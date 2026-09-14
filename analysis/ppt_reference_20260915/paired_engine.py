"""Identical GBM paths, two explicitly different contractual payoff representations."""
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'analysis/mc_monthly_v3_20260910')]
import reference_engine as E
from common import unpack
from module.mc_contract_v3 import payoff_v3

def project_reference(c):
    # Same maturity convention and fallback criterion as source build_source.py.
    ten=c.pay_days[-1]/365.25;n=c.nobs;N=int(round(ten*365))
    obs=np.clip(np.round(np.arange(1,n+1)*(ten/n)*365).astype(int),1,N)
    p=np.asarray(c.pmts,float)
    nonmono=bool((np.diff(p)<-1e-9).any())
    linear=nonmono or (c.coupon>0 and ten>0 and p[-1]<.5*c.coupon*ten)
    if linear:p=c.coupon*obs/365.
    return dict(B=c.ki_barrier if c.has_ki else 1.,strikes=list(c.strikes),ten=ten,c=c.coupon,
                pmts=p.tolist(),lz_barr=list(c.lz_barr),lz_pmt=list(c.lz_pmts))

def setup_reference(c,market):
    kw=project_reference(c)
    return E._setup(market['iv'],market['corr'],E.boot_z(market['iv_rates']),**kw)

def reference_payoff(logs,running,union,c,p):
    oi=np.searchsorted(union,p['obs_day']);w=logs[:,oi];run=running[:,oi]
    hit=w>=torch.tensor(p['logK']);on=torch.tensor(p['lz_on'])
    ev=hit|((run>=torch.tensor(p['loglzb']))&on)
    anyev=ev.any(1);first=ev.float().argmax(1);reg=hit.gather(1,first[:,None]).squeeze(1)
    rate=torch.tensor(p['rate']);lzp=torch.tensor(p['lzp']);df=torch.tensor(p['DF'])
    early=(1+torch.where(reg,rate[first],lzp[first]))*df[torch.tensor(p['obs_day'])[first]-1]
    final_idx=np.searchsorted(union,p['N']);end=logs[:,final_idx].exp()
    if c.has_ki:end=torch.where(running[:,final_idx]<np.log(c.ki_barrier),end,torch.ones_like(end))
    return torch.where(anyev,early,end*df[-1]).double()

def detailed_payoff(logs,running,union,c,DF):
    oi=np.searchsorted(union,c.obs_days);mi=np.searchsorted(union,c.monthly_obs_days)
    s=dict(wobs=logs[:,oi],wrun=running[:,oi],minimum=running[:,oi[-1]],final=logs[:,oi[-1]],monthly=logs[:,mi])
    value,code,monthly,count=payoff_v3(s,dict(DF=DF),c,details=True)
    return value,monthly

def independent_ledger(logs,running,union,c,DF,p=None):
    """Separate NumPy, sequential-event ledger used only during validation."""
    n=len(logs);alive=np.ones(n,bool);values=np.zeros(n);end=np.full(n,c.obs_days[-1],int)
    if p is None:
        obs=np.asarray(c.obs_days);pay=np.asarray(c.pay_days);rates=np.asarray(c.pmts)
        strikes=np.asarray(c.strikes);lzb=np.asarray([np.nan if x is None else x for x in c.lz_barr]);lzp=np.asarray(c.lz_pmts)
    else:
        obs=p['obs_day'];pay=obs;rates=p['rate'];strikes=np.asarray(c.strikes,np.float32);lzb=np.asarray([np.nan if x is None else x for x in c.lz_barr],np.float32);lzp=p['lzp'];DF=p['DF']
        end[:]=p['N']
    for j,day in enumerate(obs):
        k=np.searchsorted(union,day)
        if p is None:hit=alive&(logs[:,k].astype(float)>=np.log(strikes[j]));lz=alive&~hit&np.isfinite(lzb[j])&(running[:,k].astype(float)>=np.log(lzb[j]))
        else:hit=alive&(logs[:,k]>=np.log(strikes[j]));lz=alive&~hit&np.isfinite(lzb[j])&(running[:,k]>=np.log(lzb[j]))
        values[hit]=(1+rates[j])*DF[pay[j]-1];values[lz]=(1+lzp[j])*DF[pay[j]-1]
        end[hit|lz]=day;alive[hit|lz]=False
    terminal=c.obs_days[-1] if p is None else p['N'];k=np.searchsorted(union,terminal)
    loss=running[:,k].astype(float)<np.log(c.ki_barrier) if c.has_ki else np.ones(n,bool)
    exp=np.exp(logs[:,k].astype(float)) if p is None else np.exp(logs[:,k])
    terminal_value=np.where(loss,exp,1+c.survival_pmt if p is None else 1.)
    values[alive]=terminal_value[alive]*DF[(c.pay_days[-1] if p is None else p['N'])-1]
    if p is None:
        for day,payday,barrier,rate in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_pmts):
            eligible=(day<=end)&((logs[:,np.searchsorted(union,day)].astype(float)>=np.log(barrier)) if barrier>0 else True)
            if not c.coupon_on_redemption_day:eligible&=~((day==end)&~alive)
            values[eligible]+=rate*DF[payday-1]
    return values

def shared_values(contracts,market,n,seed,path_chunk=20000,tblock=512,audit=False):
    for c in contracts:c.validate()
    ps=[setup_reference(c,market) for c in contracts]
    N=max([p['N'] for p in ps]+[c.obs_days[-1] for c in contracts])
    maxpay=max([N]+[max([*c.pay_days,*c.monthly_pay_days]) for c in contracts])
    times=np.arange(1,maxpay+1)/365.;DF=np.exp(-E.boot_z(market['iv_rates'])(times)*times)
    fdt=(-np.diff(np.r_[0.,np.log(DF[:N])])).astype(np.float32);sig=np.asarray(market['iv'],np.float32)
    drift=(fdt[:,None]-(.5*sig**2*np.float32(1/365))[None,:]).astype(np.float32)
    L=E.chol_psd(np.asarray(market['corr'])).astype(np.float32)
    loading=torch.tensor(L*(sig*np.float32(np.sqrt(1/365)))[:,None]);cdrift=torch.tensor(drift).cumsum(0).T.contiguous()
    union=np.array(sorted({int(d) for p,c in zip(ps,contracts) for d in [*p['obs_day'],p['N'],*c.obs_days,*c.monthly_obs_days]}))
    generator=torch.Generator(device=torch.get_default_device()).manual_seed(int(seed))
    for startpath in range(0,n,path_chunk):
        size=min(path_chunk,n-startpath);carry=torch.zeros(3,size,1);minimum=torch.full((size,),float('inf'))
        logs=torch.empty(size,len(union));running=torch.empty_like(logs)
        for t0 in range(0,N,tblock):
            T=min(tblock,N-t0);X=torch.randn(3,size,T,generator=generator)
            X[2]=loading[2,0]*X[0]+loading[2,1]*X[1]+loading[2,2]*X[2]
            X[1]=loading[1,0]*X[0]+loading[1,1]*X[1];X[0]=loading[0,0]*X[0]
            X=X.cumsum(2);X+=carry;carry=X[:,:,-1:].clone();X+=cdrift[:,None,t0:t0+T]
            w=torch.minimum(torch.minimum(X[0],X[1]),X[2]);cm=torch.minimum(w.cummin(1).values,minimum[:,None]);minimum=torch.minimum(minimum,w.amin(1))
            sel=np.flatnonzero((union-1>=t0)&(union-1<t0+T))
            if len(sel):
                col=union[sel]-1-t0;logs[:,sel]=w[:,col];running[:,sel]=cm[:,col]
        ref=torch.stack([reference_payoff(logs,running,union,c,p) for c,p in zip(contracts,ps)])
        det=[detailed_payoff(logs,running,union,c,DF) for c in contracts]
        if audit:
            for k,(c,p) in enumerate(zip(contracts,ps)):
                nr=independent_ledger(logs.cpu().numpy(),running.cpu().numpy(),union,c,DF,p)
                nd=independent_ledger(logs.cpu().numpy(),running.cpu().numpy(),union,c,DF)
                np.testing.assert_allclose(ref[k].cpu().numpy(),nr,atol=2e-7,rtol=0)
                np.testing.assert_allclose(det[k][0].cpu().numpy(),nd,atol=1e-12,rtol=0)
        yield torch.stack([ref,torch.stack([v for v,_ in det])]),torch.stack([cp for _,cp in det])

@torch.no_grad()
def price_family(contracts,market,n=40000,seed=0,checkpoints=(10000,20000,40000)):
    assert checkpoints[-1]==n
    sums=np.zeros((2,len(contracts),7));comparison=np.zeros((len(contracts),4));rows=[];done=0
    for values,monthly in shared_values(contracts,market,n,seed):
        delta=values-values[:,:1];between=values[1]-values[0];dd=delta[1]-delta[0];begin=0
        while begin<values.shape[2]:
            target=next(k for k in checkpoints if k>done);stop=min(values.shape[2],begin+target-done)
            y=values[:,:,begin:stop];d=delta[:,:,begin:stop]
            cp=torch.stack([torch.zeros_like(monthly[:,begin:stop]),monthly[:,begin:stop]])
            sums+=torch.stack([y.sum(2),(y*y).sum(2),d.sum(2),(d*d).sum(2),(d.abs()>1e-12).sum(2),cp.sum(2),torch.ones_like(y).sum(2)],2).cpu().numpy()
            b=between[:,begin:stop];bd=dd[:,begin:stop]
            comparison+=torch.stack([b.sum(1),(b*b).sum(1),bd.sum(1),(bd*bd).sum(1)],1).cpu().numpy()
            done+=stop-begin;begin=stop
            if done==target:
                for a,arm in enumerate(['reference','detailed']):
                    for k,s in enumerate(sums[a]):
                        rows.append(dict(payoff=arm,case=k,paths=done,price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],affected=int(s[4]),monthly_pv_sum=s[5]))
                for k,s in enumerate(comparison):rows.append(dict(payoff='detailed_minus_reference',case=k,paths=done,price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],affected=0,monthly_pv_sum=0.))
    return rows
