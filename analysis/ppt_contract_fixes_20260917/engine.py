"""Paired original / survival-only / explicit-cashflow valuation on common paths.
Market curves, diffusion discretization and dividend drops are frozen to the
previous run. Only contract observations/cashflows differ between payoff arms.
"""
from pathlib import Path
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE.parent/'ppt_full_reproduction_20260917')]
# Import by file name to avoid importing this engine recursively.
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_legacy_engine',HERE.parent/'ppt_full_reproduction_20260917/engine.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
from module.mc_contract_v3 import payoff_v3
ARMS=('legacy','survival_only','corrected')

def legacy_values(logs,running,union,p,raw,survival=0.):
    oi=np.searchsorted(union,p['obs_day']);w=logs[:,oi];r=running[:,oi]
    hit=w>=torch.as_tensor(p['logK']);event=hit|((r>=torch.as_tensor(p['loglzb']))&torch.as_tensor(p['lz_on']))
    anyhit=event.any(1);first=event.to(torch.int8).argmax(1)
    regular=hit.gather(1,first[:,None]).squeeze(1)
    rates=torch.as_tensor(p['rate']);lz=torch.as_tensor(p['lzp']);df=torch.as_tensor(p['DF'])
    pv=(1+torch.where(regular,rates[first],lz[first]))*df[torch.as_tensor(p['obs_day'])[first]-1]
    ti=np.searchsorted(union,p['N']);terminal=logs[:,ti].exp()
    survives=torch.zeros_like(anyhit)
    if raw['B']<1:
        survives=running[:,ti]>=np.float32(np.log(raw['B']))
        terminal=torch.where(survives,torch.ones_like(terminal),terminal)
    result=torch.where(anyhit,pv,terminal*df[p['N']-1]).double()
    addition=(~anyhit&survives).double()*float(survival)*float(p['DF'][p['N']-1])
    return result,result+addition

def explicit_values(logs,running,union,c,df):
    oi=np.searchsorted(union,c.obs_days);mi=np.searchsorted(union,c.monthly_obs_days)
    paths=dict(wobs=logs[:,oi],wrun=running[:,oi],minimum=running[:,oi[-1]],final=logs[:,oi[-1]],monthly=logs[:,mi])
    return payoff_v3(paths,dict(DF=df),c)[0]

def independent_ledger(logs,running,union,c,df):
    # A sequential NumPy cash-flow ledger, separate from the vectorized engine.
    n=len(logs);alive=np.ones(n,bool);out=np.zeros(n);end=np.full(n,c.obs_days[-1],int)
    for j,day in enumerate(c.obs_days):
        k=np.searchsorted(union,day);reg=alive&(logs[:,k]>=np.log(c.strikes[j]))
        lz=np.zeros(n,bool)
        if c.lz_barr[j] is not None:lz=alive&~reg&(running[:,k]>=np.log(c.lz_barr[j]))
        out[reg]+=(1+c.pmts[j])*df[c.pay_days[j]-1]
        out[lz]+=(1+c.lz_pmts[j])*df[c.pay_days[j]-1]
        end[reg|lz]=day;alive[reg|lz]=False
    k=np.searchsorted(union,c.obs_days[-1])
    loss=running[:,k]<np.log(c.ki_barrier) if c.has_ki else np.ones(n,bool)
    last=np.where(loss,np.exp(logs[:,k]),1+c.survival_pmt)*df[c.pay_days[-1]-1]
    out[alive]+=last[alive]
    for day,pay,b,rate in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_pmts):
        eligible=day<=end
        if not c.coupon_on_redemption_day:eligible &= ~((day==end)&~alive)
        eligible &= logs[:,np.searchsorted(union,day)]>=np.log(b) if b>0 else True
        out[eligible]+=rate*df[pay-1]
    return out

@torch.no_grad()
def shared_paths(sigs,corr,curve,raw,c,events,seed,n=40000,path_chunk=20000,tblock=512):
    c.validate();p=legacy.BASE_SETUP(sigs,corr,curve,**raw)
    N=max(p['N'],c.obs_days[-1]);maxpay=max([N,*c.pay_days,*c.monthly_pay_days])
    # Preserve source curve rounding and market simulation exactly on overlapping days.
    t=np.arange(1,maxpay+1)/365.;df=np.exp(-curve(t)*t)
    fdt=-np.diff(np.r_[np.float32(0),np.log(df[:N])]).astype(np.float32)
    sig=np.asarray(sigs,np.float32);drift=fdt[:,None]-.5*sig[None,:]**2*np.float32(1/365)
    # Source _setup computes fdt from float64 DF before casting; use its drift
    # verbatim for its full horizon, extending only if actual evaluations require it.
    drift[:p['N']]=p['drift']
    for day,asset,drop in events:
        if int(day)<N:drift[int(day),int(asset)]+=np.float32(drop)
    # Discount actual cash flows with the same curve but retain float64 PV precision.
    exact_df=np.exp(-curve(t)*t)
    union=np.array(sorted(set([*p['obs_day'],p['N'],*c.obs_days,*c.monthly_obs_days])),int)
    L=torch.as_tensor(p['L']*(p['sig']*p['sq'])[:,None]);cdrift=torch.as_tensor(drift).cumsum(0).T.contiguous()
    generator=torch.Generator(device=torch.get_default_device()).manual_seed(int(seed))
    for begin in range(0,n,path_chunk):
        size=min(path_chunk,n-begin);carry=torch.zeros(3,size,1);minimum=torch.zeros(size)
        logs=torch.empty(size,len(union));running=torch.empty_like(logs)
        for t0 in range(0,N,tblock):
            length=min(tblock,N-t0);x=torch.randn(3,size,length,generator=generator)
            x[2]=L[2,0]*x[0]+L[2,1]*x[1]+L[2,2]*x[2]
            x[1]=L[1,0]*x[0]+L[1,1]*x[1];x[0]=L[0,0]*x[0]
            x=x.cumsum(2);x+=carry;carry=x[:,:,-1:].clone();x+=cdrift[:,None,t0:t0+length]
            worst=x.amin(0);run=torch.minimum(worst.cummin(1).values,minimum[:,None]);minimum=torch.minimum(minimum,worst.amin(1))
            select=np.flatnonzero((union-1>=t0)&(union-1<t0+length))
            if len(select):
                at=union[select]-1-t0;logs[:,select]=worst[:,at];running[:,select]=run[:,at]
        old,surv=legacy_values(logs,running,union,p,raw,c.survival_pmt)
        corrected=explicit_values(logs,running,union,c,exact_df)
        yield torch.stack((old,surv,corrected)),(logs,running,union,exact_df,p)

@torch.no_grad()
def price(sigs,corr,curve,raw,c,events,seed,n=40000,audit=False):
    sums=np.zeros(3);squares=np.zeros(3);dsum=np.zeros(2);dsquares=np.zeros(2)
    for values,details in shared_paths(sigs,corr,curve,raw,c,events,seed,n):
        logs,running,union,df,p=details
        if audit:
            expected=independent_ledger(logs.double().cpu().numpy(),running.double().cpu().numpy(),union,c,df)
            np.testing.assert_allclose(values[2].cpu().numpy(),expected,atol=1e-12,rtol=0)
        sums+=values.sum(1).cpu().numpy();squares+=(values*values).sum(1).cpu().numpy()
        d=values[1:]-values[:-1];dsum+=d.sum(1).cpu().numpy();dsquares+=(d*d).sum(1).cpu().numpy()
    mean=sums/n;se=np.sqrt(np.maximum(0,(squares-n*mean**2)/(n-1))/n)
    dm=dsum/n;ds=np.sqrt(np.maximum(0,(dsquares-n*dm**2)/(n-1))/n)
    return dict(prices=mean.tolist(),se=se.tolist(),step_delta=dm.tolist(),step_se=ds.tolist())
