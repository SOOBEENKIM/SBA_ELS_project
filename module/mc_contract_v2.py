"""Explicit-schedule, inception-only three-asset ELS MC pricer (version 2).

Supported: worst-of STEP / FROM_ISU LIZARD, with or without daily-close KI,
regular redemption coupons, explicit no-KI-event maturity coupon, explicit
observation and payment days. No monthly coupons, averaging, issuer calls,
settlement calendars, quanto calibration, or in-life initial states are inferred.
Day numbers and coupon accrual factors are contract inputs, not derived from nobs.
Legacy mc_engine.py remains available solely to reproduce historical labels.
"""
from dataclasses import dataclass, replace, asdict
import numpy as np
import torch
from . import features as F


@dataclass(frozen=True)
class MarketV2:
    sigs: tuple
    corr: tuple
    beta: tuple
    dividend_yields: tuple = (0., 0., 0.)

    def validate(self):
        s=np.asarray(self.sigs,float);C=np.asarray(self.corr,float)
        if s.shape!=(3,) or not np.isfinite(s).all() or (s<0).any():raise ValueError('Three nonnegative volatilities required')
        if C.shape!=(3,3) or not np.isfinite(C).all() or not np.allclose(C,C.T,atol=1e-12):raise ValueError('Invalid correlation matrix')
        if not np.allclose(np.diag(C),1) or np.linalg.eigvalsh(C).min()<-1e-10:raise ValueError('Correlation must be PSD with unit diagonal')
        if len(self.beta)!=3 or len(self.dividend_yields)!=3 or not np.isfinite([*self.beta,*self.dividend_yields]).all():raise ValueError('Invalid curve or yields')
        return self


@dataclass(frozen=True)
class ContractV2:
    name: str
    structure: str
    obs_days: tuple
    pay_days: tuple
    strikes: tuple
    coupon: float
    coupon_accruals: tuple
    lz_barr: tuple
    lz_pmts: tuple
    ki_barrier: float | None
    survival_coupon_accrual: float | None
    monitoring: str = 'daily_calendar_grid'

    @property
    def nobs(self):return len(self.obs_days)
    @property
    def has_ki(self):return self.structure.endswith(' KI')
    @property
    def pmts(self):return np.asarray(self.coupon_accruals,float)*self.coupon
    @property
    def survival_pmt(self):return self.coupon*float(self.survival_coupon_accrual) if self.has_ki else 0.

    def validate(self):
        if self.structure not in {'STEP KI','STEP no-KI','LIZARD KI','LIZARD no-KI'}:raise ValueError('Unsupported structure')
        n=self.nobs
        if not 2<=n<=12:raise ValueError('Only 2..12 redemption dates supported')
        if any(len(x)!=n for x in [self.pay_days,self.strikes,self.coupon_accruals,self.lz_barr,self.lz_pmts]):raise ValueError('Schedule length mismatch')
        o=np.asarray(self.obs_days,float);p=np.asarray(self.pay_days,float)
        if not np.isfinite([*o,*p]).all() or (o!=o.astype(int)).any() or (p!=p.astype(int)).any():raise ValueError('Days must be explicit integers')
        if (o<1).any() or (np.diff(o)<=0).any() or (p<o).any() or (np.diff(p)<=0).any():raise ValueError('Invalid observation/payment schedule')
        if not np.isfinite([self.coupon,*self.strikes,*self.coupon_accruals,*self.lz_pmts]).all():raise ValueError('Non-finite payment inputs')
        if self.coupon<0 or (np.asarray(self.strikes)<=0).any() or (np.asarray(self.coupon_accruals)<0).any() or (np.asarray(self.lz_pmts)<0).any():raise ValueError('Invalid coupon/strike/payment')
        b=np.asarray([np.nan if x is None else x for x in self.lz_barr],float)
        active=np.isfinite(b)
        if active.any()!=self.structure.startswith('LIZARD'):raise ValueError('Structure/Lizard clauses disagree')
        if ((b[active]<=0)|(b[active]>=1)).any():raise ValueError('Lizard barriers must lie in (0,1)')
        if (np.asarray(self.lz_pmts)[~active]!=0).any():raise ValueError('Payment without Lizard clause')
        if self.has_ki:
            if self.ki_barrier is None or not 0<self.ki_barrier<1:raise ValueError('Explicit KI barrier required')
            if self.survival_coupon_accrual is None or not np.isfinite(self.survival_coupon_accrual) or self.survival_coupon_accrual<0:raise ValueError('Explicit no-KI-event maturity coupon required; no silent zero default')
        elif self.ki_barrier is not None or self.survival_coupon_accrual is not None:raise ValueError('No-KI structure must not carry a KI clause')
        if self.monitoring!='daily_calendar_grid':raise ValueError('Provide a supported monitoring model explicitly')
        return self

    def serializable(self):return asdict(self)


def bump_v2(c,axis,amount):
    if axis=='coupon_regular':new=replace(c,coupon=c.coupon+amount)
    elif axis=='ki_barrier':
        if not c.has_ki:raise ValueError('Cannot change a nonexistent KI barrier')
        new=replace(c,ki_barrier=c.ki_barrier+amount)
    elif axis in ['first_strike','last_strike']:
        s=list(c.strikes);s[0 if axis=='first_strike' else -1]+=amount
        new=replace(c,strikes=tuple(s))
    else:raise ValueError('Unsupported intervention')
    return new.validate()


def setup_v2(c,m):
    c.validate();m.validate()
    N=int(c.obs_days[-1]);maxpay=int(c.pay_days[-1]);times=np.arange(1,maxpay+1)/365.
    DF=np.exp(-F.zero_curve(m.beta,times)*times)
    fdt=-np.diff(np.r_[0.,np.log(DF[:N])])
    sig=np.asarray(m.sigs,float)
    drift=(fdt[:,None]-(np.asarray(m.dividend_yields)+.5*sig**2)[None,:]/365.).astype(np.float32)
    # PSD (including a rank-deficient valid correlation) without silently repairing a bad matrix.
    try:L=np.linalg.cholesky(np.asarray(m.corr,float))
    except np.linalg.LinAlgError:
        ev,V=np.linalg.eigh(np.asarray(m.corr,float));L=V@np.diag(np.sqrt(np.maximum(ev,0)))
    return dict(N=N,DF=DF,obs=np.asarray(c.obs_days,int)-1,pay=np.asarray(c.pay_days,int)-1,
                loading=(np.diag(sig)@L/np.sqrt(365)).astype(np.float32),drift=drift)


def summaries_v2(c,m,n,seed,path_chunk=2500,tblock=256):
    """Same market paths are consumed by every contract variant in a family."""
    p=setup_v2(c,m);g=torch.Generator().manual_seed(int(seed));N=p['N']
    loading=torch.tensor(p['loading']);drift=torch.tensor(p['drift']).cumsum(0).T.contiguous()
    done=0
    while done<n:
        size=min(path_chunk,n-done);carry=torch.zeros(3,size,1)
        wmin=torch.zeros(size)  # includes inception spot=1 (log=0)
        wobs=torch.empty(size,c.nobs);wrun=torch.empty_like(wobs)
        for start in range(0,N,tblock):
            length=min(tblock,N-start)
            eps=torch.randn(3,size,length,generator=g)
            x=(loading@eps.reshape(3,-1)).reshape(3,size,length).cumsum(2)
            x+=carry;carry=x[:,:,-1:].clone();x+=drift[:,None,start:start+length]
            worst=x.amin(0);running=torch.minimum(worst.cummin(1).values,wmin[:,None])
            wmin=torch.minimum(wmin,worst.amin(1))
            for j,day in enumerate(p['obs']):
                if start<=day<start+length:
                    wobs[:,j]=worst[:,day-start];wrun[:,j]=running[:,day-start]
        yield dict(wobs=wobs,wrun=wrun,minimum=wmin,final=wobs[:,-1]),p
        done+=size


def payoff_v2(paths,p,c):
    """Amounts and discount factors are float64; compares each event in order.

    Code 2*j: regular; 2*j+1: Lizard; 2*n: no-KI-event maturity;
    2*n+1: terminal loss. Last evaluation and last payment may differ.
    """
    w=paths['wobs'].double();r=paths['wrun'].double()
    logk=torch.tensor(np.log(c.strikes),dtype=torch.float64)
    b=np.array([np.nan if x is None else x for x in c.lz_barr],float);active=np.isfinite(b)
    logb=torch.tensor(np.log(np.where(active,b,1.)),dtype=torch.float64)
    regular=w>=logk;lizard=(r>=logb)&torch.tensor(active)
    event=regular|lizard;hit=event.any(1);first=event.to(torch.int8).argmax(1)
    isreg=regular.gather(1,first[:,None]).squeeze(1)
    rates=torch.tensor(c.pmts,dtype=torch.float64);lz=torch.tensor(c.lz_pmts,dtype=torch.float64)
    cash=1+torch.where(isreg,rates[first],lz[first])
    df=torch.tensor(p['DF'][np.asarray(c.pay_days,int)-1],dtype=torch.float64)
    redeemed=cash*df[first]
    loss=paths['minimum'].double()<np.log(c.ki_barrier) if c.has_ki else torch.ones_like(hit)
    terminal=torch.where(loss,paths['final'].double().exp(),torch.full_like(w[:,0],1+c.survival_pmt))*df[-1]
    values=torch.where(hit,redeemed,terminal)
    code=torch.where(hit,2*first+(~isreg).long(),2*c.nobs+loss.long())
    return values,code


@torch.no_grad()
def price_grid_v2(cases,m,paths=40000,seed=0,checkpoints=(10000,20000,40000),path_chunk=2500):
    if cases[0][:2]!=('base',0.):raise ValueError('First case must be baseline')
    base=cases[0][2];counts=sorted(set(checkpoints))
    if counts[-1]!=paths or counts[0]<2:raise ValueError('Invalid path checkpoints')
    for _,_,c in cases:
        c.validate()
        if c.obs_days!=base.obs_days or c.pay_days!=base.pay_days or c.monitoring!=base.monitoring:raise ValueError('Shared summaries need identical dates/monitoring')
    sums=np.zeros((len(cases),7));results=[];done=0;checkpoint=0
    for summary,p in summaries_v2(base,m,paths,seed,path_chunk=path_chunk):
        evaluated=[payoff_v2(summary,p,c) for _,_,c in cases]
        pay=torch.stack([x for x,_ in evaluated]);delta=pay-pay[0]
        codes=torch.stack([x for _,x in evaluated]);begin=0
        while begin<pay.shape[1]:
            stop=min(pay.shape[1],begin+counts[checkpoint]-done)
            y=pay[:,begin:stop];d=delta[:,begin:stop];e=codes[:,begin:stop]
            sums+=torch.stack([y.sum(1),(y*y).sum(1),d.sum(1),(d*d).sum(1),
                               (d.abs()>1e-12).sum(1),(e==2*base.nobs).sum(1),
                               (e!=e[0]).sum(1)],1).numpy()
            done+=stop-begin;begin=stop
            if done==counts[checkpoint]:
                for (axis,h,_),s in zip(cases,sums):
                    results.append(dict(axis=axis,offset=h,paths=done,seed=seed,
                        price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],
                        affected=int(s[4]),survival=int(s[5]),changed_events=int(s[6])))
                checkpoint+=1
                if checkpoint==len(counts):return results
    raise RuntimeError('Incomplete simulation')
