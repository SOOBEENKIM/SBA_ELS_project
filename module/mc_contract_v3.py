"""Explicit monthly coupon cash flows alongside STEP/FROM_ISU Lizard redemption.

Version 2 remains immutable. Version 3 adds separate coupon observation/payment
dates, coupon barriers and accrual factors. No memory coupons, averaging or
settlement rules are inferred; actual dates must be supplied by a contract adapter.
"""
from dataclasses import dataclass,replace
import numpy as np
import torch
from .mc_contract_v2 import ContractV2,MarketV2,setup_v2,payoff_v2,bump_v2
from . import features as F


@dataclass(frozen=True)
class ContractV3(ContractV2):
    monthly_obs_days: tuple = ()
    monthly_pay_days: tuple = ()
    monthly_barriers: tuple = ()
    monthly_accruals: tuple = ()
    coupon_on_redemption_day: bool | None = None

    @property
    def monthly_pmts(self):return self.coupon*np.asarray(self.monthly_accruals,float)
    @property
    def has_monthly(self):return bool(self.monthly_obs_days)

    def validate(self):
        super().validate()
        n=len(self.monthly_obs_days)
        if n>60:raise ValueError('At most 60 separately specified coupons supported')
        if any(len(x)!=n for x in [self.monthly_pay_days,self.monthly_barriers,self.monthly_accruals]):raise ValueError('Monthly schedule length mismatch')
        if n:
            if not isinstance(self.coupon_on_redemption_day,bool):raise ValueError('Explicit same-day coupon/redemption rule required')
            o=np.asarray(self.monthly_obs_days,float);p=np.asarray(self.monthly_pay_days,float)
            if not np.isfinite([*o,*p,*self.monthly_barriers,*self.monthly_accruals]).all():raise ValueError('Missing monthly cash-flow inputs')
            if (o!=o.astype(int)).any() or (p!=p.astype(int)).any():raise ValueError('Monthly dates must be integer day offsets')
            if (o<1).any() or (o>self.obs_days[-1]).any() or (np.diff(o)<=0).any() or (p<o).any() or (np.diff(p)<0).any():raise ValueError('Invalid monthly observation/payment dates')
            if (np.asarray(self.monthly_barriers)<0).any() or (np.asarray(self.monthly_accruals)<0).any():raise ValueError('Invalid monthly barrier/accrual')
        elif self.coupon_on_redemption_day is not None:raise ValueError('Same-day rule without monthly coupons')
        return self


def bump_v3(c,axis,amount):
    if axis=='monthly_barrier':
        if not c.has_monthly:raise ValueError('No monthly coupon barrier to change')
        return replace(c,monthly_barriers=tuple(x+amount for x in c.monthly_barriers)).validate()
    return bump_v2(c,axis,amount)


def setup_v3(c,m):
    c.validate();p=setup_v2(c,m)
    maxpay=max([*c.pay_days,*c.monthly_pay_days]);t=np.arange(1,maxpay+1)/365.
    p['DF']=np.exp(-F.zero_curve(m.beta,t)*t)
    union=np.array(sorted(set(c.obs_days)|set(c.monthly_obs_days)),int)
    p['union_obs']=union-1
    p['redemption_index']=np.searchsorted(union,c.obs_days)
    p['monthly_index']=np.searchsorted(union,c.monthly_obs_days)
    return p


def summaries_v3(c,m,n,seed,path_chunk=2500,tblock=256):
    p=setup_v3(c,m);g=torch.Generator().manual_seed(int(seed));N=p['N']
    loading=torch.tensor(p['loading']);drift=torch.tensor(p['drift']).cumsum(0).T.contiguous()
    for done in range(0,n,path_chunk):
        size=min(path_chunk,n-done);carry=torch.zeros(3,size,1);wmin=torch.zeros(size)
        wobs=torch.empty(size,len(p['union_obs']));wrun=torch.empty_like(wobs)
        for start in range(0,N,tblock):
            length=min(tblock,N-start);eps=torch.randn(3,size,length,generator=g)
            x=(loading@eps.reshape(3,-1)).reshape(3,size,length).cumsum(2)
            x+=carry;carry=x[:,:,-1:].clone();x+=drift[:,None,start:start+length]
            worst=x.amin(0);running=torch.minimum(worst.cummin(1).values,wmin[:,None])
            wmin=torch.minimum(wmin,worst.amin(1))
            sel=np.flatnonzero((p['union_obs']>=start)&(p['union_obs']<start+length))
            if len(sel):
                col=p['union_obs'][sel]-start;wobs[:,sel]=worst[:,col];wrun[:,sel]=running[:,col]
        ridx=p['redemption_index'];midx=p['monthly_index']
        yield dict(wobs=wobs[:,ridx],wrun=wrun[:,ridx],minimum=wmin,
                   final=wobs[:,ridx[-1]],monthly=wobs[:,midx]),p


def payoff_v3(paths,p,c,details=False):
    principal,code=payoff_v2(paths,p,c)
    if not c.has_monthly:
        zeros=torch.zeros_like(principal)
        return (principal,code,zeros,zeros.long()) if details else (principal,code)
    hit=code<2*c.nobs;idx=(code//2).clamp(max=c.nobs-1)
    end=torch.tensor(c.obs_days,dtype=torch.long)[idx]
    days=torch.tensor(c.monthly_obs_days,dtype=torch.long)
    eligible=days[None,:]<=end[:,None]
    if not c.coupon_on_redemption_day:eligible&=~(hit[:,None]&(days[None,:]==end[:,None]))
    barriers=np.asarray(c.monthly_barriers,float)
    logb=np.full(len(barriers),-np.inf);np.log(barriers,out=logb,where=barriers>0)
    paid=eligible&(paths['monthly'].double()>=torch.tensor(logb,dtype=torch.float64))
    discounts=p['DF'][np.asarray(c.monthly_pay_days,int)-1]
    cash=torch.tensor(c.monthly_pmts*discounts,dtype=torch.float64)
    coupons=(paid*cash[None,:]).sum(1)
    total=principal+coupons
    # Eligibility is evaluated on evaluation days. A vested coupon is retained
    # even if its payment date falls after a later redemption payment date.
    return (total,code,coupons,paid.sum(1)) if details else (total,code)


@torch.no_grad()
def price_grid_v3(cases,m,paths=40000,seed=0,checkpoints=(10000,20000,40000),path_chunk=2500):
    if cases[0][:2]!=('base',0.):raise ValueError('First case must be baseline')
    base=cases[0][2];counts=sorted(set(checkpoints))
    if counts[-1]!=paths or counts[0]<2:raise ValueError('Invalid path checkpoints')
    for _,_,c in cases:
        c.validate()
        for field in ['obs_days','pay_days','monthly_obs_days','monthly_pay_days','monitoring']:
            if getattr(c,field)!=getattr(base,field):raise ValueError('Shared paths require identical dates/monitoring')
    sums=np.zeros((len(cases),9));results=[];done=0;checkpoint=0
    for summary,p in summaries_v3(base,m,paths,seed,path_chunk=path_chunk):
        evaluated=[payoff_v3(summary,p,c,details=True) for _,_,c in cases]
        pay=torch.stack([x[0] for x in evaluated]);delta=pay-pay[0]
        codes=torch.stack([x[1] for x in evaluated]);cp=torch.stack([x[2] for x in evaluated]);cn=torch.stack([x[3] for x in evaluated]);begin=0
        while begin<pay.shape[1]:
            stop=min(pay.shape[1],begin+counts[checkpoint]-done)
            y=pay[:,begin:stop];d=delta[:,begin:stop];e=codes[:,begin:stop]
            sums+=torch.stack([y.sum(1),(y*y).sum(1),d.sum(1),(d*d).sum(1),
                (d.abs()>1e-12).sum(1),(e==2*base.nobs).sum(1),(e!=e[0]).sum(1),
                cp[:,begin:stop].sum(1),cn[:,begin:stop].sum(1)],1).numpy()
            done+=stop-begin;begin=stop
            if done==counts[checkpoint]:
                for (axis,h,_),s in zip(cases,sums):
                    results.append(dict(axis=axis,offset=h,paths=done,seed=seed,
                        price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],
                        affected=int(s[4]),survival=int(s[5]),changed_events=int(s[6]),
                        monthly_pv_sum=s[7],monthly_payment_count_sum=int(s[8])))
                checkpoint+=1
                if checkpoint==len(counts):return results
    raise RuntimeError('Incomplete simulation')
