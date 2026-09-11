"""Structural, coupled-path, and independent dated-cashflow checks."""
import json
from dataclasses import replace
import numpy as np
import torch
from design import *
from schedule_mc import project_paths,shared_summaries,price_grid
from module.mc_contract_v3 import ContractV3,MarketV2,setup_v3,payoff_v3,price_grid_v3,summaries_v3
from module import features as F

def scalar_ledger(x,c,m):
    end=c.obs_days[-1];payday=c.pay_days[-1];amount=None;redeemed=False
    for j,day in enumerate(c.obs_days):
        if x[day]>=c.strikes[j]:
            amount=1+c.coupon*c.coupon_accruals[j];end=day;payday=c.pay_days[j];redeemed=True;break
        barrier=c.lz_barr[j]
        if barrier is not None and min(x[:day+1])>=barrier:
            amount=1+c.lz_pmts[j];end=day;payday=c.pay_days[j];redeemed=True;break
    if amount is None:
        amount=1+c.coupon*c.survival_coupon_accrual if c.has_ki and min(x[:end+1])>=c.ki_barrier else x[end]
    flows=[(payday,amount)]
    for day,pay,b,a in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_accruals):
        if day>end or (redeemed and day==end and not c.coupon_on_redemption_day):continue
        if x[day]>=b:flows.append((pay,c.coupon*a))
    # Independently expressed NS basis, not the engine discount function.
    out=0.
    for day,amount in flows:
        t=day/365;z=t/1.5;e=np.exp(-z);u=(1-e)/z
        rate=(m.beta[0]+m.beta[1]*u+m.beta[2]*(u-e))/100
        out+=amount*np.exp(-rate*t)
    return out

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);rows,excluded=build();cal=SettlementCalendar()
    assert len(excluded)==141 and len({r['family'] for r in excluded})==47
    count=0;maxdiff=0.;rng=np.random.default_rng(202609113)
    # Independent cash-flow prices for every designed baseline/changed contract.
    for r in rows:
        c,m=unpack(r);assert change_tenor(c,0,r['calendar_context'],cal)[0]==c
        cs=[unpack(dict(contract=z['contract'],market=r['market']))[0] for z in r['variants']]
        N=max(x.obs_days[-1] for x in cs);union=np.arange(1,N+1)
        # Baseline S=1 then three stresses plus eight smooth random worst paths.
        x=np.ones((12,N+1));x[1,1:]=.4;x[2,1:]=.75;x[3,1:]=1.1
        x[4:]=np.exp(np.c_[np.zeros(8),np.cumsum(rng.normal(-.0001,.018,(8,N)),axis=1)])
        logs=torch.tensor(np.log(x[:,1:]));running=torch.tensor(np.minimum.accumulate(np.log(x),axis=1)[:,1:])
        for cc in cs:
            actual=payoff_v3(project_paths(logs,running,union,cc),setup_v3(cc,m),cc)[0].numpy()
            expected=np.array([scalar_ledger(v,cc,m) for v in x])
            err=float(np.max(abs(actual-expected)));maxdiff=max(maxdiff,err)
            np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=0)
            count+=len(x)
        if r['monthly']:
            ctx=r['calendar_context'];issue=date.fromisoformat(ctx['issue'])
            for z,cc in zip(r['variants'],cs):
                if z['axis']=='tenor_months' and z['offset']>0:
                    n=len(c.monthly_obs_days)
                    for field in ['monthly_obs_days','monthly_pay_days','monthly_barriers','monthly_accruals']:
                        assert getattr(cc,field)[:n]==getattr(c,field)
                    assert len(cc.monthly_obs_days)==n+z['offset']
                if z['axis']=='tenor_months':
                    assert all(d<=cc.obs_days[-1] for d in cc.monthly_obs_days)
                    assert all(payment(d,ctx,cal)==p for d,p in zip(cc.obs_days,cc.pay_days))
    # Identical-date execution agrees exactly with the already audited V3 engine.
    identity=[]
    for r in [rows[0],rows[1],rows[-1]]:
        c,m=unpack(r)
        a=price_grid_v3([('base',0.,c)],m,paths=5000,seed=443,checkpoints=(2500,5000))
        b=price_grid([c],m,paths=5000,seed=443,checkpoints=(2500,5000))
        for u,v in zip(a,b):
            for k in ['price_sum','price_sumsq','delta_sum','delta_sumsq','affected','monthly_pv_sum']:assert u[k]==v[k],(k,u[k],v[k])
        identity.append(r['family'])
    # The reference collector records EVERY day. Projecting variable horizons
    # must agree with that full record, including KI limited to each horizon.
    m=MarketV2((.2,.3,.4),((1.,.3,.2),(.3,1.,.4),(.2,.4,1.)),(2.,-.5,.1))
    full=ContractV3('full','STEP KI',(12,24),(14,26),(.95,.8),.06,(0.,0.),(None,None),(0.,0.),.5,0.,
        monthly_obs_days=tuple(range(1,25)),monthly_pay_days=tuple(range(3,27)),monthly_barriers=(.6,)*24,monthly_accruals=(1/365,)*24,coupon_on_redemption_day=True)
    short=replace(full,name='short',obs_days=(6,12),pay_days=(8,14),monthly_obs_days=tuple(range(1,13)),monthly_pay_days=tuple(range(3,15)),monthly_barriers=(.6,)*12,monthly_accruals=(1/365,)*12)
    mixed=replace(full,name='mixed',obs_days=(7,20),pay_days=(9,22),monthly_obs_days=(3,7,13,20),monthly_pay_days=(5,9,15,22),monthly_barriers=(.6,)*4,monthly_accruals=(1/12,)*4)
    pairs=0
    for (ss,ps),(ref,rp) in zip(shared_summaries([short,mixed,full],m,5000,993,tblock=7),summaries_v3(full,m,5000,993,tblock=7)):
        # original full.monthly is the full daily log-price history
        w=ref['monthly'];run=torch.minimum(torch.zeros_like(w),w.cummin(1).values)
        for c,s in zip([short,mixed,full],ss):
            expected=project_paths(w,run,np.arange(1,25),c)
            for k in expected:assert torch.equal(s[k],expected[k]),k
        pairs+=w.shape[0]
    # Future KI after shorter maturity must not contaminate the short payoff.
    x=np.r_[1.,np.full(12,.75),np.full(12,.3)]
    logs=torch.tensor(np.log(x[1:])[None]);run=torch.tensor(np.minimum.accumulate(np.log(x))[None,1:])
    sv=payoff_v3(project_paths(logs,run,np.arange(1,25),short),setup_v3(short,m),short)[0].item()
    assert abs(sv-scalar_ledger(x,short,m))<1e-12
    result=dict(status='pass',families=len(rows),scenarios=sum(len(r['variants']) for r in rows),independent_path_contract_ledgers=count,
        max_ledger_error_krw=maxdiff*10000,exact_v3_identity_families=identity,full_daily_reference_paths=pairs,
        future_ki_after_short_maturity_excluded=True,zero_tenor_change_identity=True,
        model_capacity_excluded_families=47,protocol_sha256=sha(HERE/'protocol.json'))
    (OUT/'validation_before_mc.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
