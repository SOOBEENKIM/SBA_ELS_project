"""Monthly coupon obligations against an independent scalar cash-flow ledger."""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np
import torch
from module.mc_contract_v3 import *
from module.mc_contract_v2 import price_grid_v2

HERE=Path(__file__).resolve().parent;OUT=HERE/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
m=MarketV2((.2,.25,.3),((1.,.4,.4),(.4,1.,.4),(.4,.4,1.)),(2.,0.,0.))
c=ContractV3('monthly_test','STEP KI',(2,4),(3,7),(.9,.8),.12,(0.,0.),
    (None,None),(0.,0.),.5,0.,monthly_obs_days=(1,2,3,4),monthly_pay_days=(2,3,5,7),
    monthly_barriers=(.6,)*4,monthly_accruals=(1/12,)*4,coupon_on_redemption_day=True)

def summary(x,c):
    x=np.asarray(x,float);x=x[None] if x.ndim==1 else x;logs=np.log(x)
    return dict(wobs=torch.tensor(logs[:,c.obs_days]),
        wrun=torch.tensor(np.minimum.accumulate(logs,axis=1)[:,c.obs_days]),
        minimum=torch.tensor(logs.min(1)),final=torch.tensor(logs[:,c.obs_days[-1]]),
        monthly=torch.tensor(logs[:,c.monthly_obs_days]))

def ledger(x,c,m):
    # Contract-level scalar algorithm: generate dated cash flows, then discount.
    # Does not use the vectorized event code or its payoff helpers.
    end=c.obs_days[-1];trigger=False;principal=None;pd=c.pay_days[-1]
    for j,day in enumerate(c.obs_days):
        if x[day]>=c.strikes[j]:
            principal=1+c.coupon*c.coupon_accruals[j];end=day;pd=c.pay_days[j];trigger=True;break
        b=c.lz_barr[j]
        if b is not None and min(x[:day+1])>=b:
            principal=1+c.lz_pmts[j];end=day;pd=c.pay_days[j];trigger=True;break
    if principal is None:
        if c.has_ki and min(x[:end+1])>=c.ki_barrier:principal=1+c.coupon*c.survival_coupon_accrual
        else:principal=x[end]
    flows=[(pd,principal,'principal')]
    for day,pay,b,accr in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_accruals):
        if day>end or (trigger and day==end and not c.coupon_on_redemption_day):continue
        if x[day]>=b:flows.append((pay,c.coupon*accr,'coupon'))
    pv=sum(amount*np.exp(-float(F.zero_curve(m.beta,[day/365])[0])*day/365) for day,amount,_ in flows)
    return pv,flows

lz=replace(c,structure='LIZARD KI',lz_barr=(.65,None))
nc=replace(c,structure='STEP no-KI',ki_barrier=None,survival_coupon_accrual=None)
examples=[('coupon_without_redemption',c,[1,.7,.7,.7,.75]),
 ('coupon_and_redemption_same_day',c,[1,.7,.95,.7,.7]),
 ('no_coupons_after_redemption',c,[1,.7,.95,1.1,1.1]),
 ('missed_coupon_does_not_terminate_or_accrue_memory',c,[1,.55,.7,.7,.75]),
 ('paid_coupon_retained_on_terminal_loss',c,[1,.7,.7,.4,.45]),
 ('KI_survival_has_no_second_cumulative_coupon',c,[1,.7,.7,.7,.75]),
 ('noKI_terminal_loss_and_coupon_below_redemption_strike',nc,[1,.7,.7,.7,.75]),
 ('lizard_and_same_day_coupon',lz,[1,.7,.7,.4,.45]),
 ('regular_priority_over_lizard',replace(lz,lz_pmts=(.02,0)),[1,.95,.95,.4,.45]),
 ('monthly_barrier_equality_pays',c,[1,.6,.7,.6,.75]),
 ('coupon_breach_does_not_block_later_coupon',c,[1,.55,.7,.65,.75]),
 ('zero_barrier_unconditional_coupon',replace(c,monthly_barriers=(0.,)*4),[1,.4,.4,.4,.45]),
 ('vested_coupon_paid_after_redemption',replace(c,monthly_pay_days=(5,6,7,8)),[1,.7,.95,.3,.3]),
 ('explicit_no_same_day_coupon',replace(c,coupon_on_redemption_day=False),[1,.7,.95,.3,.3])]
rows=[]
for name,cc,x in examples:
    actual=float(payoff_v3(summary(x,cc),setup_v3(cc,m),cc)[0][0]);expected,flows=ledger(x,cc,m)
    assert abs(actual-expected)<1e-12,(name,actual,expected)
    rows.append(dict(case=name,expected=expected,actual=actual,flows=flows))
rng=np.random.default_rng(193);x=np.c_[np.ones(1000),rng.uniform(.3,1.1,(1000,4))]
for cc in [c,lz,nc,replace(lz,structure='LIZARD no-KI',ki_barrier=None,survival_coupon_accrual=None)]:
    actual=payoff_v3(summary(x,cc),setup_v3(cc,m),cc)[0].numpy()
    expect=np.array([ledger(row,cc,m)[0] for row in x]);np.testing.assert_allclose(actual,expect,atol=1e-12,rtol=0)
    up=bump_v3(cc,'coupon_regular',.01);down=bump_v3(cc,'coupon_regular',-.01)
    pu=payoff_v3(summary(x,cc),setup_v3(cc,m),up)[0].numpy();pd=payoff_v3(summary(x,cc),setup_v3(cc,m),down)[0].numpy()
    assert (pu>=actual-1e-12).all();np.testing.assert_allclose(pu-actual,actual-pd,atol=1e-12,rtol=0)
    bup=bump_v3(cc,'monthly_barrier',.05)
    pbu=payoff_v3(summary(x,cc),setup_v3(cc,m),bup)[0].numpy();assert (pbu<=actual+1e-12).all()
for bad in [replace(c,coupon_on_redemption_day=None),replace(c,monthly_accruals=(1/12,)),
            replace(c,monthly_obs_days=(1,2,3,5)),replace(c,monthly_pay_days=(0,3,5,7)),
            replace(c,survival_coupon_accrual=None)]:
    try:bad.validate()
    except ValueError:pass
    else:raise AssertionError('Incomplete monthly contract accepted')
for axis,field in [('coupon_regular','coupon'),('ki_barrier','ki_barrier'),('first_strike','strikes'),('last_strike','strikes'),('monthly_barrier','monthly_barriers')]:
    bumped=bump_v3(c,axis,.01)
    assert all(v==asdict(bumped)[k] for k,v in asdict(c).items() if k!=field)

# Ordinary contracts preserve exact V2 payoff and identical simulated path streams.
old=ContractV2('old','STEP KI',(2,4),(3,7),(.9,.8),.06,(.5,2.),(None,None),(0.,0.),.5,2.)
new=ContractV3(**asdict(old));a=price_grid_v2([('base',0.,old)],m,paths=5000,checkpoints=(2500,5000))
b=price_grid_v3([('base',0.,new)],m,paths=5000,checkpoints=(2500,5000))
for r,s in zip(a,b):
    for k,v in r.items():assert v==s[k],(k,v,s[k])

# Zero-volatility scenario checks simulator -> monthly observations -> PV end-to-end.
z=replace(m,sigs=(0.,0.,0.),beta=(0.,0.,0.));r=price_grid_v3([('base',0.,c)],z,paths=100,checkpoints=(100,))
assert abs(r[0]['price_sum']/100-1.02)<1e-12
assert r[0]['monthly_payment_count_sum']==200
result=dict(status='pass',obligation_cases=rows,independent_random_path_cases=4000,
    v2_nonmonthly_exact_regression=True,missing_monthly_terms_rejected=True,
    coupon_linearity_and_monotonicity=True,monthly_barrier_monotonicity=True,
    one_field_interventions=True,zero_vol_simulator_verified=True)
(OUT/'pricer_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('MONTHLY PRICER TESTS PASS:',len(rows),'obligations; 4000 independent paths; V2 exact regression')
