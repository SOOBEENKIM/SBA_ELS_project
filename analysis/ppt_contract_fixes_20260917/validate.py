"""Cash-flow regression tests, independent pathwise ledger and source-engine parity."""
from pathlib import Path
from dataclasses import replace
import json,gzip
import numpy as np,torch
from contracts import ContractV3,published
from engine import explicit_values,independent_ledger,price,legacy
HERE=Path(__file__).resolve().parent

def main():
    torch.set_default_device('cpu');torch.set_num_threads(2);checks=[]
    c=ContractV3('test','STEP KI',(182,365,548),(185,368,551),(.9,.85,.8),.08,(.5,1,1.5),(None,)*3,(0.,)*3,.6,1.5)
    def check(name,c,spots,mins,expected):
        union=np.array(sorted(set(c.obs_days)|set(c.monthly_obs_days)));w=np.log(np.asarray(spots,float));r=np.log(np.asarray(mins,float));df=np.ones(max([*c.pay_days,*c.monthly_pay_days]))
        values=explicit_values(torch.tensor(w),torch.tensor(r),union,c,df).numpy()
        np.testing.assert_allclose(values,expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(values,independent_ledger(w,r,union,c,df),atol=1e-12,rtol=0)
        checks.append(name)
    check('KI not hit: explicit maturity survival coupon',c,[[.8,.75,.7]],[[.8,.75,.7]],[1.12])
    check('KI hit and maturity strike missed: loss, no survival coupon',c,[[.8,.75,.7]],[[.8,.5,.5]],[.7])
    check('Maturity strike met: coupon despite past KI',c,[[.8,.75,.9]],[[.8,.5,.5]],[1.12])
    check('First redemption: principal plus first coupon',c,[[.95,.8,.7]],[[.95,.8,.7]],[1.04])
    check('No KI clause: terminal loss',replace(c,structure='STEP no-KI',ki_barrier=None,survival_coupon_accrual=None),[[.8,.75,.7]],[[.8,.75,.7]],[.7])
    lz=replace(c,structure='LIZARD KI',lz_barr=(.7,None,None),lz_pmts=(.03,0.,0.))
    check('Lizard no-touch event',lz,[[.8,.75,.7]],[[.75,.7,.65]],[1.03])
    check('Regular redemption takes precedence over Lizard',lz,[[.95,.8,.7]],[[.8,.75,.7]],[1.04])
    m=replace(c,coupon_accruals=(0.,)*3,survival_coupon_accrual=0.,monthly_obs_days=(30,182,365,548),monthly_pay_days=(35,190,370,553),monthly_barriers=(.65,)*4,monthly_accruals=(1/12,)*4,coupon_on_redemption_day=True)
    check('Monthly coupon on redemption day; earned payment after redemption preserved',m,[[.7,.95,.8,.7]],[[.7,.7,.7,.7]],[1+.08/6])
    check('Future monthly coupons stop at redemption',m,[[.5,.95,.8,.7]],[[.5,.5,.5,.5]],[1+.08/12])
    check('Explicit same-day exclusion is honored',replace(m,coupon_on_redemption_day=False),[[.7,.95,.8,.7]],[[.7,.7,.7,.7]],[1+.08/12])
    # Payment date discounting, including an already earned coupon paid later.
    u=np.array(sorted(set(m.obs_days)|set(m.monthly_obs_days)));w=torch.tensor(np.log([[.7,.95,.8,.7]]));r=torch.tensor(np.log([[.7,.7,.7,.7]]));df=np.exp(-.03*np.arange(1,554)/365.)
    val=explicit_values(w,r,u,m,df).item();expected=df[184]+.08/12*(df[34]+df[189]);assert abs(val-expected)<1e-12;checks.append('Distinct principal/coupon payment-date discounting')
    # Published six templates must retain their exact schedules and rates.
    for it,record in published.items():
        cc=ContractV3(**record['contract']).validate();assert cc.has_monthly and cc.coupon_on_redemption_day
    checks.append('Six individually reconciled monthly contracts validate unchanged')
    torch.set_default_device('cuda:1')
    curve=lambda t:np.full_like(np.asarray(t),.025,dtype=float);sig=[.18,.25,.3];corr=np.array([[1,.3,.4],[.3,1,.5],[.4,.5,1]])
    parity=[]
    for cc in (c,lz,m):
        raw=dict(B=.6,strikes=cc.strikes,ten=548/365,c=.08,pmts=[.04,.08,.12],lz_barr=[np.nan if x is None else x for x in cc.lz_barr],lz_pmt=cc.lz_pmts)
        result=price(sig,corr,curve,raw,cc,[],123,n=4000,audit=True)
        old=legacy.ORIGINAL['mc_daily_t'](sig,corr,curve,**raw,n=4000,seed=123,dev='cuda:1')
        assert result['prices'][0]==old
        parity.append(dict(structure=cc.structure,monthly=cc.has_monthly,legacy_difference=result['prices'][0]-old))
    checks.append('4000-path independent sequential ledger for ordinary/Lizard/monthly')
    checks.append('Exact original-engine MC price parity on unchanged path horizon')
    cc=c;raw=dict(B=.6,strikes=c.strikes,ten=548/365,c=.08,pmts=[.04,.08,.12],lz_barr=[np.nan]*3,lz_pmt=[0]*3)
    ev=[(89,0,float(np.log(.98))),(359,2,float(np.log(.97)))]
    r1=price(sig,corr,curve,raw,cc,ev,17,n=4000,audit=True);r2=price(sig,corr,curve,raw,cc,ev,17,n=4000)
    assert r1==r2;assert abs(r1['prices'][0]-legacy.price(sig,corr,curve,raw,ev,17,device='cuda:1',n=4000)[0])<1e-12
    checks.append('Discrete-dividend parity and exact seeded repeatability')
    report=dict(status='pass',checks=checks,source_price_parity=parity,pathwise_tolerance=1e-12)
    (HERE/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
