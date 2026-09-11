from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from module.mc_contract_v3 import ContractV3,MarketV2,bump_v3
from module import features as F
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];OUT=HERE/'results'
CFG=json.loads((HERE/'protocol.json').read_text());AXES=['base',*CFG['axes']]
STRUCTURES=['STEP KI','STEP no-KI','LIZARD KI','LIZARD no-KI']
NCON=461
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(r):
    d=r['contract'].copy()
    for k in ['obs_days','pay_days','strikes','coupon_accruals','lz_barr','lz_pmts','monthly_obs_days','monthly_pay_days','monthly_barriers','monthly_accruals']:
        if k in d:d[k]=tuple(d[k])
    m=r['market'].copy();m['sigs']=tuple(m['sigs']);m['corr']=tuple(tuple(x) for x in m['corr']);m['beta']=tuple(m['beta']);m['dividend_yields']=tuple(m.get('dividend_yields',(0.,0.,0.)))
    return ContractV3(**d).validate(),MarketV2(**m).validate()
def cases(c):
    out=[('base',0.,c)]
    for axis,offsets in CFG['axes'].items():
        if axis=='ki_barrier' and not c.has_ki:continue
        if axis=='monthly_barrier' and not c.has_monthly:continue
        for h in offsets:
            if axis=='coupon_regular' and c.coupon+h<0:continue
            # A zero coupon barrier defines unconditional payments and cannot
            # be lowered; negative barrier interventions have no valid contract.
            if axis=='monthly_barrier' and min(c.monthly_barriers)+h<0:continue
            out.append((axis,h,bump_v3(c,axis,h)))
    return out
def encode(c,m):
    order=np.argsort(m.sigs,kind='stable');corr=np.asarray(m.corr)[np.ix_(order,order)]
    vc=np.r_[np.asarray(m.sigs)[order],corr[np.triu_indices(3,1)],np.asarray(m.dividend_yields)[order]].astype('float32')
    pad=lambda x,n:np.pad(np.asarray(x,float),(0,n-len(x)))
    con=np.r_[pad(c.strikes,12),pad(c.pmts,12),pad([0 if x is None else x for x in c.lz_barr],12),pad(c.lz_pmts,12),
        pad(np.asarray(c.obs_days)/365,12),pad(np.asarray(c.pay_days)/365,12),pad(c.coupon_accruals,12),pad(np.ones(c.nobs),12),
        c.coupon,c.survival_coupon_accrual or 0.,float(c.has_ki),c.ki_barrier or 0.,
        pad(np.asarray(c.monthly_obs_days)/365,60),pad(np.asarray(c.monthly_pay_days)/365,60),pad(c.monthly_barriers,60),
        pad(c.monthly_pmts,60),pad(c.monthly_accruals,60),pad(np.ones(len(c.monthly_obs_days)),60),
        float(c.coupon_on_redemption_day or False)].astype('float32')
    assert con.shape==(NCON,) and np.isfinite(con).all()
    return F.krw_curve_nodes(m.beta),vc,con
def family_record(c,m,split,origin,**extra):
    return dict(family=c.name,split=split,origin=origin,structure=c.structure,monthly=c.has_monthly,
        contract=asdict(c),market=asdict(m),**extra)
