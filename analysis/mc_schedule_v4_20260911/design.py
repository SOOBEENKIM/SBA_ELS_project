"""Explicit schedule interventions; no observed counterfactual price imputation."""
from pathlib import Path
from dataclasses import asdict,replace
from datetime import date,timedelta
from calendar import monthrange
import json,hashlib
import numpy as np
from common import unpack,encode
from calendar_rules import SettlementCalendar

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
V3=PROJECT/'analysis/mc_monthly_v3_20260910'
OUT=HERE/'results'
CFG=json.loads((HERE/'protocol.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add_months(d,months):
    k=d.year*12+d.month-1+months;y,m=divmod(k,12);m+=1
    return date(y,m,min(d.day,monthrange(y,m)[1]))
def following(d,cal):
    while not cal.business_day(d):d+=timedelta(days=1)
    return d
def adjusted(day,ctx,cal):
    if not ctx:return int(day)
    issue=date.fromisoformat(ctx['issue'])
    return (following(issue+timedelta(days=int(day)),cal)-issue).days
def payment(day,ctx,cal,lag=0):
    if not ctx:return int(day+lag)
    issue=date.fromisoformat(ctx['issue'])
    return (cal.advance(issue+timedelta(days=int(day)),int(ctx['lag']))-issue).days

def insert_observation(c,position,ctx,cal):
    if c.nobs>=12:raise ValueError('baseline_nobs_12_exceeds_frozen_input_capacity_after_insertion')
    j={'early':0,'middle':c.nobs//2,'late':c.nobs-1}[position]
    left=c.obs_days[j-1] if j else 0;right=c.obs_days[j]
    day=adjusted(round((left+right)/2),ctx,cal)
    if not left<day<right:raise ValueError('no_intermediate_observation_day')
    w=(day-left)/(right-left)
    strike=c.strikes[j] if j==0 else (1-w)*c.strikes[j-1]+w*c.strikes[j]
    prev=c.coupon_accruals[j-1] if j else 0.
    accr=(1-w)*prev+w*c.coupon_accruals[j]
    lag=c.pay_days[j]-c.obs_days[j]
    newpay=payment(day,ctx,cal,lag)
    changes={}
    for field,value in [('obs_days',day),('pay_days',newpay),('strikes',strike),('coupon_accruals',accr),('lz_barr',None),('lz_pmts',0.)]:
        a=list(getattr(c,field));a.insert(j,value);changes[field]=tuple(a)
    out=replace(c,**changes).validate()
    # Removing the inserted clause recovers every original field exactly.
    for field in changes:assert getattr(out,field)[:j]+getattr(out,field)[j+1:]==getattr(c,field)
    for field,v in asdict(c).items():
        if field not in changes:assert v==asdict(out)[field],field
    return out,dict(insert_index=j,added_observation_day=day,added_payment_day=newpay,added_strike=strike,added_coupon_accrual=accr)

def change_tenor(c,months,ctx,cal):
    if months==0:return c,dict(actual_horizon_change_days=0)
    old=c.obs_days[-1]
    if ctx:
        issue=date.fromisoformat(ctx['issue'])
        final=(following(add_months(issue+timedelta(days=old),months),cal)-issue).days
    else:final=old+round(months*365/12)
    scale=final/old
    obs=[adjusted(round(x*scale),ctx,cal) for x in c.obs_days];obs[-1]=final
    pay=[payment(d,ctx,cal,c.pay_days[j]-c.obs_days[j]) for j,d in enumerate(obs)]
    changes=dict(obs_days=tuple(obs),pay_days=tuple(pay),coupon_accruals=tuple(x*scale for x in c.coupon_accruals),
                 lz_pmts=tuple(x*scale for x in c.lz_pmts),survival_coupon_accrual=None if c.survival_coupon_accrual is None else c.survival_coupon_accrual*scale)
    stub=False
    if c.has_monthly:
        kept=[(d,p,b,a) for d,p,b,a in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_accruals) if d<=final]
        if months<0:
            future=[j for j,d in enumerate(c.monthly_obs_days) if d>final]
            previous=kept[-1][0] if kept else 0
            if future and previous<final:
                j=future[0];accr=c.monthly_accruals[j]*(final-previous)/(c.monthly_obs_days[j]-previous)
                kept.append((final,payment(final,ctx,cal),c.monthly_barriers[j],accr));stub=True
        else:
            anchor=issue+timedelta(days=old)
            for k in range(1,months+1):
                d=(following(add_months(anchor,k),cal)-issue).days
                assert d<=final
                kept.append((d,payment(d,ctx,cal),c.monthly_barriers[-1],c.monthly_accruals[-1]))
        changes.update({field:tuple(row[k] for row in kept) for k,field in enumerate(['monthly_obs_days','monthly_pay_days','monthly_barriers','monthly_accruals'])})
    out=replace(c,**changes).validate()
    for field in ['strikes','coupon','ki_barrier','lz_barr','structure','monitoring','coupon_on_redemption_day']:
        assert getattr(c,field)==getattr(out,field),field
    assert out.nobs==c.nobs
    return out,dict(actual_horizon_change_days=final-old,horizon_ratio=scale,final_stub_coupon=stub,
                    monthly_count_change=len(out.monthly_obs_days)-len(c.monthly_obs_days))

def families():
    fs=json.loads((V3/'results/families.json').read_text())
    ts={r['item']:r for r in json.loads((V3/'results/monthly_templates.json').read_text())}
    pub={r['original_item']:r for r in json.loads((V3/'results/published_monthly_contracts.json').read_text())}
    selected=[r for r in fs if r['split']=='test' or r['origin']=='published_monthly']
    for r in selected:
        ctx=None
        if r['monthly']:
            item=r.get('template_item',r.get('original_item'));ctx=dict(issue=ts[item]['issue'],lag=int(r.get('settlement_lag_business_days',pub.get(item,{}).get('rule',{}).get('lag',0))))
            assert ctx['lag'] in (2,3)
        yield r,ctx

def variants(r,ctx,cal):
    c,m=unpack(r);result=[dict(axis='base',offset=0,contract=asdict(c),details={})];excluded=[]
    for pos in CFG['nobs']['positions']:
        if c.nobs>=12:
            excluded.append(dict(family=r['family'],axis='nobs_'+pos,offset=1,reason='13 redemption dates exceed existing engine/model 12-slot capacity'));continue
        cc,details=insert_observation(c,pos,ctx,cal)
        result.append(dict(axis='nobs_'+pos,offset=1,contract=asdict(cc),details=details))
    for h in CFG['tenor_month_offsets']:
        cc,details=change_tenor(c,h,ctx,cal)
        result.append(dict(axis='tenor_months',offset=h,contract=asdict(cc),details=details))
    return result,excluded

def build():
    cal=SettlementCalendar();rows=[];excluded=[];counts={}
    for r,ctx in families():
        cs,ex=variants(r,ctx,cal);excluded+=ex
        record=dict(r,calendar_context=ctx,variants=cs)
        c,m=unpack(r);u0,v0,x0=encode(c,m)
        for case in cs:
            cc,_=unpack(dict(contract=case['contract'],market=r['market']));u,v,x=encode(cc,m)
            assert np.array_equal(u,u0) and np.array_equal(v,v0)
            assert case['axis']=='base' or not np.array_equal(x,x0)
            case['input_changed_columns']=np.flatnonzero(x!=x0).tolist()
        rows.append(record)
    assert len(rows)==262 and len({r['family'] for r in rows})==262
    return rows,excluded

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);rows,ex=build()
    (OUT/'families.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (OUT/'excluded.json').write_text(json.dumps(ex,ensure_ascii=False,indent=2))
    print(json.dumps(dict(families=len(rows),scenarios=sum(len(r['variants']) for r in rows),excluded_variants=len(ex))))
