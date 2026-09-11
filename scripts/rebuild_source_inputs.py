"""Rebuild the sample membership and six market states from bundled raw data.

Only universe membership is taken from the old feature table, for comparison.
No historical padded coupon features or MC labels enter the current pricer.
"""
from pathlib import Path
import json,time,sys
from functools import lru_cache
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from module import features as F

def main():
    start=time.time();raw=ROOT/'data/cache/raw';cache=ROOT/'data/cache'
    ac=pd.read_csv(raw/'LAKE_V2_DART_AUTO_CALL.csv',low_memory=False)
    sc=pd.read_csv(raw/'LAKE_V2_DART_SCHD_INFO.csv',low_memory=False)
    ud=pd.read_csv(raw/'LAKE_V2_DART_UDLY_INFO.csv',low_memory=False)
    mapping=json.loads((cache/'udly_ticker_map.json').read_text());rets={}
    for t in set(mapping.values())- {None,''}:
        p=cache/('px_'+t.replace('^','_').replace('.','_')+'.parquet')
        if p.exists():
            x=pd.read_parquet(p)['close'].dropna();rets[t]=np.log(x[~x.index.duplicated()]).diff()
    curve=pd.read_parquet(cache/'krw_curve.parquet')[['call','m3','y10']]
    underlying=ud.groupby('ITEM_CD').UDLY_ID.apply(list).to_dict()
    three=ud.groupby('ITEM_CD').UDLY_ID.nunique();three=set(three[three.eq(3)].index)
    g=sc[sc.SCHD_TYPE.eq(1)].groupby('ITEM_CD')
    sched=g.agg(n=('SEQ','size'),strike_n=('STRK_1','count'),pmt_n=('PMT_1','count'),barrier=('BARR_1','min'))
    bad_prev=set(sc.loc[sc.SCHD_TYPE.eq(1)&sc.LZRD_TERM.eq('FROM_PREV'),'ITEM_CD'])
    valid=sched.n.between(2,12)&sched.n.eq(sched.strike_n)&sched.n.eq(sched.pmt_n)
    valid_ids=set(sched[valid].index)-bad_prev
    ac['issue']=pd.to_datetime(ac.ISU_DT,errors='coerce');ac['mat']=pd.to_datetime(ac.MAT_DT,errors='coerce')
    tenor=(ac.mat-ac.issue).dt.days/365.25;fair=ac.FAIR_VALUE/ac.ISU_PRC_DETAIL
    cand=ac[ac.ITEM_CD.isin(three)&ac.CUR_CD.eq('KRW')&ac.OPT_TYPE.isin(['STEP','LIZARD'])&ac.KNCK_IN_YN.isin([0,1])&fair.between(.7,1.05)&tenor.between(.5,5.)]
    @lru_cache(None)
    def market(ts,dt):
        if len(ts)!=3 or any(t not in rets for t in ts):return None
        rs=[rets[t] for t in ts];sigs=[F.vol180(r,dt) for r in rs];corr=F.corr180(rs,dt);beta=F.krw_beta(curve.asof(dt).values)
        if any(pd.isna(sigs)) or corr is None or beta is None:return None
        return dict(sigs=sigs,corr=corr.tolist(),beta=beta.tolist())
    ids=[]
    for i,r in enumerate(cand.itertuples(),1):
        if r.ITEM_CD not in valid_ids:continue
        if pd.isna(r.ANL_RTRN) or (r.KNCK_IN_YN==1 and pd.isna(sched.at[r.ITEM_CD,'barrier'])):continue
        ts=tuple(mapping.get(u) for u in underlying.get(r.ITEM_CD,[]))
        if market(ts,r.issue) is not None:ids.append(r.ITEM_CD)
        if i%20000==0:print('RAW MEMBERSHIP',i,'/',len(cand),flush=True)
    expected=pd.read_parquet(ROOT/'data/els3_dataset.parquet').item
    assert len(ids)==len(expected)==58790,(len(ids),len(expected))
    assert set(ids)==set(expected),'Raw selection differs from the original experiment'
    ref=json.loads((ROOT/'analysis/contract_value_20260910/results/mc_selected_contracts.json').read_text())
    acidx=ac.set_index('ITEM_CD');maximum=0.
    for r in ref:
        row=acidx.loc[r['item']];ts=tuple(mapping.get(u) for u in underlying[r['item']]);m=market(ts,row.issue)
        for k in ['sigs','corr','beta']:
            error=float(np.max(abs(np.asarray(m[k])-np.asarray(r['market'][k]))));maximum=max(maximum,error)
            np.testing.assert_allclose(m[k],r['market'][k],atol=1e-12,rtol=0)
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    pd.DataFrame({'item':sorted(ids)}).to_csv(out/'rebuilt_universe.csv',index=False)
    report=dict(status='pass',raw_candidates=len(cand),universe_products=len(ids),membership_identical=True,published_market_states=len(ref),market_max_abs_error=maximum,uses_old_payoff_approximations=False,seconds=time.time()-start)
    (out/'source_inputs.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
