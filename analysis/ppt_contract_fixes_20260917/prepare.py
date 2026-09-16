"""Audit all original products; retain frozen market inputs and explicit exclusions."""
from pathlib import Path
import sys,json,gzip,hashlib,time
import numpy as np
import pandas as pd
from contracts import contract
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';BASE=HERE.parent/'ppt_full_reproduction_20260917/results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);s=pd.read_parquet(BASE/'rebuilt_universe.parquet');v=pd.read_parquet(BASE/'fresh_market_inputs.parquet');ids=set(s.item)
    a=pd.read_csv(ROOT/'data/cache/raw/LAKE_V2_DART_AUTO_CALL.csv.gz',low_memory=False).set_index('ITEM_CD')
    x=pd.read_csv(ROOT/'data/cache/raw/LAKE_V2_DART_SCHD_INFO.csv.gz',low_memory=False)
    gs={(it,int(t)):g.sort_values('SEQ') for (it,t),g in x[x.ITEM_CD.isin(ids)].groupby(['ITEM_CD','SCHD_TYPE'])}
    records=[];errors=[]
    for i,it in enumerate(s.item):
        try:
            c,meta=contract(it,a.loc[it],gs.get((it,1)),gs.get((it,2)))
            records.append(dict(i=i,item=it,contract=c.serializable(),monthly=c.has_monthly,**meta))
        except ValueError as e:errors.append(dict(i=i,item=it,reason=str(e)))
        if (i+1)%10000==0:print('AUDIT',i+1,'valid',len(records),flush=True)
    (OUT/'contracts.json.gz').write_bytes(gzip.compress(json.dumps(records,allow_nan=False).encode(),mtime=0))
    pd.DataFrame(errors,columns=['i','item','reason']).to_csv(OUT/'excluded_contracts.csv',index=False)
    finalize()
def finalize():
    s=pd.read_parquet(BASE/'rebuilt_universe.parquet');v=pd.read_parquet(BASE/'fresh_market_inputs.parquet')
    a=pd.read_csv(ROOT/'data/cache/raw/LAKE_V2_DART_AUTO_CALL.csv.gz',low_memory=False).set_index('ITEM_CD')
    records=json.loads(gzip.decompress((OUT/'contracts.json.gz').read_bytes()));e=pd.read_csv(OUT/'excluded_contracts.csv')
    meta=pd.DataFrame([{k:w for k,w in r.items() if k!='contract'} for r in records]);meta['structure']=[r['contract']['structure'] for r in records]
    meta['survival_coupon']=[r['contract']['coupon']*r['contract']['survival_coupon_accrual'] if r['contract']['survival_coupon_accrual'] is not None else 0. for r in records]
    meta['market_valid']=[bool(pd.isna(v.iloc[r['i']]['fail']) and not v.iloc[r['i']].stale_px) for r in records]
    meta['fair_raw']=meta.item.map(a.FAIR_VALUE);meta['issue_price']=meta.item.map(a.ISU_PRC_DETAIL)
    meta['common_face_eligible']=meta.issue_price.between(9000,10000)&meta.fair_raw.gt(0)
    meta.to_csv(OUT/'contract_audit.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    assert len(meta)+len(e)==len(s) and meta.item.is_unique
    summary=dict(total=len(s),contract_supported=len(meta),market_and_contract_supported=int(meta.market_valid.sum()),excluded_contract_reasons=e.reason.value_counts().to_dict(),structure_counts=meta.groupby(['structure','monthly']).size().rename('n').reset_index().to_dict('records'),payment_sources=meta.payment_schedule_source.value_counts().to_dict(),lag_counts=meta.lag.value_counts().to_dict(),positive_survival_coupon=int(meta.survival_coupon.gt(0).sum()),payment_assumption='Except six individually checked disclosures: infer unique 1..5 Korean bank-day lag from final EXER_DT to MAT_DT; use that same lag on other redemption and coupon payments. Same-day earned monthly coupon is retained. These are explicit assumptions, not individually verified terms.',monitoring='Daily calendar-time grid, same as baseline; no new claim of exchange-specific daily close calendars, continuous barrier monitoring or quanto calibration.',source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'data/cache/raw/LAKE_V2_DART_AUTO_CALL.csv.gz',ROOT/'data/cache/raw/LAKE_V2_DART_SCHD_INFO.csv.gz',BASE/'rebuilt_universe.parquet',BASE/'fresh_market_inputs.parquet',BASE/'dividend_events.json.gz',HERE/'contracts.py',ROOT/'module/mc_contract_v2.py',ROOT/'module/mc_contract_v3.py',ROOT/'analysis/mc_monthly_v3_20260910/evidence/kr_bank_calendar.json']})
    (OUT/'input_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    if '--finalize-only' in sys.argv:finalize()
    else:main()
