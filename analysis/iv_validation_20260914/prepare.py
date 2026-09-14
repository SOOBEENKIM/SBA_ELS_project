"""Attach observed historical/ATM-IV market states to frozen test contracts."""
from pathlib import Path
from functools import lru_cache
from datetime import date
import sys,json,hashlib,platform
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V3=ROOT/'analysis/mc_monthly_v3_20260910'
V4=ROOT/'analysis/mc_schedule_v4_20260911'
OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(V3)]
from common import unpack,cases,encode
from module import features as F

INDEX_MAP={'^KS200':'.KS200','^GSPC':'.SPX','^HSCE':'.HSCE','^HSI':'.HSI',
           '^N225':'.N225','^NDX':'.NDX','^GDAXI':'.GDAXI','^STOXX50E':'.STOXX50','^SX7E':'.SX7E'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def preserved():
    files=[ROOT/'module'/n for n in ['mc_contract_v2.py','mc_contract_v3.py','features.py','networks.py']]
    files += [V3/n for n in ['common.py','models.py','protocol.json']]
    files += [V3/'results'/n for n in ['families.json','model_selection.json','train.npz','validation.npz','test.npz','stress.npz','mc_labels.csv']]
    files += [V4/'schedule_mc.py',V4/'results/families.json']+sorted((V3/'models').glob('*.pt'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def main():
    OUT.mkdir(exist_ok=True)
    raw=ROOT/'data/raw/iv_daily_atm'
    frames=[]
    for p in sorted(raw.glob('*.csv')):
        d=pd.read_csv(p,encoding='utf-8-sig');assert set(d)=={'calc_date','notion_ric','source_ric','iv'}
        assert d.calc_date.eq(p.stem).all(),p
        frames.append(d)
    iv=pd.concat(frames,ignore_index=True);iv['calc_date']=pd.to_datetime(iv.calc_date)
    invalid=~np.isfinite(iv.iv)|(iv.iv<=0)
    iv[invalid].to_csv(OUT/'invalid_iv_rows.csv',index=False)
    iv=iv[~invalid].copy()
    dup=iv.duplicated(['calc_date','notion_ric'],keep=False)
    if dup.any():
        assert iv[dup].groupby(['calc_date','notion_ric']).iv.nunique().max()==1,'Conflicting IV rows'
    iv=iv.drop_duplicates(['calc_date','notion_ric']).sort_values(['notion_ric','calc_date'])
    iv[iv.iv>3].to_csv(OUT/'iv_above_300_percent.csv',index=False)
    idx={ric:g.reset_index(drop=True) for ric,g in iv.groupby('notion_ric')}
    arrays={k:g.calc_date.to_numpy() for k,g in idx.items()}
    def lookup(t,dt):
        ric=INDEX_MAP.get(t,t if t in idx else None)
        if ric not in idx:return None,'missing_mapping_or_series'
        j=int(np.searchsorted(arrays[ric],np.datetime64(dt),side='right')-1)
        if j<0:return None,'no_iv_at_or_before_issue'
        r=idx[ric].iloc[j];age=(dt-r.calc_date).days
        if age>7:return None,'iv_older_than_7_calendar_days'
        # No unannounced proxy mapping. Delivered notion/source aliases are retained.
        return dict(ticker=t,notion_ric=ric,source_ric=str(r.source_ric),iv=float(r.iv),
                    iv_date=r.calc_date.date().isoformat(),age_days=age),None

    cache=ROOT/'data/cache';rdir=cache/'raw'
    ac=pd.read_csv(rdir/'LAKE_V2_DART_AUTO_CALL.csv',usecols=['ITEM_CD','ISU_DT','FAIR_VALUE','ISU_PRC_DETAIL'])
    assert not ac.ITEM_CD.duplicated().any()
    ud=pd.read_csv(rdir/'LAKE_V2_DART_UDLY_INFO.csv',usecols=['ITEM_CD','UDLY_ID'])
    ticker_map=json.loads((cache/'udly_ticker_map.json').read_text())
    udls=ud.groupby('ITEM_CD',sort=False).UDLY_ID.apply(list).to_dict()
    universe=set(pd.read_parquet(ROOT/'data/els3_dataset.parquet').item)
    ac=ac[ac.ITEM_CD.isin(universe)].copy();ac['issue']=pd.to_datetime(ac.ISU_DT)
    meta={};coverage=[]
    for r in ac.sort_values('ITEM_CD').itertuples():
        ts=tuple(ticker_map.get(u) for u in udls[r.ITEM_CD]);reasons=[];matches=[]
        if len(ts)!=3 or None in ts:reasons.append('not_three_mapped_assets')
        else:
            for t in ts:
                m,why=lookup(t,r.issue)
                if why:reasons.append(t+':'+why)
                else:matches.append(m)
        complete=not reasons
        coverage.append(dict(item=r.ITEM_CD,issue=r.issue.date().isoformat(),full_iv=complete,n_iv=len(matches),reason=';'.join(reasons)))
        meta[r.ITEM_CD]=dict(item=r.ITEM_CD,issue=r.issue,tickers=ts,iv_matches=matches,full_iv=complete,
                            fair=float(r.FAIR_VALUE/r.ISU_PRC_DETAIL*10000))
    pd.DataFrame(coverage).to_csv(OUT/'universe_iv_coverage.csv',index=False)
    rets={}
    for t in set(ticker_map.values())-{None,''}:
        p=cache/('px_'+t.replace('^','_').replace('.','_')+'.parquet')
        if p.exists():
            px=pd.read_parquet(p)['close'].dropna();rets[t]=np.log(px[~px.index.duplicated()]).diff()
    curve=pd.read_parquet(cache/'krw_curve.parquet')[['call','m3','y10']]
    @lru_cache(None)
    def historical(ts,dt):
        if any(t not in rets for t in ts):raise ValueError('missing_returns')
        rs=[rets[t] for t in ts];sig=[F.vol180(x,dt) for x in rs];corr=F.corr180(rs,dt);beta=F.krw_beta(curve.asof(dt).values)
        if not np.isfinite(sig).all() or corr is None or beta is None:raise ValueError('missing_historical_market')
        m=dict(sigs=sig,corr=corr.tolist(),beta=beta.tolist(),dividend_yields=[0.,0.,0.])
        return m

    fs=json.loads((V3/'results/families.json').read_text())
    schedule={r['family']:r for r in json.loads((V4/'results/families.json').read_text())}
    selected=[r for r in fs if r['split']=='test' or r['origin']=='published_monthly']
    pool=sorted([k for k,v in meta.items() if v['full_iv']])
    assert len(pool)>128
    rng=np.random.default_rng(2026091401)
    donors=iter(rng.permutation(pool).tolist());used=set();new=[];excluded=[];market_rows=[]
    from copy import deepcopy
    from dataclasses import asdict
    for r in selected:
        item=r.get('template_item',r.get('original_item'))
        assigned=item is None
        if assigned:
            while True:
                item=next(donors)
                if item not in used:used.add(item);break
        info=meta[item]
        if not info['full_iv']:
            excluded.append(dict(family=r['family'],item=item,origin=r['origin'],reason='full_three_asset_fresh_IV_unavailable'));continue
        try:hv=historical(info['tickers'],info['issue'])
        except ValueError as e:
            excluded.append(dict(family=r['family'],item=item,origin=r['origin'],reason=str(e)));continue
        im=deepcopy(hv);im['sigs']=[x['iv'] for x in info['iv_matches']]
        c,_=unpack(r)
        vv=[dict(axis=a,offset=h,contract=asdict(cc)) for a,h,cc in cases(c)]
        sr=schedule[r['family']]
        assert sr['variants'][0]['contract']==r['contract']
        vv += [dict(axis=x['axis'],offset=x['offset'],contract=x['contract']) for x in sr['variants'] if x['axis']!='base']
        f=dict(family=r['family'],origin=r['origin'],monthly=r['monthly'],structure=r['structure'],contract=r['contract'],
               market=hv,markets={'HV':hv,'IV':im},original_market=r['market'],variants=vv,
               market_item=item,issue=info['issue'].date().isoformat(),tickers=info['tickers'],iv_matches=info['iv_matches'],
               market_assignment='deterministic_actual_market_donor' if assigned else 'original_template_asset_date',
               fair_krw=info['fair'] if r['origin']=='published_monthly' else None)
        for arm,m in f['markets'].items():
            for v in vv:
                cc,mm=unpack(dict(contract=v['contract'],market=m));encode(cc,mm)
            market_rows.append(dict(family=f['family'],arm=arm,origin=f['origin'],monthly=f['monthly'],item=item,issue=f['issue'],
                min_sigma=min(m['sigs']),max_sigma=max(m['sigs']),sigma_outside_training=any(s<.12 or s>.45 for s in m['sigs']),
                max_iv_age=max(x['age_days'] for x in f['iv_matches']),tickers='|'.join(info['tickers'])))
        new.append(f)
    dump(OUT/'families.json',new)
    pd.DataFrame(excluded,columns=['family','item','origin','reason']).to_csv(OUT/'excluded_families.csv',index=False)
    pd.DataFrame(market_rows).to_csv(OUT/'market_assignment.csv',index=False)
    selection=json.loads((V3/'results/model_selection.json').read_text())
    for name,h in selection['checkpoint_hashes'].items():assert sha(V3/'models'/name)==h
    protocol=dict(version='iv_validation_20260914',notional=10000,paths_per_seed=40000,replicates=3,
        checkpoints=[10000,20000,40000],workers=12,path_chunk=2500,tblock=256,
        question='Same corrected GBM and held-out contract templates; replace historical sigma with delivered flat ATM IV and evaluate frozen Stage-1 price/increment errors.',
        market_policy='PPT-style 180-trading-return HV and correlation; same KRW NS curve/q=0 in HV and IV arms; only sigma changes between arms.',
        iv_asof='Latest calc_date <= issue date, maximum age 7 calendar days, all three assets required. Daily as-of comparison, not intraday point-in-time data certification.',
        iv_units='Annualized decimal volatility used as delivered; 0.20 is 20%. Expiry/tenor, option quote filters, vendor construction are not provided.',
        iv_maturity='Single flat ATM proxy; no smile or term structure inferred, including maturity interventions.',
        missing_iv='Exclude from paired experiment, record reason; never substitute HV into IV arm.',
        underlying_mapping=INDEX_MAP,stock_mapping='Exact ticker=notion_ric only; retain supplied source_ric aliases.',
        regular_market_assignment='Uniform deterministic draw without replacement from eligible source products, seed 2026091401. Contract dates remain the original explicit synthetic relative schedule.',
        monthly_market_assignment='Use original template asset identities and issue date; contracts remain synthetic unless published_monthly.',
        comparison_to_old_run='New market-state evaluation on frozen test contracts. Neither HV nor IV cohort is identical to prior random-sigma test. Old aggregate metrics are not the matched HV baseline.',
        contract_interventions='All V3 coupon/KI/first/last/monthly-barrier bumps plus existing V4 observation/maturity variants, unchanged.',
        random_numbers='Same underlying Gaussian draws for HV/IV and every contract variant within each family/seed. Paired MC standard errors for deltas and IV-HV delta differences.',
        frozen_models=dict(arms=['augmented_price','augmented_delta','affine_coupon_delta'],seeds=[47,101,233],primary=selection['selected_arm'],retraining=False),
        scope='Contract payoff unchanged; no new dividend/quanto/smile/curve calibration. Synthetic prices are not observed FAIR counterfactuals.',
        source_zip_sha256=sha(raw/'iv_daily_atm.zip'),preserved_inputs=preserved())
    dump(HERE/'protocol.json',protocol)
    audit=dict(status='pass',csv_files=len(frames),valid_rows=len(iv),invalid_rows=int(invalid.sum()),duplicate_rows=int(dup.sum()),
       first_date=iv.calc_date.min().date().isoformat(),last_date=iv.calc_date.max().date().isoformat(),notion_series=len(idx),
       iv_over_300_percent_rows=int((iv.iv>3).sum()),universe_rows=len(meta),full_fresh_iv_products=len(pool),
       requested_families=len(selected),included_families=len(new),excluded_families=len(excluded),
       included_by_origin=pd.Series([r['origin'] for r in new]).value_counts().to_dict(),
       cases_per_arm=sum(len(r['variants']) for r in new),runtime_python=platform.python_version(),
       no_old_mc_labels_used=True,no_contract_payoff_changes=True)
    dump(OUT/'input_audit.json',audit);print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
