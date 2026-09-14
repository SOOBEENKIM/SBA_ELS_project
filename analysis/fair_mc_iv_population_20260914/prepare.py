"""Prepare original-product IV inputs for FAIR-vs-MC diagnostics (no interventions)."""
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict
from datetime import date
import hashlib,json,gzip,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'analysis/mc_monthly_v3_20260910')]
from module.mc_contract_v3 import ContractV3,MarketV2
from module import features as F
from calendar_rules import SettlementCalendar

INDEX_MAP={'^KS200':'.KS200','^GSPC':'.SPX','^HSCE':'.HSCE','^HSI':'.HSI',
           '^N225':'.N225','^NDX':'.NDX','^GDAXI':'.GDAXI','^STOXX50E':'.STOXX50','^SX7E':'.SX7E'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'data/cache';raw=cache/'raw'
    ac=pd.read_csv(raw/'LAKE_V2_DART_AUTO_CALL.csv',low_memory=False).set_index('ITEM_CD')
    sc=pd.read_csv(raw/'LAKE_V2_DART_SCHD_INFO.csv',low_memory=False)
    ud=pd.read_csv(raw/'LAKE_V2_DART_UDLY_INFO.csv',low_memory=False)
    ids=set(pd.read_parquet(ROOT/'data/els3_dataset.parquet').item)
    ac=ac[ac.index.isin(ids)].sort_index();need(len(ac)==len(ids)==58790,'universe_mismatch')
    gs={(it,int(t)):g.sort_values('SEQ') for (it,t),g in sc[sc.ITEM_CD.isin(ids)].groupby(['ITEM_CD','SCHD_TYPE'])}
    mapping=json.loads((cache/'udly_ticker_map.json').read_text())
    assets=ud[ud.ITEM_CD.isin(ids)].groupby('ITEM_CD',sort=False).UDLY_ID.apply(list).to_dict()
    iv=pd.concat([pd.read_csv(p,encoding='utf-8-sig') for p in sorted((ROOT/'data/raw/iv_daily_atm').glob('*.csv'))],ignore_index=True)
    iv=iv[np.isfinite(iv.iv)&iv.iv.gt(0)].copy();iv['calc_date']=pd.to_datetime(iv.calc_date)
    need(iv.groupby(['notion_ric','calc_date']).iv.nunique().max()==1,'conflicting_iv')
    iv=iv.drop_duplicates(['notion_ric','calc_date']).sort_values('calc_date')
    series={k:g.reset_index(drop=True) for k,g in iv.groupby('notion_ric')}
    rets={}
    for t in set(mapping.values())-{None,''}:
        p=cache/('px_'+t.replace('^','_').replace('.','_')+'.parquet')
        if p.exists():
            x=pd.read_parquet(p)['close'].dropna();rets[t]=np.log(x[~x.index.duplicated()]).diff()
    curve=pd.read_parquet(cache/'krw_curve.parquet')[['call','m3','y10']]
    cal=SettlementCalendar()
    published={r['original_item']:r for r in json.loads((ROOT/'analysis/mc_monthly_v3_20260910/results/published_monthly_contracts.json').read_text())}

    @lru_cache(None)
    def market(ts,issue):
        matches=[]
        for t in ts:
            ric=INDEX_MAP.get(t,t)
            need(ric in series,'iv_missing_series')
            g=series[ric];j=int(np.searchsorted(g.calc_date.to_numpy(),np.datetime64(issue),side='right')-1)
            need(j>=0,'iv_before_history');r=g.iloc[j];age=(issue-r.calc_date).days
            need(0<=age<=7,'iv_stale')
            matches.append(dict(ticker=t,notion_ric=ric,source_ric=str(r.source_ric),iv=float(r.iv),iv_date=r.calc_date.date().isoformat(),age_days=age))
        need(all(t in rets for t in ts),'missing_returns')
        corr=F.corr180([rets[t] for t in ts],issue);beta=F.krw_beta(curve.asof(issue).values)
        need(corr is not None and beta is not None,'missing_corr_or_curve')
        m=MarketV2(tuple(x['iv'] for x in matches),tuple(map(tuple,corr)),tuple(beta)).validate()
        return asdict(m),matches

    def contract(it,r,g,q):
        # Individually reconciled terms take precedence over a missing raw field.
        if it in published:
            c=ContractV3(**published[it]['contract']).validate()
            return c,dict(payment_schedule_source='individual_disclosure',lag=published[it]['rule']['lag'])
        need(r.OPT_TYPE in ['STEP','LIZARD'] and r.RDMP_TYPE=='DOWN','unsupported_redemption_structure')
        need(r.CPN_YN in [0,1] and r.KNCK_IN_YN in [0,1],'unknown_structure_flag')
        for f in ['ISU_CALL_YN','PRCP_GRTE_RT','KNCK_IN_GRC_PRD']:
            need(pd.notna(r[f]) and r[f]==0,'unsupported_'+f)
        need(g is not None and 2<=len(g)<=12,'redemption_schedule_length')
        monthly=r.CPN_YN==1;ki=r.KNCK_IN_YN==1;coupon=float(r.ANL_RTRN)/100
        need(np.isfinite(coupon) and coupon>0,'nonpositive_annual_coupon')
        need(not monthly or (q is not None and 1<=len(q)<=60),'missing_monthly_schedule')
        if not monthly:need(q is None or q.empty,'unexpected_coupon_schedule')
        for frame in [g]+([q] if monthly else []):
            need(frame.SEQ.tolist()==list(range(1,len(frame)+1)),'incomplete_schedule_sequence')
            need(frame.STCK_MTHD.eq('ALL_MIN').all(),'not_worst_of')
            need(frame.AVG_DAYS.eq(1).all(),'averaging_not_supported')
            need(frame[['EXER_DT','STRK_1','PMT_1']].notna().all().all(),'missing_schedule_values')
            dt=pd.to_datetime(frame.EXER_DT,errors='coerce')
            need(dt.notna().all() and dt.is_monotonic_increasing and not dt.duplicated().any(),'invalid_evaluation_dates')
            for col in ['PMT_EXTRA','PMT_1_FRML','PMT_2_FRML','STRK_2','BARR_2','BARR_1_FMRL','BARR_2_FMRL']:
                need(frame[col].isna().all(),'extra_formula_or_level_'+col)
        need(g.iloc[:-1].PAY_OFF_TYPE.eq('Digital_Call').all() and g.iloc[-1].PAY_OFF_TYPE=='Digital_Call_Put','unsupported_redemption_payoff')
        need(g.iloc[:-1].PART_RTO.eq(1).all() and g.iloc[-1].PART_RTO==-1,'unsupported_participation')
        need(g.iloc[:-1].PMT_2.fillna(0).eq(0).all(),'nonterminal_second_payment')
        active=g.LZRD_BARR.notna()
        need(bool(active.any())==(r.OPT_TYPE=='LIZARD'),'lizard_structure_mismatch')
        need(g.loc[active,'LZRD_TERM'].eq('FROM_ISU').all(),'unsupported_lizard_window')
        need(g.loc[active,'LZRD_PMT'].notna().all(),'missing_lizard_payment')
        need(g.loc[~active,'LZRD_PMT'].fillna(0).eq(0).all(),'lizard_payment_without_clause')
        if ki:
            need(g.BARR_1.notna().any() and g.BARR_1.dropna().nunique()==1,'missing_or_varying_ki')
            need(g.loc[g.BARR_1.notna(),'BARR_TERM'].eq('FROM_ISU').all(),'unsupported_ki_window')
            need(pd.notna(g.iloc[-1].PMT_2),'missing_explicit_survival_coupon')
        else:
            need(g.BARR_1.isna().all(),'barrier_without_ki')
            need(pd.isna(g.iloc[-1].PMT_2) or g.iloc[-1].PMT_2==0,'second_maturity_payment_without_ki')
        issue=pd.Timestamp(r.ISU_DT)
        offsets=lambda frame:tuple((pd.to_datetime(frame.EXER_DT)-issue).dt.days.astype(int))
        obs=offsets(g);need(min(obs)>0,'evaluation_before_issue')
        if monthly:
            need(q.PAY_OFF_TYPE.eq('Digital_Call').all(),'unsupported_monthly_payoff')
            need(q.PMT_2.fillna(0).eq(0).all(),'second_monthly_payment')
            need(q.LZRD_BARR.isna().all() and q.BARR_1.isna().all(),'path_dependent_monthly_coupon')
            need(q.STRK_1.ge(0).all() and q.PMT_1.ge(0).all(),'invalid_monthly_cashflow')
            need(min(offsets(q))>0 and max(offsets(q))<=obs[-1],'monthly_evaluation_outside_life')
        candidates=[n for n in [1,2,3,4,5] if cal.advance(str(g.iloc[-1].EXER_DT),n).isoformat()==str(r.MAT_DT)]
        need(len(candidates)==1,'payment_lag_not_identifiable_from_maturity')
        lag=candidates[0]
        pay=lambda frame:tuple((cal.advance(str(d),lag)-issue.date()).days for d in frame.EXER_DT)
        c=ContractV3(name='actual_'+it,structure=str(r.OPT_TYPE)+(' KI' if ki else ' no-KI'),
            obs_days=obs,pay_days=pay(g),strikes=tuple(g.STRK_1/100),coupon=coupon,
            coupon_accruals=tuple(g.PMT_1/coupon),lz_barr=tuple(None if pd.isna(x) else float(x)/100 for x in g.LZRD_BARR),
            lz_pmts=tuple(g.LZRD_PMT.fillna(0)),ki_barrier=float(g.BARR_1.dropna().iloc[0])/100 if ki else None,
            survival_coupon_accrual=float(g.iloc[-1].PMT_2)/coupon if ki else None,
            monthly_obs_days=offsets(q) if monthly else (),monthly_pay_days=pay(q) if monthly else (),
            monthly_barriers=tuple(q.STRK_1/100) if monthly else (),monthly_accruals=tuple(q.PMT_1/coupon) if monthly else (),
            coupon_on_redemption_day=True if monthly else None).validate()
        np.testing.assert_allclose(c.pmts,g.PMT_1,rtol=0,atol=1e-12)
        if ki:need(abs(c.survival_pmt-g.iloc[-1].PMT_2)<1e-12,'survival_cashflow_mismatch')
        if monthly:np.testing.assert_allclose(c.monthly_pmts,q.PMT_1,rtol=0,atol=1e-12)
        return c,dict(payment_schedule_source='same_bank_lag_inferred_from_last_evaluation_to_raw_maturity',lag=lag)

    records=[];excluded=[];groups={}
    for i,(it,r) in enumerate(ac.iterrows(),1):
        try:
            ts=tuple(sorted(mapping.get(x,'') for x in assets[it]));need(len(ts)==3 and all(ts),'underlying_mapping')
            m,matches=market(ts,pd.Timestamp(r.ISU_DT))
            need(9000<=r.ISU_PRC_DETAIL<=10000 and np.isfinite(r.FAIR_VALUE) and r.FAIR_VALUE>0,'unsupported_price_denomination')
            c,terms=contract(it,r,gs.get((it,1)),gs.get((it,2)))
            key=hashlib.sha256(json.dumps([str(r.ISU_DT),ts,m],sort_keys=True).encode()).hexdigest()[:20]
            if key not in groups:groups[key]=dict(group=key,market=m,issue=str(r.ISU_DT),tickers=ts,iv_matches=matches,records=[])
            z=dict(item=it,structure=c.structure,monthly=c.has_monthly,issue=str(r.ISU_DT),
                   issue_price_krw=float(r.ISU_PRC_DETAIL),fair_raw_krw=float(r.FAIR_VALUE),face_krw=10000,
                   group=key,contract=asdict(c),**terms)
            groups[key]['records'].append(z);records.append(z)
        except ValueError as e:
            excluded.append(dict(item=it,reason=str(e)))
        if i%10000==0:print('PREPARE',i,'/',len(ac),'included',len(records),'groups',len(groups),flush=True)
    need(len(records)+len(excluded)==len(ids),'coverage_accounting')
    pd.DataFrame(excluded).to_csv(OUT/'excluded_products.csv',index=False)
    meta=pd.DataFrame([{k:v for k,v in r.items() if k!='contract'} for r in records]);meta.to_csv(OUT/'products.csv',index=False)
    payload=json.dumps(list(groups.values()),ensure_ascii=False,allow_nan=False).encode()
    (OUT/'groups.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    files=[ROOT/'module/mc_contract_v2.py',ROOT/'module/mc_contract_v3.py',ROOT/'module/features.py',
           ROOT/'analysis/mc_schedule_v4_20260911/schedule_mc.py',
           ROOT/'analysis/mc_monthly_v3_20260910/evidence/kr_bank_calendar.json',
           ROOT/'data/raw/iv_daily_atm/iv_daily_atm.zip',cache/'krw_curve.parquet',
           *[raw/f'LAKE_V2_DART_{n}.csv' for n in ['AUTO_CALL','SCHD_INFO','UDLY_INFO']]]
    protocol=dict(target='Original-product published FAIR vs IV-based MC; no DeepONet and no contract interventions',
        paths_per_seed=40000,replicates=3,checkpoints=[10000,20000,40000],workers=16,
        universe=len(ids),included=len(records),excluded=len(excluded),market_groups=len(groups),
        iv_policy='Same IV branch mapping: all three ATM IV as of issue date, age <=7 calendar days; no HV fallback',
        market_policy='Existing branch GBM: 180-return historical correlation, KRW NS curve, q=0, flat ATM IV, daily calendar grid',
        cashflows='Raw evaluation dates, strikes, PMT_1; explicit terminal PMT_2 for KI survival; raw monthly Digital_Call cashflows; FROM_ISU Lizard clauses',
        settlement_assumption='For unsourced products infer a unique 1..5 KR bank-day lag from last evaluation to MAT_DT, then apply it to all payments; no claim that all payment clauses were individually checked',
        monthly_assumption='Same-day earned monthly coupon is retained on redemption, as in current V3; memory coupons are not inferred',
        face_assumption='KRW 10,000 face for raw issue-price denominations 9,000..10,000; six individually sourced faces verified; remaining faces not independently reconciled',
        normalization='fair=FAIR_VALUE/issue_price; mc_iv=MC_price_per_10000_face/issue_price. Both are on the same issue-price basis',
        group_reuse='Same issue, ordered assets, IV/correlation/curve share market paths; each contract retains its own dates/horizon/payoff',
        limitations='FAIR disclosure valuation date may precede issue date used for market inputs; price gap is not proof of a pricing bug or a causal effect',
        preserved_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},groups_sha256=sha(OUT/'groups.json.gz'))
    dump(HERE/'protocol.json',protocol)
    audit=dict(included=len(records),excluded=len(excluded),groups=len(groups),
               excluded_reason_counts=pd.Series([r['reason'] for r in excluded]).value_counts().to_dict(),
               structure_counts=meta.groupby(['structure','monthly']).size().rename('n').reset_index().to_dict('records'),
               lag_counts=meta.lag.value_counts().sort_index().to_dict(),
               published_included=meta.payment_schedule_source.eq('individual_disclosure').sum().item(),seconds=time.time()-start)
    dump(OUT/'input_audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
