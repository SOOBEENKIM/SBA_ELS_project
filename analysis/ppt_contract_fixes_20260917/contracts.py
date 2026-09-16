"""Raw DART contract adapter, independent of market-input estimation.

Uses explicit PMT_2 for KI survival and monthly PMT_1 cashflows.
Payment lags outside the six sourced contracts remain declared assumptions.
Adapted from the existing 20260914 population adapter, without its IV filter.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'analysis/mc_monthly_v3_20260910')]
from module.mc_contract_v3 import ContractV3
from calendar_rules import SettlementCalendar
cal=SettlementCalendar()
published={r['original_item']:r for r in json.loads((ROOT/'analysis/mc_monthly_v3_20260910/results/published_monthly_contracts.json').read_text())}
def need(ok,why):
    if not ok:raise ValueError(why)

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
