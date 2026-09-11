"""Audit every monthly raw record; reconstruct the six sourced contracts."""
from pathlib import Path
from datetime import date
import sys,json,re,hashlib
import pandas as pd
import numpy as np
from dataclasses import asdict

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];OUT=HERE/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE/'vendor'))
import holidays
from calendar_rules import SettlementCalendar
from module.mc_contract_v3 import ContractV3,MarketV2

SOURCE_RULES={
 'KR6SH0006Y24':dict(lag=3,coupon=.0693,monthly_rate=.005775,monthly_barrier=.65,ki=None),
 'KR6KS0003LQ6':dict(lag=3,coupon=.081,monthly_rate=.00675,monthly_barrier=.60,ki=.50),
 'KR6SH0004FV2':dict(lag=2,coupon=.06,monthly_rate=.005,monthly_barrier=.65,ki=None),
 'KR6523325910':dict(lag=3,coupon=.0642,monthly_rate=.00535,monthly_barrier=.60,ki=None),
 'KR6MD0003TN4':dict(lag=2,coupon=.0861,monthly_rate=.007175,monthly_barrier=.60,ki=None),
 'KR6KS0002XR1':dict(lag=2,coupon=.0738,monthly_rate=.00615,monthly_barrier=.55,ki=None)}

def dates(g,issue):return tuple((pd.to_datetime(g.EXER_DT)-pd.Timestamp(issue)).dt.days.to_numpy(int).tolist())

def main():
    h=holidays.country_holidays('KR',years=range(2015,2032),categories=('public','bank'),language='en_US')
    spec=dict(provider='python-holidays',version=holidays.__version__,categories=['public','bank'],
        first_year=2015,last_year=2031,closed_days={d.isoformat():name for d,name in sorted(h.items())},
        source='https://holidays.readthedocs.io/en/latest/auto_gen_docs/south_korea/',
        scope='Korean bank settlement only; raw scheduled exchange evaluation dates retained; future exceptional closures require explicit updates')
    (HERE/'evidence/kr_bank_calendar.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2))
    cal=SettlementCalendar()
    # Independent weekday/known public holiday boundary examples.
    assert str(cal.advance('2024-04-26',3))=='2024-05-02' # weekend and bank Labor Day
    assert str(cal.advance('2024-09-13',2))=='2024-09-20' # Chuseok
    assert str(cal.advance('2025-10-02',2))=='2025-10-13' # National Foundation/Chuseok/Hangul
    ac=pd.read_csv(PROJECT/'data/cache/raw/LAKE_V2_DART_AUTO_CALL.csv',low_memory=False)
    sc=pd.read_csv(PROJECT/'data/cache/raw/LAKE_V2_DART_SCHD_INFO.csv',low_memory=False)
    d=pd.read_parquet(PROJECT/'data/els3_dataset.parquet')
    a=ac[ac.ITEM_CD.isin(d.item)&ac.CPN_YN.eq(1)].set_index('ITEM_CD')
    s=sc[sc.ITEM_CD.isin(a.index)];groups={(it,int(t)):g.sort_values('SEQ') for (it,t),g in s.groupby(['ITEM_CD','SCHD_TYPE'])}
    rows=[];templates=[]
    for it,r in a.iterrows():
        g=groups.get((it,1));q=groups.get((it,2));reasons=[]
        if g is None or q is None:reasons.append('missing redemption or coupon schedule')
        else:
            if len(g)>12 or not 2<=len(g):reasons.append('unsupported redemption schedule length')
            if not 1<=len(q)<=60:reasons.append('unsupported coupon schedule length')
            if r.RDMP_TYPE!='DOWN' or r.OPT_TYPE not in ['STEP','LIZARD']:reasons.append('other redemption structure')
            if pd.notna(r.ISU_CALL_YN) and r.ISU_CALL_YN!=0:reasons.append('issuer call')
            if pd.notna(r.PRCP_GRTE_RT) and r.PRCP_GRTE_RT!=0:reasons.append('principal protection')
            if pd.notna(r.KNCK_IN_GRC_PRD) and r.KNCK_IN_GRC_PRD!=0:reasons.append('KI grace period')
            for frame in [g,q]:
                if not frame.STCK_MTHD.eq('ALL_MIN').all():reasons.append('other stock aggregation')
                if frame.AVG_DAYS.isna().any() or not frame.AVG_DAYS.eq(1).all():reasons.append('averaging or missing averaging rule')
                if frame[['EXER_DT','STRK_1','PMT_1']].isna().any().any():reasons.append('missing date/strike/payment')
                dt=pd.to_datetime(frame.EXER_DT,errors='coerce')
                if dt.isna().any() or not dt.is_monotonic_increasing or dt.duplicated().any():reasons.append('invalid or duplicate dates')
                if list(frame.SEQ)!=list(range(1,len(frame)+1)):reasons.append('incomplete schedule sequence')
            if not q.PAY_OFF_TYPE.eq('Digital_Call').all():reasons.append('other coupon condition')
            if not g.PMT_1.eq(0).all() or g.PMT_2.fillna(0).ne(0).any():reasons.append('nonzero redemption premium needs product-specific reconstruction')
            if g.LZRD_TERM.dropna().ne('FROM_ISU').any():reasons.append('other Lizard monitoring')
            if g.LZRD_PMT.fillna(0).ne(0).any():reasons.append('nonzero Lizard premium needs reconstruction')
            if r.ANL_RTRN<=0 or not np.isfinite(r.ANL_RTRN):reasons.append('missing positive annual coupon')
            if pd.to_datetime(q.EXER_DT,errors='coerce').min()<=pd.Timestamp(r.ISU_DT):reasons.append('coupon before inception')
            if pd.to_datetime(q.EXER_DT,errors='coerce').max()>pd.to_datetime(g.EXER_DT,errors='coerce').max():reasons.append('coupon after last redemption evaluation')
            if int(r.KNCK_IN_YN)==1 and g.BARR_1.dropna().empty:reasons.append('missing KI barrier')
        reasons=sorted(set(reasons));rows.append(dict(item=it,structure=str(r.OPT_TYPE)+(' KI' if r.KNCK_IN_YN else ' no-KI'),
            n_monthly=0 if q is None else len(q),n_redemption=0 if g is None else len(g),
            schedule_supported=not reasons,reasons='; '.join(reasons),receipt=str(int(r.RCEPT_NO))))
        if not reasons:
            issue=r.ISU_DT;ki=int(r.KNCK_IN_YN)==1;st=str(r.OPT_TYPE)+(' KI' if ki else ' no-KI')
            templates.append(dict(item=it,structure=st,issue=issue,raw_maturity=str(r.MAT_DT),
                obs_days=dates(g,issue),strikes=list(g.STRK_1/100),coupon=float(r.ANL_RTRN)/100,
                lz_barr=[None if pd.isna(x) else float(x)/100 for x in g.LZRD_BARR],lz_pmts=list(g.LZRD_PMT.fillna(0)),
                ki_barrier=float(g.BARR_1.dropna().min()/100) if ki else None,
                monthly_obs_days=dates(q,issue),monthly_barriers=list(q.STRK_1/100),monthly_pmts=list(q.PMT_1)))
    audit=pd.DataFrame(rows);audit.to_csv(OUT/'monthly_universe_audit.csv',index=False)
    (OUT/'monthly_templates.json').write_text(json.dumps(templates,ensure_ascii=False,indent=2))
    old={r['item']:r for r in json.loads((PROJECT/'analysis/contract_value_20260910/results/mc_selected_contracts.json').read_text()) if r['origin']=='real'}
    sources={r['item']:r for r in json.loads((HERE/'evidence/sources.json').read_text())}
    actual=[];checks=[]
    for it,rule in SOURCE_RULES.items():
        row=next(t for t in templates if t['item']==it);r=a.loc[it];issue=date.fromisoformat(row['issue']);g=groups[(it,1)];q=groups[(it,2)]
        src=sources[it]['candidates'];assert len(src)==1
        txt=(HERE/'evidence'/Path(src[0]['file']).with_suffix('.txt')).read_text()
        published_dates={f'{int(y):04d}-{int(m):02d}-{int(d):02d}' for y,m,d in re.findall(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',txt)}
        assert set(g.EXER_DT)|set(q.EXER_DT)<=published_dates,(it,'raw evaluation date not in source')
        assert abs(row['coupon']-rule['coupon'])<1e-12
        np.testing.assert_allclose(row['monthly_pmts'],rule['monthly_rate'],atol=1e-12)
        np.testing.assert_allclose(row['monthly_barriers'],rule['monthly_barrier'],atol=1e-12)
        assert row['ki_barrier']==rule['ki'];assert len(q)==36
        redpay=tuple((cal.advance(x,rule['lag'])-issue).days for x in g.EXER_DT)
        cpay=tuple((cal.advance(x,rule['lag'])-issue).days for x in q.EXER_DT)
        rawmat=(date.fromisoformat(str(r.MAT_DT))-issue).days
        # Holiday changes since issuance may move a scheduled maturity. Never
        # overwrite the discrepancy silently: retain both and follow the rule.
        c=ContractV3('published_'+it,row['structure'],tuple(row['obs_days']),redpay,tuple(row['strikes']),row['coupon'],
            (0.,)*len(g),tuple(row['lz_barr']),tuple(row['lz_pmts']),row['ki_barrier'],0. if rule['ki'] else None,
            monthly_obs_days=tuple(row['monthly_obs_days']),monthly_pay_days=cpay,
            monthly_barriers=tuple(row['monthly_barriers']),monthly_accruals=tuple(q.PMT_1/row['coupon']),coupon_on_redemption_day=True).validate()
        m=MarketV2(**old[it]['market']);m.validate()
        actual.append(dict(family=c.name,split='stress',origin='published_monthly',structure=c.structure,
            original_item=it,contract=asdict(c),market=asdict(m),source=src[0],rule=rule,
            scope='Published scheduled evaluation dates and separate coupon terms; Korean bank lag applied. GBM daily-calendar-grid market, no exchange-session dynamics or disruption model.'))
        checks.append(dict(item=it,monthly_rows=len(q),all_evaluation_dates_in_published_source=True,
            published_monthly_payment=rule['monthly_rate'],annual_coupon=rule['coupon'],coupon_barrier=rule['monthly_barrier'],
            lag_business_days=rule['lag'],raw_maturity_day=rawmat,calendar_maturity_day=redpay[-1],
            maturity_date_matches=rawmat==redpay[-1],source=src[0]['url']))
    (OUT/'published_monthly_contracts.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2))
    (OUT/'published_contract_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    summary=dict(monthly_products=len(a),coupon_schedule_rows=int((s.SCHD_TYPE==2).sum()),
        raw_schedule_supported=int(audit.schedule_supported.sum()),requires_additional_reconstruction=int((~audit.schedule_supported).sum()),
        supported_structure_counts=audit[audit.schedule_supported].structure.value_counts().to_dict(),
        published_contracts_reconstructed=len(actual),settlement_calendar_version=holidays.__version__,
        calendar_tests=3,universe_pricing_claim='Raw schema audit only for full universe; six products have individually checked published payment rules')
    (OUT/'input_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.DataFrame(checks).drop(columns='source').to_string(index=False))

if __name__=='__main__':main()
