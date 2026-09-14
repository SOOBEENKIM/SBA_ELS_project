"""Freeze the actual inputs behind the presentation; never infer them from chart values."""
from pathlib import Path
from functools import lru_cache
from datetime import date
import gzip, json, hashlib, sys, zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import reference_engine as E
REF=HERE/'reference'; OUT=HERE/'results'
IVMAP={'^KS200':'.KS200','^GSPC':'.SPX','^HSCE':'.HSCE','^HSI':'.HSI',
       '^N225':'.N225','^NDX':'.NDX','^GDAXI':'.GDAXI','^STOXX50E':'.STOXX50'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    OUT.mkdir(exist_ok=True)
    # A published or resumable run keeps the exact compressed input bytes and
    # protocol. Rebuilding equal JSON with a new gzip timestamp breaks its hash.
    if (OUT/'population_jobs.json.gz').exists():
        protocol=json.loads((HERE/'protocol.json').read_text())
        for name,path in {'population':REF/'population_58790.parquet',
                          'prices':REF/'mc_iv_full.csv',
                          'iv_zip':ROOT/'data/raw/iv_daily_atm/iv_daily_atm.zip',
                          'jobs':OUT/'population_jobs.json.gz'}.items():
            assert sha(path)==protocol['files'][name],f'Frozen input changed: {name}'
        print('Frozen input files verified and reused. For a new run use fresh_run.py.',flush=True)
        return
    d=pd.read_parquet(REF/'population_58790.parquet').sort_values('isu_ord').reset_index(drop=True)
    saved=pd.read_csv(REF/'mc_iv_full.csv').set_index('item').loc[d.item].reset_index()
    assert len(d)==len(saved)==58790 and d.item.is_unique
    assert set(d.item)==set(pd.read_parquet(ROOT/'data/els3_dataset.parquet').item)
    for k in ['fair','mc']:np.testing.assert_allclose(d[k],saved[k],atol=1e-7,rtol=0)
    E.CACHE=ROOT/'data/cache'; rets=E.load_RET(pd.unique(d[['udl1','udl2','udl3']].values.ravel()))
    zpath=ROOT/'data/raw/iv_daily_atm/iv_daily_atm.zip'
    with zipfile.ZipFile(zpath) as z:
        names=sorted(x for x in z.namelist() if x.endswith('.csv') and not x.startswith('__MACOSX'))
        big=pd.concat([pd.read_csv(z.open(x)) for x in names],ignore_index=True)
    big['calc_date']=pd.to_datetime(big.calc_date)
    iv={}
    for ric,g in big.groupby('notion_ric'):
        s=g.set_index('calc_date').iv.sort_index();iv[ric]=s[~s.index.duplicated(keep='last')]
    assert np.isfinite(big.iv).all() and (big.iv>0).all()
    hvcurve=pd.read_parquet(REF/'krw_curve.parquet')[['call','m3','y10']]
    ivcurve=pd.read_parquet(REF/'krw_curve_fred.parquet')[['call','m3','y10']]
    @lru_cache(None)
    def market(ts,ordinal):
        dt=pd.Timestamp(date.fromordinal(ordinal)); hv=[E.vol180(rets[t],dt) for t in ts]
        corr=E.corr180([rets[t] for t in ts],dt); sig=[]; detail=[]
        for t,h in zip(ts,hv):
            ric=IVMAP.get(t);s=iv.get(ric);sub=s[s.index<=dt] if s is not None else None
            use=sub is not None and len(sub)>0 and not np.isnan(sub.iloc[-1])
            value=float(sub.iloc[-1]) if use else h;sig.append(value)
            detail.append(dict(ticker=t,ric=ric,source='IV' if use else 'HV',sigma=value,
                               iv_date=sub.index[-1].date().isoformat() if use else None,
                               iv_age_days=int((dt-sub.index[-1]).days) if use else None))
        assert corr is not None and np.isfinite(hv+sig).all()
        return dict(hv=hv,iv=sig,corr=corr.tolist(),hv_rates=hvcurve.asof(dt).tolist(),
                    iv_rates=ivcurve.asof(dt).tolist(),detail=detail)
    jobs=[];audit=[]
    for i,r in enumerate(d.itertuples(index=False)):
        rd=r._asdict();ts=tuple(rd[f'udl{j}'] for j in [1,2,3]);m=market(ts,int(r.isu_ord));n=int(r.nobs)
        get=lambda prefix:[float(rd[f'{prefix}_{j}']) for j in range(n)]
        lzb=get('lz_barr');lzp=get('lz_pmt')
        j=dict(item=r.item,seed=int(r.mc_seed),structure=r.opt_type+(' KI' if r.ki_yn else ' no-KI'),
               monthly=bool(r.imonth),issue=date.fromordinal(int(r.isu_ord)).isoformat(),tickers=list(ts),market=m,
               contract=dict(B=float(r.B),strikes=get('strk'),ten=float(r.tenor),c=float(r.coupon),pmts=get('pmt'),
                             lz_barr=[None if np.isnan(x) else x for x in lzb],
                             lz_pmt=[0. if np.isnan(x) else x for x in lzp]))
        jobs.append(j);niv=sum(x['source']=='IV' for x in m['detail'])
        audit.append(dict(item=r.item,issue=j['issue'],structure=j['structure'],monthly=j['monthly'],n_iv_udl=niv,
                          max_iv_age_days=max([x['iv_age_days'] for x in m['detail'] if x['iv_age_days'] is not None],default=0)))
        if (i+1)%10000==0:print('PREPARED',i+1,flush=True)
    audit=pd.DataFrame(audit)
    np.testing.assert_array_equal(audit.n_iv_udl,saved.n_iv_udl)
    with gzip.open(OUT/'population_jobs.json.gz','wt') as f:json.dump(jobs,f,allow_nan=False,separators=(',',':'))
    audit.to_csv(OUT/'input_assignment.csv',index=False)
    saved.to_csv(OUT/'reference_population_prices.csv',index=False)
    old=pd.read_parquet(REF/'legacy_23151.parquet')
    old[['item','fair','mc']].to_csv(OUT/'reference_legacy_prices.csv',index=False)
    protocol=dict(version='ppt_reference_20260915',source_trace='5_MC_implied_vol.ipynb cells 23/24 and mc_iv_full.csv; earlier 23151 chart is separate.',
      population=58790,legacy_chart_population=23151,paths_per_seed=40000,population_replicates=1,
      seed='Exact saved per-product mc_seed',device='Selected explicitly by --device; actual device is recorded in each completion and engine-validation JSON',path_chunk=20000,tblock=512,
      hv='180 pre-issue returns, std(ddof=1)*sqrt(252); minimum 60',correlation='180 common pre-issue returns; Pearson clipped off-diagonal to [-.999,.999]; source PSD repair',
      iv='Eight index mappings only; last observation <= issue date; no maximum age; missing IV uses HV per asset; no tenor or smile interpolation',
      discount=dict(hv='Nelson-Siegel lambda=1.5 from frozen krw_curve.parquet',iv='piecewise-linear zero rates (called bootstrap in source) from frozen krw_curve_fred.parquet',iv_ns_control='NS using same IV/FRED inputs'),
      drift='Integrated forwards from the same discount curve; q=0',daily_grid='1/365; round(tenor*365)',
      payment='Uniformly spaced redemption dates; discount on observation date; source per-round pmts; regular redemption before Lizard; no-KI-event terminal principal only; monthly linear-payment approximation retained',
      source_plot_units='Exactly (saved FAIR/issue-price ratio - unit-face MC ratio)*10000; not silently re-normalized. Not a verified common-cash-basis comparison when issue price differs from face.',
      modifications=['40000 paths explicitly requested instead of source IV100000','CPU and CUDA generator streams differ; device caches are kept separate'],
      synthetic_extension_status='pending payoff-policy clarification; existing main/IV experiments preserved',
      files={'population':sha(REF/'population_58790.parquet'),'prices':sha(REF/'mc_iv_full.csv'),'iv_zip':sha(zpath),'jobs':sha(OUT/'population_jobs.json.gz')})
    dump(HERE/'protocol.json',protocol)
    report=dict(status='pass',products=len(d),market_states=market.cache_info().currsize,iv_files=len(names),iv_rows=len(big),
      iv_count=audit.n_iv_udl.value_counts().sort_index().to_dict(),max_iv_age=int(audit.max_iv_age_days.max()),
      legacy_mean=float(((old.fair-old.mc)*10000).mean()),legacy_median=float(((old.fair-old.mc)*10000).median()),
      population_hv_mean=float(((saved.fair-saved.mc)*10000).mean()),population_iv_mean=float(((saved.fair-saved.mc_iv)*10000).mean()),
      iv_assignment_matches_archived=True,source_price_input_ids_match=True)
    dump(OUT/'input_validation.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
