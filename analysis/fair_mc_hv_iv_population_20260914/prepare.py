"""Attach 180-return HV to the frozen original-product IV cohort."""
from copy import deepcopy
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'results'
BASE=ROOT/'analysis/fair_mc_iv_population_20260914'
sys.path.insert(0,str(ROOT))
from module import features as F
from module.mc_contract_v3 import MarketV2

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    base_cfg=json.loads((BASE/'protocol.json').read_text())
    assert json.loads((BASE/'results/completion.json').read_text())['status']=='complete'
    groups=json.loads(gzip.decompress((BASE/'results/groups.json.gz').read_bytes()))
    tickers=sorted({t for g in groups for t in g['tickers']}); rets={}; paths=[]
    for t in tickers:
        p=ROOT/'data/cache'/('px_'+t.replace('^','_').replace('.','_')+'.parquet')
        x=pd.read_parquet(p)['close'].dropna()
        rets[t]=np.log(x[~x.index.duplicated()]).diff(); paths.append(p)
    output=[]; rows=[]
    for g in groups:
        h=deepcopy(g); h['iv_market']=deepcopy(g['market']); sig=[]
        dt=pd.Timestamp(g['issue'])
        for i,t in enumerate(g['tickers']):
            w=rets[t][rets[t].index<dt].tail(F.VOL_WIN)
            s=F.vol180(rets[t],dt)
            assert np.isfinite(s) and s>0,(g['group'],t,s)
            sig.append(s)
            rows.append(dict(group=g['group'],issue=g['issue'],ticker=t,
                hv=s,iv=g['market']['sigs'][i],return_count=int(w.count()),
                return_start=str(w.index.min().date()),return_end=str(w.index.max().date())))
        h['market']['sigs']=sig; MarketV2(**h['market']).validate()
        assert h['records']==g['records']
        assert {k:v for k,v in h['market'].items() if k!='sigs'}=={k:v for k,v in g['market'].items() if k!='sigs'}
        output.append(h)
    assert sum(len(g['records']) for g in output)==base_cfg['included']==35587
    pd.DataFrame(rows).to_csv(OUT/'volatility_assignment.csv',index=False)
    (OUT/'groups_hv.json.gz').write_bytes(gzip.compress(json.dumps(output,ensure_ascii=False,allow_nan=False).encode(),mtime=0))
    preserved=dict(base_cfg['preserved_sha256'])
    paths += [BASE/'protocol.json',BASE/'run.py',BASE/'results/groups.json.gz',
              BASE/'results/fair_mc_iv.csv',BASE/'results/mc_seed_checkpoints.csv.gz',
              BASE/'results/mc_checkpoints.csv.gz',BASE/'results/price_segments.csv']
    preserved.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    for p,h in preserved.items(): assert sha(ROOT/p)==h,p
    cfg=dict(target='Same actual products: FAIR vs HV MC and IV MC, by FAIR sextile',
        base_iv_analysis=str(BASE.relative_to(ROOT)),products=35587,groups=len(output),
        paths_per_seed=40000,replicates=3,checkpoints=[10000,20000,40000],
        hv_policy='Existing features.vol180: up to 180 log returns strictly before issue, std(ddof=1)*sqrt(252), at least 60 rows; no sigma clipping',
        iv_policy=base_cfg['iv_policy'],
        pairing='Same original contracts, cohort, sorted asset order, correlation, interest curve, dividend yields, seed, path chunk, time block, horizons and cashflows; only sigma differs',
        iv_reuse='Reuse completed IV prices and raw seed checkpoint sums; calculate missing HV arm only',
        bins='Use the same pd.qcut(FAIR/issue_price,6) membership for both arms, same cohort as IV-only plot',
        normalization=base_cfg['normalization'].replace('mc_iv','mc_hv or mc_iv'),
        assumptions_from_iv={k:base_cfg[k] for k in ['settlement_assumption','monthly_assumption','face_assumption','market_policy','limitations']},
        uncertainty='Same random draws reduce comparison noise; saved marginal payoff sums do not contain cross-products, so no paired pathwise SE is claimed. Seed-level changes are retained.',
        preserved_sha256=preserved,groups_sha256=sha(OUT/'groups_hv.json.gz'))
    dump(HERE/'protocol.json',cfg)
    a=pd.DataFrame(rows)
    audit=dict(products=35587,groups=len(groups),additional_exclusions=0,
        hv_range=[float(a.hv.min()),float(a.hv.max())],iv_range=[float(a.iv.min()),float(a.iv.max())],
        min_return_count=int(a.return_count.min()),max_return_count=int(a.return_count.max()),
        contracts_and_nonvolatility_inputs_unchanged=True)
    dump(OUT/'input_audit.json',audit); print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': main()
