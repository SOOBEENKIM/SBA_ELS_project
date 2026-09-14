"""Audit stored prices, paired increments, source copies and held-out predictions."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=json.loads((HERE/'source_manifest.json').read_text())
    for row in source['files']:assert sha(HERE/row['copy'])==row['sha256']
    for row in source['market_cache']:assert sha(ROOT/'data/cache'/row['name'])==row['sha256']
    d=pd.read_csv(OUT/'population_recomputed.csv');old=pd.read_csv(HERE/'reference/mc_iv_full.csv')
    assert len(d)==58790 and d.item.is_unique and set(d.item)==set(old.item)
    for arm in ['hv_ns','iv_ns','iv_boot']:
        assert np.isfinite(d[[arm,arm+'_se']]).all().all() and d[arm].ge(0).all() and d[arm+'_se'].ge(0).all()
    for kind,names in [('population',['reference_engine.py','run_population.py','population_jobs.json.gz']),
                       ('synthetic',['synthetic.py','paired_engine.py','reference_engine.py','synthetic_families.json.gz','mc_contract_v2.py','mc_contract_v3.py'])]:
        runtime=json.loads((OUT/f'{kind}_complete.json').read_text())['device']
        spec=json.loads((OUT/f'{kind}_run_spec_{runtime}.json').read_text())
        for name in names:
            path=OUT/name if name.endswith('.gz') else ROOT/'module'/name if name.startswith('mc_contract') else HERE/name
            assert sha(path)==spec[name],f'Executed {kind} source changed: {name}'
    labels=pd.read_csv(OUT/'synthetic_labels.csv.gz');by={p:g.set_index(['family','axis','offset']) for p,g in labels.groupby('payoff')}
    for stem in ['price_krw','delta_krw']:
        np.testing.assert_allclose(by['detailed'][stem]-by['reference'][stem],by['detailed_minus_reference'][stem],atol=1e-8,rtol=0)
    for p,g in by.items():assert np.isfinite(g[['price_krw','delta_krw','price_se_krw','delta_se_krw']]).all().all()
    tests=[]
    for split in ['train','validation','test','stress']:
        a=np.load(HERE/'synthetic/reference/results'/f'{split}.npz')
        b=np.load(HERE/'synthetic/detailed/results'/f'{split}.npz')
        for key in ['U','V','C','base','group','axis','offset']:np.testing.assert_array_equal(a[key],b[key])
    for payoff in ['reference','detailed']:
        folder=HERE/'synthetic'/payoff/'results';splits={}
        for split in ['train','validation','test','stress']:
            z=dict(np.load(folder/f'{split}.npz'));meta=pd.read_csv(folder/f'{split}_rows.csv');splits[split]=set(meta.family)
            assert len(z['Y'])==len(meta);assert all(np.isfinite(x).all() for x in z.values())
            np.testing.assert_allclose(z['Y']-z['Y'][z['base']],z['D'],atol=1e-12,rtol=0)
        for a in splits:
            for b in splits:
                if a!=b:assert not(splits[a]&splits[b])
        p=pd.read_csv(folder/'predictions.csv.gz',dtype={'seed':str})
        assert np.isfinite(p[['mc_price_krw','mc_delta_krw','pred_price_krw','pred_delta_krw']]).all().all()
        for _,g in p.groupby(['model','seed','split']):
            base=g[g.axis.eq('base')].set_index('family').pred_price_krw
            np.testing.assert_allclose(g.pred_price_krw-g.family.map(base),g.pred_delta_krw,atol=1e-8,rtol=0)
        selection=json.loads((folder/'model_selection.json').read_text());assert not selection['test_data_opened_by_training']
        assert selection['wrapper_sha256']==sha(HERE/'learn.py')
        for name,h in selection['checkpoint_hashes'].items():assert sha(folder.parent/'models'/name)==h
        tests.append(dict(payoff=payoff,train=len(splits['train']),validation=len(splits['validation']),test=len(splits['test']),stress=len(splits['stress']),checkpoints=len(selection['checkpoint_hashes'])))
    comparison=[]
    for own,ref in [('hv_ns','mc'),('iv_boot','mc_iv')]:
        e=(d[own]-d[ref])*10000
        comparison.append(dict(new=own,archived=ref,mean_difference_krw=e.mean(),mae_between_runs_krw=e.abs().mean(),p95_absolute_difference_krw=e.abs().quantile(.95)))
    for name in ['engine_validation_cuda.json','paired_engine_validation_cuda.json','all_structure_ledger_validation.json']:
        # Published run is CUDA; fresh CPU runs use their own device checks.
        check=OUT/name
        if name!='all_structure_ledger_validation.json':
            kind='population' if name.startswith('engine') else 'synthetic'
            device=json.loads((OUT/f'{kind}_complete.json').read_text())['device']
            check=OUT/name.replace('_cuda.json',f'_{device}.json')
        assert json.loads(check.read_text())['status']=='pass'
    report=dict(status='pass',source_copies_verified=len(source['files']),market_caches_verified=len(source['market_cache']),products=len(d),synthetic_paired_identity=True,same_model_inputs_across_payoffs=True,executed_source_hashes_match=True,payoff_ledger_checks_pass=True,models=tests,archived_vs_recomputed=comparison,
      scope='Numerical reproduction, input/split bookkeeping and paired-label/prediction identities; not all-product legal-terms validation')
    (OUT/'final_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
