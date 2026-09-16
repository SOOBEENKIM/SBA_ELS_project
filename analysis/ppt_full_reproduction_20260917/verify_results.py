"""Verify full pricing completeness, immutable source, MC provenance, and OOS predictions."""
from common import *
import yaml,torch,joblib
from module.data import walk_forward
from module.train import load_curve_predictor

def main():
 torch.set_num_threads(2);manifest=json.loads((REF.parent/'import_manifest.json').read_text())
 for f in manifest['selected']:assert hashlib.sha256((REF/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
 protocol=json.loads((OUT/'full_mc_protocol.json').read_text());completion=json.loads((OUT/'full_mc_completion.json').read_text())
 for name,h in protocol['inputs'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
 assert hashlib.sha256((OUT/'fresh_mc_variants.parquet').read_bytes()).hexdigest()==completion['output_sha256']
 s,v,ok=tables();assert len(s)==58790 and ok.sum()==58760 and np.isfinite(v.loc[ok,['mc_A','mc_B','se_A','se_B']]).all().all()
 assert completion['priced']==58760 and completion['total_paths_all_variants']==58760*40000*2
 universe=json.loads((OUT/'universe_validation.json').read_text());assert universe['status']=='pass' and max(universe['max_abs_difference'].values())==0
 comparisons=pd.read_csv(OUT/'market_input_comparison.csv');assert (comparisons.different==0).all()
 for raw in universe['raw_inputs']:assert hashlib.sha256((HERE.parents[1]/raw['path']).read_bytes()).hexdigest()==raw['sha256']
 assert 'checks PASS' in (OUT/'engine_validation.log').read_text()
 cfg=yaml.safe_load((REF/'config.yaml').read_text());checks=[];ids=None
 for x in 'AB':
  d,ok=data(x,'cpu');folds=walk_forward(d.n,cfg['data']['walk_forward'],cfg['data']['val_frac'],cfg['data']['val_seed']);folds=[tuple(a[ok[a]] for a in f) for f in folds];ix=np.concatenate([f[2] for f in folds]);assert len(ix)==23486
  for tr,va,te in folds:assert not set(tr)&set(va) and max(np.r_[tr,va])<min(te)
  for model in ['deeponet','xgb']:
   for seed in range(5):
    name=f'{model}_{x}_seed{seed}';p=pd.read_csv(OUT/'predictions'/(name+'.csv.gz'),dtype={'ITEM_CD':str});assert p.ITEM_CD.is_unique
    np.testing.assert_array_equal(p.ITEM_CD.to_numpy(),d.ITEM[ix]);np.testing.assert_allclose(p.mc_true,d.MC[ix],atol=1e-7,rtol=0)
    if ids is None:ids=p.ITEM_CD.tolist()
    assert ids==p.ITEM_CD.tolist()
    fp=json.loads((OUT/(name+'_fingerprint.json')).read_text());assert fp==training_fingerprint(model,x,seed,fp['device'])
    maxerr=0
    for k,(_,_,te) in enumerate(folds):
     idx=te[np.linspace(0,len(te)-1,32).astype(int)];path=str(OUT/'models'/(name+f'_fold{k}'))
     if model=='deeponet':yp=load_curve_predictor(d,path+'.pt')(idx)
     else:
      obj=joblib.load(path+'.pkl');yp=obj['model'].predict(np.column_stack([d.CURVE,d.VC,d.CON])[idx])
     err=float(max(abs(yp-p.set_index('ITEM_CD').loc[d.ITEM[idx],'mc_pred'].to_numpy())));assert err<2e-6,(name,k,err);maxerr=max(maxerr,err)
    checks.append(dict(model=model,variant=x,seed=seed,rows=len(p),max_reload_error=maxerr,**score(p)))
 pd.DataFrame(checks).to_csv(OUT/'model_verification.csv',index=False)
 result=dict(status='pass',raw_DART_rebuilt=True,contracts_identical_to_source=True,all_valid_products_repriced=True,products=58760,paths_per_product_per_variant=40000,variants=2,old_MC_used_as_training_target=False,Stage1_models=20,fold_checkpoints=80,OOS_rows_per_execution=23486,reference_files_unchanged=len(manifest['selected']),max_checkpoint_reload_error=max(c['max_reload_error'] for c in checks),MC_pricing_payoff_matches_notebook5=True,original_latest_scratch_available=False,dividend_rule='Reconstructed from notebook description and input cache; original selection code unavailable. Event dates not individually verifiable against original.',price_increment_experiment_included=False)
 dump(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
