"""Verify source preservation, pricing coverage, OOS identity and saved weights."""
from common import *
import yaml,torch,joblib,gzip
from module.data import walk_forward
from module.train import load_curve_predictor
ROOT=HERE.parents[1]
def main():
 torch.set_num_threads(2);manifest=json.loads((REF.parent/'import_manifest.json').read_text())
 for f in manifest['selected']:assert hashlib.sha256((REF/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
 protocol=json.loads((OUT/'protocol.json').read_text());audit=json.loads((OUT/'input_audit.json').read_text());done=json.loads((OUT/'completion.json').read_text())
 for file,h in {**protocol['hashes'],**audit['source_hashes']}.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
 assert hashlib.sha256((OUT/'paired_prices.parquet').read_bytes()).hexdigest()==done['output_hash']
 assert json.loads((OUT/'validation.json').read_text())['status']=='pass'
 for file,h in json.loads((OUT/'probe_protocol.json').read_text()).items():assert hashlib.sha256((HERE.parent/file).read_bytes()).hexdigest()==h,file
 prices=pd.read_parquet(OUT/'paired_prices.parquet');assert prices.item.is_unique and prices.paths.eq(40000).all() and len(prices)==audit['market_and_contract_supported']==44103
 assert np.isfinite(prices.filter(regex='^(mc|se|delta)_').to_numpy()).all()
 assert len(json.loads(gzip.decompress((OUT/'contracts.json.gz').read_bytes())))==audit['contract_supported']
 for x in 'AB':
  np.testing.assert_allclose(prices[f'mc_{x}_survival_only']-prices[f'mc_{x}_legacy'],prices[f'delta_{x}_survival'],atol=1e-12,rtol=0)
  np.testing.assert_allclose(prices[f'mc_{x}_corrected']-prices[f'mc_{x}_survival_only'],prices[f'delta_{x}_schedule_monthly'],atol=1e-12,rtol=0)
  assert prices[f'delta_{x}_survival'].min()>=-1e-12
 cfg=yaml.safe_load((REF/'config.yaml').read_text());checks=[];ids=None
 for arm in ('legacy','corrected'):
  for x in 'AB':
   d,ok=data(x,'cpu',arm);folds=walk_forward(d.n,cfg['data']['walk_forward'],cfg['data']['val_frac'],cfg['data']['val_seed']);folds=[tuple(a[ok[a]] for a in f) for f in folds];ix=np.concatenate([f[2] for f in folds])
   for tr,va,te in folds:assert not set(tr)&set(va) and max(np.r_[tr,va])<min(te)
   for model in ('deeponet','xgb'):
    for seed in range(5):
     name=f'{arm}_{model}_{x}_seed{seed}';p=pd.read_csv(OUT/'predictions'/(name+'.csv.gz'),dtype={'ITEM_CD':str});assert p.ITEM_CD.is_unique
     np.testing.assert_array_equal(p.ITEM_CD.to_numpy(),d.ITEM[ix]);np.testing.assert_allclose(p.mc_true,d.MC[ix],atol=1e-7,rtol=0)
     if ids is None:ids=p.ITEM_CD.tolist()
     assert ids==p.ITEM_CD.tolist()
     fp=json.loads((OUT/(name+'_fingerprint.json')).read_text());assert fp==training_fingerprint(model,x,seed,fp['device'],arm)
     maxerr=0.
     for k,(_,_,te) in enumerate(folds):
      idx=te[np.linspace(0,len(te)-1,32).astype(int)];path=str(OUT/'models'/(name+f'_fold{k}'))
      if model=='deeponet':yp=load_curve_predictor(d,path+'.pt')(idx)
      else:yp=joblib.load(path+'.pkl')['model'].predict(np.column_stack([d.CURVE,d.VC,d.CON])[idx])
      err=float(max(abs(yp-p.set_index('ITEM_CD').loc[d.ITEM[idx],'mc_pred'].to_numpy())));assert err<2e-6,(name,k,err);maxerr=max(maxerr,err)
     checks.append(dict(payoff=arm,model=model,variant=x,seed=seed,rows=len(p),max_reload_error=maxerr,**score(p)))
 pd.DataFrame(checks).to_csv(OUT/'model_verification.csv',index=False)
 probes=pd.read_csv(OUT/'probe_mc.csv');assert probes.groupby('item').apply(lambda g:g.axis.eq('base').sum(),include_groups=False).eq(1).all()
 assert set(probes.item)<=set(ids);pred=pd.read_csv(OUT/'probe_predictions.csv.gz');assert len(pred)==len(probes)*20 and np.isfinite(pred[['prediction','pred_delta']]).all().all()
 result=dict(status='pass',source_files_unchanged=len(manifest['selected']),matched_priced_products=len(prices),paths_per_product_per_variant=40000,market_variants=2,payoff_arms=3,stage1_configurations=40,fold_checkpoints=160,OOS_rows_per_configuration=len(ids),probe_families=probes.item.nunique(),probe_cases=len(probes),probe_paths_per_seed=40000,probe_MC_seeds=3,probe_training_leakage=False,features='Original 51 contract fields retained for a controlled comparison; payment/monthly schedules not yet encoded in model inputs',max_checkpoint_reload_error=max(r['max_reload_error'] for r in checks))
 dump(OUT/'verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
