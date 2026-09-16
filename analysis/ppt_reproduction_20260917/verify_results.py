"""Check provenance, OOS membership/targets, and checkpoint -> stored predictions."""
from common import *
import torch,joblib
from module.data import walk_forward
from module.train import load_curve_predictor
import yaml

def main():
 torch.set_num_threads(2);manifest=json.loads((HERE/'import_manifest.json').read_text())
 for row in manifest['selected']:
  assert hashlib.sha256((REF/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
 assert hashlib.sha256((REF/'data/els3_dataset.parquet').read_bytes()).hexdigest()==manifest['source_dataset']['sha256']
 cfg=yaml.safe_load((REF/'config.yaml').read_text());checks=[];eval_ids=None
 for x in 'AB':
  d,ok=data(x,'cpu');folds=walk_forward(d.n,cfg['data']['walk_forward'],cfg['data']['val_frac'],cfg['data']['val_seed'])
  folds=[tuple(ix[ok[ix]] for ix in f) for f in folds];expected=np.concatenate([f[2] for f in folds])
  for tr,va,te in folds:
   assert not set(tr)&set(va) and not set(tr)&set(te) and not set(va)&set(te)
   assert max(np.r_[tr,va])<min(te) # order-based source split; same issue dates can straddle split.
  for m in ('deeponet','xgb'):
   for seed in range(5):
    name=f'{m}_{x}_seed{seed}';q=pd.read_csv(OUT/'predictions'/(name+'.csv.gz'),dtype={'ITEM_CD':str});assert len(q)==23486 and q.ITEM_CD.is_unique
    np.testing.assert_array_equal(q.ITEM_CD.to_numpy(),d.ITEM[expected]);np.testing.assert_allclose(q.mc_true,d.MC[expected],atol=1e-7,rtol=0)
    if eval_ids is None:eval_ids=q.ITEM_CD.to_list()
    assert q.ITEM_CD.to_list()==eval_ids
    maxdiff=0
    for k,(_,_,te) in enumerate(folds):
     ix=te[np.linspace(0,len(te)-1,32).astype(int)];prefix=OUT/'models'/(name+f'_fold{k}')
     if m=='deeponet':yp=load_curve_predictor(d,str(prefix)+'.pt')(ix)
     else:
      ck=joblib.load(str(prefix)+'.pkl');yp=ck['model'].predict(np.concatenate([d.CURVE,d.VC,d.CON],axis=1)[ix])
     ref=q.set_index('ITEM_CD').loc[d.ITEM[ix],'mc_pred'].to_numpy();diff=float(max(abs(yp-ref)));assert diff<2e-6,(name,k,diff);maxdiff=max(maxdiff,diff)
    checks.append(dict(model=m,variant=x,seed=seed,rows=len(q),checkpoint_prediction_max_abs_difference=maxdiff,**{k:v for k,v in score(q).items() if k!='n'}))
 pd.DataFrame(checks).to_csv(OUT/'verification_models.csv',index=False)
 report=dict(status='pass',source_files_verified=len(manifest['selected']),models=20,fold_checkpoints=80,OOS_rows_per_run=23486,unique_items_shared_across_runs=True,labels_match_cache=True,checkpoint_forward_verified=True,max_prediction_difference=max(r['checkpoint_prediction_max_abs_difference'] for r in checks),MC_paths_in_reprice_audit=40000,exact_raw_to_MC_reproduction=False,reason='Missing latest scratch runner and dividend construction source; fresh MC differs from cached values.',latest_synthetic_increment_retrained=False,note='Previous synthetic experiments preserved on parent branch; not rerun with latest B labels in this PPT reproduction.')
 dump(OUT/'verification.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
