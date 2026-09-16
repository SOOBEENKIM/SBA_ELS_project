"""Reconstructed notebook-18 orchestration; invokes original, unchanged anchor functions.
Not the missing scratch/stage1_mc_variants.py. Does not train Stage 2.
Fold boundaries are formed BEFORE filtering stale inputs, consistent with 23,486 OOS rows.
"""
import argparse,time,platform,os
from common import *
import yaml,torch
from module.data import walk_forward
from model.deeponet import _anchor
from model.benchmark import _xgb_anchor

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['deeponet','xgb'],required=True);p.add_argument('--variant',choices=['A','B'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 torch.set_num_threads(4)
 cfg=yaml.safe_load((REF/'config.yaml').read_text());cfg['seed']=a.seed
 d,ok=data(a.variant,a.device)
 folds=walk_forward(d.n,cfg['data']['walk_forward'],cfg['data']['val_frac'],cfg['data']['val_seed'])
 folds=[tuple(ix[ok[ix]] for ix in f) for f in folds]
 assert sum(len(f[2]) for f in folds)==23486
 prefix=f'{a.model}_{a.variant}_seed{a.seed}';pred=OUT/'predictions';models=OUT/'models';pred.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
 dest=pred/(prefix+'.csv.gz')
 if dest.exists():
        q=pd.read_csv(dest)
        meta=OUT/(prefix+'_execution.json')
        if not meta.exists():
            dump(meta,dict(model=a.model,variant=a.variant,seed=a.seed,device=a.device,metrics=score(q),train_rows=[len(f[0]) for f in folds],val_rows=[len(f[1]) for f in folds],test_rows=[len(f[2]) for f in folds],original_anchor_unmodified=True,config=cfg,metadata_recovered_from_completed_predictions=True,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,runner='reconstructed from notebook 18 + module.data.walk_forward',deep_time_weights='Original train_curve unweighted; unchanged.'))
        print('REUSE',prefix,score(q),flush=True);return
 rows=[];times=[];t=time.monotonic()
 for k,(tr,va,te) in enumerate(folds):
  foldpath=pred/(prefix+f'_fold{k}.csv.gz');start=time.monotonic()
  if foldpath.exists():q=pd.read_csv(foldpath,dtype={'ITEM_CD':str})
  else:
   fn=_anchor if a.model=='deeponet' else _xgb_anchor
   f=fn(d,cfg,tr,va,te,save_path=str(models/(prefix+f'_fold{k}')))
   q=pd.DataFrame(dict(ITEM_CD=d.ITEM[te],isu_ord=d.ORD[te],fold=k,mc_true=d.MC[te],mc_pred=f(te)))
   assert np.isfinite(q.mc_pred).all();q.to_csv(foldpath,index=False,compression={'method':'gzip','mtime':0})
  rows.append(q);times.append(time.monotonic()-start)
  print(prefix,'fold',k,score(q),'seconds',round(times[-1],1),flush=True)
 q=pd.concat(rows,ignore_index=True);q.to_csv(dest,index=False,compression={'method':'gzip','mtime':0})
 dump(OUT/(prefix+'_execution.json'),dict(model=a.model,variant=a.variant,seed=a.seed,device=a.device,metrics=score(q),seconds=time.monotonic()-t,fold_seconds=times,train_rows=[len(f[0]) for f in folds],val_rows=[len(f[1]) for f in folds],test_rows=[len(f[2]) for f in folds],python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,original_anchor_unmodified=True,runner='reconstructed from notebook 18 + module.data.walk_forward',deep_time_weights='Original train_curve does not apply time weights despite config flag; unchanged.',config=cfg))
 print('DONE',prefix,score(q),flush=True)
if __name__=='__main__':main()
