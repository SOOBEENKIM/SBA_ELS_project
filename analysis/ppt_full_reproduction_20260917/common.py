"""Adapters for notebook 18. Original model/MC sources in reference/ are unchanged."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
REF=HERE.parent/'ppt_reproduction_20260917/reference';OUT=HERE/'results';FIG=HERE/'figures'
sys.path.insert(0,str(REF))
for p in (OUT,FIG):p.mkdir(exist_ok=True)
def tables():
 s=pd.read_parquet(OUT/'rebuilt_universe.parquet')
 v=pd.read_parquet(OUT/'fresh_mc_variants.parquet')
 assert s.item.is_unique and v.item.equals(s.item)
 ok=v.fail.isna() & ~v.stale_px
 return s,v,ok.to_numpy()
def data(variant,device):
 from module import data as md
 s,v,ok=tables();s=md.fill_lizard_neutral(s)
 sig=v[[f's{variant}_{j}' for j in (1,2,3)]].to_numpy()
 corr=np.sort(v[['rho_12','rho_13','rho_23']].clip(-.999,.999).to_numpy(),axis=1)
 eff=sig.mean(axis=1)*np.sqrt(1+np.maximum(0,1-corr.mean(axis=1)))
 vc=np.column_stack([np.sort(sig,axis=1),corr,eff])
 if variant=='B':vc=np.column_stack([vc,np.take_along_axis(v[['q_1','q_2','q_3']].to_numpy(),np.argsort(sig,axis=1),axis=1)])
 curve=v[[f'{variant.lower()}u{j}' for j in range(10)]].to_numpy()
 con=s[[md.INV.get(c,c) for c in md.CONTRACT]].to_numpy()
 d=SimpleNamespace(n=len(s),VC=vc.astype('float32'),CURVE=curve.astype('float32'),CON=con.astype('float32'),MC=v['mc_'+variant].to_numpy(dtype='float32'),ORD=s.isu_ord.to_numpy(float),ITEM=s.item.to_numpy(str),DEV=device)
 for name in ('VC','CURVE','CON','MC'):assert np.isfinite(getattr(d,name)[ok]).all(),name
 return d,ok
def score(frame):
 from sklearn.metrics import r2_score
 e=(frame.mc_pred-frame.mc_true)*10000
 return dict(n=len(frame),R2=float(r2_score(frame.mc_true,frame.mc_pred)),MAE=float(e.abs().mean()),RMSE=float(np.sqrt((e**2).mean())),MAPE=float((e.abs()/(frame.mc_true*10000)).mean()*100))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def training_fingerprint(model,variant,seed,device):
    files=[OUT/'rebuilt_universe.parquet',OUT/'fresh_mc_variants.parquet',REF/'config.yaml',REF/'module/train.py',REF/'module/networks.py',REF/'module/data.py',REF/'model/benchmark.py',REF/'model/deeponet.py',HERE/'common.py',HERE/'train_stage1.py']
    return dict(model=model,variant=variant,seed=seed,device=device,sha256={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
