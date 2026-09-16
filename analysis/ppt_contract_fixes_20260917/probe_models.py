"""Evaluate B Stage-1 OOS models on held-out one-term perturbations."""
from common import *
from types import SimpleNamespace
from module.train import load_curve_predictor
from module import data as md
import torch,joblib

def main():
 torch.set_num_threads(2);s=pd.read_parquet(BASE/'rebuilt_universe.parquet');d,ok=data('B','cpu','corrected');truth=pd.read_csv(OUT/'probe_mc.csv');families=json.loads((OUT/'probe_design.json').read_text())['families'];byitem={f['item']:f for f in families}
 con=d.CON[truth.i].copy();names=[md.INV.get(c,c) for c in md.CONTRACT];col={k:j for j,k in enumerate(names)}
 for j,r in enumerate(truth.itertuples(index=False)):
  n=int(s.iloc[r.i].nobs)
  if r.axis=='coupon_regular':
   ratio=(s.iloc[r.i].coupon+r.h)/s.iloc[r.i].coupon;con[j,col['coupon']]+=r.h
   for k in range(12):con[j,col[f'pmt_{k}']]*=ratio
  elif r.axis=='ki_barrier':con[j,col['B']]+=r.h
  elif r.axis=='first_strike':con[j,col['strk_0']]+=r.h
  elif r.axis=='last_strike':
   for k in range(n-1,12):con[j,col[f'strk_{k}']]+=r.h
 probe=SimpleNamespace(CURVE=d.CURVE[truth.i],VC=d.VC[truth.i],CON=con,DEV='cpu')
 base={r.item:j for j,r in enumerate(truth.itertuples(index=False)) if r.axis=='base'};bidx=np.array([base[it] for it in truth.item]);rows=[]
 boundaries=[int(len(s)*x) for x in (.7,.8,.9)];fold=np.searchsorted(boundaries,truth.i,side='right')
 for arm in ('legacy','corrected'):
  for model in ('deeponet','xgb'):
   for seed in range(5):
    values=np.empty(len(truth))
    for k in range(4):
     ix=np.flatnonzero(fold==k);path=OUT/'models'/f'{arm}_{model}_B_seed{seed}_fold{k}'
     if model=='deeponet':values[ix]=load_curve_predictor(probe,str(path)+'.pt')(ix)
     else:values[ix]=joblib.load(str(path)+'.pkl')['model'].predict(np.column_stack([probe.CURVE,probe.VC,probe.CON])[ix])
    q=truth.copy();q['payoff']=arm;q['model']=model;q['seed']=seed;q['prediction']=values;q['pred_delta']=(values-values[bidx])*10000;rows.append(q)
 q=pd.concat(rows,ignore_index=True);q.to_csv(OUT/'probe_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0});scores=[]
 q=q[q.axis!='base']
 for (arm,model,monthly,axis,h),g in q.groupby(['payoff','model','monthly','axis','h']):
  error=g.pred_delta-g.delta;resolved=(g.delta.abs()>1.96*g.delta_se)&g.affected.ge(30)
  scores.append(dict(payoff=arm,model=model,monthly=monthly,axis=axis,h=h,contracts=g.item.nunique(),mean_MC_delta=g.delta.mean(),delta_MAE=error.abs().mean(),delta_RMSE=np.sqrt((error**2).mean()),resolved_rows=int(resolved.sum()),direction_agreement=float((np.sign(g.loc[resolved,'pred_delta'])==np.sign(g.loc[resolved,'delta'])).mean()) if resolved.any() else np.nan))
 pd.DataFrame(scores).to_csv(OUT/'probe_metrics.csv',index=False)
 print(pd.DataFrame(scores).groupby(['payoff','model']).delta_MAE.mean())
if __name__=='__main__':main()
