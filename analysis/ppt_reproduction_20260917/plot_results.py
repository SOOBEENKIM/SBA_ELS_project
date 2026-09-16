"""Reaggregate cached MC labels and independently rerun predictions; no MC call here."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL={'A':'#e07a5f','B':'#1f6f9c'}
LAB={'A':'A: NS / HV120 / no dividend','B':'B: bootstrap / EWMA120 / discrete dividend'}
def save(fig,name):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,facecolor='white');fig.savefig(FIG/(name+'.pdf'),facecolor='white');plt.close(fig)
def main():
 s,v,ok=tables();d=s.loc[ok,['item','isu_ord','fair','opt_type','ki_yn','imonth']].copy()
 for x in 'AB':
  d['mc_'+x]=v.loc[ok,'mc_'+x];d['gap_'+x]=(d.fair-d['mc_'+x])*10000;d['ape_'+x]=abs(d['gap_'+x])/(d.fair*10000)*100
 d['year']=d.isu_ord.map(lambda o:pd.Timestamp.fromordinal(int(o)).year)
 d.to_csv(OUT/'fair_mc_products.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 rows=[]
 for x in 'AB':
  e=d['gap_'+x];rows.append(dict(variant=x,n=len(e),mean=e.mean(),median=e.median(),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),sd=e.std(),MAPE=d['ape_'+x].mean(),mc_mean=d['mc_'+x].mean()*10000))
 levels=pd.DataFrame(rows);levels.to_csv(OUT/'fair_mc_summary.csv',index=False)
 fig,ax=plt.subplots(figsize=(13.3,4.8));bins=np.linspace(min(d.gap_A.quantile(.001),d.gap_B.quantile(.001)),max(d.gap_A.quantile(.999),d.gap_B.quantile(.999)),61)
 for x in 'AB':
  ax.hist(d['gap_'+x],bins=bins,alpha=.55,color=COL[x],edgecolor='white',lw=.25,label=LAB[x]+f', mean {d["gap_"+x].mean():,.1f} KRW');ax.axvline(d['gap_'+x].mean(),color=COL[x],ls='--')
 ax.axvline(0,color='gray',ls=':');ax.set(xlabel='Fair - MC (KRW, source ratios x 10,000)',ylabel='Number of products',title=f'Fair minus MC: A vs B (n={len(d):,})');ax.legend();save(fig,'fair_mc_histogram')
 d['segment']=pd.qcut(d.fair,6);g=d.groupby('segment',observed=True).agg(n=('item','size'),gap_A=('gap_A','mean'),gap_B=('gap_B','mean'),MAPE_A=('ape_A','mean'),MAPE_B=('ape_B','mean'));g.to_csv(OUT/'fair_mc_by_segment.csv')
 fig,ax=plt.subplots(1,2,figsize=(13.3,5));pos=np.arange(len(g));labels=[f'{i.left:.3f}~{i.right:.3f}' for i in g.index]
 for j,x in enumerate('AB'):
  for k,field in enumerate(['gap','MAPE']):ax[k].bar(pos+(j-.5)*.36,g[field+'_'+x],.36,color=COL[x],label=LAB[x])
 for a,title,unit in zip(ax,['Mean fair - MC by fair-value segment','Mean |fair - MC| / fair by segment'],['KRW','%']):
  a.set_xticks(pos,labels,rotation=30,ha='right');a.set(title=title,ylabel=unit);a.axhline(0,color='gray',lw=.8);a.legend(fontsize=8)
 save(fig,'fair_mc_by_segment')
 year=d.groupby('year').agg(n=('item','size'),mean_A=('gap_A','mean'),mean_B=('gap_B','mean'),MAE_A=('gap_A',lambda e:e.abs().mean()),MAE_B=('gap_B',lambda e:e.abs().mean()),MAPE_A=('ape_A','mean'),MAPE_B=('ape_B','mean'));year.to_csv(OUT/'fair_mc_by_year.csv')
 fig,ax=plt.subplots(1,2,figsize=(13.3,4.5))
 for x in 'AB':
  ax[0].plot(year.index,year['MAE_'+x],'-o',label=LAB[x],color=COL[x]);ax[1].plot(year.index,year['MAPE_'+x],'-o',color=COL[x])
 ax[0].set(title='Fair - MC absolute gap by issue year',ylabel='MAE (KRW)');ax[1].set(title='Relative absolute gap by issue year',ylabel='MAPE (%)');ax[0].legend(fontsize=8);save(fig,'fair_mc_by_year')
 dump(OUT/'input_validation.json',dict(population=len(s),included=int(ok.sum()),excluded=int((~ok).sum()),stale=int(v.stale_px.sum()),failed=int(v.fail.notna().sum()),ids_order_match=True,aggregation='Existing MC cache: not newly simulated',fair_units='Source fair=FAIR_VALUE/ISU_PRC_DETAIL, MC is unit-face. Preserve exact notebook convention; multiplying by 10000 is not common cash normalization for discounted issue prices.'))
 print(levels.round(4).to_string(index=False),flush=True)
 preds={};scores=[]
 for m in ('deeponet','xgb'):
  for x in 'AB':
   for seed in range(5):
    p=OUT/'predictions'/f'{m}_{x}_seed{seed}.csv.gz'
    if p.exists():
     q=pd.read_csv(p);preds[(m,x,seed)]=q;scores.append(dict(model=m,variant=x,seed=seed,**score(q)))
 if not scores:return
 r=pd.DataFrame(scores);r.to_csv(OUT/'stage1_by_seed.csv',index=False)
 agg=r.groupby(['model','variant'])[['R2','MAE','RMSE','MAPE']].agg(['mean','std']);agg.to_csv(OUT/'stage1_summary.csv')
 # A five-seed figure is emitted only when all 20 executions exist.
 if len(preds)!=20:print(f'Stage1 incomplete: {len(preds)}/20; no five-seed plot.',flush=True);return
 series=[('deeponet','A','#1f6f9c'),('deeponet','B','#5aa9dd'),('xgb','A','#9b5de5'),('xgb','B','#c9a7f5')]
 for metric in ['R2','MAPE','MAE','RMSE']:
  fig,ax=plt.subplots(figsize=(10,4.5))
  for j,(m,x,col) in enumerate(series):
   val,sd=agg.loc[(m,x),(metric,'mean')],agg.loc[(m,x),(metric,'std')];ax.bar(j,val,yerr=sd,capsize=4,color=col);ax.text(j,val+sd+.015*max(agg[(metric,'mean')]),f'{val:.4f}',ha='center')
  ax.set_xticks(range(4),[f'{m}\n{x}' for m,x,c in series]);ax.set(title=f'Stage-1 {metric}: rerun, 5 seeds mean ± SD',ylabel=metric+(' (%)' if metric=='MAPE' else ''));save(fig,'stage1_'+metric.lower())
 fig,axes=plt.subplots(1,2,figsize=(13.3,4.5));foldrows=[]
 for ax,m in zip(axes,('deeponet','xgb')):
  for x in 'AB':
   q=pd.concat([preds[(m,x,seed)] for seed in range(5)]);err=(q.mc_pred-q.mc_true)*10000;ax.hist(err,bins=np.linspace(-400,400,81),alpha=.5,color=COL[x],label=f'{x}: MAE={abs(err).mean():.1f} KRW')
   for seed in range(5):
    for fold,p in preds[(m,x,seed)].groupby('fold'):foldrows.append(dict(model=m,variant=x,seed=seed,fold=fold,**score(p)))
  ax.set(title=m+' prediction error',xlabel='MC prediction - MC (KRW)');ax.legend()
 save(fig,'stage1_error_histogram');pd.DataFrame(foldrows).to_csv(OUT/'stage1_by_fold.csv',index=False)
 print(agg.round(5).to_string(),flush=True)
if __name__=='__main__':main()
