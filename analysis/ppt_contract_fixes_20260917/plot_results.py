"""Read saved results only; figures do not rerun MC or training."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C={'legacy':'#e07a5f','survival_only':'#70ad83','corrected':'#1f6f9c'}
LABEL={'legacy':'Original approximation','survival_only':'Survival coupon only','corrected':'Explicit cashflows / dates'}
def save(fig,name):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=180,facecolor='white');fig.savefig(FIG/(name+'.pdf'),facecolor='white');plt.close(fig)
def main():
 s=pd.read_parquet(BASE/'rebuilt_universe.parquet').set_index('item');q=pd.read_parquet(OUT/'paired_prices.parquet').set_index('item');meta=pd.read_csv(OUT/'contract_audit.csv.gz').set_index('item').loc[q.index]
 q['fair_ratio']=s.loc[q.index,'fair'];q['fair_face']=meta.fair_raw/10000;q['year']=s.loc[q.index,'isu_ord'].map(lambda t:pd.Timestamp.fromordinal(int(t)).year)
 summary=[];structures=[];steps=[]
 for basis in ('original_normalization','common_10000_face'):
  d=q.copy() if basis=='original_normalization' else q[meta.common_face_eligible].copy()
  fair=d.fair_ratio if basis=='original_normalization' else d.fair_face
  for x in 'AB':
   for arm in C:
    e=(fair-d[f'mc_{x}_{arm}'])*10000;ape=e.abs()/(fair*10000)*100
    summary.append(dict(basis=basis,variant=x,payoff=arm,n=len(d),mean=e.mean(),median=e.median(),MAE=e.abs().mean(),RMSE=np.sqrt((e*e).mean()),MAPE=ape.mean(),sd=e.std()))
    for (st,monthly),ix in d.groupby(['structure','monthly']).groups.items():
     g=e.loc[ix];structures.append(dict(basis=basis,variant=x,payoff=arm,structure=st,monthly=monthly,n=len(g),mean=g.mean(),MAE=g.abs().mean(),MAPE=ape.loc[ix].mean()))
   # Price segments use exactly the same products and boundaries for all arms.
   part=pd.DataFrame(dict(fair=fair));part['segment']=pd.qcut(fair,6,duplicates='drop')
   for arm in ('legacy','corrected'):
    part['gap_'+arm]=(fair-d[f'mc_{x}_{arm}'])*10000;part['ape_'+arm]=abs(part['gap_'+arm])/(fair*10000)*100
   g=part.groupby('segment',observed=True).agg(n=('fair','size'),gap_legacy=('gap_legacy','mean'),gap_corrected=('gap_corrected','mean'),MAPE_legacy=('ape_legacy','mean'),MAPE_corrected=('ape_corrected','mean'));g.to_csv(OUT/f'segments_{basis}_{x}.csv')
   fig,axes=plt.subplots(1,2,figsize=(13.4,4.8));pos=np.arange(len(g))
   for j,arm in enumerate(('legacy','corrected')):
    axes[0].bar(pos+(j-.5)*.36,g['gap_'+arm],.36,color=C[arm],label=LABEL[arm]);axes[1].bar(pos+(j-.5)*.36,g['MAPE_'+arm],.36,color=C[arm],label=LABEL[arm])
   for ax,title in zip(axes,['Mean fair - MC (KRW)','Mean |fair - MC| / fair (%)']):
    ax.set(title=title);ax.set_xticks(pos,[f'{i.left:.3f}–{i.right:.3f}' for i in g.index],rotation=30,ha='right');ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=8)
   fig.suptitle(f'{x}: contract correction, same {len(d):,} products ({basis})');save(fig,f'segments_{basis}_{x}')
   fig,ax=plt.subplots(figsize=(12.5,4.8));errors=[(fair-d[f'mc_{x}_{arm}'])*10000 for arm in ('legacy','corrected')];bins=np.linspace(min(t.min() for t in errors),max(t.max() for t in errors),85)
   for arm,e in zip(('legacy','corrected'),errors):
    ax.hist(e,bins=bins,alpha=.5,color=C[arm],label=f'{LABEL[arm]}, mean {e.mean():.1f}');ax.axvline(e.mean(),ls='--',color=C[arm])
   ax.axvline(0,color='gray',ls=':');ax.set(title=f'{x}: Fair - MC, same {len(d):,} products',xlabel='Fair - MC (KRW); '+basis,ylabel='Products');ax.legend();save(fig,f'histogram_{basis}_{x}')
 for x in 'AB':
  for (st,monthly),g in q.groupby(['structure','monthly']):
   for name,col in [('survival_coupon','delta_'+x+'_survival'),('schedule_and_monthly','delta_'+x+'_schedule_monthly')]:
    d=g[col]*10000;se=g['se_'+col]*10000 if 'se_'+col in g else None
    steps.append(dict(variant=x,structure=st,monthly=monthly,change=name,n=len(g),mean=d.mean(),median=d.median(),min=d.min(),max=d.max(),mean_paired_SE=se.mean() if se is not None else np.nan))
  assert q['delta_'+x+'_survival'].min()>-1e-12
 pd.DataFrame(summary).to_csv(OUT/'fair_mc_summary.csv',index=False);pd.DataFrame(structures).to_csv(OUT/'fair_mc_by_structure.csv',index=False);pd.DataFrame(steps).to_csv(OUT/'correction_effects.csv',index=False)
 old=pd.read_parquet(BASE/'fresh_mc_variants.parquet').set_index('item');check=[]
 for x in 'AB':
  diff=(q[f'mc_{x}_legacy']-old.loc[q.index,'mc_'+x])*10000
  check.append(dict(variant=x,n=len(q),mean=diff.mean(),mean_abs=diff.abs().mean(),max_abs=diff.abs().max(),exact_or_1e_8=int(diff.abs().lt(1e-8).sum())))
 pd.DataFrame(check).to_csv(OUT/'previous_cache_comparison.csv',index=False)
 preds={};scores=[]
 for arm in ('legacy','corrected'):
  for model in ('deeponet','xgb'):
   for x in 'AB':
    for seed in range(5):
     path=OUT/'predictions'/f'{arm}_{model}_{x}_seed{seed}.csv.gz'
     if path.exists():
      frame=pd.read_csv(path);scores.append(dict(payoff=arm,model=model,variant=x,seed=seed,**score(frame)));preds[(arm,model,x,seed)]=frame
 if not scores:return
 r=pd.DataFrame(scores);r.to_csv(OUT/'stage1_by_seed.csv',index=False);agg=r.groupby(['payoff','model','variant'])[['R2','MAE','RMSE','MAPE']].agg(['mean','std']);agg.to_csv(OUT/'stage1_summary.csv')
 if len(scores)!=40:return
 assert len({tuple(v.ITEM_CD) for v in preds.values()})==1
 for metric in ('R2','MAPE','MAE','RMSE'):
  fig,ax=plt.subplots(figsize=(11,4.8));pos=np.arange(4);labels=[]
  for k,(model,x) in enumerate([(m,x) for m in ('deeponet','xgb') for x in 'AB']):
   labels.append(f'{model} / {x}')
   for j,arm in enumerate(('legacy','corrected')):
    mean,sd=agg.loc[(arm,model,x),(metric,'mean')],agg.loc[(arm,model,x),(metric,'std')];at=k+(j-.5)*.34
    ax.bar(at,mean,.34,yerr=sd,capsize=3,color=C[arm],label=LABEL[arm] if k==0 else None);ax.text(at,mean+sd,f'{mean:.4f}',ha='center',va='bottom',fontsize=8)
  ax.set_xticks(pos,labels);ax.set(title=f'Stage-1 {metric}: same sample / features, five seeds ± SD',ylabel=metric+(' (%)' if metric=='MAPE' else ''));ax.legend(fontsize=8);save(fig,'stage1_'+metric.lower())
 fold=[]
 for (arm,m,x,seed),frame in preds.items():
  for f,group in frame.groupby('fold'):fold.append(dict(payoff=arm,model=m,variant=x,seed=seed,fold=f,**score(group)))
 pd.DataFrame(fold).to_csv(OUT/'stage1_by_fold.csv',index=False)
 if (OUT/'probe_metrics.csv').exists():
  metrics=pd.read_csv(OUT/'probe_metrics.csv');axes_names=['coupon_regular','ki_barrier','first_strike','last_strike']
  for metric,unit in [('delta_MAE','KRW'),('direction_agreement','fraction')]:
   fig,axs=plt.subplots(2,4,figsize=(16,7))
   for row,monthly in enumerate((False,True)):
    for col,axis in enumerate(axes_names):
     ax=axs[row,col]
     for arm in ('legacy','corrected'):
      g=metrics[(metrics.payoff==arm)&(metrics.model=='deeponet')&(metrics.monthly==monthly)&(metrics.axis==axis)].sort_values('h')
      if len(g):ax.plot(g.h*100,g[metric],'-o',ms=3,color=C[arm],label='Trained on '+arm+' MC')
     count=metrics[(metrics.monthly==monthly)&(metrics.axis==axis)].contracts.max()
     ax.set(title=('Monthly: ' if monthly else 'Ordinary: ')+axis+f' (n={int(count)})',xlabel='Change (percentage points)',ylabel=metric+' ('+unit+')');ax.grid(alpha=.2)
   axs[0,0].legend(fontsize=7);save(fig,'increment_'+metric+'_by_offset')
 print(pd.DataFrame(summary).round(4).to_string(index=False));print(agg.round(5).to_string())
if __name__=='__main__':main()
