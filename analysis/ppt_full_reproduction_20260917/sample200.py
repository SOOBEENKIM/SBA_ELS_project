"""Declared reconstruction of notebook16/17's evenly spaced dividend-covered sample.
58,405 eligible rows -> 200 evenly spaced row positions, same frozen row ordering.
PPT's six fair-value bin bounds and both example dates agree; original IDs are missing.
"""
from common import *
def indices(s):
 eligible=~s[['udl1','udl2','udl3']].isin(['^GDAXI','207940.KS','373220.KS']).any(axis=1)
 ix=np.flatnonzero(eligible);assert len(ix)==58405
 return ix[np.linspace(0,len(ix)-1,200).astype(int)]
def main():
 from plot_results import save,COL,LAB
 import matplotlib.pyplot as plt
 s,v,ok=tables();ix=indices(s);assert ok[ix].all();d=s.iloc[ix][['item','isu_ord','fair','udl1','udl2','udl3']].copy();d.to_csv(OUT/'declared_sample200.csv',index=False)
 for x in 'AB':
  d['mc_'+x]=v.iloc[ix]['mc_'+x];d['gap_'+x]=(d.fair-d['mc_'+x])*10000;d['ape_'+x]=d['gap_'+x].abs()/d.fair/10000*100
 d['year']=d.isu_ord.map(lambda x:pd.Timestamp.fromordinal(int(x)).year);d.to_csv(OUT/'sample200_prices.csv',index=False)
 d['segment']=pd.qcut(d.fair,6);g=d.groupby('segment',observed=True).agg(n=('item','size'),gap_A=('gap_A','mean'),gap_B=('gap_B','mean'),MAPE_A=('ape_A','mean'),MAPE_B=('ape_B','mean'));g.to_csv(OUT/'sample200_by_segment.csv')
 fig,ax=plt.subplots(1,2,figsize=(13.3,5));pos=np.arange(6)
 for j,x in enumerate('AB'):
  for k,field in enumerate(['gap','MAPE']):ax[k].bar(pos+(j-.5)*.36,g[field+'_'+x],.36,color=COL[x],label=LAB[x])
 for a,ttl,unit in zip(ax,['Mean fair - MC: reconstructed n=200','Mean |fair - MC| / fair: reconstructed n=200'],['KRW','MAPE (%)']):
  a.set_xticks(pos,[f'{i.left:.3f}~{i.right:.3f}' for i in g.index],rotation=30,ha='right');a.set(title=ttl,ylabel=unit);a.axhline(0,color='gray',lw=.8);a.legend(fontsize=7)
 save(fig,'sample200_by_segment')
 g=d.groupby('year').agg(n=('item','size'),mean_A=('gap_A','mean'),mean_B=('gap_B','mean'),MAE_A=('gap_A',lambda e:e.abs().mean()),MAE_B=('gap_B',lambda e:e.abs().mean()));g.to_csv(OUT/'sample200_by_year.csv')
 fig,ax=plt.subplots(1,2,figsize=(13.3,4.8))
 for x in 'AB':
  ax[0].plot(g.index,g['mean_'+x],'-o',color=COL[x],label=LAB[x]);ax[1].plot(g.index,g['MAE_'+x],'-o',color=COL[x])
 for a in ax:a.axhline(0,color='gray',lw=.8);a.set(xlabel='Issue year',ylabel='KRW')
 ax[0].set_title('Mean fair - MC: reconstructed n=200');ax[1].set_title('MAE by year: reconstructed n=200');ax[0].legend(fontsize=7);save(fig,'sample200_by_year')
 rows=[]
 for x in 'AB':
  e=d['gap_'+x];rows.append(dict(variant=x,n=200,mean=e.mean(),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),MAPE=d['ape_'+x].mean()))
 summary=pd.DataFrame(rows);summary.to_csv(OUT/'sample200_summary.csv',index=False)
 nb=json.loads((REF/'17_MC_bootstrap_curve.ipynb').read_text());txt='\n'.join(''.join(o.get('text',[])) for o in nb['cells'][5].get('outputs',[]));refs=[]
 for line in txt.splitlines():
  if line.startswith('before:') or line.startswith('after:'):
   numbers=list(map(float,line.split()[-6:]));refs.append(dict(variant='A' if line.startswith('before:') else 'B',mean=numbers[0],MAE=numbers[2],RMSE=numbers[3],MAPE=numbers[4]))
 reference=pd.DataFrame(refs).set_index('variant');comp=[]
 for row in rows:
  for metric in ['mean','MAE','RMSE','MAPE']:comp.append(dict(variant=row['variant'],metric=metric,source_notebook=reference.loc[row['variant'],metric],fresh_sample200=row[metric],difference=row[metric]-reference.loc[row['variant'],metric]))
 pd.DataFrame(comp).to_csv(OUT/'sample200_vs_notebook.csv',index=False)
 # Compare year counts visible in notebook17; this does not establish item-level identity.
 txt='\n'.join(''.join(o.get('text',[])) for o in nb['cells'][13].get('outputs',[]));year_counts={}
 for line in txt.splitlines():
  parts=line.split()
  if len(parts)>2 and parts[0].isdigit() and len(parts[0])==4:year_counts[int(parts[0])]=int(parts[1])
 assert d.groupby('year').size().to_dict()==year_counts

 actual_bins=[f'{i.left:.3f}~{i.right:.3f}' for i in d.segment.cat.categories]
 assert actual_bins==['0.799~0.885','0.885~0.914','0.914~0.934','0.934~0.952','0.952~0.980','0.980~1.040']
 nearest=[str(pd.Timestamp.fromordinal(int(d.iloc[np.argmin(abs(d.isu_ord.to_numpy()-pd.Timestamp(t).toordinal()))].isu_ord)).date()) for t in ['2021-07-01','2023-07-01']]
 assert nearest==['2021-06-24','2023-06-27']
 meta=dict(source_eligible=58405,sample=200,method='integer floor of linspace(0,58404,200), source row ordering',original_item_ids_available=False,PPT_bin_labels_match=True,PPT_curve_example_dates_match=True,notebook_year_counts_match=True,identity_not_independently_proven=True)
 dump(OUT/'sample200_provenance.json',meta);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
