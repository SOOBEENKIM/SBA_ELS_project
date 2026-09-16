"""Reconstruct notebook-17 curves; explicitly validate against stored B curve inputs."""
from common import *
from reprice_audit import curves
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 s,v,ok=tables();ix=np.flatnonzero(ok);sample=ix[np.linspace(0,len(ix)-1,200).astype(int)];tau=np.array([.25,.5,1,1.5,2,3,4,5,7,10]);rows=[]
 for i in sample:
  ns,bt,bp=curves(int(s.iloc[i].isu_ord));na,nb=ns(tau),bt(tau);err=max(abs(nb-v.iloc[i][[f'bu{j}' for j in range(10)]].to_numpy(float)))
  assert err<1e-10,(i,err)
  for t,a,b in zip(tau,na,nb):rows.append(dict(item=s.iloc[i]['item'],isu_ord=int(s.iloc[i].isu_ord),year=pd.Timestamp.fromordinal(int(s.iloc[i].isu_ord)).year,tenor=t,NS=a,bootstrap=b,difference_bp=(b-a)*10000,cache_max_abs_diff=err,par_reprice_max_bp=bp))
 d=pd.DataFrame(rows);d.to_csv(OUT/'curve_audit_200.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(13.3,4.5));grid=np.linspace(.05,10,200)
 for ax,date in zip(axes[:2],['2021-07-01','2023-07-01']):
  i=ix[np.argmin(abs(s.iloc[ix].isu_ord.to_numpy()-pd.Timestamp(date).toordinal()))];row=s.iloc[i];ns,bt,bp=curves(int(row.isu_ord));ax.plot(grid,ns(grid)*100,label='A: Nelson-Siegel',color='#e07a5f');ax.plot(grid,bt(grid)*100,label='B: bootstrap',color='#1f6f9c');ax.axvspan(0,row.tenor,color='gray',alpha=.12);ax.set(title=str(pd.Timestamp.fromordinal(int(row.isu_ord)).date()),xlabel='Maturity (years)',ylabel='Zero rate (%)');ax.legend(fontsize=8)
 g=d.groupby('tenor').difference_bp;axes[2].plot(tau,g.mean(),'-o');axes[2].fill_between(tau,g.quantile(.1),g.quantile(.9),alpha=.15);axes[2].axhline(0,color='gray',ls=':');axes[2].set(title='B - A: 200 fixed audit products',xlabel='Maturity (years)',ylabel='Mean / 10-90% band (bp)')
 fig.tight_layout();fig.savefig(FIG/'curve_comparison.png',dpi=180,facecolor='white');fig.savefig(FIG/'curve_comparison.pdf');plt.close(fig)
 dump(OUT/'curve_validation.json',dict(n=200,cache_max_abs_difference=float(d.cache_max_abs_diff.max()),par_reprice_max_bp=float(d.par_reprice_max_bp.max()),selection='200 evenly spaced indices of valid full-population rows, not original notebook17 missing sample IDs'))
 print('CURVES PASS',len(sample),d.cache_max_abs_diff.max(),d.par_reprice_max_bp.max())
if __name__=='__main__':main()
