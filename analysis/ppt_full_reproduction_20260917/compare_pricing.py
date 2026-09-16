"""Compare independently recomputed prices/inputs with frozen source results and PPT."""
from common import *
from market_rules import curves,K,IRS
from sample200 import indices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 s,v,ok=tables();old=pd.read_parquet(REF/'data/cache/mcvar/mc_variants.parquet');assert v.item.equals(old.item)
 joint=ok&old.fail.isna().to_numpy()&~old.stale_px.to_numpy();details=s.loc[joint,['item','isu_ord','opt_type','ki_yn']].copy();rows=[]
 for x in 'AB':
  gap=(v.loc[joint,'mc_'+x]-old.loc[joint,'mc_'+x])*10000;se=v.loc[joint,'se_'+x]*10000
  details[f'new_mc_{x}']=v.loc[joint,'mc_'+x];details[f'old_mc_{x}']=old.loc[joint,'mc_'+x];details[f'difference_{x}_KRW']=gap;details[f'MC_SE_{x}_KRW']=se
  rows.append(dict(variant=x,n=len(gap),mean_difference=gap.mean(),MAE_difference=gap.abs().mean(),RMSE_difference=np.sqrt((gap**2).mean()),max_abs_difference=gap.abs().max(),mean_MC_SE=se.mean(),p95_abs_difference=gap.abs().quantile(.95),p99_abs_difference=gap.abs().quantile(.99)))
 details.to_csv(OUT/'fresh_vs_source_prices.csv.gz',index=False,compression={'method':'gzip','mtime':0});pd.DataFrame(rows).to_csv(OUT/'fresh_vs_source_price_summary.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(13.3,4.5))
 for ax,x in zip(axes,'AB'):
  gap=details[f'difference_{x}_KRW'];limit=max(30,gap.abs().quantile(.995));ax.hist(gap,bins=np.linspace(-limit,limit,71),color='#1f6f9c',alpha=.7);ax.axvline(0,color='gray',ls=':');ax.set(title=f'{x}: fresh MC - source MC, mean {gap.mean():+.2f} KRW',xlabel='Difference (KRW)',ylabel='Products')
 fig.tight_layout();fig.savefig(FIG/'fresh_vs_source_mc.png',dpi=180,facecolor='white');fig.savefig(FIG/'fresh_vs_source_mc.pdf');plt.close(fig)
 ix=np.flatnonzero(ok);chosen=indices(s);tau=np.array([.25,.5,1,1.5,2,3,4,5,7,10]);grid=np.linspace(.05,10,200);cr=[]
 for i in chosen:
  ns,bt,bp=curves(int(s.iloc[i].isu_ord))
  for t,a,b in zip(tau,ns(tau),bt(tau)):cr.append(dict(item=s.iloc[i]['item'],isu_ord=int(s.iloc[i].isu_ord),year=pd.Timestamp.fromordinal(int(s.iloc[i].isu_ord)).year,tenor=t,NS=a,bootstrap=b,difference_bp=(b-a)*10000,par_reprice_max_bp=bp))
 c=pd.DataFrame(cr);c.to_csv(OUT/'curve_comparison_200.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(13.3,4.6))
 for ax,date in zip(axes[:2],['2021-07-01','2023-07-01']):
  i=chosen[np.argmin(abs(s.iloc[chosen].isu_ord.to_numpy()-pd.Timestamp(date).toordinal()))];row=s.iloc[i];ns,bt,_=curves(int(row.isu_ord));ax.plot(grid,ns(grid)*100,label='A: NS',color='#e07a5f');ax.plot(grid,bt(grid)*100,label='B: bootstrap',color='#1f6f9c');ax.axvspan(0,row.tenor,color='gray',alpha=.12);ax.set(title=str(pd.Timestamp.fromordinal(int(row.isu_ord)).date()),xlabel='Maturity (years)',ylabel='Zero rate (%)');ax.legend()
 g=c.groupby('tenor').difference_bp;axes[2].plot(tau,g.mean(),'-o');axes[2].fill_between(tau,g.quantile(.1),g.quantile(.9),alpha=.15);axes[2].axhline(0,color='gray',ls=':');axes[2].set(title='Mean B - A, 10-90% band',xlabel='Maturity (years)',ylabel='Difference (bp)');fig.tight_layout();fig.savefig(FIG/'curve_comparison.png',dpi=180,facecolor='white');fig.savefig(FIG/'curve_comparison.pdf');plt.close(fig)
 # Full observed issue-date history; no attempt to guess the missing original 200 IDs.
 q=v.loc[ok].copy();q['date']=s.loc[ok,'isu_ord'].map(lambda x:pd.Timestamp.fromordinal(int(x)));q=q.groupby('date')[[f'{x}u{j}' for x in 'ab' for j in range(10)]].mean()
 fig,axes=plt.subplots(1,2,figsize=(13.3,4.5))
 for ax,j,name in zip(axes,[5,9],['3Y','10Y']):
  ax.plot(q.index,q[f'au{j}']*100,label='A: NS',color='#e07a5f');ax.plot(q.index,q[f'bu{j}']*100,label='B: bootstrap',color='#1f6f9c');ax.set(title=name+' zero rate by issue date',ylabel='Zero rate (%)');ax.legend()
 fig.tight_layout();fig.savefig(FIG/'curve_history.png',dpi=180,facecolor='white');fig.savefig(FIG/'curve_history.pdf');plt.close(fig)
 # Same four series and 200-product sampling definition as PPT slide10.
 history=[]
 for i in chosen:
  dt=pd.Timestamp.fromordinal(int(s.iloc[i].isu_ord));ir=IRS[IRS.index<dt].iloc[-1];cd=float(K.asof(dt.to_period('M').start_time-pd.Timedelta(days=1)).m3);ns,bt,_=curves(int(s.iloc[i].isu_ord))
  history.append(dict(item=s.iloc[i]['item'],date=dt,NS_3Y=float(ns([3])[0])*100,Bootstrap_3Y=float(bt([3])[0])*100,IRS_3Y=float(ir['3Y']),CD91=cd))
 h=pd.DataFrame(history);h.to_csv(OUT/'sample200_curve_history.csv',index=False);fig,ax=plt.subplots(figsize=(13.3,4.5))
 for col,color,style in [('NS_3Y','#e07a5f','-'),('Bootstrap_3Y','#1f6f9c','-'),('IRS_3Y','#2a9d8f','--'),('CD91','#777777','--')]:ax.plot(h.date,h[col],style,label=col,color=color)
 ax.set(xlabel='Issue date',ylabel='Rate (%)',title='3-year rates at issue: reconstructed n=200');ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'sample200_curve_history.png',dpi=180,facecolor='white');fig.savefig(FIG/'sample200_curve_history.pdf');plt.close(fig)
 print(pd.DataFrame(rows).round(4).to_string(index=False),flush=True)
if __name__=='__main__':main()
