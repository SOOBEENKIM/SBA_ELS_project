"""Fresh 40k-path spot check against cached labels, with explicit reconstruction limits.
Payoff/RNG functions are extracted unchanged from notebook 5. IRS bootstrap follows
notebook 17. Missing dividend script means a last-360-day window is an ASSUMPTION,
not an asserted original rule. Record q/event/MC disagreement without calibrating.
"""
from common import *
import argparse,ast,time
from functools import lru_cache
from scipy.optimize import brentq
import torch
from module.features import krw_beta,zero_curve,chol_psd
CACHE=REF/'data/cache'
PROXY={'^STOXX50E':'FEZ','^GSPC':'SPY','^HSCE':'FXI','^KS200':'069500.KS','^N225':'1321.T','^HSI':'2800.HK','^NDX':'QQQ'}
SW=[1,2,3,4,5,7,10,12,15,20]
K=pd.read_parquet(CACHE/'krw_curve_fred.parquet')
IRS=pd.read_parquet(CACHE/'irs_bondweb.parquet')
nb=json.loads((REF/'5_MC_implied_vol.ipynb').read_text());cell=next(''.join(c['source']) for c in nb['cells'] if 'def mc_daily_t(' in ''.join(c.get('source',[])))
functions=[x for x in ast.parse(cell).body if isinstance(x,ast.FunctionDef) and x.name in ('_setup','mc_daily_t')]
engine={'np':np,'torch':torch,'chol_psd':chol_psd,'NPATH':40000,'DEV':'cuda'}
exec(compile(ast.Module(body=functions,type_ignores=[]),'notebook5_unchanged','exec'),engine)
original_setup=engine['_setup']
@lru_cache(None)
def curves(ordinal):
 dt=pd.Timestamp.fromordinal(int(ordinal));beta=krw_beta(K.asof(dt)[['call','m3','y10']])
 ns=lambda t:zero_curve(beta,t)
 ir=IRS[IRS.index<dt].iloc[-1];prior_month=dt.to_period('M').start_time-pd.Timedelta(days=1)
 cd=float(K.asof(prior_month).m3)/100
 tn=[0.,91/365];ln=[0.,-np.log1p(cd*91/365)];errs=[]
 for T in SW:
  rate=float(ir[f'{T}Y'])/100;tt=np.arange(1,4*T+1)*.25
  def equation(z):
   df=np.exp(np.interp(tt,tn+[float(T)],ln+[z]));return rate*.25*df.sum()+df[-1]-1
  z=brentq(equation,-10,2,xtol=1e-14);tn.append(float(T));ln.append(z)
  df=np.exp(np.interp(tt,tn,ln));errs.append((1-df[-1])/(.25*df.sum())-rate)
 def bt(t):
  t=np.maximum(np.atleast_1d(t).astype(float),1e-10);return -np.interp(t,tn,ln)/t
 return ns,bt,max(abs(np.array(errs)))*10000
@lru_cache(None)
def div_history(ticker):
 path=CACHE/('div_'+ticker.replace('^','_').replace('.','_')+'.parquet')
 if not path.exists():return None
 h=pd.read_parquet(path);delta=h.Dividends/h.Close.shift();return delta[(h.Dividends>0)&delta.notna()]
def dividends(tickers,ordinal,ten):
 dt=pd.Timestamp.fromordinal(int(ordinal));N=int(round(ten*365));drops=np.zeros((N,3),np.float32);qs=[];count=0
 for j,t in enumerate(tickers):
  if t=='^GDAXI':qs.append(0.);continue
  e=div_history(PROXY.get(t,t))
  if e is None:raise ValueError('No dividend history for '+t)
  e=e[e.index<dt]
  if not len(e):qs.append(0.);continue
  w=e[e.index>e.index[-1]-pd.Timedelta(days=360)];qs.append(float(-np.log1p(-w).sum()))
  for d,delta in w.items():
   for year in range(dt.year-d.year,dt.year-d.year+int(np.ceil(ten))+2):
    day=d+pd.DateOffset(years=year);step=(day-dt).days
    if 0<step<=N:drops[step-1,j]+=np.float32(np.log1p(-delta));count+=1
 return drops,np.array(qs),count

def run_price(row,vrow,variant,device):
 ns,bt,err=curves(int(row.isu_ord));zc=ns if variant=='A' else bt
 sig=[vrow[f's{variant}_{j}'] for j in (1,2,3)];rhos=[vrow[f'rho_{k}'] for k in ('12','13','23')]
 corr=np.array([[1,rhos[0],rhos[1]],[rhos[0],1,rhos[2]],[rhos[1],rhos[2],1]])
 n=int(row.nobs);get=lambda key:row[[f'{key}_{j}' for j in range(n)]].to_numpy(float)
 drops,q,nd=dividends([row[f'udl{j}'] for j in (1,2,3)],int(row.isu_ord),float(row.tenor))
 def setup(*a,**kw):
  result=original_setup(*a,**kw)
  if variant=='B':result['drift']=result['drift']+drops
  return result
 engine['_setup']=setup
 price=engine['mc_daily_t'](sig,corr,zc,float(row.B),get('strk'),float(row.tenor),n=40000,seed=int(row.mc_seed),c=float(row.coupon),pmts=get('pmt'),lz_barr=get('lz_barr'),lz_pmt=get('lz_pmt'),dev=device)
 return price,dict(q_maxdiff=float(max(abs(q-vrow[['q_1','q_2','q_3']].to_numpy(float)))) ,n_div=nd,n_div_cached=float(vrow.n_div),curve_maxdiff=float(max(abs(bt([.25,.5,1,1.5,2,3,4,5,7,10])-vrow[[f'bu{i}' for i in range(10)]].to_numpy(float)))),par_reprice_max_bp=err)
def main():
 p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=12);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(2)
 s,v,ok=tables();ix=np.flatnonzero(ok);chosen=ix[np.linspace(0,len(ix)-1,a.count).astype(int)];rows=[]
 for i in chosen:
  for x in 'AB':
   start=time.monotonic();val,detail=run_price(s.iloc[i],v.iloc[i],x,a.device);r=dict(item=s.iloc[i]['item'],variant=x,seed=int(s.iloc[i].mc_seed),paths=40000,cached=float(v.iloc[i]['mc_'+x]),recomputed=val,difference_KRW=(val-v.iloc[i]['mc_'+x])*10000,**detail,seconds=time.monotonic()-start);rows.append(r)
   pd.DataFrame(rows).to_csv(OUT/f'mc_reprice_audit_{a.device}.csv',index=False);print(r,flush=True)
 dump(OUT/f'mc_reprice_protocol_{a.device}.json',dict(samples=a.count,paths=40000,seed='source mc_seed',device=a.device,source_payoff='notebook5 _setup/mc_daily_t extracted unchanged; B adds log dividend drop to drift',bootstrap='notebook17 equations, cached IRS strictly preissue and previous complete month CD',dividend_window='ASSUMED last 360 calendar days anchored to last pre-issue ex-date; source scratch unavailable; compare cached q and event count',limitations=['Not exact 200-product subset from notebook16/17: IDs unavailable.','No claim of full raw-to-B replication unless q, events, curve and price checks agree.','Original terminal survival principal-only and monthly approximation intentionally retained.']))
if __name__=='__main__':main()
