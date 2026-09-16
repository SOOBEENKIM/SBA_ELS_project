"""Explicit pricing-input reconstruction from notebooks 16-18; frozen source data only."""
from pathlib import Path
from functools import lru_cache
import sys
import numpy as np,pandas as pd
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
REF=HERE.parent/'ppt_reproduction_20260917/reference';CACHE=REF/'data/cache'
sys.path.insert(0,str(REF))
from module.features import krw_beta,zero_curve
PROXY={'^STOXX50E':'FEZ','^GSPC':'SPY','^HSCE':'FXI','^KS200':'069500.KS','^N225':'1321.T','^HSI':'2800.HK','^NDX':'QQQ'}
SW=[1,2,3,4,5,7,10,12,15,20]
K=pd.read_parquet(CACHE/'krw_curve_fred.parquet')
# Two missing 20Y quotes: carry each maturity's last available historical quote.
# Tenors <=10Y used by model inputs are unaffected by this operation.
IRS=pd.read_parquet(CACHE/'irs_bondweb.parquet').ffill()
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
  # Empirical recovery of the missing runner's dividend history selection.
  # 330-day lookback matches cached q/event counts; original source statement 'one year' is underspecified.
  w=e[e.index>e.index[-1]-pd.Timedelta(days=330)];qs.append(float(-np.log1p(-w).sum()))
  for d,delta in w.items():
   for year in range(dt.year-d.year,dt.year-d.year+int(np.ceil(ten))+2):
    day=d+pd.DateOffset(years=year);step=(day-dt).days
    if 0<step<=N:drops[step-1,j]+=np.float32(np.log1p(-delta));count+=1
 return drops,np.array(qs),count

