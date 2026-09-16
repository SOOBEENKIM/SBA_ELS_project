"""Reconstruct pricing inputs from frozen prices/dividends/rates, never from cached MC inputs.
Cached input fields are read only AFTER construction for an independent comparison.
"""
from pathlib import Path
import sys,json,hashlib,time
from functools import lru_cache
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;REF=HERE.parent/'ppt_reproduction_20260917/reference';OUT=HERE/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'ppt_reproduction_20260917'))
from market_rules import curves,dividends,CACHE
MS=np.array([.25,.5,1,1.5,2,3,4,5,7,10.])
RET={};LAST={}
for f in CACHE.glob('px_*.parquet'):
 h=pd.read_parquet(f)['close'].dropna();h=h[~h.index.duplicated()];RET[f.name]=np.log(h).diff();LAST[f.name]=h.index
KRX=LAST['px__KS200.parquet']
def ret(t):return RET['px_'+t.replace('^','_').replace('.','_')+'.parquet']
@lru_cache(None)
def sig(t,ordinal):
 dt=pd.Timestamp.fromordinal(ordinal);w=ret(t);w=w[w.index<dt].tail(120)
 if len(w)<60:return np.nan,np.nan
 return float(w.std()*np.sqrt(252)),float(w.ewm(alpha=.01,adjust=True).std().iloc[-1]*np.sqrt(252))
@lru_cache(None)
def correlation(ts,ordinal):
 dt=pd.Timestamp.fromordinal(ordinal);days=KRX[KRX<dt][-120:]
 if len(days)<60:return None
 # KRX defines the CALENDAR interval, returns in each pair need not be KRX trading dates.
 rs=[ret(t).loc[(ret(t).index>=days[0])&(ret(t).index<dt)] for t in ts]
 C=pd.concat(rs,axis=1).corr(min_periods=60).to_numpy();C=np.clip(C,-.999,.999);np.fill_diagonal(C,1.)
 return C
@lru_cache(None)
def market(ts,ordinal):
 dt=pd.Timestamp.fromordinal(ordinal);ss=np.array([sig(t,ordinal) for t in ts]);c=correlation(ts,ordinal)
 lag=max((dt-ret(t)[ret(t).index<dt].index[-1]).days for t in ts)
 if c is None or not np.isfinite(c).all() or not np.isfinite(ss).all():return dict(fail='inputs',px_lag_days=lag,stale_px=lag>14)
 ns,bt,bp=curves(ordinal)
 row=dict(fail=None,px_lag_days=lag,stale_px=lag>14,curve_par_max_bp=bp)
 for j in range(3):row[f'sA_{j+1}']=ss[j,0];row[f'sB_{j+1}']=ss[j,1]
 for j,(a,b) in enumerate([(0,1),(0,2),(1,2)]):row[['rho_12','rho_13','rho_23'][j]]=c[a,b]
 row.update({f'au{j}':x for j,x in enumerate(ns(MS))});row.update({f'bu{j}':x for j,x in enumerate(bt(MS))})
 return row
@lru_cache(None)
def divs(ts,ordinal,ten):
 drop,q,count=dividends(ts,ordinal,ten);steps,assets=np.nonzero(drop)
 return q,count,[(int(i),int(j),float(drop[i,j])) for i,j in zip(steps,assets)]
def main():
 start=time.monotonic();s=pd.read_parquet(OUT/'rebuilt_universe.parquet');rows=[];events=[]
 for i,r in enumerate(s.itertuples(index=False)):
  ts=(r.udl1,r.udl2,r.udl3);m=market(ts,int(r.isu_ord)).copy();m.update(i=i,item=r.item)
  if m['fail'] is None:
   q,n,ev=divs(ts,int(r.isu_ord),float(r.tenor));m.update(n_div=n,**{f'q_{j+1}':float(q[j]) for j in range(3)})
   ns,bt,_=curves(int(r.isu_ord));m['r_ns']=float(ns([r.tenor])[0]);m['r_bt']=float(bt([r.tenor])[0])
   events.append(ev)
  else:events.append([])
  rows.append(m)
  if (i+1)%5000==0:print('INPUT',i+1,'seconds',round(time.monotonic()-start),flush=True)
 v=pd.DataFrame(rows);v.to_parquet(OUT/'fresh_market_inputs.parquet',index=False)
 # Array row IDs align to explicit ITEM_CD in fresh_market_inputs.
 packed=dict(items=s.item.tolist(),events=events)
 import gzip
 raw=json.dumps(packed,separators=(',',':')).encode()
 with (OUT/'dividend_events.json.gz').open('wb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 old=pd.read_parquet(REF/'data/cache/mcvar/mc_variants.parquet');assert old.item.equals(v.item)
 valid=v.fail.isna()&~v.stale_px;ov=old.fail.isna()&~old.stale_px
 keys=[f's{x}_{j}' for x in 'AB' for j in (1,2,3)]+['rho_12','rho_13','rho_23']+[f'q_{j}' for j in (1,2,3)]+['n_div']+[f'bu{j}' for j in range(10)]+['r_ns','r_bt']
 diff=[];bad=[]
 for key in keys:
  good=valid&ov;gap=(v.loc[good,key]-old.loc[good,key]).abs();tol=1e-7 if key!='n_div' else 0
  diff.append(dict(field=key,rows=len(gap),mean_abs_diff=float(gap.mean()),max_abs_diff=float(gap.max()),different=int((gap>tol).sum())))
  for i in gap.index[gap>tol]:bad.append(dict(item=v.at[i,'item'],field=key,fresh=float(v.at[i,key]),cached=float(old.at[i,key])))
 pd.DataFrame(diff).to_csv(OUT/'market_input_comparison.csv',index=False)
 pd.DataFrame(bad,columns=['item','field','fresh','cached']).to_csv(OUT/'market_input_differences.csv',index=False)
 mismatched=v.loc[valid!=ov,['item','fail','stale_px','px_lag_days']];mismatched.to_csv(OUT/'eligibility_differences.csv',index=False)
 report=dict(total=len(s),valid=int(valid.sum()),excluded=int((~valid).sum()),stale=int(v.stale_px.sum()),failed=int(v.fail.notna().sum()),cached_inclusion_agrees=bool((valid==ov).all()),seconds=time.monotonic()-start,original_dividend_script_available=False,dividend_rule='last preissue ex-date minus 330 calendar days; repeat annual calendar anniversaries strictly after issue through N=round(tenor*365); no future dividends',cached_values_used_for_pricing_inputs=False,input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'rebuilt_universe.parquet',OUT/'fresh_market_inputs.parquet',OUT/'dividend_events.json.gz']})
 (OUT/'input_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);print(pd.DataFrame(diff).to_string(index=False),flush=True)
if __name__=='__main__':main()
