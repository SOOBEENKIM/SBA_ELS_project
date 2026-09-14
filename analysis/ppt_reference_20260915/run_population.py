"""Reprice the frozen presentation cohort; all results are cached per product."""
from pathlib import Path
import argparse,gzip,json,hashlib,time,sys,inspect
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';sys.path.insert(0,str(HERE))
import reference_engine as E

# Preserve every path/payoff operation of the source. Add second moments and
# checkpoints after the existing per-path payoff has been computed.
src=inspect.getsource(E.mc_daily_t).replace('def mc_daily_t(', 'def mc_stats(')
src=src.replace('tot = torch.zeros((), device=d, dtype=torch.float64); done = 0',
                'tot = torch.zeros((), device=d, dtype=torch.float64); total2=0.; done = 0; records=[]')
src=src.replace('tot += torch.where(any_ev, early, sv * DF[N - 1]).sum(dtype=torch.float64); done += m',
'''values = torch.where(any_ev, early, sv * DF[N - 1]).double()
        begin=0
        while begin<m:
            target=next(k for k in (10000,20000,40000) if k>done)
            stop=min(m,begin+target-done); part=values[begin:stop]
            tot+=part.sum(); total2+=float((part*part).sum())
            done+=stop-begin;begin=stop
            if done==target:records.append(dict(paths=done,sum=float(tot),sumsq=total2))''')
src=src.replace('return float(tot.item() / n)', 'return records')
assert 'values = ' in src and 'return records' in src
namespace=dict(E.__dict__);exec(src,namespace);mc_stats=namespace['mc_stats']

def one(j,device='cpu'):
    torch.set_num_threads(1);m=j['market'];r=[]
    for arm,sigma,rates,kind in [('hv_ns',m['hv'],m['hv_rates'],'ns'),('iv_ns',m['iv'],m['iv_rates'],'ns'),('iv_boot',m['iv'],m['iv_rates'],'boot')]:
        dz=E.ns_z(rates) if kind=='ns' else E.boot_z(rates)
        vals=mc_stats(sigs=sigma,corr=m['corr'],disc_z=dz,**j['contract'],n=40000,seed=j['seed'],dev=device)
        for z in vals:r.append(dict(item=j['item'],arm=arm,seed=j['seed'],**z))
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=40);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    with gzip.open(OUT/'population_jobs.json.gz','rt') as f:jobs=json.load(f)
    if args.verify_only:
        torch.set_num_threads(1);checks=[];t=time.time()
        for j in [jobs[0],jobs[len(jobs)//2],next(j for j in jobs if j['monthly'])]:
            result=one(j,args.device);m=j['market']
            expected=E.mc_daily_t(sigs=m['iv'],corr=m['corr'],disc_z=E.boot_z(m['iv_rates']),**j['contract'],n=40000,seed=j['seed'],dev=args.device)
            actual=next(x['sum']/40000 for x in result if x['arm']=='iv_boot' and x['paths']==40000)
            assert abs(expected-actual)<1e-14,(expected,actual)
            checks.append(dict(item=j['item'],source_price=expected,added_statistics_price=actual,difference=actual-expected))
        report=dict(status='pass',device=args.device,checks=checks,seconds=time.time()-t)
        (OUT/f'engine_validation_{args.device}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True);return
    assert json.loads((OUT/f'engine_validation_{args.device}.json').read_text())['status']=='pass'
    cache=OUT/f'population_jobs_{args.device}';cache.mkdir(exist_ok=True)
    paths=[HERE/'reference_engine.py',Path(__file__),OUT/'population_jobs.json.gz']
    spec={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    spec['device']=args.device;sp=OUT/f'population_run_spec_{args.device}.json'
    if sp.exists():assert json.loads(sp.read_text())==spec
    else:sp.write_text(json.dumps(spec,indent=2))
    todo=[j for j in jobs if not (cache/(j['item']+'.json')).exists()]
    start=time.time();print('POPULATION start',len(todo),'new /',len(jobs),'total',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,j,args.device):j for j in todo}
        for i,f in enumerate(as_completed(futures),1):
            rows=f.result();dest=cache/(futures[f]['item']+'.json');tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(rows));tmp.replace(dest)
            if i%200==0 or i==len(todo):print('POPULATION',i,'/',len(todo),'seconds',round(time.time()-start,1),'ETA min',round((time.time()-start)/i*(len(todo)-i)/60,1),flush=True)
    rows=[]
    for j in jobs:rows.extend(json.loads((cache/(j['item']+'.json')).read_text()))
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'population_checkpoints.csv.gz',index=False)
    x=raw[raw.paths==40000].copy();x['price']=x['sum']/x.paths;x['se']=np.sqrt(np.maximum(0,(x.sumsq-x['sum']**2/x.paths)/(x.paths-1)/x.paths))
    result=pd.read_csv(OUT/'reference_population_prices.csv').merge(x.pivot(index='item',columns='arm',values='price'),on='item',validate='one_to_one')
    result=result.merge(x.pivot(index='item',columns='arm',values='se').add_suffix('_se'),on='item',validate='one_to_one')
    assert len(result)==58790 and result.item.is_unique and np.isfinite(result[['hv_ns','iv_ns','iv_boot']]).all().all()
    result.to_csv(OUT/'population_recomputed.csv',index=False)
    out=dict(status='complete',products=len(result),device=args.device,paths_per_seed=40000,replicates=1,arms=['hv_ns','iv_ns','iv_boot'],seconds=time.time()-start)
    (OUT/'population_complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
