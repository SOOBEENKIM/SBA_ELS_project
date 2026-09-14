"""Price all prepared actual contracts with the existing IV-branch MC engine."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import argparse,gzip,hashlib,json,os,sys,time
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
sys.path[:0]=[str(ROOT),str(ROOT/'analysis/mc_schedule_v4_20260911'),str(ROOT/'analysis/mc_monthly_v3_20260910')]
from schedule_mc import price_grid
from module.mc_contract_v3 import ContractV3,MarketV2,price_grid_v3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def inputs(g):
    return [ContractV3(**r['contract']).validate() for r in g['records']],MarketV2(**g['market']).validate()
def init():torch.set_num_threads(1)

def job(g,rep):
    init();contracts,m=inputs(g)
    seed=int(hashlib.sha256(f"{g['group']}:{rep}:FAIRIV20260914".encode()).hexdigest()[:8],16)
    t=time.time();rows=price_grid(contracts,m,paths=40000,seed=seed,checkpoints=(10000,20000,40000),path_chunk=2500,tblock=256)
    keep=[]
    for x in rows:
        keep.append(dict(item=g['records'][x['case']]['item'],paths=x['paths'],seed=seed,
                         replicate=rep,price_sum=x['price_sum'],price_sumsq=x['price_sumsq']))
    return dict(group=g['group'],replicate=rep,rows=keep,seconds=time.time()-t)

def verify(groups):
    init();seen=set();errors=[]
    for g in groups:
        cs,m=inputs(g)
        for c in cs:
            if c.has_monthly in seen:continue
            # Single-contract grouped simulation must match the preserved native engine.
            a=price_grid([c],m,paths=500,seed=713,checkpoints=(500,),path_chunk=250)[0]
            b=price_grid_v3([('base',0.,c)],m,paths=500,seed=713,checkpoints=(500,),path_chunk=250)[0]
            for k in ['price_sum','price_sumsq']:
                e=abs(a[k]-b[k]);errors.append(e);assert e<1e-8,(k,e)
            seen.add(c.has_monthly)
        if len(seen)==2:break
    assert len(seen)==2
    dump(OUT/'engine_check.json',dict(status='pass',regular_and_monthly_checked=True,max_sum_difference=max(errors),
                                   preserved_payoff_functions=True,new_model_training=False))

def aggregate(groups,jobs):
    rows=[];records=[]
    for g in groups:
        records+=g['records']
        for rep in range(3):
            p=jobs/f"{g['group']}_{rep}.json";need=json.loads(p.read_text())
            assert need['group']==g['group'] and need['replicate']==rep
            rows+=need['rows']
    raw=pd.DataFrame(rows)
    assert not raw.duplicated(['item','replicate','paths']).any()
    assert raw.groupby(['item','paths']).replicate.nunique().eq(3).all()
    allrows=[]
    for (it,n),g in raw.groupby(['item','paths']):
        total_n=int(g.paths.sum());total=float(g.price_sum.sum());sq=float(g.price_sumsq.sum())
        var=max(0.,(sq-total**2/total_n)/(total_n-1))
        allrows.append(dict(item=it,paths_per_seed=int(n),replicates=3,total_paths=total_n,
                            mc_face_krw=total/total_n*10000,mc_se_face_krw=np.sqrt(var/total_n)*10000))
    c=pd.DataFrame(allrows)
    c.to_csv(OUT/'mc_checkpoints.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    raw.to_csv(OUT/'mc_seed_checkpoints.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    meta=pd.DataFrame([{k:v for k,v in r.items() if k!='contract'} for r in records])
    d=meta.merge(c[c.paths_per_seed.eq(40000)],on='item',validate='one_to_one')
    assert len(d)==len(records) and not d.item.duplicated().any()
    d['fair']=d.fair_raw_krw/d.issue_price_krw
    d['mc_iv']=d.mc_face_krw/d.issue_price_krw
    d['fair_krw']=d.fair*10000;d['mc_iv_krw']=d.mc_iv*10000
    d['gap_krw']=(d.fair-d.mc_iv)*10000
    d['ape_pct']=abs(d.fair-d.mc_iv)/abs(d.fair)*100
    assert np.isfinite(d[['fair','mc_iv','gap_krw','ape_pct']]).all().all()
    d.sort_values('item').to_csv(OUT/'fair_mc_iv.csv',index=False)
    return len(d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=int,default=0);ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args();cfg=json.loads((HERE/'protocol.json').read_text())
    for p,h in cfg['preserved_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'groups.json.gz')==cfg['groups_sha256']
    groups=json.loads(gzip.decompress((OUT/'groups.json.gz').read_bytes()));verify(groups)
    jobs=OUT/'mc_jobs';jobs.mkdir(exist_ok=True)
    spec=dict(groups=sha(OUT/'groups.json.gz'),runner=sha(__file__),protocol=sha(HERE/'protocol.json'))
    path=OUT/'label_spec.json'
    if path.exists():assert json.loads(path.read_text())==spec,'Refuse reuse of mismatched pricing inputs'
    else:dump(path,spec)
    if args.pilot:
        # Deterministic coverage of small/large market groups, regular/monthly contracts.
        ordered=sorted(groups,key=lambda g:len(g['records']))
        indices=np.linspace(0,len(ordered)-1,args.pilot,dtype=int)
        chosen=[ordered[i] for i in indices]
    else:
        chosen=sorted(groups,key=lambda g:(not any(r['payment_schedule_source']=='individual_disclosure' for r in g['records']),-len(g['records'])))
    tasks=[(g,rep) for g in chosen for rep in range(3) if not (jobs/f"{g['group']}_{rep}.json").exists()]
    start=time.time();completed=0
    dump(OUT/'launch.json',dict(workers=args.workers,pilot=bool(args.pilot),started_unix=start,
         seed_policy='sha256(group_id:replicate:FAIRIV20260914) first 8 hex digits; replicates 0,1,2'))
    print('MC START',dict(groups=len(chosen),products=sum(len(g['records']) for g in chosen),jobs_remaining=len(tasks),workers=args.workers),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool:
        futures={pool.submit(job,g,rep):(g['group'],rep) for g,rep in tasks}
        for future in as_completed(futures):
            result=future.result();gid,rep=futures[future]
            dump(jobs/f'{gid}_{rep}.json',result);completed+=1
            if completed%100==0 or completed==len(tasks):
                state=dict(completed_new_jobs=completed,total_new_jobs=len(tasks),seconds=time.time()-start,
                           products=sum(len(g['records']) for g in chosen),pilot=bool(args.pilot))
                dump(OUT/'progress.json',state);print('MC PROGRESS',json.dumps(state),flush=True)
    if args.pilot:
        print('PILOT DONE',time.time()-start,'seconds',flush=True);return
    n=aggregate(groups,jobs)
    for p,h in cfg['preserved_sha256'].items():assert sha(ROOT/p)==h,p
    dump(OUT/'completion.json',dict(status='complete',products=n,groups=len(groups),paths_per_seed=40000,
         replicates=3,workers=args.workers,new_seconds=time.time()-start,all_input_hashes_unchanged=True))
    print('COMPLETE',n,'actual products',flush=True)

if __name__=='__main__':main()
