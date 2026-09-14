"""Calculate missing HV prices with the preserved IV runner and matched draws."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,os,platform,sys,time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'results'
BASE=ROOT/'analysis/fair_mc_iv_population_20260914'
spec=importlib.util.spec_from_file_location('preserved_iv_population_runner',BASE/'run.py')
IV=importlib.util.module_from_spec(spec);spec.loader.exec_module(IV)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(p)
def job(g,rep): return IV.job(g,rep)

def verify(groups):
    IV.init(); raw=pd.read_csv(BASE/'results/mc_seed_checkpoints.csv.gz')
    assert not raw.duplicated(['item','replicate','paths']).any()
    ids={r['item']:g['group'] for g in groups for r in g['records']}
    assert set(raw.item)==set(ids)
    for (item,rep),z in raw.groupby(['item','replicate']):
        expected=int(hashlib.sha256(f'{ids[item]}:{rep}:FAIRIV20260914'.encode()).hexdigest()[:8],16)
        assert z.seed.eq(expected).all()
    # Replay one full IV group with unchanged original runner to check seed/runtime alignment.
    g=min(groups,key=lambda g:(max(r['contract']['obs_days'][-1] for r in g['records']),len(g['records'])))
    original=dict(g,market=g['iv_market']); replay=pd.DataFrame(IV.job(original,0)['rows'])
    saved=raw[raw.item.isin(replay.item)&raw.replicate.eq(0)]
    c=replay.merge(saved,on=['item','replicate','paths','seed'],suffixes=('_new','_saved'),validate='one_to_one')
    assert len(c)==len(replay)
    errors=[]
    for k in ['price_sum','price_sumsq']:
        errors.append(float(abs(c[k+'_new']-c[k+'_saved']).max()))
        np.testing.assert_allclose(c[k+'_new'],c[k+'_saved'],rtol=0,atol=1e-8)
    seen=set(); native=[]
    for g in groups:
        cs,m=IV.inputs(g)
        for c in cs:
            if c.has_monthly in seen: continue
            a=IV.price_grid([c],m,paths=500,seed=713,checkpoints=(500,),path_chunk=250)[0]
            b=IV.price_grid_v3([('base',0.,c)],m,paths=500,seed=713,checkpoints=(500,),path_chunk=250)[0]
            for k in ['price_sum','price_sumsq']:
                native.append(abs(a[k]-b[k])); assert abs(a[k]-b[k])<1e-8
            seen.add(c.has_monthly)
        if len(seen)==2: break
    assert len(seen)==2
    result=dict(status='pass',same_seed_for_every_product_and_replicate=True,
        full_path_iv_replay_max_sum_error=max(errors),hv_regular_monthly_native_max_sum_error=max(native))
    dump(OUT/'pre_mc_verification.json',result);print('VERIFY',json.dumps(result),flush=True)

def aggregate(groups,jobs):
    rows=[]
    for g in groups:
        for rep in range(3):
            r=json.loads((jobs/f"{g['group']}_{rep}.json").read_text())
            assert r['group']==g['group'] and r['replicate']==rep
            rows+=r['rows']
    raw=pd.DataFrame(rows)
    assert not raw.duplicated(['item','replicate','paths']).any()
    assert len(raw)==35587*3*3
    ivraw=pd.read_csv(BASE/'results/mc_seed_checkpoints.csv.gz')
    paired=raw.merge(ivraw,on=['item','replicate','paths','seed'],suffixes=('_hv','_iv'),validate='one_to_one')
    assert len(paired)==len(raw)==len(ivraw)
    paired['iv_minus_hv_face_krw']=(paired.price_sum_iv-paired.price_sum_hv)/paired.paths*10000
    paired.to_csv(OUT/'seed_checkpoints_hv_iv.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    z=raw.groupby(['item','paths']).agg(total_paths=('paths','sum'),replicates=('replicate','nunique'),total=('price_sum','sum'),sq=('price_sumsq','sum')).reset_index()
    assert z.replicates.eq(3).all()
    var=np.maximum(0.,(z.sq-z.total**2/z.total_paths)/(z.total_paths-1))
    z['mc_hv_face_krw']=z.total/z.total_paths*10000
    z['mc_hv_se_face_krw']=np.sqrt(var/z.total_paths)*10000
    z=z.drop(columns=['total','sq']).rename(columns={'paths':'paths_per_seed'})
    z.to_csv(OUT/'mc_hv_checkpoints.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    iv=pd.read_csv(BASE/'results/fair_mc_iv.csv')
    d=iv.merge(z[z.paths_per_seed.eq(40000)][['item','mc_hv_face_krw','mc_hv_se_face_krw']],on='item',validate='one_to_one')
    assert len(d)==35587 and set(d.item)==set(iv.item)
    d['mc_hv']=d.mc_hv_face_krw/d.issue_price_krw; d['mc_hv_krw']=d.mc_hv*10000
    d['gap_hv_krw']=(d.fair-d.mc_hv)*10000;d['ape_hv_pct']=abs(d.fair-d.mc_hv)/abs(d.fair)*100
    d['iv_minus_hv_krw']=(d.mc_iv-d.mc_hv)*10000
    d['ape_change_iv_minus_hv_pp']=d.ape_pct-d.ape_hv_pct
    assert np.isfinite(d[['mc_hv','gap_hv_krw','ape_hv_pct','iv_minus_hv_krw']]).all().all()
    d.sort_values('item').to_csv(OUT/'fair_mc_hv_iv.csv',index=False)
    return len(d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=48);ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args();cfg=json.loads((HERE/'protocol.json').read_text())
    for p,h in cfg['preserved_sha256'].items(): assert sha(ROOT/p)==h,p
    assert sha(OUT/'groups_hv.json.gz')==cfg['groups_sha256']
    groups=json.loads(gzip.decompress((OUT/'groups_hv.json.gz').read_bytes())); verify(groups)
    if args.verify_only: return
    jobs=OUT/'mc_jobs';jobs.mkdir(exist_ok=True)
    fingerprint=dict(groups=sha(OUT/'groups_hv.json.gz'),runner=sha(__file__),protocol=sha(HERE/'protocol.json'))
    p=OUT/'label_spec.json'
    if p.exists(): assert json.loads(p.read_text())==fingerprint,'Refuse mismatched cache'
    else: dump(p,fingerprint)
    groups.sort(key=lambda g:-len(g['records']))
    tasks=[(g,rep) for g in groups for rep in range(3) if not (jobs/f"{g['group']}_{rep}.json").exists()]
    start=time.time();completed=0
    dump(OUT/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,workers=args.workers,device='cpu',threads_per_worker=1))
    dump(OUT/'launch.json',dict(started_unix=start,jobs_remaining=len(tasks),jobs_total=len(groups)*3,workers=args.workers))
    print('HV MC START',len(tasks),'jobs;',len(groups),'market groups;',args.workers,'workers',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=IV.init) as pool:
        futures={pool.submit(job,g,rep):(g['group'],rep) for g,rep in tasks}
        for future in as_completed(futures):
            r=future.result();gid,rep=futures[future]
            dump(jobs/f'{gid}_{rep}.json',r);completed+=1
            if completed%200==0 or completed==len(tasks):
                state=dict(completed_new_jobs=completed,total_new_jobs=len(tasks),seconds=time.time()-start)
                dump(OUT/'progress.json',state);print('HV MC PROGRESS',json.dumps(state),flush=True)
    n=aggregate(groups,jobs)
    for p,h in cfg['preserved_sha256'].items(): assert sha(ROOT/p)==h,p
    done=dict(status='complete',products=n,groups=len(groups),paths_per_seed=40000,replicates=3,workers=args.workers,seconds=time.time()-start,iv_results_and_engine_unchanged=True)
    dump(OUT/'completion.json',done);print('COMPLETE',json.dumps(done),flush=True)

if __name__=='__main__': main()
