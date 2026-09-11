from concurrent.futures import ProcessPoolExecutor,as_completed
import time,json,hashlib
import pandas as pd
import numpy as np
import torch
from design import *
from schedule_mc import price_grid

def preservation_files():
    paths=[PROJECT/'module/mc_contract_v2.py',PROJECT/'module/mc_contract_v3.py',V3/'common.py',V3/'models.py',V3/'protocol.json',V3/'results/model_selection.json',V3/'results/families.json']
    paths+=sorted((V3/'models').glob('*.pt'))
    paths+=[V3/'results'/f for f in ['train.npz','validation.npz','test.npz','stress.npz','mc_labels.csv','predictions.csv']]
    return {str(p.relative_to(PROJECT)):sha(p) for p in paths}

def job(r,rep):
    torch.set_num_threads(1)
    cs=[unpack(dict(contract=z['contract'],market=r['market']))[0] for z in r['variants']]
    _,m=unpack(r);seed=int(hashlib.sha256(f"{r['family']}:{rep}:MCscheduleV4".encode()).hexdigest()[:8],16)
    mc=CFG['mc'];rows=price_grid(cs,m,paths=mc['paths_per_seed'],seed=seed,checkpoints=mc['checkpoints'],path_chunk=mc['path_chunk'],tblock=mc['tblock'])
    for z in rows:
        case=r['variants'][z['case']]
        z.update(family=r['family'],replicate=rep,axis=case['axis'],offset=case['offset'],monthly=r['monthly'],structure=r['structure'],origin=r['origin'])
    return rows

def main():
    started=time.time();assert json.loads((OUT/'validation_before_mc.json').read_text())['status']=='pass'
    fs=json.loads((OUT/'families.json').read_text());selection=json.loads((V3/'results/model_selection.json').read_text())
    assert selection['selected_arm']==CFG['models']['primary']
    for f,h in selection['checkpoint_hashes'].items():assert sha(V3/'models'/f)==h
    spec={f:sha(HERE/f) for f in ['protocol.json','design.py','schedule_mc.py','run_mc.py','verify.py','results/families.json']}
    spec['preserved_inputs']=preservation_files();path=OUT/'label_spec.json'
    if path.exists():assert json.loads(path.read_text())==spec,'Specification changed; do not reuse cached labels'
    else:path.write_text(json.dumps(spec,indent=2))
    cache=OUT/'mc_jobs';cache.mkdir(exist_ok=True);raw=[];todo=[]
    for r in fs:
        for rep in range(CFG['mc']['replicates']):
            p=cache/f"{r['family']}_{rep}.json"
            if p.exists():raw.extend(json.loads(p.read_text()))
            else:todo.append((r,rep,p))
    print('MC schedule experiment: families',len(fs),'jobs',len(todo),flush=True)
    with ProcessPoolExecutor(max_workers=CFG['mc']['workers']) as pool:
        futures={pool.submit(job,r,rep):p for r,rep,p in todo}
        for i,f in enumerate(as_completed(futures),1):
            result=f.result();futures[f].write_text(json.dumps(result));raw.extend(result)
            if i%48==0 or i==len(todo):print('MC',i,'/',len(todo),'seconds',round(time.time()-started,1),flush=True)
    raw=pd.DataFrame(raw).sort_values(['family','case','replicate','paths']);raw.to_csv(OUT/'mc_seed_checkpoints.csv',index=False)
    pooled=[]
    keys=['family','case','axis','offset','monthly','structure','origin','paths']
    for key,g in raw.groupby(keys):
        r=dict(zip(keys,key));r['paths_per_seed']=r.pop('paths');n=int(g.paths.sum());r['total_paths']=n;r['replicates']=len(g)
        for stem in ['price','delta']:
            total=g[stem+'_sum'].sum();sq=g[stem+'_sumsq'].sum();mean=total/n
            var=max(0.,(sq-total*total/n)/(n-1));r[stem+'_krw']=mean*10000;r[stem+'_se_krw']=np.sqrt(var/n)*10000
        r['affected']=int(g.affected.sum());r['monthly_pv_krw']=g.monthly_pv_sum.sum()/n*10000;r['nonmonthly_pv_krw']=g.nonmonthly_pv_sum.sum()/n*10000;pooled.append(r)
    pooled=pd.DataFrame(pooled).sort_values(['family','case','paths_per_seed']);pooled.to_csv(OUT/'mc_checkpoints.csv',index=False)
    final=pooled[pooled.paths_per_seed.eq(40000)];final.to_csv(OUT/'mc_labels.csv',index=False)
    assert len(final)==3003 and final.replicates.eq(3).all() and final.total_paths.eq(120000).all()
    assert np.allclose(final.price_krw,final.monthly_pv_krw+final.nonmonthly_pv_krw,rtol=0,atol=1e-9)
    for _,g in final.groupby('family'):
        b=float(g.loc[g.axis.eq('base'),'price_krw'].iloc[0]);assert np.allclose(g.price_krw-b,g.delta_krw,rtol=0,atol=1e-9)
    assert preservation_files()==spec['preserved_inputs']
    out=dict(status='complete',families=len(fs),scenarios=len(final),jobs=len(fs)*3,paths_per_seed=40000,replicates=3,
        seconds=time.time()-started,existing_artifacts_unchanged=True,no_training=True)
    (OUT/'labels_complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
