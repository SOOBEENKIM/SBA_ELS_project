from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import pandas as pd
import torch
from common import *
from module.mc_contract_v3 import price_grid_v3
from family_design import design

def job(r,rep):
    torch.set_num_threads(1);c,m=unpack(r);cs=cases(c)
    seed=int(hashlib.sha256(f"{r['family']}:{rep}:MCv3".encode()).hexdigest()[:8],16)
    rows=price_grid_v3(cs,m,paths=CFG['mc']['paths_per_seed'],seed=seed,
        checkpoints=CFG['mc']['checkpoints'],path_chunk=CFG['mc']['path_chunk'])
    for row in rows:row.update(family=r['family'],split=r['split'],origin=r['origin'],structure=r['structure'],replicate=rep)
    return rows

def run():
    OUT.mkdir(exist_ok=True);cache=OUT/'mc_jobs';cache.mkdir(exist_ok=True)
    protocol_hash=sha(HERE/'protocol.json');engine_hash=sha(PROJECT/'module/mc_contract_v3.py')
    spec=dict(protocol_hash=protocol_hash,engine_hash=engine_hash,v2_engine_hash=sha(PROJECT/'module/mc_contract_v2.py'),generator_hash=sha(__file__),design_hash=sha(HERE/'family_design.py'),encoding_hash=sha(HERE/'common.py'),calendar_hash=sha(HERE/'evidence/kr_bank_calendar.json'))
    specpath=OUT/'label_spec.json'
    if specpath.exists():assert json.loads(specpath.read_text())==spec,'Label specification changed; use a new output version'
    else:specpath.write_text(json.dumps(spec,indent=2))
    families=design()
    assert len({r['family'] for r in families})==len(families)
    fingerprints=[]
    for r in families:
        d=r['contract'].copy();d.pop('name');fingerprints.append(hashlib.sha256(json.dumps([d,r['market']],sort_keys=True).encode()).hexdigest())
    assert len(set(fingerprints))==len(fingerprints)
    (OUT/'families.json').write_text(json.dumps(families,ensure_ascii=False,indent=2))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['contract','market']} for r in families]).to_csv(OUT/'family_splits.csv',index=False)
    started=time.time();rows=[];todo=[]
    for r in families:
        reps=1 if r['split'] in ['train','validation'] else 3
        for rep in range(reps):
            dest=cache/f'{r["family"]}_{rep}.json'
            if dest.exists():rows.extend(json.loads(dest.read_text()))
            else:todo.append((r,rep,dest))
    print('MCv3 families',len(families),'new jobs',len(todo),'cached rows',len(rows),flush=True)
    with ProcessPoolExecutor(max_workers=CFG['mc']['workers']) as pool:
        fs={pool.submit(job,r,rep):dest for r,rep,dest in todo}
        for i,f in enumerate(as_completed(fs),1):
            result=f.result();fs[f].write_text(json.dumps(result));rows.extend(result)
            if i%24==0 or i==len(fs):print('MCv3',i,'/',len(fs),'seconds',round(time.time()-started,1),flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'mc_seed_checkpoints.csv',index=False)
    keys=['family','split','origin','structure','axis','offset','paths']
    pooled=[]
    for key,g in raw.groupby(keys):
        r=dict(zip(keys,key));r['paths_per_seed']=r.pop('paths');n=int(g.paths.sum());r['total_paths']=n;r['replicates']=len(g)
        for stem in ['price','delta']:
            total=g[stem+'_sum'].sum();sq=g[stem+'_sumsq'].sum();mean=total/n
            var=max(0.,(sq-total*total/n)/(n-1));r[stem+'_krw']=mean*10000;r[stem+'_se_krw']=np.sqrt(var/n)*10000
        r['affected']=int(g.affected.sum());r['survival']=int(g.survival.sum());r['monthly_pv_krw']=g.monthly_pv_sum.sum()/n*10000;r['monthly_payment_count']=g.monthly_payment_count_sum.sum()/n;pooled.append(r)
    pooled=pd.DataFrame(pooled);pooled.to_csv(OUT/'mc_checkpoints.csv',index=False)
    final=pooled[pooled.paths_per_seed==40000].copy();final.to_csv(OUT/'mc_labels.csv',index=False)
    index=final.set_index(['family','axis','offset'])
    metadata=[]
    for split in ['train','validation','test','stress']:
        U=[];V=[];C=[];Y=[];D=[];SE=[];baseidx=[];group=[];axisidx=[];offset=[];meta=[]
        for r in [r for r in families if r['split']==split]:
            c,m=unpack(r);bi=len(Y);gi=len(set(group))
            for axis,h,cc in cases(c):
                u,v,x=encode(cc,m);label=index.loc[(r['family'],axis,h)]
                U.append(u);V.append(v);C.append(x);Y.append(label.price_krw/10000);D.append(label.delta_krw/10000);SE.append(label.delta_se_krw/10000)
                baseidx.append(bi);group.append(gi);axisidx.append(AXES.index(axis));offset.append(h)
                meta.append(dict(row=len(Y)-1,family=r['family'],split=split,origin=r['origin'],structure=r['structure'],monthly=c.has_monthly,axis=axis,offset=h,affected=int(label.affected)))
        np.savez_compressed(OUT/f'{split}.npz',U=U,V=V,C=C,Y=Y,D=D,SE=SE,base=baseidx,group=group,axis=axisidx,offset=offset)
        pd.DataFrame(meta).to_csv(OUT/f'{split}_rows.csv',index=False);metadata.extend(meta)
    pd.DataFrame(metadata).to_csv(OUT/'scenario_index.csv',index=False)
    # Report errors of increments, not errors of independent price means.
    a=pooled[pooled.paths_per_seed==20000];b=final
    z=a.merge(b,on=['family','axis','offset'],suffixes=('_20k','_40k'))
    z['change_20k_to_40k_krw']=z.delta_krw_40k-z.delta_krw_20k
    z[['family','axis','offset','change_20k_to_40k_krw','delta_se_krw_40k']].to_csv(OUT/'mc_convergence.csv',index=False)
    completion=dict(status='complete',families=len(families),scenario_rows=len(final),jobs=len(raw)//87 if False else len({(r['family'],r['replicate']) for r in rows}),
        seconds=time.time()-started,paths_per_seed=40000,unique_family_fingerprints=True,
        split_counts={s:sum(r['split']==s for r in families) for s in ['train','validation','test','stress']},
        engine_hash=engine_hash,protocol_hash=protocol_hash)
    (OUT/'labels_complete.json').write_text(json.dumps(completion,indent=2));print(json.dumps(completion),flush=True)

if __name__=='__main__':run()
