"""Paired HV/IV repricing, MC uncertainty and frozen DeepONet evaluation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
import pandas as pd
import torch
from prepare import ROOT,HERE,V3,V4,OUT,sha,dump,preserved
sys.path.insert(0,str(V4))
from schedule_mc import shared_summaries,price_grid
from common import unpack,encode
from module.mc_contract_v3 import payoff_v3
from models import load_predictor

@torch.no_grad()
def paired(r,rep,paths=40000,checkpoints=(10000,20000,40000),path_chunk=2500,tblock=256):
    torch.set_num_threads(1)
    seed=int(hashlib.sha256(f"{r['family']}:{rep}:IV20260914".encode()).hexdigest()[:8],16)
    contracts=[unpack(dict(contract=v['contract'],market=r['markets']['HV']))[0] for v in r['variants']]
    markets=[unpack(dict(contract=r['contract'],market=r['markets'][a]))[1] for a in ['HV','IV']]
    streams=[shared_summaries(contracts,m,paths,seed,path_chunk,tblock) for m in markets]
    sums=np.zeros((3,len(contracts),7));done=0;ci=0;result=[]
    for (hs,hp),(vs,vp) in zip(*streams,strict=True):
        pays=[];monthly=[]
        for ss,pp in [(hs,hp),(vs,vp)]:
            z=[payoff_v3(s,p,c,details=True) for s,p,c in zip(ss,pp,contracts)]
            pays.append(torch.stack([x[0] for x in z]));monthly.append(torch.stack([x[2] for x in z]))
        pays.append(pays[1]-pays[0]);monthly.append(monthly[1]-monthly[0]);begin=0
        while begin<pays[0].shape[1]:
            stop=min(pays[0].shape[1],begin+checkpoints[ci]-done)
            for a,(p,cp) in enumerate(zip(pays,monthly)):
                y=p[:,begin:stop];d=y-y[0];c=cp[:,begin:stop]
                sums[a]+=torch.stack([y.sum(1),(y*y).sum(1),d.sum(1),(d*d).sum(1),
                    (d.abs()>1e-12).sum(1),c.sum(1),(y-c).sum(1)],1).numpy()
            done+=stop-begin;begin=stop
            if done==checkpoints[ci]:
                for a,arm in enumerate(['HV','IV','IV_minus_HV']):
                    for j,s in enumerate(sums[a]):
                        v=r['variants'][j]
                        result.append(dict(family=r['family'],origin=r['origin'],monthly=r['monthly'],structure=r['structure'],
                            arm=arm,case=j,axis=v['axis'],offset=v['offset'],replicate=rep,paths=done,seed=seed,
                            price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],affected=int(s[4]),
                            monthly_pv_sum=s[5],nonmonthly_pv_sum=s[6]))
                ci+=1
                if ci==len(checkpoints):return result
    raise RuntimeError('Incomplete paired calculation')

def verify(fs):
    from copy import deepcopy
    errors=[]
    for monthly in [False,True]:
        r=next(x for x in fs if x['monthly']==monthly);r=deepcopy(r);r['markets']['IV']=deepcopy(r['markets']['HV'])
        n=500;got=pd.DataFrame(paired(r,0,paths=n,checkpoints=(n,),path_chunk=250,tblock=256))
        assert got.loc[got.arm.eq('IV_minus_HV'),['price_sum','delta_sum','price_sumsq','delta_sumsq']].eq(0).all().all()
        contracts=[unpack(dict(contract=v['contract'],market=r['market']))[0] for v in r['variants']]
        _,m=unpack(r);seed=int(got.seed.iloc[0])
        old=pd.DataFrame(price_grid(contracts,m,paths=n,seed=seed,checkpoints=(n,),path_chunk=250,tblock=256))
        new=got[got.arm.eq('HV')].sort_values('case')
        for k in ['price_sum','price_sumsq','delta_sum','delta_sumsq','monthly_pv_sum','nonmonthly_pv_sum']:
            e=float(abs(new[k].to_numpy()-old.sort_values('case')[k].to_numpy()).max());errors.append(e)
            np.testing.assert_allclose(new[k],old[k],rtol=0,atol=1e-9)
    for r in fs:
        a=r['markets']['HV'].copy();b=r['markets']['IV'].copy();a.pop('sigs');b.pop('sigs');assert a==b
        assert [x['iv'] for x in r['iv_matches']]==r['markets']['IV']['sigs']
        assert all(0<=x['age_days']<=7 and x['iv_date']<=r['issue'] for x in r['iv_matches'])
        keys=[(x['axis'],x['offset']) for x in r['variants']];assert len(keys)==len(set(keys))
        assert r['variants'][0]['axis']=='base' and r['variants'][0]['contract']==r['contract']
    dump(OUT/'pre_mc_verification.json',dict(status='pass',families=len(fs),native_schedule_engine_max_sum_error=max(errors),
        identical_sigma_yields_zero_difference=True,only_sigma_changes=True,iv_no_future_or_stale_rows=True))

def pool_rows(raw):
    keys=['family','origin','monthly','structure','arm','case','axis','offset','paths'];out=[]
    for key,g in raw.groupby(keys,sort=True):
        r=dict(zip(keys,key));r['paths_per_seed']=r.pop('paths');n=int(g.paths.sum());r['total_paths']=n;r['replicates']=len(g)
        for stem in ['price','delta']:
            total=g[stem+'_sum'].sum();sq=g[stem+'_sumsq'].sum();var=max(0.,(sq-total*total/n)/(n-1))
            r[stem+'_krw']=total/n*10000;r[stem+'_se_krw']=np.sqrt(var/n)*10000
        r['affected']=int(g.affected.sum());r['monthly_pv_krw']=g.monthly_pv_sum.sum()/n*10000
        r['nonmonthly_pv_krw']=g.nonmonthly_pv_sum.sum()/n*10000;out.append(r)
    return pd.DataFrame(out)

def metrics(g):
    pe=g.pred_price_krw-g.price_krw;den=((g.price_krw-g.price_krw.mean())**2).sum()
    r=dict(rows=len(g),families=g.family.nunique(),price_mae=float(abs(pe).mean()),price_rmse=float(np.sqrt(np.mean(pe**2))),
        price_r2=float(1-np.sum(pe**2)/den) if den>1e-12 else np.nan)
    h=g[g.axis.ne('base')]
    if len(h):
        e=h.pred_delta_krw-h.delta_krw;resolved=(abs(h.delta_krw)>1.96*h.delta_se_krw)&(h.affected>=30)
        r.update(delta_mae=float(abs(e).mean()),delta_rmse=float(np.sqrt(np.mean(e**2))),delta_bias=float(e.mean()),
            within_1=float((abs(e)<=1).mean()),within_5=float((abs(e)<=5).mean()),within_10=float((abs(e)<=10).mean()),
            resolved_n=int(resolved.sum()),resolved_sign=float((np.sign(h.loc[resolved,'pred_delta_krw'])==np.sign(h.loc[resolved,'delta_krw'])).mean()) if resolved.any() else np.nan)
    return r

def evaluate(fs,labels,cfg):
    torch.set_num_threads(2);base=[];meta=[];U=[];V=[];C=[]
    for r in fs:
        for arm in ['HV','IV']:
            bi=len(meta)
            for i,v in enumerate(r['variants']):
                c,m=unpack(dict(contract=v['contract'],market=r['markets'][arm]));u,w,x=encode(c,m)
                U.append(u);V.append(w);C.append(x);base.append(bi)
                meta.append(dict(family=r['family'],arm=arm,case=i))
    data=dict(U=np.asarray(U),V=np.asarray(V),C=np.asarray(C));base=np.asarray(base)
    meta=pd.DataFrame(meta).merge(labels,on=['family','arm','case'],how='left',validate='one_to_one',sort=False)
    assert not meta.price_krw.isna().any()
    assignment=pd.read_csv(OUT/'market_assignment.csv')[['family','arm','sigma_outside_training']]
    meta=meta.merge(assignment,on=['family','arm'],validate='many_to_one',sort=False)
    meta['cohort']=np.where(meta.origin.eq('published_monthly'),'published_monthly',np.where(meta.monthly,'monthly_test','regular_test'))
    meta['suite']=np.where(meta.axis.str.startswith('nobs_')|meta.axis.eq('tenor_months'),'schedule','terms')
    out=[]
    for model in cfg['frozen_models']['arms']:
        predictions=[]
        for seed in cfg['frozen_models']['seeds']:
            call,_=load_predictor(V3/'models'/f'{model}_seed{seed}.pt');p=call(data)*10000;predictions.append(p)
            g=meta.copy();g['model']=model;g['model_seed']=str(seed);g['pred_price_krw']=p;g['pred_delta_krw']=p-p[base];out.append(g)
        p=np.mean(predictions,axis=0);g=meta.copy();g['model']=model;g['model_seed']='ensemble';g['pred_price_krw']=p;g['pred_delta_krw']=p-p[base];out.append(g)
    pred=pd.concat(out,ignore_index=True);pred['price_error_krw']=pred.pred_price_krw-pred.price_krw;pred['delta_error_krw']=pred.pred_delta_krw-pred.delta_krw
    pred.to_csv(OUT/'predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    summaries=[];offsets=[];domains=[]
    for key,g in pred.groupby(['cohort','arm','model','model_seed']):
        k=dict(zip(['cohort','arm','model','model_seed'],key))
        for suite,z in [('ALL',g),('terms',g[g.suite.eq('terms')]),('schedule',g[g.suite.eq('schedule')])]:
            if len(z):summaries.append(dict(**k,axis=suite,**metrics(z)))
        for axis,z in g.groupby('axis'):summaries.append(dict(**k,axis=axis,**metrics(z)))
        for (axis,h),z in g[g.axis.ne('base')].groupby(['axis','offset']):offsets.append(dict(**k,axis=axis,offset=h,**metrics(z)))
        for outside,z in g.groupby('sigma_outside_training'):domains.append(dict(**k,sigma_outside_training=outside,**metrics(z)))
    pd.DataFrame(summaries).to_csv(OUT/'model_metrics.csv',index=False)
    pd.DataFrame(offsets).to_csv(OUT/'model_metrics_by_offset.csv',index=False)
    pd.DataFrame(domains).to_csv(OUT/'model_metrics_by_sigma_domain.csv',index=False)
    return len(pred)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    cfg=json.loads((HERE/'protocol.json').read_text());fs=json.loads((OUT/'families.json').read_text());started=time.time()
    assert preserved()==cfg['preserved_inputs'];verify(fs)
    if args.verify_only:print((OUT/'pre_mc_verification.json').read_text(),flush=True);return
    spec=dict(protocol=sha(HERE/'protocol.json'),families=sha(OUT/'families.json'),runner=sha(__file__),prepare=sha(HERE/'prepare.py'))
    p=OUT/'label_spec.json'
    if p.exists():assert json.loads(p.read_text())==spec,'Specification changed; refuse stale MC cache'
    else:dump(p,spec)
    cache=OUT/'mc_jobs';cache.mkdir(exist_ok=True);raw=[];todo=[]
    for r in fs:
        for rep in range(cfg['replicates']):
            p=cache/f"{r['family']}_{rep}.json"
            if p.exists():raw.extend(json.loads(p.read_text()))
            else:todo.append((r,rep,p))
    print('PAIRED IV MC',len(fs),'families',len(todo),'jobs; 40,000 paths/seed x 3',flush=True)
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        tasks={pool.submit(paired,r,rep):p for r,rep,p in todo}
        for i,f in enumerate(as_completed(tasks),1):
            rows=f.result();dump(tasks[f],rows);raw.extend(rows)
            if i%12==0 or i==len(todo):print('MC',i,'/',len(todo),'seconds',round(time.time()-started,1),flush=True)
    raw=pd.DataFrame(raw).sort_values(['family','arm','case','replicate','paths'])
    raw.to_csv(OUT/'mc_seed_checkpoints.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    pooled=pool_rows(raw);pooled.to_csv(OUT/'mc_checkpoints.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
    final=pooled[pooled.paths_per_seed.eq(40000)].copy();final.to_csv(OUT/'mc_labels.csv',index=False)
    assert final.replicates.eq(3).all() and final.total_paths.eq(120000).all()
    for _,g in final.groupby(['family','arm']):
        b=g.loc[g.axis.eq('base'),'price_krw'].iloc[0];np.testing.assert_allclose(g.price_krw-b,g.delta_krw,atol=1e-8,rtol=0)
    a=final[final.arm.eq('HV')].set_index(['family','case']);b=final[final.arm.eq('IV')].set_index(['family','case']);d=final[final.arm.eq('IV_minus_HV')].set_index(['family','case'])
    for k in ['price_krw','delta_krw']:np.testing.assert_allclose(b[k]-a[k],d[k],atol=1e-8,rtol=0)
    n=evaluate(fs,final[final.arm.isin(['HV','IV'])],cfg)
    assert preserved()==cfg['preserved_inputs']
    result=dict(status='complete',families=len(fs),cases_per_arm=sum(len(r['variants']) for r in fs),
        paths_per_seed=40000,replicates=3,paired_jobs=len(fs)*3,prediction_rows=n,seconds=time.time()-started,
        existing_artifacts_unchanged=True,retrained=False)
    dump(OUT/'complete.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
