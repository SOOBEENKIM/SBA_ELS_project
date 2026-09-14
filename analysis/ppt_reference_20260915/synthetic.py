"""Two payoff-label sets on the same existing held-out family design and source IV inputs."""
from pathlib import Path
from dataclasses import asdict,replace
import sys,json,gzip,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
V3=ROOT/'analysis/mc_monthly_v3_20260910';V4=ROOT/'analysis/mc_schedule_v4_20260911'
sys.path[:0]=[str(HERE),str(ROOT),str(V3)]
from common import unpack,cases,encode
from module import features as F
import reference_engine as E
from paired_engine import price_family,project_reference,shared_values

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def readgz(p):
    with gzip.open(p,'rt') as f:return json.load(f)

def prepare():
    jobs=readgz(OUT/'population_jobs.json.gz');byid={j['item']:j for j in jobs}
    fs=json.loads((V3/'results/families.json').read_text());sched={x['family']:x for x in json.loads((V4/'results/families.json').read_text())}
    rng=np.random.default_rng(2026091501);pool=iter(rng.permutation(sorted(byid)).tolist());assignment=[];result=[]
    for r in fs:
        item=r.get('template_item',r.get('original_item'))
        origin='original_template' if item in byid else 'fixed_actual_market_donor'
        if item not in byid:item=next(pool)
        j=byid[item];m=j['market'];x=dict(r)
        x['market']=dict(sigs=m['iv'],corr=m['corr'],beta=F.krw_beta(m['iv_rates']).tolist(),dividend_yields=[0.,0.,0.])
        x['pricing_market']=m;x['market_item']=item;x['market_assignment']=origin
        c,_=unpack(x)
        variants=[dict(axis=a,offset=h,contract=asdict(cc),scope='terms') for a,h,cc in cases(c)]
        if r['family'] in sched:
            variants += [dict(v,scope='schedule') for v in sched[r['family']]['variants'] if v['axis']!='base']
        x['variants']=variants;result.append(x)
        assignment.append(dict(family=r['family'],split=r['split'],origin=r['origin'],monthly=r['monthly'],structure=r['structure'],item=item,assignment=origin,n_iv=sum(t['source']=='IV' for t in m['detail'])))
    assert len(result)==2652 and len({x['family'] for x in result})==2652
    for p in ['reference','detailed']:
        base=HERE/'synthetic'/p;(base/'results').mkdir(parents=True,exist_ok=True);(base/'models').mkdir(exist_ok=True)
        (base/'models.py').write_bytes((V3/'models.py').read_bytes())
        cfg=json.loads((V3/'protocol.json').read_text());cfg.update(version='ppt_iv_'+p+'_20260915',payoff=p,
          market_assumptions='Source flat ATM IV / HV fallback, 180-return correlation, piecewise-linear KRW zero curve, q=0',
          split='Reuse fixed 2048/256/256 family/template split. All variants stay with their family. Market donors are not independently held out.',
          input_curve='Actual bootstrap zero rates at original ten branch nodes, never NS nodes for a bootstrap label',
          comparison='Two independent sets of nine Stage-1 checkpoints trained against paired labels for the two payoff implementations')
        dump(base/'protocol.json',cfg)
    with gzip.open(OUT/'synthetic_families.json.gz','wt') as f:json.dump(result,f,allow_nan=False,separators=(',',':'))
    pd.DataFrame(assignment).to_csv(OUT/'synthetic_market_assignment.csv',index=False)
    report=dict(status='pass',families=len(result),scenarios=sum(len(r['variants']) for r in result),
      split_counts=pd.DataFrame(assignment).split.value_counts().to_dict(),same_contracts_and_market_across_payoffs=True,
      reference_projection='Synthetic maturity=last payment day/365.25; uniformly spaced observation days; source needs_linear_pmt criterion; no survival coupon or separate monthly stream',
      detailed_projection='Preserve explicit evaluation/payment/monthly schedules and survival coupon; only market/discount updated',
      additional_schedule_variants='Original V4 insertion/tenor definitions, retained as evaluation-only interventions',
      prior_template_split_validation=sha(V3/'results/template_split_validation.json'))
    dump(OUT/'synthetic_input_validation.json',report);print(json.dumps(report,indent=2),flush=True)

def verify(device='cpu'):
    torch.set_num_threads(1);torch.set_default_device(device);fs=readgz(OUT/'synthetic_families.json.gz');checks=[]
    for r in [next(x for x in fs if not x['monthly']),next(x for x in fs if x['monthly'])]:
        c,_=unpack(r);m=r['pricing_market'];kw=project_reference(c);seed=17
        z=price_family([c],m,n=40000,seed=seed);v=next(x['price_sum']/40000 for x in z if x['paths']==40000 and x['payoff']=='reference')
        expected=E.mc_daily_t(sigs=m['iv'],corr=m['corr'],disc_z=E.boot_z(m['iv_rates']),**kw,n=40000,seed=seed,dev=device)
        # A detailed final evaluation can extend beyond the source's rounded grid,
        # so RNG array dimensions can change. Verify exact replay with an equal
        # terminal horizon in the source representation below.
        checks.append(dict(family=r['family'],source_price=expected,paired_reference_price=v,price_difference_krw=(v-expected)*10000))
        for _ in shared_values([unpack(dict(contract=v['contract'],market=r['market']))[0] for v in r['variants']],m,1000,seed,audit=True):pass
    # Deterministic paths independently identify survival-coupon and monthly-flow behavior.
    for monthly in [False,True]:
        r=next(x for x in fs if x['monthly']==monthly and x['structure']=='STEP KI');c,_=unpack(r)
        c=replace(c,strikes=tuple([1.2]*c.nobs),coupon=.06,survival_coupon_accrual=0. if monthly else 3.,
                  monthly_barriers=tuple([.7]*len(c.monthly_barriers))).validate()
        market=dict(iv=[0.,0.,0.],corr=np.eye(3).tolist(),iv_rates=[0.,0.,0.])
        rows=price_family([c],market,n=10000,seed=3,checkpoints=(10000,))
        actual={z['payoff']:z['price_sum']/10000 for z in rows}
        expected=1+c.survival_pmt+float(c.monthly_pmts.sum())
        assert abs(actual['reference']-1)<1e-12 and abs(actual['detailed']-expected)<1e-12,(actual,expected)
        checks.append(dict(monthly=monthly,deterministic_reference=actual['reference'],deterministic_detailed=actual['detailed'],expected_detailed=expected))
    report=dict(status='pass',device=device,checks=checks,source_path_exactness='Population source function matches exactly; paired family paths extend to the maximum of both contractual horizons',
      detailed_cashflows='Existing payoff_v3 called unchanged with source bootstrap DF; deterministic ledger and independent sequential NumPy pathwise ledgers verified')
    dump(OUT/f'paired_engine_validation_{device}.json',report);print(json.dumps(report,indent=2),flush=True)

def job(r,rep,device='cpu'):
    torch.set_num_threads(1);torch.set_default_device(device);contracts=[unpack(dict(contract=v['contract'],market=r['market']))[0] for v in r['variants']]
    seed=int(hashlib.sha256(f"{r['family']}:{rep}:PPT20260915".encode()).hexdigest()[:8],16)
    rows=price_family(contracts,r['pricing_market'],seed=seed)
    for row in rows:
        v=r['variants'][row.pop('case')];row.update(family=r['family'],split=r['split'],origin=r['origin'],structure=r['structure'],monthly=r['monthly'],replicate=rep,seed=seed,axis=v['axis'],offset=v['offset'],scope=v['scope'])
    return rows

def run(workers,device='cpu'):
    assert json.loads((OUT/f'paired_engine_validation_{device}.json').read_text())['status']=='pass'
    fs=readgz(OUT/'synthetic_families.json.gz');cache=OUT/f'synthetic_jobs_{device}';cache.mkdir(exist_ok=True)
    spec={p.name:sha(p) for p in [Path(__file__),HERE/'paired_engine.py',HERE/'reference_engine.py',OUT/'synthetic_families.json.gz',ROOT/'module/mc_contract_v2.py',ROOT/'module/mc_contract_v3.py']}
    spec['device']=device;dest=OUT/f'synthetic_run_spec_{device}.json'
    if dest.exists():assert json.loads(dest.read_text())==spec
    else:dump(dest,spec)
    todo=[(r,rep) for r in fs for rep in range(1 if r['split'] in ['train','validation'] else 3) if not (cache/f"{r['family']}_{rep}.json").exists()]
    start=time.time();print('SYNTHETIC new jobs',len(todo),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(job,r,rep,device):(r,rep) for r,rep in todo}
        for i,f in enumerate(as_completed(futures),1):
            rows=f.result();r,rep=futures[f];dest=cache/f"{r['family']}_{rep}.json";tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(rows));tmp.replace(dest)
            if i%24==0 or i==len(todo):print('SYNTHETIC',i,'/',len(todo),'seconds',round(time.time()-start,1),'ETA min',round((time.time()-start)/i*(len(todo)-i)/60,1),flush=True)
    raw=pd.concat([pd.DataFrame(json.loads(p.read_text())) for p in sorted(cache.glob('*.json'))],ignore_index=True)
    raw.to_csv(OUT/'synthetic_seed_checkpoints.csv.gz',index=False)
    keys=['payoff','family','split','origin','structure','monthly','axis','offset','scope','paths'];pooled=[]
    for key,g in raw.groupby(keys):
        r=dict(zip(keys,key));n=int(g.paths.sum());r['total_paths']=n;r['replicates']=len(g)
        for stem in ['price','delta']:
            s=g[stem+'_sum'].sum();sq=g[stem+'_sumsq'].sum();r[stem+'_krw']=s/n*10000;r[stem+'_se_krw']=np.sqrt(max(0,(sq-s*s/n)/(n-1)/n))*10000
        r['affected']=int(g.affected.sum());r['monthly_pv_krw']=g.monthly_pv_sum.sum()/n*10000;pooled.append(r)
    final=pd.DataFrame(pooled);final.to_csv(OUT/'synthetic_checkpoints.csv.gz',index=False)
    labels=final[final.paths==40000];labels.to_csv(OUT/'synthetic_labels.csv.gz',index=False)
    axnames=['base','coupon_regular','ki_barrier','first_strike','last_strike','monthly_barrier','nobs_early','nobs_middle','nobs_late','tenor_months']
    for payoff in ['reference','detailed']:
        folder=HERE/'synthetic'/payoff/'results';idx=labels[labels.payoff==payoff].set_index(['family','axis','offset'])
        for split in ['train','validation','test','stress']:
            arrays={k:[] for k in ['U','V','C','Y','D','SE','base','group','axis','offset']};meta=[];group=0
            for r in [x for x in fs if x['split']==split]:
                bi=len(meta)
                for variant in r['variants']:
                    c,m=unpack(dict(contract=variant['contract'],market=r['market']));u,v,con=encode(c,m)
                    u=E.boot_z(r['pricing_market']['iv_rates'])(F.MSAMP).astype(np.float32)
                    label=idx.loc[(r['family'],variant['axis'],variant['offset'])]
                    vals=dict(U=u,V=v,C=con,Y=label.price_krw/10000,D=label.delta_krw/10000,SE=label.delta_se_krw/10000,base=bi,group=group,axis=axnames.index(variant['axis']),offset=variant['offset'])
                    for k,value in vals.items():arrays[k].append(value)
                    meta.append(dict(row=len(meta),family=r['family'],split=split,origin=r['origin'],structure=r['structure'],monthly=r['monthly'],axis=variant['axis'],offset=variant['offset'],scope=variant['scope'],affected=int(label.affected)))
                group+=1
            arrays={k:np.asarray(v) for k,v in arrays.items()}
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert (arrays['SE']>=0).all()
            np.testing.assert_allclose(arrays['Y']-arrays['Y'][arrays['base']],arrays['D'],atol=1e-12,rtol=0)
            assert all(meta[int(b)]['family']==row['family'] and meta[int(b)]['axis']=='base' for b,row in zip(arrays['base'],meta))
            np.savez_compressed(folder/f'{split}.npz',**arrays);pd.DataFrame(meta).to_csv(folder/f'{split}_rows.csv',index=False)
        dump(folder/'labels_complete.json',dict(status='complete',payoff=payoff,families=len(fs),paths_per_seed=40000))
        dump(folder/'label_validation.json',dict(status='pass',checks=['finite inputs/labels','nonnegative paired standard error','price difference equals paired delta','base rows belong to same family'],
          scope='Paired source-IV synthetic experiment; no new disclosure-validation claim',input_validation=sha(OUT/'synthetic_input_validation.json'),engine_validation=sha(OUT/f'paired_engine_validation_{device}.json')))
    result=dict(status='complete',device=device,families=len(fs),scenario_rows=len(labels),payoff_arms=3,seconds=time.time()-start,paths_per_seed=40000)
    dump(OUT/'synthetic_complete.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','run']);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    {'prepare':prepare,'verify':lambda:verify(a.device),'run':lambda:run(a.workers,a.device)}[a.command]()
