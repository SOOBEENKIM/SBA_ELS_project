"""Reprice every supported contract, paired payoff arms, frozen A/B market inputs."""
from pathlib import Path
import sys,json,gzip,hashlib,time,os,argparse,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';BASE=HERE.parent/'ppt_full_reproduction_20260917/results';WORK=OUT/'mc_workers';WORK.mkdir(exist_ok=True)
from contracts import ContractV3
from engine import price,legacy,ARMS
sys.path.insert(0,str(HERE.parent/'ppt_full_reproduction_20260917'))
from market_rules import curves,dividends

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    files=[OUT/'contracts.json.gz',OUT/'input_audit.json',BASE/'rebuilt_universe.parquet',BASE/'fresh_market_inputs.parquet',BASE/'dividend_events.json.gz',HERE/'engine.py',HERE/'contracts.py',HERE/'run_mc.py',HERE.parent/'ppt_full_reproduction_20260917/market_rules.py']
    spec=dict(paths=40000,seed='same original mc_seed per product',arms=ARMS,market_variants={'A':'HV120, historical correlation, NS, no dividend','B':'EWMA120 lambda .99, same correlation, CD/IRS bootstrap, same discrete dividend rule'},comparison='Three payoff arms on identical random paths for each market variant. New path horizon=max(original horizon, actual last evaluation); previous cached prices are a separate check.',target='MC per unit face; Stage 2 not used',hashes={str(f.relative_to(ROOT)):sha(f) for f in files})
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==json.loads(json.dumps(spec)),'Frozen inputs changed'
    else:dest.write_text(json.dumps(spec,indent=2)+'\n')

def worker(rank,gpus):
    device='cuda:'+str(gpus[rank]);torch.cuda.set_device(gpus[rank]);torch.set_default_device(device);torch.set_num_threads(2)
    s=pd.read_parquet(BASE/'rebuilt_universe.parquet');v=pd.read_parquet(BASE/'fresh_market_inputs.parquet')
    allc=json.loads(gzip.decompress((OUT/'contracts.json.gz').read_bytes()))
    contracts=[r for r in allc if pd.isna(v.iloc[r['i']]['fail']) and not v.iloc[r['i']].stale_px][rank::len(gpus)]
    events=json.loads(gzip.decompress((BASE/'dividend_events.json.gz').read_bytes()))
    assert events['items']==s.item.tolist()
    path=WORK/f'worker_{rank}.jsonl';prior={json.loads(line)['item'] for line in path.read_text().splitlines()} if path.exists() else set()
    start=time.monotonic();new=0
    with path.open('a',buffering=1) as f:
        for k,record in enumerate(contracts):
            if record['item'] in prior:continue
            i=record['i'];r=s.iloc[i];m=v.iloc[i];c=ContractV3(**record['contract']).validate();raw=legacy.contract(r)
            ns,bt,_=curves(int(r.isu_ord));rho=[m.rho_12,m.rho_13,m.rho_23];corr=np.array([[1,rho[0],rho[1]],[rho[0],1,rho[2]],[rho[1],rho[2],1]])
            ev=events['events'][i];N=max(round(r.tenor*365),c.obs_days[-1])
            if N>round(r.tenor*365):
                drops,_,_=dividends((r.udl1,r.udl2,r.udl3),int(r.isu_ord),N/365.)
                di,dj=np.nonzero(drops);ev=[(int(a),int(b),float(drops[a,b])) for a,b in zip(di,dj)]
            result=dict(i=i,item=r['item'],paths=40000,seed=int(r.mc_seed),monthly=c.has_monthly,structure=c.structure)
            for variant,curve,drop in [('A',ns,[]),('B',bt,ev)]:
                out=price([m[f's{variant}_{j}'] for j in (1,2,3)],corr,curve,raw,c,drop,int(r.mc_seed),n=40000)
                assert np.isfinite([*out['prices'],*out['se'],*out['step_delta'],*out['step_se']]).all()
                for j,arm in enumerate(ARMS):result[f'mc_{variant}_{arm}']=out['prices'][j];result[f'se_{variant}_{arm}']=out['se'][j]
                for j,step in enumerate(['survival','schedule_monthly']):result[f'delta_{variant}_{step}']=out['step_delta'][j];result[f'se_delta_{variant}_{step}']=out['step_se'][j]
            f.write(json.dumps(result,separators=(',',':'))+'\n');new+=1
            if new%250==0:
                f.flush();os.fsync(f.fileno());print('MC',rank,k+1,'/',len(contracts),'seconds',round(time.monotonic()-start,1),flush=True)
    print('DONE',rank,len(contracts),'seconds',time.monotonic()-start,flush=True)

def finalize():
    q=pd.DataFrame([json.loads(line) for p in sorted(WORK.glob('worker_*.jsonl')) for line in p.read_text().splitlines()]).sort_values('i')
    meta=pd.read_csv(OUT/'contract_audit.csv.gz');expected=set(meta.loc[meta.market_valid,'item'])
    assert q.item.is_unique and set(q.item)==expected and q.paths.eq(40000).all()
    q.to_parquet(OUT/'paired_prices.parquet',index=False);q.to_csv(OUT/'paired_prices.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary=dict(status='complete',priced=len(q),paths_per_market_variant=int(q.paths.sum()),market_variants=2,payoff_arms=3,output_hash=sha(OUT/'paired_prices.parquet'))
    (OUT/'completion.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='1,2,3');p.add_argument('--rank',type=int);p.add_argument('--finalize-only',action='store_true');a=p.parse_args();gpus=list(map(int,a.gpus.split(',')))
    if a.finalize_only:finalize();return
    if a.rank is not None:worker(a.rank,gpus);return
    freeze();cores=sorted(os.sched_getaffinity(0))
    def task(rank):
        cmd=['taskset','-c',','.join(map(str,cores[2*rank:2*rank+2])),sys.executable,'-u',str(HERE/'run_mc.py'),'--gpus',a.gpus,'--rank',str(rank)]
        with (WORK/f'worker_{rank}.log').open('a') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        return rank,rc
    with ThreadPoolExecutor(len(gpus)) as pool:results=list(pool.map(task,range(len(gpus))))
    assert all(rc==0 for _,rc in results),results
    finalize()
if __name__=='__main__':main()
