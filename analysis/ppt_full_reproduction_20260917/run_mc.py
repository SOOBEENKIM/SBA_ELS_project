"""Full independent A/B repricing, resumable per product, frozen protocol and input hashes."""
from pathlib import Path
import sys,json,gzip,hashlib,time,os,argparse,subprocess,platform
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';WORK=OUT/'mc_workers';WORK.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'ppt_reproduction_20260917'))
from market_rules import curves
from engine import price,contract,SOURCE_HASH
FILES=[OUT/'rebuilt_universe.parquet',OUT/'fresh_market_inputs.parquet',OUT/'dividend_events.json.gz',HERE/'engine.py',HERE/'market_rules.py',HERE/'run_mc.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
 spec=dict(paths=40000,seed='original per-product mc_seed',path_chunk=20000,time_block=512,device='cuda',engine_source_sha256=SOURCE_HASH,inputs={str(p.relative_to(HERE)):sha(p) for p in FILES},python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,payoff='original notebook5 retained, including principal-only no-KI-survival and monthly linear coupon approximation',dividend_window='330 calendar days ending at latest known preissue ex-date; declared reconstruction assumption; annual calendar anniversaries',cache_MC_used_as_labels=False)
 p=OUT/'full_mc_protocol.json'
 if p.exists():assert json.loads(p.read_text())==spec,'Protocol/input changed. Use a new output directory.'
 else:p.write_text(json.dumps(spec,indent=2)+'\n')
 return spec

def worker(rank,workers):
 torch.set_num_threads(2);torch.cuda.set_device(rank%torch.cuda.device_count());device=f'cuda:{rank%torch.cuda.device_count()}'
 s=pd.read_parquet(OUT/'rebuilt_universe.parquet');v=pd.read_parquet(OUT/'fresh_market_inputs.parquet');assert v.item.equals(s.item)
 with gzip.open(OUT/'dividend_events.json.gz','rt') as f:e=json.load(f)
 assert e['items']==s.item.tolist();ok=v.fail.isna()&~v.stale_px;ix=np.flatnonzero(ok)[rank::workers]
 path=WORK/f'worker_{rank}.jsonl';prior={}
 if path.exists():
  for line in path.read_text().splitlines():
   try:r=json.loads(line)
   except json.JSONDecodeError:raise RuntimeError('Incomplete worker record; recover explicitly before resume')
   assert r['item'] not in prior;prior[r['item']]=r
 start=time.monotonic();new=0
 with path.open('a',buffering=1) as f:
  for k,i in enumerate(ix):
   row=s.iloc[i];m=v.iloc[i]
   if row['item'] in prior:continue
   ns,bt,_=curves(int(row.isu_ord));c=contract(row);rho=[m.rho_12,m.rho_13,m.rho_23];C=np.array([[1,rho[0],rho[1]],[rho[0],1,rho[2]],[rho[1],rho[2],1]])
   result=dict(i=int(i),item=str(row['item']),seed=int(row.mc_seed),paths=40000,worker=rank)
   for x,curve,events in [('A',ns,[]),('B',bt,e['events'][i])]:
    value,se=price([m[f's{x}_{j}'] for j in (1,2,3)],C,curve,c,events,int(row.mc_seed),device=device,n=40000)
    assert np.isfinite([value,se]).all() and se>=0
    result['mc_'+x]=value;result['se_'+x]=se
   f.write(json.dumps(result,separators=(',',':'))+'\n');new+=1
   if new%500==0:
    f.flush();os.fsync(f.fileno());print('MC',rank,'completed',k+1,'/',len(ix),'new',new,'seconds',round(time.monotonic()-start,1),flush=True)
 print('MC WORKER DONE',rank,'total',len(ix),'new',new,'seconds',time.monotonic()-start,flush=True)

def finalize():
 v=pd.read_parquet(OUT/'fresh_market_inputs.parquet');rows=[]
 for p in sorted(WORK.glob('worker_*.jsonl')):
  rows.extend(json.loads(line) for line in p.read_text().splitlines())
 q=pd.DataFrame(rows);ok=v.fail.isna()&~v.stale_px
 assert q.item.is_unique and set(q.item)==set(v.loc[ok,'item']) and (q.paths==40000).all()
 values=q.set_index('item')
 for col in ['mc_A','mc_B','se_A','se_B']:v[col]=v.item.map(values[col])
 v.to_parquet(OUT/'fresh_mc_variants.parquet',index=False)
 q.to_csv(OUT/'fresh_mc_prices.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 summary=dict(status='complete',population=len(v),priced=len(q),excluded=len(v)-len(q),total_paths_per_variant=int(q.paths.sum()),total_paths_all_variants=int(q.paths.sum()*2),target='independently recomputed MC prices',cache_reused=False,output_sha256=sha(OUT/'fresh_mc_variants.parquet'))
 (OUT/'full_mc_completion.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--rank',type=int);ap.add_argument('--finalize-only',action='store_true');a=ap.parse_args()
 if a.finalize_only:finalize();return
 if a.rank is not None:worker(a.rank,a.workers);return
 freeze();cores=sorted(os.sched_getaffinity(0))
 def task(rank):
  cmd=[sys.executable,'-u',str(HERE/'run_mc.py'),'--workers',str(a.workers),'--rank',str(rank)]
  if len(cores)>=a.workers*2:cmd=['taskset','-c',','.join(map(str,cores[2*rank:2*rank+2]))]+cmd
  with (WORK/f'worker_{rank}.log').open('a') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  return rank,rc
 with ThreadPoolExecutor(a.workers) as pool:rc=list(pool.map(task,range(a.workers)))
 assert all(code==0 for _,code in rc),rc
 finalize()
if __name__=='__main__':main()
