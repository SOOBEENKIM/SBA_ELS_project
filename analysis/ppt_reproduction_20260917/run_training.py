"""Run all 20 Stage-1 configurations with bounded CPU/GPU use; resume complete folds."""
from pathlib import Path
import subprocess,sys,os,json,time
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
LOG=HERE/'results/logs';LOG.mkdir(parents=True,exist_ok=True)
cores=sorted(os.sched_getaffinity(0))
def work(job):
 model,var,seed=job;name=f'{model}_{var}_seed{seed}'
 cmd=[sys.executable,str(HERE/'train_stage1.py'),'--model',model,'--variant',var,'--seed',str(seed),'--device','cuda' if model=='deeponet' else 'cpu']
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 if model=='deeponet':env['CUDA_VISIBLE_DEVICES']=str(seed%2)
 # Bound CPU threads even for the source XGB n_jobs=0 setting.
 slot=(seed*2+(var=='B'))%max(1,len(cores)//4)
 cmd=['taskset','-c',','.join(map(str,cores[slot*4:slot*4+4]))]+cmd
 with (LOG/(name+'.log')).open('a') as f:
  run=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(name,'exit',run.returncode,flush=True)
 return dict(model=model,variant=var,seed=seed,returncode=run.returncode)
def group(model,workers):
 with ThreadPoolExecutor(workers) as pool:return list(pool.map(work,[(model,v,s) for s in range(5) for v in 'AB']))
if __name__=='__main__':
 with ThreadPoolExecutor(2) as pool:
  futures=[pool.submit(group,'deeponet',2),pool.submit(group,'xgb',2)];results=sum([f.result() for f in futures],[])
 (HERE/'results/training_jobs.json').write_text(json.dumps(results,indent=2)+'\n')
 if any(x['returncode'] for x in results):sys.exit(1)
 subprocess.run([sys.executable,str(HERE/'plot_results.py')],check=True)
