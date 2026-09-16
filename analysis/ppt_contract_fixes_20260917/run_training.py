"""40 configurations: matched legacy/corrected x A/B x DeepONet/XGB x five seeds."""
from pathlib import Path
import subprocess,sys,os,json,argparse
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;LOG=HERE/'results/logs';LOG.mkdir(exist_ok=True)
cores=sorted(os.sched_getaffinity(0))
def work(job,gpu=None,slot=0):
    arm,model,var,seed=job;name=f'{arm}_{model}_{var}_seed{seed}'
    cmd=[sys.executable,'-u',str(HERE/'train_stage1.py'),'--arm',arm,'--model',model,'--variant',var,'--seed',str(seed),'--device','cuda' if model=='deeponet' else 'cpu']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    start=(slot*4)%max(4,len(cores)//4*4);cmd=['taskset','-c',','.join(map(str,cores[start:start+4]))]+cmd
    with (LOG/(name+'.log')).open('a') as f:run=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(name,'exit',run.returncode,flush=True)
    return dict(payoff=arm,model=model,variant=var,seed=seed,returncode=run.returncode)
def queue(jobs,gpu,slot):return [work(j,gpu,slot) for j in jobs]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='2,3');a=ap.parse_args();gpus=list(map(int,a.gpus.split(',')))
    jobs={m:[(arm,m,v,s) for s in range(5) for v in 'AB' for arm in ('legacy','corrected')] for m in ('deeponet','xgb')}
    with ThreadPoolExecutor(len(gpus)+2) as pool:
        futures=[pool.submit(queue,jobs['deeponet'][j::len(gpus)],gpu,j) for j,gpu in enumerate(gpus)]
        futures += [pool.submit(queue,jobs['xgb'][j::2],None,len(gpus)+j) for j in range(2)]
        results=sum([f.result() for f in futures],[])
    (HERE/'results/training_jobs.json').write_text(json.dumps(results,indent=2)+'\n')
    assert len(results)==40 and all(r['returncode']==0 for r in results),results
if __name__=='__main__':main()
