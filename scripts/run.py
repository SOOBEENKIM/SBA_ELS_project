"""Reproduce in a new isolated run directory. Frozen repository outputs are inputs."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
V3='analysis/mc_monthly_v3_20260910';V4='analysis/mc_schedule_v4_20260911';AU='analysis/mc_v3_audit_20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','mc','full']);ap.add_argument('--run-dir',required=True);args=ap.parse_args()
    dest=Path(args.run_dir).resolve();assert not dest.exists(),'Use a new run directory; frozen results are never overwritten.'
    assert dest!=ROOT and ROOT not in dest.parents or str(dest).startswith(str(ROOT/'runs')+'/'),'Run within runs/ or outside the checkout'
    critical=[ROOT/'module/mc_contract_v2.py',ROOT/'module/mc_contract_v3.py',ROOT/V3/'common.py',ROOT/V3/'models.py',ROOT/V3/'protocol.json']+list((ROOT/V3/'models').glob('*.pt'))+[ROOT/V3/'results'/n for n in ['families.json','mc_labels.csv','predictions.csv','model_selection.json']]
    original={str(p.relative_to(ROOT)):sha(p) for p in critical}
    for name in ['data','module','analysis','scripts','docs']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.gz','mc_jobs'))
    if (ROOT/'README.md').exists():shutil.copyfile(ROOT/'README.md',dest/'README.md')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/els-mpl')
    env['PYTHONPATH']=os.pathsep.join([str(dest),str(dest/V3)])
    log=dest/'execution.log';steps=[]
    def call(relative):
        t=time.time();print('RUN',relative,flush=True)
        with log.open('a') as f:
            f.write('\nRUN '+relative+'\n');f.flush()
            proc=subprocess.Popen([sys.executable,str(dest/relative)],cwd=dest,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:print(line,end='',flush=True);f.write(line)
            rc=proc.wait()
        if rc:raise RuntimeError(f'{relative} failed ({rc}); see {log}')
        steps.append(dict(script=relative,seconds=time.time()-t))
    def refresh_manifests():
        # Execution-copy hashes, distinct from the original provenance manifests.
        d=dest/V3;fs=[p for p in d.rglob('*') if p.is_file() and 'vendor' not in p.parts and 'mc_jobs' not in p.parts and p.name not in ['artifact_manifest.json','source_manifest.json']]
        (d/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p) for p in fs},indent=2))
    if args.mode in ['mc','full']:
        for sub in [V3,V4]:
            ref=dest/'reference'/Path(sub).name;ref.mkdir(parents=True)
            for name in ['mc_labels.csv','families.json']:shutil.copyfile(ROOT/sub/'results'/name,ref/name)
        shutil.rmtree(dest/V3/'results');(dest/V3/'results').mkdir()
        shutil.rmtree(dest/V4/'results');(dest/V4/'results').mkdir()
        if args.mode=='full':shutil.rmtree(dest/V3/'models');(dest/V3/'models').mkdir()
        call('scripts/rebuild_source_inputs.py')
        for name in ['prepare','verify_pricer','verify_published','generate_data','verify_labels']:call(V3+'/'+name+'.py')
        if args.mode=='full':call(V3+'/train_models.py')
        else:
            shutil.copyfile(ROOT/V3/'results/model_selection.json',dest/V3/'results/model_selection.json')
        call(V3+'/evaluate.py')
        for name in ['design','verify','run_mc','evaluate_models','diagnostics']:call(V4+'/'+name+'.py')
        call('scripts/compare_reference.py')
    else:
        call('scripts/rebuild_source_inputs.py')
        for name in ['verify_pricer','verify_published','verify_labels']:call(V3+'/'+name+'.py')
        call(V4+'/verify.py')
        call(V4+'/evaluate_models.py')
    refresh_manifests()
    call(AU+'/audit.py');call(AU+'/reproduce.py')
    call('scripts/summarize_results.py')
    assert original=={str(p.relative_to(ROOT)):sha(p) for p in critical},'Frozen source artifacts changed'
    report=dict(frozen_input_hashes=original,status='pass',mode=args.mode,steps=steps,run_dir=str(dest),frozen_source_unchanged=True)
    (dest/'execution_complete.json').write_text(json.dumps(report,indent=2));print('COMPLETE',dest,flush=True)
if __name__=='__main__':main()
