"""Import immutable, selected archive members; preserve existing project code/data."""
from pathlib import Path
import zipfile,hashlib,json,shutil,argparse
p=argparse.ArgumentParser();p.add_argument('archives',type=Path);p.add_argument('repo',type=Path);args=p.parse_args()
base=args.repo/'analysis/ppt_reproduction_20260917'; ref=base/'reference';ref.mkdir(parents=True,exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
archives={}; contents={}
for name in ('main','feat-4-structure-universe'):
 f=args.archives/('PI-DeepONet-ELS-'+name+'.zip');z=zipfile.ZipFile(f)
 contents[name]={i.filename.split('/',1)[1]:z.read(i) for i in z.infolist() if not i.is_dir()}
 archives[name]={'name':f.name,'sha256':sha(f.read_bytes()),'files':{k:{'size':len(v),'sha256':sha(v)} for k,v in contents[name].items()}}
a,b=contents['main'],contents['feat-4-structure-universe'];changes={s:sorted(k for k in set(a)|set(b) if (s=='added' and k not in a) or (s=='removed' and k not in b) or (s=='changed' and k in a and k in b and a[k]!=b[k])) for s in ('added','removed','changed')}
kept=[]
for name,raw in b.items():
 path=Path(name)
 if path.is_absolute() or '..' in path.parts: raise ValueError(name)
 keep=(name.startswith(('module/','model/','util/')) or name in ('config.yaml','requirements.txt','README.md') or name.startswith(tuple(str(n)+'_' for n in [5,15,16,17,18,19])) or name.startswith('data/cache/mcvar/') or (name.startswith('data/cache/') and path.name.startswith(('div_','px_','irs_','krw_curve','udly_ticker','_u3map','_strk_by','rates_probe'))))
 if not keep:continue
 dest=ref/path;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==raw,name
 else:dest.write_bytes(raw)
 kept.append({'path':name,'sha256':sha(raw),'bytes':len(raw)})
source=args.repo/'analysis/ppt_reference_20260915/reference/population_58790.parquet'
dest=ref/'data/els3_dataset.parquet'
if dest.exists():assert dest.read_bytes()==source.read_bytes()
else:shutil.copyfile(source,dest)
manifest={'archives':archives,'branch_comparison':changes,'selected':kept,'source_dataset':{'path':str(source.relative_to(args.repo)),'sha256':sha(source.read_bytes()),'reason':'Existing full 58,790-product snapshot; latest ZIP excludes this derived dataset. Must verify IDs against latest mc_variants.'},'missing_latest_execution_scripts':['mc_discrete_div.py','mc_boot_curve.py','mc_full_variants.py','stage1_mc_variants.py','mc_noise_check.py'],'note':'No original source files patched. New execution adapters live outside reference/. Cached MC outputs are not a fresh MC run.'}
(base/'import_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'selected':len(kept),'bytes':sum(x['bytes'] for x in kept),'changes':{k:len(v) for k,v in changes.items()}},indent=2))
