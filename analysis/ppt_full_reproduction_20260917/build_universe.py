"""Rebuild source contract universe from raw DART and frozen market caches."""
from pathlib import Path
import sys,gzip,shutil,json,hashlib,time
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REF=HERE.parent/'ppt_reproduction_20260917/reference';OUT=HERE/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REF))
from util import file_manager as fm
from module.build_source import build_source
from module.schedule import SCHED_COLS

def main():
 raw=HERE/'working_raw';raw.mkdir(exist_ok=True);manifest=[]
 for src in sorted((ROOT/'data/cache/raw').glob('LAKE_V2_DART_*.csv.gz')):
  dst=raw/src.name[:-3]
  with gzip.open(src,'rb') as r,dst.open('wb') as w:shutil.copyfileobj(r,w)
  manifest.append(dict(path=str(src.relative_to(ROOT)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
 assert len(manifest)==3
 fm.RAW=raw;fm.CACHE=REF/'data/cache'
 start=time.monotonic();d=build_source(save=False,verbose=True)
 ref=pd.read_parquet(REF/'data/els3_dataset.parquet').sort_values('isu_ord').reset_index(drop=True)
 assert d.item.is_unique and set(d.item)==set(ref.item),(len(d),len(ref),list(set(d.item)^set(ref.item))[:10])
 # Stable original ID order fixes same-day tie order and original per-product seeds.
 d=d.set_index('item').loc[ref.item].reset_index();d['mc_seed']=ref.mc_seed.to_numpy(int)
 num=['isu_ord','B','K','Kfirst','coupon','tenor','nobs','fair','pmt_linear']+SCHED_COLS+[f'u{j}' for j in range(10)]
 differences={c:float(np.nanmax(abs(d[c].to_numpy(float)-ref[c].to_numpy(float)))) if np.isfinite(d[c].to_numpy(float)).any() else 0. for c in num}
 for c in num:np.testing.assert_allclose(d[c],ref[c],atol=1e-10,rtol=0,equal_nan=True,err_msg=c)
 for c in ['udl1','udl2','udl3','opt_type','ki_yn']:np.testing.assert_array_equal(d[c],ref[c])
 d.to_parquet(OUT/'rebuilt_universe.parquet',index=False)
 report=dict(status='pass',rows=len(d),seconds=time.monotonic()-start,raw_inputs=manifest,price_labels_loaded_into_rebuilt_data=False,cached_source_used_for='comparison, same-day tie ordering and seed mapping only; no cached MC values',max_abs_difference= differences,structures=d.groupby(['opt_type','ki_yn']).size().to_dict())
 report['structures']={str(k):v for k,v in report['structures'].items()}
 (OUT/'universe_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('UNIVERSE PASS',len(d),max(differences.values()),flush=True)
if __name__=='__main__':main()
