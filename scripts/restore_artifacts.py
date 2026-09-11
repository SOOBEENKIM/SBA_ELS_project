"""Restore bundled data/results losslessly; stdlib only, no network or external paths."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inside(relative):
 p=(ROOT/relative).resolve()
 if ROOT not in p.parents:raise ValueError('Archive path outside repository')
 return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
 spec=json.loads((ROOT/'provenance/bundles.json').read_text());restored=0
 for a in spec['archives']:
  src=inside(a['archive']);assert sha(src)==a['sha256'],f'Archive checksum mismatch: {src}'
  z=zipfile.ZipFile(src) if a['format']=='zip' else None
  try:
   for entry in a['files']:
    dest=inside(entry['path'])
    if dest.exists():
     assert sha(dest)==entry['sha256'],f'Existing file differs; preserving it: {dest}'
     continue
    if args.verify_only:raise FileNotFoundError(f'Run restore first: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.restoring')
    try:
     stream=z.open(entry['member']) if z else gzip.open(src,'rb')
     with stream as r,temp.open('xb') as w:shutil.copyfileobj(r,w)
     assert sha(temp)==entry['sha256'],f'Restored checksum mismatch: {dest}'
     os.replace(temp,dest);restored+=1
    finally:
     if temp.exists():temp.unlink()
  finally:
   if z:z.close()
 print(f'PASS: {len(spec["archives"])} archives checked, {restored} files restored; original SHA-256 verified.')
if __name__=='__main__':main()
