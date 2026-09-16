"""Held-out B-market contract sensitivity checks, 40000 paths x three seeds.
These probes diagnose fixed trained models; no probe prices enter training.
Four axes retain all observation/payment dates and other terms.
"""
from pathlib import Path
import sys,json,gzip,hashlib
from dataclasses import replace
import numpy as np,pandas as pd,torch
from contracts import ContractV3
from engine import shared_paths,explicit_values,legacy
from module.mc_contract_v3 import bump_v3
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';BASE=HERE.parent/'ppt_full_reproduction_20260917/results'
sys.path.insert(0,str(BASE.parent));from market_rules import curves,dividends
SEEDS=(100101,100233,100509)
def design():
 s=pd.read_parquet(BASE/'rebuilt_universe.parquet');meta=pd.read_csv(OUT/'contract_audit.csv.gz');meta=meta[meta.market_valid & (meta.i>=int(.6*len(s)))]
 picks=[]
 for key,g in meta.groupby(['structure','monthly']):
  n=min(16,len(g));picks+=g.sort_values('i').iloc[np.linspace(0,len(g)-1,n).astype(int)].i.tolist()
 records={r['i']:r for r in json.loads(gzip.decompress((OUT/'contracts.json.gz').read_bytes()))};families=[]
 for i in sorted(picks):
  c=ContractV3(**records[i]['contract']);cases=[('base',0.,c)]
  for axis in ('coupon_regular','ki_barrier','first_strike','last_strike'):
   if axis=='ki_barrier' and not c.has_ki:continue
   steps=[.001,.0025,.005,.01] if axis=='coupon_regular' else [.005,.01,.025,.05]
   for h in [-x for x in reversed(steps)]+steps:
    try:cc=bump_v3(c,axis,h)
    except ValueError:continue
    cases.append((axis,h,cc))
  families.append(dict(i=i,item=s.iloc[i]['item'],monthly=c.has_monthly,structure=c.structure,cases=[dict(axis=ax,h=h,contract=cc.serializable()) for ax,h,cc in cases]))
 (OUT/'probe_design.json').write_text(json.dumps(dict(seeds=SEEDS,paths_per_seed=40000,market_variant='B',selection='At most 16 evenly spaced contracts per structure/monthly stratum among supported original walk-forward OOS rows; selected without inspecting model errors.',intervention='coupon changes regular/survival/monthly rate proportionally; Lizard bonus stays fixed; KI and one strike change alone. Observation and payment dates are fixed.',families=families),indent=2)+'\n')
 return families
@torch.no_grad()
def main():
 torch.set_default_device('cuda:0');torch.set_num_threads(2)
 families=design();s=pd.read_parquet(BASE/'rebuilt_universe.parquet');v=pd.read_parquet(BASE/'fresh_market_inputs.parquet');events=json.loads(gzip.decompress((BASE/'dividend_events.json.gz').read_bytes()))['events']
 files=[HERE/'probe_mc.py',HERE/'engine.py',OUT/'probe_design.json',OUT/'protocol.json',BASE/'fresh_market_inputs.parquet']
 protocol={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};pp=OUT/'probe_protocol.json'
 if pp.exists():assert json.loads(pp.read_text())==protocol,'Probe inputs changed; use a new output directory'
 else:pp.write_text(json.dumps(protocol,indent=2)+'\n')
 dest=OUT/'probe_seed_results.jsonl';prior={(r['item'],r['seed']) for r in (json.loads(line) for line in dest.read_text().splitlines())} if dest.exists() else set()
 with dest.open('a',buffering=1) as f:
  for k,fam in enumerate(families):
   i=fam['i'];r=s.iloc[i];m=v.iloc[i];cs=[ContractV3(**cc['contract']) for cc in fam['cases']];c=cs[0];raw=legacy.contract(r);_,bt,_=curves(int(r.isu_ord));rho=[m.rho_12,m.rho_13,m.rho_23];C=np.array([[1,rho[0],rho[1]],[rho[0],1,rho[2]],[rho[1],rho[2],1]])
   ev=events[i];N=max(round(r.tenor*365),c.obs_days[-1])
   if N>round(r.tenor*365):
    drops,_,_=dividends((r.udl1,r.udl2,r.udl3),int(r.isu_ord),N/365.);di,dj=np.nonzero(drops);ev=[(int(a),int(b),float(drops[a,b])) for a,b in zip(di,dj)]
   for seed in SEEDS:
    if (fam['item'],seed) in prior:continue
    total=np.zeros((len(cs),5))
    for _,details in shared_paths([m[f'sB_{j}'] for j in (1,2,3)],C,bt,raw,c,ev,seed+int(r.mc_seed),n=40000):
     logs,running,union,df,_=details;values=torch.stack([explicit_values(logs,running,union,cc,df) for cc in cs]);d=values-values[0]
     total+=torch.stack([values.sum(1),values.square().sum(1),d.sum(1),d.square().sum(1),(d.abs()>1e-12).sum(1)],1).cpu().numpy()
    f.write(json.dumps(dict(item=fam['item'],i=i,seed=seed,paths=40000,moments=total.tolist()),separators=(',',':'))+'\n')
   print('PROBE',k+1,'/',len(families),flush=True)
 rows=[]
 records=[json.loads(line) for line in dest.read_text().splitlines()]
 for fam in families:
  matched=[r for r in records if r['item']==fam['item']];assert len(matched)==3;total=sum(np.asarray(r['moments']) for r in matched);n=120000
  for j,cc in enumerate(fam['cases']):
   sm,ss,dm,ds,affected=total[j];mean=sm/n;delta=dm/n;se=np.sqrt(max(0,(ds-n*delta*delta)/(n-1))/n)
   seed_delta=[r['moments'][j][2]/40000*10000 for r in matched]
   rows.append(dict(item=fam['item'],i=fam['i'],case=j,monthly=fam['monthly'],structure=fam['structure'],axis=cc['axis'],h=cc['h'],mc=mean,delta=delta*10000,delta_se=se*10000,affected=int(affected),seed_min=min(seed_delta),seed_max=max(seed_delta)))
 pd.DataFrame(rows).to_csv(OUT/'probe_mc.csv',index=False)
 print('COMPLETE',len(families),'families',len(rows),'cases',flush=True)
if __name__=='__main__':main()
