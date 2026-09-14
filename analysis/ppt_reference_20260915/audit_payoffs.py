"""Independent sequential-cashflow checks across all sampled product structures."""
import json,gzip
from pathlib import Path
import torch
from paired_engine import shared_values,unpack

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)
with gzip.open(HERE/'results/synthetic_families.json.gz','rt') as f:families=json.load(f)
seen=set();checks=[]
for family in families:
    key=(family['structure'],family['monthly'])
    if key in seen:continue
    seen.add(key)
    contracts=[unpack({'contract':v['contract'],'market':family['market']})[0] for v in family['variants']]
    for _ in shared_values(contracts,family['pricing_market'],2000,20260915,audit=True):pass
    checks.append(dict(structure=key[0],monthly=key[1],family=family['family'],variants=len(contracts),paths=2000))
assert seen=={(f['structure'],f['monthly']) for f in families}
result=dict(status='pass',checks=checks,method='Sequential NumPy ledger vs vectorized source and detailed payoffs on identical paths; verification-only, not training labels')
(HERE/'results/all_structure_ledger_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
