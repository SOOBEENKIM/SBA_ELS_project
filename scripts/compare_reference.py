"""Compare newly simulated MC labels and frozen predictions with bundled results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];out={}
for sub,key in [('mc_monthly_v3_20260910',['family','axis','offset']),('mc_schedule_v4_20260911',['family','case'])]:
 a=pd.read_csv(R/'reference'/sub/'mc_labels.csv').set_index(key).sort_index()
 b=pd.read_csv(R/'analysis'/sub/'results/mc_labels.csv').set_index(key).sort_index()
 assert a.index.equals(b.index)
 cols=['price_krw','delta_krw','price_se_krw','delta_se_krw'];err=float(np.max(abs(a[cols].to_numpy()-b[cols].to_numpy())))
 assert err<1e-7,(sub,err)
 fa=json.loads((R/'reference'/sub/'families.json').read_text());fb=json.loads((R/'analysis'/sub/'results/families.json').read_text());assert fa==fb,sub
 out[sub]=dict(scenarios=len(a),families_identical=True,max_mc_difference_krw=err)
(R/'verification').mkdir(exist_ok=True);(R/'verification/mc_reproduction.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
