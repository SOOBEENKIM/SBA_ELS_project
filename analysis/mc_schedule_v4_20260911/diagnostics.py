import pandas as pd
import numpy as np
from design import *
from evaluate_models import metrics

p=pd.read_csv(OUT/'predictions.csv');p=p[p.training_seed.eq('ensemble')&p.model.eq(CFG['models']['primary'])]
out=[]
for key,g in p[p.axis.str.startswith('nobs')].groupby(['cohort','axis','nobs']):
    out.append(dict(zip(['cohort','axis','new_nobs'],key),base_nobs=int(key[-1]-1),mc_delta_mean=float(g.mc_delta_krw.mean()),pred_delta_mean=float(g.pred_delta_krw.mean()),**metrics(g)))
pd.DataFrame(out).to_csv(OUT/'nobs_by_initial_count.csv',index=False)
mc=pd.read_csv(OUT/'mc_labels.csv');meta=pd.read_csv(OUT/'scenario_index.csv');base=mc[mc.axis.eq('base')].set_index('family')
z=mc[mc.axis.ne('base')].merge(meta[['family','case','cohort']],on=['family','case'])
z['monthly_delta_krw']=[r.monthly_pv_krw-base.loc[r.family,'monthly_pv_krw'] for r in z.itertuples()]
z['nonmonthly_delta_krw']=[r.nonmonthly_pv_krw-base.loc[r.family,'nonmonthly_pv_krw'] for r in z.itertuples()]
assert np.allclose(z.monthly_delta_krw+z.nonmonthly_delta_krw,z.delta_krw,atol=1e-8)
z.groupby(['cohort','axis','offset'])[['monthly_delta_krw','nonmonthly_delta_krw','delta_krw']].mean().reset_index().to_csv(OUT/'cashflow_decomposition.csv',index=False)
train=[r for r in json.loads((V3/'results/families.json').read_text()) if r['split']=='train']
from collections import Counter
rows=[]
for monthly in [False,True]:
    tr=[r['contract'] for r in train if r['monthly']==monthly]
    rows.append(dict(monthly=monthly,n=len(tr),nobs_counts=dict(Counter(len(c['obs_days']) for c in tr)),
        monthly_counts=dict(Counter(len(c['monthly_obs_days']) for c in tr)),maturity_year_rounded_counts=dict(Counter(round(c['obs_days'][-1]/365,1) for c in tr))))
(OUT/'training_coverage.json').write_text(json.dumps(rows,indent=2))
print(pd.DataFrame(out).to_string(index=False))
print(json.dumps(rows,indent=2))
