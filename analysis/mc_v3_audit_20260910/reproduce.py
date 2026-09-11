"""Read-only replay of a reported sign failure; no fitting or label changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import unpack, encode, cases
from models import load_predictor

HERE=Path(__file__).resolve().parent
V3=HERE.parent/'mc_monthly_v3_20260910'
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
FAMILY='published_KR6MD0003TN4'
torch.set_num_threads(1)
r=next(r for r in json.loads((V3/'results/families.json').read_text()) if r['family']==FAMILY)
c,m=unpack(r);cs=cases(c)
data={k:np.stack(a) for k,a in zip(['U','V','C'],zip(*(encode(cc,m) for _,_,cc in cs)))}
labels=pd.read_csv(V3/'results/mc_labels.csv')
labels=labels[labels.family==FAMILY].set_index(['axis','offset'])
saved=pd.read_csv(V3/'results/predictions.csv')
saved=saved[saved.family==FAMILY]
rows=[]; maximum=0.
for arm in ['augmented_price','augmented_delta','affine_coupon_delta']:
    preds=[]
    for seed in [47,101,233]:
        call,_=load_predictor(V3/'models'/f'{arm}_seed{seed}.pt')
        preds.append(call(data))
    for seed,p in zip([47,101,233,'ensemble'],preds+[np.mean(preds,0)]):
        for i,(axis,h,_) in enumerate(cs):
            source=saved[(saved.model==arm)&(saved.seed==str(seed))&(saved.axis==axis)&np.isclose(saved.offset,h)].iloc[0]
            label=labels.loc[(axis,h)]
            error=abs((p[i]-p[0])*10000-source.pred_delta_krw)
            maximum=max(maximum,error)
            rows.append(dict(model=arm,seed=seed,axis=axis,offset=h,
                mc_price_krw=label.price_krw,mc_delta_krw=label.delta_krw,
                mc_halfwidth_krw=1.96*label.delta_se_krw,pred_price_krw=p[i]*10000,
                pred_delta_krw=(p[i]-p[0])*10000,
                baseline_error_krw=p[0]*10000-labels.loc[('base',0.),'price_krw'],
                changed_error_krw=p[i]*10000-label.price_krw,
                replay_delta_difference_krw=error))
df=pd.DataFrame(rows);df.to_csv(OUT/'failure_replay.csv',index=False)
# Different inference batch sizes can change float32 GEMM at sub-cent level.
assert maximum<.05, maximum
case=df[(df.axis=='first_strike')&np.isclose(df.offset,.05)]
print(case.to_string(index=False))
result=dict(status='reproduced',family=FAMILY,axis='first_strike',offset=.05,
            max_batch_replay_difference_krw=maximum,models_unchanged=True,training_performed=False)
(OUT/'reproduction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
