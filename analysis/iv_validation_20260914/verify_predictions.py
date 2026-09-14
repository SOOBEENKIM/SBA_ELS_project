"""Check serialized prediction rows against independently encoded cases."""
import json
import numpy as np
import pandas as pd
import torch
from prepare import ROOT,HERE,V3,OUT,dump,preserved
from common import unpack,encode
from models import load_predictor

def main():
    torch.set_num_threads(2)
    cfg=json.loads((HERE/'protocol.json').read_text())
    families={r['family']:r for r in json.loads((OUT/'families.json').read_text())}
    p=pd.read_csv(OUT/'predictions.csv.gz',dtype={'model_seed':str})
    max_price=0.;max_delta=0.;checked=0
    for arm in cfg['frozen_models']['arms']:
        for seed in cfg['frozen_models']['seeds']:
            call,_=load_predictor(V3/'models'/f'{arm}_seed{seed}.pt')
            g=p[p.model.eq(arm)&p.model_seed.eq(str(seed))]
            sampled=g.groupby(['arm','monthly','suite'],group_keys=False).sample(n=2,random_state=914)
            for r in sampled.itertuples():
                f=families[r.family];a=[]
                for j in [0,r.case]:
                    c,m=unpack(dict(contract=f['variants'][j]['contract'],market=f['markets'][r.arm]));a.append(encode(c,m))
                data=dict(zip(['U','V','C'],[np.stack([x[k] for x in a]) for k in range(3)]))
                out=call(data)*10000
                ep=abs(out[1]-r.pred_price_krw);ed=abs((out[1]-out[0])-r.pred_delta_krw)
                max_price=max(max_price,ep);max_delta=max(max_delta,ed);checked+=1
                assert ep<.01 and ed<.01,(arm,seed,r.family,r.case,ep,ed)
    assert preserved()==cfg['preserved_inputs']
    result=dict(status='pass',independently_encoded_prediction_pairs=checked,
                max_price_difference_krw=max_price,max_delta_difference_krw=max_delta,
                tolerance_krw=.01,note='Two-row vs large-batch float32 inference; frozen checkpoint and input alignment check, not a new prediction benchmark.')
    dump(OUT/'prediction_alignment_verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
