"""Reuse the three prespecified learning arms, fit separately to each payoff target."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];V3=ROOT/'analysis/mc_monthly_v3_20260910'
sys.path[:0]=[str(ROOT),str(V3),str(HERE)]
import train_models as T
from models import load_predictor
from evaluate import metric

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit_one(payoff,arm,seed):
    print('FIT START',payoff,arm,seed,flush=True)
    base=HERE/'synthetic'/payoff;T.HERE=base;T.OUT=base/'results';T.MOD=base/'models'
    T.CFG=json.loads((base/'protocol.json').read_text());T.TRAINCFG=T.CFG['learning']
    return T.train_one(arm,seed)

def train(workers):
    for payoff in ['reference','detailed']:
        b=HERE/'synthetic'/payoff
        assert json.loads((b/'results/label_validation.json').read_text())['status']=='pass'
        assert not (b/'results/evaluation_complete.json').exists(),'Do not train after test evaluation'
    cfg=json.loads((V3/'protocol.json').read_text())['learning'];rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fit_one,p,a,s):p for p in ['reference','detailed'] for a in cfg['arms'] for s in cfg['seeds']}
        for f in as_completed(futures):rows.append(dict(payoff=futures[f],**f.result()))
    for payoff in ['reference','detailed']:
        b=HERE/'synthetic'/payoff;rr=[r for r in rows if r['payoff']==payoff]
        flat=pd.DataFrame([dict(arm=r['arm'],seed=r['seed'],best_step=r['best_step'],seconds=r['seconds'],**r['validation']) for r in rr])
        flat.to_csv(b/'results/training_validation.csv',index=False)
        rank=flat.groupby('arm').selection_score.mean().sort_values()
        decision=dict(status='all_models_frozen_before_test',selected_arm=str(rank.index[0]),validation_mean_scores=rank.to_dict(),
          checkpoint_hashes={p.name:sha(p) for p in sorted((b/'models').glob('*.pt'))},test_data_opened_by_training=False,
          selection_criterion='validation price RMSE + 4 * delta RMSE; fixed before test',training_runs=len(rr),wrapper_sha256=sha(__file__))
        (b/'results/model_selection.json').write_text(json.dumps(decision,indent=2));print(payoff,json.dumps(decision,indent=2),flush=True)

def evaluate():
    torch.set_num_threads(2);allmetrics=[];allpred=[]
    for payoff in ['reference','detailed']:
        b=HERE/'synthetic'/payoff;out=b/'results';cfg=json.loads((b/'protocol.json').read_text())['learning']
        selection=json.loads((out/'model_selection.json').read_text());assert selection['test_data_opened_by_training'] is False
        for name,h in selection['checkpoint_hashes'].items():assert sha(b/'models'/name)==h
        preds=[]
        for split in ['test','stress']:
            d=dict(np.load(out/f'{split}.npz'));meta=pd.read_csv(out/f'{split}_rows.csv');base=d['base'].astype(int)
            meta['cohort']=np.where(meta.origin.eq('published_monthly'),'published_monthly_6',np.where(meta.split.eq('test'),np.where(meta.monthly,'monthly_test','regular_test'),'other_diagnostic'))
            def record(label,seed,p):
                g=meta.copy();g['payoff']=payoff;g['model']=label;g['seed']=str(seed)
                g['mc_price_krw']=d['Y']*10000;g['mc_delta_krw']=d['D']*10000;g['mc_delta_se_krw']=d['SE']*10000
                g['pred_price_krw']=p*10000;g['pred_delta_krw']=(p-p[base])*10000
                g['price_error_krw']=g.pred_price_krw-g.mc_price_krw;g['delta_error_krw']=g.pred_delta_krw-g.mc_delta_krw;preds.append(g)
            for arm in cfg['arms']:
                ps=[]
                for seed in cfg['seeds']:
                    call,_=load_predictor(b/'models'/f'{arm}_seed{seed}.pt');p=call(d);ps.append(p);record(arm,seed,p)
                record(arm,'ensemble',np.mean(ps,axis=0))
            # Compare the prior models on the very same new prices and inputs.
            prior=[]
            for seed in cfg['seeds']:
                call,_=load_predictor(V3/'models'/f'affine_coupon_delta_seed{seed}.pt');prior.append(call(d))
            record('prior_frozen_affine_coupon_delta','ensemble',np.mean(prior,axis=0))
        predictions=pd.concat(preds,ignore_index=True);predictions.to_csv(out/'predictions.csv.gz',index=False);allpred.append(predictions)
        rows=[];offsets=[]
        for key,g in predictions.groupby(['payoff','cohort','model','seed','scope']):
            keys=dict(zip(['payoff','cohort','model','seed','scope'],key))
            rows.append(dict(**keys,axis='ALL',**metric(g)))
            for axis,h in g.groupby('axis'):rows.append(dict(**keys,axis=axis,**metric(h)))
            for (axis,offset),h in g[g.axis!='base'].groupby(['axis','offset']):offsets.append(dict(**keys,axis=axis,offset=offset,**metric(h)))
        pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False);pd.DataFrame(offsets).to_csv(out/'metrics_by_offset.csv',index=False);allmetrics+=rows
        (out/'evaluation_complete.json').write_text(json.dumps(dict(status='complete',prediction_rows=len(predictions),selected_arm=selection['selected_arm'],test_used_for_selection=False),indent=2))
    pd.DataFrame(allmetrics).to_csv(HERE/'results/model_metrics.csv',index=False)
    print(pd.DataFrame(allmetrics).query("seed=='ensemble' and axis=='ALL' and cohort in ['regular_test','monthly_test']").to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate']);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    (lambda:train(a.workers))() if a.command=='train' else evaluate()
