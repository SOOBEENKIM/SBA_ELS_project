"""Four controlled arms, three training seeds; test data never opened here."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import time
import pandas as pd
import torch
from common import *
from models import build,features,forward,predict,load_predictor

TRAINCFG=CFG['learning'];MOD=HERE/'models'

def scores(p,d):
    truth=d['Y'];base=d['base'].astype(int);active=d['axis']!=0
    pe=(p-truth)*10000;de=((p-p[base])-d['D'])*10000
    return dict(price_rmse=float(np.sqrt(np.mean(pe**2))),price_mae=float(np.mean(abs(pe))),
                delta_rmse=float(np.sqrt(np.mean(de[active]**2))),delta_mae=float(np.mean(abs(de[active]))),
                selection_score=float(np.sqrt(np.mean(pe**2))+4*np.sqrt(np.mean(de[active]**2))))

def train_one(arm,seed):
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    started=time.time();path=MOD/f'{arm}_seed{seed}.pt';summarypath=MOD/f'{arm}_seed{seed}.json'
    train=dict(np.load(OUT/'train.npz'));val=dict(np.load(OUT/'validation.npz'))
    spec=dict(protocol=sha(HERE/'protocol.json'),train=sha(OUT/'train.npz'),validation=sha(OUT/'validation.npz'),
              model_code=sha(HERE/'models.py'),training_code=sha(__file__))
    if path.exists() and summarypath.exists():
        result=json.loads(summarypath.read_text());assert result['spec']==spec;return result
    # All baseline/variant family ids are disjoint by the label generator's assertion.
    base=np.flatnonzero(train['axis']==0);counts=np.diff(np.r_[base,len(train['Y'])])
    # Normalization uses only training input rows (even for baseline-only arm):
    # identical input scales across the same-architecture arms; no extra labels.
    xt,stats=features(train,arm);xv,_=features(val,arm,stats)
    ym=float(train['Y'][base].mean());ys=float(train['Y'][base].std())
    yt=torch.tensor((train['Y']-ym)/ys,dtype=torch.float32)
    bt=torch.tensor(base);ct=torch.tensor(counts);net=build(arm,TRAINCFG['latent_dim'])
    opt=torch.optim.Adam(net.parameters(),lr=TRAINCFG['learning_rate'],weight_decay=TRAINCFG['weight_decay'])
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=TRAINCFG['lr_step_size'],gamma=.5)
    best=float('inf');beststate=None;wait=0;log=[]
    for step in range(1,TRAINCFG['max_steps']+1):
        net.train();families=torch.randint(0,len(base),(TRAINCFG['batch_families'],))
        b=bt[families]
        if arm=='base_price':
            pb=forward(net,arm,xt,b);loss=((pb-yt[b])**2).mean();pl=loss;dl=loss.detach()*0
        else:
            variant=b+(torch.rand(len(b))*(ct[families]-1)).long()+1
            both=torch.cat([b,variant]);pr=forward(net,arm,xt,both);pb,pv=pr.chunk(2)
            pl=(((pb-yt[b])**2).mean()+((pv-yt[variant])**2).mean())/2
            dl=(((pv-pb)-(yt[variant]-yt[b]))**2).mean()
            loss=pl+(TRAINCFG['increment_weight']*dl if arm in ['augmented_delta','affine_coupon_delta'] else 0)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();sched.step()
        if step%TRAINCFG['check_every']==0:
            net.eval();p=predict(net,arm,xv,ym,ys);s=scores(p,val)
            log.append(dict(step=step,train_price_loss=float(pl.detach()),train_delta_loss=float(dl.detach()),**s))
            if s['selection_score']<best-1e-4:
                best=s['selection_score'];beststep=step;bestmetrics=s;wait=0
                beststate={k:v.detach().clone() for k,v in net.state_dict().items()}
            else:wait+=1
            if step%500==0:print('TRAIN',arm,seed,'step',step,'val price/delta RMSE',round(s['price_rmse'],2),round(s['delta_rmse'],2),'best',round(best,2),flush=True)
            if wait>=TRAINCFG['patience_checks']:break
    assert beststate is not None;net.load_state_dict(beststate);net.eval()
    ck=dict(arm=arm,seed=seed,P=TRAINCFG['latent_dim'],state=beststate,stats=stats,ym=ym,ys=ys,
            best_step=beststep,best_validation=bestmetrics,spec=spec,input_schema='monthly_explicit_schedule_461_features')
    torch.save(ck,path)
    # Verify the exported model exactly reproduces the restored model predictions.
    loaded,_=load_predictor(path);a=loaded(val);b=predict(net,arm,xv,ym,ys)
    assert np.max(abs(a-b))==0
    result=dict(arm=arm,seed=seed,steps_run=step,best_step=beststep,seconds=time.time()-started,
                validation=bestmetrics,spec=spec,checkpoint=str(path.name),reload_max_error_krw=0.)
    summarypath.write_text(json.dumps(result,indent=2));pd.DataFrame(log).to_csv(MOD/f'{arm}_seed{seed}_history.csv',index=False)
    print('TRAIN COMPLETE',arm,seed,'best step',beststep,'seconds',round(result['seconds'],1),flush=True)
    return result

def main():
    assert not (OUT/'evaluation_complete.json').exists(), 'Test has already been opened; fitting prohibited'
    for name in ['pricer_validation.json','published_payoff_validation.json','label_validation.json']:
        assert json.loads((OUT/name).read_text())['status']=='pass', name
    MOD.mkdir(exist_ok=True);assert (OUT/'labels_complete.json').exists()
    # Existing Monte Carlo labels/checkpoints stay untouched. Only this version is trained.
    rows=[]
    with ProcessPoolExecutor(max_workers=TRAINCFG['workers']) as pool:
        fs=[pool.submit(train_one,arm,seed) for arm in TRAINCFG['arms'] for seed in TRAINCFG['seeds']]
        for f in as_completed(fs):rows.append(f.result())
    flat=pd.DataFrame([dict(arm=r['arm'],seed=r['seed'],best_step=r['best_step'],seconds=r['seconds'],**r['validation']) for r in rows])
    flat.to_csv(OUT/'training_validation.csv',index=False)
    rank=flat.groupby('arm').selection_score.mean().sort_values()
    chosen=str(rank.index[0]);files={p.name:sha(p) for p in sorted(MOD.glob('*.pt'))}
    decision=dict(status='all_models_frozen_before_test',selected_arm=chosen,
        criterion=TRAINCFG['selection_metric'],validation_mean_scores=rank.to_dict(),
        checkpoint_hashes=files,test_data_opened_by_training=False,training_runs=len(rows),
        selected_prediction='Equal-weight ensemble of all three prespecified training seeds')
    (OUT/'model_selection.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
