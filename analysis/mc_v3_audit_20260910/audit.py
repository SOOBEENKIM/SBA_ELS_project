"""Independent checks of frozen V3 data, inference and MC sign failures.

This is a diagnostic, not a new test set for model selection. No fitting occurs.
The event-time NumPy simulator applies ONLY to STEP no-KI contracts; its GBM
joint distribution is exact at the contractual observation dates and it does
not make claims about continuous/path-dependent KI or Lizard monitoring.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from common import unpack,encode,cases
from models import load_predictor,features
from module.mc_contract_v3 import summaries_v3,payoff_v3,bump_v3

HERE=Path(__file__).resolve().parent; V3=HERE.parent/'mc_monthly_v3_20260910'
PROJECT=HERE.parents[1]; OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
FS=json.loads((V3/'results/families.json').read_text())
LABELS=pd.read_csv(V3/'results/mc_labels.csv').set_index(['family','axis','offset'])
RESULT=dict(training_performed=False,frozen_outputs_modified=False)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def integrity():
    checked=[]
    for manifest,root in [('artifact_manifest.json',V3),('source_manifest.json',PROJECT)]:
        for name,value in json.loads((V3/manifest).read_text()).items():
            actual=sha(root/name);assert actual==value,(name,actual,value);checked.append(str((root/name).relative_to(PROJECT)))
    RESULT['integrity']=dict(status='pass',files=len(checked),original_and_new_pricers_preserved=True)
    print('INTEGRITY PASS',len(checked),flush=True)

def inputs():
    count=0;max_label=0.;fields={'coupon_regular':'coupon','ki_barrier':'ki_barrier',
          'first_strike':'strikes','last_strike':'strikes','monthly_barrier':'monthly_barriers'}
    for split in ['train','validation','test','stress']:
        d=dict(np.load(V3/f'results/{split}.npz'));meta=pd.read_csv(V3/f'results/{split}_rows.csv')
        row=0
        for r in [r for r in FS if r['split']==split]:
            c,m=unpack(r);baseline=row;original=asdict(c)
            for axis,h,cc in cases(c):
                ref=meta.iloc[row]
                assert ref.family==r['family'] and ref.axis==axis and abs(ref.offset-h)<1e-12
                assert int(d['base'][row])==baseline
                changed=[k for k,v in asdict(cc).items() if v!=original[k]]
                assert changed==([] if axis=='base' else [fields[axis]]),(r['family'],axis,changed)
                u,v,x=encode(cc,m)
                for key,a in zip(['U','V','C'],[u,v,x]):
                    assert np.array_equal(a,d[key][row]),(split,row,key)
                if axis!='base':
                    assert np.array_equal(d['U'][row],d['U'][baseline])
                    assert np.array_equal(d['V'][row],d['V'][baseline])
                if axis=='first_strike':
                    assert np.flatnonzero(x!=d['C'][baseline]).tolist()==[0]
                if axis=='last_strike':
                    assert np.flatnonzero(x!=d['C'][baseline]).tolist()==[c.nobs-1]
                label=LABELS.loc[(r['family'],axis,h)]
                for key,col in [('Y','price_krw'),('D','delta_krw'),('SE','delta_se_krw')]:
                    err=abs(d[key][row]*10000-label[col]);max_label=max(max_label,err);assert err<1e-8
                row+=1
        assert row==len(meta)==len(d['Y']);count+=row
        assert np.max(abs(d['Y']-d['Y'][d['base']]-d['D']))*10000<1e-7
        print('INPUT PASS',split,row,flush=True)
    train=dict(np.load(V3/'results/train.npz'))
    for arm in ['augmented_price','augmented_delta','affine_coupon_delta']:
        _,stats=features(train,arm)
        for seed in [47,101,233]:
            _,ck=load_predictor(V3/'models'/f'{arm}_seed{seed}.pt')
            for k,value in stats.items():assert np.array_equal(ck['stats'][k],value),(arm,seed,k)
            base=train['axis']==0
            assert ck['ym']==float(train['Y'][base].mean()) and ck['ys']==float(train['Y'][base].std())
    RESULT['input_and_normalization']=dict(status='pass',scenario_rows=count,
        one_contract_field_only=True,all_cached_inputs_exact=True,first_strike_changes_feature_0_only=True,
        baseline_pairing_correct=True,labels_max_error_krw=max_label,
        nine_checkpoint_normalizers_use_training_only=True)

def discount(beta,days):
    # Independent direct Nelson-Siegel expression (same documented lambda=1.5).
    t=np.asarray(days,dtype=float)/365.;x=t/1.5
    slope=-np.expm1(-x)/x
    rate=(beta[0]+beta[1]*slope+beta[2]*(slope-np.exp(-x)))/100.
    return np.exp(-rate*t)

def ledger(c,m,wobs,wrun,minimum,monthly):
    # Independently settle alive contracts sequentially, then sum dated coupons.
    n=len(wobs);alive=np.ones(n,bool);end=np.full(n,c.obs_days[-1]);pv=np.zeros(n)
    trigger=np.zeros(n,bool);principal=np.zeros(n)
    for j,(day,pay,strike,lizard) in enumerate(zip(c.obs_days,c.pay_days,c.strikes,c.lz_barr)):
        regular=alive&(wobs[:,j]>=strike)
        liz=alive&~regular&(wrun[:,j]>=lizard) if lizard is not None else np.zeros(n,bool)
        hit=regular|liz
        cash=np.where(regular,1+c.coupon*c.coupon_accruals[j],1+c.lz_pmts[j])
        principal[hit]=cash[hit]*discount(m.beta,[pay])[0]
        end[hit]=day;trigger[hit]=True;alive[hit]=False
    terminal=wobs[:,-1].copy()
    if c.has_ki:terminal[minimum>=c.ki_barrier]=1+c.coupon*c.survival_coupon_accrual
    principal[alive]=terminal[alive]*discount(m.beta,[c.pay_days[-1]])[0]
    pv+=principal
    for j,(day,pay,bar,accr) in enumerate(zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_accruals)):
        eligible=(day<=end)&(monthly[:,j]>=bar)
        if not c.coupon_on_redemption_day:eligible&=~(trigger&(day==end))
        pv[eligible]+=c.coupon*accr*discount(m.beta,[pay])[0]
    return pv,principal,pv-principal

def same_path_ledger():
    rows=[]
    for ri,r in enumerate([r for r in FS if r['origin']=='published_monthly']):
        c,m=unpack(r); cs=cases(c);maximum=0.;count=0
        for paths,p in summaries_v3(c,m,40000,74000+ri):
            w=paths['wobs'].double().exp().numpy();run=paths['wrun'].double().exp().numpy()
            minimum=paths['minimum'].double().exp().numpy();monthly=paths['monthly'].double().exp().numpy()
            for _,_,cc in cs:
                expected=ledger(cc,m,w,run,minimum,monthly)[0]
                actual=payoff_v3(paths,p,cc)[0].numpy()
                err=np.max(abs(expected-actual))*10000;maximum=max(maximum,err);assert err<1e-7,(r['family'],err)
                count+=len(w)
        rows.append(dict(family=r['family'],paths=40000,variants=len(cs),path_payoffs=count,max_error_krw=maximum))
        print('LEDGER PASS',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'independent_ledger.csv',index=False)
    RESULT['independent_cashflow_ledger']=dict(status='pass',contracts=len(rows),
        market_paths=sum(r['paths'] for r in rows),path_variant_payoffs=sum(r['path_payoffs'] for r in rows),
        max_error_krw=max(r['max_error_krw'] for r in rows))

def independent_simulator():
    rows=[];pooled=[]
    # One negative and one positive first-strike increment: no target sign imposed.
    for family in ['published_KR6MD0003TN4','published_KR6523325910']:
        r=next(r for r in FS if r['family']==family);c,m=unpack(r)
        assert c.structure=='STEP no-KI'
        times=np.array(sorted(set(c.obs_days)|set(c.monthly_obs_days)))
        t=times/365.;dt=np.diff(np.r_[0.,t]);disc=discount(m.beta,times)
        growth=-np.log(disc);rate_integrals=np.diff(np.r_[0.,growth])
        ridx=np.searchsorted(times,c.obs_days);midx=np.searchsorted(times,c.monthly_obs_days)
        sig=np.array(m.sigs);q=np.array(m.dividend_yields);chol=np.linalg.cholesky(m.corr)
        changed=bump_v3(c,'first_strike',.05);all_y=[];all_d=[];parts=[]
        for seed in [91801,91802,91803]:
            rng=np.random.default_rng(seed)
            z=rng.normal(size=(40000,len(times),3))@chol.T
            increment=z*np.sqrt(dt)[None,:,None]*sig[None,None,:]
            increment+=rate_integrals[None,:,None]-(q+.5*sig**2)[None,None,:]*dt[None,:,None]
            worst=np.exp(np.cumsum(increment,axis=1)).min(axis=2)
            running=np.minimum.accumulate(np.c_[np.ones(len(worst)),worst],axis=1)[:,1:]
            y,p,cp=ledger(c,m,worst[:,ridx],running[:,ridx],running[:,-1],worst[:,midx])
            y1,p1,cp1=ledger(changed,m,worst[:,ridx],running[:,ridx],running[:,-1],worst[:,midx])
            d=(y1-y)*10000;all_y.append(y*10000);all_d.append(d)
            parts.append([(p1-p).mean()*10000,(cp1-cp).mean()*10000])
            rows.append(dict(family=family,seed=seed,paths=40000,base_price_krw=y.mean()*10000,
                delta_krw=d.mean(),delta_se_krw=d.std(ddof=1)/np.sqrt(len(d))))
        d=np.concatenate(all_d);y=np.concatenate(all_y);old=LABELS.loc[(family,'first_strike',.05)]
        mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)))
        zscore=float((mean-old.delta_krw)/np.sqrt(se**2+old.delta_se_krw**2))
        result=dict(family=family,paths=120000,base_price_krw=float(y.mean()),delta_krw=mean,
            delta_se_krw=se,halfwidth_95_krw=1.96*se,
            original_delta_krw=float(old.delta_krw),original_halfwidth_95_krw=1.96*float(old.delta_se_krw),
            independent_difference_z=zscore,
            principal_delta_krw=float(np.mean(parts,axis=0)[0]),monthly_coupon_delta_krw=float(np.mean(parts,axis=0)[1]))
        pooled.append(result);print('INDEPENDENT NUMPY',result,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'independent_mc_seeds.csv',index=False)
    pd.DataFrame(pooled).to_csv(OUT/'independent_mc.csv',index=False)
    RESULT['independent_mc']=dict(status='calculated',scope='Two STEP no-KI contracts; independent float64 GBM at event dates',rows=pooled)

def inference_and_metrics():
    saved=pd.read_csv(V3/'results/predictions.csv');max_error=0.;checked=0
    for split in ['test','stress']:
        d=dict(np.load(V3/f'results/{split}.npz'))
        for arm in ['augmented_price','augmented_delta','affine_coupon_delta']:
            ps=[]
            for seed in [47,101,233]:
                call,_=load_predictor(V3/'models'/f'{arm}_seed{seed}.pt');ps.append(call(d))
            for seed,p in zip([47,101,233,'ensemble'],ps+[np.mean(ps,axis=0)]):
                g=saved[(saved.split==split)&(saved.model==arm)&(saved.seed==str(seed))].sort_values('row')
                assert len(g)==len(p)
                delta=(p-p[d['base']])*10000
                err=max(np.max(abs(p*10000-g.pred_price_krw)),np.max(abs(delta-g.pred_delta_krw)))
                max_error=max(max_error,err);checked+=len(p);assert err<.05
    # Recompute all aggregate errors straight from exported prices and paired base rows.
    metric=pd.read_csv(V3/'results/model_metrics.csv');count=0
    for key,g in saved.groupby(['cohort','model','seed']):
        ref=metric[(metric.cohort==key[0])&(metric.model==key[1])&(metric.seed==key[2])&(metric.axis=='ALL')].iloc[0]
        mae=abs(g.pred_price_krw-g.mc_price_krw).mean()
        h=g[g.axis!='base'];d_mae=abs(h.pred_delta_krw-h.mc_delta_krw).mean()
        assert abs(mae-ref.price_mae)<1e-8 and abs(d_mae-ref.delta_mae)<1e-8;count+=1
    RESULT['inference_and_aggregation']=dict(status='pass',prediction_rows=checked,
        max_replay_error_krw=max_error,metric_groups=count,inverse_scaling_and_paired_difference_verified=True)
    print('INFERENCE PASS',RESULT['inference_and_aggregation'],flush=True)

if __name__=='__main__':
    started=time.time();integrity();inputs();inference_and_metrics();same_path_ledger();independent_simulator();integrity()
    RESULT['status']='completed';RESULT['seconds']=time.time()-started
    (OUT/'audit.json').write_text(json.dumps(RESULT,indent=2))
    print('AUDIT COMPLETE',RESULT['seconds'],flush=True)
