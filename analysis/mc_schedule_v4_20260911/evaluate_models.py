"""Frozen-model evaluation. This script has no optimizer or checkpoint writes."""
import pandas as pd
import numpy as np
import torch
from design import *
from models import load_predictor
from run_mc import preservation_files

def metrics(g):
    pe=g.price_error_krw.to_numpy();y=g.mc_price_krw.to_numpy();den=((y-y.mean())**2).sum()
    h=g[g.axis.ne('base')];e=h.delta_error_krw.to_numpy()
    r=(abs(h.mc_delta_krw)>1.96*h.mc_delta_se_krw)&h.affected.ge(30)
    return dict(n_families=g.family.nunique(),n_rows=len(g),price_mae=float(abs(pe).mean()),price_rmse=float(np.sqrt((pe**2).mean())),
        price_r2=float(1-(pe**2).sum()/den) if den else None,
        delta_mae=float(abs(e).mean()) if len(e) else None,delta_rmse=float(np.sqrt((e**2).mean())) if len(e) else None,
        delta_bias=float(e.mean()) if len(e) else None,within_1=float((abs(e)<=1).mean()) if len(e) else None,
        within_5=float((abs(e)<=5).mean()) if len(e) else None,within_10=float((abs(e)<=10).mean()) if len(e) else None,
        resolved_n=int(r.sum()),resolved_sign=float((np.sign(h.loc[r,'mc_delta_krw'])==np.sign(h.loc[r,'pred_delta_krw'])).mean()) if r.any() else None)

def main():
    torch.set_num_threads(2);assert json.loads((OUT/'labels_complete.json').read_text())['status']=='complete'
    frozen=json.loads((OUT/'label_spec.json').read_text())['preserved_inputs'];assert preservation_files()==frozen
    fs=json.loads((OUT/'families.json').read_text());labels=pd.read_csv(OUT/'mc_labels.csv').set_index(['family','case'])
    U=[];V=[];C=[];metadata=[];base=[]
    for r in fs:
        bi=len(U)
        for j,z in enumerate(r['variants']):
            c,m=unpack(dict(contract=z['contract'],market=r['market']));u,v,x=encode(c,m);U.append(u);V.append(v);C.append(x);base.append(bi)
            mc=labels.loc[(r['family'],j)]
            metadata.append(dict(family=r['family'],case=j,axis=z['axis'],offset=z['offset'],structure=r['structure'],monthly=r['monthly'],
                cohort='public_monthly_6' if r['origin']=='published_monthly' else ('monthly_test_128' if r['monthly'] else 'regular_test_128'),
                nobs=c.nobs,horizon_days=c.obs_days[-1],monthly_coupon_count=len(c.monthly_obs_days),monthly_total_accrual=sum(c.monthly_accruals),
                mc_price_krw=mc.price_krw,mc_price_se_krw=mc.price_se_krw,mc_delta_krw=mc.delta_krw,mc_delta_se_krw=mc.delta_se_krw,affected=int(mc.affected),
                actual_horizon_change_days=z['details'].get('actual_horizon_change_days',0),final_stub_coupon=z['details'].get('final_stub_coupon',False)))
    inputs=dict(U=np.stack(U),V=np.stack(V),C=np.stack(C));np.savez_compressed(OUT/'evaluation_inputs.npz',**inputs,base=base)
    meta=pd.DataFrame(metadata);meta.to_csv(OUT/'scenario_index.csv',index=False);base=np.asarray(base);rows=[]
    for arm in CFG['models']['arms']:
        preds=[]
        for seed in CFG['models']['seeds']:
            predict,ck=load_predictor(V3/'models'/f'{arm}_seed{seed}.pt');p=predict(inputs);assert np.isfinite(p).all();preds.append(p)
            g=meta.copy();g['model']=arm;g['training_seed']=str(seed);g['pred_price_krw']=p*10000;g['pred_delta_krw']=(p-p[base])*10000;rows.append(g)
        p=np.mean(preds,axis=0);g=meta.copy();g['model']=arm;g['training_seed']='ensemble';g['pred_price_krw']=p*10000;g['pred_delta_krw']=(p-p[base])*10000;rows.append(g)
    p=pd.concat(rows,ignore_index=True);p['price_error_krw']=p.pred_price_krw-p.mc_price_krw;p['delta_error_krw']=p.pred_delta_krw-p.mc_delta_krw
    p.to_csv(OUT/'predictions.csv',index=False)
    summaries=[]
    for key,g in p.groupby(['cohort','model','training_seed']):
        head=dict(zip(['cohort','model','training_seed'],key))
        summaries.append(dict(**head,axis='ALL',offset=0,**metrics(g)))
        for (axis,h),z in g.groupby(['axis','offset']):summaries.append(dict(**head,axis=axis,offset=h,**metrics(z)))
    pd.DataFrame(summaries).to_csv(OUT/'model_metrics.csv',index=False)
    structures=[]
    for key,g in p[p.training_seed.eq('ensemble')].groupby(['cohort','model','structure','axis','offset']):
        structures.append(dict(zip(['cohort','model','structure','axis','offset'],key),**metrics(g)))
    pd.DataFrame(structures).to_csv(OUT/'metrics_by_structure.csv',index=False)
    # MC numerical precision and nested convergence, separate from model error.
    ck=pd.read_csv(OUT/'mc_checkpoints.csv');twenty=ck[ck.paths_per_seed.eq(20000)].set_index(['family','case'])
    stability=meta[meta.axis.ne('base')].copy()
    stability['change_20k_to_40k_krw']=[r.mc_delta_krw-twenty.loc[(r.family,r.case),'delta_krw'] for r in stability.itertuples()]
    seed=pd.read_csv(OUT/'mc_seed_checkpoints.csv');seed=seed[seed.paths.eq(40000)].copy();seed['delta_krw']=seed.delta_sum/seed.paths*10000
    ranges=seed.groupby(['family','case']).delta_krw.agg(['min','max','std']);stability['mc_seed_range_krw']=[ranges.loc[(r.family,r.case),'max']-ranges.loc[(r.family,r.case),'min'] for r in stability.itertuples()]
    stability.to_csv(OUT/'mc_stability.csv',index=False)
    delta=[]
    for key,g in stability.groupby(['cohort','axis','offset']):
        r=(abs(g.mc_delta_krw)>1.96*g.mc_delta_se_krw)&g.affected.ge(30)
        delta.append(dict(zip(['cohort','axis','offset'],key),n_families=len(g),mean=float(g.mc_delta_krw.mean()),median=float(g.mc_delta_krw.median()),
            minimum=float(g.mc_delta_krw.min()),maximum=float(g.mc_delta_krw.max()),positive=int(((g.mc_delta_krw>0)&r).sum()),negative=int(((g.mc_delta_krw<0)&r).sum()),unresolved=int((~r).sum()),
            median_mc_ci_halfwidth=float((1.96*g.mc_delta_se_krw).median()),max_mc_ci_halfwidth=float((1.96*g.mc_delta_se_krw).max()),
            median_abs_20k_change=float(abs(g.change_20k_to_40k_krw).median()),max_abs_20k_change=float(abs(g.change_20k_to_40k_krw).max()),
            median_mc_seed_range=float(g.mc_seed_range_krw.median()),max_mc_seed_range=float(g.mc_seed_range_krw.max())))
    pd.DataFrame(delta).to_csv(OUT/'mc_effect_summary.csv',index=False)
    # Do existing inputs/checkpoints give exactly the same baseline predictions?
    old=pd.read_csv(V3/'results/predictions.csv');old=old[old.axis.eq('base')&old.model.isin(CFG['models']['arms'])&old.seed.eq('ensemble')]
    now=p[p.axis.eq('base')&p.training_seed.eq('ensemble')]
    compare=now.merge(old[['family','model','pred_price_krw']],on=['family','model'],suffixes=('_new','_old'))
    assert len(compare)==262*3
    max_reload=float(abs(compare.pred_price_krw_new-compare.pred_price_krw_old).max());assert max_reload<.02,max_reload
    # Independent MC-seed agreement with the prior baseline labels (not identical streams).
    oldmc=pd.read_csv(V3/'results/mc_labels.csv');oldmc=oldmc[oldmc.axis.eq('base')]
    b=meta[meta.axis.eq('base')].merge(oldmc[['family','price_krw','price_se_krw']],on='family')
    b['mc_difference_krw']=b.mc_price_krw-b.price_krw;b['z_independent_mc']=(b.mc_difference_krw/np.sqrt(b.mc_price_se_krw**2+b.price_se_krw**2))
    b.to_csv(OUT/'baseline_mc_comparison.csv',index=False)
    # Training support, reported by contract type rather than mixed with zero-padded slots.
    trainfs=[r for r in json.loads((V3/'results/families.json').read_text()) if r['split']=='train'];support=[]
    for monthly in [False,True]:
        tr=[r['contract'] for r in trainfs if r['monthly']==monthly];lo=min(c['obs_days'][-1] for c in tr);hi=max(c['obs_days'][-1] for c in tr)
        maxmonths=max(len(c['monthly_obs_days']) for c in tr);ns=sorted(set(len(c['obs_days']) for c in tr))
        for key,g in meta[meta.monthly.eq(monthly)].groupby(['cohort','axis','offset']):
            support.append(dict(zip(['cohort','axis','offset'],key),train_horizon_min_days=lo,train_horizon_max_days=hi,train_nobs_values=str(ns),train_max_monthly_count=maxmonths,
                n_families=len(g),horizon_outside_train=int(((g.horizon_days<lo)|(g.horizon_days>hi)).sum()),nobs_outside_train=int((~g.nobs.isin(ns)).sum()),monthly_count_above_train=int((g.monthly_coupon_count>maxmonths).sum())))
    pd.DataFrame(support).to_csv(OUT/'training_support.csv',index=False)
    # Training-seed variability on the same MC benchmark.
    seeds=p[p.training_seed.ne('ensemble')&p.axis.ne('base')].groupby(['cohort','family','model','axis','offset']).pred_delta_krw.agg(['min','max','std']).reset_index()
    seeds['range_krw']=seeds['max']-seeds['min'];seeds.to_csv(OUT/'model_seed_dispersion.csv',index=False)
    assert preservation_files()==frozen
    result=dict(status='complete',prediction_rows=len(p),scenario_rows=len(meta),models=9,ensembles=3,no_training=True,primary_model=CFG['models']['primary'],
        prior_baseline_prediction_max_difference_krw=max_reload,existing_artifacts_unchanged=True,
        baseline_independent_mc_max_abs_z=float(abs(b.z_independent_mc).max()),baseline_mc_abs_z_gt_196=int((abs(b.z_independent_mc)>1.96).sum()))
    (OUT/'evaluation_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    s=pd.DataFrame(summaries);print(s[s.model.eq(CFG['models']['primary'])&s.training_seed.eq('ensemble')&s.axis.ne('base')].to_string(index=False))

if __name__=='__main__':main()
