"""Frozen three-arm Stage-1 evaluation on held-out monthly/regular contract families."""
import pandas as pd
import torch
from common import *
from models import load_predictor

def metric(g):
    pe=g.price_error_krw.to_numpy();y=g.mc_price_krw.to_numpy();den=((y-y.mean())**2).sum()
    out=dict(n_rows=len(g),n_families=g.family.nunique(),price_mae=float(abs(pe).mean()),price_rmse=float(np.sqrt((pe**2).mean())),
        price_r2=float(1-(pe**2).sum()/den) if den>1e-12 else None)
    h=g[g.axis!='base']
    if len(h):
        e=h.delta_error_krw.to_numpy();resolved=(abs(h.mc_delta_krw)>1.96*h.mc_delta_se_krw)&(h.affected>=30)
        out.update(delta_mae=float(abs(e).mean()),delta_rmse=float(np.sqrt((e**2).mean())),delta_bias=float(e.mean()),
            within_1=float((abs(e)<=1).mean()),within_5=float((abs(e)<=5).mean()),within_10=float((abs(e)<=10).mean()),
            resolved_n=int(resolved.sum()),resolved_sign=float((np.sign(h.loc[resolved,'mc_delta_krw'])==np.sign(h.loc[resolved,'pred_delta_krw'])).mean()) if resolved.any() else None)
    return out

def main():
    torch.set_num_threads(2);selection=json.loads((OUT/'model_selection.json').read_text());chosen=selection['selected_arm']
    for name,h in selection['checkpoint_hashes'].items():assert sha(HERE/'models'/name)==h
    assert selection['test_data_opened_by_training'] is False
    out=[];hashes={}
    for split in ['test','stress']:
        d=dict(np.load(OUT/f'{split}.npz'));meta=pd.read_csv(OUT/f'{split}_rows.csv');base=d['base'].astype(int)
        meta['cohort']=meta.origin.map({'synthetic_regular':'regular_test','synthetic_monthly':'monthly_test',
            'published_monthly':'published_monthly_6','previous_canonical':'previous_canonical_4',
            'previous_real_terms_synthetic':'previous_real_terms_82'})
        def record(model,seed,p):
            g=meta.copy();g['model']=model;g['seed']=str(seed);g['mc_price_krw']=d['Y']*10000;g['mc_delta_krw']=d['D']*10000;g['mc_delta_se_krw']=d['SE']*10000
            g['pred_price_krw']=p*10000;g['pred_delta_krw']=(p-p[base])*10000
            g['price_error_krw']=g.pred_price_krw-g.mc_price_krw;g['delta_error_krw']=g.pred_delta_krw-g.mc_delta_krw
            out.append(g[g.pred_price_krw.notna()])
        for arm in CFG['learning']['arms']:
            ps=[]
            for seed in CFG['learning']['seeds']:
                call,_=load_predictor(HERE/'models'/f'{arm}_seed{seed}.pt');p=call(d);ps.append(p);record(arm,seed,p)
            record(arm,'ensemble',np.mean(ps,0))
    p=pd.concat(out,ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
    summaries=[];offsets=[];levels=[];by_structure=[]
    for key,g in p.groupby(['cohort','model','seed']):
        keys=dict(zip(['cohort','model','seed'],key))
        for axis,h in [('ALL',g)]+list(g.groupby('axis')):summaries.append(dict(**keys,axis=axis,**metric(h)))
        levels.append(dict(**keys,**metric(g[g.axis=='base'])))
        for (axis,h),z in g[g.axis!='base'].groupby(['axis','offset']):offsets.append(dict(**keys,axis=axis,offset=h,**metric(z)))
        for st,z in g.groupby('structure'):by_structure.append(dict(**keys,structure=st,**metric(z)))
    pd.DataFrame(summaries).to_csv(OUT/'model_metrics.csv',index=False);pd.DataFrame(offsets).to_csv(OUT/'metrics_by_offset.csv',index=False)
    pd.DataFrame(levels).to_csv(OUT/'baseline_metrics.csv',index=False);pd.DataFrame(by_structure).to_csv(OUT/'metrics_by_structure.csv',index=False)
    boot=[];rng=np.random.default_rng(810)
    for cohort in ['regular_test','monthly_test']:
        h=p[(p.cohort==cohort)&p.seed.isin(['ensemble','frozen'])&(p.axis!='base')]
        for other in ['augmented_price','augmented_delta','affine_coupon_delta']:
            if other==chosen:continue
            a=h[h.model==chosen].groupby('family').delta_error_krw.apply(lambda z:abs(z).mean())
            b=h[h.model==other].groupby('family').delta_error_krw.apply(lambda z:abs(z).mean())
            diff=(a-b).dropna().to_numpy();samples=diff[rng.integers(0,len(diff),(2000,len(diff)))].mean(1)
            boot.append(dict(cohort=cohort,selected=chosen,comparator=other,selected_minus_comparator=float(diff.mean()),
                ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),n_families=len(diff)))
    pd.DataFrame(boot).to_csv(OUT/'paired_improvement_bootstrap.csv',index=False)
    aff=p[(p.model=='affine_coupon_delta')&(p.axis=='coupon_regular')]
    assert (aff.pred_delta_krw*aff.offset>=-1e-6).all()
    result=dict(status='complete',selected_arm=chosen,test_used_for_selection=False,
        prediction_rows=len(p),test_families=256,published_monthly_families=6,
        coupon_affine_positive=True,scope='Bank-settled monthly and regular contractual cash flows under explicit GBM market assumptions')
    (OUT/'evaluation_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    s=pd.DataFrame(summaries);print(s[s.cohort.isin(['monthly_test','regular_test','published_monthly_6'])&s.seed.isin(['ensemble','frozen'])&s.axis.eq('ALL')].to_string(index=False))

if __name__=='__main__':main()
