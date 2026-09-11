"""Before fitting: check simulator labels, single-field interventions and splits."""
import pandas as pd
from dataclasses import asdict
from common import *

def main():
    assert json.loads((OUT/'pricer_validation.json').read_text())['status']=='pass'
    labels=pd.read_csv(OUT/'mc_labels.csv');families=json.loads((OUT/'families.json').read_text())
    base=labels[labels.axis=='base'].set_index('family').price_krw
    err=labels.price_krw-labels.family.map(base)-labels.delta_krw
    assert abs(err).max()<1e-7
    h=labels[labels.axis=='coupon_regular'].copy();h['slope']=h.delta_krw/h.offset
    slope=h.groupby('family').slope.agg(['min','max']);assert (slope['min']>=-1e-8).all()
    assert (slope['max']-slope['min']).max()<1e-5
    for axis in ['ki_barrier','monthly_barrier']:
        h=labels[labels.axis==axis];assert (h.delta_krw*h.offset<=1e-8).all()
    raw=pd.read_csv(OUT/'mc_seed_checkpoints.csv')
    invariant=raw.axis.isin(['coupon_regular','monthly_barrier'])
    assert raw.loc[invariant,'changed_events'].eq(0).all()
    seedmap=raw[['family','replicate','seed']].drop_duplicates();assert not seedmap.seed.duplicated().any()
    assert labels[labels.split.isin(['train','validation'])].total_paths.eq(40000).all()
    assert labels[labels.split.isin(['test','stress'])].total_paths.eq(120000).all()
    allowed={'coupon_regular':{'coupon'},'ki_barrier':{'ki_barrier'},'first_strike':{'strikes'},
        'last_strike':{'strikes'},'monthly_barrier':{'monthly_barriers'}}
    for r in families:
        c,m=unpack(r)
        for axis,h,new in cases(c):
            if axis=='base':continue
            assert all(asdict(c)[k]==v for k,v in asdict(new).items() if k not in allowed[axis])
    ts=pd.read_csv(OUT/'monthly_template_splits.csv')
    assert ts.groupby('template_group').split.nunique().max()==1
    for split in ['train','validation','test','stress']:
        d=dict(np.load(OUT/f'{split}.npz'));rows=pd.read_csv(OUT/f'{split}_rows.csv')
        assert d['C'].shape==(len(rows),461);assert np.isfinite(d['C']).all()
        assert np.max(abs(d['Y']-d['Y'][d['base']]-d['D']))<1e-10
        assert rows.groupby('family').split.nunique().max()==1
    result=dict(status='pass',families=len(families),scenarios=len(labels),unique_independent_MC_seeds=len(seedmap),
        price_minus_base_equals_delta_max_error_krw=float(abs(err).max()),
        coupon_linearity_slope_spread=float((slope['max']-slope['min']).max()),
        coupon_and_monthly_barrier_leave_redemption_events_unchanged=True,
        coupon_positive_KI_and_monthly_barrier_nonpositive=True,template_groups_disjoint=True,
        paths_per_seed=40000,test_and_stress_independent_replicates=3)
    (OUT/'label_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
