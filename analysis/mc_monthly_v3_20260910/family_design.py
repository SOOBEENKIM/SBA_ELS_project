"""Prespecified family generation and template grouping before MC/test fitting."""
from datetime import date,timedelta
from dataclasses import replace
from common import *
from calendar_rules import SettlementCalendar

def market(rng):
    sig=tuple(rng.uniform(.12,.45,3));load=rng.uniform(.45,.9,3);corr=np.outer(load,load);np.fill_diagonal(corr,1.)
    return MarketV2(sig,tuple(map(tuple,corr)),tuple([rng.uniform(.005,.045)*100,rng.uniform(-.006,.006)*100,rng.uniform(-.003,.003)*100]))

def template_key(r):
    q=np.asarray(r['monthly_pmts'])/r['coupon']
    d={k:r[k] for k in ['structure','strikes','lz_barr','lz_pmts','ki_barrier','monthly_barriers']}
    d['redemption_months']=np.round(np.asarray(r['obs_days'])/(365.25/12)).tolist()
    d['coupon_months']=np.round(np.asarray(r['monthly_obs_days'])/(365.25/12)).tolist()
    d['monthly_accruals']=np.round(q,7).tolist()
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()

def template_pools():
    rows=json.loads((OUT/'monthly_templates.json').read_text())
    sourced=json.loads((OUT/'published_monthly_contracts.json').read_text());ids={r['original_item'] for r in sourced}
    held={template_key(r) for r in rows if r['item'] in ids}
    pools={s:{} for s in ['train','validation','test']};index=[]
    for r in rows:
        key=template_key(r);b=int(key[:8],16)%10
        split='stress_withheld' if key in held else ('test' if b==0 else 'validation' if b==1 else 'train')
        index.append(dict(item=r['item'],template_group=key,split=split,structure=r['structure']))
        if split in pools:pools[split].setdefault(r['structure'],[]).append(r)
    import pandas as pd
    pd.DataFrame(index).to_csv(OUT/'monthly_template_splits.csv',index=False)
    for s in pools:assert pools[s],s
    groups={s:{x['template_group'] for x in index if x['split']==s} for s in pools}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
    (OUT/'template_split_validation.json').write_text(json.dumps(dict(disjoint=True,held_stress_groups=len(held),
        counts={s:{st:len(v) for st,v in p.items()} for s,p in pools.items()},groups={s:len(v) for s,v in groups.items()}),indent=2))
    return pools

def ordinary(index,split,rng):
    st=STRUCTURES[index%4];ki=st.endswith(' KI');lz=st.startswith('LIZARD')
    accr=[np.arange(1,7)/2,np.r_[.25,.5,1.,1.5,2.,2.5,3.],np.arange(1,13)/4][int(rng.integers(0,3))]
    obs=np.round(accr*365).astype(int);jitter=rng.integers(-3,4,len(obs));jitter[-1]=0;obs+=jitter
    first=rng.uniform(.85,.98);last=rng.uniform(.55,.8)
    if rng.random()<.5: # include plateaus present in the actual contract universe
        drops=rng.choice([0.,.025,.05],len(obs)-1,p=[.4,.2,.4]);strikes=np.r_[first,first-np.cumsum(drops)]
        strikes=np.maximum(strikes,last+.01)
    else:strikes=np.linspace(first,first-rng.uniform(.03,.13),len(obs))
    strikes[-1]=min(last,strikes[-2]-.005)
    coupon=rng.uniform(.015,.12);B=rng.uniform(.35,min(.65,strikes[-1]-.02)) if ki else None
    barr=[None]*len(obs);lp=[0.]*len(obs)
    if lz:
        # Lizard dates are not forced to start at observation 1.
        count=int(rng.integers(1,min(4,len(obs))))
        for j in sorted(rng.choice(np.arange(min(4,len(obs)-1)),count,replace=False)):
            barr[j]=float(max(.45,min(.92,strikes[j]-rng.uniform(.04,.19))))
            lp[j]=float(rng.uniform(.015,.06)*accr[j])
    c=ContractV3(f'{split}-regular-{index:05d}',st,tuple(obs.tolist()),tuple((obs+int(rng.integers(1,6))).tolist()),
        tuple(strikes),float(coupon),tuple(accr),tuple(barr),tuple(lp),B,3. if ki else None).validate()
    return family_record(c,market(rng),split,'synthetic_regular')

def monthly(index,split,rng,pool,cal):
    st=sorted(pool)[index%len(pool)];r=pool[st][int(rng.integers(len(pool[st])))]
    issue=date.fromisoformat(r['issue']);lag=int(rng.choice([2,3]));coupon=float(np.clip(r['coupon']*rng.uniform(.8,1.2),.015,.14))
    pay=tuple((cal.advance(issue+timedelta(days=x),lag)-issue).days for x in r['obs_days'])
    mpay=tuple((cal.advance(issue+timedelta(days=x),lag)-issue).days for x in r['monthly_obs_days'])
    c=ContractV3(f'{split}-monthly-{index:05d}',st,tuple(r['obs_days']),pay,tuple(r['strikes']),coupon,
        (0.,)*len(pay),tuple(r['lz_barr']),tuple(r['lz_pmts']),r['ki_barrier'],0. if st.endswith(' KI') else None,
        monthly_obs_days=tuple(r['monthly_obs_days']),monthly_pay_days=mpay,
        monthly_barriers=tuple(r['monthly_barriers']),monthly_accruals=tuple(np.asarray(r['monthly_pmts'])/r['coupon']),
        coupon_on_redemption_day=True).validate()
    return family_record(c,market(rng),split,'synthetic_monthly',template_item=r['item'],template_group=template_key(r),
        settlement_lag_business_days=lag,contract_scope='Synthetic contract defined using observed schedule; payment lag and coupon explicitly varied, not original product pricing')

def design():
    pools=template_pools();cal=SettlementCalendar();out=[]
    for split,count in CFG['family_counts'].items():
        seed=int(hashlib.sha256(f'{CFG["sampling_seed"]}:{split}'.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
        assert count%2==0
        for i in range(count//2):out.append(ordinary(i,split,rng));out.append(monthly(i,split,rng,pools[split],cal))
    for r in json.loads((OUT/'published_monthly_contracts.json').read_text()):
        c,m=unpack(r);out.append(family_record(c,m,'stress','published_monthly',original_item=r['original_item'],source=r['source']))
    # Previous real-term counterparts are a transparent continuity diagnostic,
    # never a new untouched test set or source of fitted targets.
    for r in json.loads((PROJECT/'analysis/mc_contract_v2_20260910/results/families.json').read_text()):
        if r['split']!='stress':continue
        c,m=unpack(r);out.append(family_record(c,m,'stress','previous_'+r['origin'],original_item=r.get('original_item')))
    return out
