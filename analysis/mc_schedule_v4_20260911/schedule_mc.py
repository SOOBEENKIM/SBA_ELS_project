"""Paired MC for contracts with different observation dates and maturities.

Payoff functions are the already audited V3/V2 functions. Only the shared path
collector is new: sample a union of dates, and stop KI at each contract's horizon.
"""
import numpy as np
import torch
from module.mc_contract_v3 import setup_v3,payoff_v3

def project_paths(logs,running,union,c):
    ridx=np.searchsorted(union,c.obs_days);midx=np.searchsorted(union,c.monthly_obs_days)
    return dict(wobs=logs[:,ridx],wrun=running[:,ridx],minimum=running[:,ridx[-1]],
                final=logs[:,ridx[-1]],monthly=logs[:,midx])

def shared_summaries(contracts,m,n,seed,path_chunk=2500,tblock=256):
    for c in contracts:c.validate()
    setups=[setup_v3(c,m) for c in contracts]
    longest=max(range(len(contracts)),key=lambda i:contracts[i].obs_days[-1]);p=setups[longest]
    N=p['N'];union=np.array(sorted({d for c in contracts for d in [*c.obs_days,*c.monthly_obs_days]}),int)
    obs=union-1;gen=torch.Generator().manual_seed(int(seed))
    loading=torch.tensor(p['loading']);drift=torch.tensor(p['drift']).cumsum(0).T.contiguous()
    for done in range(0,n,path_chunk):
        size=min(path_chunk,n-done);carry=torch.zeros(3,size,1);wmin=torch.zeros(size)
        wobs=torch.empty(size,len(union));wrun=torch.empty_like(wobs)
        for start in range(0,N,tblock):
            length=min(tblock,N-start);eps=torch.randn(3,size,length,generator=gen)
            x=(loading@eps.reshape(3,-1)).reshape(3,size,length).cumsum(2)
            x+=carry;carry=x[:,:,-1:].clone();x+=drift[:,None,start:start+length]
            worst=x.amin(0);running=torch.minimum(worst.cummin(1).values,wmin[:,None]);wmin=torch.minimum(wmin,worst.amin(1))
            sel=np.flatnonzero((obs>=start)&(obs<start+length))
            if len(sel):
                cols=obs[sel]-start;wobs[:,sel]=worst[:,cols];wrun[:,sel]=running[:,cols]
        yield [project_paths(wobs,wrun,union,c) for c in contracts],setups

@torch.no_grad()
def price_grid(contracts,m,paths=40000,seed=0,checkpoints=(10000,20000,40000),path_chunk=2500,tblock=256):
    counts=sorted(set(checkpoints));assert counts[-1]==paths and counts[0]>=2
    sums=np.zeros((len(contracts),8));out=[];done=0;checkpoint=0
    for summaries,ps in shared_summaries(contracts,m,paths,seed,path_chunk,tblock):
        results=[payoff_v3(s,p,c,details=True) for s,p,c in zip(summaries,ps,contracts)]
        pay=torch.stack([x[0] for x in results]);delta=pay-pay[0];monthly=torch.stack([x[2] for x in results]);begin=0
        while begin<pay.shape[1]:
            stop=min(pay.shape[1],begin+counts[checkpoint]-done)
            y=pay[:,begin:stop];d=delta[:,begin:stop];cp=monthly[:,begin:stop];principal=y-cp
            sums+=torch.stack([y.sum(1),(y*y).sum(1),d.sum(1),(d*d).sum(1),(d.abs()>1e-12).sum(1),cp.sum(1),principal.sum(1),torch.ones_like(y).sum(1)],1).numpy()
            done+=stop-begin;begin=stop
            if done==counts[checkpoint]:
                for j,s in enumerate(sums):
                    out.append(dict(case=j,paths=done,seed=seed,price_sum=s[0],price_sumsq=s[1],delta_sum=s[2],delta_sumsq=s[3],affected=int(s[4]),monthly_pv_sum=s[5],nonmonthly_pv_sum=s[6]))
                checkpoint+=1
                if checkpoint==len(counts):return out
    raise RuntimeError('Incomplete simulation')
