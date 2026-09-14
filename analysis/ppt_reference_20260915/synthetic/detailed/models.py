"""Explicit-schedule Stage-1 DeepONet and separate coupon-affine ablation."""
import numpy as np
import torch
from torch import nn
from module.networks import CurveOperatorV2,mlp

AFFINE_KEEP=np.array([i for i in range(461) if i not in set(range(12,24))|set(range(280,340))|{96}],int)

class AffineCouponDeepONet(nn.Module):
    """V(c)=A(other inputs)+(c-.06)*100*positive B(other inputs).

    Applied only when regular/survival payments are c times fixed accrual factors,
    and Lizard payments and all event conditions are independent of c.
    """
    def __init__(self,nvc,ncon,P):
        super().__init__();self.core=CurveOperatorV2(nvc,ncon,P)
        self.slope_t=mlp(ncon,P);self.slope_bias=nn.Parameter(torch.tensor(-1.5))
    def V(self,u,v,c,coupon):
        b=self.core.branch(torch.cat([self.core.cnn(u.unsqueeze(1)).squeeze(-1),v],-1))
        intercept=(b*self.core.t(c)).sum(-1)+self.core.b0
        slope=torch.nn.functional.softplus((b*self.slope_t(c)).sum(-1)+self.slope_bias)
        return intercept+(coupon-.06)*100*slope

def build(arm,P):
    return AffineCouponDeepONet(9,len(AFFINE_KEEP),P) if arm=='affine_coupon_delta' else CurveOperatorV2(9,461,P)

def features(data,arm,stats=None):
    arrays={'u':np.asarray(data['U'],np.float32),'v':np.asarray(data['V'],np.float32),
            'c':np.asarray(data['C'],np.float32)}
    coupon=arrays['c'][:,96].copy()
    if arm=='affine_coupon_delta':arrays['c']=arrays['c'][:,AFFINE_KEEP]
    if stats is None:
        stats={}
        for k,a in arrays.items():
            stats[k+'m']=a.mean(0);sd=a.std(0);stats[k+'s']=np.where(sd<1e-6,1.,sd)
    result=[torch.tensor((arrays[k]-stats[k+'m'])/stats[k+'s'],dtype=torch.float32) for k in ['u','v','c']]
    return (*result,torch.tensor(coupon)),stats

def forward(net,arm,x,idx):
    return net.V(*[a[idx] for a in x]) if arm=='affine_coupon_delta' else net.V(*[a[idx] for a in x[:3]])

@torch.no_grad()
def predict(net,arm,x,ym,ys,batch=4096):
    result=[]
    for i in range(0,len(x[0]),batch):result.append((forward(net,arm,x,slice(i,i+batch))*ys+ym).numpy())
    return np.concatenate(result).astype(float)

def load_predictor(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    net=build(ck['arm'],ck['P']);net.load_state_dict(ck['state']);net.eval()
    def call(data):
        x,_=features(data,ck['arm'],ck['stats'])
        return predict(net,ck['arm'],x,ck['ym'],ck['ys'])
    return call,ck
