"""Payoff/RNG from frozen notebook 5; deterministic dividend drops and MC SE added.
Original functions are compiled from the supplied notebook, without editing reference/.
The only sampler instrumentation accumulates squared discounted payoffs for standard error.
"""
from pathlib import Path
import ast,json,copy,hashlib
import numpy as np,torch
HERE=Path(__file__).resolve().parent;REF=HERE.parent/'ppt_reproduction_20260917/reference'
import sys
sys.path.insert(0,str(REF))
from module.features import chol_psd
nb=json.loads((REF/'5_MC_implied_vol.ipynb').read_text());source=next(''.join(c['source']) for c in nb['cells'] if 'def mc_daily_t(' in ''.join(c.get('source',[])))
functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('_setup','mc_daily_t')]
ORIGINAL={'np':np,'torch':torch,'chol_psd':chol_psd,'NPATH':40000,'DEV':'cuda'}
exec(compile(ast.Module(body=functions,type_ignores=[]),'original_notebook5','exec'),ORIGINAL)
transformed=copy.deepcopy(functions)
class Moments(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name!='mc_daily_t':return node
  out=[]
  for stmt in node.body:
   if isinstance(stmt,ast.While):
    out.extend(ast.parse('tot2 = torch.zeros((), device=d, dtype=torch.float64)').body)
    body=[]
    for sub in stmt.body:
     if isinstance(sub,ast.AugAssign) and isinstance(sub.target,ast.Name) and sub.target.id=='tot':
      # Original tot += torch.where(...).sum(dtype=torch.float64).
      assert isinstance(sub.value,ast.Call) and isinstance(sub.value.func,ast.Attribute) and sub.value.func.attr=='sum'
      expression=ast.unparse(sub.value.func.value)
      body.extend(ast.parse(f'pv = ({expression}).to(torch.float64)\ntot += pv.sum()\ntot2 += pv.square().sum()').body)
     else:body.append(sub)
    stmt.body=body
   if isinstance(stmt,ast.Return):
    out.extend(ast.parse('mean = tot / n\nvariance = torch.clamp((tot2 - n * mean.square()) / (n - 1), min=0.0)\nreturn float(mean.item()), float(torch.sqrt(variance / n).item())').body)
   else:out.append(stmt)
  node.body=out;return node
transformed=[Moments().visit(n) for n in transformed];tree=ast.fix_missing_locations(ast.Module(body=transformed,type_ignores=[]))
ENV={'np':np,'torch':torch,'chol_psd':chol_psd,'NPATH':40000,'DEV':'cuda'}
exec(compile(tree,'notebook5_with_payoff_moments','exec'),ENV)
BASE_SETUP=ENV['_setup']
SOURCE_HASH=hashlib.sha256(source.encode()).hexdigest()
def price(sigs,corr,curve,contract,events,seed,device='cuda',n=40000):
 def setup(*a,**kw):
  result=BASE_SETUP(*a,**kw)
  if events:
   drops=np.zeros_like(result['drift'])
   for step,asset,logdrop in events:drops[int(step),int(asset)]+=np.float32(logdrop)
   result['drift']=result['drift']+drops
  return result
 ENV['_setup']=setup
 return ENV['mc_daily_t'](sigs,corr,curve,**contract,seed=seed,dev=device,n=n,path_chunk=20000,tblock=512)
def contract(row):
 n=int(row.nobs);get=lambda key:[float(row[f'{key}_{j}']) for j in range(n)]
 return dict(B=float(row.B),strikes=get('strk'),ten=float(row.tenor),c=float(row.coupon),pmts=get('pmt'),lz_barr=get('lz_barr'),lz_pmt=get('lz_pmt'))
def regression():
 torch.set_num_threads(2);sigs=[.18,.25,.3];corr=np.array([[1,.3,.4],[.3,1,.5],[.4,.5,1]]);curve=lambda t:np.full_like(np.asarray(t),.025,dtype=float)
 cases=[dict(B=.6,strikes=[.9,.85,.8],ten=1.5,c=.08,pmts=[.04,.08,.12],lz_barr=[np.nan]*3,lz_pmt=[0]*3),dict(B=1.,strikes=[.9,.85,.8],ten=1.5,c=.08,pmts=[.04,.08,.12],lz_barr=[.7,np.nan,np.nan],lz_pmt=[.025,0,0])]
 for case in cases:
  baseline=ORIGINAL['mc_daily_t'](sigs,corr,curve,**case,n=40000,seed=123,dev='cuda')
  current,se=price(sigs,corr,curve,case,[],123,'cuda')
  assert abs(current-baseline)<1e-12,(current,baseline)
  assert se>0
  # Identical spot/curve/contract inputs and random stream are exactly repeatable.
  again,se2=price(sigs,corr,curve,case,[],123,'cuda');assert current==again and se==se2
 print('MC engine original-price equality and deterministic-repeat checks PASS',flush=True)
if __name__=='__main__':regression()
