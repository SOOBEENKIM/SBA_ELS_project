"""Dated cash-flow ledgers for all six DART-checked monthly contracts."""
from common import *
import torch
from module.mc_contract_v3 import payoff_v3,setup_v3

def main():
    rows=[]
    for r in json.loads((OUT/'published_monthly_contracts.json').read_text()):
        c,m=unpack(r);p=setup_v3(c,m)
        for label,ratio in [('first_redemption',1.01),('coupon_stream_to_maturity',.7),('loss_without_coupons',.3)]:
            x=np.full(c.obs_days[-1]+1,ratio);x[0]=1
            ledger=[];end=c.obs_days[-1];jpay=c.nobs-1;principal=ratio
            for j,day in enumerate(c.obs_days):
                if ratio>=c.strikes[j] or (c.lz_barr[j] is not None and ratio>=c.lz_barr[j]):
                    end=day;jpay=j;principal=1.;break
            else:
                if c.has_ki and ratio>=c.ki_barrier:principal=1.
            ledger.append((c.pay_days[jpay],principal))
            for day,pay,b,accr in zip(c.monthly_obs_days,c.monthly_pay_days,c.monthly_barriers,c.monthly_accruals):
                if day<=end and ratio>=b:ledger.append((pay,c.coupon*accr))
            expected=sum(amount*np.exp(-float(F.zero_curve(m.beta,[day/365])[0])*day/365) for day,amount in ledger)
            s=dict(wobs=torch.tensor(np.log(x[list(c.obs_days)]))[None],wrun=torch.tensor(np.log(x[list(c.obs_days)]))[None],
                minimum=torch.tensor([np.log(ratio)]),final=torch.tensor([np.log(ratio)]),monthly=torch.tensor(np.log(x[list(c.monthly_obs_days)]))[None])
            observed,_,cp,n=payoff_v3(s,p,c,details=True)
            assert abs(float(observed[0])-expected)<1e-12,(r['original_item'],label)
            rows.append(dict(item=r['original_item'],case=label,expected_price_krw=expected*10000,
                engine_price_krw=float(observed[0])*10000,monthly_coupon_count=int(n[0]),
                monthly_coupon_pv_krw=float(cp[0])*10000,cashflow_ledger=ledger))
    (OUT/'published_payoff_validation.json').write_text(json.dumps(dict(status='pass',cases=rows),ensure_ascii=False,indent=2))
    print('PUBLISHED PAYOFF VALIDATION PASS',len(rows),'source-defined cash-flow scenarios')

if __name__=='__main__':main()
