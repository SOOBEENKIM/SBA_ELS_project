"""Matched HV/IV FAIR sextile figures from completed original-product MC prices."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results';FIG=HERE/'figures'
BASE=ROOT/'analysis/fair_mc_iv_population_20260914'

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def md(rows):
    cols=list(rows[0]);s='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    for r in rows:s+='| '+' | '.join(f'{r[k]:,.2f}' if isinstance(r[k],float) else str(r[k]) for k in cols)+' |\n'
    return s

def figures(d,suffix=''):
    d=d.copy();d['segment']=pd.qcut(d.fair,6)
    g=d.groupby('segment',observed=True).agg(n=('item','size'),
        mean_gap_hv_krw=('gap_hv_krw','mean'),mean_gap_iv_krw=('gap_krw','mean'),
        mape_hv_pct=('ape_hv_pct','mean'),mape_iv_pct=('ape_pct','mean'),
        mean_iv_minus_hv_krw=('iv_minus_hv_krw','mean'))
    g['mape_change_iv_minus_hv_pp']=g.mape_iv_pct-g.mape_hv_pct
    assert g.n.sum()==len(d) and len(g)==6
    g.to_csv(OUT/f'price_segments_hv_iv{suffix}.csv')
    x=np.arange(6);w=.39;colors=['#E4775C','#2175A0']
    fig,ax=plt.subplots(1,2,figsize=(13.333,5.0),layout='constrained')
    for a,cols in zip(ax,[['mean_gap_hv_krw','mean_gap_iv_krw'],['mape_hv_pct','mape_iv_pct']]):
        for j,(col,color,label) in enumerate(zip(cols,colors,['Historical (HV)','Implied (IV)'])):
            a.bar(x+(j-.5)*w,g[col],width=w,color=color,label=label,edgecolor='white',linewidth=.4)
        a.set_xticks(x,[f'{i.left:.3f}–{i.right:.3f}\nn={int(n):,}' for i,n in zip(g.index,g.n)],rotation=25,ha='right',fontsize=9)
        a.set_xlabel('FAIR / issue price segment (low → high)')
        a.grid(axis='y',color='#ddd',alpha=.65,lw=.6)
        a.legend(frameon=True,fontsize=9)
    ax[0].axhline(0,color='#555',lw=.8)
    ax[0].set(title='Pricing gap by price segment',ylabel='Mean (FAIR − MC) [KRW]')
    ax[1].set(title='|FAIR − MC| / FAIR by price segment',ylabel='MAPE (%)',ylim=(0,None))
    scope=' · STEP KI only' if suffix else ''
    fig.suptitle(f'Same {len(d):,} actual products{scope} · HV versus IV · 40,000 paths/seed × 3 seeds\nBoth prices normalized to KRW 10,000 invested',fontsize=11)
    for ext in ['png','svg','pdf']:
        target=FIG/f'fair_mc_hv_iv_by_price_segment{suffix}.{ext}'
        fig.savefig(target,dpi=220,bbox_inches='tight')
        if ext=='svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    return g

def main():
    cfg=json.loads((HERE/'protocol.json').read_text())
    assert json.loads((OUT/'completion.json').read_text())['status']=='complete'
    d=pd.read_csv(OUT/'fair_mc_hv_iv.csv')
    assert len(d)==cfg['products'] and not d.item.duplicated().any()
    iv=pd.read_csv(BASE/'results/fair_mc_iv.csv')
    joined=d.merge(iv,on='item',suffixes=('_new','_saved'),validate='one_to_one')
    assert len(joined)==len(iv)==len(d)
    for c in ['fair','mc_iv','gap_krw','ape_pct','mc_face_krw','issue_price_krw']:
        np.testing.assert_allclose(joined[c+'_new'],joined[c+'_saved'],rtol=0,atol=1e-9)
    for col,want in [('gap_hv_krw',(d.fair_raw_krw-d.mc_hv_face_krw)/d.issue_price_krw*10000),
                     ('ape_hv_pct',abs(d.fair_raw_krw-d.mc_hv_face_krw)/d.fair_raw_krw*100),
                     ('iv_minus_hv_krw',(d.mc_face_krw-d.mc_hv_face_krw)/d.issue_price_krw*10000)]:
        np.testing.assert_allclose(d[col],want,rtol=0,atol=1e-8)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'font.family':'DejaVu Sans','font.size':11})
    g=figures(d)
    original=pd.read_csv(BASE/'results/price_segments.csv')
    np.testing.assert_array_equal(g.n,original.n)
    np.testing.assert_allclose(g.mean_gap_iv_krw,original.err_mean_KRW,rtol=0,atol=1e-8)
    np.testing.assert_allclose(g.mape_iv_pct,original.MAPE_pct,rtol=0,atol=1e-10)
    step=d[d.structure.eq('STEP KI')]
    if len(step)>=60:figures(step,'_step_ki')
    summary=dict(products=len(d),hv_mean_gap_krw=float(d.gap_hv_krw.mean()),iv_mean_gap_krw=float(d.gap_krw.mean()),
        hv_median_gap_krw=float(d.gap_hv_krw.median()),iv_median_gap_krw=float(d.gap_krw.median()),
        hv_mae_krw=float(abs(d.gap_hv_krw).mean()),iv_mae_krw=float(abs(d.gap_krw).mean()),
        hv_mape_pct=float(d.ape_hv_pct.mean()),iv_mape_pct=float(d.ape_pct.mean()),
        mape_change_iv_minus_hv_pp=float(d.ape_change_iv_minus_hv_pp.mean()),
        mean_iv_minus_hv_krw=float(d.iv_minus_hv_krw.mean()),
        fraction_products_with_smaller_absolute_gap_using_iv=float((abs(d.gap_krw)<abs(d.gap_hv_krw)).mean()),
        bins_with_lower_iv_mape=int((g.mape_iv_pct<g.mape_hv_pct).sum()),
        hv_mape_strictly_decreasing=bool(np.all(np.diff(g.mape_hv_pct)<0)),
        iv_mape_strictly_decreasing=bool(np.all(np.diff(g.mape_iv_pct)<0)))
    dump(OUT/'summary.json',summary)
    raw=pd.read_csv(OUT/'seed_checkpoints_hv_iv.csv.gz')
    assert len(raw)==len(d)*9 and not raw.duplicated(['item','replicate','paths']).any()
    seed=raw.merge(d[['item','fair_raw_krw','issue_price_krw']],on='item',validate='many_to_one')
    for arm in ['hv','iv']:
        seed['mc_'+arm+'_face']=seed['price_sum_'+arm]/seed.paths*10000
        seed['gap_'+arm+'_krw']=(seed.fair_raw_krw-seed['mc_'+arm+'_face'])/seed.issue_price_krw*10000
        seed['ape_'+arm+'_pct']=abs(seed.fair_raw_krw-seed['mc_'+arm+'_face'])/seed.fair_raw_krw*100
    sg=seed.groupby(['paths','replicate']).agg(n=('item','size'),mean_gap_hv_krw=('gap_hv_krw','mean'),mean_gap_iv_krw=('gap_iv_krw','mean'),mape_hv_pct=('ape_hv_pct','mean'),mape_iv_pct=('ape_iv_pct','mean')).reset_index()
    sg['mape_change_iv_minus_hv_pp']=sg.mape_iv_pct-sg.mape_hv_pct
    assert sg.n.eq(len(d)).all();sg.to_csv(OUT/'seed_summary.csv',index=False)
    checkpoints=pd.read_csv(OUT/'mc_hv_checkpoints.csv.gz').pivot(index='item',columns='paths_per_seed',values='mc_hv_face_krw')
    convergence=dict(hv_mean_abs_20k_to_40k_face_krw=float(abs(checkpoints[40000]-checkpoints[20000]).mean()),
        hv_median_mc_se_face_krw=float(d.mc_hv_se_face_krw.median()),hv_p95_mc_se_face_krw=float(d.mc_hv_se_face_krw.quantile(.95)))
    dump(OUT/'convergence_summary.json',convergence)
    for p,h in cfg['preserved_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    verification=dict(status='pass',products=len(d),same_products_in_both_arms=True,iv_prices_unchanged=True,
        iv_sextile_statistics_match_saved_iv_plot=True,common_normalization_passed=True,
        all_three_seed_results_aligned=True,all_preserved_hashes_match=True)
    dump(OUT/'verification.json',verification)
    table=[]
    for interval,r in g.iterrows():
        table.append({'공정가 구간':f'{interval.left:.3f}–{interval.right:.3f}','상품 수':int(r.n),
            'HV 평균 FAIR−MC(원)':float(r.mean_gap_hv_krw),'IV 평균 FAIR−MC(원)':float(r.mean_gap_iv_krw),
            'HV MAPE(%)':float(r.mape_hv_pct),'IV MAPE(%)':float(r.mape_iv_pct)})
    change='감소' if summary['mape_change_iv_minus_hv_pp']<0 else '증가'
    text=f'''# 동일 상품의 HV·IV MC 공정가 구간별 비교

동일한 실상품 **{len(d):,}개**에서 변동성 입력만 역사적 변동성(HV)과 내재변동성(IV)으로 바꾸어 공정가와 비교했습니다.
원문 계약, 평가·지급 일정, 상관계수, 금리곡선, 배당률, 난수 시드, 경로 수, 정규화 기준은 동일합니다.
완료된 IV 계산값은 그대로 재사용하고 부족했던 HV 가격만 추가로 계산했습니다. DeepONet은 이 분석에 사용하지 않았습니다.

![동일 상품 HV·IV 분위별 가격 차이](figures/fair_mc_hv_iv_by_price_segment.png)

{md(table)}

전체 평균 FAIR−MC는 HV **{summary['hv_mean_gap_krw']:,.2f}원**, IV **{summary['iv_mean_gap_krw']:,.2f}원**입니다.
전체 MAPE는 HV **{summary['hv_mape_pct']:.2f}%**, IV **{summary['iv_mape_pct']:.2f}%**로,
IV 적용 시 **{abs(summary['mape_change_iv_minus_hv_pp']):.2f}%p {change}**했습니다.
6개 구간 중 IV MAPE가 낮은 구간은 **{summary['bins_with_lower_iv_mape']}개**이고,
상품별 절대 가격 차이가 줄어든 비율은 **{summary['fraction_products_with_smaller_absolute_gap_using_iv']*100:.2f}%**입니다.

평균 절대 가격 차이(MAE)는 HV **{summary['hv_mae_krw']:,.2f}원**, IV **{summary['iv_mae_krw']:,.2f}원**입니다.
부호가 있는 평균 FAIR−MC는 양수·음수 차이가 상쇄될 수 있으므로, 그 값이 0에 가까워졌다는 사실만으로 오차 개선을 판단하지 않습니다.
4만 경로의 개별 3개 시드에서도 IV−HV MAPE 변화는 **{sg.loc[sg.paths.eq(40000),'mape_change_iv_minus_hv_pp'].min():+.4f}~{sg.loc[sg.paths.eq(40000),'mape_change_iv_minus_hv_pp'].max():+.4f}%p**였습니다.

이 수치는 관측 공정가에 대한 MC 가격의 차이이며, DeepONet의 예측오차 또는 검증된 시장 정답에 대한 오차는 아닙니다.
IV와 HV 중 어느 쪽이 공정가에 가까운지와 실무 가격모형으로서 어느 쪽이 타당한지는 구분해서 해석해야 합니다.

## 계산 기준

- 시드당 **40,000경로 × 3개 시드**. HV·IV 모두 같은 난수 시드와 경로 분할을 사용했습니다.
- HV는 기존 `module/features.py`의 `vol180`을 사용했습니다. 발행일보다 앞선 180개 로그수익률의 표본표준편차 × √252이며, 이번 표본에는 모두 180개가 존재합니다.
- IV는 앞서 적용한 발행일 당일 또는 이전 최신 ATM IV이며 최대 7달력일 이내입니다. 단일 변동성이고 기간구조·스마일은 없습니다.
- 각 상품의 공정가·HV MC·IV MC 모두 같은 실제 발행가로 나눴습니다. 왼쪽 금액은 다시 10,000을 곱해 발행금액 1만 원 기준으로 표시했습니다.
- `pd.qcut(FAIR/발행가, 6)`으로 같은 6분위 구간을 두 군에 적용했습니다.
- 왼쪽: 구간 내 평균 `(fair − mc) × 10,000`. 오른쪽: 구간 내 평균 `abs(fair − mc) / abs(fair) × 100`.
- 같은 표본과 구간을 사용하므로 IV 막대는 [기존 IV 단독 분석](../fair_mc_iv_population_20260914/REPORT.md)의 수치와 같습니다.
- 지급일 추정, 액면 10,000원 가정, 개별 공시 대조 6개라는 범위와 공정가·MC 기준일 차이 가능성도 기존 분석과 동일합니다.
- 두 군의 경로별 교차곱은 저장하지 않았으므로 paired pathwise 표준오차를 산출했다고 주장하지 않습니다. 시드별 비교값을 별도로 저장했습니다.

HV의 3시드 평균 가격에서 2만→4만 경로 평균 절대 변화는 액면 기준 **{convergence['hv_mean_abs_20k_to_40k_face_krw']:.2f}원**입니다.

## 저장 파일과 재현

- `prepare.py`: 기존 IV 표본과 계약을 고정하고 HV 입력만 추가합니다.
- `run.py`: 기존 IV 실행 코드의 MC 호출을 재사용해 HV만 계산하며 동일 시드를 검증합니다.
- `report.py`: 동일 6분위의 그림·표·해설을 저장된 계산값에서 생성합니다.
- `results/fair_mc_hv_iv.csv`: 상품별 공정가와 두 MC 가격, 가격 차이 및 상대오차.
- `results/volatility_assignment.csv`: 시장군·기초자산별 HV, IV, HV 수익률 기간.
- `results/seed_checkpoints_hv_iv.csv.gz`: 상품·시드·경로 수별 두 군의 원시 합계·제곱합과 가격 차이.
- `results/seed_summary.csv`: 3개 시드와 1만/2만/4만 경로별 집계. 본문의 3시드 평균 가격 기준 집계와 구별합니다.
- `results/price_segments_hv_iv.csv`: 그림의 정확한 막대값.
- `results/pre_mc_verification.json`, `results/verification.json`: 동일 시드, 기존 IV 재계산 일치, 계약 엔진·정규화·구간별 검증.
- `figures/`: PNG, SVG, PDF. STEP KI 10,002개만의 비교도 별도 저장했습니다.

기존 IV 분석이 완료된 저장소에서 실행합니다. `requirements.txt`의 의존성이 필요합니다.
이번 실행은 Python 3.11 / PyTorch 2.9.1 CPU 환경입니다.

```bash
python scripts/restore_artifacts.py
python analysis/fair_mc_hv_iv_population_20260914/prepare.py
python analysis/fair_mc_hv_iv_population_20260914/run.py --workers 48
python analysis/fair_mc_hv_iv_population_20260914/report.py
```

계산된 결과로 그림만 다시 만들 때는 마지막 `report.py`만 실행합니다.
'''
    (HERE/'REPORT.md').write_text(text)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
