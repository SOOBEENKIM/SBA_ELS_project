"""PPT-style FAIR sextiles and FAIR-minus-IV-MC histogram, IV only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';FIG=HERE/'figures'

def md(rows):
    h=list(rows[0]);s='| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'
    for r in rows:s+='| '+' | '.join(f'{r[k]:,.2f}' if isinstance(r[k],float) else str(r[k]) for k in h)+' |\n'
    return s

def make_figures(d,suffix=''):
    assert not d.item.duplicated().any()
    z=d.copy();z['price_seg']=pd.qcut(z.fair,6)
    g=z.groupby('price_seg',observed=True).agg(n=('item','size'),fair_mean=('fair','mean'),mc_mean=('mc_iv','mean'),
        err_mean_KRW=('gap_krw','mean'),err_median_KRW=('gap_krw','median'),MAPE_pct=('ape_pct','mean'))
    assert len(g)==6 and g.n.sum()==len(d)
    g.to_csv(OUT/f'price_segments{suffix}.csv')
    labels=[f'{iv.left:.3f}–{iv.right:.3f}' for iv in g.index];x=np.arange(6)
    fig,ax=plt.subplots(1,2,figsize=(13.333,4.9),layout='constrained')
    ax[0].bar(x,g.err_mean_KRW,color=np.where(g.err_mean_KRW>0,'#e07a5f','#1f6f9c'),edgecolor='white',linewidth=.5)
    ax[1].bar(x,g.MAPE_pct,color='#2a9d8f',edgecolor='white',linewidth=.5)
    ax[0].axhline(0,color='#444',lw=.8)
    ax[0].set(ylabel='Mean (FAIR − IV MC) [KRW]',title='Pricing gap (FAIR − IV MC) by price segment')
    ax[1].set(ylabel='MAPE (%)',title='|FAIR − IV MC| / FAIR by price segment')
    for a in ax:
        a.set_xticks(x,[f'{v}\nn={n:,}' for v,n in zip(labels,g.n)],rotation=25,ha='right',fontsize=9)
        a.set_xlabel('FAIR / issue price segment (low → high)')
        a.grid(axis='y',color='#dddddd',lw=.5,alpha=.65)
    title=f'IV-based MC · {len(d):,} actual products'
    if suffix:title+=' · STEP KI only'
    fig.suptitle(title+' · both prices normalized to KRW 10,000 invested',fontsize=12)
    fig.savefig(FIG/f'fair_mc_by_price_segment{suffix}.png',dpi=220,bbox_inches='tight');plt.close(fig)
    diff=d.gap_krw
    fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
    ax.hist(diff,bins=90,color='#5aa9dd',edgecolor='white',linewidth=.3)
    ax.axvline(diff.mean(),color='#c0392b',lw=2,label=f'Mean {diff.mean():,.0f} KRW')
    ax.axvline(diff.median(),color='#c0392b',lw=1.2,ls='--',label=f'Median {diff.median():,.0f} KRW')
    ax.axvline(0,color='#888',lw=1,ls=':')
    ax.set(xlabel='FAIR − IV MC (KRW, both prices issue-price normalized)',ylabel='Number of actual products',
           title=f'FAIR minus IV MC theoretical price · n={len(d):,}')
    ax.grid(axis='y',color='#ddd',lw=.5,alpha=.6);ax.legend(frameon=False)
    fig.savefig(FIG/f'fair_minus_mc_hist{suffix}.png',dpi=220,bbox_inches='tight');plt.close(fig)
    return g


def main():
    complete=json.loads((OUT/'completion.json').read_text());assert complete['status']=='complete'
    cfg=json.loads((HERE/'protocol.json').read_text());audit=json.loads((OUT/'input_audit.json').read_text())
    d=pd.read_csv(OUT/'fair_mc_iv.csv');assert len(d)==cfg['included']==complete['products']
    np.testing.assert_allclose(d.gap_krw,(d.fair_raw_krw-d.mc_face_krw)/d.issue_price_krw*10000)
    np.testing.assert_allclose(d.ape_pct,abs(d.fair_raw_krw-d.mc_face_krw)/d.fair_raw_krw*100)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'font.family':'DejaVu Sans','font.size':11})
    g=make_figures(d)
    stepki=d[d.structure.eq('STEP KI')]
    if len(stepki)>=60:make_figures(stepki,'_step_ki')
    summary=dict(products=len(d),mean_gap_krw=float(d.gap_krw.mean()),median_gap_krw=float(d.gap_krw.median()),
        mae_krw=float(abs(d.gap_krw).mean()),rmse_krw=float(np.sqrt(np.mean(d.gap_krw**2))),mape_pct=float(d.ape_pct.mean()),
        low_fair_bin_mape=float(g.MAPE_pct.iloc[0]),high_fair_bin_mape=float(g.MAPE_pct.iloc[-1]),
        mape_strictly_decreasing_across_six_bins=bool(np.all(np.diff(g.MAPE_pct)<0)),
        mc_above_fair_fraction=float(d.gap_krw.lt(0).mean()))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    checkpoints=pd.read_csv(OUT/'mc_checkpoints.csv.gz')
    pivot=checkpoints.pivot(index='item',columns='paths_per_seed',values='mc_face_krw')
    convergence=dict(mean_abs_change_20k_to_40k=float(abs(pivot[40000]-pivot[20000]).mean()),
        median_price_se_krw=float(d.mc_se_face_krw.median()),p95_price_se_krw=float(d.mc_se_face_krw.quantile(.95)))
    (OUT/'convergence_summary.json').write_text(json.dumps(convergence,indent=2)+'\n')
    rows=[]
    for i,(iv,r) in enumerate(g.iterrows(),1):
        rows.append({'공정가 구간':f'{iv.left:.3f}–{iv.right:.3f}','상품 수':int(r.n),
                     '평균 FAIR−MC(원)':float(r.err_mean_KRW),'MAPE(%)':float(r.MAPE_pct)})
    report=f'''# IV MC와 공정가의 가격구간별 분석

현재 IV 브랜치의 MC 엔진으로 실제 상품 {len(d):,}개의 원계약 가격을 계산하고,
동일 상품의 공시 공정가와 비교했습니다. MC 입력은 ATM IV이며, HV 모델이나 DeepONet 예측값을 섞지 않았습니다.
합성 계약의 조건 변경 실험과 별개로, 이 분석에는 변경 전 원계약만 포함합니다.

## 계산과 그림

- 발행가 정규화: `fair = 원문 FAIR_VALUE / 실제 발행가`, `mc_iv = 액면 10,000원 MC 가격 / 실제 발행가`.
- 차이(원): `(fair − mc_iv) × 10,000`. 두 값 모두 같은 발행금액 기준입니다.
- 상대오차(%): `abs(fair − mc_iv) / abs(fair) × 100`.
- 공정가를 `pd.qcut(fair, 6)`으로 6분위로 나누어 각 구간의 평균 차이와 평균 상대오차를 그렸습니다.
  구간당 상품 수가 비슷하며 가격 구간의 폭은 서로 다릅니다.
- 히스토그램은 상품별 차이를 90개 bin으로 집계하고 평균·중앙값을 표시합니다.
- 상품별 시드당 40,000경로 × 3개 시드. 동일 시장·발행일의 상품끼리는 경로를 재사용하되,
  각 상품의 관측일·지급일·만기·지급조건은 별도로 적용했습니다.

![공정가 구간별 FAIR−IV MC와 상대오차](figures/fair_mc_by_price_segment.png)

{md(rows)}

![FAIR−IV MC 분포](figures/fair_minus_mc_hist.png)

평균 차이는 **{summary['mean_gap_krw']:,.2f}원**, 중앙값은 **{summary['median_gap_krw']:,.2f}원**,
전체 MAPE는 **{summary['mape_pct']:.2f}%**입니다.
가장 낮은 공정가 구간의 MAPE는 {summary['low_fair_bin_mape']:.2f}%, 가장 높은 구간은 {summary['high_fair_bin_mape']:.2f}%입니다.
6개 구간에서 MAPE가 순서대로 모두 감소했는지: **{summary['mape_strictly_decreasing_across_six_bins']}**.
음의 FAIR−MC는 MC 가격이 공시 공정가보다 높다는 뜻입니다. 이 수치는 DeepONet 예측오차가 아닙니다.

## 분석 대상과 제외

원래 표본 ID {cfg['universe']:,}개 중 {cfg['included']:,}개를 계산했고 {cfg['excluded']:,}개를 제외했습니다.
선택은 MC 결과를 보기 전에 원문 필드와 IV 존재 여부로 결정했습니다. 상품별 사유는 results/excluded_products.csv에 있습니다.

{md([{'사유':k,'건수':v} for k,v in audit['excluded_reason_counts'].items()])}

{md([{'구조':r['structure'],'월지급':r['monthly'],'건수':r['n']} for r in audit['structure_counts']])}

개별 공시 대조까지 한 상품은 {audit['published_included']}개이며, 나머지는 원문 데이터의 일정·지급률을 사용한 모형 진단입니다.
공시 전문을 전량 대조한 가격이라는 뜻은 아닙니다.

## 적용 가정과 범위

1. IV는 각 기초자산의 발행일 이전 또는 당일 최신 관측값을 사용하고, 7일 초과 또는 하나라도 누락되면 제외했습니다.
   역사적 변동성을 IV 대신 채우지 않았습니다. 상관계수는 발행 전 180개 정렬 수익률, 금리는 기존 KRW NS 곡선입니다.
2. 기존 V2/V3 지급 엔진과 V4 공통경로 수집 코드를 그대로 호출했습니다. 만기 생존 쿠폰은 원문 PMT_2로,
   월 쿠폰은 별도 일정의 PMT_1로 구성했습니다. 추정되지 않는 추가 지급식·평균가격·미지원 조건은 제외했습니다.
3. 공시 대조 상품 이외의 지급일은 마지막 평가일→원문 만기일에 맞는 유일한 1~5 은행영업일 지연을 구하고,
   동일 지연을 다른 지급일에도 적용했습니다. 상품별 약관으로 모두 확정한 지급 규칙은 아닙니다.
4. 액면 10,000원을 가정하고 발행가 9,000~10,000원 상품을 포함했습니다. 개별 공시 대조 상품 외에는 액면을 전량 대조하지 않았습니다.
   발행가 정규화는 FAIR와 MC 모두에 동일하게 적용했습니다.
5. GBM·일별 모니터링·배당률 0·단일 ATM IV 가정은 IV 브랜치와 같습니다. 변동성 표면·만기구조·quanto 보정은 없습니다.
6. 공시 FAIR 기준일과 MC에 사용한 발행일 시장자료가 다를 수 있습니다. 따라서 관측 차이 전체를 MC 오류나 한 요인의 인과효과로 해석하지 않습니다.

3개 시드 평균 MC의 2만→4만 경로 평균 절대 가격 변화는 액면 기준 {convergence['mean_abs_change_20k_to_40k']:.2f}원입니다.
상품별 MC 표준오차의 중앙값은 {convergence['median_price_se_krw']:.2f}원, 95분위는 {convergence['p95_price_se_krw']:.2f}원입니다.

## 재현 파일

- prepare.py: 원문 계약과 시장자료 연결, 사전 제외 기준, protocol.json과 입력 해시 생성.
- run.py: 고정 MC 엔진 호출, 3개 시드 및 1만/2만/4만 체크포인트, 중단 후 동일 작업 재개.
- report.py: FAIR 6분위 집계, 히스토그램과 현재 문서 생성.
- results/fair_mc_iv.csv: 상품별 FAIR, IV MC, 차이, 상대오차.
- results/price_segments.csv: 그림의 6개 막대에 대응하는 정확한 집계값.
- results/mc_seed_checkpoints.csv.gz: 상품·시드·경로 수별 원시 합계와 제곱합.

저장소 최상위에서 기존 Python 의존성을 설치한 환경으로 실행합니다. 앞의 두 명령은 저장된 원문 자료와 IV ZIP을 복원합니다. 이미 계산된 결과로 그림만 다시 만들려면 `report.py`만 실행하면 됩니다.

```bash
python scripts/restore_artifacts.py
python analysis/iv_validation_20260914/restore_input.py
python analysis/fair_mc_iv_population_20260914/prepare.py
python analysis/fair_mc_iv_population_20260914/run.py --workers 16
python analysis/fair_mc_iv_population_20260914/report.py
python analysis/fair_mc_iv_population_20260914/verify_results.py
```
'''
    if len(stepki)>=60:
        report+=f'\nSTEP KI에 한정한 {len(stepki):,}개 표본의 동일 집계도 별도로 저장했습니다.\n\n![STEP KI 구간별 결과](figures/fair_mc_by_price_segment_step_ki.png)\n'
    (HERE/'REPORT.md').write_text(report)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
