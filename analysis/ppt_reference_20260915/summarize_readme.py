"""Write a compact, result-derived summary; update root README only explicitly."""
from pathlib import Path
import argparse,json
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
START='<!-- PPT_REFERENCE_RESULTS -->';END='<!-- /PPT_REFERENCE_RESULTS -->'
NAME={'reference':'원본 지급 근사','detailed':'상세 지급 구현'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--update-root',action='store_true');args=parser.parse_args()
    assert json.loads((OUT/'final_verification.json').read_text())['status']=='pass'
    prefix=HERE.relative_to(ROOT).as_posix()
    selections={p:json.loads((HERE/'synthetic'/p/'results/model_selection.json').read_text())['selected_arm'] for p in ['reference','detailed']}
    lines=['## 발표 기준 재현과 두 지급 구현 비교 (2026-09-15)','',
      '발표 그림을 만든 원본 코드·저장 표본·시장 입력을 확인해 별도 실험으로 재현했다. 기존 main과 앞선 IV 실험은 보존했다. 학습 타깃은 MC 이론가이며 Stage 2는 사용하지 않았다.','',
      '- 원본 그림의 표본은 앞부분 23,151개, 이후 HV·IV 비교 58,790개였다. 저장 가격을 재집계해 평균 차이 −507원 및 HV −556원 / IV −520원을 확인했다.',
      '- 제공된 ATM IV CSV 2,404개가 원본에서 사용한 파일과 모두 일치했다. 같은 58,790개 상품에서 원본 지급 근사로 HV·IV 가격을 4만 경로 재계산했다.',
      '- 기준 계약 2,652개·기준 및 변경 시나리오 88,181개에서 원본 근사와 상세 지급 구현을 같은 시장 입력·경로로 비교했다. 합성 학습·검증은 4만 경로×1시드, 시험·진단은 4만 경로×3시드다.',
      '- 두 지급 구현 각각 가격 손실 / 가격+증분 손실 / 가격+증분 손실+쿠폰 선형 구조를 3시드로 학습해 총 18개 DeepONet을 평가했다.','',
      '### 실제 상품 58,790개 재계산','',
      '| MC 설정 | 평균 FAIR−MC | MAE | MAPE |','|---|---:|---:|---:|']
    pop=json.loads((OUT/'recomputed_plot_summary.json').read_text())
    for name,row in [*pop['hv_iv'].items(),('IV + NS',pop['iv_discount']['IV + NS'])]:
        lines.append(f"| {name} | {row['mean']:.2f} | {row['mae']:.2f} | {row['mape']:.2f}% |")
    lines += ['', '위 차이는 원본 그림의 정규화 식 `(FAIR/발행가 − 단위 액면 MC)×10,000`을 재현한 값이다. 발행가와 액면가가 다른 경우 완전히 같은 금액 기준은 아니다. 원본 IV 그림의 할인 구현은 세 금리점의 선형 제로금리 보간이며, IV+NS 대조군을 추가했다.','',
      f'![원본 조건 HV·IV 비교]({prefix}/figures/recomputed_hv_iv_segments.png)','',
      '### 동일 합성 계약의 평균 MC 증분','',
      '| 상품 | 변경 | 원본 지급 근사 | 상세 지급 구현 |','|---|---|---:|---:|']
    e=pd.read_csv(OUT/'payoff_effect_comparison.csv')
    for r in e[~e.axis.isin(['tenor_months','nobs_early','nobs_middle','nobs_late'])].to_dict('records'):
        lines.append(f"| {r['상품']} | {r['변경']} | {r['원본 근사 ΔMC']:+.2f}원 | {r['상세 지급 ΔMC']:+.2f}원 |")
    lines += ['', '### 선택 DeepONet의 시험 성능','',
      '검증 점수에 따른 선택: '+', '.join(f'{NAME[p]} `{m}`' for p,m in selections.items())+'. 가격·증분 지표는 쿠폰·배리어·행사가 변경 시험에 대한 3개 학습 시드 평균 예측이다.','',
      '| 지급 구현 | 상품 | 가격 R² | 가격 MAE | 증분 MAE: 가격만 학습→선택 모델 | 방향 일치율 | 일정 변경 증분 MAE |','|---|---|---:|---:|---:|---:|---:|']
    metrics=pd.read_csv(OUT/'model_metrics.csv');metrics=metrics[metrics.seed.eq('ensemble')&metrics.axis.eq('ALL')]
    for payoff in ['reference','detailed']:
        selected=selections[payoff]
        for cohort,label in [('regular_test','일반형'),('monthly_test','월지급형')]:
            m=metrics[metrics.payoff.eq(payoff)&metrics.cohort.eq(cohort)]
            t=m[m.model.eq(selected)&m.scope.eq('terms')].iloc[0];s=m[m.model.eq(selected)&m.scope.eq('schedule')].iloc[0];old=m[m.model.eq('augmented_price')&m.scope.eq('terms')].iloc[0]
            lines.append(f'| {NAME[payoff]} | {label} | {t.price_r2:.4f} | {t.price_mae:.2f}원 | {old.delta_mae:.2f}→{t.delta_mae:.2f}원 | {100*t.resolved_sign:.2f}% | {s.delta_mae:.2f}원 |')
    lines += ['', '계약조건 증분의 평균 오차는 감소했지만 방향 불일치와 관측 횟수·만기 변경의 큰 오차가 남았다. 모든 상품·변경 폭에서 증분을 안정적으로 예측한다고 결론 내릴 수 없다. 원본 근사에서는 월 쿠폰 배리어가 가격에 반영되지 않으므로 해당 MC 증분 0원을 정확한 월지급 구현의 결과로 해석하지 않는다.', '',
      f'전체 설정·표·민감도·예측 그림은 [실험 보고서]({prefix}/REPORT.md), 데이터부터 다시 실행하거나 저장된 결과로 그림만 만드는 방법은 [실행 안내]({prefix}/EXECUTION.md)에 있다.','']
    summary='\n'.join(lines)
    (HERE/'SUMMARY.md').write_text(summary.replace(prefix+'/',''))
    if args.update_root:
        path=ROOT/'README.md';old=path.read_text();block=START+'\n\n'+summary+'\n'+END
        if START in old:
            start=old.index(START);end=old.index(END,start)+len(END);old=old[:start]+block+old[end:]
        else:old=old.rstrip()+'\n\n'+block+'\n'
        path.write_text(old)
    print('Summary written; root README updated:',args.update_root)

if __name__=='__main__':main()
