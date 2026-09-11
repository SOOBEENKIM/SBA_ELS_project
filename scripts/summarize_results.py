"""Build the final README section directly from completed MC/model result tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'analysis/mc_monthly_v3_20260910/results';B=ROOT/'analysis/mc_schedule_v4_20260911/results'
def table(cols,rows):
 return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
 labels=pd.read_csv(A/'mc_labels.csv');metrics=pd.read_csv(A/'model_metrics.csv');sm=pd.read_csv(B/'model_metrics.csv');effects=pd.read_csv(B/'mc_effect_summary.csv')
 rows=[]
 for axis,h,title in [('coupon_regular',.01,'연 쿠폰 +1%p'),('ki_barrier',.05,'낙인 배리어 +5%p'),('first_strike',.05,'1차 행사가 +5%p'),('last_strike',.05,'만기 행사가 +5%p'),('monthly_barrier',.05,'월 쿠폰 지급 배리어 +5%p')]:
  row=[title]
  for origin in ['synthetic_regular','synthetic_monthly']:
   q=labels[labels.origin.eq(origin)&labels.split.eq('test')&labels.axis.eq(axis)&np.isclose(labels.offset,h)]
   row.append('해당 없음' if q.empty else f'{q.delta_krw.mean():+.2f}원 ({q.family.nunique()}개)')
  rows.append(row)
 primary='affine_coupon_delta'
 modelrows=[]
 for cohort,title in [('regular_test','일반형'),('monthly_test','월지급형')]:
  for arm,name in [('augmented_price','합성 가격 손실'),('augmented_delta','가격 + 증분 손실'),(primary,'가격 + 증분 손실 + 쿠폰 선형 구조 (선택)')]:
   r=metrics[metrics.cohort.eq(cohort)&metrics.model.eq(arm)&metrics.seed.eq('ensemble')&metrics.axis.eq('ALL')].iloc[0]
   modelrows.append([title,name,f'{r.price_r2:.4f}',f'{r.price_mae:.2f}',f'{r.delta_mae:.2f}',f'{r.resolved_sign*100:.2f}%'])
 sr=[]
 for axis,h,title in [('nobs_early',1,'관측 +1회: 첫 평가 전'),('nobs_middle',1,'관측 +1회: 중간'),('nobs_late',1,'관측 +1회: 마지막 구간')]+[('tenor_months',h,f'만기 {h:+d}개월') for h in [-12,-6,-3,-1,1,3,6,12]]:
  rr=[title]
  for cohort in ['regular_test_128','monthly_test_128']:
   x=effects[effects.cohort.eq(cohort)&effects.axis.eq(axis)&effects.offset.eq(h)].iloc[0]
   y=sm[sm.cohort.eq(cohort)&sm.model.eq(primary)&sm.training_seed.eq('ensemble')&sm.axis.eq(axis)&sm.offset.eq(h)].iloc[0]
   rr.extend([f'{x["mean"]:+.2f}',f'{y.delta_mae:.2f}'])
  sr.append(rr)
 text='''## 최종 실험 결과 (2026-09-11까지)

아래 결과는 수정된 계약 현금흐름을 사용한 실제 실행 결과다. 과거 잘못된 MC 구현의 수치는 제외했다. **모든 원화 금액은 액면 10,000원 기준**이며, 실제 관측 공정가의 인과효과가 아니라 **정의된 MC 모형 안에서 계약조건을 변경한 반사실 가격 차이**다. 쿠폰·배리어·행사가 실험은 계약별 시장·일정·나머지 조건을 고정하고, 관측 횟수·만기 실험은 명시된 규칙에 따라 종속 일정과 쿠폰 이자기간도 조정했다.

### 완료된 실행 범위

- 계약조건 실험: 기준 계약 2,652개, 기준·변경 시나리오 85,440개. 학습 2,048개 / 검증 256개 / 시험 256개 / 진단 92개(공시 확인 월지급형 6개 + 이전 참고 계약 86개).
- 일반형·월지급형 각각 시험 128개. 월지급형은 계약 템플릿 그룹을 분리하고 공시 비교군의 템플릿도 학습에서 제외했다.
- MC는 **시드당 40,000경로**. 학습·검증은 1시드, 시험·진단은 3개 독립 시드(계약당 합계 120,000경로). 10,000→20,000→40,000은 같은 실행의 누적 체크포인트다.
- 세 학습 방식 × 학습 시드 47·101·233 = 9개 DeepONet. 모델 선택은 검증 자료로 끝냈고 선택 방식의 3시드 평균을 사용했다. PI 손실과 Stage 2는 사용하지 않았다.
- 관측 횟수·만기 추가 평가: 기존 시험 256개 + 공시 6개, 총 262개 기준 계약·3,003개 시나리오. 모든 계약에 40,000경로 × 3시드. 위의 9개 모델을 재학습 없이 평가했다. 이미 살펴본 시험군의 진단 확장이므로 새 독립 최종시험이라고 부르지 않는다.

### 조건별 MC 가격 변화

계약별 증분의 단순평균이다. 가격 변화량과 예측 오차는 서로 다른 수치다.

'''+table(['변경 조건','일반형 평균 ΔMC','월지급형 평균 ΔMC'],rows)+'''

쿠폰 증가의 양(+) 효과와 KI 배리어 증가의 음(-) 효과를 확인했다. 1차 행사가의 평균 효과는 음(-)이지만 양(+)인 계약도 있다. 원금 회수 시점과 추가 쿠폰 지급 효과가 달라 계약별로 증분의 방향과 크기가 달라졌다.

### DeepONet의 가격·증분 예측 성능

아래 가격 지표는 시험 계약의 기준·변경 가격 전체, 증분 지표는 변경 시나리오에서 계산했다. 각 행은 해당 방식의 3개 학습 시드 평균 예측이다. 방향 일치율은 |ΔMC| > 1.96 × paired MC SE이면서 영향 경로가 30개 이상인 경우만 평가했다.

'''+table(['상품','학습 방식','가격 R²','가격 MAE(원)','증분 MAE(원)','MC 방향 일치율'],modelrows)+'''

같은 합성 데이터에서 가격만 학습한 경우보다 선택 모델의 평균 증분 오차는 일반형 **14.92→7.43원**, 월지급형 **11.70→6.41원**으로 감소했다. 월지급형 증분 MAE만 보면 가격+증분 손실 모델(6.19원)이 선택 모델보다 작다. 선택은 전체 검증 점수로 결정했으며 시험 결과에 따라 바꾸지 않았다. 이 실험에는 새로운 데이터의 기준 가격만 학습한 별도 대조군이 없어, 합성 자료 추가 효과를 손실 변경 효과와 독립적으로 입증한 것은 아니다.

### 관측 횟수·만기 변경에 대한 추가 평가

관측 +1회의 위치별 정의와 만기 변경에 따른 지급·이자기간 규칙은 [실험 과정](docs/EXPERIMENT.md)에 설명했다. 관측 +1회는 기존 12회 계약을 제외하여 일반형 81개·월지급형 128개에서, 만기 변경은 각각 128개에서 평가했다. 공시 6개는 별도 진단 결과 파일에 포함되어 있다.

'''+table(['변경 조건','일반형 평균 ΔMC(원)','일반형 증분 MAE(원)','월지급형 평균 ΔMC(원)','월지급형 증분 MAE(원)'],sr)+'''

![관측 횟수·만기 변경의 MC 증분과 선택 모델 예측](docs/assets/schedule_counterfactuals.png)

### 결론과 남은 실패

**일부 축의 평균 증분 오차는 개선됐지만, 모든 계약조건 변화에서 가격 증분을 안정적으로 예측한다는 결과는 아니다.**

- 공시 확인 월지급형 32410의 1차 행사가 +5%p: MC **−9.55원**, 선택 DeepONet **+9.44원**으로 방향이 달랐다. 독립 NumPy MC에서도 **−9.51원**을 확인했다.
- 월지급형에서 관측을 중간·마지막 구간에 1회 추가하면 MC는 평균 **+8.82원·+9.66원**, 모델은 **−375.97원·−429.40원**을 예측했다. 평균 증분 오차는 **384.78원·439.06원**이다.
- 일정 추가 실험은 동일 가중치의 새로운 평가 축이므로, 앞의 6.41원과 직접 비교해 학습 후 성능이 퇴보했다고 해석하지 않는다. 학습 표본에 부족한 일정에 대한 일반화 실패가 확인된 것이다. 분포 부족만이 유일한 원인이라고 입증한 것은 아니다.
- 독립 현금흐름 계산·입력/정규화·모델 재추론을 점검했지만 산업용 pricer 전체 검증을 완료했다는 뜻은 아니다. 현재 GBM·역사 변동성/상관·q=0·일별 달력 격자 가정에 따른 모형 오차가 남는다.
- 가격 R²는 일반형 0.9703, 월지급형 0.9224였다. 1/5/10원 이내 비율은 증분 예측 오차를 평가하기 위한 진단 지표다. **현재 결과에서는 모든 조건 변화에 걸친 가격 정확도와 증분 안정성이 확인되지 않았다.**
'''
 (ROOT/'docs').mkdir(exist_ok=True);(ROOT/'docs/RESULTS.md').write_text(text.replace('](docs/',']('))
 p=ROOT/'README.md'
 if p.exists():prefix=p.read_text().split('<!-- RESULTS -->')[0];p.write_text(prefix+'<!-- RESULTS -->\n\n'+text)
 experiment=ROOT/'docs/EXPERIMENT.md'
 if experiment.exists():
  methods=experiment.read_text().replace('# 최종 실험 과정','# Report 2 — 최종 MC·DeepONet·반사실 실험',1).replace('](RESULTS.md)','](docs/RESULTS.md)').replace('](../provenance/','](provenance/')
  (ROOT/'Report_2.md').write_text(methods+'\n\n'+text)
 print('RESULTS regenerated from MC labels and ensemble metrics',flush=True)
if __name__=='__main__':main()
