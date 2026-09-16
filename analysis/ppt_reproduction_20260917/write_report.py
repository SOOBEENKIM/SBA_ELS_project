"""Create the report from measured results and notebook-embedded reference text."""
from common import *
import re

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
 level=pd.read_csv(OUT/'fair_mc_summary.csv').set_index('variant');r=pd.read_csv(OUT/'stage1_by_seed.csv');agg=r.groupby(['model','variant'])[['R2','MAE','RMSE','MAPE']].agg(['mean','std']);ver=json.loads((OUT/'verification.json').read_text());curve=json.loads((OUT/'curve_validation.json').read_text())
 nb=json.loads((REF/'18_stage1_mc_variants.ipynb').read_text());out='\n'.join(''.join(o.get('text',[])) for c in nb['cells'] for o in c.get('outputs',[]));block=out.split('=== 5시드 평균 ± 표준편차')[1].split('=== 시드별 R²')[0];refs=[];model=None
 for line in block.splitlines():
  cells=line.split()
  if cells and cells[0] in ('deeponet','xgb'):model=cells.pop(0)
  if cells and cells[0] in ('A','B') and len(cells)==9:
   var=cells[0];nums=list(map(float,cells[1:]));refs.append(dict(model=model,variant=var,**{metric:nums[j*2] for j,metric in enumerate(['R2','MAE','RMSE','MAPE'])}))
 ref=pd.DataFrame(refs).set_index(['model','variant']);assert len(ref)==4
 comparisons=[]
 for (m,x),row in ref.iterrows():
  for metric in ['R2','MAE','RMSE','MAPE']:comparisons.append(dict(model=m,variant=x,metric=metric,notebook_reference=float(row[metric]),rerun_mean=float(agg.loc[(m,x),(metric,'mean')]),rerun_std=float(agg.loc[(m,x),(metric,'std')]),delta=float(agg.loc[(m,x),(metric,'mean')]-row[metric])))
 pd.DataFrame(comparisons).to_csv(OUT/'notebook_vs_rerun.csv',index=False)
 measured=table(['지표','A: NS·HV120·무배당','B: 부트스트랩·EWMA120·이산배당'],[[k,f'{level.loc["A",col]:,.4f}',f'{level.loc["B",col]:,.4f}'] for k,col in [('공정가−MC 평균 (원)','mean'),('MAE (원)','MAE'),('RMSE (원)','RMSE'),('MAPE (%)','MAPE'),('차이의 표준편차 (원)','sd')]])
 learned=table(['모델','사양','노트북 R²','재실행 R² 평균 ± SD','노트북 MAPE (%)','재실행 MAPE (%) 평균 ± SD'],[[m,x,f'{ref.loc[(m,x),"R2"]:.4f}',f'{agg.loc[(m,x),("R2","mean")]:.4f} ± {agg.loc[(m,x),("R2","std")]:.4f}',f'{ref.loc[(m,x),"MAPE"]:.4f}',f'{agg.loc[(m,x),("MAPE","mean")]:.4f} ± {agg.loc[(m,x),("MAPE","std")]:.4f}'] for m,x in agg.index])
 audits=[]
 for device in ['cuda','cpu']:
  path=OUT/f'mc_reprice_audit_{device}.csv'
  if not path.exists() and device=='cuda':path=OUT/'mc_reprice_audit.csv'
  if path.exists():
   q=pd.read_csv(path)
   for x,g in q.groupby('variant'):audits.append([device,x,len(g),f'{g.difference_KRW.abs().mean():.4f}',f'{g.difference_KRW.abs().max():.4f}',f'{g.q_maxdiff.max():.2e}',bool((g.n_div==g.n_div_cached).all())])
 report=f'''# 2026-09-17 발표 실험 재현

제공된 `main`과 `feat/4-structure-universe` ZIP을 비교하고 최신 branch의 필요한 파일만 별도 `reference/`에 보존했다. 기존 main과 이전 실험 코드·결과를 덮어쓰지 않았다. 작업 branch는 `codex/ppt-pricer-reproduction-20260917`이다.

## 실행 범위와 재현 수준

| 작업 | 실행 결과 | 성격 |
|---|---|---|
| ZIP 비교 및 필요한 파일 반입 | 153개 파일, 해시 검증 | 원본 수정 없음 |
| 전체 MC 가격 분포·6분위·연도별 분석 | 58,760개 | 제공된 MC 캐시 재집계; 새로운 전체 MC 계산 아님 |
| Stage 1 DeepONet·XGBoost A/B | 각 5시드·4폴드, 총 80개 가중치 | 실제 재학습 완료, MC만 학습·평가 |
| 부트스트랩 금리곡선 | 200개 상품 | 새로 계산하여 저장된 10노드 입력과 비교 |
| MC A/B 가격 재계산 | 12개 상품 × A/B × CPU/CUDA, 각 40,000경로 | 실제 새 MC 실행; 원본 캐시와 차이 존재 |
| 노트북 16·17의 배당 200개 원표본 | 미재현 | 표본 ID와 중간 가격 CSV가 ZIP에 없음 |
| 최신 B 기반 합성 계약·증분 학습 | 미실행 | 이전 실험 결과는 상위 branch에 보존, 최신 결과로 혼합하지 않음 |

**최신 가격 캐시에서 출발하는 분석과 Stage 1 재학습은 완료했다. 원천 데이터부터 원본과 동일한 MC 가격을 만드는 전 과정은 완전 재현으로 판정하지 않았다.** `scratch/mc_discrete_div.py`, `mc_boot_curve.py`, `mc_full_variants.py`, `stage1_mc_variants.py`, `mc_noise_check.py` 및 일부 결과 CSV는 두 ZIP 모두에 없다. 기존에 받은 scratch와 별개로 최신 노트북이 참조하는 파일들이다.

## 반입한 자료

- 최신 `module/`, `model/`, `util/`, `config.yaml`, 노트북 5·15~19를 원문대로 보존함.
- 배당 캐시 46개, 기초자산 종가, CD/FRED·산업은행 IRS 자료, `mc_variants.parquet`를 복사함. 현재 Yahoo에서 다시 내려받아 값이 바뀌는 일을 피하고 제공 스냅샷을 사용함.
- 파생 상품 데이터는 기존 58,790개 스냅샷을 사용함. 최신 MC 캐시와 상품 식별자 및 정렬 순서가 전부 일치하고, NS 만기 금리도 일치함. 최신 원천 생성 스크립트로 DART부터 재생성한 것은 아님.
- `import_manifest.json`에 ZIP 및 각 파일 SHA-256, main/branch 차이, 반입 대상을 기록함. main은 최신 배당·IRS·A/B 캐시를 포함하지 않아 실행 기준으로 혼합하지 않음.

## A/B 설정

| 항목 | A | B |
|---|---|---|
| MC 경로 | 상품별 40,000, 저장된 mc_seed | 동일 |
| 변동성 | 발행 전 자산별 120거래일 HV | 같은 120거래일 EWMA λ=0.99 |
| 상관 | 한국 120거래일 창의 쌍별 Pearson | 동일 |
| 금리 | NS λ=1.5, 콜·3개월·국고10년 | 직전 영업일 IRS 및 직전 완결월 CD91, log DF 보간 |
| 배당 | 없음 | 이산배당, DAX는 0 |
| Stage 1 입력 | 곡선 10 + VC 7 + 계약 51 | 곡선 10 + VC 10(배당 q 3 포함) + 계약 51 |
| 예측 타깃 | mc_A | mc_B |

이 최신 A/B 비교는 IV 실험과 다르다. 양쪽 모두 역사적 수익률 기반 변동성을 사용한다. 이전 IV 실험 자료는 그대로 보존했다. Stage 2는 호출하지 않는다.

전체 {ver['OOS_rows_per_run']:,}개 OOS는 원래 58,790행에서 60/70/80/90/100% 경계를 만든 뒤 각 구간에서 부적합 행을 제거한 결과다. 오래된 종가 30개를 제외하며 이 중 18개는 MC 입력 실패도 겹친다. 30개와 18개를 따로 합산하지 않는다. train 내 12% validation과 seed=0을 사용하고, 학습 시드는 0~4이다. 동일 발행일이 경계 양쪽에 걸릴 수 있는 원본의 행 기준 분할을 유지했다.

`train_stage1.py`는 누락된 실행 파일 대신 작성한 연결 코드이다. 원본 `_anchor`, `_xgb_anchor`, `walk_forward`를 호출하며 학습 함수·네트워크·하이퍼파라미터는 바꾸지 않았다. XGBoost는 시간감쇠 가중을 적용하지만, 원본 `train_curve`에는 그 가중 적용이 없어 DeepONet은 무가중으로 그대로 실행했다. 노트북의 두 모델 모두 시간감쇠를 사용한다는 설명과 실제 코드의 차이이다.

## 공정가와 MC 분포 재집계

{measured}

평균 차이와 평균 절대 차이는 감소하고 분포 표준편차는 커진다. 공정가와의 일치가 MC 계약 구현의 정확성을 증명하는 것은 아니다.

원본과 동일하게 `fair = FAIR_VALUE / ISU_PRC_DETAIL`와 단위 액면 MC 차이에 10,000을 곱했다. 발행가와 액면가가 다른 상품까지 공통 현금 기준으로 재정규화한 숫자는 아니며, 이 재현에서 그 정의를 조용히 바꾸지 않았다.

![MC 차이 분포](figures/fair_mc_histogram.png)

![공정가 6분위별 차이](figures/fair_mc_by_segment.png)

![연도별 차이](figures/fair_mc_by_year.png)

## Stage 1 실제 재학습 결과

{learned}

노트북 참고 수치는 첨부 노트북의 저장된 출력에서 읽었고, 재실행 수치는 이번에 저장한 OOS 예측으로 다시 계산했다. 출력 참고 수치를 새 실행 결과로 복사하지 않았다. 두 결과는 가깝지만 동일하지 않다. 누락된 실행 래퍼, 실행 장치·라이브러리와 세부 입력 구성 차이가 완전히 해소되지 않았으므로 수치의 완전 일치는 주장하지 않는다.

B에서 R²가 높아져도 MAPE·원 단위 오차가 커질 수 있다. 서로 다른 MC 타깃 분포를 학습하므로 R² 증가만으로 전반적 정확도 향상이라고 결론내리지 않는다. 이 표는 가격 수준 예측이며 **계약조건 변경에 대한 증분 안정성 평가가 아니다.**

![Stage 1 R2](figures/stage1_r2.png)

![Stage 1 MAPE](figures/stage1_mape.png)

![Stage 1 오차 분포](figures/stage1_error_histogram.png)

시드별·폴드별 숫자는 `results/stage1_by_seed.csv`, `stage1_by_fold.csv`, 원본 출력과 비교는 `notebook_vs_rerun.csv`에 있다. 20회 평가에서 동일한 23,486개 상품과 MC 타깃을 확인했다. 저장된 80개 가중치의 폴드별 32개 상품 재예측을 검증했으며 최대 가격비율 차이는 {ver['max_prediction_difference']:.2e}이다.

## 금리곡선·MC 재계산 점검

공개된 노트북의 부트스트랩 식을 구현해 200개 고정 표본을 계산했다. 캐시의 B 10노드와 최대 차이 {curve['cache_max_abs_difference']:.2e}, 입력 IRS 재가격 최대 차이 {curve['par_reprice_max_bp']:.2e}bp로 일치했다. 이 200개는 유효 모집단에 등간격으로 선택한 감사 표본이며 원본 노트북 17의 누락된 표본 ID를 재현한 것이 아니다.

![금리곡선](figures/curve_comparison.png)

MC는 노트북 5의 `_setup`·`mc_daily_t` 함수 원문을 추출해서 호출했다. B에만 로그 배당락을 추가했다. 원본의 만기 미낙인 생존 시 원금 지급·월지급 누적근사를 유지했으며 기존 별도 상세 지급 엔진과 혼합하지 않았다. 이 원본 근사를 모든 실제 약관에 맞는 구현으로 인증하는 검증은 아니다.

{table(['장치','사양','상품','|새 MC−캐시| 평균 (원)','최대 (원)','q 최대 차이','배당 횟수 일치'],audits)}

배당 창은 마지막 과거 배당일로부터 360일을 가정했다. 원본 실행 파일이 없어 노트북의 ‘1년’이라는 설명을 확정적인 365일 코드로 볼 수 없었다. 점검한 12개에서는 q·이벤트 횟수가 맞지만 전체 배당 규칙의 동일성을 입증하지 못했다. q와 횟수 일치만으로 모든 배당일이 같다고 볼 수도 없다. CPU·CUDA 모두 MC 캐시와 가격 차이가 남았다. **이를 전부 난수 오차라고 단정하지 않으며, 원본 실행 코드·seed 사용 및 이벤트 구성 확인이 남아 있다.** 새 MC 값으로 제공 캐시를 덮어쓰거나 이 값을 최신 학습 타깃으로 교체하지 않았다.

## 실행 방법

저장소 루트, Python 3.11 및 `reference/requirements.txt`에 맞춘 환경에서 실행한다. 이번 실행은 기존 ELS 이미지의 torch 2.9.1+cu126, pandas 2.3.0, XGBoost 3.2.0을 사용했다. 모델 학습과 MC CUDA 점검에는 RTX 3090을 사용했다.

```bash
python analysis/ppt_reproduction_20260917/plot_results.py
python analysis/ppt_reproduction_20260917/run_training.py
python analysis/ppt_reproduction_20260917/plot_curves.py
python analysis/ppt_reproduction_20260917/reprice_audit.py --count 12 --device cuda
python analysis/ppt_reproduction_20260917/reprice_audit.py --count 12 --device cpu
python analysis/ppt_reproduction_20260917/verify_results.py
python analysis/ppt_reproduction_20260917/write_report.py
```

`run_training.py`는 완료된 예측·폴드를 재사용한다. 완전히 새 학습은 실험 폴더 전체를 다른 이름으로 복사한 다음 복사본의 `results/`와 `figures/`를 비워 실행한다. 입력·코드가 바뀐 상태에서 이전 체크포인트를 재사용하지 않는다. 그림 재생성만 할 때 MC·학습을 다시 할 필요는 없다.

전체 원천→MC 완전 재현을 마무리하려면 최신 scratch 실행 파일과 노트북 16·17의 상품 ID/중간 CSV가 필요하다. 이후 최신 B의 합성 계약·증분 실험을 진행할 때는 이 미확정 가격 생성 규칙부터 고정해야 한다.
'''
 (HERE/'REPORT.md').write_text(report)
 root=HERE.parents[1]/'README.md';s=root.read_text();marker='## 2026-09-17 발표 실험 재현';s=s.split(marker)[0].rstrip();extra=f'''\n\n{marker}

최신 A/B MC 캐시와 원본 모델 함수로 가격분포 분석 및 Stage 1 각 5시드 재학습을 실행했다. A는 NS·HV120·무배당, B는 CD/IRS 부트스트랩·EWMA120·이산배당이다. main의 실험은 그대로 보존했다.

{measured}

{learned}

전체 MC는 제공 캐시를 재집계했고, 별도로 12개 상품을 각 4만 경로로 다시 가격했다. 누락된 최신 실행 스크립트 때문에 원천부터 MC까지 완전히 동일한 재현으로 보지는 않는다. 최신 B의 합성 계약·증분 학습 결과는 아직 포함하지 않는다.

[실행 과정·비교표·그림·한계](analysis/ppt_reproduction_20260917/REPORT.md) · [검증 결과](analysis/ppt_reproduction_20260917/results/verification.json)
''';root.write_text(s+extra)
 print('REPORT and README updated from measured CSVs')
if __name__=='__main__':main()
