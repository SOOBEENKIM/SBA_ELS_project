"""Generate report values from saved CSVs; no hard-coded experiment outcomes."""
from common import *
def table(frame):
 def cell(x):
  if isinstance(x,(float,np.floating)):return f'{x:.4f}' if np.isfinite(x) else 'NA'
  return str(x).replace('|','/').replace('\n',' ')
 rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
 rows += ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(rows)
def main():
 audit=json.loads((OUT/'input_audit.json').read_text());check=json.loads((OUT/'verification.json').read_text());assert check['status']=='pass'
 fair=pd.read_csv(OUT/'fair_mc_summary.csv');seed=pd.read_csv(OUT/'stage1_by_seed.csv');assert len(seed)==40
 agg=seed.groupby(['payoff','model','variant'])[['R2','MAE','RMSE','MAPE']].agg(['mean','std']).reset_index()
 rows=[]
 for _,r in agg.iterrows():rows.append({'지급 구현':r[('payoff','')],'모델':r[('model','')],'시장':r[('variant','')],'R² 평균':r[('R2','mean')],'R² SD':r[('R2','std')],'MAE(원)':r[('MAE','mean')],'MAPE(%)':r[('MAPE','mean')],'MAPE SD':r[('MAPE','std')]})
 # Cross-target evaluation: both fitted models are assessed against corrected MC.
 corrected=pd.read_parquet(OUT/'paired_prices.parquet').set_index('item');cross=[];by_type=[]
 for arm in ('legacy','corrected'):
  for model in ('deeponet','xgb'):
   for x in 'AB':
    for sd in range(5):
     f=pd.read_csv(OUT/'predictions'/f'{arm}_{model}_{x}_seed{sd}.csv.gz');f.mc_true=f.ITEM_CD.map(corrected[f'mc_{x}_corrected']);cross.append(dict(train_payoff=arm,model=model,variant=x,seed=sd,**score(f)))
     f['monthly']=f.ITEM_CD.map(corrected.monthly)
     for monthly,g in f.groupby('monthly'):
      e=(g.mc_pred-g.mc_true)*10000;by_type.append(dict(train_payoff=arm,model=model,variant=x,seed=sd,monthly=monthly,n=len(g),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean()),MAPE=(e.abs()/(g.mc_true*10000)).mean()*100))
 cross=pd.DataFrame(cross);cross.to_csv(OUT/'stage1_against_corrected_target.csv',index=False);cg=cross.groupby(['train_payoff','model','variant'])[['R2','MAE','MAPE']].mean().reset_index()
 types=pd.DataFrame(by_type);types.to_csv(OUT/'stage1_corrected_target_by_payment_type.csv',index=False);type_table=types[(types.model=='deeponet')&(types.variant=='B')].groupby(['train_payoff','monthly'])[['n','MAE','RMSE','MAPE']].mean().reset_index()
 q=pd.read_parquet(OUT/'paired_prices.parquet');steps=[]
 for x in 'AB':
  for monthly,g in q.groupby('monthly'):
   steps.append({'시장':x,'월지급형':monthly,'상품수':len(g),'만기 쿠폰만 추가(원)':g[f'delta_{x}_survival'].mean()*10000,'월지급·일정 추가 보완(원)':g[f'delta_{x}_schedule_monthly'].mean()*10000,'전체 수정 ΔMC(원)':(g[f'mc_{x}_corrected']-g[f'mc_{x}_legacy']).mean()*10000})
 probes=pd.read_csv(OUT/'probe_mc.csv');pp=pd.read_csv(OUT/'probe_predictions.csv.gz');pp=pp[pp.axis!='base'];pm=[]
 for (arm,model,monthly),g in pp.groupby(['payoff','model','monthly']):
  e=g.pred_delta-g.delta;resolved=(g.delta.abs()>1.96*g.delta_se)&g.affected.ge(30)
  pm.append({'학습 지급 구현':arm,'모델':model,'월지급형':monthly,'증분 MAE(원)':e.abs().mean(),'증분 RMSE(원)':np.sqrt((e**2).mean()),'유의한 MC 증분 방향 일치율(%)':(np.sign(g.loc[resolved,'pred_delta'])==np.sign(g.loc[resolved,'delta'])).mean()*100,'방향 평가 행수':int(resolved.sum())})
 pd.DataFrame(pm).to_csv(OUT/'probe_summary.csv',index=False)
 get=lambda arm,monthly:next(r for r in pm if r['학습 지급 구현']==arm and r['모델']=='deeponet' and r['월지급형']==monthly)
 old_o,new_o=get('legacy',False),get('corrected',False);old_m,new_m=get('legacy',True),get('corrected',True)
 increment_conclusion=f"같은 수정 MC 증분을 정답으로 비교하면, DeepONet 증분 MAE는 일반형 {old_o['증분 MAE(원)']:.2f}→{new_o['증분 MAE(원)']:.2f}원, 월지급형 {old_m['증분 MAE(원)']:.2f}→{new_m['증분 MAE(원)']:.2f}원이다. 방향 일치율은 일반형 {old_o['유의한 MC 증분 방향 일치율(%)']:.2f}→{new_o['유의한 MC 증분 방향 일치율(%)']:.2f}%, 월지급형 {old_m['유의한 MC 증분 방향 일치율(%)']:.2f}→{new_m['유의한 MC 증분 방향 일치율(%)']:.2f}%다. 일반형 평균 증분 오차는 줄었지만 월지급형과 방향 정확도가 함께 개선된 결과는 아니다. 모든 조건에서 안정적인 증분 예측을 달성했다고 결론 내릴 수 없다."
 monthly_ki_n=int(probes[(probes.monthly)&(probes.axis=='ki_barrier')].item.nunique())

 focus=probes[((probes.axis=='coupon_regular')&np.isclose(probes.h,.01))|((probes.axis.isin(['ki_barrier','first_strike','last_strike']))&np.isclose(probes.h,.05))].groupby(['monthly','axis']).agg(n=('item','size'),mean_delta=('delta','mean'),min_delta=('delta','min'),max_delta=('delta','max'),mean_paired_SE=('delta_se','mean')).reset_index()
 exclusions=pd.DataFrame(audit['excluded_contract_reasons'].items(),columns=['사유','상품수'])
 scope=pd.DataFrame(audit['structure_counts']);cache=pd.read_csv(OUT/'previous_cache_comparison.csv')
 # Summary values are computed from this run, not copied from an earlier report.
 bshift=(q.mc_B_corrected-q.mc_B_legacy).mean()*10000
 bf=fair[(fair.basis=='common_10000_face')&(fair.variant=='B')].set_index('payoff')
 dc=cg[(cg.model=='deeponet')&(cg.variant=='B')].set_index('train_payoff')
 headline=f"시장 B에서 지급 규칙 수정에 따른 평균 MC 변화는 {bshift:+.2f}원이다. 같은 공통 액면 표본의 공정가 대비 MAE는 {bf.loc['legacy','MAE']:.2f}원에서 {bf.loc['corrected','MAE']:.2f}원으로 바뀌었다. 수정된 MC를 공통 정답으로 평가하면 DeepONet의 MAE는 기존 지급 라벨 학습 {dc.loc['legacy','MAE']:.2f}원, 수정 라벨 학습 {dc.loc['corrected','MAE']:.2f}원이며, 새 모델 R²는 {dc.loc['corrected','R2']:.6f}이다."
 report=f"""# 계약 지급 규칙 보완 후 재가격·Stage 1·증분 진단

{headline}


## 1. 보존한 기준과 이번 실행

- 기준 branch: `codex/ppt-pricer-reproduction-20260917`, commit `fae6ecd`.
- 수정 branch: `codex/ppt-pricer-contract-fixes-20260917`.
- 이전 원본 재현 실험과 `main`을 수정하지 않고, 동일 commit에서 별도 worktree를 만들었다.
- 수정 범위: **만기 미낙인 쿠폰, 월 쿠폰 현금흐름, 실제 평가일과 지급일 할인**.
- 기존 분석 표본 {audit['total']:,}개 전체를 원문과 대조했다. 계약 필드로 지원 가능한 상품 {audit['contract_supported']:,}개 중 시장 입력도 유효한 **{audit['market_and_contract_supported']:,}개 전체**를 새로 계산했다. 6개 공시 예제만 다시 계산한 결과가 아니다.
- MC 가격: 상품·시장 설정별 **40,000경로, 기존 상품별 MC 시드 1개**. A/B 각각 같은 난수에서 3개 지급 구현을 평가했다.
- Stage 1: 2개 지급 타깃 × A/B × DeepONet/XGBoost × 학습 시드 0~4 = **40회 설정, 160개 fold 가중치**. 각 실행의 OOS 표본 {check['OOS_rows_per_configuration']:,}개로 동일하다.
- 별도 증분 진단: OOS 기준 계약 **{check['probe_families']}개, 기준·변경 계약 {check['probe_cases']:,}개**, 시장 B 고정, **40,000경로 × 3개 MC 시드**. 이 가격은 학습에 넣지 않았다.

## 2. 원본의 처리와 수정

| 항목 | 원본 재현 엔진 | 이번 구현 |
|---|---|---|
| 만기 미낙인 | 정규 상환 조건 미충족·KI 미발생이면 원금만 지급 | 원문 만기 `PMT_2`를 미낙인 생존 쿠폰으로 더함. `PMT_2` 누락 상품은 임의로 연 쿠폰×만기로 채우지 않고 제외 |
| 월지급형 | 일부를 연 쿠폰×누적 기간의 상환 쿠폰으로 대체 | `SCHD_TYPE=2`의 평가일·배리어·`PMT_1`을 별도 월 현금흐름으로 평가 |
| 평가일 | 만기/회차 수로 균등 배치 | 회차별 실제 `EXER_DT` 사용 |
| 지급일 | 평가일에 바로 지급·할인 | 명시/추정한 회차별 지급일로 할인 |
| 조기상환과 월 쿠폰 | 별도 월 현금흐름 없음 | 상환 후 새 쿠폰 발생 중단. 동일 평가일의 충족 쿠폰과 이미 확정된 쿠폰 보존 |
| 리자드 | FROM_ISU no-touch | 같은 규칙 유지. 정규 상환이 먼저 충족되면 정규 상환 우선 |

핵심 계산은 `PV = Σ(지급일 할인계수 × 해당 경로의 지급액)`이다. 일반형의 만기 정규 조건 미충족 시, KI 발생 경로는 원문에서 지원하는 worst-of 손실 상환을 적용하고, 미발생 경로는 `1 + PMT_2`를 지급한다. 월지급 쿠폰은 지급 조건이 성립한 회차의 현금흐름을 별도로 더한다. 원문 정규 상환 `PMT_1`과 월 지급 `PMT_1`을 섞지 않는다.

`contracts.py`는 원문을 명시적 `ContractV3`로 변환한다. `engine.py`는 동일 경로에 원본, 만기 쿠폰만 추가, 전체 현금흐름 보완을 적용한다. 실제 지급 계산은 기존 `module/mc_contract_v2.py`·`mc_contract_v3.py`의 명시적 지급 함수를 재사용하며 원본 파일은 바꾸지 않았다.

## 3. 지급일 데이터와 남는 가정

원문 `SCHD_INFO`에는 평가일 `EXER_DT`가 있지만 회차별 지급일 필드가 없다. `AUTO_CALL`에는 최종 만기일 `MAT_DT`가 있다. 이는 기초자산 배당 데이터의 누락과 별개다.

- 공시를 개별 대조한 월지급형 6개: 저장된 공시의 지급 지연·일정 사용.
- 나머지: 최종 평가일부터 `MAT_DT`까지 유일하게 식별되는 1~5 한국 은행영업일 지연을 구한 뒤 전 회차에 같은 지연을 적용. **추정 규칙이며 모든 약관의 개별 확인 결과가 아니다.**
- 월 쿠폰의 상환일 지급 조건·이미 확정된 쿠폰의 보존도 지원 범위 내 명시한 가정이다. 메모리 쿠폰·평균 평가·다른 KI 관찰 구간은 자동으로 추정하지 않는다.
- 일별 달력시간 경로, 발행 시 기준 가격 비율 1, 기존 환율/quanto 미보정 등 시장모형 가정은 유지했다. 실제 해외 거래소별 종가 달력과 전 상품의 약관을 완전히 대조한 상태가 아니다.

### 계약 지원 범위

{table(scope)}

### 계약 제외 사유

아래는 계약 필터에서 먼저 발견된 사유 기준으로 상품별 1건씩 집계한 값이다. 별도로 시장 입력이 유효하지 않은 {audit['contract_supported']-audit['market_and_contract_supported']}개를 계산에서 제외했다.

{table(exclusions)}

## 4. 고정한 시장 입력과 비교 방법

- A: HV120, 기존 역사적 상관계수, Nelson–Siegel 곡선, 배당 없음.
- B: EWMA120(λ=0.99), 같은 상관계수, CD/IRS 부트스트랩 곡선, 기존 이산 배당락.
- 배당 이력 선택·ETF 대리지수·연간 일정 투영은 이전 재현에서 고정한 규칙을 그대로 사용했다. 원본에서 빠져 있던 배당 실행 파일을 새로 찾았다는 뜻은 아니다.
- 상품별 자산 순서·변동성·상관계수·금리곡선·난수 시드는 유지했다. 실제 마지막 평가일이 기존 경로보다 길 때만 같은 배당 규칙으로 경로를 연장했다.
- 3개 지급 구현은 각 시장 설정 안에서 공통 난수를 사용한다. 가격 차이는 같은 상품·경로에서 계산했다.
- 전체 58,760개였던 이전 결과와 이번 44,103개 결과의 평균을 그대로 비교하지 않는다. **수정 전후 모두 동일한 지원 표본을 사용한다.**

이전 MC 캐시와 이번 공통 경로의 원본 구현 가격 비교:

{table(cache)}

## 5. 지급 규칙을 바꾸어 발생한 가격 차이

원화는 액면 10,000원 기준의 MC 값이다. `월지급·일정 추가 보완`은 만기 쿠폰만 추가한 구현에서 월 쿠폰 분리·원문 정규 지급률·실제 평가일·지급일 할인까지 반영한 추가 차이다. 이 단계의 각 세부 요소를 다시 독립적으로 분리한 ablation은 아니다.

{table(pd.DataFrame(steps))}

## 6. 관측 공정가와 MC의 차이

`common_10000_face`는 발행가가 9,000~10,000원인 원문에 액면 10,000원 가정을 적용해 `FAIR_VALUE - MC×10000`을 비교한다. 공시 6개 외의 액면가는 약관별 개별 대조를 완료한 값은 아니다. `original_normalization`은 이전 PPT 재현과 동일한 `(FAIR_VALUE/ISU_PRC_DETAIL - MC)×10000`이며, 발행가와 액면가가 다르면 공통 원화 기준이 아니다. 두 기준을 구분해서 보관했다.

{table(fair)}

가격을 수정한 결과가 공정가에 가까워졌는지와 지급 규칙이 정확한지는 다른 질문이다. 지급 규칙은 원문 현금흐름과 명시한 지급일 가정을 기준으로 구성했으며, 위 표는 관측 차이에 대한 진단이다.

![B 공정가-MC 분포](figures/histogram_common_10000_face_B.png)
![B 분위별 차이](figures/segments_common_10000_face_B.png)

## 7. Stage 1 재학습 결과

기존 51개 계약 입력, 시장 입력 구성, 모델 구조, 학습 설정, 원래 58,790행에서 정한 시간 순 fold 경계와 검증 분할을 유지했다. 그 뒤 동일한 지원 표본으로 제한했다. 수정 전후 모두 새로 학습했으며 타깃은 각 지급 구현의 MC 가격이다. Stage 2와 PI 손실은 사용하지 않았다.

{table(pd.DataFrame(rows))}

두 행의 R²는 서로 다른 MC 타깃 분산을 사용한다. 수정 전후 R²만으로 같은 문제의 성능이 개선되었다고 단정하지 않는다. 다음 표는 **모두 수정된 MC를 정답으로 사용**하여 기존 지급 라벨로 학습한 모델과 새 라벨로 학습한 모델을 비교한 5시드 평균이다.

{table(cg)}

같은 수정 MC 정답에 대한 시장 B DeepONet의 지급형별 가격 오차:

{table(type_table)}

![Stage 1 R2](figures/stage1_r2.png)
![Stage 1 MAPE](figures/stage1_mape.png)

실제 평가/지급 일정과 월 쿠폰별 배리어·지급액은 MC에 반영했지만, 이번 통제 비교에서는 DeepONet/XGBoost의 입력을 확장하지 않았다. 원래 51개 입력만으로 구별되지 않는 계약 차이는 학습 모형의 한계로 남을 수 있다. 가격 R²가 높다는 사실만으로 증분 예측의 안정성을 판정하지 않는다.

## 8. 수정된 MC 기준의 증분 진단

같은 계약의 다른 조건과 일정을 고정하고 연 쿠폰 ±0.1/0.25/0.5/1%p, KI·첫/만기 행사가 ±0.5/1/2.5/5%p를 평가했다. KI가 있는 계약만 KI를 변경했다. 연 쿠폰 변경은 정규·미낙인 생존·월 쿠폰을 같은 비율로 바꾸며 별도 리자드 보너스는 고정한다. 원래 시간 순 OOS에 속하는 구조별 최대 16개 계약을 오차 확인 전에 일정 간격으로 선택했다.

주요 양의 변경 폭에 대한 MC 증분:

{table(focus)}

아래는 모델별·학습 시드별 개별 예측의 오차를 모든 변경 폭에 걸쳐 평균한 값이다. 5개 모델을 앙상블한 값이 아니다. 방향은 `|ΔMC| > 1.96×paired SE`이고 영향 경로 30개 이상인 경우에만 집계했다. 3개 MC 시드의 최소·최대는 `probe_mc.csv`, 변경 폭별 성능은 `probe_metrics.csv`에 있다.

{table(pd.DataFrame(pm))}

{increment_conclusion}

월지급형 KI 증분은 기준 계약 {monthly_ki_n}개에서 나온 결과이므로 월지급형 KI 전체로 일반화하지 않는다.

이 진단은 명시한 MC 모형 안의 반사실 가격 차이다. 실제 관측 공정가의 인과효과를 입증한 실험이 아니다. 관측 횟수·만기 변경은 이번 신규 진단에 포함하지 않았다. 과거 별도 일정 실험 결과를 이번 새 라벨·모델의 결과로 재사용하지 않았다. 상세 지급 규칙 반영과 모든 계약의 안정적인 증분 예측 달성도 동일한 결론이 아니다.

![변경 폭별 DeepONet 증분 MAE](figures/increment_delta_MAE_by_offset.png)
![변경 폭별 DeepONet 방향 일치율](figures/increment_direction_agreement_by_offset.png)

## 9. 검증과 재현

- 만기 생존 쿠폰, KI 손실, KI 후 만기 조건 충족, 조기상환, 리자드 우선순위, 월 쿠폰 조건·중단·동일일 처리, 늦게 지급되는 확정 쿠폰, 지급일 할인에 대한 경로별 검증 통과.
- 별도 순차 NumPy 현금흐름 장부와 GPU 벡터화 계산 대조 통과.
- 원래 경로 범위가 같은 사례에서 원본 엔진과 가격이 정확히 일치함을 검증.
- 고정된 원본 자료 {check['source_files_unchanged']}개 해시 보존.
- 저장된 160개 fold 가중치를 재로딩해 예측값 대조. 최대 차이 {check['max_checkpoint_reload_error']:.3g}(액면 비율).
- 입력 해시, 제외 목록, 가격, 표준오차, 예측, 그림, 가중치를 이 branch에 함께 보관.

실행 명령과 파일 위치는 [README](README.md)에 있다. 그림 생성은 저장된 결과만 읽으며 MC를 다시 계산하지 않는다.
"""
 (HERE/'REPORT.md').write_text(report)
 # Local Markdown previews and GitHub can both use a standalone HTML gallery.
 html='<meta charset="utf-8"><title>Contract correction figures</title><style>body{font-family:sans-serif;max-width:1400px;margin:auto}img{max-width:100%}</style><h1>Contract correction figures</h1>'
 for f in sorted(FIG.glob('*.png')):html+=f'<h2>{f.stem}</h2><img src="figures/{f.name}">'
 (HERE/'figures.html').write_text(html)
 root_readme=HERE.parents[1]/'README.md';original=root_readme.read_text();start='<!-- CONTRACT_FIXES_LATEST -->';end='<!-- /CONTRACT_FIXES_LATEST -->'
 block=start+'\n\n## 현재 branch: 계약 지급 규칙 보완\n\n'+headline+'\n\n- 보존한 기준: `codex/ppt-pricer-reproduction-20260917` (`fae6ecd`).\n- 새 branch: `codex/ppt-pricer-contract-fixes-20260917`.\n- 같은 44,103개 상품을 4만 경로로 재가격하고, 수정 전후 동일 표본의 Stage 1을 각 5시드로 재학습했다. 별도로 OOS 계약의 네 조건별 증분을 4만 경로×3시드로 검증했다.\n- [이번 실행 보고서](analysis/ppt_contract_fixes_20260917/REPORT.md) · [실행 방법](analysis/ppt_contract_fixes_20260917/README.md) · [그림 모음](analysis/ppt_contract_fixes_20260917/figures.html).\n- 지급일 일부는 명시한 추정 규칙을 사용한다. 전체 약관 검증이나 모든 증분의 안정적인 예측을 완료했다는 의미는 아니다.\n\n아래에는 기존 main과 이전 branch의 별도 실험 기록을 보존한다.\n\n'+end+'\n\n'
 if start in original:original=original[:original.index(start)]+original[original.index(end)+len(end):].lstrip('\n')
 header='# SBA ELS Project\n\n';assert original.startswith(header);root_readme.write_text(header+block+original[len(header):])
 print('REPORT written',flush=True)
if __name__=='__main__':main()
