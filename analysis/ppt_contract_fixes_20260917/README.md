# 계약 지급 규칙 보완 실험

기준 commit: `fae6ecdec62723a6576a68daae487e181c7d2622`
보존 branch: `codex/ppt-pricer-reproduction-20260917`
작업 branch: `codex/ppt-pricer-contract-fixes-20260917`

이 폴더가 이번 branch의 신규 실행 경로다. 이전 실행 폴더와 원본 코드는 그대로 보존한다.

## 변경 사항

- 미낙인 만기 지급: 원문 `PMT_2`를 명시적으로 사용. 누락된 값을 연 쿠폰으로 임의 대체하지 않는다.
- 월지급형: 원문 `SCHD_TYPE=2`의 평가일·배리어·쿠폰을 별도 현금흐름으로 평가. 조기상환 이후 쿠폰 중단, 상환일에 발생한 쿠폰과 이미 지급이 확정된 쿠폰 보존.
- 평가·지급일: 실제 `EXER_DT` 사용. 공시를 개별 대조한 6개는 명시된 지급 규칙 사용. 나머지는 최종 평가일과 `MAT_DT`로 식별되는 1~5 은행영업일 지연을 전 회차에 동일 적용한다는 가정. 식별 불가 제외.
- 변동성·상관계수·금리곡선·배당 이력/투영 규칙·원래 상품별 난수 시드는 고정.
- 원본/만기 쿠폰만 보완/전체 현금흐름 보완을 같은 경로에서 계산.

## 실행

저장소 루트에서 Python 3.11 및 기존 CUDA 환경 사용. 입력은 같은 저장소의 압축 원문과 이전 전체 재현 실험에 고정한 시장 입력이다. 새로운 외부 다운로드가 필요하지 않다.

```bash
python -m pip install -r analysis/ppt_contract_fixes_20260917/requirements.txt
python analysis/ppt_contract_fixes_20260917/validate.py
python analysis/ppt_contract_fixes_20260917/prepare.py
python analysis/ppt_contract_fixes_20260917/run_mc.py --gpus 0,3,3
python analysis/ppt_contract_fixes_20260917/run_training.py --gpus 0,3
python analysis/ppt_contract_fixes_20260917/probe_mc.py
python analysis/ppt_contract_fixes_20260917/probe_models.py
python analysis/ppt_contract_fixes_20260917/plot_results.py
python analysis/ppt_contract_fixes_20260917/verify_results.py
python analysis/ppt_contract_fixes_20260917/write_report.py
```

`validate.py`는 GPU 1, `probe_mc.py`는 GPU 0을 사용한다. GPU 배치가 다른 환경에서는 `CUDA_VISIBLE_DEVICES`와 코드의 device 설정을 함께 맞춘다. MC는 완료된 상품별 JSONL에서 재개하고, 학습은 완료된 fold를 재사용한다. 입력이나 계산 코드가 바뀌면 기존 결과와 섞지 않고 새 실행 폴더에서 실행한다.

그림만 다시 만들 때는 `plot_results.py`만 실행한다. MC를 다시 돌리지 않는다.

## 출력 및 해석

- [REPORT.md](REPORT.md): 표본·제외 사유·수정 전후 가격/모델 성능·증분 진단.
- `results/paired_prices.parquet`: 같은 경로의 A/B 시장 × 3개 지급 구현 가격과 표준오차.
- `results/contract_audit.csv.gz`: 상품별 지원 여부, 지급 지연, 원문 가격 단위.
- `results/stage1_by_seed.csv`: 40개 학습 설정의 실제 OOS 결과. Stage 2는 실행하지 않음.
- `results/probe_mc.csv`, `probe_predictions.csv.gz`: 고정 일정에서 4개 계약조건의 양·음 변경 폭별 MC/모델 증분.

공정가와의 근접성이 지급 규칙의 정확성을 입증하지 않는다. 계약 보완 효과는 동일 상품끼리 비교하고, MC 근사 성능은 각 모델의 예측값과 명시된 MC 타깃으로 평가한다. 기존 51개 계약 입력을 고정했으므로 상세 월 쿠폰/평가/지급 일정까지 모델에 새로 제공한 실험은 아니다. 관측 횟수·만기 변경과 전체 상품 약관의 개별 대조는 이번 신규 증분 진단 범위에 포함하지 않는다.
