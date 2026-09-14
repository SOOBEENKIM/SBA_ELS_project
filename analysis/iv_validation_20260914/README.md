# ATM IV를 적용한 MC·DeepONet 추가 검증

현재 보완된 MC 계약 엔진을 사용해 역사적 변동성(HV)과 제공된 단일 ATM IV의 가격·증분을 비교한다. 예측 타깃은 MC 이론가이며 기존 Stage 1 DeepONet 9개를 고정해 평가한다. Stage 2, 관측 공정가 학습 및 새 모델 학습은 사용하지 않는다.

- 결과와 해석: [REPORT.md](REPORT.md)
- 입력·계약·계산·모델 설정: [protocol.json](protocol.json)
- 입력 감사: [results/input_audit.json](results/input_audit.json)
- 표본별 IV 출처: [results/market_assignment.csv](results/market_assignment.csv)
- 제외 사유: [results/excluded_families_detailed.csv](results/excluded_families_detailed.csv)
- 조건별 MC 증분: [results/mc_effect_summary.csv](results/mc_effect_summary.csv)
- 모델별·시드별 평가: [results/model_metrics.csv](results/model_metrics.csv)

## 비교 범위

기존 시험 계약 256개와 공시 월지급형 6개에서 출발한다. 세 자산의 유효한 IV가 모두 있는 236개(일반형 128, 월지급형 시험 102, 공시 6)를 포함하고 26개는 제외했다. IV 이력 시작 전 또는 매핑/시계열이 없는 경우를 HV 대체로 채우지 않는다. 날짜 as-of 최대 7달력일 규칙을 사용한다.

일반형 합성 계약에는 고정 난수로 선택한 실상품의 자산·발행일 시장 상태를 부여한다. 월지급형은 원래 일정 템플릿의 자산·발행일을 사용한다. 합성 계약의 지급 규칙은 유지하므로 해당 실상품의 공정가를 합성 계약의 정답으로 사용할 수 없다.

HV/IV 쌍에서 변동성 외에는 동일하다. 상관은 발행 전 최대 180거래일 역사적 상관, 금리곡선은 동일한 KRW NS, 배당은 q=0이다. IV의 옵션 만기 정보는 제공되지 않아 단일 고정 변동성으로만 적용한다. 만기 변경에서도 IV는 고정한다. 기간구조·스마일을 보정한 실험은 아니다.

시드당 40,000경로×3시드이며 10,000/20,000/40,000은 누적 체크포인트다. 각 군 9,907개 기준·변경 시나리오를 계산한다. 기존 쿠폰·KI·행사가·월 지급 배리어 실험과 관측 추가·만기 변경 실험을 모두 포함한다. 평가 계약의 학습 변동성 범위 이탈 여부는 별도 기록한다.

## 재현

저장소 최상위에서 실행한다. 기존 `requirements.txt`의 의존성이 필요하다. 이번 실행은 격리된 Python 3.11.16 / PyTorch 2.9.1+cpu / NumPy 2.3.0 / pandas 2.3.0 환경에서 수행했다. MC와 추론은 CPU이며 모델 학습은 실행하지 않는다.

```bash
python scripts/restore_artifacts.py
python analysis/iv_validation_20260914/restore_input.py
PYTHONPATH=. python analysis/iv_validation_20260914/prepare.py
PYTHONPATH=. python analysis/iv_validation_20260914/run.py --verify-only
PYTHONPATH=. python analysis/iv_validation_20260914/run.py
PYTHONPATH=. python analysis/iv_validation_20260914/verify_predictions.py
PYTHONPATH=. python analysis/iv_validation_20260914/report.py
```

`restore_input.py`는 원본 ZIP의 SHA-256과 추출한 모든 CSV 바이트를 검사한다. 파일이 다르면 덮어쓰지 않고 중단한다. 원본 ZIP과 출처 JSON은 `data/raw/iv_daily_atm/`에 있고, 추출 CSV는 재현 가능한 작업 파일로 Git에서 제외한다.

`prepare.py`는 기존 계약과 일정 변형을 그대로 읽고 새 시장 상태와 실험 프로토콜을 만든다. `run.py`는 프로토콜·코드·계약 해시가 같은 경우에만 완료된 MC 작업을 재사용한다. 중단되었으면 같은 명령으로 이어서 실행할 수 있다. 설정을 바꾸려면 새 실험 폴더를 사용한다.

`run.py`의 기본 동시 실행은 CPU worker 12개다. 각 worker의 PyTorch 쓰레드는 1개다. 실행 전체에서 기존 엔진·인코더·학습 자료·모델 가중치 해시를 확인한다. 새 실험 폴더와 원본 IV 사본 이외의 과거 결과를 수정하지 않는다.

모델 재평가 없이 저장된 결과에서 보고서만 갱신하려면 `report.py`를 실행한다. Git의 `families.json.gz`만 있고 작업용 JSON이 없다면 먼저 아래 명령으로 복원한다.

```bash
python -c "import gzip,pathlib; p=pathlib.Path('analysis/iv_validation_20260914/results/families.json'); p.write_bytes(gzip.decompress(p.with_suffix('.json.gz').read_bytes())) if not p.exists() else None"
```

## 결과 파일의 의미

- `mc_labels.csv`: HV, IV 및 paired IV−HV의 가격·증분과 표준오차. `IV_minus_HV`는 별도의 가격모형이 아니라 같은 난수에서 계산한 차이다.
- `mc_seed_checkpoints.csv.gz`: 3시드 및 누적 경로별 합계·제곱합. 각 시드의 원 자료로 MC 불확실성을 재계산할 수 있다.
- `mc_effect_summary.csv`: 상품군·변경 조건·변경 폭별 평균, 범위, 부호와 MC 불확실성.
- `predictions.csv.gz`: 기존 3개 모델 방식×3개 학습 시드 및 방식별 앙상블의 모든 예측.
- `model_metrics.csv`: 가격·증분 오차, MC에서 구분 가능한 부호 일치율 및 금액별 허용오차 비율.
- `model_metrics_by_sigma_domain.csv`: 적어도 한 자산의 σ가 기존 학습 범위 12~45% 밖인지에 따른 성능.
- `mc_convergence.csv.gz`, `mc_seed_dispersion.csv`, `model_seed_dispersion.csv`: 경로 수 및 MC·학습 시드별 안정성.
- `paired_model_error_bootstrap.csv`: 고정 계약군에서 IV−HV의 계약별 평균 증분 절대오차 차이에 대한 family bootstrap. 새로운 실상품 모집단의 인과효과가 아니다.
- `pre_mc_verification.json`, `post_run_verification.json`: 기존 엔진 일치, 동일 σ 대조, 시드 공유, 증분 정의와 기존 자료 보존 검증.

이 결과를 이전 임의 시장 상태의 Report_2 평균과 직접 비교해 IV 효과로 해석하지 않는다. IV 효과 비교의 기준은 이번 실험 안에서 구성한 동일 계약·동일 시장 조건의 HV/IV 쌍이다.
