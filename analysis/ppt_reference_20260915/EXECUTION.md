# 실행 순서

이 branch의 기준은 `analysis/ppt_reference_20260915`다. 원본 가격의 재집계와 새 MC 실행을 구분한다. 아래 Python 명령은 저장소 루트에서 실행한다.

## 저장된 결과 읽기·그림 재생성

```bash
python scripts/restore_artifacts.py
python analysis/ppt_reference_20260915/plot.py
python analysis/ppt_reference_20260915/plot.py --fresh
python analysis/ppt_reference_20260915/report.py
python analysis/ppt_reference_20260915/verify_results.py
```

MC 재계산이나 재학습 없이 저장된 가격·예측으로 그림과 표를 만든다. 저장된 가중치로 예측을 다시 계산하려면 `learn.py evaluate`를 먼저 실행한다.

## 데이터 준비부터 새 실행

기존 결과를 보존하도록 새 폴더를 만든다. 아래 `ppt_repeat_001`은 사용하지 않은 폴더 이름으로 바꿀 수 있다. 실행 코드·원본 입력 스냅샷·출처 기록을 복사하며 기존 MC 라벨·가중치·시험 결과는 복사하지 않는다.

```bash
python scripts/restore_artifacts.py
python analysis/ppt_reference_20260915/fresh_run.py --name ppt_repeat_001
```

아래 실행 순서에서 **`ppt_reference_20260915`를 새 폴더 이름 `ppt_repeat_001`로 바꿔서 실행**한다.

```bash
python scripts/restore_artifacts.py
python analysis/ppt_reference_20260915/prepare.py
python analysis/ppt_reference_20260915/plot.py
python analysis/ppt_reference_20260915/run_population.py --verify-only --device cuda
python analysis/ppt_reference_20260915/run_population.py --workers 1 --device cuda
python analysis/ppt_reference_20260915/plot.py --fresh
python analysis/ppt_reference_20260915/synthetic.py prepare
python analysis/ppt_reference_20260915/synthetic.py verify --device cuda
python analysis/ppt_reference_20260915/synthetic.py run --workers 1 --device cuda
python analysis/ppt_reference_20260915/audit_payoffs.py
python analysis/ppt_reference_20260915/learn.py train --workers 6
python analysis/ppt_reference_20260915/learn.py evaluate
python analysis/ppt_reference_20260915/finalize_metadata.py
python analysis/ppt_reference_20260915/report.py
python analysis/ppt_reference_20260915/verify_results.py
```

CUDA가 없는 환경에서는 MC 명령의 `--device cpu`를 사용한다. 장치가 바뀌면 난수 스트림이 달라지므로 CPU와 GPU 캐시는 분리한다. 가격 계산식은 동일하다. 위 명령은 Python 3.11 및 저장소 requirements 환경을 사용한다.

CUDA 실험에는 CUDA 지원 PyTorch가 필요하다. 이번 저장 결과의 MC는 PyTorch 2.9.1+cu126 / RTX 3090, 학습은 PyTorch 2.9.1 CPU에서 실행했다. 여러 GPU에서는 해당 GPU만 `CUDA_VISIBLE_DEVICES`로 노출하고 `run_population.py` 대신 `multi_gpu_population.py --workers 2 --device cuda`를 실행할 수 있다. worker 수는 보이는 GPU 수에 맞춘다. 원래 상품 계산 함수는 동일하다. 실제 실행도 단일 GPU로 저장한 27,230개 상품을 재사용하고, 나머지 31,560개만 두 GPU로 계산했다. 실행 로그를 분리해 보관했다.

MC 완료 이후 그림과 보고서를 다시 만들 때는 `plot.py`, `plot.py --fresh`, `report.py`만 실행한다. 모델은 저장된 가중치로 `learn.py evaluate`를 실행할 수 있다. 시험 평가를 이미 실행한 폴더에서 재학습하려 하면 중단하므로 새 학습 가설은 새 실험 폴더에 둔다.

`prepare.py`는 이미 고정된 입력 파일이 있으면 해시를 확인하고 재사용한다. 중단된 MC는 동일 코드·입력·장치에서 같은 `run` 명령을 다시 실행하면 저장된 상품/가족별 캐시 이후부터 이어간다. 데이터·계약·시장 설정을 바꿀 때는 새 폴더를 사용한다. 반복 실행에서도 기존 시험군이 새로운 독립 시험군이 되는 것은 아니다.

`finalize_metadata.py`는 이전 실험에서 상속한 설명 문장만 현재 두 지급 구현·이전 동결 비교군에 맞게 명확히 한다. 숫자 설정·라벨·가중치는 바꾸지 않으며 수정 전후 문장을 기록한다. 최종 검증 뒤 `summarize_readme.py`를 실행하면 결과에서 요약을 생성한다. 저장소 README에도 반영하려면 `--update-root`를 명시한다.

실제 상품은 원본과 같은 상품별 시드로 40,000경로를 계산한다. 합성 계약은 학습·검증 1시드, 시험·진단 3시드의 40,000경로다. 원본 지급 근사와 상세 지급 구현은 같은 경로를 공유한다. 학습 목표는 각 구현의 MC 가격이며 관측 공정가나 Stage 2 잔차가 아니다.
