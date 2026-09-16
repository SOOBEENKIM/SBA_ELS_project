# 오토콜 ELS 이론가 DeepONet 대리모델 (Surrogate Pricing)

한국형 **3-기초자산 worst-of 스텝다운 오토콜 ELS(주가연계증권)**의 **몬테카를로 이론가(MC 이론가)**를
재현하는 연산자학습(operator-learning) 대리모델 **DeepONet**. 재시뮬레이션 대비 약 **10⁴–10⁵배** 빠르게
가격을 산출한다. 금융감독 공시(DART)와 시장데이터를 결합해 학습하고, 정보누수 없는 **walk-forward**
프로토콜로 강력한 테이블형 머신러닝 벤치마크와 비교 평가한다.

---

## 무엇을 예측하는가

타깃은 공시 공정가치가 아니라 **MC 이론가 `mc`** 다. MC 엔진(`module/mc.py`, 미래에셋 36449 방식:
일별 GBM + 촐레스키(180일 역사 상관) + 180일 역사 변동성 + KRW Nelson-Siegel 할인, 40,000 경로)은
**무보정(k = 1)** 이 정본이다 — 변동성 프리미엄 보정을 탐색했으나 근거·통용성이 약해 기각했다.
학습된 DeepONet은 이 가격을 forward 한 번으로 재현하므로 모델은 **MC 대리모델**이다: 프라이서와 동일한
입력(바스켓 vol·corr + KRW 곡선 + 전체 계약)을 받아 µs 단위로 추론한다.

두 모델 계열을 비교한다.

- **직접(단일 단계)** — 테이블형 회귀기가 `ml.csv` 전체 특성으로 `mc` 를 한 번에 예측.
- **하이브리드(2단계)** — `y = MC_hat(stage-1 앵커) + residual_hat(stage-2)`. stage-2 타깃은
  `mc − MC_hat`(앵커 자신의 근사오차)이며 시장 마진 항은 쓰지 않는다.

> `model/` 디렉토리는 **플러그인 레지스트리**(`model/registry.py`)다. 물리기반 연산자
> (PI-DeepONet / BS-PINN)를 나중에 `run(D, cfg)` 함수를 가진 `model/*.py` 파일 하나로 끼워 넣으면
> 파이프라인 수정 없이 자동 편입된다. `model/_template.py` 참고.

---

## 결과 — walk-forward OOS (n = 23,151; test = 미래 9,261건)

| 모델 | 계열 | R² | MAE | MAPE % | Spearman ρ |
|---|---|---:|---:|---:|---:|
| **deeponet_hybrid_l1**    | DeepONet 하이브리드 (L1 앵커)     | **0.964** | 0.0034 | 0.35 | 0.988 |
| **deeponet_hybrid_s2don** | DeepONet 앵커 + DeepONet 잔차     | **0.961** | 0.0035 | 0.37 | 0.979 |
| **deeponet_hybrid**       | DeepONet 하이브리드 (MSE 앵커)    | **0.956** | 0.0038 | 0.39 | 0.978 |
| **deeponet_hybrid_mape**  | DeepONet 하이브리드 (MAPE 앵커)   | **0.954** | 0.0037 | 0.39 | 0.988 |
| xgb_hybrid                | XGB 앵커 + XGB 잔차               | 0.904 | 0.0061 | 0.63 | 0.963 |
| bench_catboost            | 직접 (CatBoost)                   | 0.902 | 0.0064 | 0.66 | 0.964 |
| bench_xgboost             | 직접 (XGBoost)                    | 0.854 | 0.0076 | 0.78 | 0.950 |
| bench_gbm                 | 직접 (HistGB)                     | 0.842 | 0.0081 | 0.84 | 0.945 |
| bench_lgbm                | 직접 (LightGBM)                   | 0.836 | 0.0081 | 0.84 | 0.952 |
| bench_ridge               | 직접 (Ridge)                      | 0.701 | 0.0124 | 1.26 | 0.895 |

**핵심 결과.** 4개 DeepONet 하이브리드는 **R² ≈ 0.95–0.96** 로 밀집해 모든 직접 ML 벤치마크를 앞선다.
최상위 DeepONet 하이브리드는 최상위 직접 벤치마크(CatBoost)를 약 6 R²포인트 앞선다. 잔차모델보다 **2단계
구조 자체가 더 결정적**이다: XGB 하이브리드(0.904)는 직접 CatBoost(0.902) 바로 위에 놓이지만, DeepONet
앵커는 하이브리드를 그 둘보다 훨씬 위로 끌어올린다. 앵커 손실함수(MAPE / L1 / MSE) 순위는 신경망
run-to-run 노이즈 안에서 뒤바뀌므로 하나의 군집으로 본다.

**단계별 R²(하이브리드).** Stage-1(`mc` 재현) ≈ 0.90–0.96 · Stage-2(잔차 `mc − MC_hat`) ≈ 0.12–0.19
(DeepONet 하이브리드 기준; `xgb_hybrid` 은 ≈ 0) · Final(`mc`) ≈ 0.90–0.96.
**속도**(`4_MC_pricing`): MC 재시뮬 ≈ 2.7 s/상품 vs DeepONet 로드+forward ≈ 22 µs/상품 →
전체 OOS 기준 **약 70,000배 빠름**, MAE ≈ 0.0045.

지표: `result/statistics/`  ·  그림: `result/image/`.

---

## 레포 구조

```
config.yaml              # 모든 하이퍼파라미터의 단일 출처
0_data.ipynb             # raw + cache  -> data/els3_dataset.parquet (base, MC 없음)
1_MC_recompute.ipynb     # mc + recent_margin 추가 -> data/ml.csv, data/deeponet.csv 생성
2_run.ipynb              # 레지스트리 자동발견 -> 전 모델 학습 -> result/predictions/ + 가중치
3_evaluate.ipynb         # 가중치 로드 -> OOS forward -> 지표 + 그림
4_MC_pricing.ipynb       # MC 재시뮬 vs DeepONet 대리모델: 속도 + 가격 일치도
EDA_raw_filter.ipynb     # raw 필터 퍼널 + 공정가비율 / 만기 EDA

model/                   # 모델 플러그인(레지스트리 자동발견): benchmark, deeponet, stage2, _template
module/                  # 데이터 + 특성/MC 엔진 + 네트워크 + 학습 + 추론
util/                    # file_manager(경로), utils(설정/시드), metric, plot, speed
data/                    # cache/ 와 raw/ 만 추적 대상 입력(아래 참고)
result/                  # predictions/, image/, statistics/, models/(폴드별 저장 가중치)
paper/                   # 참고 논문 [1]-[5]
```

### 데이터 — 추적 대상 vs 재생성 대상

입력은 **`data/cache/`** 와 **`data/raw/`** 뿐이며, `data/` 의 나머지는 파생물로 git-ignore 된다.

- `data/raw/` — raw DART CSV(`AUTO_CALL`, `SCHD_INFO`, `UDLY_INFO`). **git-ignore**(>100 MB); 로컬 보관.
- `data/cache/` — 기초자산 일별종가 `px_*.parquet`, KRW 곡선, 티커/행사가 매핑, 그리고 40k 경로 MC
  결과가 캐시된 **28개 `mccal_shard_*.json`**(→ 재시뮬 없이 수 초 만에 `mc` 복원).
- `data/els3_dataset.parquet`, `data/ml.csv`, `data/deeponet.csv` — 노트북 0–1 이 **재생성**.

---

## 재현 (오직 `cache` + `raw` 로부터)

```bash
pip install -r requirements.txt          # Python 3.11 (conda env `abacus`), JAX/PyTorch CPU

# 0) raw + cache 로 base 데이터셋 재유도 (아직 MC 없음)
jupyter nbconvert --to notebook --execute --inplace 0_data.ipynb

# --- MC 이론가 ---
# 전량 재계산은 무겁다(단일코어 ~14h). 샤드 결과가 data/cache/ 에 캐시돼 있어 `combine` 이 수 초 만에
# `mc` 를 복원한다. 처음부터 다시 계산하려면(진행률은 별도 터미널의 monitor 로):
#   N=28; seq 0 $((N-1)) | xargs -P $N -I {} python -m module.mc_shards shard {} $N
#   python -m module.mc_shards monitor $N   # 누적 진행률 n/23151 · 처리율 · ETA 한 줄 갱신
python -m module.mc_shards combine 28    # els3_dataset.parquet 에 mc + recent_margin 기록 (+ heartbeat 정리)

# 1) 모델링 CSV(ml.csv, deeponet.csv) 생성 + MC-vs-공정가 EDA 그림
jupyter nbconvert --to notebook --execute --inplace 1_MC_recompute.ipynb
# 2) 발견된 전 모델을 walk-forward 로 학습 -> result/predictions/ + result/models/
jupyter nbconvert --to notebook --execute --inplace 2_run.ipynb
# 3) 저장 가중치를 로드해 OOS test 를 forward -> result/{statistics,image}/
jupyter nbconvert --to notebook --execute --inplace 3_evaluate.ipynb
# 4) MC 재시뮬 vs DeepONet 대리모델: 속도 + 가격 일치도
jupyter nbconvert --to notebook --execute --inplace 4_MC_pricing.ipynb
```

모든 하이퍼파라미터는 **`config.yaml`**, 모든 경로는 **`util/file_manager.py`** 로 관리한다.
`combine` 단계만 분리돼 있는 것은 그것이 유일한 고비용 단계이기 때문이며, 그 외 노트북 0→4 는 위에서
아래로 순차 실행된다.

---

## 패키지 레퍼런스 — `model/`

`run(D, cfg) -> {name: DataFrame}` 를 노출하는 모든 `model/*.py` 는 `2_run` 이 자동발견·실행한다.
모델 추가 = 파일 하나 넣기(`_` 로 시작하는 파일은 발견에서 제외).

### `model/registry.py`
플러그인 발견·실행.
- `discover()` → `run(D, cfg)` 를 노출하는 `model` 하위모듈들의 `(name, run)` 목록(이름 정렬).
  `registry` 자기 자신, `_` 접두 파일, `run` 없는 모듈은 제외.
- `run_all(D, cfg, verbose=True)` → 각 모델을 실행해 예측 dict 를 병합. 모델명 중복 시 `ValueError`.

### `model/benchmark.py`
테이블형 ML 모델 전체 — 전부 공정가가 아니라 `mc` 를 예측.
- **직접 벤치마크**(단일 단계): `bench_ridge`, `bench_gbm`, `bench_lgbm`, `bench_xgboost`,
  `bench_catboost`. `predict_direct_tab` 로 `ml.csv` 전체 regime 특성 위에서 학습.
- `_xgb_anchor(D, cfg, tr, va, te, save_path=None)` — stage-1 XGB 앵커. **DeepONet 앵커와 동일한 블록**
  (`[곡선 u0-9 | vol·corr | 계약]`)으로 `mc` 를 예측(순수 아키텍처 비교). `predict(idx)` 콜백 반환·`.pkl` 저장.
- `run(D, cfg)` → 직접 5종 + `xgb_hybrid`(2단계 XGB 앵커 + XGB 잔차).

### `model/deeponet.py`
DeepONet-Curve 하이브리드(2단계, MC 타깃, 마진 없음).
- `_anchor` / `_anchor_l1` / `_anchor_mape` — `train_curve` 로 `D.MC` 를 MSE / L1 / MAPE 손실로
  학습하는 stage-1 `CurveOperatorV2` 앵커. 각각 `predict(idx)` 콜백 반환.
- `run(D, cfg)` → `deeponet_hybrid`(MSE 앵커 + XGB 잔차), `deeponet_hybrid_l1`, `deeponet_hybrid_mape`,
  `deeponet_hybrid_s2don`(MSE 앵커 + **DeepONet** 잔차 `don_resid`).

### `model/stage2.py`
공용 stage-2 잔차모델(하이브리드끼리 stage-1 앵커만 다름). 입력은 deeponet 특성 블록
`D.DON = [곡선 u0-9 | vol·corr·sig_eff | 계약]`.
- `xgb_resid(D, cfg, tr, va, te, target, ...)` — 기본 stage-2: `D.DON` 위 XGBoost
  (`deeponet_hybrid`/`_l1`/`_mape`·`xgb_hybrid` 사용). `load_xgb_resid(D, path)` 로 재로드.
- `don_resid(D, cfg, tr, va, te, target, ...)` — DeepONet stage-2: `MarginOperator`
  (branch = vol·corr·sig_eff, trunk = `[곡선 | 계약]`), train 폴드 표준화 + early stopping
  (`deeponet_hybrid_s2don` 사용). `load_don_resid(D, path)` 로 재로드.

### `model/_template.py`
새 플러그인 템플릿. 계약 규약(필수 예측 컬럼 `ITEM_CD, isu_ord, y_true, y_pred`; 하이브리드는
`mc_true, mc_pred, resid_true, resid_pred` 추가), `D` 의 필드, 현행 MC-타깃 관례
(`predict_hybrid(..., target=D.MC, use_margin=False)`)를 문서화.

---

## 패키지 레퍼런스 — `module/`

### `module/data.py`
데이터 스키마·생성·로딩 — 특성 그룹 정의의 단일 출처.
- 상수: `VOLCORR`(sig1-3, rho12/13/23, sig_eff), `UC`(곡선 u0-9), `CONTRACT`(strk0-11 + 배리어 +
  쿠폰 + 만기), 그리고 `BASE`/`REG`/`CAT`(테이블형)·타깃/앵커 이름.
- `build_datasets()` → **`data/ml.csv`**(테이블형 `BASE+REG+CAT`)와 **`data/deeponet.csv`**
  (연산자 입력: 곡선 + vol·corr·sig_eff + 계약 + r) 생성. 둘 다 `mc` / `fair` / `recent_margin` 포함.
- `load(cfg)` → `SimpleNamespace` `D`: `MC`, `FAIR`, `rm`, `ITEM`, `ORD`, branch/trunk 배열
  `VC`(n,7) / `CURVE`(n,10) / `CON`(n,15), stage-2 블록 `DON`(n,32), walk-forward 폴드.
- `walk_forward(n, bounds, val_frac, val_seed)` → 확장윈도우 `[(tr, va, te), …]` 폴드. 각 train
  윈도우에서 랜덤 validation 부분집합을 떼어냄(early stopping 용).
- 헬퍼: `time_weights`(반감기 365.25일 시간감쇠 가중), `to_tensor`, `zstats`(train 폴드 표준화), `featnum`.

### `module/build_source.py`
raw + cache 로 base 데이터셋 재유도(MC 는 여기서 계산하지 않음).
- `build_source(save=True, verbose=True)` → raw DART 테이블 + 캐시 가격/곡선 로드, 후보
  3-기초자산 **STEP / KRW / fair∈[0.70,1.05] / tenor∈[0.5,5]** 오토콜 필터, 180일 vol·corr,
  KRW NS 곡선 노드, 계약 특성·모멘텀 유도 후 `data/els3_dataset.parquet` 에 `fair` +
  `recent_mktvol`(인과적 90일 이동평균 변동성)와 함께 저장.

### `module/features.py`
공용 특성 계산(`0_data` 와 MC 엔진 간 DRY).
- `nsb`, `zero_curve`, `krw_beta`, `krw_curve_nodes` — KRW Nelson-Siegel 곡선 적합·노드 추출.
- `vol180(ret, dt)`, `corr180(rets, dt)` — 기준일 이전 180거래일 역사 변동성 / 3자산 상관.
  `chol_psd(corr)` — 고윳값 클리핑 PSD 보정 폴백이 있는 촐레스키.

### `module/mc.py`
MC 이론가 프라이싱 엔진(재산출 노트북·샤드 스크립트 공용).
- `load_market()` → 로그수익률 시계열, KRW 곡선, 상품→기초자산 / 행사가 매핑(`u3map`/`strk_by` 캐시).
  `load_kmap()` → 보정맵 또는 `None`(**정본 = 무보정**).
- `mc_daily(...)` — 촐레스키 상관 충격 일별 GBM, 일별 KI/오토콜 관측, KRW NS 할인(청크 단위).
  `price_one(mk, kmap, item, iord, B, c, ten, sig_eff, …)` → 상품 1건 프라이싱,
  `(mc, vol1-3, rho12/13/23, r_krw, k)` 반환. 상수: `NPATH`(40,000), `MC_COLS`.

### `module/mc_shards.py`
전량 MC 재산출을 독립 샤드로 병렬화(Windows OOM 방지 위해 `ProcessPool` 회피).
- CLI `main()`: `shard <k> <N>` 은 `tasks[k::N]` 프라이싱 → `data/cache/mccal_shard_<k>.json`
  (200개마다 `scratch/mc_progress/shard_<k>.json` 에 진행 heartbeat 기록); `monitor <N> [초]` 는 전
  샤드 heartbeat 를 합산해 누적 진행률·처리율·ETA·완료샤드 수를 한 줄로 갱신 표시; `combine <N>` 은
  전 샤드를 병합해 `els3_dataset.parquet` 에 `mc` + MC 입력 컬럼 + `recent_margin` 기록(+ heartbeat
  정리); `test` 는 앞 30행을 단일프로세스로 검증.

### `module/networks.py`
DeepONet 아키텍처.
- `CurveOperatorV2(nvc, ncon, P)` — stage-1 앵커: branch = 곡선 1D-CNN + vol·corr 융합,
  trunk = 계약 MLP; 내적 헤드.(스팟 입력 없음 — 스칼라 가격이 타깃.)
- `MarginOperator(nb, nt, P)` — stage-2 잔차 DeepONet: branch(시장상태)·trunk(계약) 내적 + 학습형 bias.
  `mlp(d, p)` — 공용 3층 Tanh MLP 빌더.

### `module/train.py`
stage-1 앵커 학습.
- `train_curve(D, cfg, tr, te, target, va=None, loss="mse", return_predict=False, save_path=None)` —
  표준화된 곡선/vol·corr/계약 특성 위에서 `CurveOperatorV2` 를 `target`·`loss`
  (`mse`/`l1`/`huber`/`mape`)로 학습. validation early stopping + `.pt` 저장. train/test 예측(및 선택적
  `predict(idx)` 콜백) 반환.
- `load_curve_predictor(D, path)` — 가중치 로드 → forward 전용 `predict(idx)`.
- `_opt`, `_EarlyStop`(best 가중치 복원, validation 기반), `_loss_fn` — `model/stage2.py` 도 재사용.

### `module/pipeline.py`
walk-forward OOS 예측 조립 + 폴드별 가중치를 `result/models/<name>_fold{k}.{pt|pkl}` 로 저장.
- `predict_direct_tab(D, cfg, model_key, name=None, target=None)` — 폴드별 단일단계 벤치마크
  (target 기본값 `D.MC`).
- `predict_hybrid(D, cfg, anchor_fn, resid_fn=None, name=None, target=None, use_margin=True)` — 2단계
  조립: 앵커 학습 → 잔차 타깃 `hybrid_residual_target(target, mc_hat, rm)`(stage-1 **예측값** 기준이라
  학습·추론 일치) → 잔차모델 학습/로드(기본 `xgb_resid`) → `y_pred = mc_hat + margin + resid`.
  MC 파이프라인은 `target=D.MC, use_margin=False` 로 호출.

### `module/tabular.py`
직접 벤치마크와 XGB stage-2 가 공용하는 테이블형 회귀기.
- `fit_tab(D, cfg, model, tr, te, feat, target, tw=True, va=None, save_path=None, return_predictor=False)`
  — `ridge`/`gbm`/`lgbm`/`cat`/`xgb` 를 시간감쇠 가중·early stopping(xgb/lgbm/cat)으로 적합, `.pkl` 저장.
  `load_tab_predictor(D, path)` 로 재로드 → `predict(idx)`.

### `module/infer.py`
저장 가중치로 예측 재현(`3_evaluate` 이 예측 CSV 대신 이걸 사용).
- `_DIRECT`(직접 5종)·`_HYBRID`(2단계 5종) 레지스트리.
- `predict_all_from_weights(D, cfg)` → 각 모델의 폴드 가중치를 로드해 OOS test 인덱스를 forward,
  파이프라인과 동일 스키마의 DataFrame 반환(`y_true = mc`; 하이브리드는 단계별 R² 용 컬럼도 포함).

---

## `util/`

- `file_manager.py` — 중앙 경로 관리(`fm.dataset("ml"|"deeponet")`, `fm.image`, `fm.stat`,
  `fm.prediction`, `fm.RESULT/"models"`).
- `utils.py` — `load_config()`, `set_seed()`, 디바이스 선택.
- `metric.py` — `metrics()`(R²/MAE/RMSE/MAPE/Bias/Spearman)·`stage_r2()`(하이브리드 stage-1/2/final).
- `plot.py` — 학술/PPT 그림 기본값(import 시 rcParams 를 **dpi 400 · 투명배경 · PPT 콘텐츠밴드
  13.33″** 로 설정): `metric_bars`, `scatter_grid`, `stage_r2_bar`, `speed_bar`.
- `speed.py` — 전 10개 모델(직접 + 하이브리드)의 추론 지연 측정.

---

## 설정 (`config.yaml`)

`seed`, `device`; `data.walk_forward` 폴드 경계 `[0.6,0.7,0.8,0.9,1.0]`(test = 미래 40%, 확장 4폴드),
`val_frac`, `time_decay`; `train`(연산자망 반복수 / lr / early stopping); `networks.P`(잠재차원);
벤치마크·stage-2 공용 `tabular` 트리 하이퍼파라미터(`xgb`/`lgbm`/`cat`/`gbm`/`ridge`).

---

## 참고문헌 (PDF 는 `paper/`)

1. Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." *Nature Machine Intelligence* 3.3 (2021): 218–229.
2. Wang, Wang, Perdikaris. "Learning the solution operator of parametric PDEs with physics-informed DeepONets." *Science Advances* 7.40 (2021): eabi8605.
3. Lee, Huh, Jeong. "DeepONet-based surrogate modeling for bond option pricing." *AIMS Mathematics* 11.3 (2026): 5853.
4. Andreou, Han, Li. "Stock options pricing via machine learning methods combined with firm characteristics." (2023).
5. Jiao et al. "Solving forward and inverse PDE problems on unknown manifolds via physics-informed neural operators." *arXiv:2407.05477* (2024).
