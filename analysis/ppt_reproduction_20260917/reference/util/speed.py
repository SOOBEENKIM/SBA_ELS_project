# -*- coding: utf-8 -*-
"""예측(추론) 속도 측정 — 학습된 모델로 test 폴드를 예측하는 데 걸리는 시간만 측정.
 학습 시간 제외(즉시 견적 지연시간). fold-0 기준, 반복 median.
 이론가(MC) 예측 경로 전체를 잰다: 2단계 하이브리드(stage1 앵커 forward + stage2 잔차 predict, 마진 없음)
 + 직접(단일 단계) 벤치마크(ml.csv 전 특성 → MC)."""
import time
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from module.data import time_weights
from module.tabular import fit_tab
from module.train import train_curve
from model.stage2 import xgb_resid, don_resid


def _time(fn, repeat=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def measure(D, cfg, repeat=20):
    """반환: index=model, cols=[n, batch_ms, per_product_us, products_per_sec, kind].
     현재 파이프라인 = 이론가(MC) 2단계: y = 앵커(MC_hat) + stage2 잔차(MC−MC_hat), 마진 없음."""
    tr, va, te = D.WF[0]
    N = len(te)
    tw = cfg["data"]["time_decay"]
    rows = []

    def add(name, predict_fn, kind="hybrid"):
        sec = _time(lambda: predict_fn(te), repeat=repeat)
        rows.append(dict(model=name, kind=kind, n=N, batch_ms=sec * 1e3,
                         per_product_us=sec / N * 1e6, products_per_sec=N / sec))

    # stage-1 앵커 (MC 이론가) forward 예측기
    *_, a_mse = train_curve(D, cfg, tr, te, target=D.MC, va=va, return_predict=True)
    *_, a_l1 = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="l1", return_predict=True)
    *_, a_mape = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="mape", return_predict=True)
    # XGB 앵커 (matched-입력 = deeponet 블록)
    XP = cfg["tabular"]["xgb"]; X = D.DON; w = time_weights(D, tr) if tw else None
    _es = va is not None and len(va) > 0
    _kw = dict(n_estimators=XP["n_estimators"], learning_rate=XP["learning_rate"], max_depth=XP["max_depth"],
               subsample=XP["subsample"], colsample_bytree=XP["colsample_bytree"], n_jobs=0, random_state=cfg["seed"])
    if _es:
        _kw["early_stopping_rounds"] = cfg["train"].get("es_rounds", 50)
    ma = XGBRegressor(**_kw)
    if _es:
        ma.fit(X[tr], D.MC[tr], sample_weight=w, eval_set=[(X[va], D.MC[va])], verbose=False)
    else:
        ma.fit(X[tr], D.MC[tr], sample_weight=w)
    a_xgb = lambda idx: ma.predict(X[idx]).astype("float32")

    # stage-2 잔차모델 (target = 이론가 잔차 MC−MC_hat; mse 앵커 기준으로 학습해 타이밍)
    rt = (D.MC - a_mse(np.arange(D.n))).astype("float32")
    *_, r_xgb = xgb_resid(D, cfg, tr, va, te, rt, return_predict=True)     # 공용 stage2 (4모델)
    *_, r_don = don_resid(D, cfg, tr, va, te, rt, return_predict=True)     # DeepONet stage2 (s2don)

    # 2단계 이론가 예측 = 앵커 + 잔차 (마진 없음)
    add("deeponet_hybrid", lambda idx: a_mse(idx) + r_xgb(idx))
    add("deeponet_hybrid_l1", lambda idx: a_l1(idx) + r_xgb(idx))
    add("deeponet_hybrid_mape", lambda idx: a_mape(idx) + r_xgb(idx))
    add("deeponet_hybrid_s2don", lambda idx: a_mse(idx) + r_don(idx))
    add("xgb_hybrid", lambda idx: a_xgb(idx) + r_xgb(idx))

    # 직접(단일 단계) 벤치마크 = ml.csv 전 특성으로 MC 직접 예측 (앵커/잔차 없음)
    for key, nm in [("ridge", "bench_ridge"), ("gbm", "bench_gbm"), ("lgbm", "bench_lgbm"),
                    ("xgb", "bench_xgboost"), ("cat", "bench_catboost")]:
        _, p_dir = fit_tab(D, cfg, key, tr, te, "regime", D.MC, tw=bool(tw), va=va, return_predictor=True)
        add(nm, p_dir, kind="direct")

    return pd.DataFrame(rows).set_index("model").sort_values("per_product_us")
