# -*- coding: utf-8 -*-
"""머신러닝 벤치마크 — 전부 **이론가(MC) 예측**(fair 아님).
 (1) 직접(단일 단계) 벤치마크: ml.csv 전 특성(regime)으로 MC 를 바로 예측 — Ridge/GBM/LightGBM/XGBoost/CatBoost.
 (2) XGB 하이브리드(2단계): stage-1 앵커=XGB→MC(MC_hat), stage-2=XGB→이론가 잔차(MC_true−MC_hat), 마진 없음.
     앵커 입력을 DeepONet 앵커와 동일하게 맞춤(순수 아키텍처 비교): 곡선 u0-9 + vol·corr + 전체 계약(strk0-11,B,coupon,tenor).
 → 직접 vs 2단계 하이브리드 비교."""
import numpy as np
from xgboost import XGBRegressor

from module.pipeline import predict_hybrid, predict_direct_tab
from module.data import time_weights

# 직접(단일 단계) 벤치마크: fit_tab model_key -> 예측 CSV/가중치 이름
_DIRECT = {"ridge": "bench_ridge", "gbm": "bench_gbm", "lgbm": "bench_lgbm",
           "xgb": "bench_xgboost", "cat": "bench_catboost"}


def _xgb_anchor(D, cfg, tr, va, te, save_path=None):
    """stage-1 앵커: XGBoost가 무차익가 MC를 예측.
    입력을 DeepONet-Curve 앵커와 동일하게 맞춤 → 순수 아키텍처 비교.
    X = [곡선 u0-9 | vol·corr(7) | 계약(strk0-11 + B + coupon + tenor)]. va→val 조기종료. save_path→.pkl 저장."""
    XP = cfg["tabular"]["xgb"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    w = time_weights(D, tr) if cfg["data"]["time_decay"] else None
    es = va is not None and len(va) > 0
    kw = dict(n_estimators=XP["n_estimators"], learning_rate=XP["learning_rate"],
              max_depth=XP["max_depth"], subsample=XP["subsample"],
              colsample_bytree=XP["colsample_bytree"], n_jobs=0, random_state=cfg["seed"])
    if es:
        kw["early_stopping_rounds"] = cfg["train"].get("es_rounds", 50)
    m = XGBRegressor(**kw)
    if es:
        m.fit(X[tr], D.MC[tr], sample_weight=w, eval_set=[(X[va], D.MC[va])], verbose=False)
    else:
        m.fit(X[tr], D.MC[tr], sample_weight=w)
    if save_path:
        import os, joblib
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": m, "kind": "xgb_anchor_curve_vc_con"}, save_path + ".pkl")
    return lambda idx: m.predict(X[idx]).astype("float32")


def run(D, cfg):
    out = {}
    # (1) 직접(단일 단계) 벤치마크: ml.csv 전 특성으로 공정가(FAIR) 직접 예측
    for key, name in _DIRECT.items():
        out[name] = predict_direct_tab(D, cfg, key, name=name, target=D.FAIR)
    # (2) XGB 하이브리드(2단계): XGB 앵커(MC_hat) + recent_margin + XGB 잔차(FAIR−MC_hat−rm)
    out["xgb_hybrid"] = predict_hybrid(D, cfg, _xgb_anchor, name="xgb_hybrid",
                                       target=D.FAIR, use_margin=True)
    return out
