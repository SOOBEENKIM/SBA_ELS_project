# -*- coding: utf-8 -*-
"""평가 지표: R2/MAE/RMSE/Spearman + 하이브리드 stage별 R2."""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)   # 절대 백분율 오차
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE%": float(ape.mean() * 100),                              # 평균 절대 %오차 (핵심)
        "MdAPE%": float(np.median(ape) * 100),                         # 중앙값 %오차 (이상치 강건)
        "Bias%": float(((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)).mean() * 100),  # 부호있는 %편향
        "Spearman": float(spearmanr(y_true, y_pred).correlation),
    }


def metrics_by(y_true, y_pred, group, min_n=30):
    """그룹 라벨별 지표 표. 표본 min_n 미만 그룹은 '(small)' 하나로 합친다.
     반환: index=group, columns=[n, R2, MAE, RMSE, MAPE%, MdAPE%, Bias%, Spearman], n 내림차순."""
    import pandas as pd
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    g = pd.Series(np.asarray(group, dtype=object)).astype(str)
    cnt = g.value_counts()
    g = g.where(g.map(cnt) >= min_n, "(small)")
    rows = {}
    for lab in g.unique():
        m = (g == lab).values
        if m.sum() < 2:                                   # R2/Spearman 계산 불가
            continue
        rows[lab] = {"n": int(m.sum()), **metrics(y_true[m], y_pred[m])}
    out = pd.DataFrame(rows).T
    out["n"] = out["n"].astype(int)
    return out.sort_values("n", ascending=False)


def stage_r2(df):
    """2단계 하이브리드 예측 DataFrame(mc_true/mc_pred/resid_true/resid_pred/y_true/y_pred)에서 stage별 R2.
     직접(단일 단계) 벤치마크엔 stage 컬럼이 없어 None 반환(→ stage_r2 그림 제외)."""
    need = {"mc_true", "mc_pred", "resid_true", "resid_pred", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        return None
    return {
        "stage1_MC_R2": r2_score(df["mc_true"], df["mc_pred"]),
        "stage2_resid_R2": r2_score(df["resid_true"], df["resid_pred"]),
        "final_MC_R2": r2_score(df["y_true"], df["y_pred"]),
    }
