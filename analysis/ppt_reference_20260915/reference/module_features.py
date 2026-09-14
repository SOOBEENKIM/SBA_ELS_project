# -*- coding: utf-8 -*-
"""공용 피처 유틸 (DRY) — KRW Nelson-Siegel 곡선 + 180일 역사 변동성·상관.
 0_data(데이터셋 branch 피처 재유도)와 MC 재산출 노트북이 공유한다.
 곡선/변동성 정의는 데이터셋을 만든 프라이서와 동일하게 유지(재현성)."""
import numpy as np
import pandas as pd

LAM = 1.5
KR_TAUS = np.array([0.08, 0.25, 10.0])                                  # 콜·3M 은행간·10Y 국채 (FRED)
MSAMP = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10], dtype=float)   # branch 곡선 노드 u0..u9
VOL_WIN = 180
CORR_WIN = 180


def nsb(t):
    """Nelson-Siegel 기저 [level, slope, curvature] (LAM 고정)."""
    t = np.atleast_1d(np.asarray(t, float)); x = t / LAM; e = np.exp(-x)
    return np.stack([np.ones_like(t), (1 - e) / x, (1 - e) / x - e], 1)


def zero_curve(beta, times):
    """NS 제로금리(소수) 곡선 = nsb(times)@beta/100."""
    return (nsb(np.maximum(np.atleast_1d(times), 1e-6)) @ beta) / 100.0


def krw_beta(row):
    """row=[call, m3, y10] (발행일 asof 값) → NS beta. NaN 있으면 None."""
    row = np.asarray(row, float)
    if np.isnan(row).any():
        return None
    beta, *_ = np.linalg.lstsq(nsb(KR_TAUS), row, rcond=None)
    return beta


def krw_curve_nodes(beta):
    """branch 곡선 노드 u0..u9 = MSAMP 만기점의 KRW NS 제로금리."""
    return zero_curve(beta, MSAMP).astype(np.float32)


def chol_psd(corr):
    """PSD 보정 촐레스키."""
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(corr); ev = np.clip(ev, 1e-8, None)
        c2 = V @ np.diag(ev) @ V.T; d = np.sqrt(np.diag(c2)); c2 = c2 / np.outer(d, d)
        return np.linalg.cholesky(c2)


def vol180(ret, dt):
    """발행 전 180 거래일 로그수익률 연율 변동성(std×√252). 표본<60이면 nan."""
    w = ret[ret.index < dt].tail(VOL_WIN)
    return float(w.std() * np.sqrt(252)) if len(w) >= 60 else np.nan


def corr180(rets, dt):
    """발행 전 180 거래일 정렬 3자산 Pearson 상관행렬. 자료 부족시 None."""
    dfr = pd.concat([r[r.index < dt] for r in rets], axis=1).dropna().tail(CORR_WIN)
    if dfr.shape[1] < 3 or len(dfr) < 60:
        return None
    C = np.corrcoef(dfr.values, rowvar=False)
    C = np.clip(C, -0.999, 0.999); np.fill_diagonal(C, 1.0)
    return C
