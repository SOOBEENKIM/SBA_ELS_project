# -*- coding: utf-8 -*-
"""ATM 내재변동성 + 차용 기간구조 -> ELS 만기 변동성 σ(T).

 보유 자료는 기초자산별 **단일 만기 ATM IV** 뿐이다 (data/raw/iv_daily_atm, 만기·행사가 차원 없음).
 그래서 기간구조는 공표 변동성지수에서 **형태만** 빌려온다:

     σ_udl(T) = ATM_udl · σ_VIX(T) / σ_VIX(T_ref)

 σ_VIX(T) 는 같은 날의 VIX9D·VIX·VIX3M·VIX6M(9·30·93·182일)으로 만든 **총분산 곡선**에서 읽는다.

     w(T) = σ²(T)·T,   σ(T) = sqrt(w(T) / T)

 - 노드 사이는 w 에 대해 **선형보간** — IV 자체를 보간하면 캘린더 차익 부재가 깨지므로 반드시 w 축에서.
 - 노드는 w 가 비감소가 되도록 누적최댓값으로 정리한다 (역전 구간의 캘린더 차익 제거).
 - 최장 노드(182일) 이후는 마지막 구간 기울기로 **선형 외삽** = 6개월 이후 선도분산 일정.
   ELS 만기(대부분 3년)는 항상 이 외삽 구간이다.

 한계(자문 시 확인 필요):
   - 미국 S&P 500 의 기간구조 형태를 유로·홍콩·한국·일본 지수에 그대로 차용한다.
   - 6개월 이후는 시장 자료가 아니라 외삽이다.
   - ATM 이라 하방 스큐는 담지 못한다.
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd

from util import file_manager as fm

T_REF_DAYS = 30                         # 보유 ATM IV 의 만기 가정 (1개월, 관례)
VIX_NODES = [("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 182)]

# ELS 기초자산(야후식 티커) -> ATM IV 원천 notion_ric.
#   ^STOXX50E 는 원천에 EURO STOXX 50(.STOXX50E) 이 없어 STOXX Europe 50(.STOXX50) 으로 대용한다.
#   다른 지수(영국·스위스 방어주 포함)라 IV 가 HV 보다 평균 2.8%p 낮다 -> variant 로 분리해 다룬다.
IVMAP = {"^KS200": ".KS200", "^GSPC": ".SPX", "^HSCE": ".HSCE", "^HSI": ".HSI",
         "^N225": ".N225", "^NDX": ".NDX", "^GDAXI": ".GDAXI", "^STOXX50E": ".STOXX50"}
PROXY_SUSPECT = {"^STOXX50E"}


def load_atm_iv(root=None):
    """notion_ric -> 날짜 인덱스 IV Series (소수, 0.18 = 18%)."""
    root = Path(root) if root else fm.ROOT
    files = sorted(glob.glob(str(root / "data/raw/iv_daily_atm/iv_daily_atm/*.csv")))
    if not files:
        raise FileNotFoundError("data/raw/iv_daily_atm/iv_daily_atm/*.csv 가 없다 (zip 해제 필요)")
    big = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    big["calc_date"] = pd.to_datetime(big["calc_date"])
    out = {}
    for ric, g in big.groupby("notion_ric"):
        s = g.set_index("calc_date")["iv"].sort_index()
        out[ric] = s[~s.index.duplicated(keep="last")].astype(float)
    return out


def load_vix_term(root=None):
    """VIX 기간구조 노드 패널 (열 = 노드 라벨, 값 = 소수 변동성)."""
    root = Path(root) if root else fm.ROOT
    cols = {}
    for lab, _ in VIX_NODES:
        v = pd.read_parquet(root / "data/cache/volidx" / f"{lab}.parquet").iloc[:, 0]
        v.index = pd.to_datetime(v.index)
        cols[lab] = v[~v.index.duplicated(keep="last")].astype(float) / 100.0
    return pd.DataFrame(cols).sort_index()


def total_variance_curve(sig_nodes, days_nodes):
    """노드 (σ, 일수) -> (T[년], w) 로, w 는 캘린더 차익이 없도록 비감소로 정리한다."""
    s = np.asarray(sig_nodes, float)
    t = np.asarray(days_nodes, float) / 365.0
    ok = np.isfinite(s) & (s > 0)
    s, t = s[ok], t[ok]
    if len(t) < 2:
        return None
    o = np.argsort(t)
    t, s = t[o], s[o]
    w = np.maximum.accumulate(s ** 2 * t)            # 비감소 강제
    return t, w


def sigma_at(t, w, T):
    """총분산 곡선에서 만기 T[년] 의 변동성. 노드 사이 선형보간, 마지막 노드 이후 선형 외삽."""
    if T <= t[0]:
        return float(np.sqrt(w[0] / t[0]))          # 최단 노드보다 짧으면 그 노드의 변동성
    if T <= t[-1]:
        return float(np.sqrt(np.interp(T, t, w) / T))
    slope = max((w[-1] - w[-2]) / (t[-1] - t[-2]), 0.0)   # 선도분산 (음수 불가)
    return float(np.sqrt((w[-1] + slope * (T - t[-1])) / T))


def term_ratio(vix_row, T, t_ref_days=T_REF_DAYS):
    """σ_VIX(T) / σ_VIX(T_ref). vix_row 는 VIX_NODES 순서의 소수 변동성."""
    cur = total_variance_curve(vix_row, [d for _, d in VIX_NODES])
    if cur is None:
        return np.nan
    t, w = cur
    return sigma_at(t, w, T) / sigma_at(t, w, t_ref_days / 365.0)


def asof(series, dt):
    """dt 이하의 마지막 관측값 (미래정보 차단)."""
    s = series[series.index <= dt]
    return float(s.iloc[-1]) if len(s) else np.nan
