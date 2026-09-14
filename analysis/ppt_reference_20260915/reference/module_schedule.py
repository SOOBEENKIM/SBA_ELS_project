# -*- coding: utf-8 -*-
"""raw SCHD_INFO(SCHD_TYPE=1) -> 상품별 고정폭 스케줄 테이블 (DRY: build_source 와 data 투영이 공유).

 단위: STRK_1 / LZRD_BARR 는 raw 가 퍼센트 -> /100. PMT_1 / LZRD_PMT 는 raw 가 이미 소수 -> 그대로.
 패딩: strk/pmt 는 마지막값 forward-fill(만기 조건이 반복되는 것과 같은 의미).
       리자드는 '그 회차에만 붙는 조항'이므로 forward-fill 하지 않고 NaN 으로 둔다.
 제외: 회차 2 미만 / NSTRK 초과 / 행사가·지급률 결측 (sched_ok=False, sched_drop 에 사유)."""
import numpy as np
import pandas as pd

from util import file_manager as fm

NSTRK = 12
LZ_BARR_NEUTRAL = 1.2      # 스팟(1.0) 위 -> no-touch 조건이 즉시 깨져 발동 불가 (관측 최대 0.90 바깥)
LZ_PMT_NEUTRAL = 0.0

SCHED_COLS = ([f"strk_{j}" for j in range(NSTRK)] + [f"pmt_{j}" for j in range(NSTRK)]
              + [f"lz_barr_{j}" for j in range(NSTRK)] + [f"lz_pmt_{j}" for j in range(NSTRK)])


def _pad_ffill(vals, k=NSTRK):
    """길이 k 로 고정: 짧으면 마지막값 forward-fill."""
    v = list(vals[:k])
    return v + [v[-1]] * (k - len(v))


def _pad_nan(pairs, k=NSTRK):
    """(회차 인덱스, 값) 목록을 길이 k 배열의 해당 위치에만 채우고 나머지는 NaN."""
    out = [np.nan] * k
    for i, v in pairs:
        if 0 <= i < k:
            out[i] = v
    return out


def build_schedules(sc):
    """raw SCHD_INFO DataFrame -> ITEM_CD 인덱스 고정폭 스케줄 테이블.
     컬럼: nobs, SCHED_COLS(48), lz_from_prev, sched_ok, sched_drop."""
    s1 = sc[sc["SCHD_TYPE"] == 1].sort_values(["ITEM_CD", "SEQ"])
    rows = []
    for it, g in s1.groupby("ITEM_CD", sort=True):
        strk = g["STRK_1"].tolist(); pmt = g["PMT_1"].tolist()
        lzb = g["LZRD_BARR"].tolist(); lzp = g["LZRD_PMT"].tolist()
        term = g["LZRD_TERM"].tolist()
        n = len(strk)
        drop = ""
        if n < 2:
            drop = "nobs<2"
        elif n > NSTRK:
            drop = f"nobs>{NSTRK}"
        elif any(pd.isna(x) for x in strk):
            drop = "strk_na"
        elif any(pd.isna(x) for x in pmt):
            drop = "pmt_na"
        if drop:
            rec = {c: np.nan for c in SCHED_COLS}
            rec.update(ITEM_CD=it, nobs=n, lz_from_prev=False, pmt_nonmono=False,
                       sched_ok=False, sched_drop=drop)
            rows.append(rec)
            continue
        # 누적 지급률은 회차가 갈수록 줄 수 없다. raw 에 1회차가 만기값으로 들어간 오류가 있다(0.67%).
        pv = [float(x) for x in pmt]
        nonmono = any(b < a - 1e-9 for a, b in zip(pv[:-1], pv[1:]))
        pairs_b = [(i, float(b) / 100.0) for i, b in enumerate(lzb) if pd.notna(b)]
        pairs_p = [(i, float(p) if pd.notna(p) else 0.0)
                   for i, (p, b) in enumerate(zip(lzp, lzb)) if pd.notna(b)]
        rec = {"ITEM_CD": it, "nobs": n,
               "lz_from_prev": any(t == "FROM_PREV" for t in term),
               "pmt_nonmono": nonmono, "sched_ok": True, "sched_drop": ""}
        for j, v in enumerate(_pad_ffill([float(x) / 100.0 for x in strk])):
            rec[f"strk_{j}"] = v
        for j, v in enumerate(_pad_ffill([float(x) for x in pmt])):
            rec[f"pmt_{j}"] = v
        for j, v in enumerate(_pad_nan(pairs_b)):
            rec[f"lz_barr_{j}"] = v
        for j, v in enumerate(_pad_nan(pairs_p)):
            rec[f"lz_pmt_{j}"] = v
        rows.append(rec)
    return pd.DataFrame(rows).set_index("ITEM_CD")


def load_schedules():
    """raw CSV 를 읽어 build_schedules 적용."""
    sc = pd.read_csv(fm.RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
    return build_schedules(sc)
