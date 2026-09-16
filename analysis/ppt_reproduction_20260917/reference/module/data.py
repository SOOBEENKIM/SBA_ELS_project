# -*- coding: utf-8 -*-
"""데이터 스키마·생성·로딩. 정통 DeepONet 재설계:
 branch = 시장상태(바스켓 vol·corr + 수익률곡선), trunk = 계약(전체 STRK 스케줄 + 배리어 + 쿠폰 + 만기).
 PI-DeepONet만 trunk에 평가좌표(S,τ,I)를 추가로 넣고 BS-PDE 물리를 건다.
 컬럼명은 raw DART와 직접 비교되도록 raw 이름 활용. 데이터셋은 CSV.
"""
import bisect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from util import file_manager as fm
from .schedule import NSTRK, SCHED_COLS, LZ_BARR_NEUTRAL, LZ_PMT_NEUTRAL

# ===== 원천 컬럼명 -> 데이터셋 컬럼명 (raw 이름/산식 활용) =====
RENAME = {
    "fair": "FAIR_VALUE/ISU_PRC_DETAIL",
    "coupon": "ANL_RTRN/100",
    "B": "BARR_1/100",
    "tenor": "TENOR",
    "isu_ord": "ISU_DT_ordinal",
    "mc": "MC",
    # pass-through
    "amt": "ACT_ISU_AMT", "sbrt": "SB_RT", "dvrt": "DV_RT", "prcp": "PRCP_GRTE_RT",
    "kigrc": "KNCK_IN_GRC_PRD", "issuer": "ISU_ORG", "risk": "RISK_GRADE",
    "ptype": "PRODUCT_TYPE", "rdmp": "RDMP_TYPE", "iyear": "ISU_DT_year",
    "imonth": "ISU_DT_month", "subdays": "SUB_END_DT-SUB_START_DT",
    "K": "STRK_1_last/100", "Kfirst": "STRK_1_first/100",
    "opt_type": "OPT_TYPE", "ki_yn": "KNCK_IN_YN",   # 구조 (STEP/LIZARD × 낙인/노낙인)
    "item": "ITEM_CD",   # 행 식별 인덱스 (훈련 미사용, 예측→상품 추적용)
}
INV = {v: k for k, v in RENAME.items()}


def _new(name):
    return RENAME.get(name, name)


# ===== 피처 그룹 =====
# tabular (ml.csv): 벤치마크 + 하이브리드 stage-2 마진모델
BASE = [_new(c) for c in ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_mean", "rho", "r", "B",
        "K", "Kfirst", "coupon", "tenor", "nobs", "cpn_spread", "b_over_k", "stepdown",
        "mom6m", "amt", "sbrt", "dvrt", "prcp", "kigrc", "iyear", "subdays"]]
REG = ["recent_margin", "recent_mktvol", "curve_level", "curve_slope", "curve_curv", "issue_intensity"]
CAT = [_new(c) for c in ["issuer", "risk", "ptype", "rdmp", "imonth"]]
# 인과(as-of, 미래정보 無) stage-2 드라이버: 발행사 마진편차(shrinkage)·신뢰도·IV−HV 스프레드·IV수준.
# 오차분석의 발행사 이질성·위험프리미엄을 명시적으로 모델링 (절대수준 regime 피처는 OOS 실패 → 상대신호만).
# 균등 m_issuer 는 m_issuer_ewma 가 흡수+노이즈라 제거(ablation: 빼면 +0.004). 5개 세트가 OOS 최적.
DRIVER = ["m_issuer_ewma", "m_issuer_cnt", "iv_hv_spread", "sig_iv_lr", "recent_mktvol"]

# 연산자망 (deeponet.csv)
VOLCORR = ["sig1", "sig2", "sig3", "rho12", "rho13", "rho23", "sig_eff"]   # branch: 바스켓 변동성·상관
UC = [f"u{j}" for j in range(10)]                                         # branch: 수익률곡선 10노드
STRK = [f"strk_{j}" for j in range(NSTRK)]                                # trunk: 오토콜 행사가 스케줄(/100, 패딩)
PMT = [f"pmt_{j}" for j in range(NSTRK)]                                  # trunk: 회차별 누적 지급률(소수)
LZB = [f"lz_barr_{j}" for j in range(NSTRK)]                              # trunk: 리자드 배리어(없으면 중립 1.2)
LZP = [f"lz_pmt_{j}" for j in range(NSTRK)]                               # trunk: 리자드 축소 지급률(없으면 0)

# ml BASE 에도 행사가·지급률 스케줄 포함. K/Kfirst 는 파생피처(b_over_k, stepdown)용으로 유지.
# 리자드 배열은 ml 에 넣지 않고 CAT 의 OPT_TYPE 으로 구조를 구분한다(GBM 은 범주형이 더 유리).
BASE = BASE + STRK + PMT
CONTRACT = STRK + PMT + LZB + LZP + [_new("B"), _new("coupon"), _new("tenor")]   # trunk 51
CAT = CAT + [_new("opt_type"), _new("ki_yn")]

INDEX = _new("item")      # ITEM_CD (행 식별 인덱스, 훈련 미사용)
TARGET = _new("fair")     # FAIR_VALUE/ISU_PRC_DETAIL
ANCHOR = _new("mc")       # MC
MARGIN = "recent_margin"
ORDER = _new("isu_ord")   # ISU_DT_ordinal
SRC_ORDER = "isu_ord"
TENOR = _new("tenor")     # TENOR
BARR = _new("B")          # BARR_1/100
COUPON = _new("coupon")   # ANL_RTRN/100
RF = "r"                  # 무위험금리 (물리용)
SIGEFF = "sig_eff"
CSV_ENC = "utf-8-sig"


def featnum(feat):
    return BASE + (REG if feat == "regime" else []) + (DRIVER if feat == "driver" else [])


def fill_lizard_neutral(df):
    """리자드 없는 회차의 NaN 을 중립값으로 채운다 (parquet 은 NaN 유지, CSV/모델 입력만 채움).
     lz_barr=1.2 는 스팟 위라 no-touch 조건이 즉시 깨져 '발동 불가' 가 명확하다."""
    for j in range(NSTRK):
        b, p = f"lz_barr_{j}", f"lz_pmt_{j}"
        if b in df.columns:
            df[b] = df[b].fillna(LZ_BARR_NEUTRAL)
        if p in df.columns:
            df[p] = df[p].fillna(LZ_PMT_NEUTRAL)
    return df


def _load_source():
    """원천 = data/els3_dataset.parquet (0_data build_source + 1_MC_recompute).
     행사가·지급률·리자드 스케줄이 이미 parquet 에 있으므로 raw 를 다시 읽지 않는다."""
    df = pd.read_parquet(fm.source()).sort_values(SRC_ORDER).reset_index(drop=True)
    o = df[SRC_ORDER].tolist()
    df["issue_intensity"] = [bisect.bisect_left(o, o[i]) - bisect.bisect_left(o, o[i] - 90)
                             for i in range(len(df))]
    missing = [c for c in SCHED_COLS if c not in df.columns]
    assert not missing, f"parquet 에 스케줄 컬럼 없음 (0_data build_source 재실행 필요): {missing[:4]}"
    return fill_lizard_neutral(df)


# ===== 0_data: 데이터셋 생성 (CSV) =====
def build_datasets():
    """원천 -> data/ml.csv, data/deeponet.csv 생성.
     ml = tabular(BASE+REG+CAT), deeponet = 연산자망 입력(곡선 u0-9 + vol·corr·sig_eff + 계약 + r). 둘 다 공통(fair·mc·recent_margin) 포함."""
    fm.ensure_dirs()
    df = _load_source()
    common = [INDEX, ORDER, TARGET, ANCHOR, MARGIN]   # INDEX(ITEM_CD) 선두 — 모든 데이터셋에 인덱스 컬럼

    def write(name, new_cols):
        new_cols = list(dict.fromkeys(new_cols))
        old_cols = [INV.get(c, c) for c in new_cols]
        out = df[old_cols].rename(columns=RENAME)
        out.to_csv(fm.dataset(name), index=False, encoding=CSV_ENC)
        return list(out.columns)

    drv = [c for c in DRIVER if c in df.columns]        # 드라이버는 존재할 때만 포함(inject_drivers 선행 시)
    cols_ml = write("ml", [INDEX] + BASE + REG + drv + CAT + common)
    # DeepONet 연산자망 입력: 곡선 u0-9 + vol·corr·sig_eff + 계약(strk0-11,B,coupon,tenor) + r (stage1 앵커/stage2 D.DON 공용)
    cols_dn = write("deeponet", [INDEX] + UC + VOLCORR + CONTRACT + [RF] + common)
    return {"ml": len(df), "deeponet": len(df),
            "columns": {"ml": cols_ml, "deeponet": cols_dn}}


# ===== 1_run: 통합 로딩 =====
def load(cfg):
    ml = pd.read_csv(fm.dataset("ml"), encoding=CSV_ENC).sort_values(ORDER).reset_index(drop=True)
    don = pd.read_csv(fm.dataset("deeponet"), encoding=CSV_ENC).sort_values(ORDER).reset_index(drop=True)
    n = len(ml)
    assert len(don) == n, "데이터셋 행수 불일치"
    for c in CAT:
        ml[c] = ml[c].astype(str).astype("category")

    dev = "cuda" if (cfg.get("device") != "cpu" and torch.cuda.is_available()) else "cpu"
    D = SimpleNamespace()
    D.n = n; D.SMAX = float(cfg["data"]["smax"]); D.DEV = dev
    D.ml = ml
    D.FAIR = ml[TARGET].values.astype("float32")
    D.MC = ml[ANCHOR].values.astype("float32")
    D.rm = ml[MARGIN].values.astype("float32")
    D.ITEM = ml[INDEX].astype(str).values   # ITEM_CD 인덱스 (훈련 미사용, 예측 추적용)
    D.ORD = ml[ORDER].values.astype(float)
    # 연산자망 입력 (branch=vol·corr+곡선, trunk=계약) — deeponet.csv
    D.VC = don[VOLCORR].values.astype("float32")      # (n, 7)
    D.CURVE = don[UC].values.astype("float32")        # (n, 10)
    D.CON = don[CONTRACT].values.astype("float32")    # (n, 51) = strk/pmt/lz_barr/lz_pmt ×12 + BARR + coupon + TENOR
    D.R = don[RF].values.astype("float32")            # r
    D.TEN = don[TENOR].values.astype("float32")       # 만기
    D.SIGEFF = don[SIGEFF].values.astype("float32")   # σ_eff
    # stage-2 잔차모델 입력 = deeponet.csv 특성 블록 [곡선|vol·corr·sig_eff|계약] (= 앵커 입력, 이론가 결정 특성만; ml aux/범주형 제외)
    D.DON = np.concatenate([D.CURVE, D.VC, D.CON], axis=1).astype("float32")   # (n, 68) = 10+7+51
    # CONTRACT 내 인덱스 (물리/payoff용)
    D.iK = CONTRACT.index(f"strk_{NSTRK - 1}")        # 만기 행사가 = strk_{last}
    D.iKlast = CONTRACT.index(f"strk_{NSTRK - 1}")
    D.iBARR = CONTRACT.index(BARR)
    D.iCOUPON = CONTRACT.index(COUPON)
    D.iTEN = CONTRACT.index(TENOR)
    D.WF = walk_forward(n, cfg["data"]["walk_forward"],
                        cfg["data"].get("val_frac", 0.0), cfg["data"].get("val_seed", 0))
    return D


def walk_forward(n, bounds, val_frac=0.0, val_seed=0):
    """확장윈도우 walk-forward. 반환: [(tr, va, te)].
     tr=학습(과거, val 제외) / va=validation(train에서 랜덤샘플, 미래참조 허용, early stopping용) / te=test(미래 OOS).
     val_frac=0 이면 va=빈배열 (기존 동작)."""
    rng = np.random.default_rng(val_seed)
    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr_all = np.arange(0, int(a * n))
        te = np.arange(int(a * n), int(b * n))
        if val_frac and len(tr_all) > 10:
            nv = max(1, int(round(val_frac * len(tr_all))))
            va = np.sort(rng.choice(tr_all, size=nv, replace=False))
            tr = np.setdiff1d(tr_all, va, assume_unique=True)
        else:
            va = np.empty(0, dtype=int); tr = tr_all
        folds.append((tr, va, te))
    return folds


def time_weights(D, tr, half_life=365.25):
    """시간감쇠 표본가중 0.5^(dt/half_life). half_life=None → 균등가중(1). dt=학습최신일 대비 경과일."""
    if half_life is None:
        return np.ones(len(tr), dtype="float32")
    return (0.5 ** ((D.ORD[tr].max() - D.ORD[tr]) / half_life)).astype("float32")


def to_tensor(a, dev):
    return torch.tensor(np.asarray(a, dtype="float32"), device=dev)


ZCLIP = 10.0    # 표준화값 한계 (정규 표본이면 절대 안 걸리는 수준)


def zstats(a, tr):
    """train 기준 표준화 통계."""
    return a[tr].mean(0), a[tr].std(0) + 1e-8


def znorm(a, m, s, clip=ZCLIP):
    """train 기준 표준화 + 클리핑.

     train 폴드에서 상수인 열(예: 리자드 lz_pmt_5/6/7 — 그 회차에 리자드가 붙은 상품이
     train 구간에 하나도 없다)은 std 가 eps(1e-8) 라, test 에 값이 나타나면 |z| 가
     2,500만까지 튄다. 트리 모델은 스케일 불변이라 무사하지만 tanh MLP 는 포화돼 발산한다.
     (실측: deeponet_hybrid_s2don stage2 R² −1.84)"""
    return np.clip((a - m) / s, -clip, clip)
