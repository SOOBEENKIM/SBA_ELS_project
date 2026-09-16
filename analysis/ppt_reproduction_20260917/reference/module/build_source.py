# -*- coding: utf-8 -*-
"""raw + cache → els3_dataset (원천 재유도). exp15 로직을 정식 이식하되 '현재 정의'로 빌드:
   변동성·상관 = 180일 역사(module.features), 수익률곡선 = KRW Nelson-Siegel. MC 는 여기서 계산하지 않음(1_MC_recompute).
 유니버스 = 3-star KRW 4구조(STEP/LIZARD × 낙인/노낙인). 노낙인은 B=1.0 으로 인코딩(항상 knocked-in).
 산출: data/els3_dataset.parquet (보조피처 + branch[sig/rho/sig_eff/u/r/curve] + fair + recent_mktvol
       + 구조[opt_type/ki_yn] + 기초자산[udl1..3/udl_key] + 스케줄[strk/pmt/lz_barr/lz_pmt ×12]). item·isu_ord 포함."""
import io, sys, json, time
from datetime import date
import numpy as np
import pandas as pd
import bisect

from util import file_manager as fm
from . import features as F
from .schedule import load_schedules, NSTRK, SCHED_COLS

STRUCTS = (("STEP", 1), ("STEP", 0), ("LIZARD", 1), ("LIZARD", 0))
FAIR_LO, FAIR_HI = 0.70, 1.05
TEN_LO, TEN_HI = 0.5, 5.0


def encode_barrier(ki_yn, barr_pct):
    """낙인형: BARR_1/100. 노낙인: 1.0.
     1.0 은 '만기 미상환이면 항상 worst 수취' 를 뜻하고, 엔진이 B>=1.0 을 명시적으로 always-KI 로 처리한다.
     노낙인인데 BARR_1 이 붙은 상품이 1건 있으나 KNCK_IN_YN 을 신뢰한다."""
    if int(ki_yn) == 1:
        return float(barr_pct) / 100.0 if pd.notna(barr_pct) else np.nan
    return 1.0


def needs_linear_pmt(pmts, c, ten, nonmono):
    """상환 스케줄이 계약 쿠폰을 담고 있지 않으면 선형 c*t (기존 엔진 동작) 로 대체해야 한다.

     (1) 누적 지급률 비단조 = raw 오류 (1회차 슬롯에 만기값)
     (2) 마지막 누적 지급률이 c*tenor 의 절반에도 못 미침 = 월지급형 등 '쿠폰 미기재'.
         이걸 그대로 쓰면 쿠폰이 통째로 빠져 mc 가 액면 대비 약 570원 낮아진다(실측).
     반대 방향(스케줄이 c*tenor 보다 많이 지급, 예: ANL_RTRN=0 인데 스케줄엔 지급률 있음)은
     스케줄이 정답이므로 건드리지 않는다."""
    if nonmono:
        return True
    if c <= 0 or ten <= 0:
        return False
    return bool(float(pmts[-1]) < 0.5 * float(c) * float(ten))


def structure_mask(ac, three_index):
    """3-star ∧ KRW ∧ (OPT_TYPE, KNCK_IN_YN) ∈ STRUCTS ∧ 공정가/만기 범위."""
    struct = pd.Series(False, index=ac.index)
    for o, k in STRUCTS:
        struct |= (ac["OPT_TYPE"].eq(o) & ac["KNCK_IN_YN"].eq(k))
    return (ac["ITEM_CD"].isin(three_index) & ac["CUR_CD"].eq("KRW") & struct
            & ac["fair"].between(FAIR_LO, FAIR_HI) & ac["tenor"].between(TEN_LO, TEN_HI))


def _safe(t):
    return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"


def build_source(save=True, verbose=True):
    CA, RAW = fm.CACHE, fm.RAW
    mapping = json.loads((CA / "udly_ticker_map.json").read_text(encoding="utf-8"))
    RET = {}
    for uid, t in mapping.items():
        if not t:
            continue
        f = CA / _safe(t)
        if t in RET or not f.exists():
            continue
        s = pd.read_parquet(f)["close"].dropna(); s = s[~s.index.duplicated()]
        RET[t] = np.log(s).diff()
    KRW = pd.read_parquet(CA / "krw_curve.parquet")[["call", "m3", "y10"]]

    ac = pd.read_csv(RAW / "LAKE_V2_DART_AUTO_CALL.csv", low_memory=False)
    sc = pd.read_csv(RAW / "LAKE_V2_DART_SCHD_INFO.csv", low_memory=False)
    ud = pd.read_csv(RAW / "LAKE_V2_DART_UDLY_INFO.csv", low_memory=False)
    for c in ("ISU_DT", "MAT_DT", "SUB_START_DT", "SUB_END_DT"):
        ac[c] = pd.to_datetime(ac[c], errors="coerce")
    ac["tenor"] = (ac["MAT_DT"] - ac["ISU_DT"]).dt.days / 365.25
    ac["fair"] = ac["FAIR_VALUE"] / ac["ISU_PRC_DETAIL"]
    nu = ud.groupby("ITEM_CD")["UDLY_ID"].nunique(); three = nu[nu == 3].index
    u3map = ud[ud.ITEM_CD.isin(three)].groupby("ITEM_CD")["UDLY_ID"].apply(list)
    SCHED = load_schedules()
    barr_pct = sc[sc.SCHD_TYPE == 1].groupby("ITEM_CD")["BARR_1"].min()
    cand = ac[structure_mask(ac, three)]
    if verbose:
        print("4구조 후보:", len(cand))
        print("  ", cand.groupby(["OPT_TYPE", "KNCK_IN_YN"]).size().to_dict())
    drops = {}

    def mom6m(rets, dt, win=126):
        vals = []
        for r in rets:
            if r is None:
                return 0.0
            w = r[r.index < dt].tail(win)
            if len(w) < 40:
                return 0.0
            vals.append(float(w.sum()))
        return float(np.mean(vals)) if vals else 0.0

    rows = []; t0 = time.time()
    for it, rw in cand.set_index("ITEM_CD").iterrows():
        try:
            ts = [mapping.get(x) for x in u3map.get(it, [])]
            if len(ts) != 3 or any(t is None for t in ts):
                drops["티커 매핑 실패"] = drops.get("티커 매핑 실패", 0) + 1; continue
            rets = [RET.get(t) for t in ts]
            if any(r is None for r in rets):
                drops["가격이력 없음"] = drops.get("가격이력 없음", 0) + 1; continue
            dt = rw.ISU_DT
            sigs = [F.vol180(r, dt) for r in rets]                 # 현재 정의: 180일 역사 변동성
            if any(pd.isna(sigs)):
                drops["180일 변동성 부족"] = drops.get("180일 변동성 부족", 0) + 1; continue
            corr = F.corr180(rets, dt)                             # 180일 역사 상관
            if corr is None:
                drops["180일 상관 부족"] = drops.get("180일 상관 부족", 0) + 1; continue
            if it not in SCHED.index or not bool(SCHED.at[it, "sched_ok"]):
                why = SCHED.at[it, "sched_drop"] if it in SCHED.index else "스케줄 없음"
                drops[f"스케줄: {why}"] = drops.get(f"스케줄: {why}", 0) + 1; continue
            if bool(SCHED.at[it, "lz_from_prev"]):                 # FROM_PREV 리자드는 엔진 미지원
                drops["리자드 FROM_PREV"] = drops.get("리자드 FROM_PREV", 0) + 1; continue
            srow = SCHED.loc[it]
            nobs_ = int(srow["nobs"])
            strikes = [float(srow[f"strk_{j}"]) for j in range(nobs_)]
            B = encode_barrier(rw.KNCK_IN_YN, barr_pct.get(it, np.nan))
            c = rw.ANL_RTRN / 100; ten = rw.tenor
            if any(pd.isna(x) for x in [B, c, ten]):
                drops["배리어/쿠폰/만기 결측"] = drops.get("배리어/쿠폰/만기 결측", 0) + 1; continue
            beta = F.krw_beta(KRW.asof(dt).values)                 # KRW NS 곡선
            if beta is None:
                continue
            u = F.krw_curve_nodes(beta); r = float(F.zero_curve(beta, np.array([ten]))[0])
            iu = np.triu_indices(3, 1); rr = np.sort(np.clip(corr[iu], -0.999, 0.999))
            rho12, rho13, rho23 = float(rr[0]), float(rr[1]), float(rr[2])
            rho_v = float(np.mean(rr))
            ss = np.sort(sigs); sig1, sig2, sig3 = float(ss[0]), float(ss[1]), float(ss[2])
            smean = float(np.mean(sigs)); sig_eff = smean * float(np.sqrt(1.0 + max(0.0, 1.0 - rho_v)))
            Klast = float(strikes[-1]); Kfst = float(strikes[0])
            sd = (rw.SUB_END_DT - rw.SUB_START_DT).days if pd.notna(rw.SUB_END_DT) and pd.notna(rw.SUB_START_DT) else np.nan
            rec = dict(item=it, sig1=sig1, sig2=sig2, sig3=sig3, rho12=rho12, rho13=rho13, rho23=rho23,
                       sig_mean=smean, sig_max=sig3, sig_min=sig1, rho=rho_v, sig_eff=sig_eff,
                       cpn_spread=float(c - r), b_over_k=float(B / Klast) if Klast > 0 else 1.0,
                       stepdown=float(Kfst - Klast), mom6m=mom6m(rets, dt), isu_ord=int(dt.toordinal()),
                       r=r, B=float(B), Kfirst=Kfst, K=Klast, coupon=float(c), tenor=float(ten), nobs=nobs_,
                       fair=float(rw.fair), issuer=str(rw.ISU_ORG), risk=str(rw.RISK_GRADE),
                       ptype=str(rw.PRODUCT_TYPE), rdmp=str(rw.RDMP_TYPE), imonth=str(int(dt.month)),
                       amt=float(rw.ACT_ISU_AMT) if pd.notna(rw.ACT_ISU_AMT) else np.nan,
                       sbrt=float(rw.SB_RT) if pd.notna(rw.SB_RT) else np.nan,
                       dvrt=float(rw.DV_RT) if pd.notna(rw.DV_RT) else np.nan,
                       prcp=float(rw.PRCP_GRTE_RT) if pd.notna(rw.PRCP_GRTE_RT) else np.nan,
                       kigrc=float(rw.KNCK_IN_GRC_PRD) if pd.notna(rw.KNCK_IN_GRC_PRD) else np.nan,
                       iyear=float(dt.year), subdays=float(sd))
            for j in range(10):
                rec[f"u{j}"] = float(u[j])
            rec["curve_level"] = float(u.mean()); rec["curve_slope"] = float(u[9] - u[0])
            rec["curve_curv"] = float(2 * u[4] - u[0] - u[9])
            rec["opt_type"] = str(rw.OPT_TYPE)
            rec["ki_yn"] = int(rw.KNCK_IN_YN)
            ts_sorted = sorted(ts)                       # 기초자산 티커 (평가 분해용 식별자)
            rec["udl1"], rec["udl2"], rec["udl3"] = ts_sorted
            rec["udl_key"] = "|".join(ts_sorted)
            for cname in SCHED_COLS:
                rec[cname] = float(srow[cname]) if pd.notna(srow[cname]) else np.nan
            # 스케줄이 계약 쿠폰을 담지 못한 경우(raw 비단조 0.67% + 월지급형 등 13%)는
            # 선형 쿠폰 c*t 로 대체 = 기존 엔진 동작
            rec["pmt_linear"] = int(needs_linear_pmt(
                [rec[f"pmt_{q}"] for q in range(nobs_)], c, ten, bool(srow["pmt_nonmono"])))
            if rec["pmt_linear"]:
                Nd = int(round(ten * 365))
                od = np.clip(np.round(np.arange(1, nobs_ + 1) * (ten / nobs_) * 365).astype(int), 1, Nd)
                for j, v in enumerate(c * (od / 365.0)):
                    rec[f"pmt_{j}"] = float(v)
                for j in range(nobs_, NSTRK):            # 패딩은 마지막값 forward-fill
                    rec[f"pmt_{j}"] = rec[f"pmt_{nobs_ - 1}"]
            rows.append(rec)
        except Exception as e:
            key = f"예외: {type(e).__name__}"
            drops[key] = drops.get(key, 0) + 1
            continue
    df = pd.DataFrame(rows).sort_values("isu_ord").reset_index(drop=True)
    if verbose:
        print(f"built {len(df)} products in {time.time()-t0:.0f}s")
        print("  구조별:", df.groupby(["opt_type", "ki_yn"]).size().to_dict())
        if drops:
            print("  탈락 사유:", dict(sorted(drops.items(), key=lambda x: -x[1])))
    # recent_mktvol: 발행 전 90일 발행분 평균 sig_mean (인과적)
    o = df["isu_ord"].tolist(); sm = df["sig_mean"].values; rmv = np.zeros(len(df))
    for i in range(len(df)):
        hi = bisect.bisect_left(o, o[i]); lo = bisect.bisect_left(o, o[i] - 90)
        rmv[i] = sm[lo:hi].mean() if hi > lo else (sm[:hi].mean() if hi > 0 else sm[i])
    df["recent_mktvol"] = rmv.astype("float32")
    if save:
        out = fm.DATA / "els3_dataset.parquet"
        df.to_parquet(out)
        if verbose:
            print("saved", out, "rows", len(df))
    return df
