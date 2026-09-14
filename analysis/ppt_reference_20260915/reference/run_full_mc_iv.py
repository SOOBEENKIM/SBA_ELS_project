# -*- coding: utf-8 -*-
"""전량 mc_iv 산출 (100k 경로, GPU) — 검증된 노트북 로직의 배치 실행본.
 체크포인트(scratch/mc_iv_ckpt.npz)로 재개 가능. 완료시 result/statistics/mc_iv_full.csv 저장."""
import glob, io, sys, time, urllib.request
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, RAW = ROOT / "data", ROOT / "data/cache", ROOT / "data/raw"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NPATH = 100_000
DISC = (sys.argv[1] if len(sys.argv) > 1 else "boot").lower()   # 'ns' | 'boot'
print(f"device {DEV} | NPATH {NPATH} | discount {DISC}", flush=True)

IVMAP = {"^KS200": ".KS200", "^GSPC": ".SPX", "^HSCE": ".HSCE", "^HSI": ".HSI",
         "^N225": ".N225", "^NDX": ".NDX", "^GDAXI": ".GDAXI", "^STOXX50E": ".STOXX50"}
KR_TAUS = np.array([0.08, 0.25, 10.0])
FRED_KR = {"call": "IRSTCI01KRM156N", "m3": "IR3TIB01KRM156N", "y10": "IRLTLT01KRM156N"}


def _safe(t): return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"

def load_RET(tickers):
    RET = {}
    for t in tickers:
        f = CACHE / _safe(t)
        if f.exists():
            s = pd.read_parquet(f)["close"].dropna(); s = s[~s.index.duplicated()]
            RET[t] = np.log(s).diff()
    return RET

def load_IV():
    files = sorted(glob.glob(str(RAW / "iv_daily_atm/iv_daily_atm/*.csv")))
    big = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    big["calc_date"] = pd.to_datetime(big["calc_date"]); iv = {}
    for ric, g in big.groupby("notion_ric"):
        s = g.set_index("calc_date")["iv"].sort_index(); iv[ric] = s[~s.index.duplicated(keep="last")]
    return iv

def fred_csv(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode()
    d = pd.read_csv(io.StringIO(raw)); d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"]); d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index("date")["v"].dropna()

def load_krw_curve():
    cp = CACHE / "krw_curve_fred.parquet"
    if cp.exists(): return pd.read_parquet(cp)
    cur = pd.DataFrame({k: fred_csv(s) for k, s in FRED_KR.items()}).sort_index().ffill(); cur.to_parquet(cp); return cur

def vol180(ret, dt):
    w = ret[ret.index < dt].tail(180)
    return float(w.std() * np.sqrt(252)) if len(w) >= 60 else np.nan

def corr180(rets, dt):
    dfr = pd.concat([r[r.index < dt] for r in rets], axis=1, sort=True).dropna().tail(180)
    if dfr.shape[1] < 3 or len(dfr) < 60: return None
    C = np.corrcoef(dfr.values, rowvar=False); C = np.clip(C, -0.999, 0.999); np.fill_diagonal(C, 1.0); return C

def chol_psd(corr):
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(corr); ev = np.clip(ev, 1e-8, None)
        c2 = V @ np.diag(ev) @ V.T; d = np.sqrt(np.diag(c2)); c2 = c2 / np.outer(d, d); return np.linalg.cholesky(c2)

def boot_z(row):
    row = np.asarray(row, float)
    if np.isnan(row).any(): return None
    z = row / 100.0
    return lambda times: np.interp(np.atleast_1d(times), KR_TAUS, z)

def _nsb(t):
    t = np.atleast_1d(np.asarray(t, float)); x = t / 1.5; e = np.exp(-x)
    return np.stack([np.ones_like(t), (1 - e) / x, (1 - e) / x - e], 1)

def ns_z(row):
    row = np.asarray(row, float)
    if np.isnan(row).any(): return None
    beta, *_ = np.linalg.lstsq(_nsb(KR_TAUS), row, rcond=None)
    return lambda times: (_nsb(np.maximum(np.atleast_1d(times), 1e-6)) @ beta) / 100.0

def _setup(sigs, corr, disc_z, B, strikes, ten, c, pmts, lz_barr, lz_pmt):
    nobs = len(strikes); N = int(round(ten * 365)); dt = 1 / 365
    obs_day = np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N); obs_t = obs_day / 365.0
    times = np.arange(1, N + 1) * dt
    DF = np.exp(-disc_z(times) * times)
    fdt = (-(np.diff(np.concatenate([[0.0], np.log(DF)])))).astype(np.float32)
    sig = np.asarray(sigs, np.float32)
    drift = (fdt[:, None] - (0.5 * sig ** 2 * np.float32(dt))[None, :]).astype(np.float32)
    rate = np.asarray(c * obs_t if pmts is None else pmts, np.float32)[:nobs]
    lzb = (np.full(nobs, np.nan, np.float32) if lz_barr is None else np.asarray(lz_barr, np.float32)[:nobs])
    lzp = (np.zeros(nobs, np.float32) if lz_pmt is None else np.nan_to_num(np.asarray(lz_pmt, np.float32))[:nobs])
    on = np.isfinite(lzb)
    return dict(N=N, nobs=nobs, obs_day=obs_day, DF=DF.astype(np.float32),
                L=chol_psd(np.asarray(corr)).astype(np.float32), sig=sig, drift=drift,
                sq=np.float32(np.sqrt(dt)), logB=float(np.log(B)),
                logK=np.log(np.asarray(strikes, np.float32)), rate=rate, lzp=lzp, lz_on=on,
                loglzb=np.log(np.where(on, lzb, 1.0)).astype(np.float32))

def mc_daily_t(sigs, corr, disc_z, B, strikes, ten, n=NPATH, seed=0, c=0.0, pmts=None,
               lz_barr=None, lz_pmt=None, dev=None, path_chunk=20_000, tblock=512):
    P = _setup(sigs, corr, disc_z, B, strikes, ten, c, pmts, lz_barr, lz_pmt)
    d = torch.device(dev or DEV); f32 = torch.float32
    N, nobs = P["N"], P["nobs"]
    L = torch.as_tensor(P["L"] * (P["sig"] * P["sq"])[:, None], device=d)
    cdrift = torch.as_tensor(P["drift"], device=d).cumsum(0).T.contiguous()
    DF = torch.as_tensor(P["DF"], device=d); logK = torch.as_tensor(P["logK"], device=d)
    rate = torch.as_tensor(P["rate"], device=d); lzp = torch.as_tensor(P["lzp"], device=d)
    loglzb = torch.as_tensor(P["loglzb"], device=d); lz_on = torch.as_tensor(P["lz_on"], device=d)
    obs_day = torch.as_tensor(P["obs_day"], device=d, dtype=torch.long)
    has_lz = bool(P["lz_on"].any()); always_ki = bool(B >= 1.0); logB = P["logB"]
    g = torch.Generator(device=d).manual_seed(int(seed))
    tot = torch.zeros((), device=d, dtype=torch.float64); done = 0
    while done < n:
        m = min(path_chunk, n - done)
        carry = torch.zeros(3, m, 1, device=d)
        wmin = torch.full((m,), float("inf"), device=d)
        wobs = torch.empty(m, nobs, device=d); wrun = torch.empty(m, nobs, device=d) if has_lz else None
        for t0 in range(0, N, tblock):
            T = min(tblock, N - t0)
            X = torch.randn(3, m, T, device=d, dtype=f32, generator=g)
            X[2] = L[2, 0] * X[0] + L[2, 1] * X[1] + L[2, 2] * X[2]
            X[1] = L[1, 0] * X[0] + L[1, 1] * X[1]; X[0] = L[0, 0] * X[0]
            X = X.cumsum(2); X += carry; carry = X[:, :, -1:].clone(); X += cdrift[:, None, t0:t0 + T]
            w = torch.minimum(torch.minimum(X[0], X[1]), X[2])
            if has_lz: cmin = torch.minimum(torch.cummin(w, dim=1).values, wmin[:, None])
            wmin = torch.minimum(wmin, w.amin(dim=1))
            sel = ((obs_day - 1 >= t0) & (obs_day - 1 < t0 + T)).nonzero(as_tuple=True)[0]
            if sel.numel():
                col = obs_day[sel] - 1 - t0; wobs[:, sel] = w[:, col]
                if has_lz: wrun[:, sel] = cmin[:, col]
        hit = wobs >= logK
        ev = (hit | ((wrun >= loglzb) & lz_on)) if has_lz else hit
        any_ev = ev.any(dim=1); first = ev.to(f32).argmax(dim=1)
        is_reg = hit.gather(1, first[:, None]).squeeze(1); pr = torch.where(is_reg, rate[first], lzp[first])
        early = (1.0 + pr) * DF[obs_day[first] - 1]
        wT = torch.minimum(torch.minimum(carry[0, :, 0] + cdrift[0, -1], carry[1, :, 0] + cdrift[1, -1]),
                           carry[2, :, 0] + cdrift[2, -1])
        eT = torch.exp(wT); sv = eT if always_ki else torch.where(wmin < logB, eT, torch.ones_like(eT))
        tot += torch.where(any_ev, early, sv * DF[N - 1]).sum(dtype=torch.float64); done += m
    return float(tot.item() / n)


# ---- load ----
df = pd.read_parquet(DATA / "els3_dataset.parquet").sort_values("isu_ord").reset_index(drop=True)
RET = load_RET(pd.unique(df[["udl1", "udl2", "udl3"]].values.ravel()))
IV = load_IV(); KRW = load_krw_curve()
print(f"상품 {len(df):,} | RET {len(RET)} | IV RIC {len(IV)} | KRW {KRW.shape}", flush=True)

def iv_asof(ticker, dt):
    ric = IVMAP.get(ticker)
    if ric is None or ric not in IV: return np.nan
    s = IV[ric]; s = s[s.index <= dt]; return float(s.iloc[-1]) if len(s) else np.nan

def price_iv(r):
    dt = pd.Timestamp(date.fromordinal(int(r["isu_ord"]))); ts = [r["udl1"], r["udl2"], r["udl3"]]
    if any(t not in RET for t in ts): return (np.nan, 0)
    corr = corr180([RET[t] for t in ts], dt)
    if corr is None: return (np.nan, 0)
    sigs, niv = [], 0
    for t in ts:
        h = vol180(RET[t], dt); iv = iv_asof(t, dt)
        if not np.isnan(iv): sigs.append(iv); niv += 1
        else: sigs.append(h)
    if any(np.isnan(sigs)): return (np.nan, 0)
    krow = KRW.asof(dt).values
    dz = ns_z(krow) if DISC == "ns" else boot_z(krow)
    if dz is None: return (np.nan, 0)
    k = int(r["nobs"])
    strikes = [float(r[f"strk_{j}"]) for j in range(k)]; pmts = [float(r[f"pmt_{j}"]) for j in range(k)]
    lzb = [float(r[f"lz_barr_{j}"]) for j in range(k)]; lzp = [float(r[f"lz_pmt_{j}"]) for j in range(k)]
    m = mc_daily_t(sigs, corr, dz, float(r["B"]), strikes, float(r["tenor"]), n=NPATH, seed=int(r["mc_seed"]),
                   c=float(r["coupon"]), pmts=pmts, lz_barr=lzb, lz_pmt=lzp)
    return (m, niv)

# ---- resume ----
ck = ROOT / f"scratch/mc_iv_{DISC}_ckpt.npz"
mc_iv = np.full(len(df), np.nan); n_iv = np.zeros(len(df), dtype=int)
start = 0
if ck.exists():
    z = np.load(ck); mc_iv = z["mc_iv"]; n_iv = z["n_iv"]
    start = int(np.where(np.isnan(mc_iv))[0][0]) if np.isnan(mc_iv).any() else len(df)
    print(f"resume from {start:,}", flush=True)

t0 = time.time()
for i in range(start, len(df)):
    m, niv = price_iv(df.iloc[i]); mc_iv[i] = m; n_iv[i] = niv
    if (i + 1) % 500 == 0:
        el = time.time() - t0; rate = (i + 1 - start) / el
        print(f"{i+1:,}/{len(df):,} ({el/60:.1f}m, {rate:.1f}/s, ETA {(len(df)-i-1)/rate/60:.0f}m)", flush=True)
    if (i + 1) % 4000 == 0:
        np.savez(ck, mc_iv=mc_iv, n_iv=n_iv)

np.savez(ck, mc_iv=mc_iv, n_iv=n_iv)
out = df[["item", "isu_ord", "opt_type", "ki_yn", "fair", "mc"]].copy()
out["mc_iv"] = mc_iv.astype("float32"); out["n_iv_udl"] = n_iv
op = ROOT / f"result/statistics/mc_iv_{DISC}_full.csv"; op.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(op, index=False, encoding="utf-8-sig")
ok = out["mc_iv"].notna()
print(f"\nDONE {int(ok.sum()):,}/{len(df):,} in {(time.time()-t0)/60:.1f}m | "
      f"IV3 {int((out.n_iv_udl==3).sum()):,} | mc_iv-mc mean {(out.mc_iv-out.mc)[ok].mean()*10000:+.0f}원 | saved {op.name}", flush=True)
