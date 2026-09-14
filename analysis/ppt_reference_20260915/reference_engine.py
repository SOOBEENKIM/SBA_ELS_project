"""Frozen source functions; defaults are CPU / 40,000 paths. See reference and source_manifest.json."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
DEV="cpu"
NPATH=40000
KR_TAUS=np.array([0.08,0.25,10.0])
def _safe(t): return "px_" + t.replace("^", "_").replace(".", "_") + ".parquet"

def load_RET(tickers):
    RET = {}
    for t in tickers:
        f = CACHE / _safe(t)
        if f.exists():
            s = pd.read_parquet(f)["close"].dropna(); s = s[~s.index.duplicated()]
            RET[t] = np.log(s).diff()
    return RET

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
