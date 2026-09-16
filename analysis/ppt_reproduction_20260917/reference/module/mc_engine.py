# -*- coding: utf-8 -*-
"""torch MC 엔진 (CUDA 기본 / CPU 폴백) — 페이오프 규칙의 정본 구현.

 노트북 4의 V3 이식: 시간축 블록 + carry 로 경로를 만들고, 선형성
 cumsum(drift + s*Z) = cumdrift + s*cumsum(Z) 로 drift pass 를 없앴으며,
 worst-of 비교는 log-space 에서 한다 (min exp(x) >= K  <=>  min x >= log K).
 관측일 판정은 argmax(first event) 로 완전 벡터화. 상품당 약 27 ms (RTX 3060 Ti).

 규칙 우선순위: 조기상환(회차별 지급률) > 리자드(FROM_ISU no-touch, 축소 지급률) > 만기(낙인 판정).
 B >= 1.0 은 노낙인 = 만기 미상환이면 항상 worst 수취."""
import numpy as np
import torch

from . import features as F
from .mc import NPATH, MC_COLS, prepare_one

DEFAULT_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def payoff_from_path_summary_t(wobs, wrun, wmin, wT, logK, rate, lz_on, loglzb,
                               lzp, always_ki, logB, DF, obs_day):
    """경로요약 -> 할인 페이오프. mc_daily_t 가 쓰는 페이오프 블록을 그대로 함수로 분리한 것.

     PI 물리항(module.pi_physics)이 '실측 MC 규약'과 일치하는지 테스트에서 직접 대조하려고 노출한다.
     정규 상환이 리자드보다 우선. 만기 미상환 시 노낙인(B>=1)은 항상 worst, 낙인은 배리어 터치 여부."""
    f32 = wobs.dtype
    hit = wobs >= logK
    has_lz = bool(lz_on.any())
    ev = (hit | ((wrun >= loglzb) & lz_on)) if has_lz else hit
    any_ev = ev.any(dim=1)
    first = ev.to(f32).argmax(dim=1)
    is_reg = hit.gather(1, first[:, None]).squeeze(1)
    pr = torch.where(is_reg, rate[first], lzp[first])                # 정규 상환이 리자드보다 우선
    early = (1.0 + pr) * DF[obs_day[first] - 1]
    eT = torch.exp(wT)
    sv = eT if always_ki else torch.where(wmin < logB, eT, torch.ones_like(eT))
    return torch.where(any_ev, early, sv * DF[-1])


def _setup(sigs, corr, beta, B, strikes, ten, c, pmts, lz_barr, lz_pmt):
    nobs = len(strikes); N = int(round(ten * 365)); dt = 1 / 365
    obs_day = np.clip(np.round(np.arange(1, nobs + 1) * (ten / nobs) * 365).astype(int), 1, N)
    obs_t = obs_day / 365.0
    times = np.arange(1, N + 1) * dt
    DF = np.exp(-F.zero_curve(beta, times) * times)
    fdt = (-(np.diff(np.concatenate([[0.0], np.log(DF)])))).astype(np.float32)
    sig = np.asarray(sigs, np.float32)
    drift = (fdt[:, None] - (0.5 * sig ** 2 * np.float32(dt))[None, :]).astype(np.float32)
    rate = np.asarray(c * obs_t if pmts is None else pmts, np.float32)[:nobs]
    lzb = (np.full(nobs, np.nan, np.float32) if lz_barr is None
           else np.asarray(lz_barr, np.float32)[:nobs])
    lzp = (np.zeros(nobs, np.float32) if lz_pmt is None
           else np.nan_to_num(np.asarray(lz_pmt, np.float32))[:nobs])
    on = np.isfinite(lzb)
    return dict(N=N, nobs=nobs, obs_day=obs_day, DF=DF.astype(np.float32),
                L=F.chol_psd(np.asarray(corr)).astype(np.float32), sig=sig, drift=drift,
                sq=np.float32(np.sqrt(dt)), logB=float(np.log(B)),
                logK=np.log(np.asarray(strikes, np.float32)), rate=rate, lzp=lzp, lz_on=on,
                loglzb=np.log(np.where(on, lzb, 1.0)).astype(np.float32))


def mc_daily_t(sigs, corr, beta, B, strikes, ten, n=NPATH, seed=0, c=0.0, pmts=None,
               lz_barr=None, lz_pmt=None, dev=None, path_chunk=20_000, tblock=512):
    """일별 MC (torch). pmts=None 이면 (1 + c*경과연수), 주어지면 (1 + pmts[j]).
     lz_barr[j] 가 유한하면 정규 조건 실패 경로에 대해 발행 이후 최저 worst >= 배리어일 때 리자드 상환."""
    P = _setup(sigs, corr, beta, B, strikes, ten, c, pmts, lz_barr, lz_pmt)
    d = torch.device(dev or DEFAULT_DEV); f32 = torch.float32
    N, nobs = P["N"], P["nobs"]
    L = torch.as_tensor(P["L"] * (P["sig"] * P["sq"])[:, None], device=d)   # s*sqrt(dt) 를 L 에 흡수
    cdrift = torch.as_tensor(P["drift"], device=d).cumsum(0).T.contiguous()  # (3,N) 누적 drift
    DF = torch.as_tensor(P["DF"], device=d)
    logK = torch.as_tensor(P["logK"], device=d)
    rate = torch.as_tensor(P["rate"], device=d)
    lzp = torch.as_tensor(P["lzp"], device=d)
    loglzb = torch.as_tensor(P["loglzb"], device=d)
    lz_on = torch.as_tensor(P["lz_on"], device=d)
    obs_day = torch.as_tensor(P["obs_day"], device=d, dtype=torch.long)
    has_lz = bool(P["lz_on"].any()); always_ki = bool(B >= 1.0); logB = P["logB"]
    g = torch.Generator(device=d).manual_seed(int(seed))
    tot = torch.zeros((), device=d, dtype=torch.float64); done = 0
    while done < n:
        m = min(path_chunk, n - done)
        carry = torch.zeros(3, m, 1, device=d)
        wmin = torch.full((m,), float("inf"), device=d)
        wobs = torch.empty(m, nobs, device=d)
        wrun = torch.empty(m, nobs, device=d) if has_lz else None
        for t0 in range(0, N, tblock):
            T = min(tblock, N - t0)
            X = torch.randn(3, m, T, device=d, dtype=f32, generator=g)
            X[2] = L[2, 0] * X[0] + L[2, 1] * X[1] + L[2, 2] * X[2]     # 하삼각 상관(역순 in-place)
            X[1] = L[1, 0] * X[0] + L[1, 1] * X[1]
            X[0] = L[0, 0] * X[0]
            X = X.cumsum(2)
            X += carry
            carry = X[:, :, -1:].clone()                                 # drift 제외 누적
            X += cdrift[:, None, t0:t0 + T]
            w = torch.minimum(torch.minimum(X[0], X[1]), X[2])           # (m,T) log worst-of
            if has_lz:
                cmin = torch.minimum(torch.cummin(w, dim=1).values, wmin[:, None])
            wmin = torch.minimum(wmin, w.amin(dim=1))
            sel = ((obs_day - 1 >= t0) & (obs_day - 1 < t0 + T)).nonzero(as_tuple=True)[0]
            if sel.numel():
                col = obs_day[sel] - 1 - t0
                wobs[:, sel] = w[:, col]
                if has_lz:
                    wrun[:, sel] = cmin[:, col]
        wT = torch.minimum(torch.minimum(carry[0, :, 0] + cdrift[0, -1],
                                         carry[1, :, 0] + cdrift[1, -1]),
                           carry[2, :, 0] + cdrift[2, -1])
        payoff = payoff_from_path_summary_t(wobs, wrun, wmin, wT, logK, rate, lz_on,
                                            loglzb, lzp, always_ki, logB, DF, obs_day)
        tot += payoff.sum(dtype=torch.float64)
        done += m
    return float(tot.item() / n)


def price_one_t(mk, kmap, item, iord, B, c, ten, sig_eff, n=NPATH, seed=0, strikes=None,
                pmts=None, lz_barr=None, lz_pmt=None, dev=None):
    """price_one 과 같은 9개 튜플 (mc, vol1..3, rho12/13/23, r_krw, k)."""
    P = prepare_one(mk, kmap, item, iord, ten, sig_eff, strikes=strikes)
    if P is None:
        return (np.nan,) * len(MC_COLS)
    mc = mc_daily_t(P["sigs"], P["corr"], P["beta"], B, P["strikes"], ten, n=n, seed=seed,
                    c=c, pmts=pmts, lz_barr=lz_barr, lz_pmt=lz_pmt, dev=dev)
    return (mc, *P["audit"], P["rkrw"], P["k"])
