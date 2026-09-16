# -*- coding: utf-8 -*-
"""PI 앵커용 collocation 샘플러 — **fold train 계약만** 사용.

 물리항은 라벨이 필요 없으므로 어디서든 뽑을 수 있는데, 그게 곧 누출 경로다.
 그래서 (1) 학습 인덱스만 받고 (2) 매 스텝 assert_train_only 로 검증하며
 (3) 샘플 범위(스팟·러닝최소 하한)도 fold train 계약의 행사가·배리어 분위수에서 도출한다.

 4종을 분리해 뽑는다:
   sample_interior      PDE 잔차용 내부점 (이벤트 시각 근방은 회피 — 그 지점은 불연속)
   sample_event         관측일 pre/post 상태 쌍 (정규상환 / 리자드 / 연속 3모드)
   sample_terminal      만기 (tau=0)
   sample_ki_boundary   배리어 ±eps 양측 상태 쌍 (KI 지시자만 다른 쌍)
"""
import numpy as np
import torch

from .pi_physics import event_tau, feasible_ki_states


class PICollocationSampler:
    """D.CON 레이아웃 의존: [0:12]=strk, [12:24]=pmt, [24:36]=lz_barr, [36:48]=lz_pmt, D.iBARR=배리어."""

    def __init__(self, D, train_idx, seed=0):
        self.D = D
        self.train_idx = np.asarray(train_idx, dtype=int)
        if not len(self.train_idx):
            raise ValueError("collocation 학습 유니버스가 비었습니다")
        self.rng = np.random.default_rng(seed)
        con = D.CON[self.train_idx]
        ks = con[:, :12].ravel(); ks = ks[(ks > .2) & (ks < 1.2)]
        bs = con[:, D.iBARR]; bs = bs[(bs > .2) & (bs < 1.)]
        # 계약 행사가/배리어가 실제로 사는 구간 + 여유. 학습분포 밖에서 물리를 강제하면 노이즈만 늘어난다.
        self.spot_lo = float(max(.35, np.quantile(ks, .01) - .15))
        self.spot_hi = float(min(1.30, np.quantile(ks, .99) + .25))
        self.run_lo = float(max(.25, np.quantile(bs, .01) - .12)) if len(bs) else .35
        self.train_items = set(D.ITEM[self.train_idx].astype(str))

    # ---- 내부 유틸 ----
    def _contracts(self, n):
        return self.train_idx[self.rng.integers(0, len(self.train_idx), n)]

    def _state(self, idx, spot, run, tau, ki, event_pre=0., frac=0.):
        """(n, 10) 상태행렬 조립. 규약은 module.pi_physics docstring 참고."""
        return np.c_[spot, run, tau, ki,
                     np.full(len(idx), event_pre),
                     np.asarray(frac) + np.zeros(len(idx))].astype("float32")

    def _ki(self, idx, run):
        barr = self.D.CON[idx, self.D.iBARR]
        return feasible_ki_states(torch.tensor(run), torch.tensor(barr)).numpy()

    # ---- 4종 샘플러 ----
    def sample_interior(self, n, device):
        """PDE 잔차용. 관측일(불연속점)에서 2.5%·만기 이상 떨어진 tau 만 채택."""
        idx = self._contracts(n)
        spot = self.rng.uniform(self.spot_lo, self.spot_hi, (n, 3)).astype("float32")
        run = np.minimum(spot, self.rng.uniform(self.run_lo, 1., (n, 3))).astype("float32")
        tau = np.empty(n, "float32"); frac = np.empty(n, "float32")
        for q, i in enumerate(idx):
            nobs = int(self.D.ml.iloc[i].nobs); ten = float(self.D.TEN[i])
            events = ten - np.arange(1, nobs + 1) * ten / nobs
            for _ in range(100):                       # 이벤트 근방이면 재추첨 (최대 100회)
                x = self.rng.uniform(.03 * ten, .97 * ten)
                if np.min(np.abs(events - x)) > .025 * ten:
                    break
            tau[q] = x; frac[q] = 1 - x / ten
        return idx, torch.tensor(self._state(idx, spot, run, tau, self._ki(idx, run), 0., frac), device=device)

    def sample_event(self, n, device):
        """관측일 pre/post 상태 쌍. mode 0=정규상환, 1=리자드(가능할 때만), 2=연속(둘 다 실패)."""
        idx = self._contracts(n)
        nobs = self.D.ml.iloc[idx].nobs.to_numpy(int)
        j = np.array([self.rng.integers(0, x) for x in nobs])
        K = self.D.CON[idx, j]; lzb = self.D.CON[idx, 24 + j]
        mode = self.rng.integers(0, 3, n)
        spot = np.empty((n, 3), "float32"); run = np.empty((n, 3), "float32")
        for q in range(n):
            if mode[q] == 0:                                       # 정규 상환: worst >= K
                spot[q] = self.rng.uniform(K[q] + .005, K[q] + .12, 3)
                run[q] = np.minimum(spot[q], self.rng.uniform(self.run_lo, 1., 3))
            elif mode[q] == 1 and lzb[q] < 1.15 and K[q] > lzb[q] + .03:   # 리자드가 실현 가능한 계약만
                lo = max(lzb[q] + .005, self.spot_lo); hi = K[q] - .005
                spot[q] = self.rng.uniform(lo, hi, 3); spot[q, 0] = K[q] - .01
                run[q] = self.rng.uniform(lzb[q] + .002, np.minimum(spot[q], 1.))
            else:                                                  # 연속: 정규·리자드 모두 실패
                spot[q] = self.rng.uniform(self.spot_lo, max(self.spot_lo + .01, K[q] - .01), 3)
                spot[q, 0] = K[q] - .03
                run[q] = np.minimum(spot[q], self.rng.uniform(self.run_lo, 1., 3))
                run[q, 0] = min(run[q, 0], spot[q, 0], lzb[q] - .01)
        ki = self._ki(idx, run)
        tau = event_tau(torch.tensor(self.D.TEN[idx]), torch.tensor(nobs), torch.tensor(j)).numpy()
        frac = (j + 1) / nobs
        pre = self._state(idx, spot, run, tau, ki, 1., frac)
        post = self._state(idx, spot, run, tau, ki, 0., frac)
        return idx, j, torch.tensor(pre, device=device), torch.tensor(post, device=device)

    def sample_terminal(self, n, device):
        """만기(tau=0) 경계."""
        idx = self._contracts(n)
        spot = self.rng.uniform(self.spot_lo, self.spot_hi, (n, 3)).astype("float32")
        run = np.minimum(spot, self.rng.uniform(self.run_lo, 1., (n, 3))).astype("float32")
        state = self._state(idx, spot, run, np.zeros(n), self._ki(idx, run), 0., 1.)
        return idx, torch.tensor(state, device=device)

    def sample_ki_boundary(self, n, device, eps=.002):
        """배리어 바로 위(KI=0) vs 바로 아래(KI=1) 쌍. 낙인 계약(B<1) 만 대상."""
        idx = self._contracts(n)
        b = self.D.CON[idx, self.D.iBARR]
        keep = b < 1.
        idx = idx[keep]; b = b[keep]; n = len(idx)
        spot = np.maximum(b[:, None] + .05, self.rng.uniform(.75, 1.1, (n, 3))).astype("float32")
        above = np.repeat((b + eps)[:, None], 3, 1).astype("float32")
        below = np.repeat((b - eps)[:, None], 3, 1).astype("float32")
        tau = (self.rng.uniform(.1, .9, n) * self.D.TEN[idx]).astype("float32")
        frac = 1 - tau / self.D.TEN[idx]
        s0 = self._state(idx, spot, above, tau, np.zeros(n), 0., frac)
        s1 = self._state(idx, spot, below, tau, np.ones(n), 0., frac)
        return idx, torch.tensor(s0, device=device), torch.tensor(s1, device=device)

    def assert_train_only(self, idx):
        """collocation 이 비학습 계약을 건드리면 즉시 실패 (누출 방지 가드)."""
        if not set(self.D.ITEM[np.asarray(idx)].astype(str)).issubset(self.train_items):
            raise AssertionError("collocation 이 학습 외 계약을 사용했습니다")
