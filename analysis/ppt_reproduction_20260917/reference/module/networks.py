# -*- coding: utf-8 -*-
"""정통 DeepONet 재설계.
 CurveOperatorV2 : branch=CNN(곡선)+vol·corr, trunk=MLP(계약). 데이터기반. stage-1 앵커.
 MarginOperator  : stage-2 잔차 DeepONet(내적). branch=시장상태, trunk=계약 → 내적으로 가격.
 PIStateDeepONet : CurveOperatorV2 + trunk 에 경로상태(스팟·러닝최소·잔존만기·KI) 증강. PI 앵커."""
import torch
import torch.nn as nn


def mlp(d, p=128):
    return nn.Sequential(nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p))


class MarginOperator(nn.Module):
    """stage-2 잔차 DeepONet: branch=MLP(vol·corr·sig_eff), trunk=MLP(곡선+계약). 내적으로 잔차 회귀."""
    def __init__(self, nb, nt, P):
        super().__init__()
        self.b = mlp(nb, P)          # branch: 리스크(vol·corr) + 금리
        self.t = mlp(nt, P)          # trunk: 나머지(계약+발행+범주형 one-hot)
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, bx, tx):
        return (self.b(bx) * self.t(tx)).sum(-1) + self.b0


class CurveOperatorV2(nn.Module):
    """DeepONet-Curve: branch=1D-CNN(곡선)+vol·corr 융합, trunk=계약. 스팟 없음."""
    def __init__(self, nvc, ncon, P):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.branch = nn.Sequential(nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P))
        self.t = mlp(ncon, P)        # trunk: 계약
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, curve, vc, con):
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)          # (n, 32)
        branch = self.branch(torch.cat([hc, vc], -1))          # (n, P)  시장상태(곡선+vol·corr)
        return (branch * self.t(con)).sum(-1) + self.b0


class PIStateDeepONet(nn.Module):
    """PI 앵커: CurveOperatorV2 와 같은 branch(곡선 CNN + vol·corr) + trunk 에 **경로상태 10차원** 증강.

     데이터항만 쓰면 CurveOperatorV2 와 사실상 동등하지만(발행시점 상태로 고정 평가),
     trunk 가 임의의 (S, M, tau) 를 받으므로 PDE 잔차·이벤트 경계조건을 걸 수 있다.
     상태 규약은 module.pi_physics 의 docstring 참고 (S1..3, M1..3, tau, KI, event_pre, obs_frac)."""

    state_dim = 10

    def __init__(self, nvc, ncon, P):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.branch = nn.Sequential(nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P))
        self.t = mlp(ncon + self.state_dim, P)     # trunk: 계약 + 경로상태
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, curve, vc, con, state):
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)              # (n, 32)
        branch = self.branch(torch.cat([hc, vc], -1))              # (n, P)  시장상태
        return (branch * self.t(torch.cat([con, state], -1))).sum(-1) + self.b0


class CurveOperatorMoE(nn.Module):
    """arm A4b (XPINN 게이팅): CurveOperatorV2 공유 branch/trunk + K 전문가 헤드 + 배리어 게이트(soft routing).
     불연속(낙인 배리어)에서 영역 특화하되 전량 데이터로 학습(하드분할의 데이터굴주 회피). ib=계약 내 배리어 인덱스."""
    def __init__(self, nvc, ncon, P, K=3, ib=0):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                 nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.branch = nn.Sequential(nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P))
        self.t = mlp(ncon, P)
        self.heads = nn.Linear(P, K)                            # K 전문가 (공유 상호작용 z 위)
        self.b0 = nn.Parameter(torch.zeros(K))
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, K))   # 배리어 기준 soft 게이트
        self.ib = ib

    def V(self, curve, vc, con):
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)
        branch = self.branch(torch.cat([hc, vc], -1))
        z = branch * self.t(con)                                # (n, P) 공유 상호작용
        experts = self.heads(z) + self.b0                       # (n, K) 전문가별 가격
        g = torch.softmax(self.gate(con[:, self.ib:self.ib + 1]), -1)   # (n, K) 배리어 게이트
        return (g * experts).sum(-1)
