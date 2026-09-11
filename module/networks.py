"""Stage-1 CurveOperatorV2 used by all final experiments; unused legacy classes excluded."""
import torch

import torch.nn as nn

def mlp(d, p=128):
    return nn.Sequential(nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p))

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
