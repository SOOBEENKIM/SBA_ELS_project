# -*- coding: utf-8 -*-
"""연산자망 학습 (stage-1 앵커).
 train_curve : DeepONet-Curve (CNN 곡선+vol·corr, trunk=계약). target=MC(이론가 앵커) 또는 FAIR.
 _EarlyStop/_opt 은 stage-2 (model.stage2) 에서도 공용으로 import 한다.
"""
import numpy as np
import torch

from .data import to_tensor, zstats, znorm
from .networks import CurveOperatorV2, CurveOperatorMoE


def _opt(net, cfg):
    return torch.optim.Adam(net.parameters(), cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


class _EarlyStop:
    """val 손실 기반 조기종료 + 최적 가중치 복원. va 비면 비활성."""
    def __init__(self, net, cfg, va, dev):
        self.on = va is not None and len(va) > 0
        self.net = net
        self.every = cfg["train"].get("es_every", 100)
        self.patience = cfg["train"].get("es_patience", 6)
        self.vat = torch.tensor(np.asarray(va), device=dev) if self.on else None
        self.best = float("inf"); self.best_state = None; self.wait = 0

    def step(self, it, val_loss_fn):
        """(it+1) 마다 호출; 개선 없으면 True(=중단) 반환."""
        if not self.on or (it + 1) % self.every != 0:
            return False
        with torch.no_grad():
            vloss = float(val_loss_fn(self.vat))
        if vloss < self.best - 1e-7:
            self.best = vloss
            self.best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self):
        if self.on and self.best_state is not None:
            self.net.load_state_dict(self.best_state)


# ===== DeepONet-Curve (데이터기반, stage-1 앵커) =====
def _loss_fn(loss):
    """정규화 스케일 손실: mse(기본) | l1(MAE) | huber. (mape 는 train_curve 에서 원본 스케일로 별도 처리)"""
    if loss == "l1":
        return lambda p, y: (p - y).abs().mean()
    if loss == "huber":
        return lambda p, y: torch.nn.functional.huber_loss(p, y, delta=1.0)
    return lambda p, y: ((p - y) ** 2).mean()


def train_curve(D, cfg, tr, te, target, va=None, loss="mse", return_predict=False, save_path=None,
                moe_k=0, moe_ib=0):
    """target=D.MC → 하이브리드 앵커(이론가) / target=D.FAIR → 직접.
    loss: 'mse'(기본)|'l1'|'huber'|'mape'. mape 는 원본 스케일 상대오차. va → val 조기종료.
    moe_k>0 이면 CurveOperatorMoE(K=moe_k, 배리어게이트 ib=moe_ib) 사용 (arm A4b)."""
    dev = D.DEV; B = cfg["train"]["batch"]; NIT = cfg["train"]["nit"]; P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    um, us = zstats(D.CURVE, tr); vm, vs = zstats(D.VC, tr); cm, cs = zstats(D.CON, tr)
    Un = to_tensor(znorm(D.CURVE, um, us), dev)
    Vn = to_tensor(znorm(D.VC, vm, vs), dev)
    Cn = to_tensor(znorm(D.CON, cm, cs), dev)
    net = (CurveOperatorMoE(Vn.shape[1], Cn.shape[1], P, K=moe_k, ib=moe_ib) if moe_k
           else CurveOperatorV2(Vn.shape[1], Cn.shape[1], P)).to(dev); opt = _opt(net, cfg)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8); Y = to_tensor((target - ym) / ysd, dev)
    Torig = to_tensor(target, dev)                 # 원본 스케일 타깃 (mape loss용)
    ntr = len(tr); trt = torch.tensor(tr, device=dev)
    lf = _loss_fn(loss)

    def bloss(pr, idx):                            # 배치 손실 (mape 는 역정규화한 원본 스케일 상대오차)
        if loss == "mape":
            po = pr * ysd + ym
            return ((po - Torig[idx]).abs() / Torig[idx].abs().clamp(min=1e-6)).mean()
        return lf(pr, Y[idx])

    es = _EarlyStop(net, cfg, va, dev)
    for it in range(NIT):
        bb = trt[torch.randint(0, ntr, (B,), device=dev)]
        pr = net.V(Un[bb], Vn[bb], Cn[bb])
        opt.zero_grad(); bloss(pr, bb).backward(); opt.step()
        if es.step(it, lambda vi: bloss(net.V(Un[vi], Vn[vi], Cn[vi]), vi)):
            break
    es.restore(); net.eval()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state": net.state_dict(), "P": P, "nvc": Vn.shape[1], "ncon": Cn.shape[1],
                    "um": um, "us": us, "vm": vm, "vs": vs, "cm": cm, "cs": cs, "ym": ym, "ysd": ysd},
                   save_path if save_path.endswith(".pt") else save_path + ".pt")

    def p(idx):
        with torch.no_grad():
            i = torch.tensor(idx, device=dev)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()
    return (p(tr), p(te), p) if return_predict else (p(tr), p(te))


def load_curve_predictor(D, path):
    """저장된 DeepONet-Curve 가중치를 로드해 predict(idx)->np.array 반환 (forward만; 재학습 없음)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = CurveOperatorV2(ck["nvc"], ck["ncon"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"]); net.eval()
    Un = to_tensor(znorm(D.CURVE, ck["um"], ck["us"]), D.DEV)
    Vn = to_tensor(znorm(D.VC, ck["vm"], ck["vs"]), D.DEV)
    Cn = to_tensor(znorm(D.CON, ck["cm"], ck["cs"]), D.DEV)
    ym, ysd = ck["ym"], ck["ysd"]

    def predict(idx):
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=D.DEV)
            return (net.V(Un[i], Vn[i], Cn[i]) * ysd + ym).cpu().numpy()
    return predict
