# -*- coding: utf-8 -*-
"""DeepONet-Curve (정통 재설계, 데이터기반): branch=1D-CNN(수익률곡선)+바스켓 vol·corr, trunk=계약.
 **공정가(FAIR) 2단계 예측**: stage-1=MC 이론가 앵커(MC_hat), stage-2=공정가 잔차(FAIR−MC_hat−recent_margin).
   Fair_hat = MC_hat + recent_margin + Residual_hat
 stage-2는 XGB(deeponet_hybrid) 또는 DeepONet(deeponet_hybrid_s2don), 앵커 loss는 MSE/L1/MAPE 비교."""
from module.train import train_curve
from module.pipeline import predict_hybrid
from model.stage2 import don_resid


def _anchor(D, cfg, tr, va, te, save_path=None):
    *_, predict = train_curve(D, cfg, tr, te, target=D.MC, va=va,
                              save_path=save_path, return_predict=True)
    return predict


def _anchor_l1(D, cfg, tr, va, te, save_path=None):
    """cost function 벤치마크: MC 앵커를 L1(MAE) 로 학습."""
    *_, predict = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="l1",
                              save_path=save_path, return_predict=True)
    return predict


def _anchor_mape(D, cfg, tr, va, te, save_path=None):
    """cost function 벤치마크: MC 앵커를 MAPE(상대오차) 로 학습."""
    *_, predict = train_curve(D, cfg, tr, te, target=D.MC, va=va, loss="mape",
                              save_path=save_path, return_predict=True)
    return predict


def run(D, cfg):
    T = dict(target=D.FAIR, use_margin=True)  # 공정가 2단계: y=MC_hat+recent_margin+resid, 잔차=FAIR−MC_hat−rm
    return {
        "deeponet_hybrid": predict_hybrid(D, cfg, _anchor, name="deeponet_hybrid", **T),           # 앵커 MSE
        # stage-1 동일(DeepONet MC 앵커), stage-2만 DeepONet(branch=리스크+금리, trunk=나머지)
        "deeponet_hybrid_s2don": predict_hybrid(D, cfg, _anchor, resid_fn=don_resid, name="deeponet_hybrid_s2don", **T),
        "deeponet_hybrid_l1": predict_hybrid(D, cfg, _anchor_l1, name="deeponet_hybrid_l1", **T),   # 앵커 L1
        "deeponet_hybrid_mape": predict_hybrid(D, cfg, _anchor_mape, name="deeponet_hybrid_mape", **T),  # 앵커 MAPE
    }
