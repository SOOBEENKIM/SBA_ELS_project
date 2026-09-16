# -*- coding: utf-8 -*-
"""walk-forward OOS 예측 조립: 직접(tabular 벤치마크)·하이브리드. 반환 DataFrame.
 name 주면 폴드별 학습 모델 가중치를 result/models/<name>_fold{k}.{pt|pkl} 로 저장."""
import numpy as np
import pandas as pd

from .tabular import fit_tab
from model.stage2 import ml_resid
from util import file_manager as fm


def _mp(name, k):
    """폴드별 모델 저장 경로 접두(확장자는 각 학습함수가 붙임). name=None이면 저장 안 함."""
    if name is None:
        return None
    d = fm.RESULT / "models"; d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{name}_fold{k}")


def hybrid_residual_target(fair, mc_hat, rm):
    """stage-2 잔차 타깃 = FAIR − stage1예측(MC_hat) − recent_margin.
    (기존엔 MC 참값을 썼으나, stage1 예측값 기준으로 재정의 → 학습·추론 일치 + stage1 근사오차 흡수.)"""
    return (np.asarray(fair) - np.asarray(mc_hat) - np.asarray(rm)).astype("float32")


def predict_direct_tab(D, cfg, model_key, name=None, target=None):
    """직접(단일 단계) tabular 회귀 벤치마크 — 이론가(MC)를 ml.csv 전 특성으로 바로 예측(2단계 아님).
     target=None → D.MC(이론가). regime 피처. save_path 주면 폴드별 .pkl 저장(infer 재현용)."""
    tgt = D.MC if target is None else target
    rows = []
    for k, (tr, va, te) in enumerate(D.WF):
        yp = fit_tab(
            D, cfg, model_key, tr, te, "regime", tgt, tw=cfg["data"]["time_decay"], va=va,
            save_path=_mp(name, k)
        )
        rows.append(
            pd.DataFrame({"ITEM_CD": D.ITEM[te], "isu_ord": D.ORD[te], "y_true": tgt[te], "y_pred": yp})
        )
    return pd.concat(rows, ignore_index=True)


def predict_hybrid(D, cfg, anchor_fn, resid_fn=None, name=None, target=None, use_margin=True):
    """2단계 하이브리드: y = stage1예측(MC_hat) [+ recent_margin] + resid_hat(stage2).

    target=None → D.FAIR (기본, 공정가치 예측). target=D.MC → 이론가 2단계 예측.
    use_margin=True → recent_margin 항 포함(공정가치용). False → 제외(이론가 예측용, 시장마진 없음).
    stage-2 잔차 타깃 = target − MC_hat − (recent_margin if use_margin else 0)  (stage1 예측값 기준, train/val/test 모두).
    anchor_fn(D,cfg,tr,va,te,save_path=) -> anchor_predict(idx)->np.ndarray[float32].
    resid_fn(D,cfg,tr,va,te,target,save_path=) -> (resid_tr, resid_te). None이면 config 마진모델(fit_tab).
    name 주면 stage-1 → <name>_anchor_fold{k}, stage-2 → <name>_resid_fold{k} 저장.
    """
    tgt = D.FAIR if target is None else target
    rm_full = D.rm if use_margin else np.zeros(D.n, dtype="float32")
    rows = []
    for k, (tr, va, te) in enumerate(D.WF):
        anchor_predict = anchor_fn(D, cfg, tr, va, te,
                                   save_path=_mp(f"{name}_anchor" if name else None, k))
        sel = np.concatenate([tr, va, te]) if len(va) else np.concatenate([tr, te])
        mc_hat = np.full(D.n, np.nan, dtype="float32")
        mc_hat[sel] = np.asarray(anchor_predict(sel), dtype="float32")
        rt = np.full(D.n, np.nan, dtype="float32")
        rt[sel] = hybrid_residual_target(tgt[sel], mc_hat[sel], rm_full[sel])
        # stage-2 잔차모델: 기본=model.stage2.ml_resid(ml 전체특성, cfg['margin']), resid_fn 주면 그것
        rfn = ml_resid if resid_fn is None else resid_fn
        _, resid_pred = rfn(D, cfg, tr, va, te, rt,
                            save_path=_mp(f"{name}_resid" if name else None, k))
        mc_te = mc_hat[te]
        y_pred = mc_te + rm_full[te] + resid_pred
        rows.append(
            pd.DataFrame(
                {
                    "ITEM_CD": D.ITEM[te],
                    "isu_ord": D.ORD[te],
                    "y_true": tgt[te],
                    "y_pred": y_pred,
                    "mc_true": D.MC[te],
                    "mc_pred": mc_te,
                    "resid_true": rt[te],
                    "resid_pred": resid_pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
