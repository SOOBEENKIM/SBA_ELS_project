# -*- coding: utf-8 -*-
"""저장된 model weight 을 로드해 OOS test set 을 직접 forward → 예측 DataFrame 조립.
 (3_evaluate 가 예측 CSV 대신 이걸 써서 가중치→예측 재현)
 전 모델이 **이론가(MC) 2단계 예측**: y = MC_hat(stage1) + resid(stage2), 마진 없음, y_true = MC.
 result/models/<name>_fold{k}.{pt|pkl} 규약은 pipeline 의 저장 규약과 동일."""
import numpy as np
import pandas as pd
import joblib

from util import file_manager as fm
from .train import load_curve_predictor
from .pi_train import load_pi_predictor
from .tabular import load_tab_predictor
from model.stage2 import load_xgb_resid, load_don_resid, load_ml_resid


def _mp(name, k):
    return str(fm.RESULT / "models" / f"{name}_fold{k}")


def _load_xgb_anchor(D, path):
    """benchmark._xgb_anchor 저장 .pkl → predict(idx). 입력 X=[곡선|vol·corr|계약]."""
    m = joblib.load(path)["model"]
    X = np.concatenate([D.CURVE, D.VC, D.CON], axis=1)
    return lambda idx: m.predict(X[idx]).astype("float32")


# 직접(단일 단계) 벤치마크: fit_tab .pkl → 이론가(MC) 직접 예측
_DIRECT = ["bench_ridge", "bench_gbm", "bench_lgbm", "bench_xgboost", "bench_catboost"]

# 이론가(MC) 2단계(하이브리드) 모델 구성: name -> (anchor_type, resid_type)
_HYBRID = {
    "deeponet_hybrid": ("curve", "mltab"),        # 앵커 MSE
    "deeponet_hybrid_l1": ("curve", "mltab"),     # 앵커 L1
    "deeponet_hybrid_mape": ("curve", "mltab"),   # 앵커 MAPE
    "deeponet_hybrid_s2don": ("curve", "margin"),  # stage-2 = DeepONet
    "xgb_hybrid": ("xgb", "mltab"),               # 앵커 XGB
}

# 구조별 라우팅 하이브리드 (model/pi_route.py): 앵커가 폴드당 2개라 위 (anchor,resid) 규약에 안 맞아 분리.
# name -> 재현에 쓸 시드. 다중 시드 결과는 result/statistics/exp_pi_route_seeds.csv 참고.
_ROUTED = {"deeponet_hybrid_pi_lizard": 0}


def _anchor_loader(D, atype, base):
    if atype == "curve":
        return load_curve_predictor(D, base + ".pt")
    if atype == "pi":
        return load_pi_predictor(D, base + ".pt")      # PI 앵커(상태증강) — 발행시점 상태에서 평가
    if atype == "xgb":
        return _load_xgb_anchor(D, base + ".pkl")
    raise ValueError(atype)


def _resid_loader(D, rtype, base):
    if rtype == "mltab":
        return load_ml_resid(D, base + ".pkl")         # 기본 stage2: ml 전체특성 (cfg['margin'])
    if rtype == "tab":
        return load_xgb_resid(D, base + ".pkl")        # XGB on deeponet 블록 D.DON
    if rtype == "margin":
        return load_don_resid(D, base + ".pt")         # DeepONet stage2 (deeponet 블록)
    raise ValueError(rtype)


def predict_all_from_weights(D, cfg):
    """저장 가중치로 전 모델의 공정가(FAIR) OOS 예측을 조립. pipeline 산출 DataFrame 과 동일 스키마.
     직접(단일 단계) 벤치마크: y = 예측(FAIR) (stage 컬럼 없음).
     2단계 하이브리드: y = MC_hat + recent_margin + resid, y_true = FAIR,
                      resid_true = FAIR − MC_hat − recent_margin, mc_true = MC(이론가 참값)."""
    FAIR, MC, rm = D.FAIR, D.MC, D.rm
    out = {}
    # 직접(단일 단계) 벤치마크
    for name in _DIRECT:
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            pred = load_tab_predictor(D, _mp(name, k) + ".pkl")
            yp = np.asarray(pred(te), dtype="float32")
            rows.append(pd.DataFrame({
                "ITEM_CD": D.ITEM[te], "isu_ord": D.ORD[te],
                "y_true": FAIR[te], "y_pred": yp,
            }))
        out[name] = pd.concat(rows, ignore_index=True)
    # 2단계 하이브리드
    for name, (atype, rtype) in _HYBRID.items():
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            anchor = _anchor_loader(D, atype, _mp(f"{name}_anchor", k))
            resid = _resid_loader(D, rtype, _mp(f"{name}_resid", k))
            mc_te = np.asarray(anchor(te), dtype="float32")
            r_te = np.asarray(resid(te), dtype="float32")
            y = mc_te + rm[te] + r_te
            rows.append(pd.DataFrame({
                "ITEM_CD": D.ITEM[te], "isu_ord": D.ORD[te],
                "y_true": FAIR[te], "y_pred": y,
                "mc_true": MC[te], "mc_pred": mc_te,
                "resid_true": (FAIR[te] - mc_te - rm[te]).astype("float32"), "resid_pred": r_te,
            }))
        out[name] = pd.concat(rows, ignore_index=True)
    # 구조별 라우팅 하이브리드 (앵커 2개 → 예측 시 계약조건으로 선택). 없으면 조용히 건너뛴다.
    from model.pi_route import load_routed_predictor
    for name, seed in _ROUTED.items():
        if not (fm.RESULT / "models" / f"{name}_seed{seed}_anchor_base_fold0.pt").exists():
            continue
        rows = []
        for k, (tr, va, te) in enumerate(D.WF):
            anchor = load_routed_predictor(D, name, seed, k)
            resid = load_ml_resid(D, _mp(f"{name}_seed{seed}_resid", k) + ".pkl")
            mc_te = np.asarray(anchor(te), dtype="float32")
            r_te = np.asarray(resid(te), dtype="float32")
            rows.append(pd.DataFrame({
                "ITEM_CD": D.ITEM[te], "isu_ord": D.ORD[te],
                "y_true": FAIR[te], "y_pred": mc_te + rm[te] + r_te,
                "mc_true": MC[te], "mc_pred": mc_te,
                "resid_true": (FAIR[te] - mc_te - rm[te]).astype("float32"), "resid_pred": r_te,
            }))
        out[name] = pd.concat(rows, ignore_index=True)
    return out
