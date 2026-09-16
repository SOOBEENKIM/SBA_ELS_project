# -*- coding: utf-8 -*-
"""stage-2 잔차모델 전용 (하이브리드 공용). 하이브리드는 stage1 앵커만 다르고 stage2는 여기 것을 공유.
 입력은 **deeponet 특성 블록 D.DON**(곡선 u0-9 | vol·corr·sig_eff | 계약; 이론가 결정 특성, ml aux 제외).

  ml_resid        : **기본 stage2**. cfg['margin'] 설정으로 ml 전체특성(BASE+CAT) 위에서 잔차 회귀.
                    마진(fair-mc)은 발행사·발행금액·청약일수 같은 시장/발행자 현상이라 그 특성이 필요하다.
  xgb_resid       : XGB on D.DON(이론가 결정 특성만). 이론가 잔차용.
  don_resid       : DeepONet stage2. branch=vol·corr·sig_eff, trunk=[곡선|계약]. (deeponet_hybrid_s2don)
 각 resid_fn(D,cfg,tr,va,te,target,save_path=) -> (resid_tr, resid_te). 로더는 predict(idx)->np.ndarray.

 주의: 2026-07-10 rebase 때 기본 stage2 가 ml 전체특성 -> D.DON 으로 바뀌면서 공정가 R² 가
 0.66 -> 0.44 로 떨어졌다(STEP×KI 실측, tests/verify_step_ki_reproduction.py). 기본값을 되돌린다.
"""
import numpy as np
import torch

from module.data import to_tensor, zstats, znorm, time_weights
from module.networks import MarginOperator
from module.tabular import fit_tab, load_tab_predictor
from module.train import _EarlyStop, _opt


def ml_resid(D, cfg, tr, va, te, target, return_predict=False, save_path=None):
    """기본 stage-2: ml 전체특성(cfg['margin'].feature_set) 으로 잔차 회귀.
     target = FAIR − MC_hat − recent_margin. 저장/로드는 fit_tab 규약(.pkl).
     cfg['margin'].half_life=='auto' 이면 비정상성 대응으로 시간감쇠 반감기를 **시간기반 inner-val**(tr 최근 15%)
     에서 fold별 선택(test 미사용 → 미래정보 無). 잔차 MSE 최소화는 fair MSE 최소화와 동치(mc·rm 고정 오프셋)."""
    mg = cfg["margin"]
    hl = mg.get("half_life", 365.25)
    obj = mg.get("loss")                              # None|reg:pseudohubererror 등 (이분산 강건손실)
    if hl == "auto":
        if cfg["data"].get("time_decay", True):
            srt = tr[np.argsort(D.ORD[tr])]; cut = max(1, int(len(srt) * 0.85))
            itr, ival = srt[:cut], srt[cut:]
            hl = 365.25
            if len(ival) >= 20:
                best = (365.25, np.inf)
                for h in [None, 730, 365.25, 182, 91, 45]:
                    pv = fit_tab(D, cfg, mg["model"], itr, ival, mg["feature_set"], target,
                                 tw=True, va=ival, half_life=h, objective=obj)
                    m = float(((target[ival] - pv) ** 2).mean())
                    if m < best[1]:
                        best = (h, m)
                hl = best[0]
        else:
            hl = 365.25
    out = fit_tab(D, cfg, mg["model"], tr, te, mg["feature_set"], target,
                  tw=cfg["data"]["time_decay"], va=va, save_path=save_path,
                  return_predictor=return_predict, half_life=hl, objective=obj)
    if return_predict:
        yp, p = out
        return p(tr), yp, p
    return None, out


def load_ml_resid(D, path):
    """ml_resid 저장 .pkl → predict(idx)."""
    return load_tab_predictor(D, path)


def xgb_resid(D, cfg, tr, va, te, target, return_predict=False, save_path=None):
    """기본 stage-2: XGB on deeponet 블록 D.DON. target=잔차(MC−MC_hat 등)."""
    from xgboost import XGBRegressor
    XP = cfg["tabular"]["xgb"]; X = D.DON
    w = time_weights(D, tr) if cfg["data"]["time_decay"] else None
    es = va is not None and len(va) > 0
    kw = dict(n_estimators=XP["n_estimators"], learning_rate=XP["learning_rate"], max_depth=XP["max_depth"],
              subsample=XP["subsample"], colsample_bytree=XP["colsample_bytree"], n_jobs=0, random_state=cfg["seed"])
    if es:
        kw["early_stopping_rounds"] = cfg["train"].get("es_rounds", 50)
    m = XGBRegressor(**kw)
    if es:
        m.fit(X[tr], target[tr], sample_weight=w, eval_set=[(X[va], target[va])], verbose=False)
    else:
        m.fit(X[tr], target[tr], sample_weight=w)
    if save_path:
        import os, joblib
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model": m, "kind": "resid_don_xgb"}, save_path + ".pkl")

    def p(idx):
        return m.predict(X[idx]).astype("float32")
    return (p(tr), p(te), p) if return_predict else (p(tr), p(te))


def load_xgb_resid(D, path):
    """xgb_resid 저장 .pkl → predict(idx). 입력 = deeponet 블록 D.DON."""
    import joblib
    m = joblib.load(path)["model"]
    return lambda idx: m.predict(D.DON[np.asarray(idx)]).astype("float32")


def don_resid(D, cfg, tr, va, te, target, return_predict=False, save_path=None):
    """DeepONet stage-2: branch=D.VC(vol·corr·sig_eff), trunk=[곡선 u0-9 | 계약]. 표준화는 train 폴드 기준."""
    dev = D.DEV; B = cfg["train"]["batch"]; NIT = cfg["train"]["nit"]; P = cfg["networks"]["P"]
    torch.manual_seed(cfg["seed"])
    Bmat = D.VC; Tmat = np.concatenate([D.CURVE, D.CON], axis=1)
    bm, bs = zstats(Bmat, tr); tm, ts = zstats(Tmat, tr)
    Bt = to_tensor(znorm(Bmat, bm, bs), dev); Tt = to_tensor(znorm(Tmat, tm, ts), dev)
    ym, ysd = float(target[tr].mean()), float(target[tr].std() + 1e-8)
    Y = to_tensor((target - ym) / ysd, dev)
    net = MarginOperator(Bt.shape[1], Tt.shape[1], P).to(dev); opt = _opt(net, cfg)
    trt = torch.tensor(tr, device=dev)
    if cfg["data"]["time_decay"]:
        wsamp = torch.tensor(time_weights(D, tr), device=dev, dtype=torch.float32)
    else:
        wsamp = torch.ones(len(tr), device=dev)
    es = _EarlyStop(net, cfg, va, dev)
    for it in range(NIT):
        bb = trt[torch.multinomial(wsamp, B, replacement=True)]
        pr = net.V(Bt[bb], Tt[bb])
        opt.zero_grad(); ((pr - Y[bb]) ** 2).mean().backward(); opt.step()
        if es.step(it, lambda vi: ((net.V(Bt[vi], Tt[vi]) - Y[vi]) ** 2).mean()):
            break
    es.restore(); net.eval()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state": net.state_dict(), "P": P, "nb": Bt.shape[1], "nt": Tt.shape[1],
                    "bm": bm, "bs": bs, "tm": tm, "ts": ts, "ym": ym, "ysd": ysd}, save_path + ".pt")

    def p(idx):
        with torch.no_grad():
            i = torch.tensor(idx, device=dev)
            return (net.V(Bt[i], Tt[i]) * ysd + ym).cpu().numpy()
    return (p(tr), p(te), p) if return_predict else (p(tr), p(te))


def load_don_resid(D, path):
    """don_resid 저장 .pt → predict(idx). deeponet 블록(branch=VC, trunk=[곡선|계약])."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = MarginOperator(ck["nb"], ck["nt"], ck["P"]).to(D.DEV)
    net.load_state_dict(ck["state"]); net.eval()
    Bmat = D.VC; Tmat = np.concatenate([D.CURVE, D.CON], axis=1)
    Bt = to_tensor(znorm(Bmat, ck["bm"], ck["bs"]), D.DEV)
    Tt = to_tensor(znorm(Tmat, ck["tm"], ck["ts"]), D.DEV)
    ym, ysd = ck["ym"], ck["ysd"]

    def predict(idx):
        with torch.no_grad():
            i = torch.tensor(np.asarray(idx), device=D.DEV)
            return (net.V(Bt[i], Tt[i]) * ysd + ym).cpu().numpy()
    return predict
