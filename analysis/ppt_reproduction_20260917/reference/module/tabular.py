# -*- coding: utf-8 -*-
"""tabular 회귀기 (직접 벤치마크 + 하이브리드 stage-2 마진모델 공용). config 파라미터 사용."""
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from .data import featnum, CAT, time_weights


def load_tab_predictor(D, path):
    """fit_tab 이 저장한 .pkl 로드 → predict(idx)->np.array (재학습 없이 forward)."""
    import joblib
    ck = joblib.load(path)
    m = ck["model"]; ct = ck["ct"]; X = D.ml[ck["num"] + ck["cat"]]
    if ct is not None:   # ridge: ColumnTransformer 적용
        return lambda idx: m.predict(ct.transform(X.iloc[idx]))
    return lambda idx: m.predict(X.iloc[idx])


def fit_tab(D, cfg, model, tr, te, feat, target, tw=True, va=None, save_path=None, return_predictor=False,
           half_life=365.25, objective=None):
    """train 에만 fit, 시간가중. target: 전체 길이 배열, feat: 'base'|'regime'|'driver'.
    va(있으면): xgb/lgbm/cat 는 val 조기종료(early stopping)에 사용 (ridge/gbm 은 미사용).
    half_life: 시간감쇠 반감기(일). None=균등. (stage-2 ml_resid 가 inner-val 로 선택해 넘길 수 있음)
    save_path 주면 학습 모델+전처리를 <save_path>.pkl 로 저장(로드용).
    return_predictor=True 이면 (preds, predictor) 반환 — predictor(idx)는 D.ml 행 idx의 예측(추론 속도 측정용)."""
    num = featnum(feat)
    X = D.ml[num + CAT]
    Xtr = X.iloc[tr]
    ytr = target[tr]
    w = time_weights(D, tr, half_life) if tw else None
    P = cfg["tabular"]
    es = va is not None and len(va) > 0
    ES_R = cfg["train"].get("es_rounds", 50)
    ct = None
    if es:
        Xva = X.iloc[va]; yva = target[va]

    if model == "ridge":
        ct = ColumnTransformer(
            [
                (
                    "num",
                    SkPipe(
                        [
                            ("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler()),
                        ]
                    ),
                    num,
                ),
                ("c", OneHotEncoder(handle_unknown="ignore"), CAT),
            ]
        ).fit(Xtr)
        m = Ridge(alpha=P["ridge"]["alpha"]).fit(
            ct.transform(Xtr), ytr, sample_weight=w
        )
        predictor = lambda idx: m.predict(ct.transform(X.iloc[idx]))
    elif model == "gbm":
        m = HistGradientBoostingRegressor(
            categorical_features=CAT, random_state=0, **P["gbm"]
        )
        m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "lgbm":
        m = lgb.LGBMRegressor(random_state=0, verbose=-1, **P["lgbm"])
        fkw = dict(categorical_feature=CAT, sample_weight=w)
        if es:
            fkw["eval_set"] = [(Xva, yva)]
            fkw["callbacks"] = [lgb.early_stopping(ES_R, verbose=False)]
        m.fit(Xtr, ytr, **fkw)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "cat":
        m = CatBoostRegressor(random_seed=0, verbose=0, cat_features=CAT, **P["cat"])
        if es:
            m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva), early_stopping_rounds=ES_R)
        else:
            m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    elif model == "xgb":
        xkw = dict(tree_method="hist", enable_categorical=True, random_state=0, **P["xgb"])
        if objective:                                 # stage-2 강건손실(예: reg:pseudohubererror, 이분산 대응)
            xkw["objective"] = objective
        if es:
            xkw["early_stopping_rounds"] = ES_R
        m = xgb.XGBRegressor(**xkw)
        if es:
            m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
        else:
            m.fit(Xtr, ytr, sample_weight=w)
        predictor = lambda idx: m.predict(X.iloc[idx])
    else:
        raise ValueError(model)
    if save_path:
        import os, joblib
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({"model_key": model, "model": m, "ct": ct, "num": num, "cat": CAT, "feat": feat},
                    save_path + ".pkl")
    preds = predictor(te)
    return (preds, predictor) if return_predictor else preds
