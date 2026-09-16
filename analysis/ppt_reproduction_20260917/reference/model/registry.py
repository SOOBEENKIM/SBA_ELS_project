# -*- coding: utf-8 -*-
"""model/ 플러그인 자동발견. model/*.py 가 `run(D, cfg) -> {name: DataFrame}` 규약만 지키면
 여기서 자동 발견·실행된다. 새 모델은 파일 하나만 model/ 에 넣으면 2_run 에 자동 편입.
 규약·입력(D) 설명은 model/_template.py 참고."""
import importlib
import pkgutil

import model as _pkg

_SKIP = {"registry"}   # 언더스코어(_template 등)로 시작하는 모듈은 자동 제외


def discover():
    """run(D, cfg) 를 노출하는 model 하위 모듈들을 (이름, run) 리스트(이름순)로 반환."""
    found = []
    for mi in pkgutil.iter_modules(_pkg.__path__):
        if mi.name.startswith("_") or mi.name in _SKIP:
            continue
        mod = importlib.import_module(f"model.{mi.name}")
        fn = getattr(mod, "run", None)
        if callable(fn):
            found.append((mi.name, fn))
    return sorted(found, key=lambda x: x[0])


def run_all(D, cfg, verbose=True):
    """발견된 모든 모델의 run(D,cfg) 를 실행해 예측 dict 병합 반환."""
    preds = {}
    for name, fn in discover():
        out = fn(D, cfg)
        dup = preds.keys() & out.keys()
        if dup:
            raise ValueError(f"중복 모델 이름 {sorted(dup)} (module 'model.{name}'). 예측 이름은 유일해야 합니다.")
        preds.update(out)
        if verbose:
            print(f"[model.{name}] -> {list(out.keys())}")
    return preds
