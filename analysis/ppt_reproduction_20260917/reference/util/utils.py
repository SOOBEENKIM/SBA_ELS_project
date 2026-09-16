# -*- coding: utf-8 -*-
"""범용 헬퍼: config 로딩, 시드 고정, 디바이스 선택."""
import random
import numpy as np
import torch
import yaml

from . import file_manager as fm

_CONFIG = None


def load_config(path=None):
    """config.yaml 로딩 (1회 캐시)."""
    global _CONFIG
    if path is None:
        path = fm.CONFIG
    with open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config():
    return _CONFIG if _CONFIG is not None else load_config()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(cfg=None):
    cfg = cfg or get_config()
    d = cfg.get("device", "auto")
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
