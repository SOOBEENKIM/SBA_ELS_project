# -*- coding: utf-8 -*-
"""파일 경로 중앙 관리. AbacusDirectoryPath 형식 차용 (클래스 기반 경로 + get_* 게터).

하위호환: 기존 코드가 쓰던 모듈 레벨 API(ROOT, dataset(), prediction() 등)는
싱글턴 PATH 를 감싸 그대로 제공하며, 전부 pathlib.Path 를 반환한다.
"""
import os
import glob
from pathlib import Path
from typing import Optional

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ELSDirectoryPath:
    def __init__(self, base_directory: str):
        self.base_directory = base_directory

        ############################################################
        # Config
        ############################################################
        self.config_file_name = os.path.join(self.base_directory, "config.yaml")

        ############################################################
        # Data (원천 · 모델링 데이터셋 3종)
        ############################################################
        self.data_dir_name = os.path.join(self.base_directory, "data")
        self.raw_dir_name = os.path.join(self.data_dir_name, "raw")
        self.cache_dir_name = os.path.join(self.data_dir_name, "cache")
        self.source_file_name = os.path.join(
            self.data_dir_name, "els3_dataset.parquet"
        )
        self.ml_dataset_file_name = os.path.join(self.data_dir_name, "ml.csv")
        self.deeponet_dataset_file_name = os.path.join(
            self.data_dir_name, "deeponet.csv"
        )

        ############################################################
        # Result (예측 · 그림 · 통계)
        ############################################################
        self.result_dir_name = os.path.join(self.base_directory, "result")
        self.prediction_dir_name = os.path.join(self.result_dir_name, "predictions")
        self.image_dir_name = os.path.join(self.result_dir_name, "image")
        self.statistic_dir_name = os.path.join(self.result_dir_name, "statistics")

        ############################################################
        # Scratch
        ############################################################
        self.scratch_dir_name = os.path.join(self.base_directory, "scratch")

        # 데이터셋 이름 -> 경로
        self.dataset_files = {
            "ml": self.ml_dataset_file_name,
            "deeponet": self.deeponet_dataset_file_name,
        }

    ############################################################
    # Base / Config
    ############################################################
    def get_base_dir(self):
        """프로젝트 루트 경로 반환"""
        return self.base_directory

    def get_config_file(self):
        """config.yaml 경로 반환"""
        return self.config_file_name

    def ensure_dirs(self):
        """필요한 디렉토리 생성"""
        for d in (
            self.data_dir_name,
            self.raw_dir_name,
            self.cache_dir_name,
            self.result_dir_name,
            self.prediction_dir_name,
            self.image_dir_name,
            self.statistic_dir_name,
            self.scratch_dir_name,
        ):
            os.makedirs(d, exist_ok=True)

    ############################################################
    # Data
    ############################################################
    def get_data_dir(self):
        """data 디렉토리 경로 반환"""
        return self.data_dir_name

    def get_raw_dir(self):
        """원천 CSV(raw) 디렉토리 경로 반환"""
        return self.raw_dir_name

    def get_cache_dir(self):
        """cache 디렉토리 경로 반환"""
        return self.cache_dir_name

    def get_source_file(self):
        """원천(피처 완성본) parquet 경로 반환"""
        return self.source_file_name

    def get_dataset_file(self, name: str):
        """모델링 데이터셋 경로 반환 (name: ml | deeponet)"""
        return self.dataset_files[name]

    ############################################################
    # Result
    ############################################################
    def get_result_dir(self):
        """result 디렉토리 경로 반환"""
        return self.result_dir_name

    def get_prediction_file(self, name: str):
        """모델별 예측 csv 경로 반환 (디렉토리 자동 생성)"""
        os.makedirs(self.prediction_dir_name, exist_ok=True)
        return os.path.join(self.prediction_dir_name, f"{name}.csv")

    def get_image_file(self, name: str):
        """그림(png) 경로 반환 (디렉토리 자동 생성)"""
        os.makedirs(self.image_dir_name, exist_ok=True)
        return os.path.join(self.image_dir_name, f"{name}.png")

    def get_statistic_file(self, name: str):
        """통계치(csv) 경로 반환 (디렉토리 자동 생성)"""
        os.makedirs(self.statistic_dir_name, exist_ok=True)
        return os.path.join(self.statistic_dir_name, f"{name}.csv")

    def list_prediction_files(self):
        """저장된 예측 csv 경로 목록 반환"""
        return sorted(glob.glob(os.path.join(self.prediction_dir_name, "*.csv")))

    ############################################################
    # Scratch
    ############################################################
    def get_scratch_dir(self):
        """scratch 디렉토리 경로 반환"""
        return self.scratch_dir_name


# ============================================================================
# 모듈 싱글턴 + 하위호환 API (기존 코드는 fm.ROOT / fm.dataset() 등을 그대로 사용)
# 모든 반환값은 pathlib.Path 로 감싸 기존 .exists()/.glob()/relative_to()/"/" 를 지원.
# ============================================================================
PATH = ELSDirectoryPath(ROOT_DIRECTORY)

ROOT = Path(ROOT_DIRECTORY)
CONFIG = Path(PATH.config_file_name)
DATA = Path(PATH.data_dir_name)
RAW = Path(PATH.raw_dir_name)
CACHE = Path(PATH.cache_dir_name)
RESULT = Path(PATH.result_dir_name)
PRED = Path(PATH.prediction_dir_name)
IMAGE = Path(PATH.image_dir_name)
STAT = Path(PATH.statistic_dir_name)
SCRATCH = Path(PATH.scratch_dir_name)
DATASETS = {k: Path(v) for k, v in PATH.dataset_files.items()}


def ensure_dirs():
    PATH.ensure_dirs()


def dataset(name: str) -> Path:
    """모델링 데이터셋 경로 (ml | deeponet)."""
    return Path(PATH.get_dataset_file(name))


def source() -> Path:
    """원천(피처 완성본) 경로."""
    return Path(PATH.get_source_file())


def prediction(name: str) -> Path:
    return Path(PATH.get_prediction_file(name))


def image(name: str) -> Path:
    return Path(PATH.get_image_file(name))


def stat(name: str) -> Path:
    return Path(PATH.get_statistic_file(name))


def list_predictions():
    """저장된 예측 csv 경로 목록."""
    return [Path(p) for p in PATH.list_prediction_files()]
