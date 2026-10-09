"""Bộ nạp module upstream GĐ1 với namespace cách ly, không làm ô nhiễm sys.path."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from stage1_compat.constants import UPSTREAM_DIR


def load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    """Nạp file python thành module độc lập, không thêm thư mục vào sys.path."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file module upstream: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể tạo module spec cho {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Lazy-loaded cache
_cached_modules: dict[str, ModuleType] = {}


def get_upstream_mobilenetv3() -> ModuleType:
    if "mobilenetv3" not in _cached_modules:
        _cached_modules["mobilenetv3"] = load_module_from_file(
            "stage1_compat_upstream_mobilenetv3",
            UPSTREAM_DIR / "models" / "mobilenetv3.py"
        )
    return _cached_modules["mobilenetv3"]


def get_upstream_evaluate() -> ModuleType:
    if "evaluate" not in _cached_modules:
        _cached_modules["evaluate"] = load_module_from_file(
            "stage1_compat_upstream_evaluate",
            UPSTREAM_DIR / "utils" / "evaluate.py"
        )
    return _cached_modules["evaluate"]


def get_upstream_data_partition() -> ModuleType:
    if "data_partition" not in _cached_modules:
        _cached_modules["data_partition"] = load_module_from_file(
            "stage1_compat_upstream_data_partition",
            UPSTREAM_DIR / "utils" / "data_partition.py"
        )
    return _cached_modules["data_partition"]


def get_upstream_reproducibility() -> ModuleType:
    if "reproducibility" not in _cached_modules:
        _cached_modules["reproducibility"] = load_module_from_file(
            "stage1_compat_upstream_reproducibility",
            UPSTREAM_DIR / "utils" / "reproducibility.py"
        )
    return _cached_modules["reproducibility"]
