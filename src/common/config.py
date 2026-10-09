"""
src.common.config: Nạp và xử lý cấu hình YAML.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
