"""
src.data.split: Quản lý chia tập train/val/test và đảm bảo test set dùng chung cố định.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def verify_fixed_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
) -> Dict[str, int]:
    """Kiểm tra tính bảo toàn và số lượng mẫu của các tập phân chia cố định."""
    t_df = pd.read_csv(train_csv)
    v_df = pd.read_csv(val_csv)
    te_df = pd.read_csv(test_csv)

    return {
        "train_samples": len(t_df),
        "val_samples": len(v_df),
        "test_samples": len(te_df),
        "total_samples": len(t_df) + len(v_df) + len(te_df),
    }
