"""Read-only Stage 1 baseline reference registry for Stage 2 FedAvg-only scope.

GĐ2 CHỈ TRAIN FEDAVG:
- Tuyệt đối không huấn luyện lại Centralized hoặc Local-only trong GĐ2.
- Kết quả Centralized và Local-only lịch sử chỉ được đọc từ snapshot GĐ1.
- Mọi baseline đều được gắn nhãn reference_only=True và strict_comparison_eligible=False.
- Kiểm tra toàn vẹn SHA256 của các tệp kết quả GĐ1; từ chối so sánh strict khi khác giao thức (AdamW vs SGD, khác split).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from stage1_compat.baselines import BaselineRegistry, BaselineEntry


def get_stage1_baseline_registry() -> BaselineRegistry:
    """Trả về BaselineRegistry chỉ đọc cho các kết quả GĐ1 đã xác minh SHA256."""
    return BaselineRegistry()


def load_historical_baselines(suite_dir: Path | str | None = None) -> Dict[str, Any]:
    """Tải danh sách baseline lịch sử làm tham chiếu (historical_reference)."""
    registry = BaselineRegistry()

    suite_counts_str = "38.984/4.400/10.921"
    if suite_dir is not None:
        s_path = Path(suite_dir) / "suite.json"
        if s_path.is_file():
            s_data = json.loads(s_path.read_text(encoding="utf-8"))
            counts = s_data.get("counts", {})
            suite_counts_str = f"{counts.get('train', 38984):,}/{counts.get('val', 4400):,}/{counts.get('test', 10921):,}".replace(",", ".")

    return {
        "status": "historical_reference",
        "reference_only": True,
        "strict_comparison_eligible": False,
        "mismatch_reasons": [
            f"Khác split: GĐ1 dùng 39.084/4.328/10.893; GĐ2 v3 dùng {suite_counts_str}",
            "Khoảng 7.900+ ảnh test GĐ2 thuộc tập train GĐ1 (theo split đã pin)",
            "Khác optimizer: GĐ1 dùng AdamW; GĐ2 FedAvg dùng SGD",
            "Khác selection policy: GĐ1 Local-only chọn final epoch; GĐ2 chọn best validation",
            "Không có baseline Local-only cho alpha=0.1, quantity hoặc feature skew"
        ],
        "entries": registry.list_all(),
    }
