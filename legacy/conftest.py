"""Cấu hình pytest cho code lịch sử (pipeline Kaggle / stage*).

Chạy riêng, không thuộc lượt test mặc định:
    .\.venv\Scripts\python.exe -m pytest legacy/tests -q

`legacy/` được đưa vào sys.path và PYTHONPATH để `import fl_training`,
`import stage1_compat`... và các subprocess `python -m fl_training.cli` vẫn chạy.
Đường dẫn dữ liệu (`data/`, `results/`) vẫn tính từ gốc repo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parent
REPO = LEGACY.parent

for entry in (str(LEGACY), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(LEGACY), str(REPO), os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)
