"""
src.federated.telemetry: Đo lường độ trễ mạng và hiệu năng truyền thông (LOSS != LATENCY).
Tách bạch hoàn toàn giữa hàm mất mát (loss) và thời gian thực thi (latency).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ClientRoundTelemetry:
    client_id: int
    round_num: int
    local_train_time_s: float = 0.0
    download_time_ms: float = 0.0
    upload_time_ms: float = 0.0
    network_rtt_ms: float = 0.0
    bytes_downloaded: int = 0
    bytes_uploaded: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    num_examples: int = 0
    trained_update_norm: float = 0.0
    sent_update_norm: float = 0.0
    # Backward-compatible alias used by the existing dashboard/results files.
    # It intentionally mirrors the actual payload norm sent to the server.
    update_norm: float = 0.0


@dataclass
class ServerRoundTelemetry:
    round_num: int
    aggregation_time_ms: float = 0.0
    total_round_time_s: float = 0.0
    clients_participating: int = 0
    global_loss: float = 0.0
    global_accuracy: float = 0.0
    global_macro_f1: float = 0.0
    client_metrics: List[ClientRoundTelemetry] = field(default_factory=list)


class TelemetryManager:
    """Quản lý lưu trữ và xuất báo cáo độ trễ / mạng."""

    def __init__(self, log_dir: Optional[str | Path] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("artifacts/metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[ServerRoundTelemetry] = []

    def record_round(self, round_telemetry: ServerRoundTelemetry) -> None:
        self.history.append(round_telemetry)
        self._save_json()

    def _save_json(self) -> None:
        file_path = self.log_dir / "telemetry_history.json"
        data = [asdict(r) for r in self.history]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Tính toán các chỉ số tổng hợp (mean, median, P95)."""
        if not self.history:
            return {}

        train_times = []
        rtts = []
        upload_bytes = []

        for r in self.history:
            for c in r.client_metrics:
                train_times.append(c.local_train_time_s)
                rtts.append(c.network_rtt_ms)
                upload_bytes.append(c.bytes_uploaded)

        return {
            "total_rounds": len(self.history),
            "train_time_s": {
                "mean": round(float(np.mean(train_times)), 3) if train_times else 0.0,
                "median": round(float(np.median(train_times)), 3) if train_times else 0.0,
                "p95": round(float(np.percentile(train_times, 95)), 3) if train_times else 0.0,
            },
            "network_rtt_ms": {
                "mean": round(float(np.mean(rtts)), 2) if rtts else 0.0,
                "median": round(float(np.median(rtts)), 2) if rtts else 0.0,
                "p95": round(float(np.percentile(rtts, 95)), 2) if rtts else 0.0,
            },
            "total_uploaded_mb": round(sum(upload_bytes) / (1024 * 1024), 2),
        }
