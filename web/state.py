"""Trạng thái dùng chung của web app (background task + Flower network demo)."""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Dict, Optional

from src import registry as reg

from web.network_demo import NetworkDemoManager

# Trạng thái tác vụ nền
CURRENT_TASK: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "completed" | "failed" | "cancelled"
    "experiment": None,
    "type": None,  # "centralized" | "federated" | "federated-network"
    "progress": 0,
    "message": "Hệ thống sẵn sàng",
    "task_id": None,
    "config": None,
    "error": None,
    "live_progress": [],
    "cancel_requested": False,
}

NETWORK_DEMO = NetworkDemoManager()

SMOKE_RUNS_ROOT = reg.abs_path(reg.NETWORK_SMOKE_RUNS_DIR)


def latest_completed_smoke_run() -> Optional[Path]:
    """Recover only a fully audited local smoke run after a web restart."""
    if not SMOKE_RUNS_ROOT.is_dir():
        return None
    for run_dir in sorted(SMOKE_RUNS_ROOT.iterdir(), key=lambda path: path.name, reverse=True):
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        try:
            session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            audits = history["audit_rounds"]
            if (session.get("kind") == "flower_localhost_synthetic_smoke"
                    and session.get("status") == "completed"
                    and session.get("scientific_valid") is False
                    and session.get("published_result_reproduction") is False
                    and session.get("client_processes") == 5
                    and len(audits) == session.get("rounds") == 2
                    and [audit.get("round") for audit in audits] == [1, 2]
                    and all(audit.get("accepted_clients") == 5 for audit in audits)
                    and (run_dir / "global.pt").is_file()):
                return run_dir
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def check_port_open(host: str = "127.0.0.1", port: int = 8080, timeout: float = 0.3) -> bool:
    """Kiểm tra xem Flower gRPC Server có đang mở port lắng nghe thật không."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_and_verify_dataset_root(candidate: Optional[str]) -> Optional[Path]:
    """Kiểm tra sự tồn tại của thư mục ảnh thật PlantVillage."""
    import os

    candidates = []
    if candidate:
        candidates.append(Path(candidate))
    if os.environ.get("DATASET_ROOT"):
        candidates.append(Path(os.environ["DATASET_ROOT"]))
    if os.environ.get("PLANTVILLAGE_DATASET_ROOT"):
        candidates.append(Path(os.environ["PLANTVILLAGE_DATASET_ROOT"]))
    candidates.extend([
        Path("data/PlantVillage"),
        Path("data/plantvillage"),
        Path("../PlantVillage-Dataset/raw/color"),
        Path("PlantVillage-Dataset/raw/color"),
        Path("/home/ubuntu/fl-project/data/PlantVillage"),
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            subdirs = [p for p in c.iterdir() if p.is_dir()]
            if len(subdirs) >= 10:
                return c.resolve()
    return None
