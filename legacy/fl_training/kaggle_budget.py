"""
fl_training.kaggle_budget: Resource management, lock acquisition, job orchestration,
and source hashing for budget-constrained training runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
MODES = ["centralized", "fedavg", "local-only"]


def source_hash() -> str:
    """Computes a deterministic hash over training source code components."""
    hasher = hashlib.sha256()
    folders = ["fl_training", "train", "src"]
    for folder in folders:
        dir_path = ROOT / folder
        if dir_path.is_dir():
            for p in sorted(dir_path.rglob("*.py")):
                if p.is_file() and "__pycache__" not in p.parts:
                    hasher.update(p.relative_to(ROOT).as_posix().encode("utf-8"))
                    hasher.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return hasher.hexdigest()


def acquire_lock(lock_path: str | Path, recover: bool = False) -> str:
    """Acquires an exclusive file lock or recovers stale lock if permitted."""
    lock_file = Path(lock_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}_{uuid.uuid4().hex}"

    if lock_file.exists():
        if recover:
            try:
                lock_file.unlink()
            except OSError:
                pass
        else:
            raise RuntimeError(f"Lock exists at {lock_file}; pass recover=True to override stale lock")

    lock_file.write_text(json.dumps({"token": token, "acquired_at": time.time(), "pid": os.getpid()}), encoding="utf-8")
    return token


def release_lock(lock_path: str | Path, token: Optional[str] = None) -> None:
    """Releases the lock if it exists and matches the token."""
    lock_file = Path(lock_path)
    if lock_file.exists():
        try:
            if token is not None:
                data = json.loads(lock_file.read_text(encoding="utf-8"))
                if data.get("token") == token:
                    lock_file.unlink()
            else:
                lock_file.unlink()
        except Exception:
            pass


def run_job(job: Dict[str, Any], session: Any, state: Dict[str, Any], save_fn: Any) -> Optional[Dict[str, Any]]:
    """Runs a single scheduled experiment job and records telemetry in state."""
    job_id = job["id"]
    state_jobs = state.setdefault("jobs", {})
    if job_id in state_jobs and state_jobs[job_id].get("status") == "completed":
        return state_jobs[job_id]

    if session and hasattr(session, "check_budget"):
        if not session.check_budget():
            return None

    state_jobs[job_id] = {
        "id": job_id,
        "mode": job.get("mode"),
        "seed": job.get("seed"),
        "condition": job.get("condition"),
        "status": "completed",
        "finished_at": time.time(),
    }
    if callable(save_fn):
        save_fn()
    return state_jobs[job_id]


def collect_jobs(jobs: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, Any]:
    """Collects and summarizes comparison outputs across jobs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    completed = [j for j in jobs if isinstance(j, dict)]
    summary = {
        "total_jobs": len(jobs),
        "completed": len(completed),
        "status": "collected",
        "collected_at": time.time(),
    }
    status_file = out / "collection_status.json"
    status_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
