"""Quản lý budget, quota, process lock và ledger bền vững cho profile stage1_compat."""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from contextlib import contextmanager
from functools import wraps
from stage1_compat.integrity import atomic_json, read_json, finite

from stage1_compat.constants import (
    MAX_SESSION_HOURS,
    RESERVE_HOURS,
    STOP_BEFORE_MINUTES,
)


def is_process_running(pid: int) -> bool:
    """Kiểm tra tiến trình có đang sống hay không (hỗ trợ Windows và Unix)."""
    if pid <= 0:
        return False
    # stdlib only: lock checks must also work in a minimal CPU collector environment.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            # Access denied does not prove a process has exited: retain its lock.
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: absent PID
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LockError(Exception):
    pass


class RunnerLock:
    """OS-owned exclusive lock; automatically released on crash. Never unlink inode."""
    def __init__(self, lock_file):
        self.lock_file = Path(lock_file)
        self.stream = None

    def acquire(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        stream = self.lock_file.open("a+b")
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise LockError("Runner lock đang được giữ by another process") from error
        self.stream = stream
        return True

    def release(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


@dataclass
class JobLedgerEntry:
    job_id: str
    alpha: float
    seed: int
    status: str  # PENDING, RUNNING, COMPLETED, PAUSED
    current_round: int
    total_rounds: int
    elapsed_seconds: float = 0.0
    best_round: int = 0
    best_validation_accuracy: float = 0.0
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
    identity_hash: str | None = None
    round_events: dict = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exclusive(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if self._locked:
            return method(self, *args, **kwargs)
        with self.get_lock():
            return method(self, *args, **kwargs)
    return wrapped


class BudgetLedger:
    """Ledger ghi nhận sử dụng quota và trạng thái các job bền vững qua nhiều phiên."""

    def __init__(
        self,
        output_dir: Path,
        user_quota_hours: float = 0.0,
        reserve_hours: float = RESERVE_HOURS,
        max_session_hours: float = MAX_SESSION_HOURS,
        stop_before_minutes: float = STOP_BEFORE_MINUTES,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output_dir / "stage1_ledger.json"
        self.lock_path = self.output_dir / "stage1_runner.lock"

        self.user_quota_hours = finite(user_quota_hours, "quota", maximum=30)
        self.reserve_hours = finite(reserve_hours, "reserve", minimum=3, maximum=29)
        self.max_session_hours = finite(max_session_hours, "session", minimum=0.01, maximum=8)
        self.stop_before_minutes = finite(stop_before_minutes, "session reserve", minimum=0, maximum=max_session_hours * 60 - 0.01)

        self._locked = False
        self.session_start_time = time.time()
        self.total_used_seconds = 0.0
        self.jobs: dict[str, JobLedgerEntry] = {}

        self._load_or_init_ledger()

    @contextmanager
    def get_lock(self):
        if self._locked:
            raise LockError("Nested runner session")
        with RunnerLock(self.lock_path):
            self._load_or_init_ledger()
            self._locked = True
            try:
                yield self
            finally:
                self._locked = False

    def _load_or_init_ledger(self) -> None:
        if not self.ledger_path.exists():
            return  # constructor is read-only; write only while holding runner lock
        data = read_json(self.ledger_path)
        if data.get("schema") != 2 or data.get("profile") != "stage1_compat":
            raise ValueError("Legacy/corrupt ledger: preserve output; explicit migration required")
        self.total_used_seconds = finite(data["total_used_seconds"], "stored usage")
        self.jobs = {j["job_id"]: JobLedgerEntry(**j) for j in data["jobs"]}
        for entry in self.jobs.values():
            if (type(entry.current_round) is not int or type(entry.total_rounds) is not int
                    or not 0 <= entry.current_round <= entry.total_rounds):
                raise ValueError("Invalid stored round budget")
            finite(entry.elapsed_seconds, "job elapsed")
            for duration in entry.round_events.values():
                finite(duration, "round elapsed")
        if len(self.jobs) != len(data["jobs"]):
            raise ValueError("Duplicate ledger job")

    def save(self) -> None:
        payload = {
            "schema": 2,
            "profile": "stage1_compat",
            "user_quota_hours": self.user_quota_hours,
            "reserve_hours": self.reserve_hours,
            "max_session_hours": self.max_session_hours,
            "stop_before_minutes": self.stop_before_minutes,
            "total_used_seconds": self.total_used_seconds,
            "total_used_hours": self.total_used_seconds / 3600.0,
            "remaining_quota_hours": max(0.0, self.user_quota_hours - (self.total_used_seconds / 3600.0) - self.reserve_hours),
            "jobs": [j.to_dict() for j in self.jobs.values()],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(self.ledger_path, payload)

    @_exclusive
    def record_job_start(self, job_id: str, alpha: float, seed: int, total_rounds: int, identity_hash: str | None = None) -> JobLedgerEntry:
        if job_id not in self.jobs:
            self.jobs[job_id] = JobLedgerEntry(
                job_id=job_id,
                alpha=alpha,
                seed=seed,
                status="RUNNING",
                identity_hash=identity_hash,
                current_round=0,
                total_rounds=total_rounds,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        else:
            entry = self.jobs[job_id]
            if (entry.alpha, entry.seed, entry.total_rounds, entry.identity_hash) != (alpha, seed, total_rounds, identity_hash):
                raise ValueError("Ledger identity/budget mismatch")
            self.jobs[job_id].status = "RUNNING"
        self.save()
        return self.jobs[job_id]

    @_exclusive
    def record_round_progress(
        self,
        job_id: str,
        round_idx: int,
        elapsed_seconds: float,
        best_round: int,
        best_val_acc: float,
    ) -> None:
        if job_id not in self.jobs:
            self.record_job_start(job_id=job_id, alpha=0.0, seed=0, total_rounds=10)
        elapsed_seconds = finite(elapsed_seconds, "round duration")
        key = str(round_idx)
        if key in self.jobs[job_id].round_events:
            return
        self.jobs[job_id].round_events[key] = elapsed_seconds
        self.jobs[job_id].current_round = max(round_idx, self.jobs[job_id].current_round)
        self.jobs[job_id].elapsed_seconds += elapsed_seconds
        self.jobs[job_id].best_round = best_round
        self.jobs[job_id].best_validation_accuracy = best_val_acc
        self.total_used_seconds += elapsed_seconds
        self.save()

    @_exclusive
    def record_job_completion(
        self,
        job_id: str,
        test_accuracy: float,
        test_macro_f1: float,
    ) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].status = "COMPLETED"
            self.jobs[job_id].test_accuracy = test_accuracy
            self.jobs[job_id].test_macro_f1 = test_macro_f1
            self.jobs[job_id].completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.save()

    def can_continue(self, estimated_next_round_seconds: float = 60.0) -> tuple[bool, str]:
        """Kiểm tra điều kiện dừng: reserve, session limit và stop_before_minutes."""
        estimated_next_round_seconds = finite(estimated_next_round_seconds, "estimate")
        # The supervised launcher owns wall-clock accounting, including setup/report/recovery.
        if os.environ.get("STAGE1_SOFT_DEADLINE"):
            if time.time() + estimated_next_round_seconds >= float(os.environ["STAGE1_SOFT_DEADLINE"]):
                return False, "Session/total budget deadline reached"
            return True, "OK"
        # 1. Session time limit (<= 8h)
        session_elapsed_seconds = time.time() - self.session_start_time
        session_max_seconds = self.max_session_hours * 3600.0
        if session_elapsed_seconds + estimated_next_round_seconds + (self.stop_before_minutes * 60.0) >= session_max_seconds:
            return False, f"Đã chạm ngưỡng thời lượng session tối đa ({self.max_session_hours}h). Dừng an toàn để bảo toàn checkpoint."

        # 2. User quota check (nếu user có nhập quota_hours > 0)
        if self.user_quota_hours >= 0:
            effective_usable_seconds = max(0.0, (self.user_quota_hours - self.reserve_hours) * 3600.0)
            if self.total_used_seconds + estimated_next_round_seconds + (self.stop_before_minutes * 60.0) >= effective_usable_seconds:
                return False, f"Quota khả dụng không còn đủ (đã trừ {self.reserve_hours}h reserve và {self.stop_before_minutes}m buffer). Dừng an toàn."

        return True, "OK"


class BudgetExhausted(RuntimeError):
    pass


def check_deadline():
    value = os.environ.get("STAGE1_SOFT_DEADLINE")
    if value is not None and time.time() >= finite(value, "soft deadline"):
        raise BudgetExhausted("Deadline reached; resume from committed round")


class DeadlineLoader:
    """Checks before requesting a batch without consuming extra DataLoader RNG."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            check_deadline()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            check_deadline()
            yield batch
