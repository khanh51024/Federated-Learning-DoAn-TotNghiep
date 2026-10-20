"""Bounded localhost Flower smoke session owned by the web process.

The session uses the legacy scratch PlantVillage partition contract with
random-noise images (no image folder is configured for the web demo). Its metrics
are engineering telemetry, never published PlantVillage/PlantDoc validation results.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src import registry as reg


ROOT = Path(__file__).resolve().parents[1]
PARTITION = ROOT / "data/partitions_stage2_scratch_v3/label01"

_ROUND_KEYS = (
    "round", "accepted_clients", "suspicious_clients", "rejected_clients", "round_fit_elapsed_ms",
    "mean_probe_rtt_ms", "parent_model_hash", "global_model_hash", "mean_received_learnable_update_norm",
)
_CLIENT_KEYS = (
    "client_id", "num_examples", "local_train_time_s", "received_learnable_update_norm",
    "learnable_update_relative", "status", "data_mode",
)


def _brief_rounds(rounds: Any) -> list[dict[str, Any]]:
    """Rút gọn audit_rounds của server Flower thành thứ giao diện cần để chứng minh máy đã tính thật."""
    brief: list[dict[str, Any]] = []
    for record in rounds if isinstance(rounds, list) else []:
        if not isinstance(record, dict):
            continue
        row = {key: record.get(key) for key in _ROUND_KEYS}
        row["clients"] = [
            {key: client.get(key) for key in _CLIENT_KEYS}
            for client in record.get("clients", []) if isinstance(client, dict)
        ]
        brief.append(row)
    return brief


class NetworkDemoManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._spawned: dict[str, dict[str, Any]] = {}
        self._logs: list[Any] = []
        self._running = False
        self._final_exit_codes: dict[str, int | None] = {}
        self.run_dir: Path | None = None

    def detach_run(self) -> None:
        """Bỏ liên kết với phiên Flower đã xong để các tác vụ khác không hiển thị RTT/số client của phiên cũ.

        Không đụng gì khi đang chạy. Cố ý KHÔNG đặt run_dir = None trong `_close()`: sau khi dừng, bài kiểm
        và trang vẫn cần đọc phiên vừa rồi."""
        with self._lock:
            if self._running:
                return
            self.run_dir = None
            self._final_exit_codes = {}
            self._spawned = {}

    def prepare(self) -> None:
        """Reset cancellation before the API schedules the background task."""
        with self._lock:
            if self._running:
                raise RuntimeError("A Flower demo is already running")
            self._stop.clear()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            processes = {name: process.poll() for name, process in self._processes.items()}
            pids = {name: spec.get("pid") for name, spec in self._spawned.items()}
            run_dir = self.run_dir
            active = self._running
            final_codes = dict(self._final_exit_codes)
        progress = None
        if run_dir is not None:
            progress_path = run_dir / "progress.json"
            if progress_path.is_file():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        rows = processes or final_codes  # sau khi phiên đóng, _processes rỗng: dùng mã thoát đã chụp lúc đóng
        return {
            "run_dir": str(run_dir) if run_dir else None,
            "active": active,
            "processes": [
                {"name": name, "pid": pids.get(name), "running": exit_code is None, "exit_code": exit_code}
                for name, exit_code in sorted(rows.items())
            ],
            "rounds_brief": _brief_rounds(progress.get("rounds") if progress else None),
            "parent_model_hash": progress.get("parent_model_hash") if progress else None,
            "server_running": processes.get("server") is None and "server" in processes,
            "client_processes_running": sum(
                exit_code is None for name, exit_code in processes.items() if name.startswith("client_")
            ),
            "process_exit_codes": processes,
            "round_completed": progress.get("completed_round", 0) if progress else 0,
            "round_in_progress": progress.get("round_in_progress", 0) if progress else 0,
            "connected_clients": progress.get("connected_clients", 0) if progress else 0,
            "accepted_clients_last_round": progress.get("accepted_clients", 0) if progress else 0,
            "mean_probe_rtt_ms": (
                progress["rounds"][-1].get("mean_probe_rtt_ms")
                if progress and progress.get("rounds") else None
            ),
            "stop_requested": self._stop.is_set(),
        }

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def _close(self) -> None:
        with self._lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=4)
        for stream in self._logs:
            stream.close()
        self._logs.clear()
        with self._lock:
            self._final_exit_codes = {name: process.returncode for name, process in self._processes.items()}
            self._processes.clear()
            self._running = False

    def _spawn(self, name: str, args: list[str], run_dir: Path) -> subprocess.Popen[bytes]:
        stream = (run_dir / f"{name}.log").open("wb")
        self._logs.append(stream)
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        argv = [sys.executable, "-u", *args]
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._lock:
            self._processes[name] = process
            self._spawned[name] = {"argv": argv, "pid": process.pid, "log": f"{name}.log",
                                   "started_at_utc": datetime.now(timezone.utc).isoformat()}
        return process

    def _write_session(self, run_dir: Path, metadata: dict[str, Any]) -> None:
        """Ghi session.json kèm đúng argv + PID đã spawn, để trang Flower hiển thị lệnh thật chứ không phải lệnh minh hoạ."""
        with self._lock:
            metadata["commands"] = {name: dict(spec) for name, spec in self._spawned.items()}
        (run_dir / "session.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(
        self,
        *,
        port: int,
        rounds: int = 2,
        local_epochs: int = 1,
        seed: int = 42,
        reject_suspicious_updates: bool = False,
        max_update_norm: float | None = None,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
        timeout_seconds: int = 900,
    ) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise RuntimeError("A Flower demo is already running")
            self._running = True
            self._processes.clear()
            self._spawned.clear()
            self._final_exit_codes = {}
        run_dir = reg.abs_path(reg.NETWORK_SMOKE_RUNS_DIR) / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        metadata = {
            "kind": "flower_localhost_synthetic_smoke",
            "scientific_valid": False,
            "published_result_reproduction": False,
            "client_processes": 5,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "smoke_samples_per_present_class": 1,
            "seed": seed,
            "reject_suspicious_updates": reject_suspicious_updates,
            "max_update_norm": max_update_norm,
            "bind_address": f"127.0.0.1:{port}",
            "connect_address": f"127.0.0.1:{port}",
            "partition_dir": str(PARTITION),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_session(run_dir, metadata)

        def report() -> None:
            if status_callback is not None:
                status_callback(self.snapshot())

        try:
            if not PARTITION.is_dir():
                raise FileNotFoundError(f"Missing smoke partition: {PARTITION}")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Flower port {port} is already in use")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port + 1)) == 0:
                    raise RuntimeError(f"RTT probe port {port + 1} is already in use")

            server_args = [
                "-m", "src.federated.server_app", "--mode", "deployment",
                "--host", "127.0.0.1", "--port", str(port),
                "--num-clients", "5", "--rounds", str(rounds),
                "--local-epochs", str(local_epochs), "--seed", str(seed),
                "--smoke", "--evaluation-partition-dir", str(PARTITION),
                "--metrics-file", str(run_dir / "history.json"),
                "--checkpoint-file", str(run_dir / "global.pt"),
                "--progress-file", str(run_dir / "progress.json"),
            ]
            server_args.append("--reject-suspicious-updates" if reject_suspicious_updates else "--observe-only")
            if max_update_norm is not None:
                server_args.extend(["--max-update-norm", str(max_update_norm)])
            server = self._spawn("server", server_args, run_dir)
            self._write_session(run_dir, metadata)  # lệnh của server phải nằm trên đĩa kể cả khi vòng chờ mở cổng sau đó timeout
            deadline = time.monotonic() + min(timeout_seconds, 60)
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    raise InterruptedError("Flower demo stopped by user")
                if server.poll() is not None:
                    raise RuntimeError(f"Flower server exited during startup ({server.returncode})")
                with socket.socket() as sock:
                    sock.settimeout(0.25)
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.3)
            else:
                raise TimeoutError("Flower server did not open its gRPC port within 60 seconds")
            report()

            for client_id in range(5):
                if self._stop.is_set():
                    raise InterruptedError("Flower demo stopped by user")
                self._spawn(f"client_{client_id:02d}", [
                    "-m", "src.federated.client_app", "--client-id", str(client_id),
                    "--server", f"127.0.0.1:{port}", "--partition-dir", str(PARTITION),
                    "--smoke", "--local-epochs", str(local_epochs), "--seed", str(seed),
                ], run_dir)
                self._write_session(run_dir, metadata)
                report()

            deadline = time.monotonic() + timeout_seconds
            while server.poll() is None:
                if self._stop.is_set():
                    raise InterruptedError("Flower demo stopped by user")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Chạy thử Flower vượt quá {timeout_seconds} giây")
                report()
                time.sleep(0.5)
            report()
            if self._stop.is_set():
                raise InterruptedError("Flower demo stopped by user")
            if server.returncode != 0:
                raise RuntimeError(f"Flower server failed (exit {server.returncode}); see {run_dir / 'server.log'}")
            history_path = run_dir / "history.json"
            checkpoint_path = run_dir / "global.pt"
            if not history_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError("Flower server exited without final history/checkpoint")
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if len(history.get("audit_rounds", [])) != rounds or any(
                row.get("accepted_clients") != 5 for row in history["audit_rounds"]
            ):
                raise RuntimeError("Flower run did not audit 5 clients in every round")
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["status"] = "completed"
            self._write_session(run_dir, metadata)
            return {"run_dir": str(run_dir), "rounds": rounds, "accepted_clients": 5}
        except BaseException as exc:
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["status"] = "cancelled" if isinstance(exc, InterruptedError) else "failed"
            metadata["error"] = str(exc)
            self._write_session(run_dir, metadata)
            raise
        finally:
            self._close()
            report()
