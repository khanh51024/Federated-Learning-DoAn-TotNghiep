"""
fl_training.progress: Single-progress-bar renderer and structured JSONL event logging.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm


class EventLogger:
    """
    Writer of events.jsonl for the ServerApp:
    Ensures structured, atomic-flushed events are communicated to the launcher.
    """

    def __init__(
        self, log_path: Path, run_id: str, attempt_id: int = 1,
        client_id: Optional[int] = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.attempt_id = int(attempt_id)
        self.client_id = None if client_id is None else int(client_id)
        self.seq = 0
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        existing.get("run_id") == self.run_id
                        and int(existing.get("attempt_id", -1)) == self.attempt_id
                        and existing.get("client_id") == self.client_id
                    ):
                        self.seq = max(self.seq, int(existing.get("seq", 0)))

    def _emit(self, event_type: str, round_num: int, payload: Optional[Dict[str, Any]] = None) -> None:
        self.seq += 1
        event = {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "seq": self.seq,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": event_type,
            "round": int(round_num),
            "payload": payload or {},
        }
        if self.client_id is not None:
            event["client_id"] = self.client_id
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def log_phase(self, round_num: int, phase_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        p = {"phase": phase_name}
        if payload:
            p.update(payload)
        self._emit("phase", round_num, p)

    def log_round_completed(
        self,
        round_num: int,
        val_loss: float,
        best_loss: float,
        lr: float,
        bad_rounds: int,
        patience: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        duration: Optional[float] = None,
        is_best: bool = False,
    ) -> None:
        payload = {
            "val_loss": float(val_loss),
            "best_loss": float(best_loss),
            "lr": float(lr),
            "bad_rounds": int(bad_rounds),
            "patience": int(patience),
            "is_best": bool(is_best),
        }
        if train_loss is not None:
            payload["train_loss"] = float(train_loss)
        if train_acc is not None:
            payload["train_accuracy"] = float(train_acc)
        if duration is not None:
            payload["duration_seconds"] = float(duration)
        self._emit("round_completed", round_num, payload)

    def log_stopped(self, round_num: int, reason: str, payload: Optional[Dict[str, Any]] = None) -> None:
        p = {"reason": reason}
        if payload:
            p.update(payload)
        self._emit("stopped", round_num, p)

    def log_failed(self, round_num: int, error_message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        p = {"error": error_message}
        if payload:
            p.update(payload)
        self._emit("failed", round_num, p)

    def log_client_epoch(self, record: Dict[str, Any]) -> None:
        self._emit("client_epoch", int(record["round"]), dict(record))

    def log_checkpoint(self, round_num: int, paths: Dict[str, str]) -> None:
        self._emit("checkpoint", round_num, paths)


class ProgressRenderer:
    """
    Launcher-owned single progress bar.
    Monitors events.jsonl and renders either a tqdm bar (TTY) or clean periodic log lines.
    """

    def __init__(
        self,
        event_log_path: Path,
        max_rounds: int,
        description: str = "Training",
        enabled: bool = True,
        is_tty: Optional[bool] = None,
        client_logs_dir: Optional[Path] = None,
    ):
        self.event_log_path = Path(event_log_path)
        self.max_rounds = int(max_rounds)
        self.description = description
        self.enabled = bool(enabled)
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty

        self.last_read_pos = 0
        self.partial_line = ""
        self.client_logs_dir = Path(client_logs_dir) if client_logs_dir else None
        self.client_positions: Dict[Path, int] = {}
        self.client_partials: Dict[Path, str] = {}
        self.completed_rounds = 0
        self.pbar: Optional[tqdm] = None

        if self.enabled and self.is_tty:
            self.pbar = tqdm(
                total=self.max_rounds,
                desc=self.description,
                unit="round",
                dynamic_ncols=True,
            )

    def poll(self) -> Optional[str]:
        """
        Read new events from events.jsonl and update progress bar.
        Returns 'stopped', 'failed', or None.
        """
        if not self.event_log_path.exists():
            return None

        status = None
        with open(self.event_log_path, "r", encoding="utf-8") as f:
            f.seek(self.last_read_pos)
            chunk = f.read()
            self.last_read_pos = f.tell()
        combined = self.partial_line + chunk
        lines = combined.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.partial_line = lines.pop()
        else:
            self.partial_line = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            etype = event.get("type")
            r = event.get("round", 0)
            payload = event.get("payload", {})

            if etype == "round_completed":
                self.completed_rounds = r
                val_loss = payload.get("val_loss", 0.0)
                best_loss = payload.get("best_loss", 0.0)
                lr = payload.get("lr", 0.0)
                bad = payload.get("bad_rounds", 0)
                pat = payload.get("patience", 0)

                postfix = {
                    "val_loss": f"{val_loss:.4f}",
                    "best": f"{best_loss:.4f}",
                    "lr": f"{lr:.6f}",
                    "bad/pat": f"{bad}/{pat}",
                }

                if self.pbar is not None:
                    self.pbar.n = r
                    self.pbar.set_postfix(postfix)
                    self.pbar.refresh()
                elif self.enabled:
                    print(
                        f"[{self.description}] Round {r}/{self.max_rounds} - "
                        f"val_loss={val_loss:.4f}, best={best_loss:.4f}, lr={lr:.6f}, bad={bad}/{pat}"
                    )

            elif etype == "phase":
                phase = payload.get("phase", "")
                if self.pbar is not None:
                    self.pbar.set_description(f"{self.description} ({phase})")

            elif etype == "client_epoch":
                if self.pbar is not None:
                    cid = payload.get("client_id", "?")
                    ep = payload.get("local_epoch", "?")
                    loss = float(payload.get("loss", 0.0))
                    self.pbar.set_description(f"{self.description} (r{r} c{cid} e{ep}, loss={loss:.4f})")

            elif etype == "checkpoint":
                if self.pbar is not None:
                    self.pbar.set_description(f"{self.description} (checkpoint)")

            elif etype == "stopped":
                status = "stopped"
                reason = payload.get("reason", "Early stopped")
                if self.pbar is not None:
                    self.pbar.set_postfix_str(f"Stopped: {reason}")
                elif self.enabled:
                    print(f"[{self.description}] Stopped at round {r}: {reason}")

            elif etype == "failed":
                status = "failed"
                err = payload.get("error", "Error")
                if self.pbar is not None:
                    self.pbar.set_postfix_str(f"FAILED: {err}")
                elif self.enabled:
                    print(f"[{self.description}] FAILED at round {r}: {err}")

        self._poll_client_logs()
        return status

    def _poll_client_logs(self) -> None:
        if self.client_logs_dir is None or not self.client_logs_dir.exists():
            return
        for path in sorted(self.client_logs_dir.glob("client_*.jsonl")):
            pos = self.client_positions.get(path, 0)
            with open(path, "r", encoding="utf-8") as f:
                f.seek(pos)
                chunk = f.read()
                self.client_positions[path] = f.tell()
            combined = self.client_partials.get(path, "") + chunk
            lines = combined.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                self.client_partials[path] = lines.pop()
            else:
                self.client_partials[path] = ""
            for line in lines:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if event.get("type") != "client_epoch":
                    continue
                payload = event.get("payload", {})
                cid = event.get("client_id", "?")
                r = event.get("round", 0)
                ep = payload.get("local_epoch", "?")
                loss = payload.get("loss", 0.0)
                if self.pbar is not None:
                    self.pbar.set_description(
                        f"{self.description} (r{r} c{cid} e{ep}, loss={loss:.4f})"
                    )

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
