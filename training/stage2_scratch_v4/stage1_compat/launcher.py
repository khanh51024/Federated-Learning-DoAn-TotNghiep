"""One owned worker process, bounded deadlines, durable accounting across sessions."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from stage1_compat.accounting import BudgetSession, apply_usage_observations, quota_remaining_from_state
from stage1_compat.budget import RunnerLock
from stage1_compat.integrity import atomic_json, read_json, finite


def stop_owned_tree(process):
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def peak_ram_bytes(pid):
    if os.name != "nt":
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
            return max(int(line.split()[1]) * 1024 for line in lines if line.startswith(("VmHWM:", "VmRSS:")))
        except (OSError, ValueError):
            return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
                ("peak", "working", "pqp", "qp", "pqnp", "qnp", "page", "peakpage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x410, False, pid)
        if not handle:
            return None
        try:
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            fn = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            return int(counters.peak) if fn(handle, ctypes.byref(counters), counters.cb) else None
        finally:
            kernel.CloseHandle(handle)
    except OSError:
        return None


def supervise(command, env, log_path, session):
    with Path(log_path).open("ab") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        peak = None
        timed_out = False
        try:
            while process.poll() is None:
                ram = peak_ram_bytes(process.pid)
                if ram is not None:
                    peak = max(peak or 0, ram)
                session.tick()
                if session.hard_remaining() <= 0:
                    timed_out = True
                    stop_owned_tree(process)
                    break
                try:
                    process.wait(timeout=min(1, max(.01, session.hard_remaining())))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                stop_owned_tree(process)
        return {"returncode": process.returncode, "timed_out": timed_out, "peak_ram_bytes": peak}


def launch(argv, args):
    quota = finite(args.quota_hours, "remaining quota", maximum=30)
    already = finite(args.already_used_hours, "already used", maximum=30)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "account_budget.v2.json"
    with RunnerLock(output / "account_budget.lock"):
        state = read_json(state_path) if state_path.exists() else {"schema": 2, "used_seconds": 0.0}
        if state.get("schema") != 2:
            raise ValueError("Budget schema mismatch")
        # Never silently discard an old, independently-accounted ledger.
        old_ledger = output / "stage1_ledger.json"
        if not state_path.exists() and old_ledger.exists() and read_json(old_ledger).get("schema") != 2:
            raise ValueError("Legacy ledger requires explicit accounting migration; preserve old output")
        save = lambda: atomic_json(state_path, state)
        apply_usage_observations(state, already_used_hours=already, quota_remaining_hours=quota)
        session = BudgetSession(state, save, 30, 3, 8, 15,
                                quota_remaining_hours=quota_remaining_from_state(state))
        # Optional outer notebook deadline is shared by calibration and main commands.
        outer_deadline = os.environ.get("STAGE1_SESSION_DEADLINE")
        if outer_deadline is not None:
            remaining = max(0.0, finite(outer_deadline, "outer deadline") - time.time())
            session.session_hard_limit = min(session.session_hard_limit, remaining)
            session.session_soft_limit = max(0.0, session.session_hard_limit - 15 * 60)
        if session.hard_remaining() <= 60:
            atomic_json(output / "launch_status.json", {"status": "PAUSED_QUOTA", "scientific_stage2_complete": False})
            return 0
        session.begin_lease(session.hard_remaining())
        deadlines = session.deadlines_unix()
        env = os.environ.copy()
        env.update(STAGE1_SUPERVISED="1", STAGE1_SOFT_DEADLINE=str(deadlines["soft"]),
                   STAGE1_HARD_DEADLINE=str(deadlines["hard"]),
                   STAGE1_TOTAL_DEADLINE=str(time.time() + session.total_remaining()), MPLBACKEND="Agg")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        result = {"status": "FAILED"}
        try:
            result = supervise([sys.executable, "-m", "stage1_compat", *argv], env,
                               output / f"{args.command}-{stamp}.log", session)
            result["status"] = "PAUSED_QUOTA" if result["timed_out"] or result["returncode"] == 75 else ("COMPLETED_COMMAND" if result["returncode"] == 0 else "FAILED")
            return 0 if result["timed_out"] or result["returncode"] == 75 else result["returncode"]
        finally:
            # Collection is charged and bounded by the same parent deadline.
            if session.hard_remaining() > 5:
                collected = supervise([sys.executable, "-m", "stage1_compat", "collect", "--output-dir", str(output)],
                                      env, output / f"collect-{stamp}.log", session)
                result["collection"] = collected
            result["scientific_stage2_complete"] = False
            atomic_json(output / "launch_status.json", result)
            atomic_json(output / f"{args.command}-{stamp}-status.json", result)
            session.end_lease()
