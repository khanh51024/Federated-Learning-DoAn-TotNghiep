"""Calibrate, run/resume and collect a fixed-protocol matrix within a 30h ceiling.

python -m fl_training.kaggle_budget --config configs/kaggle_30h.yaml --action calibrate
python -m fl_training.kaggle_budget --config configs/kaggle_30h.yaml --action run
No Kaggle API credentials are used. Preserve the whole output directory across sessions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid

import yaml
from .budget_state import (
    BudgetSession, apply_usage_observations, atomic_json, choose_rounds,
    quota_remaining_from_state,
)

ROOT = Path(__file__).resolve().parents[1]
MODES = ("centralized", "local-only", "fedavg")


def source_hash():
    from .config import compute_source_fingerprint
    return compute_source_fingerprint(ROOT)


def write_session_provenance(output, action, config_path, dataset_root, partition_index):
    packages = {}
    for name in ("torch", "torchvision", "flwr", "ray", "numpy", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {"available": False, "device": None}
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        pass
    record = {
        "schema_version": 1, "action": action, "recorded_at_unix": time.time(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "cuda": cuda, "source_fingerprint": source_hash(),
        "matrix_config": str(Path(config_path).resolve()),
        "matrix_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "dataset_root": str(dataset_root), "partition_index": str(partition_index),
        "quota_source": "user input only; Kaggle account quota is not auto-detected",
        "scientific_stage2_complete": False,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    destination = output / "provenance" / f"{stamp}__{os.getpid()}__{action}.json"
    atomic_json(destination, record)
    return destination


def stop_tree(process):
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        # Each job gets its own process group; include its Flower/Ray descendants.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(1)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def acquire_lock(path, recover=False):
    if path.exists() and recover:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("host") == socket.gethostname():
            pid = int(old["pid"])
            if os.name == "nt":
                probe = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True,
                )
                alive = probe.returncode == 0 and f'"{pid}"' in probe.stdout
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
                else:
                    alive = True
            if alive:
                raise RuntimeError("Budget runner PID is still active; cannot recover its lock")
        old["recovered_at_unix"] = time.time()
        atomic_json(path.with_suffix(".recovered.json"), old)
        path.unlink()
    token = uuid.uuid4().hex
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Another runner or stale lock exists; use --recover only after the old session has stopped") from exc
    with os.fdopen(fd, "w") as file:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "token": token,
                   "created_at_unix": time.time()}, file)
        file.flush()
        os.fsync(file.fileno())
    return token


def release_lock(path, token):
    if not path.exists():
        return
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Runner lock became unreadable; refusing to delete it") from exc
    if current.get("token") != token:
        raise RuntimeError("Runner lock ownership changed; refusing to delete another runner's lock")
    path.unlink()


def matching_attempts(directory, cfg):
    found = []
    if not directory.exists():
        return found
    for run in directory.iterdir():
        if not run.is_dir() or not (run / "resolved_config.yaml").exists():
            continue
        try:
            old = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(old, dict):
            continue
        if (old.get("semantic_config_hash") == cfg.semantic_config_hash
                and old.get("data", {}).get("protocol_fingerprint") == cfg.data.protocol_fingerprint
                and old.get("federation", {}).get("max_rounds") == cfg.federation.max_rounds
                and old.get("mode") == cfg.mode):
            found.append(run)
    return sorted(found, key=lambda p: (p / "resolved_config.yaml").stat().st_mtime_ns, reverse=True)


def completed_run(attempts, calibration=False):
    for run in attempts:
        summary_path = run / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accepted = {"calibration_completed"} if calibration else {"completed", "completed_with_warnings"}
        if summary.get("status") not in accepted:
            continue
        if calibration and summary.get("artifact_scope") != "calibration_no_test":
            continue
        is_local = summary.get("mode") == "local-only"
        if not calibration and summary.get("artifact_scope") not in {None, "main_evaluated"}:
            continue
        if not (run / "resolved_config.yaml").exists():
            continue
        if is_local:
            expected = len(summary.get("clients", []))
            required_names = ("last.pt",) if calibration else ("last.pt", "best.pt", "test_metrics.json")
            if expected == 0 or any(
                not all((run / f"client_{i:02d}" / name).exists() for name in required_names)
                for i in range(expected)
            ):
                continue
        elif (not (run / "last.pt").exists()
              or (not calibration and not (run / "test_metrics.json").exists())):
            continue
        if calibration or (run / "best.pt").exists() or summary.get("mode") == "local-only":
            return run
    return None


def sample_resources(process):
    """Best-effort measured host/GPU memory for calibration provenance."""
    result = {"rss_bytes": None, "gpu_memory_mb": None}
    if os.name != "nt":
        try:
            for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    result["rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            pass
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        values = [float(line.strip()) for line in query.stdout.splitlines() if line.strip()]
        if query.returncode == 0 and values:
            result["gpu_memory_mb"] = sum(values)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return result


def measure_run(run_dir, mode, wall_seconds, rounds, resources=None):
    import pandas as pd
    histories = sorted(run_dir.glob("client_*/history.csv")) if mode == "local-only" else [run_dir / "history.csv"]
    if not histories or any(not path.exists() for path in histories):
        raise RuntimeError("No calibration history")
    frames = [pd.read_csv(path) for path in histories]
    if any(frame.empty or "duration_seconds" not in frame for frame in frames):
        raise RuntimeError("Incomplete calibration history")
    training = sum(float(frame["duration_seconds"].sum()) for frame in frames)
    rate = training / rounds
    # No test data is read during calibration. Reserve two extra round-equivalents
    # for final test/plots, in addition to measured setup and round-zero validation.
    measured = resources or {}
    return {
        "seconds_per_round": rate,
        "fixed_seconds": max(0, wall_seconds-training) + 2*rate,
        "measured_wall_seconds": wall_seconds,
        "calibration_rounds": rounds,
        "client_models_measured": len(histories) if mode == "local-only" else 1,
        "training_validation_seconds": training,
        "startup_checkpoint_postprocess_seconds": max(0, wall_seconds-training),
        "max_process_rss_bytes": measured.get("max_rss_bytes"),
        "max_gpu_memory_mb": measured.get("max_gpu_memory_mb"),
        "gpu_memory_scope": "nvidia-smi compute processes; null when unavailable",
        "final_evaluation_report_reserve_round_equivalents": 2,
        "test_used_for_selection": False,
    }


def run_job(job, session, state, save, calibration=False):
    from .sweep import _result_row
    attempts = matching_attempts(job["directory"], job["cfg"])
    done = completed_run(attempts, calibration=calibration)
    if done is not None:
        if not calibration:
            _result_row(done, job["condition"], job["seed"], job["mode"],
                        job["cfg"].data.partition_dir)
        return done
    session.tick()
    remaining = session.remaining()
    if remaining <= 120:
        return None
    deadlines = session.deadlines_unix()
    command = [sys.executable, "-m", "fl_training.cli"]
    if job["mode"] == "fedavg":
        command += ["train", "--config", str(job["config"]), "--no-progress"]
        resume = next((p / "last.pt" for p in attempts if (p / "last.pt").exists()), None)
        if resume:
            command += ["--resume", str(resume)]
    else:
        command += ["baseline", "--mode", job["mode"], "--config", str(job["config"])]
        if attempts:
            command += ["--resume-dir", str(attempts[0])]
    entry = state["jobs"].setdefault(job["id"], {"attempts": 0, "wall_seconds": 0.0})
    entry.update(status="running", attempts=entry["attempts"]+1)
    environment = os.environ.copy()
    environment["FL_TRAINING_SOFT_DEADLINE_UNIX"] = str(deadlines["soft"])
    environment["FL_TRAINING_HARD_DEADLINE_UNIX"] = str(deadlines["hard"])
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    log_path = job["directory"] / f"attempt-{entry['attempts']:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    session.begin_lease(session.hard_remaining())
    started = time.monotonic()
    next_resource_sample = started
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
            while process.poll() is None:
                session.tick()
                now = time.monotonic()
                if now >= next_resource_sample:
                    sample = sample_resources(process)
                    if sample["rss_bytes"] is not None:
                        entry["max_rss_bytes"] = max(entry.get("max_rss_bytes", 0), sample["rss_bytes"])
                    if sample["gpu_memory_mb"] is not None:
                        entry["max_gpu_memory_mb"] = max(entry.get("max_gpu_memory_mb", 0), sample["gpu_memory_mb"])
                    next_resource_sample = now + 15
                if session.hard_remaining() <= 0:
                    stop_tree(process)
                    break
                time.sleep(min(2, max(0.05, session.hard_remaining())))
        if process.returncode != 0:
            entry["status"] = "paused" if time.time() >= deadlines["soft"] else "failed"
            if entry["status"] == "failed":
                raise RuntimeError(f"Job {job['id']} failed; inspect {log_path}")
            return None
        done = completed_run(matching_attempts(job["directory"], job["cfg"]), calibration=calibration)
        if done is None:
            raise RuntimeError(f"No completed artifacts after {job['id']}")
        if not calibration:
            _result_row(done, job["condition"], job["seed"], job["mode"],
                        job["cfg"].data.partition_dir)
        entry["status"] = "completed"
        return done
    except BaseException:
        entry["status"] = "interrupted" if process and process.poll() is None else "failed"
        if process:
            stop_tree(process)
        raise
    finally:
        entry["wall_seconds"] += time.monotonic() - started
        session.end_lease()
        save()


def collect_jobs(jobs, output):
    from .sweep import _result_row, _write_comparison
    rows, coverage = [], []
    for job in jobs:
        raw_candidates = ([path for path in job["directory"].iterdir() if path.is_dir()]
                          if job["directory"].exists() else [])
        attempts = matching_attempts(job["directory"], job["cfg"])
        try:
            run = completed_run(attempts)
            if run is None:
                failed = []
                for attempt in attempts:
                    summary_path = attempt / "summary.json"
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        if summary.get("status") in {"failed", "postprocessing_failed"}:
                            failed.append({"run_dir": str(attempt), "error": summary.get("error")})
                if failed:
                    coverage.append({"job": job["id"], "status": "failed", "attempts": failed})
                elif raw_candidates and not attempts:
                    coverage.append({"job": job["id"], "status": "invalid",
                                     "reason": "Run directories exist but resolved config/identity is invalid",
                                     "candidate_runs": [str(path) for path in raw_candidates]})
                else:
                    coverage.append({"job": job["id"], "status": "missing",
                                     "reason": "No main job with required evaluation artifacts"})
                continue
            rows.append(_result_row(run, job["condition"], job["seed"], job["mode"],
                                    job["cfg"].data.partition_dir))
            coverage.append({"job": job["id"], "status": "completed", "run_dir": str(run)})
        except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            coverage.append({"job": job["id"], "status": "invalid", "reason": str(exc),
                             "candidate_runs": [str(path) for path in attempts]})
    output.mkdir(parents=True, exist_ok=True)
    from .research_reporting import write_research_summary
    # Invalidate research outputs even if the comparison consistency guard raises.
    try:
        _write_comparison(rows, output)
    except Exception:
        write_research_summary([], jobs, output)
        raise
    write_research_summary(rows, jobs, output)
    counts = {name: sum(item["status"] == name for item in coverage)
              for name in ("completed", "missing", "failed", "invalid")}
    status = {"expected_jobs": len(jobs), "completed_jobs": len(rows), **counts,
              "matrix_complete": bool(jobs) and len(rows) == len(jobs),
              "scientific_stage2_complete": False,
              "scope": "configured budget-limited matrix; inspect research_status.json for multi-seed coverage",
              "jobs": coverage}
    atomic_json(output / "collection_status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kaggle_30h.yaml")
    parser.add_argument("--action", choices=["plan", "calibrate", "run", "collect"], default="plan")
    parser.add_argument("--dataset-root")
    parser.add_argument("--partition-index")
    parser.add_argument("--output-root")
    parser.add_argument("--already-used-hours", type=float,
                        help="Authoritative lower bound of GPU hours already used; never subtracts ledger time")
    parser.add_argument("--quota-remaining-hours", type=float,
                        help="Current Kaggle account quota remaining; supplied by the user, not auto-detected")
    parser.add_argument("--charge-hours", type=float, default=0,
                        help="Additional GPU session/setup time used outside this runner")
    parser.add_argument("--charge-id", help="Stable unique id making --charge-hours idempotent across reruns")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from .config import load_training_config
    from .sweep import _condition_name
    base = yaml.safe_load((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    for section, values in spec.get("overrides", {}).items():
        base[section].update(values)
    if args.dataset_root:
        base["data"]["dataset_root"] = str(Path(args.dataset_root).resolve())
    if args.partition_index:
        base["data"]["partition_index"] = str(Path(args.partition_index).resolve())
    output = Path(args.output_root or spec["output_root"]).resolve()
    seeds = [int(value) for value in spec.get("seeds", [42])]
    conditions = spec["conditions"]
    if spec.get("research_protocol") == "stage2_full_v1":
        from .research_reporting import validate_research_spec
        validate_research_spec(spec)
    if len(set(seeds)) != len(seeds) or not seeds or not conditions:
        raise ValueError("Seeds/conditions must be nonempty and unique")
    if len({_condition_name(c) for c in conditions}) != len(conditions):
        raise ValueError("Duplicate conditions")
    budget = spec["budget"]
    for name, value in (("charge-hours", args.charge_hours),
                        ("already-used-hours", args.already_used_hours),
                        ("quota-remaining-hours", args.quota_remaining_hours)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 30):
            raise ValueError(f"{name} must be finite and between 0 and 30")
    if args.charge_hours and not args.charge_id:
        raise ValueError("--charge-id is required when --charge-hours is nonzero")
    required_budget = {
        "total_hours", "reserve_hours", "session_hours", "session_reserve_minutes",
        "calibration_rounds", "min_rounds", "max_rounds", "safety_factor",
    }
    if required_budget - set(budget):
        raise ValueError(f"Missing budget fields: {sorted(required_budget - set(budget))}")
    if budget["total_hours"] != 30 or budget["reserve_hours"] != 3:
        raise ValueError("This protocol requires total_hours=30 and reserve_hours=3")
    if budget["session_hours"] > 8 or budget["session_reserve_minutes"] < 15:
        raise ValueError("Session limit must be <=8h and stop-new-work reserve must be >=15 minutes")
    if not 10 <= int(budget["min_rounds"]) <= int(budget["max_rounds"]) <= 20:
        raise ValueError("Fixed common round range must be 10..20")
    if float(budget["safety_factor"]) != 1.5:
        raise ValueError("This protocol requires safety_factor=1.5")
    output.mkdir(parents=True, exist_ok=True)
    write_session_provenance(
        output, args.action, args.config, base["data"]["dataset_root"],
        base["data"]["partition_index"],
    )

    def make_job(condition, seed, mode, rounds, calibration=False):
        name = _condition_name(condition)
        scope = "calibration" if calibration else "main"
        ident = f"{scope}/{name}__seed-{seed}/{mode}"
        mode_folder = mode.replace("-", "_")
        raw = copy.deepcopy(base)
        raw["data"].update(alpha=None, quantity_alpha=None, feature_skew="none")
        raw["data"].update(condition)
        raw["training"]["seed"] = seed
        raw["federation"]["max_rounds"] = rounds
        raw["early_stopping"]["enabled"] = False
        directory = output / scope / f"{name}__seed-{seed}" / mode_folder
        raw["output"].update(root=str(directory), progress=False,
                              evaluate_test_after_train=not calibration)
        config = output / "configs" / (ident.replace("/", "__") + ".yaml")
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        cfg = load_training_config(config, base_dir=ROOT, mode="train" if mode == "fedavg" else mode)
        return {"id": ident, "config": config, "directory": directory, "cfg": cfg,
                "mode": mode, "condition": condition, "seed": seed}

    # Validate every partition before spending time on calibration. Dry-run does not
    # change an existing frozen budget plan or comparison.
    probe_jobs = [make_job(c, seed, m, budget["max_rounds"]) for c in conditions for seed in seeds for m in MODES]
    if args.action == "plan":
        atomic_json(output / "requested_plan.json", {"jobs": [j["id"] for j in probe_jobs],
            "total_hours": budget["total_hours"], "reserve_hours": budget["reserve_hours"],
            "rounds": "chosen after timing-only calibration", "max_rounds": budget["max_rounds"]})
        print(f"Validated {len(probe_jobs)} jobs. Calibration will freeze a common round budget.")
        return 0
    state_path = output / "budget_state.json"
    identity_raw = [{"id": j["id"], "semantic": j["cfg"].semantic_config_hash,
                     "rounds": budget["max_rounds"]} for j in probe_jobs]
    protocol_budget = {key: budget[key] for key in sorted(required_budget)}
    identity = hashlib.sha256(json.dumps({"jobs": identity_raw, "budget": protocol_budget,
                              "source": source_hash()}, sort_keys=True).encode()).hexdigest()
    if args.action == "collect":
        if not state_path.exists():
            raise RuntimeError("Missing budget_state.json; restore the complete output directory")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("Collect config/source/protocol differs from the frozen budget state")
        rounds = state.get("rounds")
        if rounds is None:
            raise RuntimeError("Calibration not complete; no main round budget has been frozen")
        jobs = [make_job(c, seed, m, rounds) for c in conditions for seed in seeds for m in MODES]
        print(json.dumps(collect_jobs(jobs, output / "comparison"), indent=2))
        return 0
    lock = output / "runner.lock"
    lock_token = acquire_lock(lock, args.recover)
    session = None
    jobs = []
    state = {}
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["identity"] != identity:
                raise ValueError("Frozen matrix/code/config changed. Restore the original version or use a new output root and account for already-used quota")
        else:
            initial_used = args.already_used_hours
            if initial_used is None:
                initial_used = float(budget.get("already_used_hours", 0))
            if not math.isfinite(initial_used) or not 0 <= initial_used <= 30:
                raise ValueError("already_used_hours must be finite and between 0 and 30")
            state = {"schema_version": 2, "identity": identity, "used_seconds": initial_used*3600,
                     "rounds": None, "jobs": {}, "calibration": {}, "external_charges": {}}
        apply_usage_observations(
            state, already_used_hours=args.already_used_hours,
            quota_remaining_hours=args.quota_remaining_hours,
            charge_hours=args.charge_hours, charge_id=args.charge_id,
        )
        quota_remaining = quota_remaining_from_state(state)
        save = lambda: atomic_json(state_path, state)
        session = BudgetSession(state, save, budget["total_hours"], budget["reserve_hours"],
                                budget["session_hours"], budget["session_reserve_minutes"],
                                quota_remaining_hours=quota_remaining)
        if state["rounds"] is None:
            for mode in MODES:
                if mode in state["calibration"]:
                    continue
                job = make_job(conditions[0], seeds[0], mode, budget["calibration_rounds"], True)
                print("Calibrating", mode, flush=True)
                run = run_job(job, session, state, save, calibration=True)
                if run is None:
                    return 75
                wall = state["jobs"].get(job["id"], {}).get("wall_seconds", 0)
                state["calibration"][mode] = measure_run(
                    run, mode, wall, budget["calibration_rounds"], state["jobs"].get(job["id"]),
                )
                save()
            session.tick()
            rounds, estimate = choose_rounds(state["calibration"], [j["mode"] for j in probe_jobs],
                session.total_remaining(), budget["min_rounds"], budget["max_rounds"], budget["safety_factor"])
            state.update(rounds=rounds, estimated_main_seconds=estimate)
            save()
            print(f"Frozen common budget: {rounds} rounds; conservative main estimate {estimate/3600:.2f} hours", flush=True)
        jobs = [make_job(c, seed, m, state["rounds"]) for c in conditions for seed in seeds for m in MODES]
        atomic_json(output / "frozen_plan.json", {"rounds": state["rounds"], "estimated_hours": state["estimated_main_seconds"]/3600,
                    "jobs": [j["id"] for j in jobs], "scientific_stage2_complete": False})
        if args.action == "calibrate":
            state["status"] = "protocol_frozen_main_not_started"
            save()
            return 0
        for job in jobs:
            print("Running/resuming", job["id"], flush=True)
            if run_job(job, session, state, save) is None:
                state["status"] = "paused_budget_or_session"
                save()
                return 75
            collect_jobs(jobs, output / "comparison")
        state["status"] = "matrix_completed"
        save()
        return 0
    finally:
        try:
            if jobs:
                result = collect_jobs(jobs, output / "comparison")
                print(f"Collected {result['completed_jobs']}/{result['expected_jobs']} jobs", flush=True)
            if session:
                session.tick()
        finally:
            release_lock(lock, lock_token)


if __name__ == "__main__":
    raise SystemExit(main())
