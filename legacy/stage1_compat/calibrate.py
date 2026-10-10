"""Calibration uses the same five-client loop as main, never the test evaluator."""
import os
import time
from pathlib import Path
from stage1_compat.integrity import atomic_json, read_json, digest, finite
from stage1_compat.identity import source_fingerprint


def calibration_forecast(timings, startup_seconds):
    if len(timings) != 2:
        raise ValueError("Both alpha conditions must be measured")
    rate = max(finite(t["elapsed_seconds"], "calibration time", minimum=0.001) / t["rounds"] for t in timings)
    # Entire observed job duration includes initialization/checkpoint; conservatively amortized per round.
    safe = 1.5 * rate
    reserve = 1.5 * (finite(startup_seconds, "startup") + 4 * rate)
    return safe, reserve, 20 * safe + reserve


def validate_calibration(output, dataset_hash, device):
    report = read_json(Path(output) / "calibration" / "calibration_report.json")
    body = {k: v for k, v in report.items() if k != "budget_sha256"}
    if (digest(body) != report.get("budget_sha256") or report["source_sha256"] != source_fingerprint()
            or report["dataset_sha256"] != dataset_hash or report["device"] != device
            or report["rounds_per_job"] != 10 or report["safety_margin"] != 1.5
            or report["test_evaluated"] is not False):
        raise ValueError("Calibration source/data/device/protocol mismatch")
    finite(report["safe_round_seconds"], "calibrated rate", minimum=0.001)
    return report


def run_calibration(data_dir=None, output_dir=None, user_quota_hours=0.0, calibration_rounds=1, device_str=None):
    if not os.environ.get("STAGE1_SUPERVISED"):
        raise RuntimeError("Use python -m stage1_compat calibrate")
    if type(calibration_rounds) is not int or not 1 <= calibration_rounds <= 2:
        raise ValueError("Calibration rounds must be 1 or 2")
    import torch
    from stage1_compat.config import get_default_profile_config
    from stage1_compat.preflight import run_preflight
    from stage1_compat.data import load_stage1_datasets, load_stage1_partition
    from stage1_compat.runner import run_single_fedavg_job
    from stage1_compat.budget import BudgetLedger
    started = time.monotonic()
    cfg = get_default_profile_config(data_dir, output_dir, user_quota_hours)
    output = Path(cfg.output_dir)
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    checked = run_preflight(cfg.data_dir, output)
    if checked.status != "PASSED":
        raise ValueError("Preflight failed")
    frozen = read_json(checked.frozen_manifest_path)
    target = output / "calibration" / "calibration_report.json"
    if target.exists():
        return validate_calibration(output, frozen["identity_hash"], device)
    train, val, test, names = load_stage1_datasets(cfg.data_dir)
    train.identity_context = {"dataset_sha256": frozen["identity_hash"], "scope": "calibration_no_test"}
    ledger = BudgetLedger(output / "calibration", user_quota_hours=user_quota_hours)
    timings = []
    startup = time.monotonic() - started
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with ledger.get_lock():
        ledger._load_or_init_ledger()
        for job in cfg.jobs:
            job.job_id = "calibration_" + job.job_id
            job.rounds = calibration_rounds
            result = run_single_fedavg_job(job, train, val, test, names,
                load_stage1_partition(job.alpha, train.targets), output / "calibration", ledger,
                torch.device(device), calibration=True)
            if result["status"] != "CALIBRATED_NO_TEST":
                return result
            # Resume already-measured rounds using their persisted timing, never zero time.
            result["elapsed_seconds"] = max(result["elapsed_seconds"], sum(result["round_seconds"]))
            timings.append(result)
    if len({t["identity"]["context"]["initialization_sha256"] for t in timings}) != 1:
        raise ValueError("Calibration jobs have different initialization")
    safe, reserve, forecast = calibration_forecast(timings, startup)
    remaining = max(0.0, float(os.environ["STAGE1_TOTAL_DEADLINE"]) - time.time())
    report = {"status": "SUCCESS" if forecast <= remaining else "QUOTA_WARNING",
        "device": device, "source_sha256": source_fingerprint(), "dataset_sha256": frozen["identity_hash"],
        "rounds_per_job": 10, "safety_margin": 1.5, "safe_round_seconds": safe,
        "final_reserve_seconds": reserve, "forecast_total_seconds": forecast,
        "remaining_budget_seconds": remaining, "quota_sufficient": forecast <= remaining,
        "initialization_sha256": timings[0]["identity"]["context"]["initialization_sha256"],
        "test_evaluated": False, "measurements": timings, "startup_seconds": startup,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "memory_note": "Parent launcher records process peak RAM; no GPU speed guarantee before measurement"}
    report["budget_sha256"] = digest(report)
    atomic_json(target, report)
    return report
