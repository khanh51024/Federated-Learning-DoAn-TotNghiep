from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

OUT = reg.abs_path(reg.FINAL_ACCEPTANCE_JSON)


def run_command(name: str, cmd: List[str], timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "name": name,
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def git_commit() -> Dict[str, Any]:
    """Commit + trạng thái dirty tại thời điểm chạy gate (để PROJECT_STATUS biết gate còn đúng không)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
        return {"commit": head or None, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command final technical/scientific acceptance gate.")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--skip-network", action="store_true", help="Skip real local gRPC process tests (kết quả sẽ KHÔNG được tính là 'full tests')")
    args = ap.parse_args()

    # Dấu vân tay mã nguồn lấy TRƯỚC khi chạy test: sửa code trong lúc chạy sẽ làm kết quả bị coi là cũ
    fingerprint = reg.code_fingerprint()
    started_at = datetime.now().isoformat(timespec="seconds")

    py = sys.executable
    checks: List[Dict[str, Any]] = []

    pre_demo_cmd = [py, "scripts/pre_demo_check.py"]
    if args.dataset_root is not None:
        pre_demo_cmd += ["--dataset-root", str(args.dataset_root)]
    checks.append(run_command("pre_demo_check", pre_demo_cmd, 180))

    core_tests = [
        "tests/test_fedavg.py",
        "tests/test_model.py",
        "tests/test_parameters.py",
        "tests/test_partition.py",
        "tests/test_integration.py",
        "tests/test_round_trip_feedback_loop.py",
        "tests/test_update_validator.py",
        "tests/test_data_scarcity.py",
        "tests/test_real_data_guard.py",
    ]
    checks.append(run_command("core_tests", [py, "-m", "pytest", *core_tests, "-q"], 300))

    scientific_tests = [
        "tests/test_artifact_scientific_status.py",
        "tests/test_web_provenance_guard.py",
        "tests/test_promote_verified_run.py",
        "tests/test_network_evaluation_guard.py",
        "tests/test_network_heldout_evaluation.py",
        "tests/test_network_reproducibility_guard.py",
        "tests/test_dataset_real_mode.py",
        "tests/test_client_partition_preflight.py",
    ]
    checks.append(run_command("scientific_guard_tests", [py, "-m", "pytest", *scientific_tests, "-q"], 360))

    network_tests = [
        "tests/test_flower_live_socket.py",
        "tests/test_flower_multi_round_network.py",
        "tests/test_audited_flower_network.py",
        "tests/test_five_client_network.py",
        "tests/test_faulty_client_network_detection.py",
        "tests/test_network_rtt_and_checkpoint.py",
        "tests/test_attack_matrix_network.py",
        "tests/test_web_federated_engine.py",
    ]
    if not args.skip_network:
        checks.append(run_command("network_tests", [py, "-m", "pytest", *network_tests, "-q"], 900))

    # Mọi file test còn lại (registry, web gate, status, bundle...) cũng phải chạy: không để test "mồ côi"
    listed = set(core_tests) | set(scientific_tests) | set(network_tests)
    all_tests = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_*.py"))
    other_tests = [t for t in all_tests if t not in listed]
    checks.append(run_command("other_tests", [py, "-m", "pytest", *other_tests, "-q"], 600))

    readiness = load_json(reg.abs_path(reg.PRE_DEMO_CHECK_JSON))
    technical_pass = all(c["status"] == "pass" for c in checks)
    scientific_pass = technical_pass and readiness.get("overall") == "ready"

    blockers = []
    if technical_pass and not scientific_pass:
        for item in readiness.get("checks", []):
            if item.get("status") != "pass":
                blockers.append({
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "detail": item.get("detail"),
                })

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "git": git_commit(),
        "code_fingerprint": fingerprint,
        "skip_network": bool(args.skip_network),
        "test_files": all_tests,
        "technical_acceptance": "pass" if technical_pass else "fail",
        "scientific_acceptance": "pass" if scientific_pass else "blocked",
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "readiness_overall": readiness.get("overall"),
        "checks": checks,
        "scientific_blockers": blockers,
        "interpretation": {
            "technical_pass": "Core, Flower/network, validator, RTT, checkpoint, web routing, registry gate and scientific guard tests passed.",
            "scientific_pass": "Technical acceptance passed AND real-data/provenance/multi-seed scientific gates are complete.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))  # ASCII: console Windows (cp1258) không in được tiếng Việt

    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
