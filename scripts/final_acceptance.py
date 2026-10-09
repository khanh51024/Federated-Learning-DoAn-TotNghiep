from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_acceptance.json"


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


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command final technical/scientific acceptance gate.")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--skip-network", action="store_true", help="Skip real local gRPC process tests")
    args = ap.parse_args()

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
        "tests/test_flower_parity.py",
        "tests/test_real_flower_parity.py",
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

    if not args.skip_network:
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
        checks.append(run_command("network_tests", [py, "-m", "pytest", *network_tests, "-q"], 480))

    readiness = load_json(ROOT / "artifacts" / "pre_demo_check.json")
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
        "technical_acceptance": "pass" if technical_pass else "fail",
        "scientific_acceptance": "pass" if scientific_pass else "blocked",
        "dataset_root": str(args.dataset_root) if args.dataset_root is not None else None,
        "readiness_overall": readiness.get("overall"),
        "checks": checks,
        "scientific_blockers": blockers,
        "interpretation": {
            "technical_pass": "Core, Flower/network, validator, RTT, checkpoint, web routing, and scientific guard tests passed.",
            "scientific_pass": "Technical acceptance passed AND real-data/provenance/multi-seed scientific gates are complete.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if technical_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
