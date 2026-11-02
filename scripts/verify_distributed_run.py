"""Audit a completed v4 FedAvg run without reading sealed-split metrics or logits.

Used after the GPU/Ray preflight and after each full run. A DONE marker alone is
insufficient: every round must contain five clients and an independent merge
check. Flower additionally requires exact transport checks in both directions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RELEASE_SHA = "68e574519a1a9a53f4e71060d6d6ef70122ff946d16e41981b5930d3da92d51b"
AGGREGATION_ATOL = 1e-7
FEDSGD_MAX_ABS = 1e-10
FEDSGD_RELATIVE_L2 = 1e-9


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def audit_run(run_dir: Path, *, preflight: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    for name in ("DONE", "config.json", "history.jsonl", "final.json", "checkpoint_best.pt", "logits_best.npz"):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}")
    if errors:
        return {"run": run_dir.name, "passed": False, "errors": errors}
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected = {"method": "fedavg", "budget": 3 if preflight else 40,
                "local_epochs": 1, "batch_size": 128, "lr": 0.02,
                "channels_last": True, "amp": "bf16", "limit": 3000 if preflight else 0,
                "release_manifest_sha256": RELEASE_SHA}
    if preflight:
        expected.update(runtime="flower", seed=42, scheme="label_alpha_0_1")
    for key, value in expected.items():
        if cfg.get(key) != value:
            errors.append(f"config.{key}: expected {value!r}, got {cfg.get(key)!r}")
    if not str(cfg.get("device", "")).startswith("cuda"):
        errors.append("run did not use CUDA")
    if not str(cfg.get("train_transform", "")).startswith("gpu_v2 ("):
        errors.append("run did not use gpu_v2 augmentation")
    if len(cfg.get("train_sizes", [])) != 5:
        errors.append("expected five training shards")
    # The trainer does not summarize the sealed split in new final.json files.
    # Deliberately access only initialization provenance here.
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    w0 = final.get("w0_sha256")
    if not isinstance(w0, str) or len(w0) != 64 or any(c not in "0123456789abcdef" for c in w0):
        errors.append("missing or invalid w0_sha256")
    by_step: dict[int, dict[str, Any]] = {}
    duplicates = 0
    for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            step = row["step"]
            duplicates += int(step in by_step)
            by_step[step] = row  # interrupted runs can replay the uncommitted last step
    if sorted(by_step) != list(range(expected["budget"])):
        errors.append("history does not cover every expected round")
    flower = cfg.get("runtime", "simulation") == "flower"
    comparisons, bytes_up, bytes_down, worst, seconds = 0, 0, 0, 0.0, 0.0
    for step, row in sorted(by_step.items()):
        train = row.get("train", {})
        clients = train.get("clients", [])
        if sorted(c.get("client", -1) for c in clients) != list(range(5)):
            errors.append(f"round {step}: expected each client 0..4 exactly once")
        check = train.get("aggregation_check", {})
        diff = check.get("max_abs_diff")
        if not _finite(diff) or diff < 0 or diff > AGGREGATION_ATOL or check.get("tensors_compared", 0) <= 0:
            errors.append(f"round {step}: invalid aggregation check")
        else:
            worst = max(worst, diff)
        elapsed = row.get("seconds")
        if not _finite(elapsed) or elapsed < 0:
            errors.append(f"round {step}: invalid elapsed time")
        else:
            seconds += elapsed
        for client in clients:
            for key in ("delta_l2", "l2_to_merged", "l2_to_merged_without_bn", "loss"):
                if not _finite(client.get(key)):
                    errors.append(f"round {step}, client {client.get('client')}: invalid {key}")
            if flower:
                if client.get("transport_down_exact") is not True or client.get("transport_up_exact") is not True:
                    errors.append(f"round {step}, client {client.get('client')}: transport mismatch or missing check")
                else:
                    comparisons += 2
            for key in ("bytes_up", "bytes_down"):
                value = client.get(key)
                if not _finite(value) or value <= 0:
                    errors.append(f"round {step}, client {client.get('client')}: invalid {key}")
                elif key == "bytes_up":
                    bytes_up += int(value)
                else:
                    bytes_down += int(value)
    return {"run": run_dir.name, "passed": not errors, "errors": errors,
            "runtime": cfg.get("runtime", "simulation"), "seed": cfg.get("seed"), "scheme": cfg.get("scheme"),
            "rounds": len(by_step), "replayed_history_rows": duplicates, "w0_sha256": w0,
            "transport_comparisons": comparisons, "aggregation_max_abs_diff": worst,
            "aggregation_atol": AGGREGATION_ATOL, "bytes_up": bytes_up, "bytes_down": bytes_down,
            "round_seconds_sum": seconds, "checkpoint_bytes": (run_dir / "checkpoint_best.pt").stat().st_size}


def audit_equivalence(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = [c for c in report["checks"] if c["check"] == "A_fedsgd" and c["dtype"] == "float64"]
    passed = len(checks) == 2 and {c["scheme"] for c in checks} == {"iid", "label_alpha_0_1"}
    for check in checks:
        passed = passed and all(_finite(check.get(k)) and 0 <= check[k] <= bound for k, bound in (
            ("max_abs", FEDSGD_MAX_ABS), ("l2_relative_to_centralized_travel", FEDSGD_RELATIVE_L2)))
    return {"passed": bool(passed), "max_abs_limit": FEDSGD_MAX_ABS, "relative_l2_limit": FEDSGD_RELATIVE_L2,
            "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--equivalence", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.run and not args.equivalence:
        parser.error("provide --run and/or --equivalence")
    report = {}
    if args.run:
        report["run"] = audit_run(args.run, preflight=args.preflight)
    if args.equivalence:
        report["equivalence"] = audit_equivalence(args.equivalence)
    report["passed"] = all(value["passed"] for value in report.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
