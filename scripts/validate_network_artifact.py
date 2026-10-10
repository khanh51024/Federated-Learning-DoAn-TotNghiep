from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
from src import registry as reg  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.mobilenetv3 import compute_model_hash


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a Flower network run as a scientific artifact.")
    ap.add_argument("--metrics", default=reg.NETWORK_FLOWER_HISTORY_JSON)
    ap.add_argument("--checkpoint", default=reg.FEDERATED_NETWORK_GLOBAL_PT)
    ap.add_argument("--expected-clients", type=int, default=5)
    ap.add_argument("--expected-rounds", type=int, default=5)
    ap.add_argument("--output", default=reg.NETWORK_ARTIFACT_VALIDATION_JSON)
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    checkpoint_path = Path(args.checkpoint)
    checks: Dict[str, Any] = {}
    details: Dict[str, Any] = {}

    checks["metrics_exists"] = metrics_path.is_file()
    checks["checkpoint_exists"] = checkpoint_path.is_file()
    if not checks["metrics_exists"] or not checks["checkpoint_exists"]:
        report = {"ok": False, "checks": checks, "details": details}
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        checks["require_real_clients"] = metrics.get("require_real_clients") is True
        checks["require_real_evaluation"] = metrics.get("require_real_evaluation") is True
        checks["expected_clients"] = int(metrics.get("num_clients_required", -1)) == args.expected_clients
        checks["expected_rounds"] = int(metrics.get("rounds", -1)) == args.expected_rounds
        checks["final_test_metrics_present"] = isinstance(metrics.get("final_test_metrics"), dict)
        checks["checkpoint_test_metrics_present"] = isinstance(ckpt.get("test_metrics"), dict)
        checks["seed_present"] = isinstance(metrics.get("seed"), int) and isinstance(ckpt.get("seed"), int)
        checks["seed_matches"] = metrics.get("seed") == ckpt.get("seed")
        checks["checkpoint_source"] = ckpt.get("source") == "flower_network_fedavg"
        model_state = ckpt.get("model_state")
        checks["checkpoint_model_state_present"] = isinstance(model_state, dict) and len(model_state) > 0
        recomputed_model_hash = compute_model_hash(model_state) if checks["checkpoint_model_state_present"] else None
        checks["checkpoint_model_hash_matches"] = (
            isinstance(ckpt.get("model_hash"), str)
            and recomputed_model_hash == ckpt.get("model_hash")
        )

        audit = metrics.get("audit_rounds", []) or []
        checks["audit_round_count"] = len(audit) == args.expected_rounds
        all_real = True
        all_client_counts = True
        no_rejected_synthetic = True
        for round_row in audit:
            clients = round_row.get("clients", []) or []
            all_client_counts = all_client_counts and len(clients) == args.expected_clients
            for client in clients:
                all_real = all_real and client.get("data_mode") == "real"
                no_rejected_synthetic = no_rejected_synthetic and client.get("status") != "rejected_synthetic_data"
        checks["all_clients_real"] = all_real
        checks["all_rounds_have_expected_clients"] = all_client_counts
        checks["no_synthetic_rejections"] = no_rejected_synthetic

        final_test = metrics.get("final_test_metrics") or {}
        details["final_test_metrics"] = final_test
        details["seed"] = metrics.get("seed")
        details["checkpoint_sha256"] = sha256_file(checkpoint_path)
        details["metrics_sha256"] = sha256_file(metrics_path)
        details["checkpoint_model_hash"] = ckpt.get("model_hash")
        details["recomputed_model_hash"] = recomputed_model_hash
        report = {"ok": all(bool(v) for v in checks.values()), "checks": checks, "details": details}
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))  # stdout ASCII: console Windows không in được tiếng Việt
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
