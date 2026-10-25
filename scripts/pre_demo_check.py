from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

OUT = reg.abs_path(reg.PRE_DEMO_CHECK_JSON)


def add(checks: List[Dict[str, Any]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bundle_context() -> Any:
    """(update_status, manifest) NẾU gói web đang dùng dựng từ run train_v4 (protocol đang công bố); None = dùng cách kiểm cũ (Kaggle scratch)."""
    try:
        from scripts import update_status as us

        manifest = us._run_bundle_manifest()
    except Exception:
        return None
    return (us, manifest) if manifest else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-demo readiness gate for the thesis project.")
    ap.add_argument("--dataset-root", type=Path, default=None)
    args = ap.parse_args()

    checks: List[Dict[str, Any]] = []

    # Environment
    try:
        flwr_version = importlib.metadata.version("flwr")
        add(checks, "flower_installed", "pass", f"flwr={flwr_version}")
    except Exception as exc:
        add(checks, "flower_installed", "fail", f"Flower unavailable: {exc}")

    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        add(checks, "pytorch", "pass", f"torch={torch.__version__}, device={device}")
    except Exception as exc:
        add(checks, "pytorch", "fail", f"PyTorch unavailable: {exc}")

    # Required deployment scripts
    required = [
        "scripts/start_real_flower_server.ps1",
        "scripts/start_real_flower_client.ps1",
        "scripts/preflight_plantvillage.py",
        "scripts/preflight_client_partition.py",
        "scripts/run_real_multiseed_report.ps1",
        "scripts/validate_artifact_provenance.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    add(
        checks,
        "deployment_scripts",
        "pass" if not missing else "fail",
        "all required scripts exist" if not missing else f"missing: {missing}",
    )

    ctx = bundle_context()

    # Dataset gate
    if args.dataset_root is None and ctx:
        # Protocol đang công bố: dữ liệu là release v4 (không phải PlantVillage scratch) -> kiểm audit rò rỉ của release và ràng buộc với gói web
        manifest = ctx[1]
        audit = load_json(reg.abs_path(reg.RELEASE_LEAKAGE_AUDIT_JSON))
        pinned = (audit.get("release_binding") or {}).get("pinned_release_manifest_sha256")
        if audit.get("passed") is True and pinned and pinned == manifest.get("release_manifest_sha256"):
            add(checks, "release_data", "pass",
                f"release {audit.get('release_name')} {pinned[:12]}…: audit rò rỉ đạt và khớp gói {manifest.get('bundle_id')}")
        else:
            add(checks, "release_data", "warn",
                f"audit release chưa đạt hoặc không khớp gói {manifest.get('bundle_id')} (passed={audit.get('passed')}, pinned={str(pinned)[:12]})")
    elif args.dataset_root is None:
        add(checks, "plantvillage_real_data", "warn", "No --dataset-root supplied; scientific training cannot be validated.")
    else:
        preflight_out = reg.abs_path(reg.DATA_PREFLIGHT_DEMO_GATE_JSON)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "preflight_plantvillage.py"),
                str(args.dataset_root),
                "--full-path-check",
                "--output",
                str(preflight_out.relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        payload = load_json(preflight_out)
        add(
            checks,
            "plantvillage_real_data",
            "pass" if proc.returncode == 0 and payload.get("ok") else "fail",
            f"root={args.dataset_root}; ok={payload.get('ok', False)}",
        )

    # Canonical provenance
    provenance = load_json(reg.abs_path(reg.PROVENANCE_VALIDATION_JSON))
    bundle_prov = ctx[0]._bundle_provenance(ctx[1]) if ctx else None
    if bundle_prov is not None:
        # Kết quả hiện hành là gói v4b; slot canonical của pipeline Kaggle cũ chỉ còn metrics, ngoài phạm vi cổng này
        legacy = provenance.get("status") or "không có"
        add(checks, "canonical_provenance", "pass" if bundle_prov.status == ctx[0].DONE else "warn",
            f"{bundle_prov.detail} [pipeline Kaggle cũ, chỉ còn metrics: {legacy}]")
    elif provenance.get("status") == "verified":
        add(checks, "canonical_provenance", "pass", "canonical metrics/checkpoints verified")
    else:
        issue_codes = [i.get("code") for i in provenance.get("issues", [])]
        add(checks, "canonical_provenance", "warn", f"UNVERIFIED: {issue_codes}")

    # Network artifact scientific validity
    network_history = load_json(reg.abs_path(reg.NETWORK_FLOWER_HISTORY_JSON))
    audits = network_history.get("audit_rounds", []) or []
    clients = [c for r in audits for c in (r.get("clients", []) or [])]
    require_real = bool(network_history.get("require_real_clients", False))
    require_real_evaluation = bool(network_history.get("require_real_evaluation", False))
    final_test_metrics = network_history.get("final_test_metrics")
    all_real = bool(clients) and all(c.get("data_mode") == "real" for c in clients)
    has_real_test = (
        require_real_evaluation
        and isinstance(final_test_metrics, dict)
        and all(k in final_test_metrics for k in ("accuracy", "loss", "macro_f1"))
    )
    network_ckpt = reg.abs_path(reg.FEDERATED_NETWORK_GLOBAL_PT)
    if network_ckpt.exists() and require_real and all_real and has_real_test:
        add(
            checks,
            "network_global_artifact",
            "pass",
            "checkpoint exists, all audited clients used real data, and real held-out final test metrics are present",
        )
    elif network_ckpt.exists():
        add(
            checks,
            "network_global_artifact",
            "warn",
            "chưa có run mạng KHOA HỌC: checkpoint hiện có chỉ là smoke kỹ thuật "
            f"(require_real={require_real}, all_real={all_real}, real_test={has_real_test})",
        )
    else:
        add(checks, "network_global_artifact", "warn", "no network global checkpoint yet")

    # Multi-seed real summary
    multi = load_json(reg.abs_path(reg.REPORT_EXPERIMENTS_DIR) / "multiseed_real_summary.json")
    v4_multi = ctx[0]._v4_multiseed(ctx[1]) if ctx else None
    if v4_multi is not None and v4_multi.status == ctx[0].DONE:
        # Chỉ xác nhận đủ hạt giống × mô hình cùng release; ngưỡng chưa đạt nằm trong chuỗi chi tiết
        add(checks, "multiseed_real_summary", "pass", v4_multi.detail)
    elif multi.get("scientific_valid"):
        add(checks, "multiseed_real_summary", "pass", "3-seed real summary available and complete")
    else:
        add(checks, "multiseed_real_summary", "warn", "no complete verified multi-seed real summary yet")

    fail_count = sum(c["status"] == "fail" for c in checks)
    warn_count = sum(c["status"] == "warn" for c in checks)
    overall = "ready" if fail_count == 0 and warn_count == 0 else "technical_demo_ready" if fail_count == 0 else "not_ready"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": checks,
        "meaning": {
            "ready": "Ready for scientific/demo reporting.",
            "technical_demo_ready": "Code/demo can run, but one or more scientific evidence gates are incomplete.",
            "not_ready": "At least one required technical gate failed.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))  # stdout ASCII: console Windows không in được tiếng Việt
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
