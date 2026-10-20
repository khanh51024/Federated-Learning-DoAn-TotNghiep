"""Read-only projection of verified training metadata; no checkpoint loading in HTTP requests."""
from __future__ import annotations

import math
from typing import Any

from src import registry as reg


def _rounded(value: Any, digits: int = 6) -> Any:  # digits = significant digits, so 1.5e-08 learning rates survive
    """Shorten floats after validation so the summary payload stays small (the page shows 2-4 decimals)."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    return value


def training_report(evidence: dict[str, Any], load_json) -> dict[str, Any]:
    from web.demo_evidence import _sha256_cached

    report = load_json(reg.TRAINING_PROTOCOL_REPORT_JSON)
    if report is None:
        return {"available": False, "reason": "Chưa có báo cáo cấu hình; chạy scripts/report_training_protocol.py trên máy có bundle và manifest release."}
    for key in ("protocol_id", "release_manifest_sha256", "validation_manifest_sha256", "w0_fingerprint", "seed"):
        if report.get(key) != evidence.get(key):
            raise ValueError(f"training report: {key} khác bundle")
    hashes = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    if report.get("checkpoint_sha256") != hashes:
        raise ValueError("training report: SHA-256 checkpoint khác bundle")
    # Verify small bundle sources as well as model identity. No dataset images needed at request time.
    sources = report.get("sources") or {}
    if not sources:
        raise ValueError("training report: thiếu nguồn")
    for relative, expected in sources.items():
        path = (reg.ROOT / relative).resolve()
        if not path.is_relative_to(reg.abs_path(reg.DEMO_BUNDLE_DIR).resolve()) or not path.is_file() or _sha256_cached(path) != expected:
            raise ValueError("training report: nguồn thay đổi hoặc thiếu")
    models = report["models"]
    if [m["model_id"] for m in models] != list(hashes):
        raise ValueError("training report: danh sách mô hình khác bundle")
    for model in models:
        overall = model["validation_overall"]
        if overall["count"] != sum(evidence["validation_counts"].values()):
            raise ValueError("training report: số ảnh validation khác bundle")
        hist = model["history"]
        key = "epoch" if model["model_id"] == "centralized" else "round"
        if hist and [h[key] for h in hist] != list(range(1, len(hist) + 1)):
            raise ValueError("training report: lịch sử bị đứt vòng")
    fed = next(m for m in models if m["model_id"] == "fedavg")
    if report["transport_mode"] != "in_process_simulation" or len(fed["history"]) != fed["selected_step"]:
        raise ValueError("training report: runtime/lịch sử không khớp")
    for row in fed["history"]:
        clients = row["clients"]
        if len(clients) != len(evidence["shard_counts"]) or {c["client_id"] for c in clients} != set(evidence["shard_counts"]):
            raise ValueError("training report: thiếu client")
        for client in clients:
            n = evidence["shard_counts"][client["client_id"]]
            if client["sample_count_n_k"] != n or abs(client["weight_p_k"] - n / sum(evidence["shard_counts"].values())) > 1e-9:
                raise ValueError("training report: sai trọng số client")
            learnable, total = client.get("delta_l2_learnable"), client.get("delta_l2")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (learnable, total)) or not 0 <= learnable <= total + 1e-3:
                raise ValueError("training report: delta_l2_learnable thiếu hoặc không hợp lệ")
    splits = {s["name"]: s for s in report["release"]["splits"]}
    if splits["val"]["manifest_sha256"] != evidence["validation_manifest_sha256"] or splits["val"]["by_source"] != evidence["validation_counts"]:
        raise ValueError("training report: validation manifest khác bundle")
    if splits["legacy_diagnostic_test"]["manifest_sha256"] != reg.HOLDOUT_MANIFEST_SHA256 or splits["legacy_diagnostic_test"]["by_source"] != reg.HOLDOUT_COUNTS:
        raise ValueError("training report: test manifest khác release")
    if splits["train"]["count"] != sum(evidence["shard_counts"].values()) or any(s["count"] != sum(s["by_source"].values()) for s in splits.values()):
        raise ValueError("training report: sai tổng ảnh")
    # Local checkpoints carry no history; Centralized (83 epochs) and FedAvg (27 rounds) are small enough to serve whole.
    # The page draws Centralized only from these fields; per-epoch duration stays in the committed report.
    projected_models = _rounded([{**m, "history": [{k: v for k, v in row.items() if k != "duration_sec"} for row in m["history"]]}
                                 if m["model_id"] == "centralized" else m for m in models])
    return {"available": True, "models": projected_models,
            **{k: report[k] for k in ("transport_mode", "release", "checkpoint_sha256", "seed")}}
