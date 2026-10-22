"""Độ chính xác của mô hình trên các nhóm bệnh mà từng client CÓ hoặc KHÔNG CÓ ảnh học: Centralized, FedAvg và Local-only của client đó.

Trả lời câu hỏi "học chung có giúp client nhận ra bệnh mà dữ liệu riêng của nó không có không?" bằng số liệu, và thay câu hạn chế
"chưa đo độ công bằng giữa client" của giao diện. Chỉ ĐỌC: logits đã lưu của lần chấm cục bộ (data/reeval/logits.npz cho tập kiểm tra,
data/reeval/holdout_logits.npz cho tập giữ riêng) và báo cáo shard (artifacts/client_shard_class_counts.json); không chạy lại mô hình.
Báo cáo chỉ được ghi khi danh tính khớp gói đã kiểm hash (protocol, release, validation, SHA-256 checkpoint) và phân vùng khớp n_k của gói.

Chạy: python scripts/report_client_fairness.py   ->  artifacts/client_fairness_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

CLIENTS = tuple(f"client_{i:02d}" for i in range(5))


def group_accuracy(correct: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    return {"n": n, "correct": int(correct[mask].sum()) if n else 0}


def analyze(targets: np.ndarray, logits: dict[str, np.ndarray], client_models: dict[str, str], present: dict[str, np.ndarray]) -> dict[str, Any]:
    """Mỗi client: số ảnh và số ảnh đúng của Centralized, FedAvg, Local-only của client đó, tách theo lớp client có hoặc không có ảnh học."""
    correct = {m: logits[m].argmax(axis=1) == targets for m in logits}
    out: dict[str, Any] = {}
    for client, local_id in client_models.items():
        has = present[client][targets]               # ảnh thuộc lớp mà client có ảnh học
        entry: dict[str, Any] = {}
        for name, mask in (("classes_with_images", has), ("classes_without_images", ~has)):
            block = {"n": int(mask.sum())}
            for model in ("centralized", "fedavg", local_id):
                block[model if model != local_id else "local"] = group_accuracy(correct[model], mask)["correct"]
            entry[name] = block
        out[client] = entry
    return out


def build_report() -> dict[str, Any]:
    from scripts.report_statistics import _jsonl, _load_split_logits
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        raise ValueError(evidence.get("reason"))
    shards = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))
    if shards.get("protocol_id") != evidence["protocol_id"]:
        raise ValueError("báo cáo shard khác protocol của gói; chạy scripts/report_client_shards.py")
    for client, n_k in evidence["shard_counts"].items():
        if shards["clients"][client]["n_k"] != n_k:
            raise ValueError(f"n_k của {client} khác gói")
    classes = list(evidence["classes"])
    present = {c: np.array([int(shards["clients"][c]["per_class"].get(name, 0)) > 0 for name in classes]) for c in CLIENTS}

    manifests = reg.abs_path(evidence_release_dir(evidence)) / "manifests"
    rows = {"val": _jsonl(manifests / "val.jsonl"), "test": _jsonl(manifests / f"{reg.HOLDOUT_SPLIT}.jsonl")}
    cal_rows = _jsonl(manifests / "calibration.jsonl")
    client_models = {c: f"local_{i:02d}" for i, c in enumerate(CLIENTS)}
    report: dict[str, Any] = {
        "evidence": "tính trên logits đã lưu của lần chấm cục bộ; không chạy lại mô hình",
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "definition": "lớp client CÓ ảnh học = số ảnh của lớp đó trong shard của client lớn hơn 0 (báo cáo shard của gói)",
        "clients": {c: {"n_k": shards["clients"][c]["n_k"], "classes_with_images": int(present[c].sum()), "classes_total": len(classes)} for c in CLIENTS},
        "splits": {},
    }
    for split, split_rows in rows.items():
        logits, reason = _load_split_logits(evidence, split, split_rows, cal_rows, rows["val"])
        if logits is None:
            report["splits"][split] = {"available": False, "reason": reason}
            continue
        targets = np.array([r["class_id"] for r in split_rows], dtype=np.int64)
        report["splits"][split] = {"available": True, "count": len(split_rows), "by_client": analyze(targets, logits, client_models, present)}
    return report


def evidence_release_dir(evidence: dict[str, Any]) -> str:
    from src import bundle

    return bundle.release_dir(bundle.load_manifest())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = build_report()
    out = reg.abs_path(reg.CLIENT_FAIRNESS_JSON)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for split, block in report["splits"].items():
        if not block.get("available"):
            print(f"[fairness] {split}: bỏ qua — {block.get('reason')}")
            continue
        for client, data in block["by_client"].items():
            has, no = data["classes_with_images"], data["classes_without_images"]
            pct = lambda b, k: f"{100 * b[k] / b['n']:.1f}%" if b["n"] else "—"
            print(f"[fairness] {split}/{client}: lớp có ảnh (n={has['n']}) FedAvg {pct(has, 'fedavg')} · Local {pct(has, 'local')} | "
                  f"lớp thiếu (n={no['n']}) FedAvg {pct(no, 'fedavg')} · Local {pct(no, 'local')}")
    print(f"[fairness] wrote {reg.CLIENT_FAIRNESS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
