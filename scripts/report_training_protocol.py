"""Báo cáo cấu hình và lịch sử huấn luyện của gói mô hình đang dùng (artifacts/active_bundle.json); không huấn luyện gì.

Nguồn là chính gói: `import_manifest.json` (khối `training` của từng mô hình, lấy từ config.json/final.json của run lúc nhập),
`<mô hình>/summary.json` (lịch sử từng epoch/vòng), `validation_metrics.json`, checkpoint (đã băm) và manifest release. Không còn đọc
checkpoint để lấy lịch sử như thời gói v6/v7 (bản đó dùng checkpoint của nhánh train cũ; bản của nó nằm trong git và trong
artifacts/report_snapshots/<gói>/ khi lùi gói).

Điều CHƯA ghi lại được ở gói train bằng scripts/train_v4.py: độ lệch ‖w_k − w_toàn cục‖₂ của từng client (`delta_l2`) và byte thực đo trên mạng.
Báo cáo ghi `delta_l2_recorded: false`, để trường đó None; web ẩn biểu đồ tương ứng. Byte mỗi chiều là kích thước float32 của trạng thái mô hình
(mô phỏng trong tiến trình, không phải đo mạng).

Chạy: python scripts/report_training_protocol.py   ->  artifacts/training_protocol_report.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def state_payload_bytes(state: dict[str, Any]) -> int:
    """Byte của trạng thái mô hình khi gửi dạng float32 (mọi tensor dấu phẩy động; số nguyên như num_batches_tracked giữ int64)."""
    total = 0
    for tensor in state.values():
        total += int(tensor.numel()) * (4 if tensor.is_floating_point() else int(tensor.element_size()))
    return total


def config_block(training: dict[str, Any]) -> dict[str, Any]:
    """Các khóa web hiển thị (report-training.js), điền từ khối `training` của gói."""
    fedavg = training["method"] == "fedavg"
    return {
        "batch_size": training["batch_size"], "lr": training["lr"], "optimizer": training["optimizer"],
        "weight_decay": None, "amp": training["precision"], "init": "ImageNet-1K V1 + đầu 38 lớp theo hạt giống",
        "preprocessing_version": "canonical_v1", "scheduler_policy": training["lr_schedule"],
        "train_transform": training["train_transform"], "loss_weighting": training["loss_weighting"], "selection": training["selection"],
        "units": training["units"],
        **({"rounds": training["budget"], "local_epochs": training["local_epochs"], "client_optimizer": training["optimizer"],
            "optimizer_state": training["optimizer_state"]} if fedavg else {"epochs": training["budget"] * (training["local_epochs"] if training["method"] == "local" else 1)}),
    }


def build_report() -> dict:
    from scripts.evaluate_published_models import source_group
    from src import bundle, registry as reg
    from src.common.checkpoint_io import load_checkpoint
    from web.demo_evidence import STORE, load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        raise ValueError(evidence.get("reason"))
    manifest = bundle.load_manifest(STORE)
    trainings = manifest.get("training")
    if not trainings:
        raise ValueError("gói không có khối `training` (gói cũ nhập từ nhánh train: dùng bản đã lưu ở artifacts/report_snapshots)")
    report: dict[str, Any] = {key: evidence[key] for key in (
        "protocol_id", "release_manifest_sha256", "validation_manifest_sha256", "w0_fingerprint", "seed")}
    report.update(schema_version=2, bundle_id=manifest.get("bundle_id"), checkpoint_sha256={}, models=[], sources={},
                  delta_l2_recorded=False)

    def read_json(relative):
        path = STORE / relative
        raw = path.read_bytes()
        report["sources"][str(path.relative_to(reg.ROOT)).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    import io

    for model in evidence["models"]:
        mid = model["model_id"]
        path = STORE / model["checkpoint_relative_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != model["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed: {mid}")
        checkpoint = load_checkpoint(io.BytesIO(raw))
        if checkpoint.get("release_manifest_sha256") != evidence["release_manifest_sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {mid}: release_manifest_sha256")
        report["checkpoint_sha256"][mid] = digest
        local = mid.startswith("local_")
        directory = f"local/{mid.replace('local_', 'client_')}" if local else mid
        validation = read_json(directory + "/validation_metrics.json")
        summary = read_json(directory + "/summary.json")
        training = trainings[mid]
        selected = int(training["best_step"])
        if int(validation["checkpoint_round_or_epoch"]) != selected or validation["checkpoint_sha256"] != digest:
            raise ValueError(f"selected step or checkpoint mismatch: {mid}")
        if checkpoint.get("seed") != evidence["seed"] and mid in {"centralized", "fedavg"}:
            raise ValueError(f"seed mismatch: {mid}")
        history = summary["history"]
        item = {"model_id": mid, "selected_step": selected, "checkpoint_bytes": len(raw), "config": config_block(training),
                "validation_overall": validation["validation"]["overall"], "completed_steps": len(history),
                "total_runtime_seconds": round(sum(h["seconds"] for h in history), 1), "status": "completed", "history": []}
        if mid == "centralized":
            item["train_accuracy_by_source"] = None       # bộ train v4 không đo độ chính xác train theo nguồn
        payload = state_payload_bytes(checkpoint["model_state"])
        for row in history:
            entry = {k: row[k] for k in ("epoch", "train_loss", "val_loss", "val_accuracy", "plantdoc_macro_f1_supported",
                                         "plantvillage_macro_f1_supported", "lr", "selection_score") if k in row}
            entry["duration_sec"] = row["seconds"]
            if mid == "fedavg":
                entry["round"] = entry.pop("epoch")
                entry["clients"] = [{**c, "delta_l2": None, "delta_l2_learnable": None, "payload_down_bytes": payload, "payload_up_bytes": payload}
                                    for c in row["clients"]]
            if not local:
                item["history"].append(entry)
        if mid == "fedavg":
            for c in item["history"][0]["clients"]:
                if c["sample_count_n_k"] != evidence["shard_counts"][c["client_id"]]:
                    raise ValueError("n_k của FedAvg khác phân vùng của gói")
        report["models"].append(item)

    release = reg.abs_path(reg.RELEASE_DIR)
    raw = (release / "release_manifest.json").read_bytes()
    if evidence["release_manifest_sha256"] not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.rstrip(b"\r\n") + b"\r\n").hexdigest()}:
        raise ValueError("release identity mismatch")
    release_manifest = json.loads(raw)
    splits = []
    for name in ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine"):
        data = (release / "manifests" / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != release_manifest["manifest_hashes"][name]:
            raise ValueError(f"split hash mismatch: {name}")
        counts = Counter(source_group(json.loads(line)["source_dataset"]) for line in data.decode("utf-8").splitlines() if line.strip())
        splits.append({"name": name, "count": sum(counts.values()), "by_source": dict(counts),
                       "manifest_sha256": release_manifest["manifest_hashes"][name]})
    report["release"] = {"name": release_manifest["release_name"], "crop_policy": release_manifest.get("crop_policy", release_manifest.get("preprocessing_version")),
                         "splits": splits}
    report["transport_mode"] = "in_process_simulation"
    first = trainings["fedavg"]
    report["note"] = (f"Lịch sử lấy từ history.jsonl của các run train_v4 ({first['budget']} vòng FedAvg, chọn vòng {first['best_step']}). "
                      "Byte mỗi chiều là kích thước float32 của trạng thái mô hình, mô phỏng trong tiến trình, không phải lưu lượng mạng đo được; "
                      "độ lệch ‖Δ‖₂ của từng client chưa được ghi lại.")
    return report


if __name__ == "__main__":
    from src import registry as reg
    result = build_report()
    output = reg.abs_path(reg.TRAINING_PROTOCOL_REPORT_JSON)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(result['models'])} checkpoints")
