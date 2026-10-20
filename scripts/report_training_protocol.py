"""Extract report data from hash-verified published checkpoints; never train a model."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_report() -> dict:
    from src import registry as reg
    from src.common.checkpoint_io import load_checkpoint
    from web.demo_evidence import load_evidence, STORE

    evidence = load_evidence()
    if not evidence.get("available"):
        raise ValueError(evidence.get("reason"))
    report = {key: evidence[key] for key in (
        "protocol_id", "release_manifest_sha256", "validation_manifest_sha256", "w0_fingerprint", "seed")}
    report.update(schema_version=1, checkpoint_sha256={}, models=[], sources={})

    def read_json(relative):
        path = STORE / relative
        raw = path.read_bytes()
        report["sources"][str(path.relative_to(reg.ROOT)).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    central_summary = read_json("centralized/summary.json")
    local_summary = read_json("local/combined_summary.json")
    for model in evidence["models"]:
        mid = model["model_id"]
        path = STORE / model["checkpoint_relative_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != model["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed: {mid}")
        checkpoint = load_checkpoint(io.BytesIO(raw))
        for key in ("release_manifest_sha256", "w0_fingerprint"):
            if checkpoint.get(key) != evidence[key]:
                raise ValueError(f"checkpoint identity mismatch: {mid}: {key}")
        report["checkpoint_sha256"][mid] = digest
        local = mid.startswith("local_")
        source = "local/" + mid.replace("local_", "client_") if local else mid
        validation = read_json(source + "/validation_metrics.json")
        selected = int(checkpoint.get("epoch", checkpoint.get("round")))
        if int(validation["checkpoint_round_or_epoch"]) != selected:
            raise ValueError(f"selected step mismatch: {mid}")
        if validation["checkpoint_sha256"] != digest:
            raise ValueError(f"validation checkpoint mismatch: {mid}")
        cfg = checkpoint.get("resolved_config") or {}
        if not local and (checkpoint.get("seed") != evidence["seed"] or cfg.get("preprocessing_version") != evidence["transform"]):
            raise ValueError(f"config identity mismatch: {mid}")
        config_keys = ("batch_size", "lr", "optimizer", "weight_decay", "amp", "epochs", "rounds", "local_epochs",
                       "client_lr", "client_optimizer", "client_momentum", "client_weight_decay", "init",
                       "preprocessing_version", "plantdoc_sampling_ratio", "sampler_repeat_policy", "scheduler_policy")
        item = {"model_id": mid, "selected_step": selected, "checkpoint_bytes": len(raw),
                "config": {k: cfg[k] for k in config_keys if k in cfg},
                "validation_overall": validation["validation"]["overall"], "history": []}
        if mid == "centralized":
            item.update(completed_steps=len(central_summary["history"]), total_runtime_seconds=central_summary["total_runtime_seconds"],
                        status=central_summary["status"],
                        train_accuracy_by_source={name: block["accuracy"] for name, block in central_summary["best_checkpoint_train"]["per_source"].items()})
            history = central_summary["history"]
        elif local:
            item["completed_steps"] = local_summary["rounds_per_client"]
            history = []  # local checkpoints contain no per-round history or resolved config
        else:
            import re
            result_text = (STORE / "fedavg/RESULTS.md").read_text(encoding="utf-8")
            match = re.search(r"completed at round (\d+)", result_text)
            if not match:
                raise ValueError("missing FedAvg completion evidence")
            item["completed_steps"] = int(match[1])
            history = checkpoint["history"]
        for row in history:
            entry = {k: row[k] for k in ("epoch", "round", "train_loss", "val_loss", "val_accuracy",
                     "plantdoc_macro_f1_supported", "plantvillage_macro_f1_supported", "duration_sec", "lr", "lr_next_round") if k in row}
            if mid == "fedavg":
                events = row["telemetry_events"]
                events = [e for e in events if e.get("event_type") == "client_round_execution"]
                if len(events) != 5 or {e["client_id"] for e in events} != set(evidence["shard_counts"]):
                    raise ValueError("missing client events")
                entry["clients"] = []
                for event in events:
                    n_k = evidence["shard_counts"][event["client_id"]]
                    if event["transport_mode"] != "in_process_simulation" or event["sample_count_n_k"] != n_k:
                        raise ValueError("unexpected transport or shard")
                    if abs(event["weight_p_k"] - n_k / sum(evidence["shard_counts"].values())) > 1e-9:
                        raise ValueError("unexpected aggregation weight")
                    client = {k: event[k] for k in ("client_id", "sample_count_n_k", "weight_p_k",
                        "train_loss", "delta_l2", "payload_down_bytes", "payload_up_bytes", "optimizer_steps_applied")}
                    # `delta_l2` gom mọi tensor float, kể cả thống kê BatchNorm (running_mean/var). Tách hai phần để biết
                    # trọng số học được thực sự trôi bao nhiêu. Chuẩn từng tensor đã làm tròn 6 chữ số nên kiểm lại bằng sai số tương đối.
                    layer = event["layer_delta_norms"]
                    bn_sq = sum(v * v for name, v in layer.items() if name.endswith(("running_mean", "running_var")))
                    learn_sq = sum(v * v for name, v in layer.items() if not name.endswith(("running_mean", "running_var")))
                    if abs((bn_sq + learn_sq) ** 0.5 - event["delta_l2"]) > 1e-3 * max(event["delta_l2"], 1.0):
                        raise ValueError("layer_delta_norms không khớp delta_l2")
                    # Chỉ lưu phần học được (làm tròn như web hiển thị) để payload /api/evidence/summary nhỏ; phần BatchNorm
                    # suy ra được: sqrt(delta_l2^2 - delta_l2_learnable^2).
                    client["delta_l2_learnable"] = round(learn_sq ** 0.5, 4)
                    entry["clients"].append(client)
            item["history"].append(entry)
        report["models"].append(item)

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    raw = (release / "release_manifest.json").read_bytes()
    if evidence["release_manifest_sha256"] not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.rstrip(b"\r\n") + b"\r\n").hexdigest()}:
        raise ValueError("release identity mismatch")
    manifest = json.loads(raw)
    splits = []
    for name in ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine"):
        raw = (release / "manifests" / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["manifest_hashes"][name]:
            raise ValueError(f"split hash mismatch: {name}")
        counts = Counter(json.loads(line)["source_domain"] for line in raw.decode("utf-8").splitlines() if line.strip())
        splits.append({"name": name, "count": sum(counts.values()), "by_source": dict(counts),
                       "manifest_sha256": manifest["manifest_hashes"][name]})
    report["release"] = {"name": manifest["release_name"], "crop_policy": manifest["crop_policy"], "splits": splits}
    report["transport_mode"] = "in_process_simulation"
    report["note"] = "History is extracted from the selected checkpoint (FedAvg rounds 1–27); completion at 32 is documented in RESULTS.md. Payload bytes are simulated, not measured network traffic."
    return report


if __name__ == "__main__":
    from src import registry as reg
    result = build_report()
    output = reg.abs_path(reg.TRAINING_PROTOCOL_REPORT_JSON)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}: {len(result['models'])} checkpoints")
