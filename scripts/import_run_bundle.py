"""Nhập các run do scripts/train_v4.py sinh ra thành MỘT gói mô hình cho web (artifacts/model_bundles/<tên>/).

Gói có đúng bố cục của gói cũ (import_branch_demo.py) để các bộ đọc hiện có dùng lại được, nhưng mọi nhận dạng nằm ở `import_manifest.json`
(đọc qua src/bundle.py): release, tập validation, hạt giống, tên mô hình, giao thức train. Không còn hằng số SHA/tên mô hình viết cứng ở nơi khác.

Làm gì:
  1. kiểm 7 run (Centralized, FedAvg, Local-only client 0-4) cùng hạt giống: có DONE, cùng release, cùng mã train, không phải run thử (`--limit`);
  2. chép checkpoint nguyên byte, chép hợp đồng (taxonomy, transform canonical_v1) và kiểm thứ tự 38 lớp khớp release;
  3. ĐÁNH GIÁ LẠI cả 7 checkpoint trên CPU (cùng đường suy luận với web) trên tập validation của release, ghi validation_metrics.json, bảng so sánh,
     summary.json theo từng cách, artifact_manifest.json (kích thước + SHA-256 từng tệp);
  4. ghi import_manifest.json rồi đổi tên thư mục tạm thành thư mục gói (không ghi đè gói đã có).

Chưa đổi con trỏ `artifacts/active_bundle.json`: việc đó do scripts/swap_web_models.py làm sau khi báo cáo đã sinh lại.

    python scripts/import_run_bundle.py --runs runs/v4b --seed 42 --name v4b_s42 --label v4b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bundle, registry as reg  # noqa: E402

CONTRACT_FILES = (
    "plant_data_contract/plant_data_contract/taxonomy.py",
    "plant_data_contract/plant_data_contract/transforms.py",
    "plant_data_contract/plant_data_contract/models.py",
)
HIGH_CONFIDENCE = 0.9
METRICS_SOURCE = "validation_comparison.csv do đồ án tự tính trên CPU từ checkpoint đã băm (scripts/import_run_bundle.py), cùng đường suy luận với web"
EVIDENCE_LEVEL = "locally_trained_and_reevaluated"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    """Ghi theo byte với xuống dòng LF cố định: mã băm trong artifact_manifest phải giống nhau trên Windows và Linux (write_text trên Windows đổi thành CRLF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- danh sách run <-> mô hình của gói
def model_plan(seed: int, label: str) -> list[dict[str, Any]]:
    """7 mô hình của gói: id giữ nguyên như gói cũ (centralized, fedavg, local_00..04) để mọi bộ đọc dùng lại được; chỉ tên hiển thị đổi."""
    plan = [
        {"model_id": "centralized", "dir": "centralized", "file": "checkpoint_best.pt", "run": f"centralized_s{seed}", "method": "centralized",
         "client": None, "display_name": f"Centralized {label}", "phase": label, "label_method": f"Centralized {label}"},
        {"model_id": "fedavg", "dir": "fedavg", "file": "checkpoint_best.pt", "run": f"fedavg_s{seed}", "method": "fedavg", "client": None,
         "display_name": f"FedAvg {label}", "phase": label, "label_method": f"FedAvg {label}"},
    ]
    for k in range(5):
        plan.append({"model_id": f"local_{k:02d}", "dir": "local", "file": f"client_{k:02d}/checkpoint_best.pt", "run": f"local_c{k}_s{seed}",
                     "method": "local", "client": k, "display_name": f"Local {k:02d}", "phase": label, "label_method": "Local-only"})
    return plan


def load_run(runs: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = runs / spec["run"]
    for name in ("DONE", "final.json", "config.json", "history.jsonl", "checkpoint_best.pt"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{spec['run']}: thiếu {name} (run chưa xong hoặc chưa kéo về)")
    final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    config = final["config"]
    if config.get("limit"):
        raise ValueError(f"{spec['run']}: là run thử (--limit), không nhập vào gói")
    if config["method"] != spec["method"] or config.get("client") != spec["client"]:
        raise ValueError(f"{spec['run']}: method/client trong config không khớp tên run")
    return {"dir": run_dir, "final": final, "config": config, "history": read_jsonl(run_dir / "history.jsonl")}


def check_runs_consistent(runs: dict[str, dict[str, Any]], seed: int, release_sha: str) -> dict[str, Any]:
    """Cả 7 run phải cùng release, cùng mã train và cùng công thức (khác nhau chỉ ở cách học và client)."""
    identity: dict[str, set[Any]] = {k: set() for k in ("release_manifest_sha256", "code_fingerprint", "git_commit", "batch_size", "lr",
                                                         "momentum", "weight_decay", "label_smoothing", "train_transform", "budget",
                                                         "amp", "loss_weighting", "selection")}
    for model_id, run in runs.items():
        config = run["config"]
        if config["seed"] != seed:
            raise ValueError(f"{model_id}: hạt giống {config['seed']} khác {seed}")
        for key in identity:
            identity[key].add(json.dumps(config.get(key)))
    bad = {k: sorted(v) for k, v in identity.items() if len(v) != 1}
    if bad:
        raise ValueError(f"các run không cùng công thức/mã/release: {bad}")
    if json.loads(next(iter(identity["release_manifest_sha256"]))) != release_sha:
        raise ValueError("release của các run khác release trên đĩa (data/dataset_sources/mixed/...)")
    return {k: json.loads(next(iter(v))) for k, v in identity.items()}


# --------------------------------------------------------------------------- số liệu (thuần numpy, có test)
def quality_block(targets: np.ndarray, preds: np.ndarray, conf: np.ndarray, classes: list[str]) -> dict[str, Any]:
    from scripts.evaluate_published_models import per_class_recall

    crops = np.array([c.split("___", 1)[0] for c in classes])
    right_crop = crops[targets] == crops[preds]
    wrong = targets != preds
    hcw = wrong & (conf >= HIGH_CONFIDENCE)
    n = max(1, len(targets))
    return {
        "count": int(len(targets)),
        "crop_accuracy": float(right_crop.mean()) if len(targets) else 0.0,
        "wrong_crop_count": int((~right_crop).sum()),
        "right_crop_wrong_disease_count": int((right_crop & wrong).sum()),
        "high_confidence_wrong_rate": float(hcw.sum()) / n,
        "high_confidence_wrong_count": int(hcw.sum()),
        "per_class_recall": per_class_recall(targets, preds, classes),
    }


def validation_metrics(logits: np.ndarray, rows: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """Khối `validation` của validation_metrics.json: overall, by_source (plantdoc, plantvillage, field_other), quality, top_confusions."""
    from collections import Counter

    from scripts.evaluate_published_models import SOURCES, group_scores, present_sources, softmax, source_group

    targets = np.array([r["class_id"] for r in rows], dtype=np.int64)
    sources = np.array([source_group(r["source_dataset"]) for r in rows])
    probs = softmax(logits)
    preds, conf = probs.argmax(axis=1), probs.max(axis=1)

    def block(mask: np.ndarray) -> dict[str, Any]:
        scores = group_scores(targets[mask], preds[mask], conf[mask], classes)
        scores.pop("correct", None)       # định dạng của gói cũ không có trường này
        return scores

    present = present_sources(sources)
    return {
        "overall": block(np.ones(len(rows), dtype=bool)),
        "by_source": {s: block(sources == s) for s in present},
        "quality": {"high_confidence_threshold": HIGH_CONFIDENCE,
                    "by_source": {s: quality_block(targets[sources == s], preds[sources == s], conf[sources == s], classes) for s in present}},
        "top_confusions": [{"target": classes[t], "predicted": classes[p], "count": int(c)}
                           for (t, p), c in Counter((int(t), int(p)) for t, p in zip(targets, preds) if t != p).most_common(20)],
        "sources": list(SOURCES),
    }


def comparison_rows(display_name: str, metrics: dict[str, Any], checkpoint_sha: str, release_sha: str, validation_sha: str) -> list[dict[str, Any]]:
    return [{"model": display_name, "source": source, "count": block["count"], "top1_accuracy": block["top1_accuracy"],
             "macro_f1_supported": block["macro_f1_supported"], "crop_accuracy": block["crop_accuracy"], "checkpoint_sha256": checkpoint_sha,
             "release_manifest_sha256": release_sha, "validation_manifest_sha256": validation_sha}
            for source, block in metrics["by_source"].items()]


def training_info(config: dict[str, Any], final: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Giao thức train của một run, lấy từ config.json/final.json (web hiển thị, không viết cứng trong JS)."""
    fedavg = config["method"] == "fedavg"
    return {
        "method": config["method"], "client": config.get("client"), "seed": config["seed"],
        "optimizer": f"SGD động lượng {config['momentum']}, weight decay {config['weight_decay']}",
        "optimizer_state": "làm mới mỗi vòng ở từng client" if fedavg else "giữ xuyên suốt",
        "lr": config["lr"], "lr_schedule": f"khởi động {config['warmup']} {'vòng' if fedavg else 'epoch'}, sau đó cosine về 0",
        "batch_size": config["batch_size"], "label_smoothing": config["label_smoothing"],
        "units": "vòng" if fedavg else "epoch",
        "budget": config["budget"], "local_epochs": config.get("local_epochs", 1), "steps_total": len(history),
        "best_step": int(final["best_step"]) + 1, "selection": config["selection"],
        "precision": config.get("amp", "fp16"), "train_transform": config["train_transform"],
        "loss_weighting": config["loss_weighting"], "train_sizes": config["train_sizes"],
        "scheme": config.get("scheme"), "release_name": config["release_name"],
        "code_fingerprint": config["code_fingerprint"], "git_commit": config["git_commit"],
        "gpu": config.get("gpu"), "torch": config.get("torch"),
    }


def history_entries(history: list[dict[str, Any]], shard_sizes: list[int] | None = None) -> list[dict[str, Any]]:
    """Một dòng mỗi epoch/vòng từ history.jsonl của train_v4. FedAvg thêm `clients` (n_k, trọng số p_k, mất mát và số bước từng client);
    `delta_l2` KHÔNG được bộ train ghi lại nên để None (web ẩn phần đó thay vì bịa)."""
    out = []
    total = sum(shard_sizes) if shard_sizes else 0
    for item in history:
        train = item["train"]
        by_source = item["val"].get("by_source") or {}
        entry = {"epoch": item["step"] + 1, "lr": item["lr"], "val_accuracy": item["val"]["top1"], "val_loss": None,
                 "selection_score": item["val"]["selection_score"], "seconds": item["seconds"],
                 "plantdoc_macro_f1_supported": (by_source.get("PlantDoc") or {}).get("macro_f1"),
                 "plantvillage_macro_f1_supported": (by_source.get("PlantVillage") or {}).get("macro_f1")}
        if "clients" in train:       # FedAvg: mất mát gộp theo số bước của từng client
            steps = sum(c["steps"] for c in train["clients"]) or 1
            entry["train_loss"] = sum(c["loss"] * c["steps"] for c in train["clients"]) / steps
            entry["clients"] = [{"client_id": f"client_{c['client']:02d}", "train_loss": c["loss"], "optimizer_steps_applied": c["steps"],
                                 **({"sample_count_n_k": int(shard_sizes[c["client"]]), "weight_p_k": shard_sizes[c["client"]] / total} if shard_sizes else {})}
                                for c in train["clients"]]
        else:
            entry["train_loss"] = train["loss"]
        out.append(entry)
    return out


def class_coverage(histogram: dict[str, int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.evaluate_published_models import source_group

    seen = {int(k) for k, v in histogram.items() if int(v) > 0}
    classes = [int(r["class_id"]) for r in rows]
    groups = [source_group(r["source_dataset"]) for r in rows]

    def fraction(names: tuple[str, ...] | None) -> float:
        picked = [c for c, g in zip(classes, groups) if names is None or g in names]
        return sum(1 for c in picked if c in seen) / max(1, len(picked))

    return {"train_label_count": len(seen), "missing_class_ids": sorted(set(range(38)) - seen),
            "validation_seen_label_fraction": fraction(None), "plantdoc_validation_seen_label_fraction": fraction(("plantdoc",)),
            "plantvillage_validation_seen_label_fraction": fraction(("plantvillage",))}


def artifact_manifest(directory: Path, branch: str, release_sha: str, validation_sha: str) -> dict[str, Any]:
    files = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        files[path.relative_to(directory).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"branch": branch, "validation_manifest_sha256": validation_sha, "release_manifest_sha256": release_sha, "files": files}


# --------------------------------------------------------------------------- nhập gói
def count_groups(rows: list[dict[str, Any]]) -> dict[str, int]:
    from scripts.evaluate_published_models import SOURCES, source_group

    counts = {s: 0 for s in SOURCES}
    for r in rows:
        counts[source_group(r["source_dataset"])] += 1
    return {k: v for k, v in counts.items() if v}


def evaluate_on_cpu(plan: list[dict[str, Any]], staging: Path, rows: list[dict[str, Any]], dataset_root: Path,
                    batch_size: int, workers: int, cache: Path | None = None) -> dict[str, np.ndarray]:
    import torch

    from scripts.evaluate_published_models import _load_contract_transforms, _load_model, collect_logits

    # Dấu vân tay: byte từng checkpoint + danh sách ảnh validation. Khớp thì dùng lại logits (nhập lại gói không phải suy luận lại ~10 phút).
    fingerprint = hashlib.sha256(json.dumps([{s["model_id"]: sha256_file(staging / s["dir"] / s["file"]) for s in plan},
                                             [r["sample_id"] for r in rows]], sort_keys=True).encode()).hexdigest()
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                print(f"[bundle] dùng lại logits đã lưu ({cache.name})", flush=True)
                return {s["model_id"]: data[s["model_id"]] for s in plan}
    torch.set_num_threads(max(1, min(10, torch.get_num_threads())))
    transform = _load_contract_transforms(staging)
    models = {spec["model_id"]: _load_model(staging / spec["dir"] / spec["file"]) for spec in plan}
    print(f"[bundle] đánh giá lại {len(models)} mô hình trên {len(rows)} ảnh validation (CPU)", flush=True)
    logits = collect_logits(rows, dataset_root, transform, models, batch_size, workers)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fingerprint=fingerprint, **logits)
    return logits


def import_runs(runs_dir: Path, seed: int, name: str, label: str, target: Path, release: Path, dataset_root: Path,
                batch_size: int = 64, workers: int = 5, logits: dict[str, np.ndarray] | None = None) -> Path:
    if target.exists():
        raise FileExistsError(f"Gói đã có, không ghi đè: {target}")
    plan = model_plan(seed, label)
    runs = {spec["model_id"]: load_run(runs_dir, spec) for spec in plan}
    release_manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    release_sha = sha256_file(release / "release_manifest.json")
    recipe = check_runs_consistent(runs, seed, release_sha)
    class_to_idx = json.loads((release / "taxonomy.json").read_text(encoding="utf-8"))["class_to_idx"]
    classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    if len(classes) != 38 or [class_to_idx[c] for c in classes] != list(range(38)):
        raise ValueError("taxonomy.json của release không phải 38 lớp đánh số 0..37")
    val_rows = read_jsonl(release / "manifests" / "val.jsonl")
    test_rows = read_jsonl(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
    validation_sha = release_manifest["manifest_hashes"]["val"]
    if sha256_file(release / "manifests" / "val.jsonl") != validation_sha:
        raise ValueError("val.jsonl khác mã băm trong release_manifest.json")

    legacy = reg.abs_path(bundle.LEGACY_DIR) / "contract"
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        # hợp đồng: cùng taxonomy 38 lớp và transform canonical_v1 với gói cũ; kiểm thứ tự lớp khớp release
        (staging / "contract").mkdir()
        for relative in CONTRACT_FILES:
            shutil.copy2(legacy / Path(relative).name, staging / "contract" / Path(relative).name)
        from scripts.import_branch_demo import classes_from_taxonomy

        if classes_from_taxonomy(staging / "contract" / "taxonomy.py") != classes:
            raise ValueError("thứ tự 38 lớp của hợp đồng khác release")
        for spec in plan:
            destination = staging / spec["dir"] / spec["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runs[spec["model_id"]]["dir"] / "checkpoint_best.pt", destination)

        started = time.time()
        if logits is None:
            logits = evaluate_on_cpu(plan, staging, val_rows, dataset_root, batch_size, workers,
                                     cache=reg.abs_path(f"data/reeval/bundle_{name}_logits.npz"))
        print(f"[bundle] đánh giá xong sau {time.time() - started:.0f} giây", flush=True)

        w0 = None
        try:
            from scripts.train_v4 import build_model
            from src.model.mobilenetv3 import compute_model_hash, get_parameters

            w0 = compute_model_hash(get_parameters(build_model(seed)))
        except Exception as exc:  # noqa: BLE001 - không có trọng số ImageNet ngoại tuyến: ghi None, web không kiểm W0
            print(f"[bundle] không tính được W0 ({exc}); ghi null", file=sys.stderr)

        shard_sizes = runs["fedavg"]["config"]["train_sizes"]
        shard_total = int(sum(shard_sizes))
        spec_path = release / "partition_spec_v4.json"
        histograms = {}
        if spec_path.is_file():
            stats = json.loads(spec_path.read_text(encoding="utf-8"))["scenarios"][recipe_scheme(runs)]["client_stats"]
            histograms = {k: v["class_histogram"] for k, v in stats.items()}
        all_rows: list[dict[str, Any]] = []
        metrics: dict[str, dict[str, Any]] = {}
        checkpoint_sha: dict[str, str] = {}
        for spec in plan:
            model_id = spec["model_id"]
            run = runs[model_id]
            checkpoint_sha[model_id] = sha256_file(staging / spec["dir"] / spec["file"])
            metrics[model_id] = validation_metrics(logits[model_id], val_rows, classes)
            model_dir = staging / spec["dir"] / Path(spec["file"]).parent
            write_json(model_dir / "validation_metrics.json", {
                "checkpoint": f"{runs_dir.as_posix()}/{spec['run']}/checkpoint_best.pt", "checkpoint_sha256": checkpoint_sha[model_id],
                "checkpoint_round_or_epoch": int(run["final"]["best_step"]) + 1, "release_manifest_sha256": release_sha,
                "validation": {"manifest": (release / "manifests" / "val.jsonl").as_posix(), "manifest_sha256": validation_sha, **metrics[model_id]},
            })
            all_rows += comparison_rows(spec["display_name"], metrics[model_id], checkpoint_sha[model_id], release_sha, validation_sha)

        header = ["model", "source", "count", "top1_accuracy", "macro_f1_supported", "crop_accuracy", "checkpoint_sha256",
                  "release_manifest_sha256", "validation_manifest_sha256"]
        for directory in ("centralized", "fedavg", "local"):
            with (staging / directory / "validation_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerows(all_rows)

        training = {spec["model_id"]: training_info(runs[spec["model_id"]]["config"], runs[spec["model_id"]]["final"], runs[spec["model_id"]]["history"])
                    for spec in plan}
        test_blocks = {spec["model_id"]: runs[spec["model_id"]]["final"]["legacy_diagnostic_test"]["by_source"] for spec in plan}

        def recorded_test(model_id: str) -> dict[str, Any]:
            blocks = test_blocks[model_id]
            return {"per_source": {s.lower(): {"count": int(b["n"]), "accuracy": float(b["top1"])} for s, b in blocks.items()
                                   if s.lower() in ("plantdoc", "plantvillage")}}

        common = {"seed": seed, "w0_fingerprint": w0, "release_manifest_sha256": release_sha}
        write_json(staging / "centralized" / "summary.json", {
            **common, "history": history_entries(runs["centralized"]["history"]), "best_epoch": training["centralized"]["best_step"],
            "total_runtime_seconds": round(sum(h["seconds"] for h in runs["centralized"]["history"]), 1),
            "legacy_diagnostic_test": recorded_test("centralized"), "training": training["centralized"]})
        write_json(staging / "fedavg" / "summary.json", {
            **common, "history": history_entries(runs["fedavg"]["history"], shard_sizes), "best_round": training["fedavg"]["best_step"],
            "legacy_diagnostic_test": recorded_test("fedavg"), "training": training["fedavg"]})
        clients = {}
        for k in range(5):
            model_id = f"local_{k:02d}"
            run = runs[model_id]
            if run["config"]["train_sizes"][0] != shard_sizes[k]:
                raise ValueError(f"{model_id}: n_k khác shard tương ứng của FedAvg")
            by = metrics[model_id]["by_source"]
            clients[f"client_{k:02d}"] = {
                "n_k": int(shard_sizes[k]),
                **({"class_coverage": class_coverage(histograms[f"client_{k:02d}"], val_rows)} if histograms else {}),
                "best_validation": {"round": training[model_id]["best_step"], "accuracy": run["final"]["val"]["top1"],
                                    "plantdoc_macro_f1_supported": by.get("plantdoc", {}).get("macro_f1_supported"),
                                    "plantvillage_macro_f1_supported": by.get("plantvillage", {}).get("macro_f1_supported")},
            }
            write_json(staging / "local" / f"client_{k:02d}" / "summary.json", {
                **common, "client": k, "n_k": int(shard_sizes[k]), "history": history_entries(run["history"]),
                "legacy_diagnostic_test": recorded_test(model_id), "training": training[model_id]})
        write_json(staging / "local" / "combined_summary.json", {
            "backend": "local_only", "partition_scheme": recipe_scheme(runs), "rounds_per_client": runs["local_00"]["config"]["budget"],
            **common, "clients": clients})

        code = {"git_commit": recipe["git_commit"], "code_fingerprint": recipe["code_fingerprint"]}
        branches = {"centralized": f"{runs_dir.as_posix()}/centralized_s{seed}", "fedavg": f"{runs_dir.as_posix()}/fedavg_s{seed}",
                    "local": f"{runs_dir.as_posix()}/local_c*_s{seed}"}
        for directory, branch in branches.items():
            write_json(staging / directory / "artifact_manifest.json", artifact_manifest(staging / directory, branch, release_sha, validation_sha))

        manifest = {
            "bundle_id": name, "protocol_id": f"mixed_{release.name}_pretrained_seed{seed}_validation",
            "evidence_level": EVIDENCE_LEVEL, "metrics_source": METRICS_SOURCE,
            "release_manifest_sha256": release_sha, "release_name": release.name, "release_dir": release.relative_to(reg.abs_path(".")).as_posix()
            if release.is_relative_to(reg.abs_path(".")) else release.as_posix(),
            "validation_manifest_sha256": validation_sha, "validation_counts": count_groups(val_rows),
            "w0_fingerprint": w0, "w0_in_checkpoints": False, "seed": seed, "partition_scheme": recipe_scheme(runs),
            "num_classes": 38, "transform": "canonical_v1",
            "contract_hashes": {relative: sha256_file(staging / "contract" / Path(relative).name) for relative in CONTRACT_FILES},
            "classes": classes, "shard_total": shard_total,
            "holdout": {"manifest_sha256": release_manifest["manifest_hashes"][reg.HOLDOUT_SPLIT], "counts": count_groups(test_rows)},
            "sources": {directory: {"branch": branch, "commit": code["git_commit"]} for directory, branch in branches.items()},
            "models": [{k: spec[k] for k in ("model_id", "dir", "file", "display_name", "phase")}
                       | {"method": spec["label_method"], "registry_id": f"web_{spec['model_id']}"} for spec in plan],
            "training": training, "recipe": recipe, "code": code,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(staging / "import_manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def recipe_scheme(runs: dict[str, dict[str, Any]]) -> str:
    return str(runs["fedavg"]["config"].get("scheme") or "label_alpha_0_1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="thư mục chứa các run của scripts/train_v4.py (ví dụ runs/v4b)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True, help="tên thư mục gói, ví dụ v4b_s42")
    parser.add_argument("--label", required=True, help="nhãn hiển thị cạnh tên mô hình, ví dụ v4b -> 'Centralized v4b'")
    parser.add_argument("--release", type=Path, default=None, help="thư mục release (mặc định data/dataset_sources/mixed/pv_pd_v4)")
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    release = args.release or reg.abs_path(f"{reg.DATASET_SOURCES_DIR}/mixed/pv_pd_v4")
    target = args.target or reg.abs_path(f"{bundle.BUNDLES_DIR}/{args.name}")
    path = import_runs(args.runs, args.seed, args.name, args.label, target, release, reg.abs_path(reg.DATASET_SOURCES_DIR), args.batch_size, args.workers)
    print(f"[bundle] đã tạo {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
