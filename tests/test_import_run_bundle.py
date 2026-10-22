"""scripts/import_run_bundle.py: gói dựng từ các run train_v4 phải được chính web (web.demo_evidence.load_evidence) chấp nhận,
và mọi nhận dạng của gói nằm ở import_manifest.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small

from scripts import import_run_bundle as imp
from src import bundle, registry as reg
from web.demo_evidence import load_evidence

SEED = 7
LABEL = "t"
SHARDS = [4, 3, 2, 5, 3]
SOURCES = ["PlantVillage", "PlantVillage", "PlantDoc", "PlantDoc", "mendeley_x", "mendeley_x"]


def legacy_classes() -> list[str]:
    from scripts.import_branch_demo import classes_from_taxonomy

    return classes_from_taxonomy(reg.abs_path(bundle.LEGACY_DIR) / "contract" / "taxonomy.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, root: Path, name: str, classes: list[str], count: int) -> list[dict]:
    rows = []
    for i in range(count):
        rel = f"imgs/{name}_{i}.jpg"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (20 + 9 * i, 100, 60)).save(root / rel)
        rows.append({"sample_id": f"{name}{i}", "relative_path": rel, "class_id": i % 38, "target_class": classes[i % 38],
                     "source_dataset": SOURCES[i % len(SOURCES)], "group_id": f"{name}{i}"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


@pytest.fixture()
def world(tmp_path):
    classes = legacy_classes()
    dataset_root = tmp_path / "dataset_sources"
    release = dataset_root / "mixed" / "pv_pd_v9"
    (release / "manifests").mkdir(parents=True)
    val_rows = write_rows(release / "manifests" / "val.jsonl", dataset_root, "val", classes, 12)
    write_rows(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl", dataset_root, "test", classes, 6)
    (release / "taxonomy.json").write_text(json.dumps({"class_to_idx": {c: i for i, c in enumerate(classes)}}), encoding="utf-8")
    (release / "release_manifest.json").write_text(json.dumps({"manifest_hashes": {
        "val": sha(release / "manifests" / "val.jsonl"), reg.HOLDOUT_SPLIT: sha(release / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")}}), encoding="utf-8")
    (release / "partition_spec_v4.json").write_text(json.dumps({"scenarios": {"label_alpha_0_1": {"client_stats": {
        f"client_{k:02d}": {"class_histogram": {str(c): (1 if c < 10 + k else 0) for c in range(38)}} for k in range(5)}}}}), encoding="utf-8")
    release_sha = sha(release / "release_manifest.json")

    model = mobilenet_v3_small(weights=None, num_classes=38)
    runs = tmp_path / "runs"
    for spec in imp.model_plan(SEED, LABEL):
        run = runs / spec["run"]
        run.mkdir(parents=True)
        config = {"method": spec["method"], "seed": SEED, "client": spec["client"], "scheme": "label_alpha_0_1", "budget": 2, "local_epochs": 1,
                  "batch_size": 128, "lr": 0.02, "momentum": 0.9, "weight_decay": 1e-4, "label_smoothing": 0.1, "warmup": 1, "limit": 0,
                  "train_transform": "gpu_v2", "loss_weighting": "domain_balanced", "selection": "mean macro-F1", "amp": "bf16", "release_name": "pv_pd_v9",
                  "release_manifest_sha256": release_sha, "code_fingerprint": "f" * 64, "git_commit": "c" * 40, "gpu": "gpu", "torch": "2",
                  "train_sizes": SHARDS if spec["method"] != "local" else [SHARDS[spec["client"]]] if spec["client"] is not None else [sum(SHARDS)]}
        torch.save({"model_state": model.state_dict(), "step": 1, "selection_score": 0.5, "method": spec["method"], "seed": SEED,
                    "client": spec["client"], "release_manifest_sha256": release_sha, "code_fingerprint": "f" * 64}, run / "checkpoint_best.pt")
        train = {"clients": [{"client": k, "loss": 1.0 + k, "steps": 2 + k} for k in range(5)], "steps": 20} if spec["method"] == "fedavg" else {"loss": 1.5, "steps": 4}
        history = [{"step": s, "lr": 0.02, "train": train, "val": {"top1": 0.4 + s / 10, "selection_score": 0.3 + s / 10}, "seconds": 1.0} for s in range(2)]
        (run / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history), encoding="utf-8")
        final = {"run": spec["run"], "best_step": 1, "selection_score": 0.4, "config": config, "val": {"top1": 0.5},
                 "legacy_diagnostic_test": {"by_source": {"PlantDoc": {"n": 2, "top1": 0.5}, "PlantVillage": {"n": 2, "top1": 1.0}, "mendeley_x": {"n": 2, "top1": 0.0}}}}
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "final.json").write_text(json.dumps(final), encoding="utf-8")
        (run / "DONE").write_text("done\n", encoding="utf-8")
    rng = np.random.default_rng(0)
    logits = {spec["model_id"]: rng.normal(size=(len(val_rows), 38)).astype(np.float32) for spec in imp.model_plan(SEED, LABEL)}
    return {"tmp": tmp_path, "runs": runs, "release": release, "dataset_root": dataset_root, "classes": classes, "logits": logits, "val_rows": val_rows}


def build(world, name="b1", **kw):
    return imp.import_runs(world["runs"], SEED, name, LABEL, world["tmp"] / "bundles" / name, world["release"], world["dataset_root"],
                           logits=world["logits"], **kw)


def test_the_web_accepts_a_bundle_built_from_runs(world, monkeypatch):
    import scripts.train_v4 as trainer

    monkeypatch.setattr(trainer, "build_model", lambda seed: torch.nn.Linear(2, 2))
    target = build(world)
    evidence = load_evidence(target)
    assert evidence["available"], evidence.get("reason")
    assert [m["model_id"] for m in evidence["models"]] == ["centralized", "fedavg", *(f"local_{k:02d}" for k in range(5))]
    assert [m["name"] for m in evidence["models"]][:2] == ["Centralized t", "FedAvg t"]
    assert evidence["validation_counts"] == {"plantdoc": 4, "plantvillage": 4, "field_other": 4}
    assert evidence["w0_in_checkpoints"] is False and evidence["w0_fingerprint"]
    assert sum(evidence["shard_counts"].values()) == sum(SHARDS)
    assert len(evidence["centralized_history"]) == 2 and evidence["centralized_history"][0]["val_loss"] is None
    manifest = bundle.load_manifest(target)
    assert manifest["protocol_id"] == f"mixed_pv_pd_v9_pretrained_seed{SEED}_validation" and manifest["evidence_level"] == imp.EVIDENCE_LEVEL
    assert manifest["training"]["fedavg"]["units"] == "vòng" and manifest["training"]["centralized"]["units"] == "epoch"


def test_the_web_can_build_and_run_a_model_from_the_bundle(world, monkeypatch):
    import scripts.train_v4 as trainer
    from web import demo_evidence

    monkeypatch.setattr(trainer, "build_model", lambda seed: torch.nn.Linear(2, 2))
    target = build(world)
    result = demo_evidence.predict_published(Image.new("RGB", (60, 50), (30, 120, 40)), "fedavg", target)
    assert result["ready"] and result["method"] == "FedAvg t" and len(result["top3"]) == 3


@pytest.mark.parametrize("edit", [
    lambda m: m.update(release_manifest_sha256="1" * 64),
    lambda m: m.update(seed=1),
    lambda m: m.update(shard_total=1),
    lambda m: m["models"][0].update(display_name="Centralized renamed"),
    lambda m: m.update(validation_counts={"plantdoc": 9, "plantvillage": 9, "field_other": 9}),
])
def test_editing_the_manifest_after_import_is_refused(world, monkeypatch, edit):
    import scripts.train_v4 as trainer

    monkeypatch.setattr(trainer, "build_model", lambda seed: torch.nn.Linear(2, 2))
    target = build(world)
    manifest = json.loads((target / "import_manifest.json").read_text(encoding="utf-8"))
    edit(manifest)
    (target / "import_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_evidence(target)["available"] is False


def test_import_refuses_unfinished_trial_inconsistent_runs_and_existing_target(world, monkeypatch):
    import scripts.train_v4 as trainer

    monkeypatch.setattr(trainer, "build_model", lambda seed: torch.nn.Linear(2, 2))
    (world["runs"] / "local_c3_s7" / "DONE").unlink()
    with pytest.raises(FileNotFoundError, match="local_c3_s7"):
        build(world, "x1")
    (world["runs"] / "local_c3_s7" / "DONE").write_text("done\n", encoding="utf-8")

    final = json.loads((world["runs"] / "fedavg_s7" / "final.json").read_text(encoding="utf-8"))
    final["config"]["limit"] = 100
    (world["runs"] / "fedavg_s7" / "final.json").write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="run thử"):
        build(world, "x2")
    final["config"]["limit"] = 0
    final["config"]["lr"] = 0.5                      # khác công thức của các run còn lại
    (world["runs"] / "fedavg_s7" / "final.json").write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="không cùng công thức"):
        build(world, "x3")
    final["config"]["lr"] = 0.02
    (world["runs"] / "fedavg_s7" / "final.json").write_text(json.dumps(final), encoding="utf-8")

    target = build(world, "ok")
    with pytest.raises(FileExistsError):
        build(world, "ok")
    assert target.is_dir() and not (target.parent / ".ok.staging").exists()
    assert not (world["tmp"] / "bundles" / "x1").exists()        # lần lỗi không để lại thư mục gói dở


def test_validation_metrics_group_sources_and_keep_the_legacy_shape(world):
    rows = world["val_rows"]
    metrics = imp.validation_metrics(world["logits"]["centralized"], rows, world["classes"])
    assert set(metrics["by_source"]) == {"plantdoc", "plantvillage", "field_other"}
    assert metrics["overall"]["count"] == 12 and "correct" not in metrics["overall"]
    assert metrics["by_source"]["plantdoc"]["count"] == 4
    assert set(metrics["quality"]["by_source"]["plantdoc"]) >= {"per_class_recall", "wrong_crop_count", "high_confidence_wrong_rate"}
    assert all(c["count"] > 0 for c in metrics["top_confusions"])


def test_class_coverage_counts_validation_images_of_classes_the_client_saw(world):
    coverage = imp.class_coverage({"0": 3, "1": 0, "2": 5}, world["val_rows"])
    assert coverage["train_label_count"] == 2 and 1 in coverage["missing_class_ids"] and 0 not in coverage["missing_class_ids"]
    assert 0.0 < coverage["validation_seen_label_fraction"] < 1.0
