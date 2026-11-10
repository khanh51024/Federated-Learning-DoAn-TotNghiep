"""scripts/import_pc_package.py: gói kết quả PC v3 (13 mô hình, có int8) phải được chính web chấp nhận, chỉ dùng tập validation,
và mọi sai lệch về danh tính (release, hạt giống, SHA, vòng cuối) phải bị từ chối."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_small

from scripts import import_pc_package as imp
from scripts import import_run_bundle as base
from src import bundle, registry as reg
from tests.test_int8_torchscript import _int8_torchscript_bytes
from web.demo_evidence import load_evidence

SEED = 42
SHARDS = [4, 3, 2, 5, 3]
STEPS = {"centralized_adamw": 3, "centralized_sgd": 3, "fedavg": 2, "scaffold": 2}
W0 = "e" * 64
SOURCES = ["PlantVillage", "PlantVillage", "PlantDoc", "PlantDoc"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_contract() -> Path:
    return reg.abs_path(bundle.LEGACY_DIR) / "contract"


def legacy_classes() -> list[str]:
    from scripts.import_branch_demo import classes_from_taxonomy

    return classes_from_taxonomy(legacy_contract() / "taxonomy.py")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def history_for(kind: str, steps: int) -> list[dict]:
    rows = []
    for s in range(1, steps + 1):
        row = {"val_accuracy": 0.4 + s / 10, "val_loss": 1.0, "val_macro_f1_raw": 0.3, "duration_sec": 1.5,
               "plantdoc_macro_f1_supported": 0.2, "plantvillage_macro_f1_supported": 0.5}
        if kind == "centralized":
            row.update(epoch=s, train_loss=2.0 / s, lr=0.001)
        else:
            row.update(round=s, lr_next_round=0.01, client_telemetry=[
                {"client_id": f"client_{k:02d}", "loss": 1.0 + k, "steps": 2 + k, "raw_images_n_k": SHARDS[k], "optimizer_steps_applied": 2 + k}
                for k in range(5)])
        rows.append(row)
    return rows


@pytest.fixture()
def world(tmp_path, monkeypatch):
    classes = legacy_classes()
    dataset_root = tmp_path / "dataset_sources"
    release = dataset_root / "mixed" / "pv_pd_v9"
    (release / "manifests").mkdir(parents=True)
    val_rows = []
    for i in range(8):
        rel = f"imgs/val_{i}.jpg"
        (dataset_root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 30), (20 + 9 * i, 100, 60)).save(dataset_root / rel)
        val_rows.append({"sample_id": f"v{i}", "relative_path": rel, "class_id": i % 38, "target_class": classes[i % 38],
                         "source_dataset": SOURCES[i % len(SOURCES)], "group_id": f"v{i}"})
    (release / "manifests" / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val_rows), encoding="utf-8")
    # train.jsonl hiện diện cho phần kiểm giao nhóm; KHÔNG tạo calibration.jsonl và legacy_diagnostic_test.jsonl: nhập gói mà đọc chúng sẽ lỗi
    (release / "manifests" / "train.jsonl").write_text(json.dumps({"group_id": "t0"}) + "\n", encoding="utf-8")
    (release / "taxonomy.json").write_text(json.dumps({"class_to_idx": {c: i for i, c in enumerate(classes)}}), encoding="utf-8")
    holdout = {"manifest_sha256": "9" * 64, "counts": {"plantdoc": 4, "plantvillage": 5}}
    release_raw = json.dumps({"manifest_hashes": {"val": sha(release / "manifests" / "val.jsonl"), reg.HOLDOUT_SPLIT: holdout["manifest_sha256"]},
                              "counts": {"legacy_diagnostic_test_samples": 9}})
    (release / "release_manifest.json").write_bytes(release_raw.encode() + b"\n")   # LF cuối tệp; nhóm PC băm bản CRLF
    release_sha = hashlib.sha256(release_raw.encode() + b"\r\n").hexdigest()
    for k in range(5):
        write_rows = [{"class_id": c % 38} for c in range(SHARDS[k])]
        (release / "partitions" / "label_alpha_0_1").mkdir(parents=True, exist_ok=True)
        (release / "partitions" / "label_alpha_0_1" / f"client_{k:02d}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in write_rows), encoding="utf-8")
    monkeypatch.setattr(bundle, "LEGACY_HOLDOUT", holdout)
    monkeypatch.setattr(bundle, "LEGACY_SHARD_TOTAL", sum(SHARDS))

    models = tmp_path / "pkg" / "pc_v3_ket_qua" / "models"
    (models / "contract").mkdir(parents=True, exist_ok=True)
    for name in ("taxonomy.py", "transforms.py", "models.py"):
        shutil.copy2(legacy_contract() / name, models / "contract" / name)
    state = mobilenet_v3_small(weights=None, num_classes=38).state_dict()
    plan = imp.model_plan()
    int8_bytes = _int8_torchscript_bytes()
    for spec in plan:
        path = models / spec["source"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec["format"] == bundle.FORMAT_TORCHSCRIPT_INT8:
            path.write_bytes(int8_bytes)
            info = {"sha256": sha(path), "bytes": path.stat().st_size, "parent_fp32_role": spec["fp32_role"]}
        else:
            step = STEPS.get(spec["role"], 2)
            ckpt = {"release_manifest_sha256": release_sha, "seed": SEED, "w0_fingerprint": W0}
            if spec["kind"] == "local":
                ckpt.update(model_state_dict=state, round=2)
                step = 2
            elif spec["kind"] == "centralized":
                ckpt.update(model_state=state, epoch=step, history=history_for("centralized", step))
            else:
                ckpt.update(model_state=state, round=step, history=history_for("federated", step))
            torch.save(ckpt, path)
            info = {"sha256": sha(path), "bytes": path.stat().st_size, "checkpoint_step": step}
            if spec["kind"] == "local":
                info["metrics_validation_pv_pd_v3"] = {"n_k": SHARDS[spec["client"]]}
        write_json(path.with_name("model_info.json"), info)
    rng = np.random.default_rng(0)
    logits = {spec["model_id"]: rng.normal(size=(len(val_rows), 38)).astype(np.float32) for spec in plan}
    for spec in plan:      # số "của nhóm PC" = chính số đo lại, để bảng đối chiếu có dữ liệu
        write_json((models / spec["source"]).with_name("validation_metrics.json"),
                   {"validation": base.validation_metrics(logits[spec["model_id"]], val_rows, classes)})
    write_json(models / "import_manifest.json", {
        "bundle_id": "pc_v3_test", "release_name": "pv_pd_v9", "release_manifest_sha256": release_sha, "seed": SEED, "w0_fingerprint": W0,
        "partition_scheme": "label_alpha_0_1", "classes": classes, "validation_counts": {"plantdoc": 4, "plantvillage": 4},
        "contract_hashes": {f"plant_data_contract/plant_data_contract/{n}": sha(models / "contract" / n) for n in ("taxonomy.py", "transforms.py", "models.py")},
        "selected_solution": {"label": "SCAFFOLD"}})
    return {"tmp": tmp_path, "package": tmp_path / "pkg" / "pc_v3_ket_qua", "models": models, "release": release, "dataset_root": dataset_root,
            "logits": logits, "plan": plan, "val_rows": val_rows}


def build(world, name="b1", **kw):
    return imp.import_package(world["package"], name, world["tmp"] / "bundles" / name, world["release"], world["dataset_root"],
                              logits=world["logits"], **kw)


def test_the_web_accepts_the_13_model_bundle_and_it_is_validation_only(world):
    target = build(world)
    evidence = load_evidence(target)
    assert evidence["available"], evidence.get("reason")
    ids = [m["model_id"] for m in evidence["models"]]
    assert ids == ["centralized", "centralized_sgd", "fedavg", "scaffold", *(f"local_{k:02d}" for k in range(5)),
                   "centralized_int8", "centralized_sgd_int8", "fedavg_int8", "scaffold_int8"]
    by_id = {m["model_id"]: m for m in evidence["models"]}
    assert by_id["fedavg_int8"]["format"] == bundle.FORMAT_TORCHSCRIPT_INT8 and by_id["fedavg_int8"]["fp32_of"] == "fedavg"
    assert "format" not in by_id["fedavg"]
    manifest = bundle.load_manifest(target)
    assert manifest["evidence_scope"] == imp.EVIDENCE_SCOPE == "validation_only" and "training" not in manifest
    assert manifest["release_manifest_sha256"] != sha(world["release"] / "release_manifest.json")      # giữ mã băm (CRLF) mà checkpoint mang
    assert manifest["holdout"]["counts"] == {"plantdoc": 4, "plantvillage": 5} and manifest["shard_total"] == sum(SHARDS)
    assert manifest["w0_in_checkpoints"] is True and manifest["protocol_id"].endswith("_pc_r4_seed42_validation")
    assert not (world["release"] / "manifests" / "calibration.jsonl").exists()


def test_every_directory_carries_the_same_comparison_csv_with_13_models_x_sources(world):
    target = build(world)
    csvs = {sha(p) for p in target.rglob("validation_comparison.csv")}
    assert len(csvs) == 1
    lines = (target / "centralized" / "validation_comparison.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 13 * 2
    assert (target / "fedavg" / "summary.json").is_file() and (target / "centralized" / "summary.json").is_file()


def test_federated_history_comes_from_the_checkpoint_with_client_weights(world):
    target = build(world)
    summary = json.loads((target / "fedavg" / "summary.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in summary["history"]] == [1, 2] and summary["best_round"] == 2
    clients = summary["history"][0]["clients"]
    assert [c["sample_count_n_k"] for c in clients] == SHARDS and abs(sum(c["weight_p_k"] for c in clients) - 1) < 1e-9
    centralized = json.loads((target / "centralized" / "summary.json").read_text(encoding="utf-8"))
    assert len(centralized["history"]) == 3 and centralized["history"][0]["train_loss"] == 2.0


def test_int8_is_evaluated_per_image_and_comparison_report_is_written(world):
    report = world["tmp"] / "report"
    build(world, report_dir=report)
    summary = json.loads((report / "KIEM_LAI_VALIDATION.json").read_text(encoding="utf-8"))
    assert summary["out_of_tolerance"] == 0 and summary["int8_evaluated_per_image"] is True and summary["comparisons"] > 0
    assert (report / "KIEM_LAI_VALIDATION.csv").read_text(encoding="utf-8").startswith("model_id,model,source,count,metric")


def test_collect_logits_runs_int8_one_image_at_a_time(tmp_path):
    from scripts.evaluate_published_models import collect_logits, load_published
    from torchvision import transforms

    rows = []
    for i in range(3):
        Image.new("RGB", (50, 40), (30 * i + 10, 90, 150)).save(tmp_path / f"i{i}.jpg")
        rows.append({"relative_path": f"i{i}.jpg"})
    path = tmp_path / "int8.pt"
    path.write_bytes(_int8_torchscript_bytes())
    model = load_published(path, bundle.FORMAT_TORCHSCRIPT_INT8)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    got = collect_logits(rows, tmp_path, transform, {"m": model}, batch_size=3, workers=0, single_image=frozenset({"m"}))["m"]
    images = torch.stack([transform(Image.open(tmp_path / r["relative_path"]).convert("RGB")) for r in rows])
    with torch.inference_mode():
        expected = np.concatenate([model(images[i:i + 1]).numpy() for i in range(3)])
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("edit", [
    lambda m: m.update(release_manifest_sha256="1" * 64),
    lambda m: m.update(seed=1),
    lambda m: m.update(shard_total=1),
    lambda m: m["models"][0].update(display_name="renamed"),
    lambda m: m["models"][-1].update(fp32_checkpoint_sha256="2" * 64),
])
def test_editing_the_manifest_after_import_is_refused(world, edit):
    target = build(world)
    manifest = json.loads((target / "import_manifest.json").read_text(encoding="utf-8"))
    edit(manifest)
    (target / "import_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_evidence(target)["available"] is False


def test_a_corrupt_package_is_refused_before_anything_is_written(world):
    models = world["models"]
    checkpoint = models / "scaffold" / "checkpoint_best.pt"
    data = bytearray(checkpoint.read_bytes())
    data[-5] ^= 0xFF
    checkpoint.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        build(world)
    assert not (world["tmp"] / "bundles" / "b1").exists() and not (world["tmp"] / "bundles" / ".b1.staging").exists()


def test_a_checkpoint_from_another_release_or_seed_is_refused(world):
    path = world["models"] / "fedavg" / "checkpoint_best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["seed"] = 7
    torch.save(ckpt, path)
    info_path = path.with_name("model_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(sha256=sha(path), bytes=path.stat().st_size)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="release/seed/W0"):
        build(world)


def test_an_int8_file_tied_to_the_wrong_parent_is_refused(world):
    info_path = world["models"] / "int8_dynamic_linear" / "fedavg" / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["parent_fp32_role"] = "scaffold"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="bản gốc"):
        build(world)


def test_import_refuses_an_existing_target_and_a_release_with_other_hash(world):
    build(world)
    with pytest.raises(FileExistsError):
        build(world)
    manifest = json.loads((world["models"] / "import_manifest.json").read_text(encoding="utf-8"))
    manifest["release_manifest_sha256"] = "3" * 64
    (world["models"] / "import_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="release_manifest_sha256"):
        build(world, name="b2")
