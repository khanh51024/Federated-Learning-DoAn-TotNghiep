"""Tập kiểm thử niêm phong (scripts/run_sealed_test.py): khóa một lần, kiểm trước không đọc dữ liệu, không đường nào khác đọc số của split này."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from scripts import eval_sweep_v4 as sweep
from scripts import run_sealed_test as sealed

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 38
SEEDS = (42, 123, 2026)


def _row(sample_id: str, class_id: int, kind: str = "lab", trust: str | None = None) -> dict:
    domain, dataset = {"lab": ("plantvillage", "PlantVillage"), "pd": ("plantdoc", "PlantDoc"), "field": ("plantdoc", "mendeley_x")}[kind]
    return {"sample_id": sample_id, "class_id": class_id, "source_domain": domain, "source_dataset": dataset, "group_id": sample_id, "parent_scene_id": sample_id, "label_trust": trust}


@pytest.fixture()
def release(tmp_path: Path) -> dict:
    release_dir = tmp_path / "release"
    calibration = [_row(f"c{i}", i % 2) for i in range(4)] + [_row("cp0", 0, "pd"), _row("cp1", 1, "pd", "reviewed_quarantine"), _row("cf0", 0, "field"), _row("cf1", 1, "field")]
    test = [_row(f"t{i}", i % 2) for i in range(2)] + [_row("tp0", 0, "pd"), _row("tp1", 1, "pd")]
    val = [_row("v0", 0)]
    hashes = {}
    (release_dir / "manifests").mkdir(parents=True)
    for name, rows in (("train", val), ("val", val), ("calibration", calibration), ("legacy_diagnostic_test", test)):
        path = release_dir / "manifests" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = release_dir / "release_manifest.json"
    manifest.write_text(json.dumps({"release_name": "tiny", "manifest_hashes": hashes, "partition_shard_hashes": {}}), encoding="utf-8")
    return {"dir": release_dir, "sha": hashlib.sha256(manifest.read_bytes()).hexdigest(), "calibration": calibration, "test": test, "val": val}


def _run(runs: Path, release: dict, name: str, method: str, seed: int, *, scheme: str | None = None, client: int | None = None, wrong: tuple[str, ...] = (), **extra) -> None:
    run = runs / name
    run.mkdir(parents=True)
    arrays = {}
    for split, rows in (("val", release["val"]), ("calibration", release["calibration"]), ("legacy_diagnostic_test", release["test"])):
        logits = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
        for i, row in enumerate(rows):
            logits[i, 37 if row["sample_id"] in wrong else row["class_id"]] = 1.0
        arrays[split] = logits
        arrays[f"{split}_sample_ids"] = np.array([r["sample_id"] for r in rows])
    np.savez_compressed(run / "logits_best.npz", **arrays)
    config = {"method": method, "seed": seed, "client": client, "scheme": scheme, "limit": 0, "release_manifest_sha256": release["sha"], **extra}
    (run / "final.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    (run / "DONE").write_text("x", encoding="utf-8")


def _full_matrix(runs: Path, release: dict) -> None:
    for seed in SEEDS:
        _run(runs, release, f"centralized_s{seed}", "centralized", seed)
        _run(runs, release, f"fedavg_s{seed}", "fedavg", seed, scheme="a", wrong=("c0", "c1", "cp0", "tp0"))
        _run(runs, release, f"prox_s{seed}", "fedavg", seed, scheme="a", wrong=("c0",), algorithm={"name": "fedprox", "mu": 0.01})
        for k in range(5):
            _run(runs, release, f"local_c{k}_s{seed}", "local", seed, scheme="a", client=k, wrong=("c0", "c1", "c2", "c3"))


def test_runs_once_and_refuses_a_second_time(tmp_path: Path, release: dict):
    runs = tmp_path / "runs"
    _full_matrix(runs, release)
    result = tmp_path / "out" / "sealed.json"
    report = sealed.run_sealed([runs], release["dir"], ["a|fedprox(mu=0.01)"], result, dry_run=False)
    assert result.is_file() and report["sealed_manifest_sha256"] == hashlib.sha256((release["dir"] / "manifests" / "calibration.jsonl").read_bytes()).hexdigest()
    before = result.read_bytes()
    with pytest.raises(SystemExit, match="đã có"):
        sealed.run_sealed([runs], release["dir"], ["a|fedprox(mu=0.01)"], result, dry_run=False)
    with pytest.raises(SystemExit, match="đã có"):
        sealed.run_sealed([runs], release["dir"], ["a|fedavg"], result, dry_run=True)
    assert result.read_bytes() == before


def test_numbers_split_by_source_and_include_all_three_baselines(tmp_path: Path, release: dict):
    runs = tmp_path / "runs"
    _full_matrix(runs, release)
    report = sealed.run_sealed([runs], release["dir"], ["a|fedprox(mu=0.01)"], tmp_path / "s.json", dry_run=False)
    cells = {(c["scheme"], c["variant"]): c for c in report["cells"]}
    assert set(cells) == {(None, "centralized"), ("a", "fedavg"), ("a", "local"), ("a", "fedprox(mu=0.01)")}

    assert cells[(None, "centralized")]["parts"]["sealed_all"]["top1"]["mean"] == pytest.approx(1.0)
    assert "gap_top1_points" not in cells[(None, "centralized")]["parts"]["sealed_all"]

    fedavg = cells[("a", "fedavg")]["parts"]
    assert {name: fedavg[name]["n"] for name in fedavg} == {"sealed_all": 8, "plantvillage": 4, "plantdoc": 2, "plantdoc_without_reviewed_quarantine": 1, "field_other": 2, "plantdoc_extended": 4}
    assert fedavg["sealed_all"]["top1"]["mean"] == pytest.approx(5 / 8)
    assert fedavg["plantvillage"]["top1"]["mean"] == pytest.approx(0.5)
    assert fedavg["plantdoc"]["top1"]["mean"] == pytest.approx(0.5)
    assert fedavg["field_other"]["top1"]["mean"] == pytest.approx(1.0)
    assert fedavg["plantdoc_extended"]["top1"]["mean"] == pytest.approx(0.5)            # cp0 và tp0 sai trên 4 ảnh PlantDoc của cả hai tập
    assert fedavg["sealed_all"]["gap_top1_points"]["mean"] == pytest.approx(100 * 3 / 8)
    assert "vs_fedavg" not in fedavg["sealed_all"]

    prox = cells[("a", "fedprox(mu=0.01)")]["parts"]["sealed_all"]
    assert prox["vs_fedavg"]["diff_points"] == pytest.approx(100 * 2 / 8)
    assert cells[("a", "local")]["parts"]["plantvillage"]["top1"]["mean"] == pytest.approx(0.0)  # trung bình 5 client


def test_dry_run_checks_without_opening_logits_or_writing_a_result(tmp_path: Path, release: dict):
    runs = tmp_path / "runs"
    _full_matrix(runs, release)
    for run in runs.iterdir():
        (run / "logits_best.npz").unlink()                                              # kiểm trước không được cần logit
    result = tmp_path / "s.json"
    report = sealed.run_sealed([runs], release["dir"], ["a|fedprox(mu=0.01)"], result, dry_run=True)
    assert report["dry_run"] is True and not result.exists()
    assert len(report["cells"]["a|local"]) == 15


def test_incomplete_matrix_stops_before_reading_anything(tmp_path: Path, release: dict):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        _run(runs, release, f"centralized_s{seed}", "centralized", seed)
        _run(runs, release, f"fedavg_s{seed}", "fedavg", seed, scheme="a")
    for k in range(4):                                                                  # Local-only thiếu một client, thiếu hạt giống
        _run(runs, release, f"local_c{k}_s42", "local", 42, scheme="a", client=k)
    _run(runs, release, "prox_s42", "fedavg", 42, scheme="a", algorithm={"name": "fedprox", "mu": 0.01})
    result = tmp_path / "s.json"
    with pytest.raises(SystemExit, match="chỉ có 1 hạt giống"):
        sealed.run_sealed([runs], release["dir"], ["a|fedprox(mu=0.01)"], result, dry_run=False)
    assert not result.exists()
    with pytest.raises(SystemExit, match="dạng 'cách chia"):
        sealed.plan_cells([], ["khong_co_gach"])


def test_trial_runs_and_other_releases_never_count(tmp_path: Path, release: dict):
    runs = tmp_path / "runs"
    _run(runs, release, "centralized_s42", "centralized", 42, limit=10)
    _run(runs, release, "centralized_s123", "centralized", 123, release_manifest_sha256="khac")
    _run(runs, release, "centralized_s2026", "centralized", 2026)
    assert [r["name"] for r in sealed.find_runs([runs], release["sha"])] == ["centralized_s2026"]


def test_no_other_path_reports_accuracy_on_the_sealed_split():
    """Mọi đường khác chỉ đọc val và tập thử; train_v4 không tóm tắt split này vào final.json."""
    assert sealed.SEALED_SPLIT not in sweep.FAIRNESS_SPLITS
    source = (ROOT / "scripts" / "eval_sweep_v4.py").read_text(encoding="utf-8")
    assert '"calibration"' not in source
    train = (ROOT / "scripts" / "train_v4.py").read_text(encoding="utf-8")
    assert re.search(r"summarize\(logits\[name\], eval_rows\[name\]\) for name in eval_rows if name != SEALED_SPLIT", train)
    assert 'SEALED_SPLIT = "calibration"' in train
