"""Bộ gộp kết quả theo cách chia × thuật toán × hạt giống (scripts/eval_sweep_v4.py) và các chỉ số công bằng (scripts/report_client_fairness.py)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import eval_sweep_v4 as sweep
from scripts.report_client_fairness import class_recall, client_accuracy, fairness_summary


# --------------------------------------------------------------------------- công thức
def test_scheme_alpha_reads_every_naming_form():
    assert sweep.scheme_alpha("iid") == (math.inf, None)
    assert sweep.scheme_alpha("label_alpha_0_1") == (0.1, None)
    assert sweep.scheme_alpha("label_alpha_100_0") == (100.0, None)
    assert sweep.scheme_alpha("label_alpha_0_1_p123") == (0.1, 123)
    assert sweep.scheme_alpha("quantity_alpha_0_1") == (None, None)


def test_variant_name_separates_what_makes_a_run_differ_from_fedavg():
    assert sweep.variant_of({"method": "fedavg"}) == "fedavg"
    assert sweep.variant_of({"method": "centralized"}) == "centralized"
    assert sweep.variant_of({"method": "centralized", "optimizer": "adamw"}) == "centralized+adamw"
    assert sweep.variant_of({"method": "fedavg", "algorithm": {"name": "fedprox", "mu": 0.01}}) == "fedprox(mu=0.01)"
    assert sweep.variant_of({"method": "fedavg", "algorithm": {"name": "scaffold", "momentum": 0.9}}) == "scaffold"
    assert sweep.variant_of({"method": "fedavg", "logit_adjust_tau": 1.0}) == "fedavg+la=1.0"
    assert sweep.variant_of({"method": "fedavg", "runtime": "flower"}) == "fedavg+flower"


def test_gap_closure_matches_the_formula_in_the_requirement():
    assert sweep.gap_closure(0.85, 0.80, 0.90) == pytest.approx(0.5)     # (85 − 80) / (90 − 80)
    assert sweep.gap_closure(0.75, 0.80, 0.90) == pytest.approx(-0.5)    # tệ hơn FedAvg thì tỷ lệ âm, không bị cắt về 0
    assert sweep.gap_closure(0.95, 0.80, 0.90) == pytest.approx(1.5)     # vượt cả Centralized
    assert sweep.gap_closure(0.85, 0.90, 0.90) is None                   # FedAvg không thua Centralized: công thức không có nghĩa


def test_rounds_to_convergence_requires_staying_above_the_threshold():
    assert sweep.rounds_to_convergence([0.1, 0.5, 0.80, 0.70, 0.795, 0.80], tolerance=0.01) == 5    # vòng 4 còn tụt nên tính từ vòng 5
    assert sweep.rounds_to_convergence([0.8, 0.8, 0.8], tolerance=0.01) == 1
    assert sweep.rounds_to_convergence([0.1, 0.9, 0.5], tolerance=0.01) is None                     # vòng cuối tụt xuống: chưa hội tụ
    assert sweep.rounds_to_convergence([None, None], tolerance=0.01) is None


def test_communication_uses_logged_bytes_and_falls_back_to_model_size_for_old_runs():
    def row(score: float, logged: bool) -> dict:
        client = {"client": 0, "loss": 1.0, "steps": 1}
        if logged:
            client |= {"bytes_up": 100, "bytes_down": 200}
        return {"val": {"selection_score": score}, "train": {"clients": [client, dict(client)]}, "seconds": 2.0}

    logged = sweep.communication_of([row(0.5, True), row(0.9, True), row(0.9, True)], model_bytes=1000)
    assert logged["bytes_from_history"] is True
    assert logged["rounds_to_convergence"] == 2
    assert logged["mb_up_total"] == pytest.approx(600 / 1e6)
    assert logged["mb_total"] == pytest.approx(1800 / 1e6)
    assert logged["mb_total_to_convergence"] == pytest.approx(1200 / 1e6)
    assert logged["train_seconds"] == pytest.approx(6.0)

    old = sweep.communication_of([row(0.9, False)], model_bytes=1000)
    assert old["bytes_from_history"] is False
    assert old["mb_total"] == pytest.approx(4000 / 1e6)                  # 2 client, mỗi client 1000 byte mỗi chiều


def test_significance_needs_interval_above_zero_and_same_sign_in_every_seed():
    clusters = [str(i) for i in range(400)]
    base = np.zeros(400, dtype=bool)
    better = np.zeros(400, dtype=bool)
    better[:100] = True
    rng = np.random.default_rng(1)
    clear = sweep.compare_to_fedavg({1: better, 2: better, 3: better}, {1: base, 2: base, 3: base}, clusters, rng)
    assert clear["better_significant"] is True
    assert clear["diff_points"] == pytest.approx(25.0)
    assert clear["ci95_points"][0] > 0

    # trung bình dương nhưng một hạt giống ngược dấu: không được kết luận vượt
    worse = np.ones(400, dtype=bool)
    mixed = sweep.compare_to_fedavg({1: better, 2: better, 3: base}, {1: base, 2: base, 3: worse[:400] & (np.arange(400) < 10)}, clusters, rng)
    assert mixed["diff_points_by_seed"][3] < 0
    assert mixed["better_significant"] is False

    tied = sweep.compare_to_fedavg({1: base}, {1: base}, clusters, rng)
    assert tied["better_significant"] is False
    assert sweep.compare_to_fedavg({1: base}, {2: base}, clusters, rng) is None


def test_client_accuracy_weights_recall_by_the_shard_label_distribution():
    targets = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 0, 1, 0, 0, 0])                                  # recall: lớp 0 = 1, lớp 1 = 0,5, lớp 2 = 0
    recall = class_recall(targets, preds, num_classes=4)
    assert recall[:3] == pytest.approx([1.0, 0.5, 0.0])
    assert math.isnan(recall[3])

    assert client_accuracy(recall, np.array([9.0, 1.0, 0.0, 0.0]))["accuracy"] == pytest.approx(0.95)   # 0,9×1 + 0,1×0,5
    assert client_accuracy(recall, np.array([0.0, 0.0, 10.0, 0.0]))["accuracy"] == pytest.approx(0.0)

    # lớp 3 không có ảnh trong tập đo: bỏ và chuẩn hóa lại, phần bị bỏ phải lộ ra ở coverage
    partial = client_accuracy(recall, np.array([5.0, 0.0, 0.0, 5.0]))
    assert partial == {"accuracy": pytest.approx(1.0), "coverage": pytest.approx(0.5)}
    assert client_accuracy(recall, np.array([0.0, 0.0, 0.0, 5.0])) == {"accuracy": None, "coverage": 0.0}


def test_fairness_summary_reports_every_metric_in_the_requirement():
    out = fairness_summary([0.9, 0.5, 0.7], [100, 10, 50])
    assert out["mean"] == pytest.approx(0.7)
    assert out["std_between_clients"] == pytest.approx(np.std([0.9, 0.5, 0.7]))
    assert out["best"] == pytest.approx(0.9)
    assert out["worst"] == pytest.approx(0.5)
    assert out["gap_best_worst"] == pytest.approx(0.4)
    assert out["smallest_client_accuracy"] == pytest.approx(0.5)          # client ít ảnh nhất, không phải client kém nhất
    assert fairness_summary([0.2, 0.8], [1000, 10])["smallest_client_accuracy"] == pytest.approx(0.8)
    assert fairness_summary([None], [1])["mean"] is None


def test_plot_candidate_chooses_by_validation_score_and_skips_cells_below_three_seeds():
    make = lambda val, low=False: {"below_min_seeds": low, "parts": {"p": {}}, "val_selection_score": {"mean": val}}   # noqa: E731
    series = {"fedprox": {"0.1": [make(0.70), make(0.90, low=True), make(0.80)]}}
    assert sweep.plot_candidate(series, "fedprox", 0.1, "p")["val_selection_score"]["mean"] == 0.80
    assert sweep.plot_candidate(series, "fedprox", 0.5, "p") is None


# --------------------------------------------------------------------------- toàn đường, trên release nhỏ dựng tại chỗ
NUM_CLASSES = 38


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(sample_id: str, class_id: int, domain: str = "plantvillage", dataset: str = "PlantVillage") -> dict:
    return {"sample_id": sample_id, "class_id": class_id, "source_domain": domain, "source_dataset": dataset, "group_id": sample_id, "parent_scene_id": sample_id,
            "relative_path": f"images/{sample_id}.jpg"}


@pytest.fixture()
def tiny_release(tmp_path: Path) -> dict:
    """Train 10 ảnh, val 4 ảnh, test 6 ảnh (PlantVillage và PlantDoc giữ riêng); hai cách chia `label_alpha_0_1` và `label_alpha_1_0` chung một phân vùng."""
    root = tmp_path / "dataset"
    release_dir = root / "mixed" / "pv_pd_v4"
    train = [_row(f"tr{i}", i % 2) for i in range(10)]
    val = [_row(f"va{i}", i % 2) for i in range(4)]
    test = [_row(f"te{i}", i % 2) for i in range(4)] + [_row(f"pd{i}", i % 2, "plantdoc", "PlantDoc") for i in range(2)]
    calibration = [_row("ca0", 0)]
    (root / "images").mkdir(parents=True)
    for row in train:
        (root / row["relative_path"]).write_bytes(b"x" * 1000)
    hashes = {}
    for name, rows in (("train", train), ("val", val), ("calibration", calibration), ("legacy_diagnostic_test", test)):
        path = release_dir / "manifests" / f"{name}.jsonl"
        _write_jsonl(path, rows)
        hashes[name] = _sha(path)
    shards = {}
    for scheme in ("label_alpha_0_1", "label_alpha_1_0"):
        shards[scheme] = {}
        for k in range(5):
            path = release_dir / "partitions" / scheme / f"client_{k:02d}.jsonl"
            _write_jsonl(path, train[2 * k:2 * k + 2])
            shards[scheme][path.name] = _sha(path)
    manifest = release_dir / "release_manifest.json"
    manifest.write_text(json.dumps({"release_name": "tiny", "manifest_hashes": hashes, "partition_shard_hashes": shards}), encoding="utf-8")
    return {"root": root, "release_dir": release_dir, "sha": _sha(manifest), "val": val, "test": test, "v3_ids": {r["sample_id"] for r in test}}


RECIPE = {"budget": 2, "local_epochs": 1, "batch_size": 8, "lr": 0.02, "momentum": 0.9, "weight_decay": 1e-4, "label_smoothing": 0.1, "warmup": 1, "train_transform": "t", "train_size": 224,
          "eval_transform": "e", "loss_weighting": "domain_balanced", "channels_last": False, "amp": "fp16", "selection": "s"}


def _run(runs: Path, release: dict, name: str, method: str, seed: int, *, scheme: str | None = None, client: int | None = None, wrong: dict | None = None, **extra) -> None:
    """Run giả: đoán đúng toàn bộ trừ các ảnh trong `wrong` (mỗi split một danh sách chỉ số)."""
    import torch

    wrong = wrong or {}
    run = runs / name
    run.mkdir(parents=True)
    arrays = {}
    for split, rows in (("val", release["val"]), ("legacy_diagnostic_test", release["test"])):
        logits = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
        for i, row in enumerate(rows):
            logits[i, 37 if i in wrong.get(split, ()) else row["class_id"]] = 1.0
        arrays[split] = logits
        arrays[f"{split}_sample_ids"] = np.array([r["sample_id"] for r in rows])
    np.savez_compressed(run / "logits_best.npz", **arrays)
    config = {**RECIPE, "method": method, "seed": seed, "client": client, "scheme": scheme, "limit": 0, "release_manifest_sha256": release["sha"], **extra}
    (run / "final.json").write_text(json.dumps({"config": config, "best_step": 1, "selection_score": 0.9}), encoding="utf-8")
    clients = [{"client": k, "bytes_up": 1000, "bytes_down": 1000} for k in range(5)] if method == "fedavg" else []
    (run / "history.jsonl").write_text("".join(json.dumps({"val": {"selection_score": s}, "train": {"clients": clients}, "seconds": 1.0}) + "\n" for s in (0.5, 0.9)), encoding="utf-8")
    torch.save({"model_state": {"w": torch.zeros(10, dtype=torch.float32)}}, run / "checkpoint_best.pt")
    (run / "DONE").write_text("x", encoding="utf-8")


def test_report_from_runs_on_disk(tmp_path: Path, tiny_release: dict):
    runs = tmp_path / "runs"
    seeds = (42, 123, 2026)
    for seed in seeds:
        _run(runs, tiny_release, f"centralized_s{seed}", "centralized", seed)
        _run(runs, tiny_release, f"fedavg_s{seed}", "fedavg", seed, scheme="label_alpha_0_1", wrong={"legacy_diagnostic_test": [0, 1]})
        _run(runs, tiny_release, f"fedprox_s{seed}", "fedavg", seed, scheme="label_alpha_0_1", wrong={"legacy_diagnostic_test": [0]}, algorithm={"name": "fedprox", "mu": 0.01})
        _run(runs, tiny_release, f"fedavg_s{seed}_a1", "fedavg", seed, scheme="label_alpha_1_0")
        for k in range(5):
            _run(runs, tiny_release, f"local_c{k}_s{seed}", "local", seed, scheme="label_alpha_0_1", client=k, wrong={"legacy_diagnostic_test": [0, 1, 2]})
    _run(runs, tiny_release, "centralized_adamw_s42", "centralized", 42, optimizer="adamw", lr=0.001)
    _run(runs, tiny_release, "fedavg_s42_khac_lr", "fedavg", 42, scheme="label_alpha_0_1", tag="x", lr=0.5)
    _run(runs, tiny_release, "fedavg_s42_thu", "fedavg", 42, scheme="label_alpha_0_1", limit=10)
    _run(runs, tiny_release, "fedavg_s42_release_khac", "fedavg", 42, scheme="label_alpha_0_1", release_manifest_sha256="abc")
    (runs / "dang_chay").mkdir()
    (runs / "dang_chay" / "config.json").write_text("{}", encoding="utf-8")

    report = sweep.build_report([runs], tiny_release["release_dir"], tiny_release["root"], tiny_release["v3_ids"])
    cells = {(c["scheme"], c["variant"]): c for c in report["cells"]}

    reasons = " | ".join(f"{e['run']}: {e['reason']}" for e in report["excluded"])
    for fragment in ("fedavg_s42_khac_lr", "fedavg_s42_thu", "fedavg_s42_release_khac", "dang_chay"):
        assert fragment in reasons
    assert len(report["excluded"]) == 4

    central = cells[(None, "centralized")]
    assert central["seeds"] == list(seeds) and central["below_min_seeds"] is False
    assert central["parts"]["all_test"]["top1"]["mean"] == pytest.approx(1.0)
    assert "gap_top1_points" not in central["parts"]["all_test"]

    # đối chứng AdamW khác công thức là chủ đích: được giữ, ghi rõ khác gì, và bị đánh dấu dưới 3 hạt giống
    adamw = cells[(None, "centralized+adamw")]
    assert adamw["recipe_differences"]["lr"] == (0.02, 0.001)
    assert adamw["below_min_seeds"] is True
    assert adamw["parts"]["all_test"]["gap_top1_points"]["mean"] == pytest.approx(0.0)

    fedavg = cells[("label_alpha_0_1", "fedavg")]
    part = fedavg["parts"]["all_test"]
    assert part["n"] == 6
    assert part["top1"]["mean"] == pytest.approx(4 / 6)
    assert part["gap_top1_points"]["mean"] == pytest.approx(100 * 2 / 6)
    assert fedavg["parts"]["plantdoc_holdout_v3"]["top1"]["mean"] == pytest.approx(1.0)   # hai ảnh sai đều là PlantVillage: PlantDoc phải tách đúng
    assert fedavg["parts"]["plantvillage_test"]["top1"]["mean"] == pytest.approx(0.5)
    assert "gap_closure_top1" not in part and "vs_fedavg" not in part

    prox = cells[("label_alpha_0_1", "fedprox(mu=0.01)")]["parts"]["all_test"]
    assert prox["gap_closure_top1"] == pytest.approx(0.5)                                  # (5/6 − 4/6) / (1 − 4/6)
    assert prox["vs_fedavg"]["diff_points"] == pytest.approx(100 / 6)
    assert prox["vs_fedavg"]["diff_points_by_seed"] == {s: pytest.approx(100 / 6) for s in seeds}

    # Local-only lấy trung bình 5 client trong hạt giống
    local = cells[("label_alpha_0_1", "local")]
    assert local["parts"]["all_test"]["top1"]["mean"] == pytest.approx(3 / 6)
    assert "communication" not in local

    comm = fedavg["communication"]
    assert comm["bytes_from_history"] is True
    assert comm["mb_total"]["mean"] == pytest.approx(2 * 5 * 2000 / 1e6)
    assert comm["rounds_to_convergence"]["mean"] == pytest.approx(2)
    assert comm["raw_images_mb"] == pytest.approx(10 * 1000 / 1e6)
    assert comm["mb_total_vs_raw"] == pytest.approx(2.0)

    # công bằng: FedAvg sai 2 ảnh PlantVillage (te0 lớp 0, te1 lớp 1) trong tập thử; mọi client có đủ hai lớp 0 và 1 với tỉ lệ 1:1
    # recall lớp 0 = 2/3 (te0 sai, te2 và pd0 đúng), lớp 1 = 2/3 → mọi client 2/3, không chênh lệch
    fair = fedavg["fairness"]["legacy_diagnostic_test"]["all"]
    assert fair["mean"]["mean"] == pytest.approx(2 / 3)
    assert fair["gap_best_worst"]["mean"] == pytest.approx(0.0)
    assert fair["coverage_min"] == pytest.approx(1.0)
    assert fedavg["fairness"]["val"]["all"]["mean"]["mean"] == pytest.approx(1.0)
    assert fedavg["fairness"]["n_k"] == [2, 2, 2, 2, 2]

    sweep.write_tables(report, tmp_path)
    table = (tmp_path / "table_main.csv").read_text(encoding="utf-8")
    assert "fedprox(mu=0.01)" in table and "label_alpha_1_0" in table
    assert "smallest_client_%" in (tmp_path / "table_fairness.csv").read_text(encoding="utf-8")

    figures = sweep.plot_figures(report, tmp_path)
    assert figures == ["curves_by_alpha.png", "gap_by_alpha.png"]
    assert (tmp_path / "curves_by_alpha.png").stat().st_size > 0

    # một run không có Centralized làm mốc thì phải dừng, không ra bảng thiếu khoảng cách
    alone = tmp_path / "runs_alone"
    _run(alone, tiny_release, "fedavg_s42", "fedavg", 42, scheme="label_alpha_0_1")
    with pytest.raises(SystemExit, match="Centralized"):
        sweep.build_report([alone], tiny_release["release_dir"], tiny_release["root"], tiny_release["v3_ids"])


def test_personalized_block_uses_each_client_own_model(tmp_path: Path, tiny_release: dict):
    runs = tmp_path / "runs"
    _run(runs, tiny_release, "centralized_s42", "centralized", 42)
    _run(runs, tiny_release, "fedbn_s42", "fedavg", 42, scheme="label_alpha_0_1", algorithm={"name": "fedbn"})
    saved = {}
    for split, rows in (("val", tiny_release["val"]), ("legacy_diagnostic_test", tiny_release["test"])):
        for k in range(5):
            logits = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
            for i, row in enumerate(rows):
                logits[i, 37 if k == 0 else row["class_id"]] = 1.0            # client 0 sai toàn bộ, bốn client còn lại đúng toàn bộ
            saved[f"{split}_client_{k}"] = logits
    np.savez_compressed(runs / "fedbn_s42" / "logits_personalized.npz", **saved)
    report = sweep.build_report([runs], tiny_release["release_dir"], tiny_release["root"], tiny_release["v3_ids"])
    cell = next(c for c in report["cells"] if c["variant"] == "fedbn")
    assert cell["fairness"]["val"]["all"]["gap_best_worst"]["mean"] == pytest.approx(0.0)
    personal = cell["fairness_personalized"]["val"]["all"]
    assert personal["gap_best_worst"]["mean"] == pytest.approx(1.0)
    assert personal["by_client_mean"] == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0])
