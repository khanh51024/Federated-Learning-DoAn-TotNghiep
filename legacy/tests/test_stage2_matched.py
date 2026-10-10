"""Regression tests for the controlled GĐ1/GĐ2 comparison and data boundary."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from stage2_matched.data import (
    CONDITIONS, PROTOCOL, ManifestDataset, assign_clients, semantic_rows,
)
from stage2_matched.experiment import make_job, paired_delta, trainer_contract
from stage1_compat.config import JobConfig
from stage1_compat.data import EVAL_TRANSFORM
from src.data.transforms import create_deterministic_client_profile


@pytest.fixture
def rows():
    return [{"relative_path": f"c{c}/g{g}_{i}.png", "label": c, "class_name": f"c{c}",
             "group_id": f"c{c}_g{g}", "domain_id": -1}
            for c in range(38) for g in range(30) for i in range(2)]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_whole_groups_conservation_reproducibility(rows, condition):
    clients, diagnostic = assign_clients(rows, CONDITIONS[condition])
    again, _ = assign_clients(rows, CONDITIONS[condition])
    assert clients == again and diagnostic["conserved"]
    assert min(map(len, clients.values())) >= 10
    seen, ownership = set(), {}
    for cid, items in clients.items():
        for row in items:
            assert row["relative_path"] not in seen
            seen.add(row["relative_path"])
            assert ownership.setdefault(row["group_id"], cid) == cid
    assert seen == {r["relative_path"] for r in rows}
    assert all(r["domain_id"] == -1 for r in rows)  # no mutation of shared pool


def test_feature_dirichlet_changes_domain_allocation_not_domain_or_label(rows):
    controls, _ = assign_clients(rows, CONDITIONS["feature100"])
    skew, _ = assign_clients(rows, CONDITIONS["feature01"])
    flat = lambda clients: [r for items in clients.values() for r in items]
    assert semantic_rows(flat(controls), True) == semantic_rows(flat(skew), True)
    assert semantic_rows(flat(skew)) == semantic_rows(rows)
    def imbalance(clients):
        return np.mean([np.abs(np.bincount([r["domain_id"] for r in items], minlength=5) / len(items) - .2).sum()
                        for items in clients.values()])
    assert imbalance(skew) > imbalance(controls) + .3


def test_stage1_trainer_is_reused_without_hyperparameter_drift():
    original = trainer_contract(JobConfig(job_id="historical"))
    for name in CONDITIONS:
        assert trainer_contract(make_job(name, 42)) == original
    smoke = make_job("label01", 42, True)
    assert smoke.rounds == 2 and not smoke.pretrained and smoke.num_clients == 5


def fake_result(condition="label100", accuracy=.99):
    job = make_job(condition, 42)
    return {"resolved_config": job.to_dict(), "test_metrics": {"accuracy": accuracy, "macro_f1": accuracy},
            "identity": {"seed": 42, "runtime": {"torch": "test"}, "context": {
                "scope": PROTOCOL, "smoke": False, "condition": condition, "common": {"test": "same"},
                "matched_source_sha256": "source", "matched_suite_sha256": "suite",
                "initialization_sha256": "initialization", "train_device": "cpu", "evaluation_device": "cpu"}}}


def test_high_scores_and_positive_delta_are_not_suppressed():
    delta = paired_delta(fake_result(), fake_result("label01", 1.0))
    assert delta["accuracy_delta_pp"] == pytest.approx(1.0)


@pytest.mark.parametrize("key,value", [("scope", "stage1_compat"), ("smoke", True),
    ("common", {"test": "different"}), ("initialization_sha256", "different"),
    ("matched_source_sha256", "different"), ("train_device", "cuda")])
def test_refuse_unmatched_comparison(key, value):
    target = fake_result("label01")
    target["identity"]["context"][key] = value
    with pytest.raises(ValueError):
        paired_delta(fake_result(), target)


def test_refuse_changed_optimizer():
    target = fake_result("label01")
    target["resolved_config"]["optimizer"] = "SGD"
    with pytest.raises(ValueError):
        paired_delta(fake_result(), target)


def test_eval_is_clean_and_train_applies_domain(tmp_path, monkeypatch):
    from stage2_matched import data
    array = np.tile(np.arange(64, 192, dtype=np.uint8), (128, 1))
    image = Image.fromarray(np.stack([array] * 3, axis=-1))
    image.save(tmp_path / "sample.png")
    row = {"relative_path": "sample.png", "label": 3, "domain_id": 0}
    profile = create_deterministic_client_profile(0, 42, "moderate").to_dict()
    eval_data = ManifestDataset([row], tmp_path, profiles=[profile])
    assert torch.equal(eval_data[0][0], EVAL_TRANSFORM(image))
    monkeypatch.setattr(data, "TRAIN_TRANSFORM", EVAL_TRANSFORM)
    train = ManifestDataset([row], tmp_path, True, [profile])
    assert not torch.equal(train[0][0], eval_data[0][0])
    assert torch.equal(train[0][0], train[0][0])
    assert train[0][1] == 3


def test_source_fingerprint_includes_upstream_and_adapter(monkeypatch, tmp_path):
    from stage2_matched import experiment
    for folder in ("stage2_matched", "stage1_compat/upstream", "src/data", "fl_training"):
        path = tmp_path / folder
        path.mkdir(parents=True)
        (path / "test.py").write_text("original")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    previous = experiment.source_hash()
    for relative in ("stage1_compat/upstream/test.py", "stage2_matched/test.py"):
        (tmp_path / relative).write_text("changed")
        current = experiment.source_hash()
        assert previous != current
        previous = current


@pytest.fixture
def frozen_suite(tmp_path, monkeypatch):
    from stage2_matched import data
    samples = {}
    for split in ("train", "val", "test"):
        samples[split] = [{"relative_path": f"c{c}/{split}{g}_{i}.png", "label": c, "class_name": f"c{c}",
                           "group_id": f"{split}_{c}_{g}", "leaf_group_id": f"{split}_{c}_{g}"}
                          for c in range(38) for g in range(5) for i in range(2)]
    class Splitter:
        class_names = [f"c{c}" for c in range(38)]
        leaf_audit = {"scope": "unit_test"}
        def __init__(self, *args, **kwargs):
            pass
        def scan_dataset(self):
            pass
        def split_global_test(self):
            return deepcopy(samples["train"]), deepcopy(samples["test"]), deepcopy(samples["val"])
    monkeypatch.setattr(data, "DatasetPartitioner", Splitter)
    images = {r["relative_path"]: {"sha256": data.digest(r["relative_path"])}
              for rows in samples.values() for r in rows}
    monkeypatch.setattr(data, "build_or_update_source_content_cache",
                        lambda *args, **kwargs: {"images": images, "total_images": len(images)})
    leaf_map = tmp_path / "leaf.json"
    leaf_map.write_text("{}")
    root = tmp_path / "suite"
    data.prepare(root, tmp_path, leaf_map, smoke=True)
    data.preflight(root)
    return root


def repin(root, name):
    """Exercise semantic checks even when a modified file hash is repinned."""
    from stage2_matched.data import manifest_hash
    path = root / "suite.json"
    suite = json.loads(path.read_text())
    suite["conditions"][name]["manifest_sha256"] = manifest_hash(root / name)
    path.write_text(json.dumps(suite))


def test_frozen_file_tamper_rejected(frozen_suite):
    from stage2_matched.data import preflight
    path = frozen_suite / "label100/global_val.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest or condition changed"):
        preflight(frozen_suite)


def test_client_domain_must_match_central_manifest(frozen_suite):
    from stage2_matched.data import preflight, read_rows, write_rows
    path = frozen_suite / "feature01/clients/client_00.csv"
    rows = read_rows(path)
    rows[0]["domain_id"] = (rows[0]["domain_id"] + 1) % 5
    write_rows(path, rows)
    repin(frozen_suite, "feature01")
    with pytest.raises(ValueError, match="domain assignments differ"):
        preflight(frozen_suite)


def test_holdout_must_not_receive_feature_transform(frozen_suite):
    from stage2_matched.data import preflight, read_rows, write_rows
    path = frozen_suite / "feature01/global_val.csv"
    rows = read_rows(path)
    rows[0]["domain_id"] = 0
    write_rows(path, rows)
    repin(frozen_suite, "feature01")
    with pytest.raises(ValueError, match="Holdouts must stay clean"):
        preflight(frozen_suite)


def test_holdout_identity_includes_groups(frozen_suite):
    from stage2_matched.data import preflight, read_rows, write_rows
    path = frozen_suite / "label1/global_test.csv"
    rows = read_rows(path)
    rows[0]["group_id"] = "unknown-new-group"
    write_rows(path, rows)
    repin(frozen_suite, "label1")
    with pytest.raises(ValueError, match="identical train pool, holdouts"):
        preflight(frozen_suite)
