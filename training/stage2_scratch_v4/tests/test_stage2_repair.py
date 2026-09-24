"""Regression tests for the 09/09/2026 Stage-2 repair plan."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("flwr")
import torch
import yaml
from PIL import Image

from fl_training.checkpoint import save_round_checkpoint
from fl_training.config import load_training_config
from fl_training.data import build_evaluation_loader, build_training_loader
from fl_training.progress import ProgressRenderer
from fl_training.reporting import build_diagnostics, generate_report, normalize_confusion_matrix
from fl_training.server_app import _parse_and_validate_replies, _select_clients
from fl_training.strategy import aggregate_fedavg_state_dicts

ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "images"
    (root / "class_a").mkdir(parents=True)
    values = np.zeros((260, 300, 3), dtype=np.uint8)
    values[:, :, 0] = np.arange(300, dtype=np.uint8)[None, :]
    values[:, :, 1] = np.arange(260, dtype=np.uint8)[:, None]
    Image.fromarray(values).save(root / "class_a" / "sample.png")
    manifest = tmp_path / "client.csv"
    manifest.write_text(
        "relative_path,label,class_name,client_id,split,group_id,feature_skew,seed\n"
        "class_a/sample.png,0,class_a,0,train,g0,moderate,42\n",
        encoding="utf-8",
    )
    return root, manifest


def test_training_augmentation_is_sample_epoch_deterministic(tmp_path):
    root, manifest = _manifest(tmp_path)
    loader = build_training_loader(manifest, root, batch_size=1, seed=91, client_id=0)
    loader.set_epoch(0)
    first = next(iter(loader))[0]
    loader.set_epoch(0)
    repeat = next(iter(loader))[0]
    loader.set_epoch(7)
    other_epoch = next(iter(loader))[0]
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other_epoch)


def test_feature_profile_changes_train_but_eval_is_clean(tmp_path):
    root, manifest = _manifest(tmp_path)
    none = build_training_loader(
        manifest, root, batch_size=1, seed=42, client_id=0, feature_skew="none"
    )
    moderate = build_training_loader(
        manifest, root, batch_size=1, seed=42, client_id=0, feature_skew="moderate"
    )
    none.set_epoch(0)
    moderate.set_epoch(0)
    assert not torch.equal(next(iter(none))[0], next(iter(moderate))[0])
    clean = build_evaluation_loader(manifest, root, batch_size=1)
    first = next(iter(clean))[0]
    clean.dataset.set_epoch(10)
    assert torch.equal(first, next(iter(clean))[0])


@pytest.mark.parametrize("field,value", [
    ("lr", float("nan")), ("momentum", 1.0), ("num_workers", -1),
])
def test_config_rejects_invalid_training_numbers(tmp_path, field, value):
    with open(ROOT / "configs" / "train_smoke.yaml", "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    raw["training"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_config(path, base_dir=ROOT)


def test_fraction_train_is_deterministic_and_effective():
    cfg = load_training_config(ROOT / "configs" / "train_fedavg.yaml", base_dir=ROOT)
    cfg.federation.fraction_train = 0.5
    assert _select_clients(cfg, 3) == _select_clients(cfg, 3)
    assert len(_select_clients(cfg, 3)) == 5


def _reply(client_id=0, source=10, round_num=2, n_k=3, value=None, has_error=False):
    tensor = np.asarray([1.0], dtype=np.float32) if value is None else value
    metrics = {
        "num-examples": n_k, "processed-examples": n_k, "local-epochs": 1,
        "epoch-1-loss": 1.0, "epoch-1-accuracy": 0.0, "epoch-1-examples": n_k,
        "epoch-1-steps": 1, "epoch-1-lr": 0.01, "epoch-1-duration": 0.1,
        "epoch-1-started": 1.0, "epoch-1-ended": 1.1,
    }
    array = SimpleNamespace(numpy=lambda: tensor)
    return SimpleNamespace(
        metadata=SimpleNamespace(src_node_id=source, group_id=str(round_num)),
        has_error=lambda: has_error,
        error="synthetic client error",
        content={
            "arrays": {"weight": array}, "metrics": metrics,
            "identity": {"client_id": client_id, "round": round_num, "n_k": n_k},
        },
    )


def test_reply_validation_rejects_wrong_round_and_count():
    with pytest.raises(ValueError, match="round"):
        _parse_and_validate_replies([_reply(round_num=1)], {10: 0}, {0: 3}, 2, 1)
    with pytest.raises(ValueError, match="manifest count"):
        _parse_and_validate_replies([_reply(n_k=2)], {10: 0}, {0: 3}, 2, 1)
    with pytest.raises(ValueError, match="Unexpected reply source"):
        _parse_and_validate_replies([_reply(source=11)], {10: 0}, {0: 3}, 2, 1)
    with pytest.raises(RuntimeError, match="returned an error"):
        _parse_and_validate_replies([_reply(has_error=True)], {10: 0}, {0: 3}, 2, 1)


def test_aggregator_rejects_negative_duplicate_and_shape():
    base = {"client_id": 0, "num-examples": 1, "state_dict": {"w": torch.ones(1)}}
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_fedavg_state_dicts([{**base, "num-examples": -1}])
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_fedavg_state_dicts([base, {**base, "num-examples": 2}])
    with pytest.raises(ValueError, match="shape"):
        aggregate_fedavg_state_dicts([base], expected_shapes={"w": (2,)})


def test_progress_keeps_partial_json_line(tmp_path):
    path = tmp_path / "events.jsonl"
    renderer = ProgressRenderer(path, 2, enabled=False, is_tty=False)
    path.write_text('{"type":"round_completed","round":1,"payload":', encoding="utf-8")
    assert renderer.poll() is None
    with open(path, "a", encoding="utf-8") as file:
        file.write('{"val_loss":1,"best_loss":1,"lr":0.1}}\n')
    renderer.poll()
    assert renderer.completed_rounds == 1


def test_snapshot_retention_only_managed_files(tmp_path):
    state = {"w": torch.ones(1)}
    unmanaged = tmp_path / "round_notes.pt"
    unmanaged.write_text("keep", encoding="utf-8")
    for round_num in range(1, 4):
        save_round_checkpoint(
            tmp_path, round_num, state, state, 1, 1.0, 0.1, {}, {}, [],
            {"semantic_config_hash": "x"}, checkpoint_every_n_rounds=1, keep_last_n=2,
        )
    assert not (tmp_path / "round_0001.pt").exists()
    assert (tmp_path / "round_0002.pt").exists()
    assert (tmp_path / "round_0003.pt").exists()
    assert unmanaged.exists()


def test_report_normalization_and_synthetic_diagnostics(tmp_path):
    cm = np.asarray([[2, 0], [0, 0]])
    normalized = normalize_confusion_matrix(cm)
    assert normalized[0].sum() == 1
    assert normalized[1].sum() == 0
    history = pd.DataFrame({"round": [1, 2, 3], "val_loss": [1.0, 1.1, 1.3], "bad_rounds": [1, 2, 3]})
    findings = build_diagnostics(history, {}, [], {"best_round": 0, "last_round": 3})
    ids = {finding["id"] for finding in findings}
    assert {"validation_divergence", "validation_plateau", "no_improvement_over_initialization"} <= ids

    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({
        "round": [1], "train_loss": [1.2], "val_loss": [1.1],
        "train_accuracy": [0.2], "val_accuracy": [0.1], "val_macro_f1": [0.05],
        "lr": [0.01], "bad_rounds": [1],
    }).to_csv(run / "history.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"run_id": "synthetic", "best_round": 0, "last_round": 1}), encoding="utf-8")
    result = generate_report(run)
    assert result["scientific_stage2_complete"] is False
    assert (run / "report" / "learning_curves.png").exists()
