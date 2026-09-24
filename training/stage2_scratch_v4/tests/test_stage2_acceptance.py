"""Independent numerical and artifact/protocol regression checks for Stage 2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
pytest.importorskip("flwr")
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from fl_training.checkpoint import model_state_sha256, save_round_checkpoint, load_checkpoint
from fl_training.config import load_training_config, compute_protocol_fingerprint
from fl_training.evaluation_artifacts import bind_evaluation, validate_evaluation_partition, write_evaluation
from fl_training.model import create_mobilenet_v3_small
from fl_training.server_app import _parse_and_validate_replies
from fl_training.strategy import aggregate_fedavg_state_dicts
from fl_training.sweep import _latest_run, _write_comparison, run_sweep
from fl_training.task import evaluate_model
from tests.test_stage2_repair import _reply

ROOT = Path(__file__).resolve().parents[1]


def test_all_mobilenet_tensors_match_independent_fedavg_oracle(tmp_path):
    reference = create_mobilenet_v3_small(38, None).state_dict()
    generator = torch.Generator().manual_seed(913)
    replies = []
    for cid, count in enumerate((11, 23, 47)):
        state = {}
        for key, tensor in reference.items():
            state[key] = (torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
                          if tensor.is_floating_point() else torch.full_like(tensor, cid + 2))
        replies.append({"client_id": cid, "num-examples": count, "state_dict": state})
    input_hashes = [model_state_sha256(r["state_dict"]) for r in replies]
    actual, metrics = aggregate_fedavg_state_dicts(
        list(reversed(replies)), {k: tuple(v.shape) for k, v in reference.items()},
        {k: v.dtype for k, v in reference.items()},
    )
    assert set(actual) == set(reference)
    assert metrics["num-examples"] == 81
    maximum = 0.0
    for key, tensor in actual.items():
        assert tensor.shape == reference[key].shape and tensor.dtype == reference[key].dtype
        assert tensor.device.type == "cpu"
        if tensor.is_floating_point():
            # NumPy float64 oracle: no production aggregator helper is reused.
            expected = sum(r["state_dict"][key].numpy().astype(np.float64) * r["num-examples"]
                           for r in replies) / 81
            np.testing.assert_allclose(tensor.numpy(), expected, rtol=2e-6, atol=5e-7)
            maximum = max(maximum, float(np.max(np.abs(tensor.numpy() - expected))))
        else:
            assert torch.equal(tensor, torch.full_like(tensor, 4))
    assert input_hashes == [model_state_sha256(r["state_dict"]) for r in replies]
    save_round_checkpoint(tmp_path, 1, actual, actual, 1, 1.0, .01, {}, {}, [],
                          {"semantic_config_hash": "oracle"}, is_best=True)
    restored = load_checkpoint(tmp_path / "last.pt")["model_state_dict"]
    assert model_state_sha256(restored) == model_state_sha256(actual)
    (tmp_path / "oracle.json").write_text(json.dumps({
        "tensor_entries": len(actual), "floating_elements": sum(t.numel() for t in actual.values() if t.is_floating_point()),
        "counts": [11, 23, 47], "sum_n_k": 81, "max_abs_error": maximum,
        "checkpoint_exact": True,
    }, indent=2), encoding="utf-8")


def test_numpy_float64_and_implicit_schema():
    reply = lambda cid, v: {"client_id": cid, "num-examples": cid + 1, "state_dict": {"w": v}}
    output, _ = aggregate_fedavg_state_dicts([
        reply(0, np.array([1 + 1e-10], dtype=np.float64)),
        reply(1, np.array([1 + 4e-10], dtype=np.float64)),
    ])
    assert output["w"].dtype == torch.float64
    assert output["w"].item() == pytest.approx(1 + 3e-10, abs=1e-15)
    with pytest.raises(ValueError, match="shape"):
        aggregate_fedavg_state_dicts([reply(0, torch.ones(1)), reply(1, torch.ones(2))])
    with pytest.raises(ValueError, match="dtype"):
        aggregate_fedavg_state_dicts([reply(0, torch.ones(1)), reply(1, torch.ones(1, dtype=torch.float64))])
    with pytest.raises(ValueError, match="empty model"):
        aggregate_fedavg_state_dicts([{"client_id": 0, "num-examples": 1, "state_dict": {}}])


def valid_reply():
    reply = _reply()
    reply.content["metrics"].update({"loss-sum": 3.0, "correct": 0, "optimizer-steps": 1})
    return reply


@pytest.mark.parametrize("key,value", [
    ("num-examples", 3.7), ("loss-sum", float("nan")), ("correct", 4),
    ("epoch-1-loss", float("inf")), ("epoch-1-accuracy", 1.5),
    ("epoch-1-examples", 2), ("epoch-1-steps", 2), ("local-epochs", 2),
])
def test_reply_metrics_cannot_be_truncated_or_inconsistent(key, value):
    reply = valid_reply()
    reply.content["metrics"][key] = value
    with pytest.raises(ValueError):
        _parse_and_validate_replies([reply], {10: 0}, {0: 3}, 2, 1, batch_size=3)


def test_valid_reply_metrics():
    parsed = _parse_and_validate_replies([valid_reply()], {10: 0}, {0: 3}, 2, 1, batch_size=3)
    assert parsed[0]["loss-sum"] == 3


def test_metrics_match_sklearn_including_absent_class():
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 0, 1, 2, 0, 2])
    logits = torch.nn.functional.one_hot(predictions, num_classes=4).float() * 3
    result = evaluate_model(torch.nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=3), num_classes=4)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(4), zero_division=0)
    assert result["accuracy"] == accuracy_score(labels, predictions)
    assert result["macro_f1"] == pytest.approx(f1.mean())
    assert result["macro_precision"] == pytest.approx(precision.mean())
    assert result["macro_recall"] == pytest.approx(recall.mean())
    assert result["weighted_f1"] == pytest.approx(np.average(f1, weights=support))
    assert result["loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    np.testing.assert_array_equal(result["confusion_matrix"], confusion_matrix(labels, predictions, labels=range(4)))
    assert result["classes_with_support"] == 3
    bad = logits.clone(); bad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        evaluate_model(torch.nn.Identity(), DataLoader(TensorDataset(bad, labels), batch_size=3), num_classes=4)


def test_evaluations_keep_model_identity_and_reject_changed_partition(tmp_path):
    manifest = tmp_path / "global_test.csv"
    manifest.write_text("relative_path,label\na,0\n", encoding="utf-8")
    fp = compute_protocol_fingerprint(tmp_path)
    payload = {"model_state_dict": {"w": torch.ones(2)}, "round": 2,
               "config": {"data": {"protocol_fingerprint": fp}, "run_id": "test"}}
    validate_evaluation_partition(payload, tmp_path)
    original = {"accuracy": 1.0, "confusion_matrix": [[1]]}
    first = bind_evaluation(copy.deepcopy(original), payload, tmp_path / "best.pt", manifest, fp)
    write_evaluation(first, tmp_path / "out")
    payload["model_state_dict"]["w"].add_(1)
    second = bind_evaluation(copy.deepcopy(original), payload, tmp_path / "last.pt", manifest, fp)
    write_evaluation(second, tmp_path / "out")
    assert first["evaluation_id"] != second["evaluation_id"]
    assert len(list((tmp_path / "out" / "evaluations").iterdir())) == 2
    assert json.loads((tmp_path / "out" / "test_metrics.json").read_text())["checkpoint"] == "last.pt"
    alias = bind_evaluation(copy.deepcopy(original), payload, tmp_path / "model_final.pt", manifest, fp)
    write_evaluation(alias, tmp_path / "out")
    assert alias["model_state_sha256"] == second["model_state_sha256"]
    assert alias["evaluation_id"] != second["evaluation_id"]
    assert len(list((tmp_path / "out" / "evaluations").iterdir())) == 3
    manifest.write_text("relative_path,label\nb,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_evaluation_partition(payload, tmp_path)


def test_sweep_collection_filters_seed_protocol_and_failure(tmp_path):
    for name, seed, fp, status in (("correct", 42, "p", "completed"), ("wrong_seed", 43, "p", "completed"),
                                   ("wrong_protocol", 42, "q", "completed"), ("failed", 42, "p", "failed")):
        run = tmp_path / name; run.mkdir()
        (run / "summary.json").write_text(json.dumps({"status": status}))
        (run / "resolved_config.yaml").write_text(yaml.safe_dump({"training": {"seed": seed}, "data": {"protocol_fingerprint": fp}}))
    assert _latest_run(tmp_path, 42, "p").name == "correct"
    with pytest.raises(RuntimeError):
        _latest_run(tmp_path, 99, "p")


def test_sweep_dry_run_preserves_comparison_and_split_seed(tmp_path):
    spec = {"base_config": str(ROOT / "configs/train_smoke.yaml"), "output_root": str(tmp_path),
            "sweep_id": "dry", "modes": ["fedavg"], "seeds": [42, 43],
            "conditions": [{"scenario": "label_skew", "alpha": .1, "feature_skew": "none"}]}
    path = tmp_path / "spec.yaml"; path.write_text(yaml.safe_dump(spec))
    out = tmp_path / "dry"; out.mkdir()
    comparison = out / "comparison.csv"; comparison.write_text("existing-results")
    run_sweep(path)
    assert comparison.read_text() == "existing-results"
    for cfg_path in (out / "configs").glob("*.yaml"):
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["data"]["split_seed"] == 42
        assert f"seed-{cfg['training']['seed']}" in cfg["output"]["root"]


@pytest.mark.parametrize("field,value", [("local_epochs", 1.5), ("amp", "false")])
def test_config_rejects_coerced_types(tmp_path, field, value):
    raw = yaml.safe_load((ROOT / "configs/train_smoke.yaml").read_text())
    raw["training"][field] = value
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_training_config(path, base_dir=ROOT)


@pytest.mark.parametrize("fail_checkpoint", [False, True])
def test_server_early_stop_and_failed_checkpoint_do_not_publish_uncommitted_round(tmp_path, monkeypatch, fail_checkpoint):
    import fl_training.server_app as server
    from fl_training.progress import EventLogger

    cfg = load_training_config(ROOT / "configs/train_smoke.yaml", base_dir=ROOT)
    cfg.early_stopping.enabled = True
    cfg.early_stopping.patience_rounds = 1
    cfg.early_stopping.warmup_rounds = 0
    cfg.federation.max_rounds = 4
    monkeypatch.delenv("FL_TRAINING_RESUME_CHECKPOINT", raising=False)
    monkeypatch.setattr(server, "create_mobilenet_v3_small", lambda **kwargs: torch.nn.Linear(1, 38))
    monkeypatch.setattr(server, "build_evaluation_loader", lambda *a, **kw: None)
    monkeypatch.setattr(server, "get_client_sample_count", lambda path, cid: 3)
    monkeypatch.setattr(server, "evaluate_model", lambda *a: {"loss": 1., "accuracy": 0., "macro_f1": 0.})

    class Grid:
        calls = 0
        def get_node_ids(self):
            return [10, 11]
        def send_and_receive(self, messages, timeout):
            result = []
            for msg in messages:
                node = msg.metadata.dst_node_id
                if msg.metadata.message_type == "query":
                    result.append(SimpleNamespace(has_error=lambda: False,
                        metadata=SimpleNamespace(src_node_id=node),
                        content={"identity": {"node_id": node, "partition_id": node - 10}}))
                else:
                    self.calls += 1
                    reply = valid_reply()
                    reply.metadata.src_node_id = node
                    reply.metadata.group_id = msg.metadata.group_id
                    reply.content["identity"].update(client_id=node - 10, round=int(msg.metadata.group_id))
                    reply.content["arrays"] = msg.content["arrays"]
                    result.append(reply)
            return result

    grid = Grid()
    if fail_checkpoint:
        real_save = server.save_round_checkpoint
        def broken(*args, **kwargs):
            if args[1] == 1:
                raise OSError("injected disk failure")
            return real_save(*args, **kwargs)
        monkeypatch.setattr(server, "save_round_checkpoint", broken)
        with pytest.raises(OSError, match="disk failure"):
            server._run(grid, None, cfg, tmp_path, EventLogger(tmp_path / "events.jsonl", cfg.run_id))
        assert load_checkpoint(tmp_path / "last.pt")["round"] == 0
        assert not (tmp_path / "history.csv").exists()
        assert not (tmp_path / "weights.jsonl").exists()
    else:
        server._run(grid, None, cfg, tmp_path, EventLogger(tmp_path / "events.jsonl", cfg.run_id))
        assert grid.calls == 2  # only one round, two clients
        assert load_checkpoint(tmp_path / "last.pt")["round"] == 1
        assert len(pd.read_csv(tmp_path / "history.csv")) == 1


def test_manifest_audit_rejects_duplicate_sample_weight_inflation(tmp_path):
    import shutil
    from fl_training.prepare import audit_manifest_directory
    bundle = tmp_path / "bundle"
    shutil.copytree(ROOT / "data/partitions_train_v1/smoke_bundle_2c", bundle)
    client = bundle / "clients/client_00.csv"
    rows = client.read_text(encoding="utf-8").splitlines()
    client.write_text("\n".join(rows + [rows[1]]) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="n_k would be inflated"):
        audit_manifest_directory(bundle, require_all_classes=False)


def test_single_tty_bar_handles_baseline_epoch_and_phases(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import fl_training.progress as progress
    factory = MagicMock()
    monkeypatch.setattr(progress, "tqdm", factory)
    events = tmp_path / "events.jsonl"
    logger = progress.EventLogger(events, "acceptance")
    renderer = progress.ProgressRenderer(events, 2, is_tty=True)
    logger.log_phase(1, "training")
    logger.log_client_epoch({"round": 1, "client_id": 0, "local_epoch": 1, "loss": 1.0})
    renderer.poll()
    assert "r1 c0 e1" in factory.return_value.set_description.call_args.args[0]
    for phase in ("aggregate", "validation", "checkpoint", "evaluate", "plot"):
        logger.log_phase(1, phase)
        renderer.poll()
        assert phase in factory.return_value.set_description.call_args.args[0]
    renderer.close()
    factory.assert_called_once()
    factory.return_value.close.assert_called_once()


def test_reply_and_diagnostics_account_for_amp_skips():
    from fl_training.reporting import build_diagnostics
    reply = valid_reply()
    reply.content["metrics"].update({"optimizer-steps": 0, "skipped-optimizer-steps": 1,
                                     "epoch-1-steps": 0, "epoch-1-skipped-steps": 1})
    parsed = _parse_and_validate_replies([reply], {10: 0}, {0: 3}, 2, 1, batch_size=3)
    assert parsed[0]["optimizer-steps"] == 0
    assert parsed[0]["skipped-optimizer-steps"] == 1
    findings = build_diagnostics(pd.DataFrame([{"skipped_optimizer_steps": 1}]), {}, [], {})
    assert any(item["id"] == "amp_skipped_steps" for item in findings)
    reply.content["metrics"]["skipped-optimizer-steps"] = 2
    with pytest.raises(ValueError, match="skipped steps differ"):
        _parse_and_validate_replies([reply], {10: 0}, {0: 3}, 2, 1, batch_size=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real GradScaler skip")
def test_cuda_amp_counts_real_skipped_sgd_updates():
    from fl_training.task import train_local
    from fl_training.reproducibility import seed_everything
    seed_everything(42)
    model = torch.nn.Linear(2, 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    calls = []
    handle = optimizer.register_step_post_hook(lambda *args: calls.append(1))
    # Finite logits/loss, deliberately non-finite backward gradient.
    gradient_hook = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    try:
        state, metrics = train_local(model, DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long)), batch_size=2),
                                     optimizer, 1, "cuda", amp=True)
        assert metrics["optimizer-steps"] == len(calls) == 0
        assert metrics["skipped-optimizer-steps"] == 2
        assert all(torch.equal(value, initial[key].cpu()) for key, value in state.items())
    finally:
        handle.remove()
        gradient_hook.remove()
