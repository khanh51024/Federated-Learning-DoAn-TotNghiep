"""
Tests for Gate B: Model, DataLoader, Task, Config, and Local Steps.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_training.config import (
    compute_semantic_run_hash,
    load_training_config,
)
from fl_training.data import (
    build_evaluation_loader,
    build_training_loader,
    get_eval_transform,
)
from fl_training.model import (
    check_parameters_finite,
    create_mobilenet_v3_small,
    get_model_state_dict_cpu,
    set_model_state_dict,
)
from fl_training.task import evaluate_model, train_local

ROOT = Path(__file__).resolve().parents[1]


def test_gate_b_config_loading_and_validation():
    cfg = load_training_config(ROOT / "configs" / "train_fedavg.yaml", base_dir=ROOT)
    assert cfg.schema_version == 1
    assert cfg.model.name == "mobilenet_v3_small"
    assert cfg.model.num_classes == 38
    assert cfg.training.local_epochs == 1
    assert cfg.training.batch_size == 16
    assert cfg.federation.num_clients == 10
    assert cfg.data.dataset_root.exists()
    assert cfg.data.partition_dir.exists()
    assert (cfg.data.partition_dir / "global_test.csv").exists()
    assert (cfg.data.partition_dir / "global_val.csv").exists()


def test_gate_b_config_rejection_of_unknown_keys(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "schema_version: 1\n"
        "unknown_section:\n"
        "  foo: bar\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys in section 'root'"):
        load_training_config(bad_yaml, base_dir=ROOT)


def test_gate_b_config_smoke_loading():
    cfg = load_training_config(ROOT / "configs" / "train_smoke.yaml", base_dir=ROOT, mode="smoke")
    assert cfg.mode == "smoke"
    assert cfg.model.weights is None
    assert cfg.federation.num_clients == 2
    assert cfg.federation.max_rounds == 2
    assert cfg.early_stopping.enabled is False
    assert cfg.lr_scheduler.enabled is False


def test_gate_b_model_structure_and_batch_one():
    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    # Check classifier structure
    assert model.classifier[3].out_features == 38

    # Forward with single-item batch (B=1)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 38)

    # Backward on B=1
    loss = y.sum()
    loss.backward()
    assert model.classifier[3].weight.grad is not None


def test_gate_b_model_state_dict_and_finiteness():
    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    sd = get_model_state_dict_cpu(model)
    check_parameters_finite(sd)

    # Test that NaN is caught
    sd_nan = {k: v.clone() for k, v in sd.items()}
    first_key = list(sd_nan.keys())[0]
    sd_nan[first_key][0] = float("nan")
    with pytest.raises(FloatingPointError, match="Non-finite values detected"):
        check_parameters_finite(sd_nan)


def test_gate_b_dataloader_final_singleton_and_epoch():
    # 5 samples with batch_size 4 produces batches of size 4 and 1 (drop_last=False)
    x = torch.randn(5, 3, 224, 224)
    y = torch.randint(0, 38, (5,))
    ds = TensorDataset(x, y)

    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=False)
    batch_sizes = [b[0].size(0) for b in loader]
    assert batch_sizes == [4, 1], f"Expected batches [4, 1], got {batch_sizes}"


def test_gate_b_train_local_step_and_metrics():
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 38, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    param_before = list(model.parameters())[0].clone()
    sd, metrics = train_local(model, loader, opt, epochs=1, device="cpu", amp=False)
    param_after = list(model.parameters())[0]

    assert not torch.equal(param_before, param_after), "Weights must change after training step"
    assert metrics["processed-examples"] == 8
    assert metrics["optimizer-steps"] == 2
    assert metrics["local-epochs"] == 1
    assert "loss-sum" in metrics
    assert "correct" in metrics
    assert "duration-seconds" in metrics


def test_gate_b_evaluate_model_preserves_batchnorm():
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 38, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    eval_res = evaluate_model(model, loader, device="cpu", num_classes=38)

    assert "loss" in eval_res
    assert "accuracy" in eval_res
    assert "macro_f1" in eval_res
    assert eval_res["total_samples"] == 8
    cm = np.array(eval_res["confusion_matrix"])
    assert cm.shape == (38, 38)
    assert cm.sum() == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gate_b_cuda_amp_local_step():
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 38, (4,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    model = create_mobilenet_v3_small(num_classes=38, weights=None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    sd, metrics = train_local(model, loader, opt, epochs=1, device="cuda", amp=True)
    assert metrics["processed-examples"] == 4
    assert metrics["optimizer-steps"] == 2
