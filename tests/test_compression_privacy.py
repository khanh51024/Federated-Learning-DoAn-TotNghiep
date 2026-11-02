"""Nén cập nhật mỗi vòng (src/federated/compression.py), DP phía máy chủ (src/federated/privacy.py), FedBN+FedProx và tinh chỉnh đầu phân loại (scripts/finetune_personal.py)."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from scripts import eval_gd4_compression as gd4
from scripts import finetune_personal as ft
from src.federated import algorithms as alg
from src.federated import compression as comp
from src.federated import privacy as dp
from src.model.mobilenetv3 import create_model, get_parameters


def _states():
    torch.manual_seed(0)
    model = create_model(num_classes=38, pretrained=False)
    parent = get_parameters(model)
    local = {k: (v + 0.01 * torch.randn_like(v) if v.is_floating_point() else v + 2) for k, v in parent.items()}
    return parent, local


def test_one_quantization_implementation_is_shared_with_the_gd4_measurement():
    assert gd4.fake_quant_int8 is comp.fake_quant_int8 and gd4.is_quantizable is comp.is_quantizable


@pytest.mark.parametrize("mode,ratio", [("fp16", 0.51), ("delta_int8", 0.27)])
def test_encoded_update_is_really_smaller_and_decodes_close_to_original(mode: str, ratio: float):
    parent, local = _states()
    payload = comp.encode_update(local, parent, mode)
    assert alg.state_bytes(payload) < (ratio + 0.03) * alg.state_bytes(local)     # đúng tỉ lệ đã đo ở GĐ4
    decoded = comp.decode_update(payload, parent, mode)
    assert decoded.keys() == parent.keys()
    for key, value in parent.items():
        assert decoded[key].dtype == value.dtype
        if comp.is_quantizable(value):
            assert float((decoded[key] - local[key]).abs().max()) < 2e-3, key
        else:
            assert torch.equal(decoded[key], local[key]), key                     # BatchNorm, bias, bộ đếm: giữ nguyên


def test_delta_quantization_recovers_the_parent_when_the_client_did_not_move():
    parent, _ = _states()
    decoded = comp.decode_update(comp.encode_update(parent, parent, "delta_int8"), parent, "delta_int8")
    for key, value in parent.items():
        assert torch.equal(decoded[key], value), key


def test_unknown_compression_is_rejected():
    parent, local = _states()
    with pytest.raises(ValueError, match="không có"):
        comp.encode_update(local, parent, "zip")
    with pytest.raises(ValueError, match="không có"):
        comp.decode_update(local, parent, "zip")


def test_compressed_payload_survives_the_flower_record():
    pytest.importorskip("flwr")
    from src.federated import flower_app as fa

    parent, local = _states()
    payload = comp.encode_update(local, parent, "delta_int8")
    record = fa.state_to_record(payload)
    assert fa.record_bytes(record) < 0.3 * alg.state_bytes(local)
    back = fa.record_to_state(record)
    assert all(back[k].dtype == v.dtype and torch.equal(back[k], v) for k, v in payload.items())


# --------------------------------------------------------------------------- DP
def test_rdp_epsilon_matches_the_closed_form_and_moves_the_right_way():
    z, rounds, delta = 2.0, 40, 0.01
    closed = rounds / (2 * z * z) + math.sqrt(2 * rounds * math.log(1 / delta)) / z      # α tốt nhất cho công thức liên tục
    assert dp.rdp_epsilon(z, rounds, delta) == pytest.approx(closed, rel=1e-3)
    assert dp.rdp_epsilon(4.0, rounds, delta) < dp.rdp_epsilon(2.0, rounds, delta) < dp.rdp_epsilon(1.0, rounds, delta)  # nhiễu lớn hơn thì ε nhỏ hơn
    assert dp.rdp_epsilon(2.0, 80, delta) > dp.rdp_epsilon(2.0, 40, delta)                                               # nhiều vòng hơn thì ε lớn hơn
    assert dp.rdp_epsilon(0.0, rounds, delta) == math.inf
    with pytest.raises(ValueError):
        dp.rdp_epsilon(1.0, rounds, 0.0)


def test_clip_scales_only_trainable_delta_down_to_the_bound():
    parent = {"w": torch.zeros(4), "bn.running_mean": torch.zeros(2), "n": torch.tensor(1)}
    local = {"w": torch.tensor([3.0, 4.0, 0.0, 0.0]), "bn.running_mean": torch.ones(2), "n": torch.tensor(9)}
    clipped, norm = dp.clip_update(local, parent, {"w"}, clip=1.0)
    assert norm == pytest.approx(5.0)
    assert clipped["w"].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert torch.equal(clipped["bn.running_mean"], local["bn.running_mean"]) and int(clipped["n"]) == 9
    same, norm = dp.clip_update(local, parent, {"w"}, clip=10.0)
    assert norm == pytest.approx(5.0) and torch.equal(same["w"], local["w"])


def test_noise_has_the_requested_std_and_is_reproducible_per_round():
    merged = {"w": torch.zeros(200000), "bn.running_var": torch.ones(3)}
    noisy = dp.add_noise(merged, {"w"}, std=0.5, seed=7)
    assert float(noisy["w"].std()) == pytest.approx(0.5, rel=0.02) and abs(float(noisy["w"].mean())) < 0.01
    assert torch.equal(noisy["bn.running_var"], merged["bn.running_var"])                 # thống kê BN không nhiễu: giới hạn đã ghi ở tài liệu
    assert torch.equal(dp.add_noise(merged, {"w"}, 0.5, 7)["w"], noisy["w"])
    assert not torch.equal(dp.add_noise(merged, {"w"}, 0.5, 8)["w"], noisy["w"])


def test_noise_std_follows_the_sensitivity_of_weighted_averaging():
    assert dp.noise_std(2.0, 3.0, [10, 10, 10, 10, 10]) == pytest.approx(2.0 * 3.0 * 0.2)
    assert dp.noise_std(2.0, 3.0, [70, 10, 10, 5, 5]) == pytest.approx(2.0 * 3.0 * 0.7)


# --------------------------------------------------------------------------- FedBN + FedProx
def test_fedbnprox_penalty_skips_batchnorm_and_keeps_client_batchnorm():
    torch.manual_seed(0)
    model = create_model(num_classes=38, pretrained=False)
    algo = alg.make_algorithm("fedbnprox", model, 2, mu=0.5)
    assert isinstance(algo, alg.FedBN) and algo.name == "fedbnprox"
    parent = get_parameters(model)
    hooks = algo.make_hooks(0, model, parent, 0.1, torch.device("cpu"))
    bn_params = [name for name, _ in model.named_parameters() if name in algo.bn_keys]
    assert bn_params and not set(bn_params) & set(hooks.anchor)
    assert float(hooks.extra_loss(model, None, None)) == 0.0

    # dời tham số BatchNorm không phát sinh phạt; dời tham số khác thì có, đúng công thức (μ/2)·‖Δ‖²
    moved = copy.deepcopy(model)
    params = dict(moved.named_parameters())
    with torch.no_grad():
        params[bn_params[0]].add_(1.0)
    assert float(hooks.extra_loss(moved, None, None)) == 0.0
    other = next(name for name in params if name not in algo.bn_keys)
    with torch.no_grad():
        params[other].add_(1.0)
    assert float(hooks.extra_loss(moved, None, None)) == pytest.approx(0.5 * 0.5 * params[other].numel(), rel=1e-4)

    # FedBN gốc: hành vi giữ BN riêng không đổi
    algo.after_local(0, moved, parent, 0.1, 1, 10)
    assert bn_params[0] in algo.client_bn[0]
    assert torch.equal(algo.start_state(0, parent)[bn_params[0]], params[bn_params[0]].detach())


def test_plain_fedprox_still_penalizes_batchnorm():
    """Thuật toán đối chứng không đổi hành vi vì có cải tiến."""
    torch.manual_seed(0)
    model = create_model(num_classes=38, pretrained=False)
    hooks = alg.make_algorithm("fedprox", model, 1, mu=0.5).make_hooks(0, model, get_parameters(model), 0.1, torch.device("cpu"))
    assert {name for name, _ in model.named_parameters()} == set(hooks.anchor)


# --------------------------------------------------------------------------- tinh chỉnh đầu phân loại
def _head():
    torch.manual_seed(0)
    return create_model(num_classes=38, pretrained=False).classifier


def _skewed(n_per_class: int = 60):
    """Hai lớp 0, 1 tách biệt rõ trên đặc trưng 576 chiều: client chỉ có đúng hai lớp này."""
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(2 * n_per_class, 576, generator=generator) * 0.1
    targets = torch.tensor([0] * n_per_class + [1] * n_per_class)
    features[:n_per_class, :288] += 1.0
    features[n_per_class:, 288:] += 1.0
    return features, targets


RECIPE = {"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-4, "label_smoothing": 0.1, "batch_size": 32}


def test_finetuning_learns_the_client_distribution_without_touching_the_shared_head():
    head = _head().eval()
    before = copy.deepcopy(head.state_dict())
    features, targets = _skewed()
    trained = ft.train_head(head, features, targets, torch.ones(len(targets)), 5, RECIPE["lr"], RECIPE["momentum"], RECIPE["weight_decay"], RECIPE["label_smoothing"], RECIPE["batch_size"], seed=3)
    with torch.inference_mode():
        accuracy = float((trained(features).argmax(1) == targets).float().mean())
    assert accuracy > 0.95
    assert not trained.training
    for key, value in head.state_dict().items():
        assert torch.equal(value, before[key])          # đầu phân loại chung của các client khác không bị ảnh hưởng


def test_epoch_choice_uses_shard_weighted_validation_and_can_keep_the_shared_head():
    head = _head().eval()
    features, targets = _skewed()
    counts = np.bincount(targets.numpy(), minlength=38).astype(float)
    good = ft.choose_epochs(head, features, targets, torch.ones(len(targets)), features, targets.numpy(), counts, RECIPE, seed=3, grid=(1, 5))
    assert good["epochs"] in (1, 5) and good["val_accuracy"] > 0.95 and set(good["tried"]) == {0, 1, 5}

    # val trên lớp mà shard không có: tinh chỉnh không giúp gì, phải giữ đầu chung (0 epoch), không chọn bừa
    val_features, _ = _skewed()
    val_targets = np.array([5] * len(val_features))
    other_counts = np.zeros(38)
    other_counts[5] = 100
    kept = ft.choose_epochs(head, features, targets, torch.ones(len(targets)), val_features, val_targets, other_counts, RECIPE, seed=3, grid=(1,))
    assert kept["epochs"] == 0 and kept["head"] is head
