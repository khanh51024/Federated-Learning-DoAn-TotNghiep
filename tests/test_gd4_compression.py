"""scripts/eval_gd4_compression.py: các phép lượng tử hóa trọng số và cách tính số byte (phần chạy mô hình thật kiểm bằng tay, cần dữ liệu ảnh)."""
from __future__ import annotations

import pytest
import torch

from scripts import eval_gd4_compression as gd4


def _state():
    torch.manual_seed(0)
    return {
        "conv.weight": torch.randn(4, 3, 3, 3),
        "fc.weight": torch.randn(5, 8),
        "fc.bias": torch.randn(5),
        "bn.running_mean": torch.randn(4),
        "bn.num_batches_tracked": torch.tensor(7),
    }


def test_only_float_tensors_with_two_or_more_dimensions_are_quantizable():
    assert gd4.is_quantizable(torch.zeros(2, 2)) and gd4.is_quantizable(torch.zeros(2, 3, 3, 3))
    assert not gd4.is_quantizable(torch.zeros(4)) and not gd4.is_quantizable(torch.zeros(2, 2, dtype=torch.int64))


@pytest.mark.parametrize("per_channel", [True, False])
def test_int8_error_is_at_most_half_a_step(per_channel):
    tensor = torch.randn(6, 10) * torch.tensor([[0.01], [0.1], [1.0], [3.0], [10.0], [0.5]])
    restored, scales = gd4.fake_quant_int8(tensor, per_channel)
    flat_max = tensor.abs().amax(dim=1) if per_channel else tensor.abs().amax().expand(6)
    step = flat_max / 127.0
    assert scales == (6 if per_channel else 1)
    assert bool(((restored - tensor).abs() <= step[:, None] / 2 + 1e-7).all())


def test_per_channel_beats_per_tensor_when_channels_differ_in_scale():
    tensor = torch.randn(6, 50) * torch.tensor([[0.001], [0.01], [0.1], [1.0], [5.0], [20.0]])
    error = {mode: float((gd4.fake_quant_int8(tensor, mode)[0] - tensor).abs().mean()) for mode in (True, False)}
    assert error[True] < error[False]


def test_all_zero_tensor_is_safe():
    restored, _ = gd4.fake_quant_int8(torch.zeros(3, 4), per_channel=True)
    assert torch.equal(restored, torch.zeros(3, 4))


def test_fp32_scheme_is_identity_and_counts_every_byte():
    state = _state()
    out, size = gd4.transform_state(state, "fp32", None)
    assert all(torch.equal(out[k], state[k]) for k in state)
    assert size == sum(t.numel() * t.element_size() for t in state.values())


def test_payload_sizes_follow_scheme():
    state = _state()
    quant = {k: t for k, t in state.items() if gd4.is_quantizable(t)}
    other = sum(t.numel() * t.element_size() for k, t in state.items() if k not in quant)
    n = sum(t.numel() for t in quant.values())
    assert gd4.transform_state(state, "fp16", None)[1] == 2 * n + other
    assert gd4.transform_state(state, "int8_per_tensor", None)[1] == n + 4 * len(quant) + other
    assert gd4.transform_state(state, "int8_per_channel", None)[1] == n + 4 * sum(t.shape[0] for t in quant.values()) + other


def test_non_quantizable_tensors_pass_through_unchanged():
    state = _state()
    out, _ = gd4.transform_state(state, "int8_per_channel", None)
    for key in ("fc.bias", "bn.running_mean", "bn.num_batches_tracked"):
        assert torch.equal(out[key], state[key])
    assert not torch.equal(out["conv.weight"], state["conv.weight"])


def test_delta_scheme_is_exact_at_w0_and_beats_absolute_int8_for_small_updates():
    torch.manual_seed(1)
    w0 = {"w": torch.randn(8, 20) * 50.0}                      # trọng số khởi tạo có biên độ lớn
    state = {"w": w0["w"] + 0.01 * torch.randn(8, 20)}         # huấn luyện chỉ dịch một chút
    same, _ = gd4.transform_state(w0, "delta_int8_per_channel", w0)
    assert torch.equal(same["w"], w0["w"])
    absolute = (gd4.transform_state(state, "int8_per_channel", None)[0]["w"] - state["w"]).abs().max()
    delta = (gd4.transform_state(state, "delta_int8_per_channel", w0)[0]["w"] - state["w"]).abs().max()
    assert float(delta) < float(absolute) / 10


def test_delta_scheme_requires_w0_and_unknown_scheme_is_rejected():
    with pytest.raises(ValueError):
        gd4.transform_state(_state(), "delta_int8_per_channel", None)
    with pytest.raises(ValueError):
        gd4.transform_state(_state(), "int4", None)


def test_onnx_node_groups_of_mobilenetv3_small(tmp_path):
    from src.model.mobilenetv3 import create_model

    path = tmp_path / "m.onnx"
    gd4.export_onnx(create_model(num_classes=38, pretrained=False), path)
    groups = gd4.classify_nodes(path)
    assert {k: len(v) for k, v in groups.items()} == {"first": 1, "depthwise": 11, "pointwise": 22, "se": 18, "gemm": 2}
    assert gd4.variant_nodes(groups, "int8_all") is None
    safe = gd4.variant_nodes(groups, "int8_se_gemm")
    assert set(safe) == set(groups["se"]) | set(groups["gemm"])
    late = gd4.variant_nodes(groups, "int8_se_gemm_pw_late")
    assert set(late) == set(safe) | set(groups["pointwise"][11:]) and not set(late) & (set(groups["first"]) | set(groups["depthwise"]))
