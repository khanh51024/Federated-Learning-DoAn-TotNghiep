"""
Unit test kiểm định trích xuất, nạp và tính mã băm trọng số mô hình (test_parameters.py).
"""

import numpy as np
import pytest
import torch

from src.model.mobilenetv3 import (
    check_parameters_finite,
    compute_model_hash,
    count_parameters,
    create_model,
    get_parameters,
    set_parameters,
)


def test_get_and_set_parameters():
    """Kiểm tra trích xuất và nạp lại parameters bảo toàn giá trị và dtype."""
    model = create_model(num_classes=38, pretrained=False)
    params = get_parameters(model)

    assert isinstance(params, dict)
    assert len(params) > 0

    # Chuyển đổi sang numpy ndarray
    np_params = {k: v.numpy() for k, v in params.items()}

    # Nạp lại vào model mới
    new_model = create_model(num_classes=38, pretrained=False)
    set_parameters(new_model, np_params)

    # Đối soát
    reloaded_params = get_parameters(new_model)
    for k in params:
        assert torch.allclose(params[k], reloaded_params[k]), f"Mismatch tại {k}"


def test_model_hash_deterministic():
    """Kiểm tra tính tất định của SHA-256 model hash: cùng trọng số -> cùng mã băm."""
    model1 = create_model(num_classes=38, pretrained=False)
    params1 = get_parameters(model1)

    model2 = create_model(num_classes=38, pretrained=False)
    set_parameters(model2, params1)
    params2 = get_parameters(model2)

    hash1 = compute_model_hash(params1)
    hash2 = compute_model_hash(params2)

    assert len(hash1) == 64
    assert hash1 == hash2, "Mã băm SHA-256 phải khớp nhau tuyệt đối giữa hai bản sao trọng số"


def test_parameters_finite_check():
    """Kiểm tra phát hiện và bắt lỗi khi tensor chứa NaN hoặc Inf."""
    t_normal = torch.tensor([1.0, 2.0])
    t_nan = torch.tensor([1.0, float("nan")])
    t_inf = torch.tensor([float("inf"), 2.0])

    # Hợp lệ
    check_parameters_finite({"weight": t_normal})

    # NaN
    with pytest.raises(FloatingPointError):
        check_parameters_finite({"weight": t_nan})

    # Inf
    with pytest.raises(FloatingPointError):
        check_parameters_finite({"weight": t_inf})


def test_parameter_counts():
    """Kiểm tra đếm tham số và dung lượng byte cho MobileNetV3-Small."""
    model = create_model(num_classes=38, pretrained=False)
    total_params, total_bytes = count_parameters(model)

    assert total_params > 500_000, f"Số tham số không hợp lý: {total_params}"
    assert total_bytes > 2_000_000, f"Dung lượng byte không hợp lý: {total_bytes}"
