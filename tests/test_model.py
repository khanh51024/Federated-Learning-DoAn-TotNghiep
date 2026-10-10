"""
Unit test kiểm tra kiến trúc MobileNetV3-Small, trích xuất tham số, và mã băm toàn vẹn SHA-256.
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


def test_mobilenetv3_small_architecture():
    """Kiểm tra MobileNetV3-Small có đúng 38 output classes và nhận input 224x224."""
    model = create_model(num_classes=38, pretrained=False)
    assert model.classifier[3].out_features == 38

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 38)


def test_parameter_extraction_and_restoration():
    """Trích xuất state dict ra CPU và nạp lại phải bảo toàn 100% giá trị tensor."""
    torch.manual_seed(42)
    model1 = create_model(num_classes=38, pretrained=False)
    params = get_parameters(model1)

    model2 = create_model(num_classes=38, pretrained=False)
    set_parameters(model2, params)

    params2 = get_parameters(model2)
    for k in params.keys():
        assert torch.equal(params[k], params2[k]), f"Tensor {k} không trùng khớp sau khi nạp"


def test_model_sha256_hash_integrity():
    """Mã băm SHA-256 phải tất định: cùng trọng số -> cùng hash (CLIENT_HASH == SERVER_HASH)."""
    torch.manual_seed(123)
    model_a = create_model(num_classes=38, pretrained=False)
    hash_a1 = compute_model_hash(get_parameters(model_a))
    hash_a2 = compute_model_hash(get_parameters(model_a))
    assert hash_a1 == hash_a2

    # Đổi 1 trọng số thì hash phải thay đổi
    with torch.no_grad():
        next(model_a.parameters()).add_(0.01)
    hash_a3 = compute_model_hash(get_parameters(model_a))
    assert hash_a1 != hash_a3


def test_finite_parameter_checking():
    """Phải phát hiện lỗi nếu tham số chứa NaN."""
    params = {"weight": torch.tensor([1.0, 2.0, float("nan")])}
    with pytest.raises(FloatingPointError):
        check_parameters_finite(params)
