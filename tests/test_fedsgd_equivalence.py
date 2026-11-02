"""Phép kiểm FedSGD trùng học tập trung (scripts/verify_fedsgd_equivalence.py): phải trùng khi đúng công thức, và phải lộ ra khi phép gộp sai hoặc có nhiều bước cục bộ."""
from __future__ import annotations

import copy

import pytest
import torch

from scripts import verify_fedsgd_equivalence as ve


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.bn(self.conv(x)).mean(dim=(2, 3)))


def _setup(dtype=torch.float64):
    torch.manual_seed(0)
    model = Small().to(dtype=dtype)
    shards = []
    for size, label in ((8, 0), (4, 1), (2, 2)):                       # cỡ khác nhau và mỗi client một lớp: lệch nhãn hoàn toàn
        generator = torch.Generator().manual_seed(size)
        shards.append((torch.randn(size, 3, 8, 8, generator=generator, dtype=dtype), torch.full((size,), label)))
    return model, shards


def test_fedsgd_equals_centralized_up_to_rounding_with_frozen_batchnorm():
    model, shards = _setup()
    reference = ve.centralized(model, shards, 6, 0.1, False)
    fed = ve.federated(model, shards, 6, 1, 0.1, False)
    gap = ve.distance(fed, reference, model)
    assert gap["max_abs"] < 1e-12
    assert gap["l2_relative_to_centralized_travel"] < 1e-11
    assert not torch.equal(dict(reference.named_parameters())["head.weight"], dict(model.named_parameters())["head.weight"])    # mô hình thật sự đã học


def test_local_steps_on_skewed_data_create_a_real_gap():
    model, shards = _setup()
    reference = ve.centralized(model, shards, 12, 0.1, False)
    gaps = [ve.distance(ve.federated(model, shards, 12 // k, k, 0.1, False), reference, model)["l2"] for k in (1, 3, 6)]
    assert gaps[0] < 1e-11 < gaps[1] < gaps[2]


def test_live_batchnorm_breaks_the_equivalence_even_with_one_step():
    model, shards = _setup()
    reference = ve.centralized(model, shards, 6, 0.1, True)
    fed = ve.federated(model, shards, 6, 1, 0.1, True)
    assert ve.distance(fed, reference, model)["l2"] > 1e-6


def test_the_check_is_sensitive_to_a_wrong_aggregation():
    """Phép kiểm chỉ có giá trị nếu gộp sai thì nó báo lệch: gộp đều thay vì theo n_k phải lộ ngay."""
    model, shards = _setup()
    reference = ve.centralized(model, shards, 1, 0.1, False)
    locals_ = []
    for images, targets in shards:
        local = copy.deepcopy(model)
        ve.set_mode(local, False)
        ve.batch_gradient_step(local, images, targets, 0.1)
        locals_.append(local.state_dict())
    wrong = copy.deepcopy(model)
    wrong.load_state_dict({k: (sum(s[k] for s in locals_) / len(locals_) if v.is_floating_point() else v) for k, v in locals_[0].items()})
    assert ve.distance(wrong, reference, model)["l2"] > 1e-4
    assert ve.distance(ve.federated(model, shards, 1, 1, 0.1, False), reference, model)["l2"] < 1e-12


def test_set_mode_keeps_dropout_off_in_both_cases():
    model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(2))
    ve.set_mode(model, True)
    assert model[0].training is False and model[1].training is True
    ve.set_mode(model, False)
    assert model[0].training is False and model[1].training is False


def test_agreement_is_one_for_the_same_model():
    model, shards = _setup()
    assert ve.agreement(model, model, shards[0][0], False) == pytest.approx(1.0)
