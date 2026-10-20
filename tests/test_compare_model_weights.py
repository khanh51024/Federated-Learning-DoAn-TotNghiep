"""Phép tính thuần của scripts/compare_model_weights.py."""
from __future__ import annotations

import numpy as np

from scripts.compare_model_weights import (
    grouped_distance,
    interpolate,
    is_bn_stat,
    layer_group,
    linear_cka,
    softmax,
    symmetric_kl,
    vector_distance,
)


def test_layer_groups_and_bn_stats():
    assert layer_group("features.2.block.0.0.weight") == "features_low_0_3"
    assert layer_group("features.7.block.1.1.bias") == "features_mid_4_8"
    assert layer_group("features.12.0.weight") == "features_high_9_12"
    assert layer_group("classifier.0.weight") == "classifier_hidden_576_to_1024"
    assert layer_group("classifier.3.bias") == "classifier_output_1024_to_38"
    assert is_bn_stat("features.0.1.running_var") and not is_bn_stat("features.0.1.weight")


def test_vector_distance_identity_and_opposite():
    v = np.arange(1, 7, dtype=np.float32)
    same = vector_distance(v, v)
    assert same["l2"] == 0 and np.isclose(same["cosine"], 1.0)
    assert np.isclose(vector_distance(v, -v)["cosine"], -1.0)


def test_grouped_distance_separates_learnable_from_bn_and_honours_skip():
    a = {"features.0.0.weight": np.ones(4, np.float32), "features.0.1.running_var": np.ones(4, np.float32),
         "classifier.3.weight": np.ones(4, np.float32), "features.0.1.num_batches_tracked": np.array(5, np.int64)}
    b = {"features.0.0.weight": np.ones(4, np.float32) * 2, "features.0.1.running_var": np.ones(4, np.float32) * 5,
         "classifier.3.weight": np.ones(4, np.float32) * 9, "features.0.1.num_batches_tracked": np.array(9, np.int64)}
    out = grouped_distance(a, b, skip=("classifier.3",))
    assert np.isclose(out["learnable:all"]["l2"], 2.0)  # sqrt(4 * 1^2); classifier.3 bị bỏ
    assert np.isclose(out["bn_stats:all"]["l2"], 8.0)  # sqrt(4 * 4^2); bộ đếm int không tính
    assert "learnable:classifier_output_1024_to_38" not in out


def test_linear_cka_is_one_for_identical_and_rotation_invariant_but_low_for_independent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 20))
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    assert np.isclose(linear_cka(x, x), 1.0)
    assert np.isclose(linear_cka(x, x @ q), 1.0)
    assert linear_cka(x, rng.standard_normal((300, 20))) < 0.2


def test_symmetric_kl_zero_for_equal_and_positive_otherwise():
    p = softmax(np.array([[2.0, 0.0, -1.0]]))
    assert symmetric_kl(p, p) < 1e-9
    assert symmetric_kl(p, softmax(np.array([[-1.0, 0.0, 2.0]]))) > 0.5


def test_interpolate_endpoints_and_counter():
    a = {"w": np.zeros(3, np.float32), "n": np.array(1, np.int64)}
    b = {"w": np.ones(3, np.float32) * 4, "n": np.array(7, np.int64)}
    assert np.allclose(interpolate(a, b, 0.0)["w"], 0) and np.allclose(interpolate(a, b, 1.0)["w"], 4)
    assert np.allclose(interpolate(a, b, 0.25)["w"], 1)
    assert int(interpolate(a, b, 0.25)["n"]) == 1 and int(interpolate(a, b, 0.75)["n"]) == 7
