"""Phần thống kê thuần numpy của scripts/report_statistics.py và phép đối chiếu test giữ riêng (không cần dataset)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import evaluate_published_models as ev
from scripts import report_statistics as st


def test_cluster_sums_groups_values_by_cluster():
    n, s = st.cluster_sums(["a", "b", "a", "c"], np.array([1.0, 0.0, 1.0, 1.0]))
    assert n.tolist() == [2.0, 1.0, 1.0]
    assert s.tolist() == [2.0, 0.0, 1.0]


def test_cluster_bootstrap_matches_estimate_and_is_wider_with_correlated_clusters():
    rng = np.random.default_rng(0)
    # 100 cụm, mỗi cụm 10 ảnh cùng đúng hoặc cùng sai (tương quan hoàn toàn trong cụm)
    clusters = [f"c{i // 10}" for i in range(1000)]
    values = np.repeat(rng.random(100) < 0.6, 10).astype(np.float64)
    clustered = st.cluster_bootstrap_ci(clusters, values, np.random.default_rng(1))
    independent = st.cluster_bootstrap_ci([f"i{i}" for i in range(1000)], values, np.random.default_rng(1))
    assert clustered["estimate"] == pytest.approx(values.mean())
    assert clustered["low"] < clustered["estimate"] < clustered["high"]
    # bỏ qua cụm làm khoảng tin cậy hẹp sai: phải hẹp hơn rõ rệt so với lấy mẫu theo cụm
    assert (independent["high"] - independent["low"]) < 0.6 * (clustered["high"] - clustered["low"])


def test_cluster_bootstrap_with_a_single_cluster_has_no_interval():
    out = st.cluster_bootstrap_ci(["x", "x"], np.array([1.0, 0.0]), np.random.default_rng(0))
    assert out["estimate"] == 0.5 and out["low"] is None and out["high"] is None


def test_mcnemar_exact_against_hand_computed_values():
    a = np.array([True] * 10 + [True] * 5)
    b = np.array([False] * 10 + [True] * 5)
    out = st.mcnemar_exact(a, b)
    assert out["a_right_b_wrong"] == 10 and out["a_wrong_b_right"] == 0
    assert out["p_value"] == pytest.approx(2 * 0.5 ** 10)
    same = st.mcnemar_exact(np.array([True, False]), np.array([False, True]))
    assert same["p_value"] == pytest.approx(1.0)


@pytest.mark.parametrize("path, expected", [
    ("D:\\data\\PlantVillage-Dataset\\raw\\color\\Soybean___healthy\\c037___RS_HL 5550 1.JPG", ("Soybean___healthy", "RS_HL ", 5550)),
    ("x/raw/color/Tomato___healthy/6ddb___GH_HL Leaf 378.JPG", ("Tomato___healthy", "GH_HL Leaf ", 378)),
    ("x/raw/color/Corn___healthy/abc___R.S_HL 0630 copy 3.jpg", ("Corn___healthy", "R.S_HL ", 630)),
    ("x/raw/color/Corn___healthy/no-separator.jpg", None),
])
def test_parse_pv_name(path, expected):
    assert st.parse_pv_name(path) == expected


def test_adjacent_reshoot_flags_only_ungrouped_neighbours():
    train = [{"original_path": "r/A___x/u___TgS_FL 0013.JPG", "group_id": "g1"}]
    query = [
        {"original_path": "r/A___x/v___TgS_FL 0014.JPG", "group_id": "solo1"},   # số liền kề, không có nhóm -> nghi chụp lại
        {"original_path": "r/A___x/w___TgS_FL 0016.JPG", "group_id": "solo2"},   # cách 3 số -> không
        {"original_path": "r/A___x/z___TgS_FL 0012.JPG", "group_id": "grouped"},  # có nhóm lá -> release đã tách, không tính
        {"original_path": "r/B___y/q___TgS_FL 0014.JPG", "group_id": "solo3"},   # khác lớp -> không
    ]
    flags = st.adjacent_reshoot_flags(query, train, singleton_groups={"solo1", "solo2", "solo3"})
    assert flags.tolist() == [True, False, False, False]


def test_compare_with_recorded_uses_rounded_accuracy_and_counts():
    own = {"count": 172, "correct": 109}
    recorded = {"count": 172, "accuracy": 0.6337}  # 0,6337 * 172 = 108,996 -> 109 ảnh
    out = ev.compare_with_recorded(own, recorded)
    assert out["reproduced"] is True and out["correct_diff_images"] == 0
    off = ev.compare_with_recorded({"count": 172, "correct": 108}, recorded)
    assert off["reproduced"] is False and off["correct_diff_images"] == -1
    assert ev.compare_with_recorded(own, None) == {"available": False}


def test_compare_with_recorded_accepts_every_count_that_rounds_to_the_recorded_value():
    # 0,9923 ghi 4 chữ số ứng với 10.759 hoặc 10.760 ảnh đúng trên 10.843 (10.759/10.843 = 0,992253 -> 0,9923)
    recorded = {"count": 10843, "accuracy": 0.9923}
    for correct in (10759, 10760):
        out = ev.compare_with_recorded({"count": 10843, "correct": correct}, recorded)
        assert out["reproduced"] is True and out["correct_diff_images"] == 0
        assert out["correct_recorded_range"] == [10759, 10760]
    assert ev.compare_with_recorded({"count": 10843, "correct": 10757}, recorded)["correct_diff_images"] == -2
    assert ev.compare_with_recorded({"count": 10843, "correct": 10762}, recorded)["correct_diff_images"] == 2
    assert ev.compare_with_recorded({"count": 10842, "correct": 10759}, recorded)["reproduced"] is False
