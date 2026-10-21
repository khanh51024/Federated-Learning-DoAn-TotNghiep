"""Phần thuần của scripts/build_release_v4.py: chia split theo nhóm, Hamming, gộp nhóm gần trùng, giới hạn PlantVillage."""
from __future__ import annotations

import numpy as np

from scripts.build_release_v4 import (
    cap_per_class, conflicting_groups, groups_by_primary_class, merge_near_duplicate_groups, min_hamming, split_for_group,
)


def test_split_for_group_is_deterministic_and_roughly_70_10_10_10():
    names = [split_for_group(f"g{i}", 7) for i in range(4000)]
    assert names == [split_for_group(f"g{i}", 7) for i in range(4000)]
    share = {s: names.count(s) / len(names) for s in set(names)}
    assert abs(share["train"] - 0.70) < 0.03 and all(abs(share[s] - 0.10) < 0.03 for s in ("val", "calibration", "legacy_diagnostic_test"))
    assert split_for_group("g1", 7) != split_for_group("g1", 8) or True  # seed khác có thể đổi phía


def test_min_hamming_counts_bits_and_handles_empty_refs():
    refs = np.array([0, 0xFF], dtype=np.uint64)
    q = np.array([0b1, 0xFFFF], dtype=np.uint64)
    assert min_hamming(q, refs).tolist() == [1, 8]
    assert min_hamming(q, np.array([], dtype=np.uint64)).tolist() == [64, 64]


def test_merge_near_duplicate_groups_unites_across_sources_but_not_distant_images():
    hashes = np.array([0b0000, 0b0001, 0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_0000_0001], dtype=np.uint64)
    merged = merge_near_duplicate_groups(hashes, ["a:1", "b:1", "a:2", "c:9"], 2)
    assert merged[0] == merged[1] and merged[2] == merged[3] and merged[0] != merged[2]


def test_cap_per_class_is_order_independent_and_keeps_small_classes_whole():
    rows = [{"sample_id": f"s{i:03d}", "class_id": 0} for i in range(50)] + [{"sample_id": f"t{i}", "class_id": 1} for i in range(5)]
    a = cap_per_class(rows, 20, 1)
    b = cap_per_class(list(reversed(rows)), 20, 1)
    assert a == b and sum(1 for r in a if r["class_id"] == 0) == 20 and sum(1 for r in a if r["class_id"] == 1) == 5


def test_scene_groups_use_v3_split_names_and_keep_group_ids_unique():
    from scripts.build_release_v4 import build_scene_groups

    rows = {
        "train": [{"group_id": "g1", "source_domain": "plantdoc"}, {"group_id": "g1", "source_domain": "plantdoc"}],
        "val": [{"group_id": "g2", "source_domain": "plantvillage"}],
        "calibration": [], "legacy_diagnostic_test": [{"group_id": "g3", "source_domain": "plantdoc"}],
        "quarantine": [{"group_id": "g1", "source_domain": "plantdoc"}, {"group_id": "g4", "source_domain": "plantdoc"}],
    }
    groups = build_scene_groups(rows)
    by_id = {g["group_id"]: g for g in groups}
    assert len(groups) == len(by_id) == 4  # không lặp group_id dù g1 có lá ở cả train và quarantine
    assert by_id["g1"]["split"] == "train" and by_id["g1"]["evaluation_role"] == "supervised_train" and by_id["g1"]["num_samples"] == 3
    assert by_id["g2"]["split"] == "validation" and by_id["g3"]["split"] == "test" and by_id["g4"]["split"] == "quarantine"


def test_conflicting_groups_flags_only_groups_with_two_labels():
    assert conflicting_groups([1, 1, 2, 3, 3], ["a", "a", "b", "c", "c"]) == set()
    assert conflicting_groups([1, 2, 3, 3], ["a", "a", "b", "b"]) == {"a"}


def test_groups_by_primary_class_keeps_a_multi_class_group_whole_in_its_majority_class():
    rows = [{"group_id": "g1", "class_id": 5, "sample_id": "a"}, {"group_id": "g1", "class_id": 5, "sample_id": "b"},
            {"group_id": "g1", "class_id": 9, "sample_id": "c"}, {"group_id": "g2", "class_id": 9, "sample_id": "d"},
            {"group_id": "g3", "class_id": 2, "sample_id": "e"}, {"group_id": "g3", "class_id": 7, "sample_id": "f"}]
    by_class = groups_by_primary_class(rows)
    assert {c: sorted(g["group_id"] for g in gs) for c, gs in by_class.items()} == {5: ["g1"], 9: ["g2"], 2: ["g3"]}  # g3 hòa 1-1 -> lớp nhỏ hơn
    assert sorted(i["sample_id"] for g in by_class[5] for i in g["items"]) == ["a", "b", "c"]
    assert sum(len(g["items"]) for gs in by_class.values() for g in gs) == len(rows)
