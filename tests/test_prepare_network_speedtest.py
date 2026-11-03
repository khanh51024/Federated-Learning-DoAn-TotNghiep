"""Chia ảnh cho phép đo mạng nhiều client: shard rời nhau, tập kiểm tra rời mọi shard, ảnh của split niêm phong bị loại."""
from __future__ import annotations

import hashlib

import pytest

from scripts.prepare_network_speedtest import pick_disjoint_shards, sealed_hashes


def _make_tree(root, classes=("a", "b"), per_class=30):
    for name in classes:
        (root / name).mkdir(parents=True)
        for i in range(per_class):
            (root / name / f"{name}_{i:03d}.jpg").write_bytes(f"{name}-{i}".encode())


def test_shards_and_eval_sets_are_disjoint(tmp_path):
    _make_tree(tmp_path)
    shards, val, test = pick_disjoint_shards(tmp_path, ["a", "b"], per_class=4, clients=5, seed=1, val_per_class=2, test_per_class=3)
    paths = [row[0] for shard in shards for row in shard] + [row[0] for row in val] + [row[0] for row in test]
    assert len(paths) == len(set(paths)) == 2 * (5 * 4 + 2 + 3)
    assert all(len(shard) == 2 * 4 for shard in shards) and len(val) == 4 and len(test) == 6
    assert {row[1] for row in val} == {0, 1} and {row[2] for row in test} == {"a", "b"}


def test_default_call_keeps_the_old_two_client_shards(tmp_path):
    _make_tree(tmp_path)
    shards, val, test = pick_disjoint_shards(tmp_path, ["a", "b"], per_class=3, clients=2, seed=42)
    assert val == [] and test == [] and [len(s) for s in shards] == [6, 6]


def test_sealed_images_are_never_picked(tmp_path):
    _make_tree(tmp_path)
    sealed = frozenset(hashlib.sha256(f"a-{i}".encode()).hexdigest() for i in range(10))     # 10 ảnh đầu của lớp a bị niêm phong
    shards, val, test = pick_disjoint_shards(tmp_path, ["a", "b"], per_class=2, clients=3, seed=7, val_per_class=1, test_per_class=1, sealed=sealed)
    picked = [row[0] for shard in shards for row in shard] + [row[0] for row in val] + [row[0] for row in test]
    forbidden = {f"a/a_{i:03d}.jpg" for i in range(10)}
    assert not forbidden & set(picked)


def test_not_enough_usable_images_is_an_error(tmp_path):
    _make_tree(tmp_path, per_class=8)
    with pytest.raises(ValueError):
        pick_disjoint_shards(tmp_path, ["a", "b"], per_class=4, clients=2, seed=1, val_per_class=1, test_per_class=1)


def test_sealed_hashes_reads_only_the_byte_hash_column(tmp_path):
    manifest = tmp_path / "calibration.jsonl"
    manifest.write_text('{"byte_sha256": "aa", "class_id": 3}\n\n{"byte_sha256": "bb", "class_id": 4}\n', encoding="utf-8")
    assert sealed_hashes(manifest) == frozenset({"aa", "bb"})
