"""Cách chia client bổ sung: tên, kiểm toán và việc src/training/data.py::load_shard chỉ nhận shard khớp mã băm đã ghim."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import build_extra_partitions as bep
from src.training import data


def _write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def _release(tmp_path, with_extra=True, extra_release_sha="rel"):
    release = {"sha256": "rel", "meta": {"partition_shard_hashes": {"iid": {"client_00.jsonl": _write_shard(tmp_path / "partitions" / "iid" / "client_00.jsonl", [{"sample_id": "a"}])}}}}
    if with_extra:
        sha = _write_shard(tmp_path / "partitions_extra" / "label_alpha_0_5" / "client_00.jsonl", [{"sample_id": "b"}])
        manifest = {"release_manifest_sha256": extra_release_sha, "schemes": {"label_alpha_0_5": {"shard_sha256": {"client_00.jsonl": sha}}}}
        (tmp_path / "partitions_extra" / "extra_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release


def test_scheme_names_follow_existing_convention():
    assert bep.scheme_name(0.1, 42) == "label_alpha_0_1"
    assert bep.scheme_name(0.5, 42) == "label_alpha_0_5"
    assert bep.scheme_name(10.0, 42) == "label_alpha_10_0"
    assert bep.scheme_name(0.1, 123) == "label_alpha_0_1_p123"


def test_load_shard_reads_release_and_extra_schemes(tmp_path):
    release = _release(tmp_path)
    assert data.load_shard(tmp_path, "iid", 0, release) == [{"sample_id": "a"}]
    assert data.load_shard(tmp_path, "label_alpha_0_5", 0, release) == [{"sample_id": "b"}]
    assert data.extra_manifest_sha256(tmp_path) is not None


def test_load_shard_rejects_unknown_scheme_tampering_and_wrong_release(tmp_path):
    release = _release(tmp_path)
    with pytest.raises(SystemExit):
        data.load_shard(tmp_path, "label_alpha_9_9", 0, release)
    (tmp_path / "partitions_extra" / "label_alpha_0_5" / "client_00.jsonl").write_text('{"sample_id": "x"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        data.load_shard(tmp_path, "label_alpha_0_5", 0, release)
    with pytest.raises(SystemExit):        # manifest bổ sung dựng cho release "rel", nhưng release đang dùng có mã băm khác
        data.load_shard(tmp_path, "label_alpha_0_5", 0, {**release, "sha256": "khac"})


def test_load_shard_without_extra_manifest_rejects_unknown_scheme(tmp_path):
    release = _release(tmp_path, with_extra=False)
    assert data.extra_manifest_sha256(tmp_path) is None
    with pytest.raises(SystemExit):
        data.load_shard(tmp_path, "label_alpha_0_5", 0, release)


def test_build_scheme_partitions_all_train_rows_without_splitting_groups(tmp_path):
    rows = []
    for class_id in range(38):
        for g in range(12):
            for i in range(2):
                rows.append({"sample_id": f"s{class_id}_{g}_{i}", "group_id": f"g{class_id}_{g}", "class_id": class_id})
    built = bep.build_scheme(rows, tmp_path, alpha=0.5, seed=42)
    assert built["ok"] and sum(built["n_k"]) == len(rows)
    assert built["audit"]["groups_split_across_clients"] == 0 and built["audit"]["union_equals_train"]
    assert built["metadata"] == {"skew_type": "label_dirichlet", "alpha_label": 0.5, "partition_seed": 42}
    for name, sha in built["shard_sha256"].items():
        assert data.sha256_file(tmp_path / "partitions_extra" / "label_alpha_0_5" / name) == sha
