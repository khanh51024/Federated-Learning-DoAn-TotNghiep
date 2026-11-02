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


# --------------------------------------------------------------------------- lệch số lượng, lệch đặc trưng
def _mixed_rows(lab_groups_per_class: int = 30, field_groups_per_class: int = 15, classes: int = 38) -> list[dict]:
    rows = []
    for class_id in range(classes):
        for domain, count in (("plantvillage", lab_groups_per_class), ("plantdoc", field_groups_per_class)):
            for g in range(count):
                for i in range(2):
                    rows.append({"sample_id": f"{domain}{class_id}_{g}_{i}", "group_id": f"{domain}{class_id}_{g}", "class_id": class_id, "source_domain": domain})
    return rows


def test_names_of_new_schemes():
    assert bep.scheme_name(0.5, 42, "quantity_alpha") == "quantity_alpha_0_5"
    assert bep.source_scheme_name(42) == "source_skew"
    assert bep.source_scheme_name(7) == "source_skew_p7"


def test_quantity_scheme_skews_sizes_but_keeps_label_mix(tmp_path):
    rows = _mixed_rows()
    built = bep.build_quantity_scheme(rows, tmp_path, quantity_alpha=0.5, seed=42)
    assert built["ok"] and sum(built["n_k"]) == len(rows)
    assert built["metadata"] == {"skew_type": "quantity_dirichlet", "alpha_quantity": 0.5, "partition_seed": 42}
    assert built["size_ratio_max_over_min"] > 2
    # trộn lớp trong client gần toàn cục: đây là chỗ khác với lệch nhãn cùng α
    label = bep.build_scheme(rows, tmp_path, alpha=0.5, seed=42)
    assert max(built["heterogeneity"]["label_tvd"].values()) < max(label["heterogeneity"]["label_tvd"].values())


def test_source_skew_gives_each_client_the_planned_field_share_with_equal_sizes(tmp_path):
    rows = _mixed_rows()
    num_field = sum(r["source_domain"] != "plantvillage" for r in rows)
    built = bep.build_source_scheme(rows, tmp_path, seed=42)
    assert built["ok"] and sum(built["n_k"]) == len(rows)
    assert built["audit"]["groups_split_across_clients"] == 0
    assert built["size_ratio_max_over_min"] < 1.05                                    # không lẫn thêm lệch số lượng

    per_client = len(rows) / 5
    fractions = list(built["heterogeneity"]["field_fraction"].values())
    assert fractions[0] == 0.0                                                         # client 0 chỉ có ảnh phòng thí nghiệm
    assert fractions == sorted(fractions)
    for fraction, share in zip(fractions, bep.FIELD_SHARE):
        assert abs(fraction - share * num_field / per_client) < 0.03
    # lớp vẫn gần toàn cục nên khác biệt chính là nguồn ảnh
    assert max(built["heterogeneity"]["label_tvd"].values()) < 0.1
    assert all(c == 38 for c in built["classes_with_images"].values())


def test_source_skew_is_reproducible_and_seed_dependent(tmp_path):
    rows = _mixed_rows()
    first = bep.build_source_scheme(rows, tmp_path / "a", seed=42)
    second = bep.build_source_scheme(rows, tmp_path / "b", seed=42)
    other = bep.build_source_scheme(rows, tmp_path / "c", seed=7)
    assert first["shard_sha256"] == second["shard_sha256"]
    assert first["shard_sha256"] != other["shard_sha256"]


def test_domain_shares_rejects_impossible_plans():
    assert bep.domain_shares(80, 20, (0.0, 0.0, 0.0, 0.5, 0.5)) == pytest.approx([0.25, 0.25, 0.25, 0.125, 0.125])
    with pytest.raises(ValueError, match="tổng bằng 1"):
        bep.domain_shares(80, 20, (0.5, 0.2, 0.1, 0.1, 0.0))
    with pytest.raises(ValueError, match="quá nhiều"):
        bep.domain_shares(20, 80, (0.0, 0.0, 0.0, 0.0, 1.0))


def test_rebuilding_a_pinned_scheme_differently_is_refused(tmp_path, monkeypatch):
    """Đã có run dùng shard cũ thì dựng lại ra shard khác phải dừng, không lặng lẽ đổi."""
    import sys

    release_dir = tmp_path / "release"
    rows = _mixed_rows(lab_groups_per_class=6, field_groups_per_class=3)
    (release_dir / "manifests").mkdir(parents=True)
    for name in ("train", "val", "calibration", "legacy_diagnostic_test"):
        (release_dir / "manifests" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (rows if name == "train" else [])), encoding="utf-8")
    (release_dir / "release_manifest.json").write_text(json.dumps({"release_name": "t", "manifest_hashes": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--release-dir", str(release_dir), "--specs", "", "--quantity-specs", "", "--source-specs", "42"])
    assert bep.main() == 0
    manifest = release_dir / "partitions_extra" / "extra_manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert bep.main() == 0                                                              # dựng lại y hệt thì qua
    saved["schemes"]["source_skew"]["shard_sha256"]["client_00.jsonl"] = "khac"
    manifest.write_text(json.dumps(saved), encoding="utf-8")
    assert bep.main() == 2
