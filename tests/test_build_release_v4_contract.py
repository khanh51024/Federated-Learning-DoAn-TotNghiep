"""Phần thuần của scripts/build_release_v4.py liên quan tới hợp đồng với bộ chạy huấn luyện và cổng rò rỉ cùng nguồn."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.build_release_v4 import (
    build_partition_spec, build_quality_audit, cap_per_class, capture_time, js_divergence_bits, merge_near_duplicate_groups,
    min_hamming_to_other_split, session_links, shannon_entropy_bits, verify_image_bytes,
)


def test_capture_time_reads_timestamps_in_names_and_ignores_the_rest():
    a, b = capture_time("20250404_154210_aug1.jpg"), capture_time("20250404-154213.jpg")
    assert a is not None and b - a == 3
    assert capture_time("IMG-20230101-WA0001.jpg") is None   # ngày không kèm giờ
    assert capture_time("Healthy (373).jpg") is None
    assert capture_time("x_20251399_250000.jpg") is None      # tháng/giờ không hợp lệ
    assert capture_time("123420250404_154210.jpg") is None    # dính vào dãy số dài hơn


def test_session_links_chain_close_shots_split_on_gaps_and_cap_session_length():
    names = ["20250404_154210.jpg", "20250404_154230.jpg", "20250404_154300.jpg", "20250404_160000.jpg", "20250404_154212.jpg", "no_time.jpg"]
    sources = ["s"] * 6
    links = session_links(sources, names, seconds=60, max_seconds=600)
    # thứ tự theo giờ: 154210(0) 154212(4) 154230(1) 154300(2) | 160000(3); không giờ(5) không nối
    assert sorted(tuple(sorted(p)) for p in links) == [(0, 4), (1, 2), (1, 4)]
    # phiên dài vô hạn bị chặn: 5 ảnh cách nhau 40 giây, tối đa 100 giây một phiên
    chained = [f"20250404_1500{s:02d}.jpg" for s in (0, 40)] + ["20250404_150120.jpg", "20250404_150200.jpg", "20250404_150240.jpg"]
    links = session_links(["s"] * 5, chained, seconds=60, max_seconds=100)
    assert (2, 3) not in links and (0, 1) in links and (1, 2) in links and (3, 4) in links   # sau 100 giây bắt đầu phiên mới
    # khác nguồn thì không nối
    assert session_links(["a", "b"], ["20250404_154210.jpg", "20250404_154211.jpg"], 60, 600) == []


def test_merge_uses_a_looser_threshold_only_inside_one_source_and_honours_flips_and_links():
    h = np.array([0, 0xFF, 0xFF << 32], dtype=np.uint64)                    # 0 và 1 cách nhau 8 bit; ảnh 2 cách ảnh 0 là 8 bit, cách ảnh 1 là 16 bit
    groups = ["g0", "g1", "g2"]
    out = merge_near_duplicate_groups(h, groups, 4, sources=["a", "a", "b"], same_source_max_hamming=10)
    assert out[0] == out[1] and out[2] != out[0]                           # cùng nguồn Hamming 8 gộp; khác nguồn Hamming 8 > 4 không gộp
    assert len(set(merge_near_duplicate_groups(h, groups, 4))) == 3        # không khai nguồn thì giữ ngưỡng cũ
    far = np.array([0xFFFF, 0], dtype=np.uint64)                           # bản gốc cách nhau 16 bit nhưng bản lật của ảnh 0 trùng ảnh 1
    flipped =merge_near_duplicate_groups(far, ["x", "y"], 4, sources=["a", "a"], same_source_max_hamming=10, flipped=np.array([0, 0], dtype=np.uint64))
    unflipped = merge_near_duplicate_groups(far, ["x", "y"], 4, sources=["a", "a"], same_source_max_hamming=10)
    assert flipped[0] == flipped[1] and unflipped[0] != unflipped[1]
    linked = merge_near_duplicate_groups(np.array([0, ~np.uint64(0)], dtype=np.uint64), ["x", "y"], 4, links=[(0, 1)])
    assert linked[0] == linked[1]


def test_cap_per_class_zero_keeps_everything():
    rows = [{"sample_id": f"s{i}", "class_id": i % 2} for i in range(10)]
    assert len(cap_per_class(rows, 0, 1)) == 10 and len(cap_per_class(rows, 3, 1)) == 6


def test_entropy_and_js_divergence_use_bits():
    assert shannon_entropy_bits([5, 5, 5, 5]) == pytest.approx(2.0)
    assert shannon_entropy_bits([0, 0]) == 0.0
    assert js_divergence_bits([3, 1], [3, 1]) == pytest.approx(0.0)
    assert js_divergence_bits([1, 0], [0, 1]) == pytest.approx(1.0)


def _rows(client_classes: dict[str, list[int]]) -> dict[str, list[dict]]:
    out, n = {}, 0
    for client, classes in client_classes.items():
        out[client] = []
        for cid in classes:
            n += 1
            out[client].append({"sample_id": f"s{n}", "group_id": f"g{n}", "class_id": cid, "source_domain": "plantvillage", "source_dataset": "PlantVillage"})
    return out


def test_partition_spec_matches_the_runner_contract_and_flags_wrong_skew():
    skew = _rows({"client_00": [0] * 30, "client_01": [1] * 30})
    iid = _rows({"client_00": [0, 1] * 15, "client_01": [0, 1] * 15})
    train = [r for rows in skew.values() for r in rows]
    schemes = {
        "label_alpha_0_1": {"shards": skew, "shard_sha256": {c: "a" * 64 for c in skew}, "structural_ok": True,
                            "metadata": {"skew_type": "label_dirichlet", "alpha_label": 0.1}},
        "iid": {"shards": iid, "shard_sha256": {c: "b" * 64 for c in iid}, "structural_ok": True, "metadata": {"skew_type": "iid"}},
    }
    spec = build_partition_spec("mixed_pv_pd_v4", "c" * 64, train, schemes, num_scene_groups=60)
    assert spec["release_manifest_sha256"] == "c" * 64 and spec["num_clients"] == 2 and spec["n_train_samples"] == 60
    label = spec["scenarios"]["label_alpha_0_1"]
    assert label["status"] == "PASS" and set(label["shard_hashes"]) == {"client_00.jsonl", "client_01.jsonl"}   # khóa có đuôi .jsonl như bộ chạy so
    assert label["metadata"] == {"skew_type": "label_dirichlet", "alpha_label": 0.1}
    stat = label["client_stats"]["client_00"]
    assert stat["supported_classes"] == 1 and stat["n_k"] == 30 and stat["class_histogram"]["0"] == 30 and stat["shannon_entropy"] == 0.0
    skewed_iid = build_partition_spec("r", "9" * 64, train, {"iid": {**schemes["iid"], "shards": skew}}, 1)
    assert skewed_iid["scenarios"]["iid"]["status"] == "FAIL"  # gọi là IID nhưng shard lệch nhãn (JS > 0,05)
    ok_iid = build_partition_spec("r", "d" * 64, [r for rows in iid.values() for r in rows], {"iid": schemes["iid"]}, 1)
    assert ok_iid["scenarios"]["iid"]["status"] == "PASS"
    broken = {**schemes["label_alpha_0_1"], "structural_ok": False}
    assert build_partition_spec("r", "e" * 64, train, {"label_alpha_0_1": broken}, 1)["scenarios"]["label_alpha_0_1"]["status"] == "FAIL"


def test_min_hamming_to_other_split_only_compares_same_source_across_splits():
    d = np.array([0, 1, 0b111, 0], dtype=np.uint64)
    out = min_hamming_to_other_split(d, d, ["a", "a", "a", "b"], ["train", "test", "train", "train"])
    assert out.tolist() == [1, 1, 2, 64]    # a0 gần a1 (Hamming 1); a1 gần a0; a2 gần a1 (2); b duy nhất -> 64
    out = min_hamming_to_other_split(np.array([0b11110000, 0b11110000], dtype=np.uint64), np.array([0, 0], dtype=np.uint64), ["a", "a"], ["train", "test"])
    assert out.tolist() == [0, 0]


def test_verify_image_bytes_reports_mismatching_and_missing_files(tmp_path):
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    good.write_bytes(b"abc")
    bad.write_bytes(b"xyz")
    rows = [{"relative_path": "good.jpg", "byte_sha256": hashlib.sha256(b"abc").hexdigest()},
            {"relative_path": "bad.jpg", "byte_sha256": hashlib.sha256(b"other").hexdigest()},
            {"relative_path": "missing.jpg", "byte_sha256": "0" * 64}]
    checked, wrong = verify_image_bytes(rows, tmp_path, workers=2)
    assert checked == 3 and sorted(wrong) == ["bad.jpg", "missing.jpg"]


def test_quality_audit_passes_only_when_every_gate_is_clean():
    rows = {"train": [{"byte_sha256": "a"}, {"byte_sha256": "b"}], "val": [{"byte_sha256": "c"}]}
    clean = {"train__val": {"group_id": 0, "parent_scene_id": 0, "byte_sha256": 0, "sample_id": 0, "image_id": 0}}
    near = {"unresolved_duplicate_candidates": 0}
    audit = build_quality_audit("r", "f" * 64, rows, clean, 3, [], near, {})
    assert audit["status"] == "PASS" and audit["current_bytes_verified"] is True and audit["within_split_duplicate_rows"] == 0
    assert audit["exact_cross_split_overlaps"]["train/val/group_id"] == 0 and audit["confirmed_or_critical_scene_leakages"] == 0
    assert build_quality_audit("r", "f" * 64, rows, clean, 3, ["x.jpg"], near, {})["status"] == "FAIL"
    assert build_quality_audit("r", "f" * 64, rows, clean, 3, [], {"unresolved_duplicate_candidates": 2}, {})["status"] == "FAIL"
    assert build_quality_audit("r", "f" * 64, {"train": [{"byte_sha256": "a"}, {"byte_sha256": "a"}]}, {}, 2, [], near, {})["within_split_duplicate_rows"] == 1
    leaked = {"train__val": {**clean["train__val"], "group_id": 1}}
    audit = build_quality_audit("r", "f" * 64, rows, leaked, 3, [], near, {})
    assert audit["status"] == "FAIL" and audit["confirmed_or_critical_scene_leakages"] == 1


def test_merge_never_joins_different_classes_and_reports_conflicting_pairs():
    h = np.array([0, 0b11, 0xFFFF], dtype=np.uint64)          # ảnh 0 và 1 cách nhau 2 bit, ảnh 2 xa cả hai
    conflicts: set[int] = set()
    out = merge_near_duplicate_groups(h, ["a", "b", "c"], 4, classes=[1, 2, 1], conflicts=conflicts)
    assert out[0] != out[1] and conflicts == {0, 1}           # khác lớp: không gộp nhóm, báo cả hai ảnh là nhãn mâu thuẫn
    same = merge_near_duplicate_groups(h, ["a", "b", "c"], 4, classes=[1, 1, 1], conflicts=conflicts)
    assert same[0] == same[1] and same[2] == "c" and conflicts == {0, 1}
