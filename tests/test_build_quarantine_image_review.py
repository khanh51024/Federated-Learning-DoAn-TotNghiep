"""scripts/build_quarantine_image_review.py và phần nhận lá cách ly của scripts/build_release_v4.py."""
from __future__ import annotations

import random

from scripts.build_quarantine_image_review import extend_sample
from scripts.build_release_v4 import promote_quarantine, split_balance


def test_extend_sample_keeps_checked_leaves_in_place_and_appends_new_ones():
    pool = [f"s{i:03d}" for i in range(100)]
    first = extend_sample([], pool, 10, random.Random(1))
    assert len(first) == 10 and first == sorted(first)
    second = extend_sample(first, pool, 25, random.Random(2))
    assert second[:10] == first and len(second) == 25 and len(set(second)) == 25          # số thứ tự cũ không đổi, không lặp lá
    assert extend_sample(second, pool, 25, random.Random(3)) == second                      # đủ rồi thì không thêm
    assert len(extend_sample(second, pool[:30], 90, random.Random(4))) <= 30 + len(second)  # kho cạn thì dừng, không lỗi


def _q(sample_id: str, target: str = "Blueberry___healthy", **kw) -> dict:
    return {"sample_id": sample_id, "source_domain": "plantdoc", "quarantine_status": "clean", "target_class": target, "class_id": 2,
            "split": "train", "supervised_eligible": False, **kw}


def test_promote_quarantine_accepts_by_class_or_by_single_leaf_and_keeps_the_v3_split():
    rows = [_q("a"), _q("b"), _q("c", "Peach___healthy", split="calibration"), _q("d", split="quarantine"), _q("e", quarantine_status="dirty"),
            _q("f", class_id=-1), _q("g", source_domain="plantvillage")]
    promoted, kept = promote_quarantine(rows, class_ok={"Peach___healthy"}, sample_ok={"b"})
    assert [r["sample_id"] for r in promoted] == ["b", "c"]
    assert all(r["supervised_eligible"] is True and r["label_trust"] == "reviewed_quarantine" for r in promoted)
    assert {r["sample_id"]: r["split"] for r in promoted} == {"b": "train", "c": "calibration"}
    assert [r["sample_id"] for r in kept] == ["a", "d", "e", "f", "g"] and all(r["supervised_eligible"] is False for r in kept)
    assert rows[1]["supervised_eligible"] is False                                           # không sửa dòng gốc


def _ext(group: str, split: str, source: str = "src", target: str = "T") -> dict:
    return {"source_dataset": source, "target_class": target, "group_id": group, "split": split}


def test_split_balance_flags_test_heavy_pairs_and_warns_about_one_big_group():
    ok = [_ext(f"g{i}", "train") for i in range(80)] + [_ext(f"v{i}", "val") for i in range(10)] + [_ext(f"t{i}", "legacy_diagnostic_test") for i in range(10)]
    assert split_balance(ok)["violations"] == [] and split_balance(ok)["warnings"] == []
    heavy = [_ext(f"g{i}", "train") for i in range(50)] + [_ext("big", "legacy_diagnostic_test") for _ in range(50)]     # 50% test do một nhóm
    result = split_balance(heavy)
    assert result["violations"] == ["src|T"] and result["pairs"]["src|T"]["largest_group_share"] == 0.5
    lopsided = [_ext("big", "train") for _ in range(30)] + [_ext(f"v{i}", "val") for i in range(70)]                       # train < 40%
    assert split_balance(lopsided)["violations"] == ["src|T"]
    warn = [_ext("big", "train") for _ in range(30)] + [_ext(f"g{i}", "train") for i in range(60)] + [_ext(f"t{i}", "legacy_diagnostic_test") for i in range(10)]
    assert split_balance(warn)["violations"] == [] and split_balance(warn)["warnings"] == ["src|T"]
    assert split_balance([_ext("a", "legacy_diagnostic_test")] * 99)["pairs"] == {}                                        # dưới 100 ảnh: không xét
