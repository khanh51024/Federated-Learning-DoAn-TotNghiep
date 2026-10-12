"""Test phần thuần của scripts/build_label_review.py: lấy mẫu tái lập, quy tắc nhận nhóm, trang không lọt </script>."""
from __future__ import annotations

from scripts import build_label_review as lr


def _rows(n_a: int, n_b: int) -> list[dict]:
    rows = [{"sample_id": f"a{i:03d}", "class_id": 0, "target_class": "A___healthy", "source_domain": "plantdoc", "quarantine_status": "clean",
             "mapping_status": "assumed_healthy", "relative_path": f"x/a{i}.jpg", "annotation_id": f"ann_a{i}"} for i in range(n_a)]
    rows += [{"sample_id": f"b{i:03d}", "class_id": 1, "target_class": "B___rust", "source_domain": "plantdoc", "quarantine_status": "clean",
              "mapping_status": "ambiguous_alias", "relative_path": f"x/b{i}.jpg", "annotation_id": f"ann_b{i}"} for i in range(n_b)]
    return rows


def test_select_clean_plantdoc_drops_conflicts_and_plantvillage():
    rows = _rows(2, 1) + [
        {"sample_id": "c", "class_id": 0, "target_class": "A___healthy", "source_domain": "plantdoc", "quarantine_status": "quarantined_label_conflict"},
        {"sample_id": "d", "class_id": 0, "target_class": "A___healthy", "source_domain": "plantvillage", "quarantine_status": "clean"},
    ]
    assert {r["sample_id"] for r in lr.select_clean_plantdoc(rows)} == {"a000", "a001", "b000"}


def test_sampling_is_reproducible_capped_and_order_independent():
    rows = _rows(100, 7)
    first = lr.sample_per_class(rows, 30, seed=1)
    again = lr.sample_per_class(list(reversed(rows)), 30, seed=1)
    assert first == again
    assert len(first["A___healthy"]) == 30 and len(first["B___rust"]) == 7  # nhóm ít lá lấy hết
    assert lr.sample_per_class(rows, 30, seed=2)["A___healthy"] != first["A___healthy"]
    assert len({r["sample_id"] for v in first.values() for r in v}) == 37  # không trùng


def test_accept_rule_needs_complete_review_and_thresholds():
    ids = [f"s{i}" for i in range(10)]
    allok = {i: "dung" for i in ids}
    assert lr.classify(allok, ids)["accepted"] is True
    one_wrong = {**allok, "s0": "sai"}  # 9/10 = 90% đúng: vừa đủ
    assert lr.classify(one_wrong, ids)["accepted"] is True
    two_wrong = {**allok, "s0": "sai", "s1": "sai"}
    assert lr.classify(two_wrong, ids)["accepted"] is False
    three_unsure = {**{i: "dung" for i in ids}, "s0": "khong_ro", "s1": "khong_ro", "s2": "khong_ro"}
    assert lr.classify(three_unsure, ids)["accepted"] is False  # 70% đúng và 30% không rõ
    partial = {i: "dung" for i in ids[:9]}
    assert lr.classify(partial, ids)["complete"] is False and lr.classify(partial, ids)["accepted"] is False


def test_page_embeds_data_without_breaking_out_of_the_script_tag():
    sampled = {"A___healthy": [_rows(1, 0)[0]]}
    page = lr.build_page(sampled, {"A___healthy": 5}, {"ann_a0": "</script><b>x"}, seed=3, per_class=30, release_sha="0" * 64, images_rel="../ds")
    assert page.count("</script>") == 1  # chỉ thẻ đóng thật
    assert "../ds/x/a0.jpg" in page and "A___healthy" in page

def test_select_clean_plantdoc_drops_unmapped_labels():
    unmapped = {"sample_id": "u", "class_id": -1, "target_class": None, "quarantine_status": "clean", "source_domain": "plantdoc"}
    assert lr.select_clean_plantdoc([unmapped] + _rows(1, 0)) == _rows(1, 0)
