"""scripts/report_data_gaps.py: trần Local-only tính từ phân vùng của release."""
from __future__ import annotations

from collections import Counter

from scripts.report_data_gaps import local_only_ceilings, shards_from_partition_spec


def test_ceiling_is_the_share_of_images_in_classes_the_client_has_seen():
    classes = ["a___x", "a___y", "b___z"]
    spec = {"scenarios": {"label_alpha_0_1": {"client_stats": {
        "client_00": {"class_histogram": {"0": 5, "1": 0, "2": 0}},
        "client_01": {"class_histogram": {"0": 1, "1": 9, "2": 3}},
    }}}}
    shards = shards_from_partition_spec(spec, classes)
    assert shards["client_00"]["per_class"] == {"a___x": 5, "a___y": 0, "b___z": 0}
    out = local_only_ceilings(classes, {"set": Counter({"a___x": 2, "a___y": 1, "b___z": 1})}, shards)
    assert out["client_00"] == {"classes_seen": 1, "set": 50.0}      # chỉ thấy lớp a___x: 2/4 ảnh có thể đoán đúng
    assert out["client_01"] == {"classes_seen": 3, "set": 100.0}
