"""Test phần thuần của scripts/eval_quarantined_plantdoc.py: chọn lá, gom theo nhóm và theo mapping_status."""
from __future__ import annotations

from scripts import eval_quarantined_plantdoc as eq

CLASSES = ["A___healthy", "B___rust", "C___blight"]


def _row(i: int, cls: int, status: str = "clean", mapping: str = "assumed_healthy", domain: str = "plantdoc") -> dict:
    return {"sample_id": f"s{i}", "class_id": cls, "target_class": CLASSES[cls], "quarantine_status": status,
            "mapping_status": mapping, "source_domain": domain}


def test_select_rows_keeps_only_clean_plantdoc():
    rows = [
        _row(0, 0), _row(1, 1, status="quarantined_label_conflict"), _row(2, 2, status="cross_split_visual_review_v3"),
        _row(3, 0, domain="plantvillage"), _row(4, 2, mapping="ambiguous_alias"),
    ]
    assert [r["sample_id"] for r in eq.select_rows(rows)] == ["s0", "s4"]


def test_summarize_counts_correct_per_class_mapping_and_most_common_prediction():
    rows = [_row(0, 0), _row(1, 0), _row(2, 0), _row(3, 1, mapping="ambiguous_alias"), _row(4, 1, mapping="ambiguous_alias")]
    perfect = [r["class_id"] for r in rows]
    always_zero = [0] * len(rows)
    preds = {m: list(always_zero) for m in eq.MODEL_ORDER}
    preds["centralized"] = list(perfect)
    out = eq.summarize(rows, preds, CLASSES)
    assert out["overall"]["n"] == 5
    assert out["overall"]["centralized"] == {"correct": 5, "top1": 1.0}
    assert out["overall"]["fedavg"] == {"correct": 3, "top1": 0.6}
    assert out["by_class"]["B___rust"]["n"] == 2
    assert out["by_class"]["B___rust"]["fedavg"]["correct"] == 0
    assert out["by_class"]["B___rust"]["fedavg"]["most_common_prediction"] == {"class": "A___healthy", "share": 1.0}
    assert out["by_mapping_status"]["assumed_healthy"]["n"] == 3
    assert out["by_mapping_status"]["ambiguous_alias"]["fedavg"]["top1"] == 0.0

def test_select_rows_drops_rows_without_taxonomy_label():
    unmapped = {"sample_id": "u", "class_id": -1, "target_class": None, "quarantine_status": "clean",
                "mapping_status": "unmapped_in_classification_yaml", "source_domain": "plantdoc"}
    assert eq.select_rows([unmapped, _row(0, 0)]) == [_row(0, 0)]
