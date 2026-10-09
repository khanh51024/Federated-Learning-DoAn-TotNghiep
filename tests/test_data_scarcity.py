from src.data.partition import calculate_data_scarcity


def test_empty_client_is_high_starvation():
    result = calculate_data_scarcity([], total_classes=38)
    assert result["class_count_missing"] == 38
    assert result["starvation_score"] == 1.0
    assert result["starvation_level"] == "high"


def test_balanced_client_has_lower_starvation_than_single_class_client():
    balanced = calculate_data_scarcity([0, 1, 2, 3] * 25, total_classes=4)
    single = calculate_data_scarcity([0] * 100, total_classes=4)

    assert balanced["class_count_present"] == 4
    assert balanced["class_count_missing"] == 0
    assert balanced["entropy_normalized"] == 1.0
    assert balanced["starvation_score"] < single["starvation_score"]
    assert single["largest_class_ratio"] == 1.0
    assert single["starvation_level"] == "high"


def test_invalid_total_classes_is_rejected():
    try:
        calculate_data_scarcity([0, 1], total_classes=0)
    except ValueError as exc:
        assert "total_classes" in str(exc)
    else:
        raise AssertionError("total_classes <= 0 must fail")
