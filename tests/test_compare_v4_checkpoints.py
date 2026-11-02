import numpy as np
import pytest

from scripts.compare_v4_checkpoints import prediction_agreement


def _write(path, ids, logits):
    path.mkdir()
    np.savez(path / "logits_best.npz", val=logits, val_sample_ids=ids,
             legacy_diagnostic_test=logits, legacy_diagnostic_test_sample_ids=ids,
             calibration=np.array([object()], dtype=object))


def test_predictions_are_paired_by_id_without_opening_sealed_arrays(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    _write(left, ["a", "b"], [[3, 0], [0, 2]])
    _write(right, ["b", "a"], [[0, 4], [2, 0]])
    result = prediction_agreement(left, right)
    assert result["legacy_diagnostic_test"] == {"n": 2, "same": 2, "agreement": 1.0}
    assert set(result) == {"val", "legacy_diagnostic_test"}


@pytest.mark.parametrize("ids", [["a", "c"], ["a", "a"]])
def test_different_or_duplicate_evaluation_ids_are_rejected(tmp_path, ids):
    left, right = tmp_path / "left", tmp_path / "right"
    _write(left, ["a", "b"], [[3, 0], [0, 2]])
    _write(right, ids, [[0, 4], [2, 0]])
    with pytest.raises(ValueError):
        prediction_agreement(left, right)
