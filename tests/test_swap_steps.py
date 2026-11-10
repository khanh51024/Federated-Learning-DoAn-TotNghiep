"""scripts/swap_web_models.py: chọn bước khi đổi gói. Gói chỉ đánh giá trên validation không được chạy bước nào mở tập kiểm thử,
tập calibration (niêm phong) hay dữ liệu ngoài miền, và mọi báo cáo của bước bị bỏ phải được gỡ khỏi chỗ web đọc."""
from __future__ import annotations

import argparse

import pytest

from scripts import swap_web_models as swap


def args(**kw) -> argparse.Namespace:
    defaults = {"skip": None, "validation_only": False, "skip_slow": False, "only": None, "start": None}
    return argparse.Namespace(**{**defaults, **kw})


def names(steps) -> list[str]:
    return [name for name, _command, _slow in steps]


def select(**kw):
    a = args(**kw)
    return swap.select_steps(a.skip_slow, a.only, a.start, swap.skipped_steps(a), a.validation_only)


def test_without_options_every_step_runs_as_before():
    assert names(select()) == [step[0] for step in swap.STEPS]


def test_validation_only_drops_every_step_that_opens_sealed_or_test_data():
    chosen = names(select(validation_only=True))
    assert not set(chosen) & set(swap.SEALED_DATA_STEPS)
    for needed in ("registry", "client_shards", "partition_schemes", "per_class", "domain_gap", "reeval_val", "error_gallery",
                   "community_export", "community_eval_features", "status"):
        assert needed in chosen, needed


def test_validation_only_makes_the_reevaluation_skip_calibration():
    commands = {name: command for name, command, _slow in select(validation_only=True)}
    assert "--validation-only" in commands["reeval_val"]
    plain = {name: command for name, command, _slow in select()}
    assert "--validation-only" not in plain["reeval_val"] and "--holdout" in plain["reeval_holdout"]


def test_the_steps_that_read_test_data_are_exactly_the_ones_that_are_dropped():
    holdout = [step for step in swap.STEPS if "--holdout" in step[1]]
    assert holdout and all(step[0] in swap.SEALED_DATA_STEPS for step in holdout)
    assert set(swap.SKIPPED_STEP_OUTPUTS) <= set(swap.SEALED_DATA_STEPS)


def test_skip_option_removes_named_steps_and_rejects_unknown_ones():
    assert "fullframe" not in names(select(skip="fullframe,external_eval"))
    with pytest.raises(SystemExit):
        select(skip="no_such_step")


def test_only_a_skipped_step_is_refused_and_from_still_filters():
    with pytest.raises(SystemExit):
        select(only="reeval_holdout", validation_only=True)
    after = names(select(start="reeval_val", validation_only=True))
    assert after[0] == "reeval_val" and "statistics" not in after and after[-1] == "status"
