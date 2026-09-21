import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fl_training.research_reporting import seed_statistics, write_research_summary, validate_research_spec
from fl_training.sweep import _condition_name


def fixture_matrix():
    jobs, rows = [], []
    for alpha in (1., .1):
        condition = dict(scenario="label_skew", alpha=alpha, feature_skew="none", split_seed=42)
        name = _condition_name(condition)
        for seed in (42, 123, 2026):
            for mode in ("centralized", "fedavg", "local-only"):
                jobs.append(dict(condition=condition, seed=seed, mode=mode))
                acc = .8 if mode == "centralized" else (.85 if alpha == 1 else .4)
                # Test would select alpha .1; validation must select alpha 1.
                val = .9 if mode == "centralized" else (.5 if alpha == 1 else .8)
                rows.append(dict(condition=name, **condition, seed=seed, mode=mode,
                                 accuracy=acc, macro_f1=acc-.1, best_val_accuracy=val,
                                 source_fingerprint="source", protocol_fingerprint=name,
                                 budget_max_rounds=10, analysis_config_hash="same"))
    return jobs, rows


def test_sample_std_and_t_interval_no_single_seed_numbers():
    stats = seed_statistics([70., 80., 90.])
    assert stats["mean"] == 80
    assert stats["std"] == 10
    assert stats["ci95_high"] == pytest.approx(80 + 4.3026527299 * 10 / 3**.5)
    one = seed_statistics([80.])
    assert one["std"] is None and one["ci95_low"] is None
    assert seed_statistics([])["mean"] is None
    with pytest.raises(ValueError):
        seed_statistics([float("nan")])


def test_paired_gap_selection_uses_validation_and_clears_stale(tmp_path):
    jobs, rows = fixture_matrix()
    write_research_summary(rows, jobs, tmp_path)
    gaps = pd.read_csv(tmp_path / "accuracy_gap_summary.csv")
    row = gaps[(gaps.alpha == 1) & (gaps.metric == "accuracy_gap_pp")].iloc[0]
    assert row["mean"] == pytest.approx(-5)
    assert row["n"] == 3
    selected = json.loads((tmp_path / "stage3_candidate.json").read_text())
    assert selected["candidate"]["alpha"] == 1
    assert selected["test_used_for_selection"] is False
    assert selected["scientific_stage2_complete"] is False
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary[summary["mode"] == "local-only"]["n"].eq(3).all()
    assert (tmp_path / "accuracy_gap_vs_alpha_test.png").exists()
    write_research_summary(rows[:-1], jobs, tmp_path)
    assert json.loads((tmp_path / "stage3_candidate.json").read_text())["candidate"] is None
    write_research_summary([], jobs, tmp_path)
    assert not (tmp_path / "accuracy_gap_vs_alpha_test.png").exists()
    assert pd.read_csv(tmp_path / "summary.csv")["mean"].isna().all()


def test_duplicate_and_cross_seed_mismatch_rejected(tmp_path):
    jobs, rows = fixture_matrix()
    with pytest.raises(ValueError, match="Duplicate"):
        write_research_summary(rows + [rows[0]], jobs, tmp_path)
    rows[3]["source_fingerprint"] = "other-code"
    with pytest.raises(ValueError, match="source_fingerprint"):
        write_research_summary(rows, jobs, tmp_path)


def test_research_matrix_covers_existing_sweeps_and_required_alpha():
    root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((root / "configs/stage2_research.yaml").read_text())
    validate_research_spec(spec)
    assert spec["seeds"] == [42, 123, 2026]
    assert len(spec["conditions"]) == 13
    assert len(spec["conditions"]) * len(spec["seeds"]) * 3 == 117
    bad = copy.deepcopy(spec)
    bad["seeds"] = [42]
    with pytest.raises(ValueError, match="3 distinct"):
        validate_research_spec(bad)
    bad = copy.deepcopy(spec)
    bad["conditions"] = [c for c in bad["conditions"] if c.get("alpha") != .5]
    with pytest.raises(ValueError, match="alpha"):
        validate_research_spec(bad)
