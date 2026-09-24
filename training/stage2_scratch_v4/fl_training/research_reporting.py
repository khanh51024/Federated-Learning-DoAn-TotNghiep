"""Paired training-seed statistics; never manufacture missing observations."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .budget_state import atomic_json

MODES = ("centralized", "fedavg", "local-only")


def validate_research_spec(spec):
    seeds = spec.get("seeds", [])
    if (len(seeds) < 3 or any(type(seed) is not int for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Research matrix requires at least 3 distinct integer training seeds")
    conditions = spec.get("conditions", [])
    alphas = {c.get("alpha") for c in conditions if c["scenario"] == "label_skew"
              and c.get("feature_skew", "none") == "none"}
    if not {100, 10, 1, .5, .1}.issubset(alphas):
        raise ValueError("Research matrix missing required label-skew alpha")
    required = [dict(scenario="iid", feature_skew="none"),
                dict(scenario="iid", feature_skew="moderate"),
                dict(scenario="label_skew", alpha=.1, feature_skew="moderate"),
                dict(scenario="label_quantity_skew", alpha=.1, quantity_alpha=.1, feature_skew="none")]
    required += [dict(scenario="quantity_skew", quantity_alpha=a, feature_skew="none") for a in (.1, 1)]
    if any(not any(all(c.get(k) == v for k, v in item.items()) for c in conditions) for item in required):
        raise ValueError("Research matrix missing declared quantity/feature/mixed condition")
    if any(c.get("split_seed") != 42 for c in conditions):
        raise ValueError("Research protocol holds split_seed=42 fixed across training seeds")


def seed_statistics(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite seed observation")
    n = len(values)
    mean = float(values.mean()) if n else None
    std = float(values.std(ddof=1)) if n > 1 else None
    # At least three independent training seeds are required for reported CI.
    half = float(t.ppf(.975, n - 1) * std / math.sqrt(n)) if n >= 3 else None
    return {"n": n, "mean": mean, "std": std,
            "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None}


def write_research_summary(rows, jobs, output: Path):
    """Called only after checkpoint/evaluation validation by collect_jobs."""
    from .sweep import _condition_name
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("summary.csv", "summary.json", "accuracy_gap*.csv", "accuracy_gap*.png",
                    "accuracy_gap*.pdf", "stage3_candidate.json", "research_status.json"):
        for path in output.glob(pattern):
            path.unlink()
    expected = {}
    for job in jobs:
        condition = job["condition"]
        name = _condition_name(condition)
        key = (name, int(job["seed"]), job["mode"])
        if key in expected:
            raise ValueError("Duplicate expected research job")
        expected[key] = condition
    actual = {}
    for row in rows:
        key = (row["condition"], int(row["seed"]), row["mode"])
        if key in actual or key not in expected:
            raise ValueError("Duplicate or unexpected research result")
        actual[key] = row
    # Protocol/config must also match across seeds, not only between methods.
    for name in {key[0] for key in actual}:
        group = [row for key, row in actual.items() if key[0] == name]
        for field in ("source_fingerprint", "protocol_fingerprint", "budget_max_rounds",
                      "analysis_config_hash"):
            if len({row.get(field) for row in group}) > 1:
                raise ValueError(f"Cannot aggregate seeds with different {field}")
    summary, gaps = [], []
    for name in sorted({key[0] for key in expected}):
        condition = next(value for key, value in expected.items() if key[0] == name)
        seeds = sorted({key[1] for key in expected if key[0] == name})
        for mode in MODES:
            wanted = [seed for seed in seeds if (name, seed, mode) in expected]
            observed = [actual[name, seed, mode] for seed in wanted if (name, seed, mode) in actual]
            for metric in ("accuracy", "macro_f1"):
                stats = seed_statistics([row[metric] * 100 for row in observed])
                summary.append({"condition": name, **condition, "mode": mode, "metric": metric,
                                "unit": "percent", "expected_seeds": len(wanted),
                                "observed_seeds": ",".join(str(row["seed"]) for row in observed),
                                "complete": bool(wanted) and len(observed) == len(wanted), **stats})
        for seed in seeds:
            central, fed = actual.get((name, seed, "centralized")), actual.get((name, seed, "fedavg"))
            if central is None or fed is None:
                continue
            val_c, val_f = central.get("best_val_accuracy"), fed.get("best_val_accuracy")
            gaps.append({"condition": name, **condition, "seed": seed,
                         "accuracy_gap_pp": 100 * (central["accuracy"] - fed["accuracy"]),
                         "macro_f1_gap_pp": 100 * (central["macro_f1"] - fed["macro_f1"]),
                         "validation_accuracy_gap_pp": (100 * (val_c - val_f)
                             if val_c is not None and val_f is not None else None)})
    pd.DataFrame(summary, columns=(list(summary[0]) if summary else
        ["condition", "mode", "metric", "n", "mean", "std", "ci95_low", "ci95_high"])).to_csv(output / "summary.csv", index=False)
    atomic_json(output / "summary.json", summary)
    pd.DataFrame(gaps, columns=(list(gaps[0]) if gaps else
        ["condition", "seed", "accuracy_gap_pp", "macro_f1_gap_pp", "validation_accuracy_gap_pp"])).to_csv(output / "accuracy_gap_paired.csv", index=False)
    gap_summary = []
    for name in sorted({row["condition"] for row in gaps}):
        group = [row for row in gaps if row["condition"] == name]
        for metric in ("accuracy_gap_pp", "macro_f1_gap_pp", "validation_accuracy_gap_pp"):
            values = [row[metric] for row in group if row[metric] is not None]
            gap_summary.append({"condition": name, "scenario": group[0]["scenario"],
                                "alpha": group[0].get("alpha"),
                                "feature_skew": group[0].get("feature_skew", "none"),
                                "metric": metric, **seed_statistics(values)})
    pd.DataFrame(gap_summary, columns=(list(gap_summary[0]) if gap_summary else
        ["condition", "metric", "n", "mean", "std", "ci95_low", "ci95_high"])).to_csv(output / "accuracy_gap_summary.csv", index=False)
    for metric, suffix in (("accuracy_gap_pp", "test"), ("validation_accuracy_gap_pp", "validation")):
        points = sorted([row for row in gap_summary if row["metric"] == metric
                         and row["scenario"] == "label_skew" and row["feature_skew"] == "none"
                         and row["n"] > 0], key=lambda row: row["alpha"])
        if not points:
            continue
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot([row["alpha"] for row in points], [row["mean"] for row in points], "o-")
        for row in points:
            if row["ci95_low"] is not None:
                axis.errorbar(row["alpha"], row["mean"], yerr=row["ci95_high"]-row["mean"], capsize=4)
        axis.axhline(0, color="grey", linewidth=.8)
        axis.set(xscale="log", xlabel="Dirichlet alpha (smaller = stronger label skew)",
                 ylabel="Centralized - FedAvg (percentage points)",
                 title=f"{suffix.title()} accuracy gap; paired seeds, 95% t CI where n >= 3")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"accuracy_gap_vs_alpha_{suffix}.{extension}")
        plt.close(fig)
    complete = bool(expected) and set(actual) == set(expected)
    multiseed = bool(summary) and all(row["expected_seeds"] >= 3 for row in summary)
    ranking = sorted([row for row in gap_summary if row["metric"] == "validation_accuracy_gap_pp"
                      and not (row["scenario"] == "iid" and row["feature_skew"] == "none")],
                     key=lambda row: (-(row["mean"] if row["mean"] is not None else -math.inf), row["condition"]))
    required_counts = {name: len({key[1] for key in expected if key[0] == name})
                       for name in {key[0] for key in expected}}
    non_iid_names = {key[0] for key, cond in expected.items()
                     if not (cond["scenario"] == "iid" and cond.get("feature_skew", "none") == "none")}
    eligible = (complete and multiseed and bool(ranking)
                and {row["condition"] for row in ranking} == non_iid_names
                and all(row["n"] == required_counts[row["condition"]] for row in ranking))
    status = "blocked_incomplete_matrix_or_validation"
    candidate = None
    if eligible:
        status = "no_positive_validation_gap"
        if ranking[0]["mean"] > 0:
            status, candidate = "candidate_requires_scientific_review", ranking[0]
    atomic_json(output / "stage3_candidate.json", {
        "status": status, "candidate": candidate, "ranking": ranking,
        "selection_basis": "largest mean paired validation accuracy gap at validation-loss-selected checkpoints",
        "test_used_for_selection": False, "scientific_stage2_complete": False,
        "limitations": "Exploratory ranking, no multiple-comparison correction; no claim of severe degradation or final Stage-3 acceptance.",
    })
    atomic_json(output / "research_status.json", {
        "expected_jobs": len(expected), "completed_jobs": len(actual), "matrix_complete": complete,
        "multi_seed_coverage_complete": complete and multiseed, "scientific_stage2_complete": False,
        "ci_method": "mean +/- t(0.975, n-1) * sample_std(ddof=1) / sqrt(n); n>=3 only",
        "scope": "training-seed uncertainty on fixed held-out split; Local-only contributes one client-mean per seed",
    })
