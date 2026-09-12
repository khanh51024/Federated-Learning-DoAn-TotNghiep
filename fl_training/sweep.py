"""Controlled Stage-2 experiment matrix and comparison artifact builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
_MPL_CACHE = Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .config import load_training_config


def _condition_name(condition: Dict[str, Any]) -> str:
    parts = [condition["scenario"]]
    if condition.get("alpha") is not None:
        parts.append(f"a{condition['alpha']}")
    if condition.get("quantity_alpha") is not None:
        parts.append(f"q{condition['quantity_alpha']}")
    parts.append(f"feature-{condition.get('feature_skew', 'none')}")
    return "__".join(parts).replace(".", "_")


def _latest_run(path: Path, seed: int, protocol_fingerprint: str,
                semantic_config_hash: str | None = None, max_rounds: int | None = None) -> Path:
    """Select only a completed run with the requested seed and data protocol."""
    matches = []
    for candidate in path.iterdir():
        if not candidate.is_dir():
            continue
        try:
            summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
            config = yaml.safe_load((candidate / "resolved_config.yaml").read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if (summary.get("status") in {"completed", "completed_with_warnings"}
                and config.get("training", {}).get("seed") == seed
                and config.get("data", {}).get("protocol_fingerprint") == protocol_fingerprint
                and (semantic_config_hash is None or config.get("semantic_config_hash") == semantic_config_hash)
                and (max_rounds is None or config.get("federation", {}).get("max_rounds") == max_rounds)):
            matches.append(candidate)
    if not matches:
        raise RuntimeError(f"No completed run for seed={seed}, protocol={protocol_fingerprint} under {path}")
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _result_row(run_dir: Path, condition: Dict[str, Any], seed: int, mode: str) -> Dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if config["training"]["seed"] != seed:
        raise ValueError("Result seed differs from requested sweep seed")
    if summary.get("status") not in {"completed", "completed_with_warnings"}:
        raise ValueError("Cannot collect an incomplete/failed run")
    protocol = config["data"]["protocol_fingerprint"]
    if summary.get("protocol_fingerprint") != protocol:
        raise ValueError("Result protocol differs from resolved config")
    if mode == "local-only":
        clients = summary["clients"]
        if len(clients) != config["federation"]["num_clients"]:
            raise ValueError("Local-only result is missing client models")
        # Verify the underlying evaluations, not just the aggregate summary.
        child_rows = [_result_row(run_dir / f"client_{i:02d}", condition, seed, "local-client")
                      for i in range(len(clients))]
        accuracy = sum(row["accuracy"] for row in child_rows) / len(child_rows)
        macro_f1 = sum(row["macro_f1"] for row in child_rows) / len(child_rows)
        duration = sum(row["duration_seconds"] for row in child_rows)
        processed = sum(row["processed_examples"] for row in child_rows)
        steps = sum(row["optimizer_steps"] for row in child_rows)
        skipped = sum(row["skipped_optimizer_steps"] for row in child_rows)
        payload_bytes = 0
    else:
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if metrics.get("protocol_fingerprint") != protocol or metrics.get("checkpoint") != "best.pt":
            raise ValueError(f"Evaluate best.pt with the matching protocol before collecting {run_dir}")
        from .checkpoint import load_checkpoint, model_state_sha256
        if metrics.get("model_state_sha256") != model_state_sha256(load_checkpoint(run_dir / "best.pt")["model_state_dict"]):
            raise ValueError("Evaluation model weights differ from best checkpoint")
        history = pd.read_csv(run_dir / "history.csv")
        accuracy, macro_f1 = metrics["accuracy"], metrics["macro_f1"]
        duration = float(history["duration_seconds"].sum())
        processed = int(history["processed_examples"].sum())
        steps = int(history["optimizer_steps"].sum())
        skipped = int(history["skipped_optimizer_steps"].sum()) if "skipped_optimizer_steps" in history else 0
        # Round history is committed with last.pt and includes resumed rounds.
        payload_bytes = int(history["payload_bytes"].sum()) if "payload_bytes" in history else 0
    if not all(pd.notna(value) and 0 <= value <= 1 for value in (accuracy, macro_f1)):
        raise ValueError("Invalid accuracy/macro-F1 in sweep result")
    return {
        "condition": _condition_name(condition), "scenario": condition["scenario"],
        "alpha": condition.get("alpha"), "quantity_alpha": condition.get("quantity_alpha"),
        "feature_skew": condition.get("feature_skew", "none"), "seed": seed,
        "mode": mode, "accuracy": accuracy, "macro_f1": macro_f1,
        "best_round": summary.get("best_round"), "completed_rounds": summary.get("completed_rounds"),
        "stop_reason": summary.get("stop_reason"), "duration_seconds": duration,
        "processed_examples": processed, "optimizer_steps": steps,
        "skipped_optimizer_steps": skipped,
        "estimated_payload_bytes": payload_bytes, "payload_scope": "model tensor upload+download; excludes protocol overhead",
        "local_accuracy_std": summary.get("test_accuracy_std"),
        "local_accuracy_min": summary.get("test_accuracy_min"),
        "local_accuracy_max": summary.get("test_accuracy_max"),
        "run_dir": str(run_dir), "protocol_fingerprint": protocol,
        "semantic_config_hash": config.get("semantic_config_hash"),
        "budget_max_rounds": config["federation"]["max_rounds"],
    }


def _write_comparison(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if frame.duplicated(["condition", "seed", "mode"]).any():
            raise ValueError("Duplicate condition/seed/mode results")
        if (frame.groupby(["condition", "seed"])["protocol_fingerprint"].nunique() > 1).any():
            raise ValueError("Baseline/FedAvg data protocols differ; accuracy gap is invalid")
        for field in ("semantic_config_hash", "budget_max_rounds"):
            if (frame.groupby(["condition", "seed"])[field].nunique() > 1).any():
                raise ValueError(f"Baseline/FedAvg {field} differs; controlled comparison is invalid")
        central = frame[frame["mode"] == "centralized"].set_index(["condition", "seed"])["accuracy"]
        frame["centralized_minus_accuracy_pp"] = [
            (central.get((row.condition, row.seed)) - row.accuracy) * 100
            if row.mode == "fedavg" and pd.notna(row.accuracy) and pd.notna(central.get((row.condition, row.seed)))
            else None
            for row in frame.itertuples()
        ]
    frame.to_csv(output_dir / "comparison.csv", index=False)
    frame.to_json(output_dir / "comparison.json", orient="records", indent=2)
    if frame.empty or frame["accuracy"].dropna().empty:
        return
    fig, axis = plt.subplots(figsize=(12, 6))
    pivot = frame.pivot_table(index="condition", columns="mode", values="accuracy", aggfunc="mean") * 100
    pivot.plot(kind="bar", ax=axis)
    axis.set(title="Stage-2 test accuracy by controlled condition", ylabel="Accuracy (%)", xlabel="Condition")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_accuracy.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "comparison_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    statistics = frame.groupby(["condition", "mode"])[["accuracy", "macro_f1"]].agg(["mean", "std", "count"])
    statistics.to_csv(output_dir / "comparison_seed_statistics.csv")
    for metric in ("accuracy", "macro_f1"):
        for axis_name, scenario in (("alpha", "label_skew"), ("quantity_alpha", "quantity_skew")):
            selected = frame[(frame["scenario"] == scenario) & (frame["feature_skew"] == "none")]
            if selected[axis_name].dropna().nunique() < 2:
                continue
            fig, axis = plt.subplots(figsize=(8, 5))
            for mode, group in selected.groupby("mode"):
                means = group.groupby(axis_name)[metric].mean().sort_index()
                axis.plot(means.index, means.values, marker="o", label=mode)
            axis.set(xscale="log", xlabel=axis_name, ylabel=metric,
                     title=f"Test {metric} versus {axis_name} (mean across training seeds)")
            axis.legend()
            fig.tight_layout()
            for extension in ("png", "pdf"):
                fig.savefig(output_dir / f"{metric}_vs_{axis_name}.{extension}", bbox_inches="tight")
            plt.close(fig)


def run_sweep(sweep_config: str | Path, execute: bool = False, collect: bool = False) -> Path:
    if execute and collect:
        raise ValueError("Choose execute or collect, not both")
    sweep_file = Path(sweep_config).resolve()
    spec = yaml.safe_load(sweep_file.read_text(encoding="utf-8"))
    package_root = Path(__file__).resolve().parents[1]
    base_config_path = (package_root / spec["base_config"]).resolve()
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    resolved_base = load_training_config(base_config_path, base_dir=package_root)
    sweep_id = spec.get("sweep_id") or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir = (package_root / spec.get("output_root", "runs/sweeps") / sweep_id).resolve()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    modes = list(spec.get("modes", ["centralized", "local-only", "fedavg"]))
    seeds = [int(seed) for seed in spec.get("seeds", [42])]
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"centralized", "local-only", "fedavg"}:
        raise ValueError("Invalid or duplicate sweep modes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid or duplicate sweep seeds")
    names = [_condition_name(c) for c in spec["conditions"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate condition names; use separate sweeps for different split seeds")
    plan_rows, result_rows = [], []
    for condition in spec["conditions"]:
        name = _condition_name(condition)
        for seed in seeds:
            generated = json.loads(json.dumps(base))
            # Clear stale skew axes inherited from the base config.
            generated["data"].update({"alpha": None, "quantity_alpha": None, "feature_skew": "none"})
            generated["data"].update(condition)
            generated["data"]["dataset_root"] = str(resolved_base.data.dataset_root)
            generated["data"]["partition_index"] = str(resolved_base.data.partition_index)
            if resolved_base.data.bundle_path is not None:
                generated["data"]["bundle_path"] = str(resolved_base.data.bundle_path)
            # Training seeds vary while the held-out data split remains fixed.
            generated["data"]["split_seed"] = int(condition.get("split_seed", resolved_base.data.split_seed))
            generated["training"]["seed"] = seed
            generated["output"]["evaluate_test_after_train"] = True
            for mode in modes:
                mode_root = output_dir / name / f"seed-{seed}" / mode.replace("-", "_")
                generated["output"]["root"] = str(mode_root)
                config_path = config_dir / f"{name}__seed-{seed}__{mode}.yaml"
                config_path.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
                available, reason, cfg = True, "ready", None
                try:
                    cfg = load_training_config(config_path, base_dir=package_root, mode=mode)
                except Exception as exc:
                    available, reason = False, str(exc)
                plan_rows.append({"condition": name, "seed": seed, "mode": mode,
                                  "config": str(config_path), "partition_available": available, "status": reason})
                if not execute and not collect:
                    continue
                if not available:
                    raise RuntimeError(f"Partition unavailable for {name}: {reason}")
                if collect and not mode_root.exists():
                    # Backward-compatible discovery, with seed/protocol validation.
                    mode_root = output_dir / name / mode.replace("-", "_")
                if execute:
                    command = [sys.executable, "-m", "fl_training.cli"]
                    command += (["train", "--config", str(config_path), "--no-progress"] if mode == "fedavg"
                                else ["baseline", "--config", str(config_path), "--mode", mode])
                    existing = set(mode_root.iterdir()) if mode_root.exists() else set()
                    # timeout_seconds is a per-round timeout, not the budget of the full experiment.
                    # An optional whole-run budget belongs explicitly to the sweep config.
                    timeout = spec.get("run_timeout_seconds")
                    process = subprocess.Popen(command, cwd=str(package_root))
                    try:
                        code = process.wait(timeout=float(timeout) if timeout is not None else None)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                        else:
                            process.terminate()
                        process.wait(timeout=15)
                        raise
                    if code:
                        raise RuntimeError(f"{mode} failed for {name}, seed {seed}, exit {code}")
                    run_dir = _latest_run(mode_root, seed, cfg.data.protocol_fingerprint,
                                          cfg.semantic_config_hash, cfg.federation.max_rounds)
                    if run_dir in existing:
                        raise RuntimeError("Execution did not produce a new completed run")
                else:
                    if not mode_root.exists():
                        raise RuntimeError(f"Missing sweep results under {mode_root}")
                    run_dir = _latest_run(mode_root, seed, cfg.data.protocol_fingerprint,
                                          cfg.semantic_config_hash, cfg.federation.max_rounds)
                result_rows.append(_result_row(run_dir, condition, seed, mode))
                # Preserve partial results if a later experiment fails.
                _write_comparison(result_rows, output_dir)
    pd.DataFrame(plan_rows).to_csv(output_dir / "sweep_plan.csv", index=False)
    (output_dir / "sweep_plan.json").write_text(json.dumps(plan_rows, indent=2), encoding="utf-8")
    # A dry-run must never erase a completed comparison.
    if execute or collect:
        _write_comparison(result_rows, output_dir)
    return output_dir
