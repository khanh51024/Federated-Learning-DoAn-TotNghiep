"""Controlled Stage-2 experiment matrix and comparison artifact builder."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

if "MPLCONFIGDIR" not in os.environ:
    if os.environ.get("STAGE2_RUNTIME_DIR"):
        _MPL_CACHE = Path(os.environ["STAGE2_RUNTIME_DIR"]) / "matplotlib"
    else:
        _MPL_CACHE = Path(tempfile.gettempdir()) / "fedavg_mpl_cache"
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

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


def _result_row(run_dir: Path, condition: Dict[str, Any], seed: int, mode: str,
                partition_dir_override: Path | None = None) -> Dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if config["training"]["seed"] != seed:
        raise ValueError("Result seed differs from requested sweep seed")
    if summary.get("status") not in {"completed", "completed_with_warnings"}:
        raise ValueError("Cannot collect an incomplete/failed run")
    expected_mode = "local-only" if mode == "local-client" else mode
    expected_config_mode = "train" if mode == "fedavg" else expected_mode
    if (config.get("mode") != expected_config_mode
            or summary.get("mode", expected_mode) != expected_mode):
        raise ValueError("Result mode differs from requested mode")
    protocol = config["data"]["protocol_fingerprint"]
    if summary.get("protocol_fingerprint") != protocol:
        raise ValueError("Result protocol differs from resolved config")
    if mode == "local-only":
        clients = summary["clients"]
        if len(clients) != config["federation"]["num_clients"]:
            raise ValueError("Local-only result is missing client models")
        # Verify the underlying evaluations, not just the aggregate summary.
        child_rows = [_result_row(run_dir / f"client_{i:02d}", condition, seed, "local-client",
                                  partition_dir_override)
                      for i in range(len(clients))]
        if [row["client_id"] for row in child_rows] != list(range(len(clients))):
            raise ValueError("Local-only client identities are incomplete or reordered")
        for child in child_rows:
            for field in ("semantic_config_hash", "source_fingerprint", "protocol_fingerprint"):
                expected = protocol if field == "protocol_fingerprint" else config.get(field)
                if child.get(field) != expected:
                    raise ValueError(f"Local-only child differs from parent {field}")
            if child["budget_max_rounds"] != config["federation"]["max_rounds"]:
                raise ValueError("Local-only child budget differs from parent")
        accuracy = sum(row["accuracy"] for row in child_rows) / len(child_rows)
        macro_f1 = sum(row["macro_f1"] for row in child_rows) / len(child_rows)
        duration = sum(row["duration_seconds"] for row in child_rows)
        processed = sum(row["processed_examples"] for row in child_rows)
        steps = sum(row["optimizer_steps"] for row in child_rows)
        skipped = sum(row["skipped_optimizer_steps"] for row in child_rows)
        payload_bytes = None
        local_accuracy_values = pd.Series([row["accuracy"] for row in child_rows], dtype=float)
        local_f1_values = pd.Series([row["macro_f1"] for row in child_rows], dtype=float)
        best_val_accuracy = None
    else:
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if (metrics.get("protocol_fingerprint") != protocol or metrics.get("checkpoint") != "best.pt"
                or metrics.get("run_id") != config.get("run_id")):
            raise ValueError(f"Evaluate best.pt with the matching protocol before collecting {run_dir}")
        from .checkpoint import load_checkpoint, model_state_sha256
        best_payload = load_checkpoint(run_dir / "best.pt")
        checkpoint_config = best_payload.get("config", {})
        if (checkpoint_config.get("mode") != config.get("mode")
                or checkpoint_config.get("training", {}).get("seed") != seed
                or checkpoint_config.get("semantic_config_hash") != config.get("semantic_config_hash")
                or checkpoint_config.get("client_id") != config.get("client_id")):
            raise ValueError("Best checkpoint config identity differs from resolved config")
        if metrics.get("model_state_sha256") != model_state_sha256(best_payload["model_state_dict"]):
            raise ValueError("Evaluation model weights differ from best checkpoint")
        identity_payload = {key: value for key, value in metrics.items() if key != "evaluation_id"}
        expected_evaluation_id = __import__("hashlib").sha256(
            json.dumps(identity_payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()[:24]
        if metrics.get("evaluation_id") != expected_evaluation_id:
            raise ValueError("Evaluation identity is missing or does not match its metrics")
        history = pd.read_csv(run_dir / "history.csv")
        best_val_accuracy = best_payload.get("metrics", {}).get("accuracy")
        if best_val_accuracy is None and "val_accuracy" in history and "round" in history:
            selected = history[history["round"] == best_payload.get("best_round")]
            if len(selected) == 1:
                best_val_accuracy = float(selected.iloc[0]["val_accuracy"])
        if best_val_accuracy is not None and not (pd.notna(best_val_accuracy) and 0 <= best_val_accuracy <= 1):
            raise ValueError("Invalid best validation accuracy")
        accuracy, macro_f1 = metrics["accuracy"], metrics["macro_f1"]
        duration = float(history["duration_seconds"].sum())
        processed = int(history["processed_examples"].sum())
        steps = int(history["optimizer_steps"].sum())
        skipped = int(history["skipped_optimizer_steps"].sum()) if "skipped_optimizer_steps" in history else 0
        # Round history is committed with last.pt and includes resumed rounds.
        payload_bytes = int(history["payload_bytes"].sum()) if mode == "fedavg" and "payload_bytes" in history else None
    if not all(pd.notna(value) and 0 <= value <= 1 for value in (accuracy, macro_f1)):
        raise ValueError("Invalid accuracy/macro-F1 in sweep result")
    partition_dir = partition_dir_override or Path(config.get("data", {}).get("partition_dir", ""))
    diagnostics = {}
    metadata_path = partition_dir / "partition_config.json"
    matrix_path = partition_dir / "client_class_matrix.csv"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        diagnostics = {
            "partition_summary_metrics": metadata.get("summary_metrics"),
            "partition_repair_diagnostics": metadata.get("split_diagnostics"),
            "leaf_map_coverage": metadata.get("leaf_group_audit"),
        }
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, index_col=0)
        diagnostics["client_class_histogram"] = matrix.astype(int).values.tolist()
    analysis_config = {key: config.get(key) for key in
                       ("model", "training", "federation", "early_stopping", "lr_scheduler")}
    analysis_config["training"] = {key: value for key, value in config.get("training", {}).items() if key != "seed"}
    return {
        "condition": _condition_name(condition), "scenario": condition["scenario"],
        "alpha": condition.get("alpha"), "quantity_alpha": condition.get("quantity_alpha"),
        "feature_skew": condition.get("feature_skew", "none"), "seed": seed,
        "mode": mode, "accuracy": accuracy, "macro_f1": macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "analysis_config_hash": hashlib.sha256(json.dumps(analysis_config, sort_keys=True).encode()).hexdigest(),
        "client_id": config.get("client_id"),
        "best_round": summary.get("best_round"), "completed_rounds": summary.get("completed_rounds"),
        "stop_reason": summary.get("stop_reason"), "duration_seconds": duration,
        "processed_examples": processed, "optimizer_steps": steps,
        "skipped_optimizer_steps": skipped,
        "payload_bytes": payload_bytes,
        "payload_label": "measured_tensor_bytes" if mode == "fedavg" else "not_applicable",
        "payload_scope": ("serialized model tensor upload+download by completed FedAvg rounds; excludes protocol overhead"
                          if mode == "fedavg" else "in-process baseline; no federated model payload"),
        "local_accuracy_std": float(local_accuracy_values.std(ddof=0)) if mode == "local-only" else None,
        "local_accuracy_min": float(local_accuracy_values.min()) if mode == "local-only" else None,
        "local_accuracy_max": float(local_accuracy_values.max()) if mode == "local-only" else None,
        "local_macro_f1_std": float(local_f1_values.std(ddof=0)) if mode == "local-only" else None,
        "local_macro_f1_min": float(local_f1_values.min()) if mode == "local-only" else None,
        "local_macro_f1_max": float(local_f1_values.max()) if mode == "local-only" else None,
        "run_dir": str(run_dir), "protocol_fingerprint": protocol,
        "source_fingerprint": config.get("source_fingerprint"),
        "semantic_config_hash": config.get("semantic_config_hash"),
        "budget_max_rounds": config["federation"]["max_rounds"],
        **diagnostics,
    }


def _write_comparison(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_patterns = (
        "comparison.csv", "comparison.json", "comparison.md", "comparison_coverage.json",
        "comparison_seed_statistics.csv", "comparison_accuracy.*", "comparison_macro_f1.*",
        "accuracy_vs_*", "macro_f1_vs_*",
    )
    for pattern in managed_patterns:
        for path in output_dir.glob(pattern):
            path.unlink()
    empty_columns = [
        "condition", "scenario", "alpha", "quantity_alpha", "feature_skew", "seed", "mode",
        "accuracy", "macro_f1", "client_id", "best_round", "completed_rounds", "stop_reason",
        "duration_seconds", "processed_examples", "optimizer_steps", "skipped_optimizer_steps",
        "payload_bytes", "payload_label", "payload_scope", "local_accuracy_std",
        "local_accuracy_min", "local_accuracy_max", "local_macro_f1_std", "local_macro_f1_min",
        "local_macro_f1_max", "run_dir", "protocol_fingerprint", "source_fingerprint",
        "semantic_config_hash", "budget_max_rounds", "partition_summary_metrics",
        "partition_repair_diagnostics", "leaf_map_coverage", "client_class_histogram",
        "centralized_minus_accuracy_pp", "centralized_minus_macro_f1_pp",
        "fedavg_minus_local_accuracy_pp", "fedavg_minus_local_macro_f1_pp",
    ]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_columns)
    if not frame.empty:
        if frame.duplicated(["condition", "seed", "mode"]).any():
            raise ValueError("Duplicate condition/seed/mode results")
        if (frame.groupby(["condition", "seed"])["protocol_fingerprint"].nunique() > 1).any():
            raise ValueError("Baseline/FedAvg data protocols differ; accuracy gap is invalid")
        for field in ("semantic_config_hash", "source_fingerprint", "budget_max_rounds"):
            if (frame.groupby(["condition", "seed"])[field].nunique() > 1).any():
                raise ValueError(f"Baseline/FedAvg {field} differs; controlled comparison is invalid")
        central = frame[frame["mode"] == "centralized"].set_index(["condition", "seed"])["accuracy"]
        frame["centralized_minus_accuracy_pp"] = [
            (central.get((row.condition, row.seed)) - row.accuracy) * 100
            if row.mode == "fedavg" and pd.notna(row.accuracy) and pd.notna(central.get((row.condition, row.seed)))
            else None
            for row in frame.itertuples()
        ]
    if not frame.empty:
        for metric in ("accuracy", "macro_f1"):
            central_values = frame[frame["mode"] == "centralized"].set_index(["condition", "seed"])[metric]
            local_values = frame[frame["mode"] == "local-only"].set_index(["condition", "seed"])[metric]
            frame[f"centralized_minus_{metric}_pp"] = [
                100 * (central_values.get((row.condition, row.seed), float("nan")) - getattr(row, metric))
                if row.mode == "fedavg" else None for row in frame.itertuples()]
            frame[f"fedavg_minus_local_{metric}_pp"] = [
                100 * (getattr(row, metric) - local_values.get((row.condition, row.seed), float("nan")))
                if row.mode == "fedavg" else None for row in frame.itertuples()]
        coverage = frame.groupby(["condition", "seed"])["mode"].agg(
            lambda values: {"centralized", "fedavg", "local-only"}.issubset(set(values)))
        (output_dir / "comparison_coverage.json").write_text(json.dumps([
            {"condition": key[0], "seed": int(key[1]), "three_modes_complete": bool(value)}
            for key, value in coverage.items()], indent=2), encoding="utf-8")
    seed_count = int(frame["seed"].nunique()) if not frame.empty else 0
    lines = ["# Training comparison", "", f"Observed training seeds: {seed_count}; incomplete results are not full scientific acceptance.",
             "Accuracy/F1 below are percentages; gaps use percentage points.",
             "Local-only is the mean of client models on global test, not site fairness.", ""]
    if frame.empty:
        lines.append("No valid completed main jobs were collected. Missing or invalid jobs are listed in collection_status.json; values are not zero-filled.")
    else:
        lines += ["| Condition | Seed | Mode | Accuracy (%) | Macro-F1 (%) | C−F Acc (pp) | F−L Acc (pp) | Time (h) |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        def gap(value):
            return "" if pd.isna(value) else f"{value:.4f}"
        lines += [f"| {r.condition} | {r.seed} | {r.mode} | {r.accuracy*100:.4f} | {r.macro_f1*100:.4f} | {gap(r.centralized_minus_accuracy_pp)} | {gap(r.fedavg_minus_local_accuracy_pp)} | {r.duration_seconds/3600:.3f} |"
                  for r in frame.itertuples()]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frame.to_csv(output_dir / "comparison.csv", index=False)
    frame.to_json(output_dir / "comparison.json", orient="records", indent=2)
    if frame.empty or frame["accuracy"].dropna().empty:
        return
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        fig, axis = plt.subplots(figsize=(12, 6))
        pivot = frame.pivot_table(index="condition", columns="mode", values=metric, aggfunc="mean") * 100
        pivot.plot(kind="bar", ax=axis)
        axis.set(title=f"Stage-2 test {label} by controlled condition",
                 ylabel=f"{label} (%)", xlabel="Condition")
        axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_{metric}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output_dir / f"comparison_{metric}.pdf", bbox_inches="tight")
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
    protocol_audit = None
    if spec.get("protocol") is not None:
        from .stage2_protocol import validate_stage2_sweep
        protocol_audit = validate_stage2_sweep(spec, resolved_base, package_root)
    sweep_id = spec.get("sweep_id") or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_dir = (package_root / spec.get("output_root", "runs/sweeps") / sweep_id).resolve()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    if protocol_audit is not None:
        (output_dir / "protocol_audit.json").write_text(json.dumps(protocol_audit, indent=2), encoding="utf-8")
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
