"""Reproducible plots, result tables, and rule-based Stage-2 diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

_MPL_CACHE = Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not np.isfinite(cm).all() or (cm < 0).any():
        raise ValueError("Confusion matrix must be finite, nonnegative and square")
    support = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)


def build_diagnostics(
    history: pd.DataFrame,
    evaluation: Dict[str, Any],
    weights: List[Dict[str, Any]],
    summary: Dict[str, Any],
    runtime_text: str = "",
    thresholds: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Apply transparent rules only when their required evidence is present."""
    findings: List[Dict[str, Any]] = []
    if "skipped_optimizer_steps" in history and history["skipped_optimizer_steps"].sum() > 0:
        findings.append({
            "id": "amp_skipped_steps", "severity": "warning",
            "condition": "GradScaler skipped at least one update because gradients were non-finite",
            "evidence": {"skipped_optimizer_steps": int(history["skipped_optimizer_steps"].sum())},
            "suggestion": "Inspect AMP scale and input magnitudes; compare a short FP32 run if skips persist. Optimizer steps exclude skipped updates.",
        })
    thresholds = thresholds or {}
    divergence_window = int(thresholds.get("divergence_window", 3))
    divergence_increase = float(thresholds.get("divergence_relative_increase", 0.02))
    plateau_rounds = int(thresholds.get("plateau_rounds", 3))
    recall_threshold = float(thresholds.get("low_recall_threshold", 0.5))
    small_client_samples = int(thresholds.get("small_client_samples", 50))
    outlier_factor = float(thresholds.get("update_norm_outlier_factor", 5.0))

    numeric = history.select_dtypes(include=[np.number]) if not history.empty else history
    if not numeric.empty and not np.isfinite(numeric.to_numpy()).all():
        findings.append({
            "id": "nonfinite_metric", "severity": "error",
            "condition": "at least one numeric history value is NaN or Inf",
            "evidence": "history.csv contains a non-finite value",
            "suggestion": "Stop the run; inspect loss scaling, LR, and input tensors before retrying.",
        })

    if len(history) >= divergence_window and "val_loss" in history:
        losses = history["val_loss"].astype(float).to_numpy()
        window = losses[-divergence_window:]
        if np.all(np.diff(window) > 0) and losses[-1] > window[0] * (1 + divergence_increase):
            findings.append({
                "id": "validation_divergence", "severity": "warning",
                "condition": (
                    f"last {divergence_window} val losses strictly increase by more than "
                    f"{divergence_increase * 100:.2f}% overall"
                ),
                "evidence": {"validation_loss_window": window.tolist()},
                "suggestion": "Try a lower LR in a new run and verify client update norms/data profiles.",
            })
        bad_rounds = int(history.iloc[-1].get("bad_rounds", 0))
        if bad_rounds >= plateau_rounds:
            findings.append({
                "id": "validation_plateau", "severity": "info",
                "condition": f"bad_rounds >= {plateau_rounds}",
                "evidence": {"bad_rounds": bad_rounds},
                "suggestion": "Run long enough for the configured scheduler/early stopping; do not tune on test.",
            })

    if summary.get("best_round") == 0 and int(summary.get("last_round", 0)) > 0:
        findings.append({
            "id": "no_improvement_over_initialization", "severity": "warning",
            "condition": "best_round == 0 after at least one training round",
            "evidence": {"best_round": 0, "last_round": summary.get("last_round")},
            "suggestion": "Check LR, labels, preprocessing, and whether the run budget is only a smoke test.",
        })

    per_class = evaluation.get("per_class", [])
    weak = [
        {"class_id": row.get("class_id"), "support": row.get("support"), "recall": row.get("recall")}
        for row in per_class
        if int(row.get("support", 0)) > 0 and float(row.get("recall", 0.0)) < recall_threshold
    ]
    if weak:
        findings.append({
            "id": "low_class_recall", "severity": "warning",
            "condition": f"test support > 0 and recall < {recall_threshold}",
            "evidence": {"affected_classes": weak},
            "suggestion": "Inspect class distribution and errors; preserve the fixed test set when tuning.",
        })

    counts = [int(row["n_k"]) for row in weights if "n_k" in row]
    if counts and min(counts) < small_client_samples:
        findings.append({
            "id": "small_client", "severity": "info",
            "condition": f"n_k < {small_client_samples}",
            "evidence": {"minimum_n_k": min(counts)},
            "suggestion": "Treat this client's update as high variance and verify partition constraints.",
        })
    norms = np.asarray([float(row["update_l2"]) for row in weights if row.get("update_l2") is not None])
    if norms.size >= 3 and float(np.median(norms)) > 0 and float(norms.max()) > outlier_factor * float(np.median(norms)):
        findings.append({
            "id": "update_norm_outlier", "severity": "warning",
            "condition": f"maximum update L2 > {outlier_factor}x median",
            "evidence": {"maximum": float(norms.max()), "median": float(np.median(norms))},
            "suggestion": "Inspect that client's data/profile and consider a lower LR in a separate run.",
        })
    if any(int(row.get("nonfinite_count", 0)) > 0 for row in weights):
        findings.append({
            "id": "nonfinite_weight", "severity": "error",
            "condition": "a client update contains NaN or Inf",
            "evidence": "weights.jsonl nonfinite_count > 0",
            "suggestion": "Reject the round and inspect the named client before training resumes.",
        })
    if "out of memory" in runtime_text.lower():
        findings.append({
            "id": "out_of_memory", "severity": "error", "condition": "runtime log contains OOM",
            "evidence": "runtime.log contains 'out of memory'",
            "suggestion": "Start a new comparable run with batch_size 8; do not change batch size mid-run.",
        })
    if "access violation" in runtime_text.lower():
        findings.append({
            "id": "runtime_shutdown_warning", "severity": "warning",
            "condition": "Windows runtime log contains access violation",
            "evidence": "runtime.log contains 'access violation'",
            "suggestion": "Verify Ray workers exited; prefer the documented WSL2 command before a full run.",
        })
    return findings


def _save_figure(fig: plt.Figure, base_path: Path) -> List[str]:
    outputs = []
    for suffix in (".png", ".pdf"):
        path = base_path.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def generate_report(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    report_dir = run_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_path / "history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    evaluation = _read_json(run_path / "test_metrics.json", {})
    summary = _read_json(run_path / "summary.json", {})
    weights = _read_jsonl(run_path / "weights.jsonl")
    runtime_text = (run_path / "runtime.log").read_text(encoding="utf-8", errors="replace") if (run_path / "runtime.log").exists() else ""
    thresholds: Dict[str, Any] = {}
    resolved_config = run_path / "resolved_config.yaml"
    if resolved_config.exists():
        with open(resolved_config, "r", encoding="utf-8") as file:
            thresholds = (yaml.safe_load(file) or {}).get("diagnostics", {})
    plot_files: List[str] = []

    if not history.empty:
        rounds = history["round"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        train_label = "train" if summary.get("mode") in {"centralized", "local-only"} else "local train (pre-aggregate)"
        axes[0, 0].plot(rounds, history["train_loss"], label=train_label)
        axes[0, 0].plot(rounds, history["val_loss"], label="global validation")
        axes[0, 0].set(title="Loss", xlabel="Round", ylabel="Cross-entropy")
        axes[0, 0].legend()
        axes[0, 1].plot(rounds, history["train_accuracy"] * 100, label=train_label)
        axes[0, 1].plot(rounds, history["val_accuracy"] * 100, label="global validation")
        axes[0, 1].set(title="Accuracy", xlabel="Round", ylabel="Percent")
        axes[0, 1].legend()
        axes[1, 0].plot(rounds, history["val_macro_f1"])
        axes[1, 0].set(title="Validation macro-F1", xlabel="Round", ylabel="Macro-F1")
        axes[1, 1].step(rounds, history["lr"], where="post")
        axes[1, 1].set(title="Learning rate used", xlabel="Round", ylabel="LR")
        fig.suptitle(f"Training curves — {summary.get('run_id', run_path.name)}")
        fig.tight_layout()
        plot_files += _save_figure(fig, report_dir / "learning_curves")

    if weights:
        weight_frame = pd.DataFrame(weights)
        if {"round", "client_id", "update_l2"}.issubset(weight_frame.columns):
            fig, axis = plt.subplots(figsize=(10, 5))
            for client_id, group in weight_frame.groupby("client_id"):
                axis.plot(group["round"], group["update_l2"], marker="o", label=f"client {client_id}")
            axis.set(title="Client update L2 norms", xlabel="Round", ylabel="L2 norm")
            axis.legend(ncol=2, fontsize=8)
            plot_files += _save_figure(fig, report_dir / "update_norms")

    if evaluation.get("confusion_matrix") is not None:
        cm = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
        normalized = normalize_confusion_matrix(cm)
        if evaluation.get("total_samples") is not None and int(cm.sum()) != evaluation["total_samples"]:
            raise ValueError("Confusion matrix count differs from evaluated sample count")
        if cm.sum() and evaluation.get("accuracy") is not None and not np.isclose(np.trace(cm) / cm.sum(), evaluation["accuracy"]):
            raise ValueError("Confusion matrix accuracy differs from evaluation accuracy")
        np.savetxt(report_dir / "confusion_matrix_normalized.csv", normalized, delimiter=",")
        labels = evaluation.get("class_names") or [
            str(row.get("class_name", row.get("class_id", idx)))
            for idx, row in enumerate(evaluation.get("per_class", []))
        ]
        for matrix, name, title in (
            (cm, "confusion_matrix_raw", "Raw confusion matrix (counts)"),
            (normalized, "confusion_matrix_normalized", "Row-normalized confusion matrix"),
        ):
            fig, axis = plt.subplots(figsize=(14, 12))
            image = axis.imshow(matrix, cmap="Blues", aspect="auto")
            fig.colorbar(image, ax=axis, fraction=0.046)
            axis.set(title=f"{title} — {evaluation.get('checkpoint', 'evaluation')}, n={cm.sum()}", xlabel="Predicted class", ylabel="True class")
            if labels:
                axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
                axis.set_yticks(range(len(labels)), labels, fontsize=6)
            plot_files += _save_figure(fig, report_dir / name)
        pd.DataFrame(evaluation.get("per_class", [])).to_csv(report_dir / "per_class_metrics.csv", index=False)
        if evaluation.get("per_class"):
            per_class = pd.DataFrame(evaluation["per_class"])
            fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
            axes[0].bar(per_class["class_id"], per_class["f1"])
            axes[0].set(ylabel="F1", title="Per-class F1 on the held-out test set", ylim=(0, 1))
            axes[1].bar(per_class["class_id"], per_class["support"])
            axes[1].set(ylabel="Samples", xlabel="True class", title="Test support per class")
            if labels:
                axes[1].set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            fig.tight_layout()
            plot_files += _save_figure(fig, report_dir / "per_class_f1_support")

    diagnostics = build_diagnostics(history, evaluation, weights, summary, runtime_text, thresholds)
    with open(report_dir / "diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2, ensure_ascii=False)
    markdown = ["# Stage-2 diagnostics", "", f"Run: `{summary.get('run_id', run_path.name)}`", ""]
    if not diagnostics:
        markdown.append("Không đủ bằng chứng để kích hoạt cảnh báo theo các quy tắc hiện tại.")
    for finding in diagnostics:
        markdown.extend([
            f"## {finding['id']} ({finding['severity']})", "",
            f"- Điều kiện: {finding['condition']}",
            f"- Bằng chứng: `{json.dumps(finding['evidence'], ensure_ascii=False)}`",
            f"- Gợi ý: {finding['suggestion']}", "",
        ])
    (report_dir / "diagnostics.md").write_text("\n".join(markdown), encoding="utf-8")

    result = {
        "schema_version": 1, "run_id": summary.get("run_id", run_path.name),
        "checkpoint": evaluation.get("checkpoint"),
        "evaluation_id": evaluation.get("evaluation_id"),
        "model_state_sha256": evaluation.get("model_state_sha256"),
        "metrics": {key: evaluation.get(key) for key in ("loss", "accuracy", "macro_f1", "total_samples")},
        "plots": plot_files, "diagnostic_count": len(diagnostics),
        "scientific_stage2_complete": False,
        "note": "Artifacts demonstrate code execution only; scientific Stage-2 requires the full controlled protocol.",
    }
    with open(report_dir / "report.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result
