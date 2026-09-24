"""Collector độc lập chạy trên CPU cho profile stage1_compat.

Thu thập kết quả FedAvg mới, đối chiếu với baseline lịch sử GĐ1,
tính reference gap (điểm phần trăm, giữ nguyên dấu âm), xuất comparison (csv/json/md)
và biểu đồ tham chiếu độc lập.
"""

import csv
from stage1_compat.integrity import atomic_json, read_json, file_hash
from stage1_compat.artifacts import validate_completed
from stage1_compat.budget import RunnerLock
import json
import os
from pathlib import Path
from typing import Any

from stage1_compat.baselines import BaselineRegistry
from stage1_compat.constants import (
    DEFAULT_ALPHAS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    SEED,
    UPSTREAM_COMMIT,
)


def _collect(
    output_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Any]:
    resolved_output = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    reports_dir = resolved_output / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Invalidate previous reports before any input validation may fail.
    for name in ("comparison.csv", "comparison.json", "comparison.md", "collection_status.json",
                 "stage1_accuracy_comparison.png", "stage1_f1_comparison.png", "stage1_alpha_gap.png"):
        (reports_dir / name).unlink(missing_ok=True)
    atomic_json(reports_dir / "collection_status.json", {"status": "COLLECTING", "scientific_stage2_complete": False})
    try:
        registry = BaselineRegistry()
    except Exception as error:
        atomic_json(reports_dir / "collection_status.json", {"status": "INVALID_BASELINES", "error": str(error),
                                                            "scientific_stage2_complete": False})
        raise
    job_statuses = {}
    errors = {}

    # 1. Đọc kết quả Centralized tham chiếu
    cent_entry = registry.get_centralized(SEED)

    # 2. Đọc kết quả Local-only tham chiếu
    local_entries = {
        alpha: registry.get_local_only(alpha, SEED)
        for alpha in DEFAULT_ALPHAS
    }

    # 3. Đọc kết quả FedAvg lịch sử (đối chiếu tái lập)
    fedavg_hist_entries = {
        alpha: registry.get_historical_fedavg(alpha, SEED)
        for alpha in DEFAULT_ALPHAS
    }

    # 4. Đọc kết quả FedAvg MỚI từ output_dir
    fedavg_new_entries: dict[float, dict[str, Any] | None] = {}
    for alpha in DEFAULT_ALPHAS:
        alpha_label = f"{int(alpha)}" if float(alpha).is_integer() else f"{alpha:g}".replace(".", "_")
        job_dir = resolved_output / f"fedavg_alpha{alpha_label}_seed{SEED}"
        metrics_file = job_dir / "fedavg_metrics.json"
        try:
            if metrics_file.exists():
                payload = validate_completed(job_dir)
                if (payload["seed"] != SEED or payload["alpha"] != alpha
                        or payload["job_id"] != job_dir.name):
                    raise ValueError("Condition/job identity mismatch")
                from stage1_compat.config import JobConfig
                from stage1_compat.identity import generate_job_identity
                expected = generate_job_identity(JobConfig(**payload["resolved_config"]), context=payload["identity"]["context"])
                payload = validate_completed(job_dir, expected)
                fedavg_new_entries[alpha] = payload
                job_statuses[alpha] = "COMPLETED"
            else:
                fedavg_new_entries[alpha] = None
                job_statuses[alpha] = "INCOMPLETE" if job_dir.exists() else "MISSING"
                failed = job_dir / "failure.json"
                if failed.exists():
                    job_statuses[alpha] = "FAILED"
                    errors[str(alpha)] = read_json(failed)
        except Exception as error:
            fedavg_new_entries[alpha] = None
            job_statuses[alpha] = "INVALID"
            errors[str(alpha)] = str(error)

    # 5. Xây dựng bảng so sánh (Comparison Table)
    rows: list[dict[str, Any]] = []

    for alpha in DEFAULT_ALPHAS:
        # Centralized row (dùng chung cho các alpha vì train union tập train)
        cent_acc = cent_entry.metrics["test_accuracy"] if cent_entry else None
        cent_f1 = cent_entry.metrics["test_macro_f1"] if cent_entry else None

        # Local-only row
        loc_entry = local_entries.get(alpha)
        loc_acc_mean = loc_entry.metrics["accuracy_mean"] if loc_entry else None
        loc_acc_std = loc_entry.metrics["accuracy_std"] if loc_entry else None  # std giữa 5 client
        loc_f1_mean = loc_entry.metrics["macro_f1_mean"] if loc_entry else None
        loc_f1_std = loc_entry.metrics["macro_f1_std"] if loc_entry else None

        # FedAvg Historical
        fed_hist = fedavg_hist_entries.get(alpha)
        fed_hist_acc = fed_hist.metrics["test_accuracy"] if fed_hist else None
        fed_hist_f1 = fed_hist.metrics["test_macro_f1"] if fed_hist else None

        # FedAvg New
        fed_new = fedavg_new_entries.get(alpha)
        if fed_new and "test_metrics" in fed_new:
            fed_new_acc = fed_new["test_metrics"]["accuracy"]
            fed_new_f1 = fed_new["test_metrics"]["macro_f1"]
            fed_status = "COMPLETED"
        else:
            fed_new_acc = None
            fed_new_f1 = None
            fed_status = job_statuses[alpha]

        # Tính Reference Gap theo điểm phần trăm (percentage points - pp)
        # gap = (acc_1 - acc_2) * 100, giữ nguyên dấu âm, KHÔNG ép đơn điệu
        gap_cent_minus_fed_pp = None
        gap_fed_minus_local_pp = None
        reproducibility_diff_pp = None

        if cent_acc is not None and fed_new_acc is not None:
            gap_cent_minus_fed_pp = (cent_acc - fed_new_acc) * 100.0

        if fed_new_acc is not None and loc_acc_mean is not None:
            gap_fed_minus_local_pp = (fed_new_acc - loc_acc_mean) * 100.0

        if fed_new_acc is not None and fed_hist_acc is not None:
            reproducibility_diff_pp = (fed_new_acc - fed_hist_acc) * 100.0

        rows.append({
            "alpha": alpha,
            "seed": SEED,
            "centralized_test_acc": cent_acc,
            "centralized_test_f1": cent_f1,
            "local_only_acc_mean": loc_acc_mean,
            "local_only_acc_std_5clients": loc_acc_std,
            "local_only_f1_mean": loc_f1_mean,
            "local_only_f1_std_5clients": loc_f1_std,
            "fedavg_historical_acc": fed_hist_acc,
            "fedavg_historical_f1": fed_hist_f1,
            "fedavg_new_acc": fed_new_acc,
            "fedavg_new_f1": fed_new_f1,
            "fedavg_new_status": fed_status,
            "reference_gap_cent_minus_fed_pp": gap_cent_minus_fed_pp,
            "reference_gap_fed_minus_local_pp": gap_fed_minus_local_pp,
            "reproducibility_diff_pp": reproducibility_diff_pp,
            "strict_gap": None,  # Bắt buộc null vì strict_comparison_eligible=False
            "strict_comparison_eligible": False,
            "provenance": "stage1_compat_reference",
            "rounds": fed_new.get("rounds") if fed_new else None,
            "best_round": fed_new.get("best_round") if fed_new else None,
            "sample_count": fed_new.get("sample_count") if fed_new else None,
            "step_count": fed_new.get("step_count") if fed_new else None,
            "elapsed_seconds": fed_new.get("elapsed_seconds") if fed_new else None,
            "local_only_acc_min": loc_entry.metrics["accuracy_min"] if loc_entry else None,
            "local_only_acc_max": loc_entry.metrics["accuracy_max"] if loc_entry else None,
            "local_only_f1_min": loc_entry.metrics["macro_f1_min"] if loc_entry else None,
            "local_only_f1_max": loc_entry.metrics["macro_f1_max"] if loc_entry else None,
        })

    # 6. Xuất comparison.csv
    csv_path = reports_dir / "comparison.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 7. Xuất comparison.json
    json_path = reports_dir / "comparison.json"
    comparison_payload = {
        "profile": "stage1_compat",
        "upstream_commit": UPSTREAM_COMMIT,
        "reference_only": True,
        "strict_comparison_eligible": False,
        "scientific_stage2_complete": False,
        "notes": [
            "Baseline Centralized và Local-only là dữ liệu lịch sử tham chiếu chỉ đọc.",
            "Local-only std là độ lệch chuẩn giữa 5 clients trên global test set, KHÔNG phải giữa các seeds.",
            "Centralized và FedAvg chọn checkpoint theo best validation accuracy; Local-only chọn final epoch.",
            "GĐ1 thiếu SHA256 ảnh lịch sử và checkpoint client Local-only.",
            "strict_gap để null; reference_gap tính bằng điểm phần trăm (pp).",
        ],
        "table": rows,
        "baseline_provenance": registry.list_all(),
        "new_run_provenance": {str(a): {k: p[k] for k in ("identity", "resolved_config", "payload")} for a, p in fedavg_new_entries.items() if p},
    }
    atomic_json(json_path, comparison_payload)

    # 8. Xuất comparison.md
    md_path = reports_dir / "comparison.md"
    md_content = generate_markdown_report(rows, comparison_payload["notes"])
    md_path.write_text(md_content, encoding="utf-8")

    # 9. Xuất collection_status.json
    status_path = reports_dir / "collection_status.json"
    completed_jobs = [r["alpha"] for r in rows if r["fedavg_new_status"] == "COMPLETED"]
    status_payload = {
        "status": "SUCCESS",
        "total_conditions": len(rows),
        "completed_new_fedavg": len(completed_jobs),
        "expected": list(DEFAULT_ALPHAS),
        **{key.lower(): [a for a, state in job_statuses.items() if state == key]
           for key in ("COMPLETED", "MISSING", "FAILED", "INVALID", "INCOMPLETE")},
        "errors": errors,
        "reference_baselines_loaded": cent_entry is not None and all(local_entries.values()),
        "strict_comparison_eligible": False,
        "scientific_stage2_complete": False,
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
        "comparison_md": str(md_path),
    }
    atomic_json(status_path, status_payload)

    # 10. Vẽ biểu đồ (Agg backend)
    plot_charts(rows, reports_dir)

    return {
        "status": "SUCCESS",
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
        "comparison_md": str(md_path),
        "collection_status": str(status_path),
        "reports_dir": str(reports_dir),
    }


def generate_markdown_report(rows: list[dict[str, Any]], notes: list[str]) -> str:
    lines = [
        "# Báo cáo đối chiếu: FedAvg Stage 1 Compat & Baseline Tham chiếu",
        "",
        "> [!IMPORTANT]",
        "> Toàn bộ đối chiếu dưới đây thuộc diện `legacy_protocol_reference` (`reference_only=True`).",
        "> `strict_comparison_eligible=False` và `scientific_stage2_complete=False`.",
        "",
        "## Bảng tổng hợp đối chiếu (Đơn vị: Accuracy & Macro-F1 [0..1], Gap [điểm phần trăm pp])",
        "",
        "| Alpha | Seed | Centralized (Val-best) | Local-only Mean±Std (Final) | FedAvg Lịch sử | FedAvg Mới | Ref Gap (Cent - Fed) | Ref Gap (Fed - Local) | Khả năng tái lập (New - Hist) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in rows:
        cent_str = f"{r['centralized_test_acc']:.4f}" if r['centralized_test_acc'] is not None else "N/A"
        loc_str = (
            f"{r['local_only_acc_mean']:.4f} ± {r['local_only_acc_std_5clients']:.4f}"
            if r['local_only_acc_mean'] is not None else "MISSING"
        )
        fed_hist_str = f"{r['fedavg_historical_acc']:.4f}" if r['fedavg_historical_acc'] is not None else "MISSING (chưa có)"
        fed_new_str = f"{r['fedavg_new_acc']:.4f}" if r['fedavg_new_acc'] is not None else "Chưa huấn luyện"
        gap1_str = f"{r['reference_gap_cent_minus_fed_pp']:+.2f} pp" if r['reference_gap_cent_minus_fed_pp'] is not None else "N/A"
        gap2_str = f"{r['reference_gap_fed_minus_local_pp']:+.2f} pp" if r['reference_gap_fed_minus_local_pp'] is not None else "N/A"
        diff_str = f"{r['reproducibility_diff_pp']:+.2f} pp" if r['reproducibility_diff_pp'] is not None else "N/A"

        lines.append(
            f"| {r['alpha']} | {r['seed']} | {cent_str} | {loc_str} | {fed_hist_str} | {fed_new_str} | {gap1_str} | {gap2_str} | {diff_str} |"
        )

    lines.extend([
        "",
        "## Ghi chú phương pháp và giới hạn",
        "",
    ])
    for note in notes:
        lines.append(f"- {note}")

    return "\n".join(lines)


def plot_charts(rows: list[dict[str, Any]], reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, filename, columns in (
        ("Accuracy (%)", "stage1_accuracy_comparison.png", ("centralized_test_acc", "fedavg_new_acc", "local_only_acc_mean")),
        ("Macro-F1 (%)", "stage1_f1_comparison.png", ("centralized_test_f1", "fedavg_new_f1", "local_only_f1_mean")),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for offset, (column, label) in enumerate(zip(columns, ("Centralized historical", "FedAvg new", "Local-only historical"))):
            valid = [(i, row[column]) for i, row in enumerate(rows) if row[column] is not None]
            if valid:
                ax.bar([i + (offset - 1) * .24 for i, _ in valid], [v * 100 for _, v in valid], .24, label=label)
        ax.set_xticks(range(len(rows)), [str(r["alpha"]) for r in rows])
        ax.set(xlabel="Dirichlet alpha", ylabel=metric, title="Historical protocol reference; missing values omitted")
        ax.legend(); fig.tight_layout(); fig.savefig(reports_dir / filename, dpi=150); plt.close(fig)
    valid = sorted((r for r in rows if r["fedavg_new_acc"] is not None), key=lambda r: r["alpha"])
    if valid:
        fig, ax = plt.subplots(figsize=(8, 5))
        for column, label in (("reference_gap_cent_minus_fed_pp", "Centralized - FedAvg"),
                              ("reference_gap_fed_minus_local_pp", "FedAvg - Local-only")):
            points = [r for r in valid if r[column] is not None]
            ax.plot([r["alpha"] for r in points], [r[column] for r in points], "o-", label=label)
        ax.axhline(0, color="grey"); ax.set_xscale("log")
        ax.set(xlabel="Dirichlet alpha", ylabel="Reference gap (percentage points)")
        ax.legend(); fig.tight_layout(); fig.savefig(reports_dir / "stage1_alpha_gap.png", dpi=150); plt.close(fig)


def run_stage1_collect(output_dir=None, data_dir=None):
    root = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    with RunnerLock(root / "collection.lock"):
        try:
            return _collect(root, data_dir)
        except Exception as error:
            reports = root / "reports"
            # A partial report generation must not leave apparently valid figures/tables.
            for pattern in ("*.png", "comparison.csv", "comparison.json", "comparison.md"):
                for path in reports.glob(pattern):
                    path.unlink(missing_ok=True)
            atomic_json(reports / "collection_status.json", {"status": "FAILED", "error": str(error),
                                                            "scientific_stage2_complete": False})
            raise
