"""Build reproducible Stage 2 reports from archived experiment metrics."""

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_alpha_seed(path: Path, prefix: str) -> tuple[float, int] | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_alpha(.+)_seed(\d+)\.json", path.name)
    if not match:
        return None
    return float(match.group(1).replace("_", ".")), int(match.group(2))


def metric_value(payload: dict, key: str) -> float:
    return float(payload["test_metrics"][key])


def discover_results() -> tuple[dict[tuple[float, int], dict], dict[tuple[float, int], dict], dict[int, dict]]:
    local = {}
    fedavg = {}
    centralized = {}
    for path in RESULTS_DIR.glob("local_only_alpha*_seed*.json"):
        parsed = parse_alpha_seed(path, "local_only")
        if parsed:
            local[parsed] = load_json(path)
    for path in RESULTS_DIR.glob("fedavg_alpha*_seed*.json"):
        parsed = parse_alpha_seed(path, "fedavg")
        if parsed:
            fedavg[parsed] = load_json(path)
    for path in RESULTS_DIR.glob("centralized_seed*_metrics.json"):
        match = re.fullmatch(r"centralized_seed(\d+)_metrics\.json", path.name)
        if match:
            centralized[int(match.group(1))] = load_json(path)
    if not centralized and (RESULTS_DIR / "centralized_metrics.json").exists():
        payload = load_json(RESULTS_DIR / "centralized_metrics.json")
        centralized[int(payload["seed"])] = payload
    return local, fedavg, centralized


def write_summary(local: dict, fedavg: dict, centralized: dict) -> list[dict]:
    keys = sorted(set(local) | set(fedavg))
    rows = []
    for alpha, seed in keys:
        local_payload = local.get((alpha, seed))
        fedavg_payload = fedavg.get((alpha, seed))
        centralized_payload = centralized.get(seed)
        row = {
            "alpha": alpha,
            "seed": seed,
            "centralized_accuracy": metric_value(centralized_payload, "accuracy") if centralized_payload else "",
            "centralized_macro_f1": metric_value(centralized_payload, "macro_f1") if centralized_payload else "",
            "fedavg_accuracy": metric_value(fedavg_payload, "accuracy") if fedavg_payload else "",
            "fedavg_macro_f1": metric_value(fedavg_payload, "macro_f1") if fedavg_payload else "",
            "local_only_accuracy_mean": local_payload.get("average_accuracy", "") if local_payload else "",
            "local_only_accuracy_std": local_payload.get("accuracy_std", "") if local_payload else "",
            "local_only_macro_f1_mean": local_payload.get("average_macro_f1", "") if local_payload else "",
            "local_only_macro_f1_std": local_payload.get("macro_f1_std", "") if local_payload else "",
        }
        if row["centralized_accuracy"] != "" and row["fedavg_accuracy"] != "":
            row["centralized_minus_fedavg_accuracy"] = row["centralized_accuracy"] - row["fedavg_accuracy"]
        else:
            row["centralized_minus_fedavg_accuracy"] = ""
        if row["fedavg_accuracy"] != "" and row["local_only_accuracy_mean"] != "":
            row["fedavg_minus_local_accuracy"] = row["fedavg_accuracy"] - row["local_only_accuracy_mean"]
        else:
            row["fedavg_minus_local_accuracy"] = ""
        rows.append(row)

    fields = list(rows[0]) if rows else ["alpha", "seed"]
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_client_metrics(local: dict) -> None:
    rows = []
    for (alpha, seed), payload in sorted(local.items()):
        for client in payload.get("clients", []):
            rows.append({
                "alpha": alpha,
                "seed": seed,
                "client_id": client["client_id"],
                "train_samples": client["train_samples"],
                "accuracy": client["accuracy"],
                "macro_f1": client["macro_f1"],
                "label_entropy": payload["partition"][client["client_id"]]["label_entropy"],
                "classes_present": payload["partition"][client["client_id"]]["classes_present"],
                "largest_class_fraction": payload["partition"][client["client_id"]]["largest_class_fraction"],
            })
    if not rows:
        return
    with (RESULTS_DIR / "client_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict], metric: str, filename: str, title: str) -> None:
    points = [row for row in rows if row[f"fedavg_{metric}"] != ""]
    if not points:
        return
    seeds = sorted({row["seed"] for row in points})
    fig, axis = plt.subplots(figsize=(8, 5))
    for seed in seeds:
        selected = sorted((row for row in points if row["seed"] == seed), key=lambda row: row["alpha"])
        axis.plot([row["alpha"] for row in selected], [row[f"centralized_{metric}"] for row in selected], "--", label=f"Centralized seed {seed}")
        axis.plot([row["alpha"] for row in selected], [row[f"fedavg_{metric}"] for row in selected], "o-", label=f"FedAvg seed {seed}")
        if all(row[f"local_only_{metric}_mean"] != "" for row in selected):
            axis.plot([row["alpha"] for row in selected], [row[f"local_only_{metric}_mean"] for row in selected], "s-", label=f"Local-only seed {seed}")
    axis.set_xscale("log")
    axis.set_xlabel("Dirichlet alpha (log scale)")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def write_report(rows: list[dict]) -> None:
    lines = [
        "# Giai đoạn 2 - Báo cáo thực nghiệm",
        "",
        f"Số cấu hình đã tổng hợp: **{len(rows)}**.",
        "",
        "Các metrics test được tính trên cùng global test split; checkpoint được chọn theo validation accuracy.",
        "",
        "| Alpha | Seed | Centralized Acc | FedAvg Acc | Local-only Acc mean | FedAvg - Local-only |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def format_value(key: str) -> str:
            value = row[key]
            return "-" if value == "" else f"{value:.4f}"
        lines.append(
            f"| {row['alpha']:g} | {row['seed']} | {format_value('centralized_accuracy')} | "
            f"{format_value('fedavg_accuracy')} | {format_value('local_only_accuracy_mean')} | "
            f"{format_value('fedavg_minus_local_accuracy')} |"
        )
    lines.extend([
        "",
        "## Artefacts",
        "",
        "- `summary.csv`: tổng hợp theo alpha và seed.",
        "- `client_metrics.csv`: metrics và thống kê partition theo client.",
        "- `accuracy_by_alpha.png`: Accuracy theo alpha.",
        "- `macro_f1_by_alpha.png`: Macro-F1 theo alpha.",
    ])
    (RESULTS_DIR / "stage2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    local, fedavg, centralized = discover_results()
    rows = write_summary(local, fedavg, centralized)
    write_client_metrics(local)
    plot_metric(rows, "accuracy", "accuracy_by_alpha.png", "Accuracy theo mức non-IID")
    plot_metric(rows, "macro_f1", "macro_f1_by_alpha.png", "Macro-F1 theo mức non-IID")
    write_report(rows)
    print(f"Đã tổng hợp {len(rows)} cấu hình vào experiments/results.")
    print("Tệp chính: summary.csv, client_metrics.csv, stage2_report.md")


if __name__ == "__main__":
    main()
