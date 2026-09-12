#!/usr/bin/env python
"""
CLI: build the whole GĐ2 non-IID scenario family and report it as one table.

Produces, for every configuration in the `sweep:` block of the config:
  data/partitions_v3/<scenario>/<collision-safe-name>/ (manifests + audits)
  data/partitions_v3/sweep_summary.csv                 (one row per configuration)
  data/partitions_v3/SWEEP_REPORT.md                   (bảng đối chiếu cho báo cáo)

The α sweep is what GĐ2/GĐ5 need for "đường cong accuracy theo α": the
heterogeneity of each split is quantified here so the later FedAvg accuracy
curve can be read against a measured non-IID level, not just a nominal α.

  python scripts/sweep_alpha.py
  python scripts/sweep_alpha.py --alphas 0.1 1.0 --no-plots
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from scripts.partition_dataset import alpha_tag, load_config  # noqa: E402
from src.data.partitioner import DatasetPartitioner  # noqa: E402

SUMMARY_COLUMNS = [
    "name", "scenario", "alpha", "quantity_alpha", "feature_skew", "group_aware",
    "seed", "num_clients", "train_samples", "test_samples",
    "client_size_min", "client_size_max", "client_size_mean", "client_size_cv",
    "client_size_gini", "classes_per_client_mean", "classes_per_client_min",
    "shannon_entropy_mean", "entropy_norm_mean", "tvd_mean", "js_mean",
    "kl_mean", "cosine_to_global_mean", "rare_class_recall_mean",
    "fairness_entropy_spread", "empty_clients", "min_size_satisfied",
    "train_leaf_groups", "images_per_group_mean",
    "groups_split_across_clients", "groups_in_both_train_and_test",
    "leakage_free", "conserved", "leaf_map_coverage", "seconds",
]


def plan_from_config(cfg: dict, alpha_override=None) -> list:
    sweep = cfg.get("sweep", {}) or {}
    base = cfg.get("partition", {}) or {}
    dataset = cfg.get("dataset", {}) or {}

    scenario = sweep.get("scenario", base.get("scenario", "label_skew"))
    alphas = alpha_override or sweep.get("alphas", [0.1, 100.0])
    seeds = sweep.get("seeds", [base.get("seed", 42)])
    num_clients = sweep.get("num_clients", base.get("num_clients", 10))
    feature_skew = sweep.get("feature_skew", base.get("feature_skew", "none"))

    plan = []
    for seed in seeds:
        for a in alphas:
            plan.append({
                "scenario": scenario, "alpha": float(a), "seed": int(seed),
                "num_clients": int(num_clients), "feature_skew": feature_skew,
                "quantity_alpha": float(base.get("quantity_alpha", 0.1)),
                "group_aware": bool(dataset.get("group_aware", True)),
            })

    for extra in sweep.get("extra_scenarios", []) or []:
        plan.append({
            "scenario": extra["scenario"],
            "alpha": float(extra.get("alpha", base.get("alpha", 0.1))),
            "quantity_alpha": float(extra.get("quantity_alpha", base.get("quantity_alpha", 0.1))),
            "seed": int(seeds[0]),
            "num_clients": int(num_clients),
            "feature_skew": str(extra.get("feature_skew", feature_skew)),
            "group_aware": bool(dataset.get("group_aware", True)),
        })

    if sweep.get("include_image_level_ablation", False):
        plan.append({
            "scenario": scenario, "alpha": float(alphas[0]), "seed": int(seeds[0]),
            "num_clients": int(num_clients), "feature_skew": feature_skew,
            "quantity_alpha": float(base.get("quantity_alpha", 0.1)),
            "group_aware": False,   # <-- the leakage ablation
        })
    return plan


def name_for(entry: dict) -> str:
    base = f"{entry['scenario']}__seed_{entry['seed']}"
    if entry["scenario"] in ("label_skew", "label_quantity_skew"):
        base += f"__alpha_{alpha_tag(entry['alpha'])}"
    if entry["scenario"] in ("quantity_skew", "label_quantity_skew"):
        base += f"__qalpha_{alpha_tag(entry['quantity_alpha'])}"
    base += f"__feature_{entry['feature_skew']}"
    if not entry["group_aware"]:
        base += "__image_level"
    return base


def out_dir_for(cfg: dict, entry: dict) -> Path:
    base = Path(cfg.get("output", {}).get("base_dir", "data/partitions"))
    return ROOT / base / entry["scenario"] / name_for(entry)


def flatten(result: dict, entry: dict, name: str, seconds: float) -> dict:
    m = result["summary_metrics"]
    gi = result["group_integrity"]
    si = result["sample_integrity"]
    sd = result["split_diagnostics"]
    la = result["leaf_group_audit"]
    row = {
        "name": name,
        "scenario": result["scenario"],
        "alpha": result["alpha"],
        "quantity_alpha": result["quantity_alpha"],
        "feature_skew": result["feature_skew"],
        "group_aware": result["group_aware"],
        "seed": result["seed"],
        "num_clients": result["num_clients"],
        "train_samples": result["train_samples"],
        "test_samples": result["test_samples"],
        "client_size_min": m["client_size_min"],
        "client_size_max": m["client_size_max"],
        "client_size_mean": m["client_size_mean"],
        "client_size_cv": m["client_size_cv"],
        "client_size_gini": m["client_size_gini"],
        "classes_per_client_mean": m["classes_per_client_mean"],
        "classes_per_client_min": m["classes_per_client_min"],
        "shannon_entropy_mean": m["shannon_entropy_mean"],
        "entropy_norm_mean": m["entropy_norm_mean"],
        "tvd_mean": m["tvd_mean"],
        "js_mean": m["js_mean"],
        "kl_mean": m["kl_mean"],
        "cosine_to_global_mean": m["cosine_to_global_mean"],
        "rare_class_recall_mean": m["rare_class_recall_mean"],
        "fairness_entropy_spread": m["fairness_entropy_spread"],
        "empty_clients": m["empty_clients"],
        "min_size_satisfied": sd["min_size_satisfied"],
        "train_leaf_groups": gi["train_leaf_groups"],
        "images_per_group_mean": gi["images_per_group_mean"],
        "groups_split_across_clients": gi["groups_split_across_clients"],
        "groups_in_both_train_and_test": gi["groups_in_both_train_and_test"],
        "leakage_free": gi["leakage_free"],
        "conserved": si["conserved"],
        "leaf_map_coverage": la["leaf_map_coverage"],
        "seconds": round(seconds, 1),
    }
    assert set(entry.keys()) <= set(row.keys()), "entry keys not represented in summary"
    return row


def write_report(rows: list, path: Path, cfg: dict) -> None:
    lines = [
        "# GĐ2 -- Báo cáo bộ kịch bản dữ liệu non-IID",
        "",
        f"Sinh lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: `{cfg.get('dataset', {}).get('path')}` · "
        f"test_ratio: {cfg.get('dataset', {}).get('test_ratio')} · "
        f"seed: {cfg.get('partition', {}).get('seed')}",
        "",
        "## 1. Mức độ không đồng nhất theo từng kịch bản",
        "",
        "Cột `classes/client` và `TVD` là hai chỉ số dùng để đọc đường cong "
        "accuracy-theo-α ở GĐ2: α càng nhỏ thì mỗi cơ sở càng ít lớp và phân bố "
        "càng xa phân bố toàn cục.",
        "",
        "| Kịch bản | α nhãn | α số lượng | Feature skew | Group-aware | n_k min–max | CV | Gini | "
        "classes/client | Entropy (bits) | Entropy chuẩn hoá | TVD | JS | cos | "
        "Rare-class recall |",
        "|---|---:|---:|---|:--:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label_alpha = r["alpha"] if r["scenario"] in ("label_skew", "label_quantity_skew") else "-"
        quantity_alpha = r["quantity_alpha"] if r["scenario"] in ("quantity_skew", "label_quantity_skew") else "-"
        lines.append(
            f"| `{r['name']}` | {label_alpha} | {quantity_alpha} | {r['feature_skew']} | "
            f"{'✔' if r['group_aware'] else '✘'} | "
            f"{r['client_size_min']:g}–{r['client_size_max']:g} | {r['client_size_cv']} | "
            f"{r['client_size_gini']} | {r['classes_per_client_mean']} | "
            f"{r['shannon_entropy_mean']} | {r['entropy_norm_mean']} | {r['tvd_mean']} | "
            f"{r['js_mean']} | {r['cosine_to_global_mean']} | {r['rare_class_recall_mean']} |"
        )

    lines += [
        "",
        "## 2. Kiểm định toàn vẹn dữ liệu (bắt buộc đạt trước khi huấn luyện)",
        "",
        "| Kịch bản | Bảo toàn mẫu | Rò rỉ nhóm lá giữa client | Rò rỉ train/test | "
        "Nhóm lá train | Ảnh/nhóm | Đạt min_samples |",
        "|---|:--:|:--:|:--:|---:|---:|:--:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {'OK' if r['conserved'] else 'FAIL'} | "
            f"{r['groups_split_across_clients']} | {r['groups_in_both_train_and_test']} | "
            f"{r['train_leaf_groups']} | {r['images_per_group_mean']} | "
            f"{'OK' if r['min_size_satisfied'] else 'WARN'} |"
        )

    ga = [r for r in rows if r["group_aware"]]
    img = [r for r in rows if not r["group_aware"]]
    lines += ["", "## 3. Diễn giải", ""]
    if ga:
        r = ga[0]
        lines += [
            f"- **Chia theo nhóm lá (group-aware).** Toàn bộ {r['train_leaf_groups']} nhóm lá "
            f"trong tập train được gán trọn vẹn cho đúng một client; số nhóm bị xé lẻ giữa các "
            f"client = **{r['groups_split_across_clients']}**, số nhóm xuất hiện ở cả train lẫn "
            f"test = **{r['groups_in_both_train_and_test']}**. Đây là điều kiện cần để khoảng cách "
            f"Centralized – Federated – Local-only phản ánh đúng thuật toán chứ không phản ánh "
            f"ảnh gần trùng lặp.",
            f"- Trung bình {r['images_per_group_mean']} ảnh/nhóm trong tập train. Với độ phủ "
            f"leaf-map {r['leaf_map_coverage']:.1%}. Ảnh không có metadata được tách thành "
            f"nhóm đơn lẻ; không thể chứng minh chống rò rỉ lá cho phần dữ liệu này.",
        ]
    if img:
        r = img[0]
        lines += [
            f"- **Đối chứng chia theo từng ảnh (`{r['name']}`).** Cấu hình này cố tình bỏ qua "
            f"nhóm lá để định lượng mức rò rỉ: {r['images_per_group_mean']} ảnh/nhóm vẫn nằm "
            f"trong tập train nhưng các ảnh cùng lá nay có thể rơi sang client khác hoặc sang "
            f"tập test. Nếu accuracy của bản này cao hơn hẳn bản group-aware ở cùng α, phần "
            f"chênh lệch đó là leakage chứ không phải chất lượng mô hình.",
        ]
    lines += [
        "- **Lệch số lượng** dùng q ~ Dirichlet(quantity_alpha), theo NIID-Bench mục IV-D, "
        "và được đo độc lập bằng CV cùng hệ số Gini của n_k, không lẫn với lệch "
        "nhãn (TVD/JS/entropy). Nhờ đó hai trục non-IID của GĐ2 báo cáo tách bạch được.",
        "- **Rare-class recall** cho biết các lớp hiếm (dưới ngưỡng toàn cục) còn được bao nhiêu "
        "client giữ lại — chỉ số này giảm mạnh khi α nhỏ và là nguồn gốc của bất công bằng "
        "giữa các cơ sở.",
        "",
        "## 4. Dùng bộ chia này với FedAvg + MobileNetV3",
        "",
        "Mỗi thư mục kịch bản có `fedavg_meta.json` chứa n_k, trọng số tổng hợp "
        "w_k = n_k / Σn_j, và contract đầu vào của MobileNetV3 (224×224, RGB, CHW, "
        "ImageNet mean/std). Nạp bằng:",
        "",
        "```python",
        "from src.data import FedAvgPartition",
        "",
        "part = FedAvgPartition('data/partitions_v3/label_skew/<partition-name>')",
        "print(part.summary())",
        "weights = part.aggregation_weights          # w_k cho FedAvg",
        "loaders = part.all_client_loaders(batch_size=32)",
        "test_loader = part.global_test_loader()",
        "central_loader = part.centralized_loader()   # mốc Centralized",
        "print(part.readiness_report(check_images=8))",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the full GĐ2 non-IID scenario family.")
    ap.add_argument("--config", type=str, default="configs/partition_config.yaml")
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="Only build entries whose name contains this substring.")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if "size_sigma" in (cfg.get("partition", {}) or {}):
        raise ValueError("Legacy size_sigma config is unsupported; replace it with quantity_alpha")
    plan = plan_from_config(cfg, alpha_override=args.alphas)
    if args.only:
        plan = [e for e in plan if args.only in name_for(e)]
    if not plan:
        print("Nothing to build (check --only / sweep config).")
        return 1

    print(f"Building {len(plan)} partition configuration(s) ...")
    rows = []
    prototype = DatasetPartitioner(
        dataset_path=cfg.get("dataset", {}).get("path", "../PlantVillage-Dataset/raw/color"),
        leaf_map_path=cfg.get("dataset", {}).get("leaf_map_path", "../PlantVillage-Dataset/leaf-map.json"),
        base_dir=str(ROOT), test_ratio=float(cfg.get("dataset", {}).get("test_ratio", 0.20)),
    )
    prototype.scan_dataset()
    for i, entry in enumerate(plan, 1):
        name = name_for(entry)
        out_dir = out_dir_for(cfg, entry)
        print(f"\n[{i}/{len(plan)}] {name}  ->  {out_dir.relative_to(ROOT)}")
        t0 = time.time()
        p = DatasetPartitioner(
            dataset_path=cfg.get("dataset", {}).get("path", "../PlantVillage-Dataset/raw/color"),
            leaf_map_path=cfg.get("dataset", {}).get("leaf_map_path", "../PlantVillage-Dataset/leaf-map.json"),
            base_dir=str(ROOT),
            test_ratio=float(cfg.get("dataset", {}).get("test_ratio", 0.20)),
            val_ratio=float(cfg.get("dataset", {}).get("val_ratio", 0.0)),
            client_val_ratio=float(cfg.get("client_val_ratio", 0.0)),
            num_clients=entry["num_clients"],
            alpha=entry["alpha"],
            quantity_alpha=entry["quantity_alpha"],
            seed=entry["seed"],
            scenario=entry["scenario"],
            feature_skew=entry["feature_skew"],
            min_samples_per_client=int(cfg.get("partition", {}).get("min_samples_per_client", 10)),
            max_retries=int(cfg.get("partition", {}).get("max_retries", 20)),
            group_aware=entry["group_aware"],
            rare_class_threshold=int(cfg.get("partition", {}).get("rare_class_threshold", 500)),
        )
        p.class_names = list(prototype.class_names)
        p.class_to_id = dict(prototype.class_to_id)
        p.all_samples = copy.deepcopy(prototype.all_samples)
        p.leaf_audit = copy.deepcopy(prototype.leaf_audit)
        result = p.partition(output_dir=out_dir)
        seconds = time.time() - t0
        m = result["summary_metrics"]
        print(f"    train={result['train_samples']} test={result['test_samples']} "
              f"n_k=[{m['client_size_min']}..{m['client_size_max']}] "
              f"classes/client={m['classes_per_client_mean']} TVD={m['tvd_mean']} "
              f"leakage_free={result['group_integrity']['leakage_free']} ({seconds:.1f}s)")
        rows.append(flatten(result, entry, name, seconds))

        if not args.no_plots:
            from scripts.generate_plots import make_plots
            make_plots(out_dir)

    base = ROOT / cfg.get("output", {}).get("base_dir", "data/partitions")
    base.mkdir(parents=True, exist_ok=True)
    summary_csv = base / "sweep_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_report(rows, base / "SWEEP_REPORT.md", cfg)

    if not args.no_plots:
        from scripts.generate_plots import make_sweep_plots
        make_sweep_plots(base)

    print("\n" + "=" * 72)
    print(f"  {len(rows)} configuration(s) built")
    print(f"  summary : {summary_csv.relative_to(ROOT)}")
    print(f"  report  : {(base / 'SWEEP_REPORT.md').relative_to(ROOT)}")
    print("=" * 72)
    bad = [r["name"] for r in rows if not r["conserved"]]
    bad += [r["name"] for r in rows if r["group_aware"] and not r["leakage_free"]]
    if bad:
        print(f"  FAILED integrity: {bad}")
        return 2

    ablation = [r for r in rows if not r["group_aware"]]
    for r in ablation:
        print(f"  EXPECTED leakage (image-level ablation '{r['name']}'): "
              f"{r['groups_split_across_clients']} group(s) span clients, "
              f"{r['groups_in_both_train_and_test']} in both train and test")
    unsatisfied = [r["name"] for r in rows if not r["min_size_satisfied"]]
    if unsatisfied:
        print(f"  WARNING min_samples_per_client not reached: {unsatisfied}")
    print("  integrity: conserved + leakage-free for every group-aware split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
