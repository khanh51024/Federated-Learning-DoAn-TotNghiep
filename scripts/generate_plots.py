#!/usr/bin/env python
"""
Plots for GĐ2 partitions.

Per partition (`make_plots`):
  1. client_class_heatmap.png    client x class counts (log colour scale)
  2. client_label_dist.png       client x class *proportions* -- this is the
                                 figure that actually shows label skew
  3. client_samples.png          n_k bar chart with mean line
  4. client_divergence.png       per-client entropy / TVD / classes-present

Across the sweep (`make_sweep_plots`):
  5. noniid_vs_alpha.png         heterogeneity metrics as a function of alpha,
                                 i.e. the x-axis of the GĐ2 accuracy curve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 110,
})


def _title_suffix(partition_dir: Path) -> str:
    cfg = partition_dir / "partition_config.json"
    if not cfg.exists():
        return partition_dir.name
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            c = json.load(f)
        bits = [f"scenario={c['scenario']}", f"alpha={c['alpha']}"]
        if c["scenario"] in ("quantity_skew", "label_quantity_skew"):
            bits.append(f"quantity_alpha={c['quantity_alpha']}")
        bits.append(f"feature_skew={c['feature_skew']}")
        bits.append(f"group_aware={c['group_aware']}")
        return f"{partition_dir.name}  ({', '.join(bits)})"
    except (json.JSONDecodeError, KeyError):
        return partition_dir.name


def make_plots(partition_dir: str | Path) -> list:
    """Render the four per-partition figures. Returns the written paths."""
    partition_dir = Path(partition_dir)
    plots = partition_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    matrix_path = partition_dir / "client_class_matrix.csv"
    stats_path = partition_dir / "statistics.csv"
    if not matrix_path.exists():
        print(f"[plots] skipped (no client_class_matrix.csv): {partition_dir}")
        return []

    matrix = pd.read_csv(matrix_path, index_col=0)
    stats = pd.read_csv(stats_path) if stats_path.exists() else None
    suffix = _title_suffix(partition_dir)
    written = []

    # --- 1. counts heatmap -------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6.5))
    data = matrix.values.astype(float)
    im = ax.imshow(np.log10(data + 1), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Lớp (38 lớp cây–bệnh PlantVillage)")
    ax.set_ylabel("Client (cơ sở nông nghiệp)")
    ax.set_title(f"Phân bố số mẫu client × lớp — log10(count+1)\n{suffix}", fontsize=11)
    fig.colorbar(im, ax=ax, label="log10(số mẫu + 1)", shrink=0.85)
    fig.tight_layout()
    p = plots / "client_class_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # --- 2. normalised label distribution ----------------------------------
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    props = data / row_sums
    fig, ax = plt.subplots(figsize=(16, 6.5))
    im = ax.imshow(props, aspect="auto", cmap="magma", vmin=0, vmax=max(0.25, float(props.max())))
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Client")
    ax.set_title("Phân bố nhãn chuẩn hoá theo dòng (mỗi client tổng = 1)\n"
                 "Ô sáng = lớp chiếm phần lớn dữ liệu cục bộ → non-IID mạnh\n" + suffix, fontsize=11)
    fig.colorbar(im, ax=ax, label="tỉ lệ trong client", shrink=0.85)
    fig.tight_layout()
    p = plots / "client_label_dist.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # --- 3. client sizes ---------------------------------------------------
    if stats is not None and "sample_count" in stats.columns:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(stats))
        bars = ax.bar(x, stats["sample_count"], color=plt.cm.viridis(np.linspace(0.15, 0.85, len(stats))),
                      edgecolor="black", linewidth=0.5)
        for b in bars:
            ax.annotate(f"{int(b.get_height())}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        mean_v = float(stats["sample_count"].mean())
        ax.axhline(mean_v, color="crimson", linestyle="--", alpha=0.8, label=f"mean = {mean_v:.0f}")
        ax.set_xticks(x)
        ax.set_xticklabels(stats["client_name"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Số mẫu huấn luyện n_k")
        ax.set_title(f"Kích thước dữ liệu mỗi client (trọng số FedAvg w_k = n_k / Σn_j)\n{suffix}",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        p = plots / "client_samples.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

        # --- 4. per-client divergence -------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        panels = [
            ("shannon_entropy", "Entropy nhãn H(p_c) [bits]", "tab:blue"),
            ("tvd_from_global", "TVD so với phân bố toàn cục", "tab:red"),
            ("num_classes", "Số lớp có mặt / 38", "tab:green"),
        ]
        for ax, (col, label, color) in zip(axes, panels):
            if col not in stats.columns:
                continue
            ax.bar(x, stats[col], color=color, edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(stats["client_name"], rotation=90, fontsize=7)
            ax.set_title(label, fontsize=10)
        fig.suptitle(f"Bất đối xứng giữa các cơ sở\n{suffix}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = plots / "client_divergence.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    print(f"[plots] {len(written)} figure(s) -> {plots}")
    return written


def make_sweep_plots(base_dir: str | Path) -> list:
    """Heterogeneity-vs-alpha curves from sweep_summary.csv."""
    base_dir = Path(base_dir)
    csv_path = base_dir / "sweep_summary.csv"
    if not csv_path.exists():
        print(f"[plots] no sweep_summary.csv in {base_dir}")
        return []

    df = pd.read_csv(csv_path)
    ls = df[(df["scenario"] == "label_skew") & df["group_aware"]].copy()
    if ls.empty:
        print("[plots] no group-aware label_skew rows to plot")
        return []
    ls = ls.sort_values("alpha")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("classes_per_client_mean", "Số lớp trung bình mỗi client", "tab:blue"),
        ("shannon_entropy_mean", "Entropy nhãn trung bình [bits]", "tab:green"),
        ("tvd_mean", "TVD trung bình so với toàn cục", "tab:red"),
        ("client_size_cv", "CV kích thước client (lệch số lượng)", "tab:purple"),
    ]
    for ax, (col, label, color) in zip(axes.ravel(), panels):
        ax.plot(ls["alpha"], ls[col], marker="o", color=color, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("α (Dirichlet, thang log)")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
        for _, r in ls.iterrows():
            ax.annotate(f"{r[col]:g}", (r["alpha"], r[col]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=color)
    fig.suptitle("Mức độ non-IID đo được theo α — trục hoành của đường cong accuracy GĐ2\n"
                 "(α nhỏ → non-IID mạnh; α lớn → gần IID)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = base_dir / "noniid_vs_alpha.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] sweep curve -> {out}")
    written = [out]

    # --- quantity-skew axis (independent of alpha) ------------------------
    qs = df[df["scenario"] == "quantity_skew"].sort_values("quantity_alpha")
    if len(qs) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        axes[0].plot(qs["quantity_alpha"], qs["client_size_cv"], marker="s", color="tab:purple",
                     label="CV của n_k")
        axes[0].plot(qs["quantity_alpha"], qs["client_size_gini"], marker="o", color="tab:orange",
                     label="Gini của n_k")
        axes[0].set_xlabel("quantity_alpha (Dirichlet)")
        axes[0].set_ylabel("bất bình đẳng dung lượng")
        axes[0].set_title("Lệch số lượng: αq nhỏ → bất bình đẳng n_k", fontsize=10)
        axes[0].legend(fontsize=9)

        width = 0.35
        x = np.arange(len(qs))
        axes[1].bar(x - width / 2, qs["client_size_min"], width, label="n_k nhỏ nhất",
                    color="tab:red")
        axes[1].bar(x + width / 2, qs["client_size_max"], width, label="n_k lớn nhất",
                    color="tab:blue")
        axes[1].set_yscale("log")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"αq={s:g}" for s in qs["quantity_alpha"]])
        axes[1].set_ylabel("số mẫu (thang log)")
        axes[1].set_title("Trải dài dung lượng giữa cơ sở lớn và cơ sở nhỏ", fontsize=10)
        axes[1].legend(fontsize=9)

        fig.suptitle("Trục lệch số lượng dùng Dirichlet độc lập với alpha lệch nhãn",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out2 = base_dir / "quantity_skew_vs_qalpha.png"
        fig.savefig(out2, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[plots] quantity-skew curve -> {out2}")
        written.append(out2)

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate partition plots.")
    ap.add_argument("--partition-dir", type=str, default=None)
    ap.add_argument("--sweep-dir", type=str, default=None,
                    help="Directory containing sweep_summary.csv (for the α curves).")
    args = ap.parse_args()

    if not args.partition_dir and not args.sweep_dir:
        print("Pass --partition-dir and/or --sweep-dir.")
        return 1
    if args.partition_dir:
        make_plots(Path(args.partition_dir))
    if args.sweep_dir:
        make_sweep_plots(Path(args.sweep_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
