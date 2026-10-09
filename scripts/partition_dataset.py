#!/usr/bin/env python
"""
CLI: partition PlantVillage into non-IID federated clients (GĐ2).

Defaults come from configs/partition_config.yaml; every field is overridable.

Examples
--------
  python scripts/partition_dataset.py --alpha 0.1
  python scripts/partition_dataset.py --scenario quantity_skew --quantity-alpha 0.1
  python scripts/partition_dataset.py --alpha 100 --feature-skew none --group-aware false
  python scripts/partition_dataset.py --config configs/partition_config.yaml --output-path data/partitions_v3/custom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data.partitioner import DatasetPartitioner  # noqa: E402


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Partition PlantVillage into non-IID federated clients (group-aware Dirichlet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="configs/partition_config.yaml")
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--leaf-map-path", type=str, default=None)
    p.add_argument("--output-path", type=str, default=None,
                   help="Defaults to a collision-safe name containing scenario, seed and parameters.")
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--scenario", type=str, default=None,
                   choices=["iid", "label_skew", "quantity_skew", "label_quantity_skew"])
    p.add_argument("--alpha", type=float, default=None,
                   help="Dirichlet concentration (label skew). 100 ~ IID, 0.1 ~ strong non-IID.")
    p.add_argument("--quantity-alpha", type=float, default=None,
                   help="Dirichlet concentration for quantity skew. Smaller means stronger skew.")
    p.add_argument("--feature-skew", type=str, default=None,
                   choices=["none", "mild", "moderate", "strong"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--test-ratio", type=float, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--client-val-ratio", type=float, default=None)
    p.add_argument("--min-samples-per-client", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--group-aware", type=str, default=None,
                   help="true/false. false splits per image (leaks near-duplicate leaves).")
    p.add_argument("--content-aware", type=str, default=None,
                   help="true/false. Also keep exact-byte duplicates in one partition group.")
    p.add_argument("--rare-class-threshold", type=int, default=None)
    return p


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "_")


def default_output_dir(cfg: dict, scenario: str, alpha: float, quantity_alpha: float,
                       feature_skew: str, seed: int, group_aware: bool, content_aware: bool = False) -> Path:
    base = Path(cfg.get("output", {}).get("base_dir", "data/partitions"))
    name = f"{scenario}__seed_{seed}"
    if scenario in ("label_skew", "label_quantity_skew"):
        name += f"__alpha_{alpha_tag(alpha)}"
    if scenario in ("quantity_skew", "label_quantity_skew"):
        name += f"__qalpha_{alpha_tag(quantity_alpha)}"
    name += f"__feature_{feature_skew}"
    if content_aware:
        name += "__content_aware"
    if not group_aware:
        name += "__image_level"
    return ROOT / base / scenario / name


def resolve_params(args: argparse.Namespace) -> dict:
    cfg = load_config(ROOT / args.config)
    ds = cfg.get("dataset", {})
    pt = cfg.get("partition", {})
    out = cfg.get("output", {})
    if "size_sigma" in pt:
        raise ValueError("Legacy size_sigma config is unsupported; replace it with quantity_alpha")

    def pick(cli, section, key, default=None):
        if cli is not None:
            return cli
        if key in section and section[key] is not None:
            return section[key]
        return default

    params = {
        "dataset_path": pick(args.dataset_path, ds, "path", "../PlantVillage-Dataset/raw/color"),
        "leaf_map_path": pick(args.leaf_map_path, ds, "leaf_map_path", "../PlantVillage-Dataset/leaf-map.json"),
        "group_aware": _as_bool(pick(args.group_aware, ds, "group_aware", True)),
        "content_aware": _as_bool(pick(args.content_aware, ds, "content_aware", False)),
        "test_ratio": float(pick(args.test_ratio, ds, "test_ratio", 0.20)),
        "val_ratio": float(pick(args.val_ratio, ds, "val_ratio", 0.0)),
        "client_val_ratio": float(pick(args.client_val_ratio, cfg, "client_val_ratio", 0.0)),
        "num_clients": int(pick(args.num_clients, pt, "num_clients", 10)),
        "scenario": str(pick(args.scenario, pt, "scenario", "label_skew")),
        "alpha": float(pick(args.alpha, pt, "alpha", 0.1)),
        "quantity_alpha": float(pick(args.quantity_alpha, pt, "quantity_alpha", 0.1)),
        "feature_skew": str(pick(args.feature_skew, pt, "feature_skew", "moderate")),
        "seed": int(pick(args.seed, pt, "seed", 42)),
        "min_samples_per_client": int(pick(args.min_samples_per_client, pt, "min_samples_per_client", 10)),
        "max_retries": int(pick(args.max_retries, pt, "max_retries", 20)),
        "rare_class_threshold": int(pick(args.rare_class_threshold, pt, "rare_class_threshold", 500)),
    }

    if args.output_path:
        out_dir = Path(args.output_path)
        params["output_dir"] = str(out_dir if out_dir.is_absolute() else ROOT / out_dir)
    else:
        params["output_dir"] = str(default_output_dir(
            {"output": out}, params["scenario"], params["alpha"], params["quantity_alpha"],
            params["feature_skew"], params["seed"], params["group_aware"], params["content_aware"]
        ))
    return params


def main() -> int:
    args = build_parser().parse_args()
    params = resolve_params(args)
    out_dir = params.pop("output_dir")

    print("=" * 72)
    print("  GĐ2 -- PLANTVILLAGE NON-IID PARTITION PIPELINE")
    print("=" * 72)
    for k, v in params.items():
        print(f"  {k:<26} {v}")
    print(f"  {'output_dir':<26} {out_dir}")
    print("-" * 72)

    partitioner = DatasetPartitioner(base_dir=str(ROOT), **params)

    print("Scanning dataset ...", flush=True)
    partitioner.scan_dataset()
    la = partitioner.leaf_audit
    print(f"  images: {la['total_images']}  classes: {len(partitioner.class_names)}")
    print(f"  leaf groups: {la['distinct_groups']} "
          f"(coverage {la['leaf_map_coverage']:.1%}, "
          f"{la['images_per_group_mean']} img/group, max {la['images_per_group_max']})")
    if la["classes_with_leaf_metadata"] < la["classes_total"]:
        print(f"  note: {la['classes_total'] - la['classes_with_leaf_metadata']} classes have no "
              f"leaf metadata -> those images fall back to singleton groups")

    print("Partitioning ...", flush=True)
    result = partitioner.partition(output_dir=out_dir)

    m = result["summary_metrics"]
    gi = result["group_integrity"]
    si = result["sample_integrity"]
    sd = result["split_diagnostics"]

    print("-" * 72)
    print("  RESULT")
    print(f"  train / test / val : {result['train_samples']} / "
          f"{result['test_samples']} / {result['val_samples']}")
    print(f"  client size        : min={m['client_size_min']} max={m['client_size_max']} "
          f"mean={m['client_size_mean']} cv={m['client_size_cv']} gini={m['client_size_gini']}")
    print(f"  classes / client   : mean={m['classes_per_client_mean']} "
          f"min={m['classes_per_client_min']} max={m['classes_per_client_max']} "
          f"(of {result['total_classes']})")
    print(f"  entropy (bits)     : mean={m['shannon_entropy_mean']} "
          f"[norm {m['entropy_norm_mean']}] max_possible={m['max_entropy_bits']}")
    print(f"  divergence         : tvd={m['tvd_mean']} js={m['js_mean']} "
          f"kl={m['kl_mean']} cos={m['cosine_to_global_mean']}")
    print(f"  rare-class recall  : mean={m['rare_class_recall_mean']} "
          f"over {m['num_rare_classes']} rare classes")
    print("-" * 72)
    print("  INTEGRITY")
    print(f"  sample conservation : {'OK' if si['conserved'] else 'FAILED'}  {si}")
    print(f"  leaf-group leakage  : {'NONE' if gi['leakage_free'] else 'DETECTED'}  "
          f"(cross-client={gi['groups_split_across_clients']}, "
          f"train/test={gi['groups_in_both_train_and_test']})")
    print(f"  min_samples_per_client satisfied: {sd['min_size_satisfied']}")
    if not sd["min_size_satisfied"]:
        print(f"  WARNING: smallest client has {sd['client_size_min']} samples "
              f"(< {sd['min_samples_per_client']})")
    print("-" * 72)
    print(f"  wrote: {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
