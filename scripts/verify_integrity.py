#!/usr/bin/env python
"""
INDEPENDENT integrity audit of a partition directory.

This script deliberately does NOT reuse the partitioner's in-memory state or its
own assertions. It re-reads the CSV manifests from disk and re-derives every
invariant from scratch, so a bug in the writer cannot hide behind a bug in the
checker.

Checks
------
 A. Manifest hygiene      : required files exist, no empty manifest, columns valid
 B. Conservation          : union(clients) == centralized_train, no duplicate rows
 C. Disjointness          : train/test, train/val, test/val share no image
 D. Leaf-group integrity  : no leaf group spans two clients or train/test
 E. Metadata consistency  : label <-> class_name agree with the dataset folders,
                            client_id/split/seed/alpha are coherent
 F. Global test coverage  : every class represented, histogram matches config
 G. FedAvg contract       : n_k in fedavg_meta.json matches the CSV row counts,
                            weights sum to 1
 H. Image readability     : a sampled subset actually opens and decodes

  python scripts/verify_integrity.py --partition-dir data/partitions_v3/label_skew/<name>
  python scripts/verify_integrity.py --all --probe-images 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REQUIRED_FIELDS = [
    "relative_path", "label", "class_name", "client_id", "split", "group_id",
]


def read_manifest(path: Path) -> list:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Report:
    def __init__(self):
        self.checks: list = []
        self.failures: list = []
        self.warnings: list = []

    def check(self, ok: bool, name: str, detail: str = "", warn_only: bool = False):
        if ok:
            self.checks.append(("PASS", name, detail))
        elif warn_only:
            self.checks.append(("WARN", name, detail))
            self.warnings.append(f"{name}: {detail}")
        else:
            self.checks.append(("FAIL", name, detail))
            self.failures.append(f"{name}: {detail}")
        return ok

    def dump(self, title: str) -> int:
        print("=" * 74)
        print(f"  {title}")
        print("=" * 74)
        for mark, name, detail in self.checks:
            line = f"  [{mark}] {name}"
            if detail:
                line += f"  -- {detail}"
            print(line)
        if self.failures:
            print("-" * 74)
            print("  FAILURES:")
            for f in self.failures:
                print(f"    x {f}")
        if self.warnings:
            print("-" * 74)
            print("  WARNINGS:")
            for w in self.warnings:
                print(f"    ! {w}")
        print("-" * 74)
        n_pass = sum(1 for m, _, _ in self.checks if m == "PASS")
        print(f"  {n_pass}/{len(self.checks)} checks passed"
              + ("" if not self.failures else f" -- {len(self.failures)} FAILED"))
        print("=" * 74)
        return 1 if self.failures else 0


def audit(partition_dir: Path, probe_images: int = 0) -> Report:
    rep = Report()
    cfg_path = partition_dir / "partition_config.json"
    meta_path = partition_dir / "fedavg_meta.json"

    # ---- A. manifest hygiene -------------------------------------------
    rep.check(cfg_path.exists(), "partition_config.json exists")
    rep.check(meta_path.exists(), "fedavg_meta.json exists")
    if not (cfg_path.exists() and meta_path.exists()):
        return rep

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_clients = int(cfg["num_clients"])
    num_classes = int(cfg["total_classes"])
    class_names = list(cfg["class_names"])

    client_rows = {}
    client_val_rows = {}
    for c in range(num_clients):
        p = partition_dir / "clients" / f"client_{c:02d}.csv"
        rep.check(p.exists(), f"client_{c:02d}.csv exists")
        if p.exists():
            rows = read_manifest(p)
            client_rows[c] = rows
            rep.check(len(rows) > 0, f"client_{c:02d}.csv non-empty",
                      f"{len(rows)} rows", warn_only=True)
            missing = [k for k in REQUIRED_FIELDS if rows and k not in rows[0]]
            rep.check(not missing, f"client_{c:02d}.csv columns",
                      f"missing {missing}" if missing else "all required columns present")
        vp = partition_dir / "clients" / f"client_{c:02d}_val.csv"
        client_val_rows[c] = read_manifest(vp) if vp.exists() else []

    test_p = partition_dir / "global_test.csv"
    rep.check(test_p.exists(), "global_test.csv exists")
    test_rows = read_manifest(test_p) if test_p.exists() else []
    val_p = partition_dir / "global_val.csv"
    val_rows = read_manifest(val_p) if val_p.exists() else []
    cen_p = partition_dir / "centralized_train.csv"
    rep.check(cen_p.exists(), "centralized_train.csv exists")
    cen_rows = read_manifest(cen_p) if cen_p.exists() else []

    all_client_rows = [r for c in sorted(client_rows) for r in client_rows[c]]
    all_client_val_rows = [r for c in sorted(client_val_rows) for r in client_val_rows[c]]

    # ---- B. conservation ------------------------------------------------
    rep.check(
        len(all_client_rows) == int(cfg["train_samples"]),
        "sum(n_k) == config.train_samples",
        f"{len(all_client_rows)} vs {cfg['train_samples']}",
    )
    rep.check(
        len(all_client_rows) == len(cen_rows),
        "centralized_train row count == union of clients",
        f"{len(cen_rows)} vs {len(all_client_rows)}",
    )
    client_paths = Counter(r["relative_path"] for r in all_client_rows)
    dupes = [p for p, n in client_paths.items() if n > 1]
    rep.check(not dupes, "no image assigned to two clients",
              f"{len(dupes)} duplicated path(s)" if dupes else "each image appears exactly once")
    cen_paths = Counter(r["relative_path"] for r in cen_rows)
    rep.check(
        set(cen_paths) == set(client_paths) and all(v == 1 for v in cen_paths.values()),
        "centralized_train is exactly the union (no dupes)",
        f"|centralized|={len(cen_paths)} |union|={len(client_paths)}",
    )

    # ---- C. disjointness ------------------------------------------------
    test_paths = set(r["relative_path"] for r in test_rows)
    val_paths = set(r["relative_path"] for r in val_rows)
    train_paths = set(client_paths)
    client_val_paths = set(r["relative_path"] for r in all_client_val_rows)
    rep.check(not (train_paths & client_val_paths), "client train/local-val disjoint",
              f"{len(train_paths & client_val_paths)} overlap")
    rep.check(not (client_val_paths & test_paths), "local-val/test disjoint",
              f"{len(client_val_paths & test_paths)} overlap")
    rep.check(not (client_val_paths & val_paths), "local-val/global-val disjoint",
              f"{len(client_val_paths & val_paths)} overlap")
    rep.check(not (train_paths & test_paths), "train/test disjoint",
              f"{len(train_paths & test_paths)} overlap")
    rep.check(not (train_paths & val_paths), "train/val disjoint",
              f"{len(train_paths & val_paths)} overlap")
    rep.check(not (test_paths & val_paths), "test/val disjoint",
              f"{len(test_paths & val_paths)} overlap")
    rep.check(
        len(train_paths) + len(client_val_paths) + len(test_paths) + len(val_paths) == int(cfg["total_images"]),
        "train + local-val + test + global-val == total_images",
        f"{len(train_paths)}+{len(client_val_paths)}+{len(test_paths)}+{len(val_paths)} vs {cfg['total_images']}",
    )

    # ---- D. leaf-group integrity ---------------------------------------
    if cfg.get("group_aware", False):
        # A leaf group legitimately contains several images, so the test is
        # "does this group_id appear under more than one client_id", NOT
        # "does it appear more than once".
        owners: dict = defaultdict(set)
        for r in all_client_rows + all_client_val_rows:
            owners[r["group_id"]].add(r["client_id"])
        cross_client = [g for g, cs in owners.items() if len(cs) > 1]
        rep.check(not cross_client, "no leaf group spans two clients",
                  f"{len(cross_client)} shared group(s) over {len(owners)} groups"
                  if cross_client else f"{len(owners)} groups, each owned by exactly one client")
        seen = set(owners)

        local_train_groups = {c: set(r["group_id"] for r in rows) for c, rows in client_rows.items()}
        local_val_groups = {c: set(r["group_id"] for r in rows) for c, rows in client_val_rows.items()}
        local_overlap = sum(len(local_train_groups.get(c, set()) & local_val_groups.get(c, set()))
                            for c in range(num_clients))
        rep.check(local_overlap == 0, "no leaf group in client train and local val",
                  f"{local_overlap} shared group(s)")

        test_groups = set(r["group_id"] for r in test_rows)
        val_groups = set(r["group_id"] for r in val_rows)
        shared_tt = seen & test_groups
        rep.check(not shared_tt, "no leaf group in both train and test",
                  f"{len(shared_tt)} shared group(s)")
        shared_tv = seen & val_groups
        rep.check(not shared_tv, "no leaf group in both train and val",
                  f"{len(shared_tv)} shared group(s)")
        rep.check(
            not (test_groups & val_groups), "no leaf group in both test and val",
            f"{len(test_groups & val_groups)} shared group(s)",
        )
    else:
        # Image-level ablation: leakage is expected, so report its magnitude
        # instead of failing. This number is the quantified cost of NOT
        # splitting on leaves.
        owners = defaultdict(set)
        for r in all_client_rows:
            owners[r["group_id"]].add(r["client_id"])
        cross_client = sum(1 for cs in owners.values() if len(cs) > 1)
        test_groups = set(r["group_id"] for r in test_rows)
        shared_tt = len(set(owners) & test_groups)
        rep.check(True, "group-aware disabled (image-level ablation)",
                  f"measured leakage: {cross_client} leaf group(s) span clients, "
                  f"{shared_tt} appear in both train and test", warn_only=True)

    # ---- E. metadata consistency ---------------------------------------
    audited_rows = all_client_rows + all_client_val_rows + test_rows + val_rows
    bad_label = [r["relative_path"] for r in audited_rows
                 if class_names[int(r["label"])] != r["class_name"]]
    rep.check(not bad_label, "label <-> class_name agree",
              f"{len(bad_label)} mismatched row(s)")

    folder_mismatch = [r["relative_path"] for r in audited_rows
                       if r["relative_path"].split("/")[-2] != r["class_name"]]
    rep.check(not folder_mismatch, "class_name == containing folder",
              f"{len(folder_mismatch)} mismatched row(s)")

    out_of_range = [r["relative_path"] for r in audited_rows
                    if not (0 <= int(r["label"]) < num_classes)]
    rep.check(not out_of_range, "all labels in [0, num_classes)",
              f"{len(out_of_range)} out-of-range row(s)")

    for c, rows in client_rows.items():
        bad_cid = [r for r in rows if int(r["client_id"]) != c]
        if bad_cid:
            rep.check(False, f"client_{c:02d}.csv client_id column",
                      f"{len(bad_cid)} row(s) with wrong client_id")
            break
        bad_split = [r for r in rows if r["split"] != "train"]
        if bad_split:
            rep.check(False, f"client_{c:02d}.csv split column",
                      f"{len(bad_split)} row(s) not marked 'train'")
            break
    else:
        rep.check(True, "client_id / split columns coherent across all clients")
    for c, rows in client_val_rows.items():
        rep.check(all(int(r["client_id"]) == c and r["split"] == "val" for r in rows),
                  f"client_{c:02d}_val.csv metadata coherent")

    rep.check(
        all(int(r["seed"]) == int(cfg["seed"]) for r in all_client_rows[:200]),
        "seed stamped consistently", f"seed={cfg['seed']}",
    )

    # ---- F. global test coverage ----------------------------------------
    hist = np.zeros(num_classes, dtype=np.int64)
    for r in test_rows:
        hist[int(r["label"])] += 1
    cfg_hist = np.array(cfg.get("global_test_class_histogram", []), dtype=np.int64)
    rep.check(
        cfg_hist.size == 0 or np.array_equal(hist, cfg_hist),
        "recorded test histogram matches global_test.csv",
    )
    empty_classes = [class_names[i] for i in np.flatnonzero(hist == 0)]
    rep.check(not empty_classes, "global test covers every class",
              f"missing: {empty_classes}" if empty_classes else f"{num_classes}/{num_classes} covered",
              warn_only=True)
    rep.check(
        len(test_rows) == int(cfg["test_samples"]),
        "global_test.csv row count == config.test_samples",
        f"{len(test_rows)} vs {cfg['test_samples']}",
    )

    # ---- G. FedAvg contract ---------------------------------------------
    n_k_meta = meta["client_sample_counts"]
    n_k_csv = {f"client_{c:02d}": len(client_rows.get(c, [])) for c in range(num_clients)}
    rep.check(n_k_meta == n_k_csv, "fedavg_meta n_k == CSV row counts",
              f"{n_k_meta}" if n_k_meta != n_k_csv else f"{list(n_k_csv.values())}")

    w = np.array(list(meta["client_aggregation_weights"].values()), dtype=np.float64)
    rep.check(abs(w.sum() - 1.0) < 1e-6, "aggregation weights sum to 1", f"sum={w.sum():.8f}")
    n_k = np.array(list(n_k_csv.values()), dtype=np.float64)
    expected = n_k / n_k.sum() if n_k.sum() else n_k
    rep.check(np.allclose(w, expected, atol=1e-6), "w_k == n_k / sum(n_j)",
              f"max|diff|={float(np.max(np.abs(w - expected))):.2e}" if n_k.sum() else "")
    rep.check(int(n_k.sum()) == int(meta["total_train_samples"]),
              "total_train_samples consistent", f"{int(n_k.sum())}")
    expected_val = {f"client_{c:02d}": len(client_val_rows[c]) for c in range(num_clients)}
    rep.check(meta.get("client_val_counts") == expected_val,
              "client_val_counts consistent", str(expected_val))

    m = meta.get("model", {})
    rep.check(m.get("input_size") == [224, 224], "MobileNetV3 input size 224x224",
              str(m.get("input_size")))
    rep.check(m.get("channels") == 3 and m.get("layout") == "CHW", "input contract RGB/CHW",
              f"{m.get('channels')}ch {m.get('layout')}")
    rep.check(np.allclose(m.get("normalize_mean", []), [0.485, 0.456, 0.406], atol=1e-6),
              "ImageNet mean recorded", str(m.get("normalize_mean")))
    rep.check(np.allclose(m.get("normalize_std", []), [0.229, 0.224, 0.225], atol=1e-6),
              "ImageNet std recorded", str(m.get("normalize_std")))
    rep.check(m.get("num_classes") == num_classes, "model num_classes == partition num_classes",
              f"{m.get('num_classes')} vs {num_classes}")

    # ---- H. image readability -------------------------------------------
    if probe_images > 0:
        anchor_rel = meta["paths"].get("manifest_anchor", "../..")
        base = (partition_dir / anchor_rel).resolve()
        pools = {
            "client_00": client_rows.get(0, []),
            "global_test": test_rows,
            "centralized": cen_rows,
        }
        for label, rows in pools.items():
            if not rows:
                continue
            idxs = np.linspace(0, len(rows) - 1, min(probe_images, len(rows))).astype(int)
            missing, bad_size = [], []
            for i in idxs:
                # pathlib accepts forward slashes on Windows, so the manifest
                # value can be joined verbatim.
                p = base / rows[int(i)]["relative_path"]
                if not p.exists():
                    missing.append(rows[int(i)]["relative_path"])
                    continue
                try:
                    with Image.open(p) as im:
                        im.convert("RGB").load()
                except Exception as exc:  # noqa: BLE001
                    bad_size.append(f"{p.name}: {exc}")
            rep.check(not missing, f"[{label}] sampled images exist on disk",
                      f"{len(missing)} missing (e.g. {missing[:2]})" if missing
                      else f"{len(idxs)} probed, all resolved")
            rep.check(not bad_size, f"[{label}] sampled images decode as RGB",
                      "; ".join(bad_size[:2]) if bad_size else f"{len(idxs)} decoded")

    return rep


def find_partitions(base: Path) -> list:
    return sorted(p.parent for p in base.rglob("fedavg_meta.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent integrity audit of a partition.")
    ap.add_argument("--partition-dir", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="Audit every partition under output.base_dir")
    ap.add_argument("--sweep-dir", type=str, default="data/partitions_v3")
    ap.add_argument("--probe-images", type=int, default=0,
                    help="Open N sampled images per split to verify paths decode (0 = skip).")
    args = ap.parse_args()

    targets = []
    if args.partition_dir:
        targets.append(Path(args.partition_dir))
    if args.all:
        targets.extend(find_partitions(ROOT / args.sweep_dir))
    if not targets:
        print("Pass --partition-dir or --all.")
        return 1

    rc = 0
    test_hashes = {}
    for t in targets:
        t = t if t.is_absolute() else ROOT / t
        rep = audit(t.resolve(), probe_images=args.probe_images)
        rc |= rep.dump(f"AUDIT: {t.relative_to(ROOT) if ROOT in t.parents else t}")
        cfg_path = t / "partition_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            test_hashes[str(t)] = cfg.get("provenance", {}).get("global_test_paths_sha256")
    if args.all:
        distinct = set(test_hashes.values())
        if None in distinct or len(distinct) != 1:
            print(f"[FAIL] partitions do not share one global-test hash: {test_hashes}")
            rc = 1
        else:
            print(f"[PASS] all {len(test_hashes)} partitions share global-test hash {next(iter(distinct))}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
