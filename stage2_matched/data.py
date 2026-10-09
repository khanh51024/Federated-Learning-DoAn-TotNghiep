"""Versioned, immutable manifests: split holdouts first, distribute train only."""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from fl_training.content_audit import audit_image_content
from fl_training.prepare import audit_manifest_directory, build_or_update_source_content_cache
from src.data.dirichlet_split import partition
from src.data.leaf_groups import group_samples
from src.data.metrics import calculate_partition_metrics
from src.data.partitioner import DatasetPartitioner, MANIFEST_FIELDS
from src.data.transforms import ClientDomainTransform, ClientFeatureProfile, create_deterministic_client_profile
from stage1_compat.data import TRAIN_TRANSFORM, EVAL_TRANSFORM
from stage1_compat.integrity import atomic_json, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "stage2_scratch_clean_v3"
CONDITIONS = {
    "label100": {"scenario": "label_skew", "alpha": 100.0},
    "label1": {"scenario": "label_skew", "alpha": 1.0},
    "label01": {"scenario": "label_skew", "alpha": 0.1},
    "quantity100": {"scenario": "quantity_skew", "quantity_alpha": 100.0},
    "quantity01": {"scenario": "quantity_skew", "quantity_alpha": 0.1},
    "label_quantity01": {"scenario": "label_quantity_skew", "alpha": 0.1, "quantity_alpha": 0.1},
    "feature100": {"scenario": "feature_dirichlet", "alpha": 100.0},
    "feature01": {"scenario": "feature_dirichlet", "alpha": 0.1},
}


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("label", "client_id", "domain_id"):
            row[key] = int(row[key])
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS + ["domain_id"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["relative_path"]))


def manifest_hash(path):
    files = list(path.glob("*.csv")) + list((path / "clients").glob("*.csv"))
    files += [path / "partition_config.json", path / "image_content.json"]
    return digest({p.relative_to(path).as_posix(): file_hash(p) for p in sorted(files)})


def semantic_rows(rows, domain=False):
    keys = ("relative_path", "label", "class_name", "group_id") + (("domain_id",) if domain else ())
    return digest([tuple(r[k] for k in keys) for r in sorted(rows, key=lambda x: x["relative_path"])])


def tiny_pool(rows, groups_per_class):
    # Whole groups, fixed independently for train/val/test and all conditions.
    rng = np.random.default_rng(20260916)
    return [r for groups in group_samples(rows, key="group_id").values()
            for i in rng.permutation(len(groups))[:groups_per_class] for r in groups[i]["items"]]


def assign_clients(rows, condition, seed=42):
    rows = deepcopy(rows)
    groups = group_samples(rows, key="group_id")
    feature = condition["scenario"] == "feature_dirichlet"
    if feature:
        # Synthetic acquisition domains are fixed per leaf group, stratified by
        # true class, BEFORE the Dirichlet draw. Labels are never overwritten.
        domains = {d: [] for d in range(5)}
        rng = np.random.default_rng(seed + 70001)
        for values in groups.values():
            for i, pos in enumerate(rng.permutation(len(values))):
                group = values[pos]
                domain = i % 5
                for row in group["items"]:
                    row["domain_id"] = domain
                domains[domain].append(group)
        groups = domains
    clients, diagnostics = partition(
        groups, "label_skew" if feature else condition["scenario"], 5,
        alpha=condition.get("alpha", 0.1), quantity_alpha=condition.get("quantity_alpha", 0.1),
        seed=seed, min_samples_per_client=10, max_retries=50,
    )
    for cid, items in clients.items():
        for row in items:
            row.update(client_id=cid, split="train", scenario=condition["scenario"],
                       alpha=condition.get("alpha", 0.1), quantity_alpha=condition.get("quantity_alpha", 0.1),
                       feature_skew="synthetic_domain" if feature else "none", seed=seed)
            if not feature:
                row["domain_id"] = -1
    return clients, diagnostics


def prepare(output, dataset_root, leaf_map, smoke=False):
    output, dataset_root, leaf_map = Path(output).resolve(), Path(dataset_root).resolve(), Path(leaf_map).resolve()
    if output.exists() and any(p.name != "source_cache.json" for p in output.iterdir()):
        raise ValueError("Refuse to replace frozen manifests; use preflight or a new output directory")
    if not leaf_map.is_file():
        raise FileNotFoundError(leaf_map)
    output.mkdir(parents=True, exist_ok=True)
    cache = build_or_update_source_content_cache(dataset_root, output / "source_cache.json", strict=True)
    
    verified_pairs = []
    review_file = ROOT / "data/duplicate_review.json"
    if not review_file.is_file():
        raise FileNotFoundError(f"Missing duplicate review file: {review_file}")
    reviews = json.loads(review_file.read_text(encoding="utf-8"))
    # Union-find both accepted and uncertain candidate pairs to guarantee zero cross-split or cross-client leakage
    verified_pairs = [(r["a"], r["b"]) for r in reviews if r.get("decision") in ("accepted", "uncertain")]

    splitter = DatasetPartitioner(str(dataset_root), str(leaf_map), base_dir=str(ROOT),
        test_ratio=0.20, val_ratio=0.08, num_clients=5, seed=42,
        group_aware=True, content_aware=True, content_hashes=cache["images"],
        verified_pairs=verified_pairs)
    splitter.scan_dataset()
    train, test, val = splitter.split_global_test()
    if smoke:
        train, val, test = tiny_pool(train, 2), tiny_pool(val, 1), tiny_pool(test, 1)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        for row in rows:
            row.update(relative_path="/".join(row["relative_path"].replace("\\", "/").split("/")[-2:]),
                       split=split, client_id=-1, domain_id=-1, feature_skew="none", seed=42)
    
    # Regression check on the 6 visually verified pairs across splits
    visually_verified_file = ROOT / "data/visually_verified_pairs.json"
    if not visually_verified_file.is_file():
        raise FileNotFoundError(f"Missing visually verified pairs file: {visually_verified_file}")
    v_pairs = json.loads(visually_verified_file.read_text(encoding="utf-8"))
    split_map = {r["relative_path"]: split for split, rows in (("train", train), ("val", val), ("test", test)) for r in rows}
    for vp in v_pairs:
        na = "/".join(vp["a"].replace("\\", "/").split("/")[-2:])
        nb = "/".join(vp["b"].replace("\\", "/").split("/")[-2:])
        if na in split_map and nb in split_map:
            assert split_map[na] == split_map[nb], (
                f"Regression failure: Verified pair crossed splits: {na} in {split_map[na]}, {nb} in {split_map[nb]}"
            )

    inventory = {r["relative_path"]: cache["images"][r["relative_path"]]["sha256"] for r in train + val + test}
    common = {"train": semantic_rows(train), "val": semantic_rows(val), "test": semantic_rows(test),
              "inventory": digest(inventory), "classes": digest(splitter.class_names)}
    suite = {"protocol": PROTOCOL, "smoke": smoke, "split_seed": 42, "num_clients": 5,
             "target_ratios": [0.72, 0.08, 0.20], "class_names": splitter.class_names,
             "counts": {"train": len(train), "val": len(val), "test": len(test)},
             "common": common, "leaf_map_sha256": file_hash(leaf_map), "leaf_audit": splitter.leaf_audit,
             "source_images": cache["total_images"], "conditions": {}}
    for name, condition in CONDITIONS.items():
        clients, diagnostics = assign_clients(train, condition)
        central = [r for cid in range(5) for r in clients[cid]]
        
        # Regression check on the 4 visually verified pairs across clients in training
        if v_pairs:
            client_map = {r["relative_path"]: r["client_id"] for r in central}
            for vp in v_pairs:
                na = "/".join(vp["a"].replace("\\", "/").split("/")[-2:])
                nb = "/".join(vp["b"].replace("\\", "/").split("/")[-2:])
                if na in client_map and nb in client_map:
                    assert client_map[na] == client_map[nb], (
                        f"Regression failure: Verified pair crossed clients in {name}: {na} (client {client_map[na]}) vs {nb} (client {client_map[nb]})"
                    )

        feature = condition["scenario"] == "feature_dirichlet"
        path = output / name
        for cid, rows in clients.items():
            write_rows(path / "clients" / f"client_{cid:02d}.csv", rows)
        for filename, rows in (("centralized_train", central), ("global_val", val), ("global_test", test)):
            write_rows(path / f"{filename}.csv", rows)
        meta = {"protocol": PROTOCOL, "smoke": smoke, "condition": name, **condition,
                "content_aware": True, "group_aware": True, "num_clients": 5, "seed": 42,
                "class_names": splitter.class_names, "n_k": [len(clients[c]) for c in range(5)],
                "split_diagnostics": diagnostics, "domain_kind": "synthetic" if feature else "none",
                "domain_profiles": [create_deterministic_client_profile(d, 42, "moderate").to_dict()
                                    for d in range(5)] if feature else [],
                "domain_assignment_sha256": semantic_rows(central, domain=True),
                "client_domain_counts": [[sum(r["domain_id"] == d for r in clients[c]) for d in range(5)]
                                         for c in range(5)] if feature else [],
                "partition_metrics": calculate_partition_metrics(clients, 38, splitter.class_names)}
        atomic_json(path / "partition_config.json", meta)
        atomic_json(path / "image_content.json", inventory)
        qa = audit_manifest_directory(path, len(inventory))
        qa["content"] = audit_image_content(path, dataset_root, hashes=inventory)
        suite["conditions"][name] = {**condition, "manifest_sha256": manifest_hash(path),
                                    "n_k": meta["n_k"], "audit": qa}
    import shutil
    shutil.copy2(review_file, output / "duplicate_review.json")
    shutil.copy2(visually_verified_file, output / "visually_verified_pairs.json")
    suite["duplicate_review_sha256"] = file_hash(output / "duplicate_review.json")
    suite["visually_verified_pairs_sha256"] = file_hash(output / "visually_verified_pairs.json")
    atomic_json(output / "suite.json", suite)
    return suite


def preflight(suite_dir, dataset_root=None, allow_legacy=False):
    root = Path(suite_dir).resolve()
    suite = json.loads((root / "suite.json").read_text(encoding="utf-8"))
    if not allow_legacy and suite["protocol"] != PROTOCOL:
        raise ValueError(f"Old suite protocol {suite.get('protocol')} is rejected; clean GĐ2 requires {PROTOCOL}")
    if (suite["protocol"] not in (PROTOCOL, "stage2_scratch_clean_v2", "stage2_matched_clean_v5") or suite["num_clients"] != 5 or suite["split_seed"] != 42
            or suite["target_ratios"] != [0.72, 0.08, 0.20] or set(suite["conditions"]) != set(CONDITIONS)):
        raise ValueError("Incompatible matched protocol")
    if not suite["smoke"] and sum(suite["counts"].values()) != 54305:
        raise ValueError("Full benchmark must cover the complete 54305-image dataset")

    # Verify duplicate_review and visually_verified_pairs files
    dup_review_path = root / "duplicate_review.json"
    if not dup_review_path.is_file():
        raise FileNotFoundError(f"Missing duplicate_review.json in {root}")
    if suite.get("duplicate_review_sha256") and file_hash(dup_review_path) != suite["duplicate_review_sha256"]:
        raise ValueError(f"duplicate_review.json hash mismatch: expected {suite['duplicate_review_sha256']}")

    visual_pairs_path = root / "visually_verified_pairs.json"
    if not visual_pairs_path.is_file():
        raise FileNotFoundError(f"Missing visually_verified_pairs.json in {root}")
    if suite.get("visually_verified_pairs_sha256") and file_hash(visual_pairs_path) != suite["visually_verified_pairs_sha256"]:
        raise ValueError(f"visually_verified_pairs.json hash mismatch: expected {suite['visually_verified_pairs_sha256']}")

    duplicate_review = json.loads(dup_review_path.read_text(encoding="utf-8"))
    visually_verified_pairs = json.loads(visual_pairs_path.read_text(encoding="utf-8"))
    if len(visually_verified_pairs) < 6:
        raise ValueError(f"Expected at least 6 visually verified pairs, found {len(visually_verified_pairs)}")

    # Pre-hash observed images once across the entire benchmark if dataset_root is provided
    observed_hashes = None
    if dataset_root is not None:
        d_root = Path(dataset_root).resolve()
        if not d_root.exists() or not d_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {d_root}")
        first_cond = list(CONDITIONS.keys())[0]
        inv_file = root / first_cond / "image_content.json"
        if not inv_file.exists():
            raise FileNotFoundError(f"Missing image_content.json in {root / first_cond}")
        inventory = json.loads(inv_file.read_text(encoding="utf-8"))
        observed_hashes = {}
        for rel_path in inventory:
            if ".." in rel_path:
                raise ValueError(f"Image path outside dataset: {rel_path}")
            img_file = d_root / rel_path
            try:
                data = img_file.read_bytes()
            except FileNotFoundError:
                raise FileNotFoundError(f"Image file not found on disk: {img_file}")
            observed_hashes[rel_path] = hashlib.sha256(data).hexdigest()

    feature_assignments = []
    audits = {}
    for name, condition in CONDITIONS.items():
        path = root / name
        entry = suite["conditions"][name]
        if any(entry.get(k) != v for k, v in condition.items()) or manifest_hash(path) != entry["manifest_sha256"]:
            raise ValueError(f"Manifest or condition changed: {name}")
        meta = json.loads((path / "partition_config.json").read_text(encoding="utf-8"))
        if (any(meta.get(k) != v for k, v in condition.items()) or meta["condition"] != name
                or not meta["content_aware"] or not meta["group_aware"] or meta["class_names"] != suite["class_names"]):
            raise ValueError(f"Invalid metadata: {name}")
        audits[name] = audit_manifest_directory(path, sum(suite["counts"].values()))
        if audits[name]["num_clients"] != 5:
            raise ValueError("Exactly five clients required")
        central = read_rows(path / "centralized_train.csv")
        val, test = (read_rows(path / f"global_{s}.csv") for s in ("val", "test"))
        common = {"train": semantic_rows(central), "val": semantic_rows(val), "test": semantic_rows(test),
                  "inventory": digest(json.loads((path / "image_content.json").read_text())),
                  "classes": digest(meta["class_names"])}
        if common != suite["common"]:
            raise ValueError("Conditions must share identical train pool, holdouts and content inventory")
        by_path = {r["relative_path"]: r for r in central}
        n_k = []
        for cid in range(5):
            rows = read_rows(path / "clients" / f"client_{cid:02d}.csv")
            n_k.append(len(rows))
            if any(r != by_path[r["relative_path"]] for r in rows):
                raise ValueError("Client rows/domain assignments differ from centralized union")
        if n_k != entry["n_k"] or n_k != meta["n_k"]:
            raise ValueError("Incorrect aggregation sample counts")
        if any(r["domain_id"] != -1 for r in val + test):
            raise ValueError("Holdouts must stay clean")
        if condition["scenario"] == "feature_dirichlet":
            if any(not 0 <= int(r["domain_id"]) < 5 for r in central):
                raise ValueError("Missing/invalid domain ID")
            if any(len({r["domain_id"] for r in g["items"]}) != 1
                   for gs in group_samples(central, "group_id").values() for g in gs):
                raise ValueError("A leaf group crosses synthetic domains")
            profiles = [create_deterministic_client_profile(d, 42, "moderate").to_dict() for d in range(5)]
            if meta["domain_profiles"] != profiles:
                raise ValueError("Feature strength differs from preregistered moderate profiles")
            feature_assignments.append(semantic_rows(central, domain=True))
        elif any(r["domain_id"] != -1 for r in central) or meta["domain_profiles"]:
            raise ValueError("Unexpected feature transform")

        # Verify verified pairs (visual + duplicate review accepted/uncertain) in this condition
        def _norm(p):
            return "/".join(p.replace("\\", "/").split("/")[-2:])

        split_map = {}
        for r in central:
            split_map[_norm(r["relative_path"])] = ("train", r["client_id"], r["group_id"])
        for r in val:
            split_map[_norm(r["relative_path"])] = ("val", -1, r["group_id"])
        for r in test:
            split_map[_norm(r["relative_path"])] = ("test", -1, r["group_id"])

        for pair in visually_verified_pairs:
            na, nb = _norm(pair["a"]), _norm(pair["b"])
            if na not in split_map or nb not in split_map:
                raise ValueError(f"Visually verified pair missing from manifests in {name}: {na}, {nb}")
            split_a, client_a, group_a = split_map[na]
            split_b, client_b, group_b = split_map[nb]
            if group_a != group_b:
                raise ValueError(f"Visually verified pair group mismatch in {name}: {na} (group {group_a}) vs {nb} (group {group_b})")
            if split_a != split_b:
                raise ValueError(f"Visually verified pair split crossing in {name}: {na} in {split_a} vs {nb} in {split_b}")
            if split_a == "train" and client_a != client_b:
                raise ValueError(f"Visually verified pair client crossing in {name}: {na} (client {client_a}) vs {nb} (client {client_b})")

        for item in duplicate_review:
            if item.get("decision") in ("accepted", "uncertain"):
                na, nb = _norm(item["a"]), _norm(item["b"])
                if na not in split_map or nb not in split_map:
                    raise ValueError(f"Review pair missing from manifests in {name}: {na}, {nb}")
                split_a, client_a, group_a = split_map[na]
                split_b, client_b, group_b = split_map[nb]
                if group_a != group_b:
                    raise ValueError(f"Review pair ({item.get('decision')}) group mismatch in {name}: {na} (group {group_a}) vs {nb} (group {group_b})")
                if split_a != split_b:
                    raise ValueError(f"Review pair ({item.get('decision')}) split crossing in {name}: {na} in {split_a} vs {nb} in {split_b}")
                if split_a == "train" and client_a != client_b:
                    raise ValueError(f"Review pair ({item.get('decision')}) client crossing in {name}: {na} (client {client_a}) vs {nb} (client {client_b})")

        if dataset_root is not None:
            expected_content = json.loads((path / "image_content.json").read_text(encoding="utf-8")) if (path / "image_content.json").exists() else None
            audits[name]["content"] = audit_image_content(
                path,
                Path(dataset_root),
                observed_hashes=observed_hashes,
                expected_hashes=expected_content,
            )
        audits[name]["duplicate_review_verified"] = True
        audits[name]["visual_pairs_verified"] = len(visually_verified_pairs)
        audits[name]["review_pairs_verified"] = len([i for i in duplicate_review if i.get("decision") in ("accepted", "uncertain")])

    if len(set(feature_assignments)) != 1:
        raise ValueError("Feature controls must share the same per-image synthetic domains")
    return suite, audits


def keep_pil(image, rng_seed=None):
    return image


class ManifestDataset(Dataset):
    def __init__(self, rows, root, training=False, profiles=(), is_train=None, domain_profiles=None):
        if is_train is not None:
            training = is_train
        if domain_profiles is not None:
            profiles = domain_profiles
        self.rows, self.root, self.training = rows, Path(root).resolve(), training
        self.targets = [r["label"] for r in rows]
        self.domains = {p["client_id"]: ClientDomainTransform(ClientFeatureProfile(**p), base_transform=keep_pil)
                        for p in profiles} if training else {}
        self.identity_context = {}
        self.round_seconds = 60.0
        self.initialization_sha256: Optional[str] = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path escapes dataset")
        with Image.open(path) as source:
            img = source.convert("RGB")
        if self.training and row["domain_id"] >= 0:
            # Stable per-image noise, independent of client ordering and alpha.
            noise_seed = int(digest(row["relative_path"])[:8], 16)
            img = self.domains[row["domain_id"]](img, rng_seed=noise_seed)
        return (TRAIN_TRANSFORM if self.training else EVAL_TRANSFORM)(img), row["label"]
