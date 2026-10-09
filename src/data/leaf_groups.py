"""
Leaf-group index for PlantVillage (group-aware splitting).

WHY THIS EXISTS
---------------
PlantVillage contains several near-duplicate photographs of the *same physical
leaf* (in the subset covered by the official leaf map: 41,111 images resolving
to 7,595 distinct leaves, i.e. ~5.4 images per leaf). A purely image-level
random split therefore leaks near-duplicates:

  * train <-> global test  -> inflated accuracy for ALL three baselines
  * client <-> client      -> facilities that are supposed to be independent
                              silently share the same leaf

Both effects distort the central measurement of the thesis (the
Centralized / Federated / Local-only gap), so the partition unit must be the
leaf, not the image.

This module resolves every image to a `group_id`:
  * images listed in `leaf-map.json`  -> "<class>:::<leaf#>"   (one physical leaf)
  * images with no leaf-map entry     -> "img::<relative_path>" (singleton group)

Singleton fallback keeps the split exact and lossless for the ~24% of images
the official leaf map does not cover (8 of 38 classes have no leaf metadata at
all); it simply degrades to image-level splitting for those images.

Leaf-map key format (produced by the dataset's own `aggregate_map.py`):
    key   = lowercase stem of the ORIGINAL filename, extension stripped
    value = ["<ClassDirName>:::<LeafNumber>", ...]   (may hold >1 class)
Dataset filenames carry a UUID prefix: "<uuid>___<original stem>.JPG".
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _normalize_leaf_number(raw: str) -> str:
    """'54.0' -> '54'; keeps ints readable without changing distinctness."""
    raw = raw.strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw


def load_leaf_map(leaf_map_path: Path | str) -> Dict[str, List[str]]:
    """Load `leaf-map.json` -> {lowercase_original_stem: [group_id, ...]}."""
    leaf_map_path = Path(leaf_map_path)
    if not leaf_map_path.exists():
        return {}
    with open(leaf_map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, List[str]] = {}
    for stem, entries in raw.items():
        if isinstance(entries, str):
            entries = [entries]
        out = []
        for e in entries:
            if ":::" not in e:
                continue
            cls, leaf_no = e.split(":::", 1)
            out.append(f"{cls}:::{_normalize_leaf_number(leaf_no)}")
        if out:
            normalized[stem.strip().lower()] = out
    return normalized


def stem_to_leaf_key(filename: str) -> str:
    """`<uuid>___RS_Early.B 8178.JPG` -> `rs_early.b 8178` (leaf-map key)."""
    stem = os.path.splitext(filename)[0]
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    return stem.strip().lower()


def resolve_group_id(
    filename: str,
    class_name: str,
    relative_path: str,
    leaf_map: Dict[str, List[str]],
) -> Tuple[str, bool]:
    """
    Return (group_id, resolved_from_leaf_map).

    When a stem is claimed by more than one class in the leaf map, the entry
    matching this image's own class directory wins (leaf numbering is
    per-class-file, so a physical leaf belongs to exactly one class).
    """
    entries = leaf_map.get(stem_to_leaf_key(filename))
    if not entries:
        return f"img::{relative_path}", False

    for e in entries:
        if e.split(":::", 1)[0] == class_name:
            return e, True

    # A key recorded under a different or ambiguous class cannot be trusted.
    # Fall back to a singleton and expose this as unresolved coverage.
    return f"img::{relative_path}", False


def attach_group_ids(
    samples: List[Dict[str, Any]],
    leaf_map: Dict[str, List[str]],
    content_hashes: Optional[Dict[str, str]] = None,
    verified_pairs: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Add a `group_id` field to every sample dict **in place** and return an
    audit summary of how well the leaf map covered the dataset.
    """
    resolved = 0
    per_class_resolved: Counter = Counter()
    per_class_total: Counter = Counter()

    base_groups: List[str] = []
    for s in samples:
        filename = s["relative_path"].rsplit("/", 1)[-1]
        gid, ok = resolve_group_id(filename, s["class_name"], s["relative_path"], leaf_map)
        base_groups.append(gid)
        s["leaf_group_id"] = gid
        per_class_total[s["class_name"]] += 1
        if ok:
            resolved += 1
            per_class_resolved[s["class_name"]] += 1

    # Exact-byte duplicates and verified near-duplicates/same-source pairs are
    # sources of leakage for images not covered by the leaf map. When hashes
    # or verified pairs are supplied, collapse the connected components formed
    # by leaf groups, byte hashes, and verified pairs. This keeps every known
    # leaf intact while preventing related images from landing in different
    # splits or clients.
    if content_hashes is not None or verified_pairs is not None:
        if content_hashes is not None and not content_hashes:
            raise ValueError('Content-aware grouping requires complete content hashes')
        parent = list(range(len(samples)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        first_by_key: Dict[str, int] = {}
        for i, gid in enumerate(base_groups):
            if gid in first_by_key:
                union(i, first_by_key[gid])
            else:
                first_by_key[gid] = i
            if content_hashes is not None:
                rel = samples[i]["relative_path"]
                digest = content_hashes.get(rel) or content_hashes.get(
                    "/".join(rel.replace("\\", "/").split("/")[-2:])
                )
                if not digest:
                    raise ValueError(f'Missing content hash for {rel}')
                key = f"bytes::{digest}"
                if key in first_by_key:
                    other = samples[first_by_key[key]]
                    if other['class_name'] != samples[i]['class_name']:
                        raise ValueError(f'Identical image bytes have conflicting labels: {rel}')
                    union(i, first_by_key[key])
                else:
                    first_by_key[key] = i

        if verified_pairs is not None:
            sample_by_rel: Dict[str, int] = {}
            for i, s in enumerate(samples):
                norm = "/".join(s["relative_path"].replace("\\", "/").split("/")[-2:])
                sample_by_rel[norm] = i

            for pair in verified_pairs:
                if isinstance(pair, dict):
                    p_a, p_b = pair["a"], pair["b"]
                else:
                    p_a, p_b = pair[0], pair[1]
                norm_a = "/".join(p_a.replace("\\", "/").split("/")[-2:])
                norm_b = "/".join(p_b.replace("\\", "/").split("/")[-2:])
                if norm_a in sample_by_rel and norm_b in sample_by_rel:
                    idx_a = sample_by_rel[norm_a]
                    idx_b = sample_by_rel[norm_b]
                    if samples[idx_a]['class_name'] != samples[idx_b]['class_name']:
                        raise ValueError(
                            f'Verified pair has conflicting labels: {norm_a} ({samples[idx_a]["class_name"]}) '
                            f'vs {norm_b} ({samples[idx_b]["class_name"]})'
                        )
                    union(idx_a, idx_b)

        roots: Dict[int, str] = {}
        for i, s in enumerate(samples):
            root = find(i)
            roots.setdefault(root, f"partition::{root:08d}")
            s["group_id"] = roots[root]
        content_group_count = len(roots)
    else:
        for s, gid in zip(samples, base_groups):
            s["group_id"] = gid
        content_group_count = None

    total = len(samples)
    group_sizes = Counter(s["group_id"] for s in samples)
    multi = [n for n in group_sizes.values() if n > 1]

    return {
        "total_images": total,
        "images_with_leaf_group": resolved,
        "leaf_map_coverage": round(resolved / total, 4) if total else 0.0,
        "distinct_groups": len(group_sizes),
        "groups_with_multiple_images": len(multi),
        "images_per_group_mean": round(total / len(group_sizes), 3) if group_sizes else 0.0,
        "images_per_group_max": max(group_sizes.values()) if group_sizes else 0,
        "classes_with_leaf_metadata": len(per_class_resolved),
        "classes_total": len(per_class_total),
        "singleton_group_images": total - resolved,
        "per_class_coverage": {
            c: round(per_class_resolved[c] / per_class_total[c], 4)
            for c in sorted(per_class_total)
        },
        "content_aware": bool(content_hashes),
        "partition_groups": content_group_count if content_group_count is not None else len(group_sizes),
    }


def group_samples(
    samples: List[Dict[str, Any]],
    key: str = "split_id",
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Collapse a flat sample list into whole, indivisible assignment units.

    `key` selects the unit of assignment:
      * "split_id"    (default) -- one physical leaf when group_aware is on,
                                   one image when it is off
      * "group_id"    -- always the true leaf, used for auditing

    Returns {class_id: [unit, ...]} where unit = {"group_id": str, "items": [...]}.
    Units are ordered by id and items by relative_path, so the structure is
    byte-stable regardless of filesystem scan order.
    """
    buckets: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for s in samples:
        buckets[s["label"]].setdefault(s.get(key, s["group_id"]), []).append(s)

    out: Dict[int, List[Dict[str, Any]]] = {}
    for class_id in sorted(buckets):
        groups = []
        for gid in sorted(buckets[class_id]):
            items = sorted(buckets[class_id][gid], key=lambda x: x["relative_path"])
            groups.append({"group_id": gid, "items": items})
        out[class_id] = groups
    return out
