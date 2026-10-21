"""Dựng release dữ liệu `mixed_pv_pd_v4`: v3 cộng ảnh ngoài đồng đã qua kiểm nhãn, lá PlantDoc cách ly đã được nhận và ảnh tự chụp.

Mã dựng v3 không còn trong repo, nên đây là bộ dựng MỚI, chạy trên các manifest v3 đã ghim (đọc và bổ sung, không sửa v3). Đầu ra ở
data/dataset_sources/mixed/pv_pd_v4/ có đúng cấu trúc mà runner huấn luyện đọc được (xem docs/DATASET_V4.md):
    release_manifest.json  taxonomy.json  label_mapping.csv  annotations/detection.json  audits/
    manifests/{train,val,calibration,legacy_diagnostic_test,quarantine,scene_groups}.jsonl
    partitions/{label_alpha_0_1,iid}/client_00..04.jsonl   (5 client, chia theo nhóm; bộ chia v3: src/data/dirichlet_split.py)
Ảnh ngoài đồng mới được nối cứng (hardlink) vào data/dataset_sources/external_v4/ để dùng chung một --dataset-root.

Quy tắc (đặt trước, ghi vào release_manifest.json):
  * Ảnh ngoài đồng chỉ vào release khi: nhãn `trusted`, hoặc nhóm (nguồn, lớp) được nhận ở data/label_review_ext/status.json.
  * Loại ảnh gần trùng (dHash 64 bit, Hamming <= 6, cả bản lật ngang) với: PlantVillage, cảnh và lá PlantDoc của v3, PlantWild (tập thử khóa) và phần thử ảnh tự chụp.
  * Ảnh gần trùng giữa các nguồn (Hamming <= 4) được gộp CÙNG MỘT nhóm để chia train/test cùng phía.
  * Chia ảnh ngoài đồng theo nhóm bằng mã băm cố định: 70% train, 10% val, 10% calibration, 10% legacy_diagnostic_test.
  * PlantVillage train bị giới hạn `--pv-cap` ảnh mỗi lớp (chọn theo mã băm); PlantVillage val/calibration/test giữ nguyên để so với số cũ.
  * Lá PlantDoc cách ly chỉ vào khi nhóm lớp được nhận ở data/label_review/status.json; giữ nguyên split dự kiến của v3.
  * Ảnh tự chụp: split theo cây (scripts/check_own_photos.py); phần thử vào legacy_diagnostic_test.
  * source_domain của mọi ảnh ngoài đồng = "plantdoc" (runner chỉ biết hai miền); nguồn thật ở `source_dataset`.

Chạy:
    python scripts/build_release_v4.py                 # dựng và kiểm rò rỉ; ghi artifacts/release_v4_report.json
    python scripts/build_release_v4.py --pv-cap 800 --seed 20261022
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

RELEASE_NAME = "mixed_pv_pd_v4"
V4_DIR = "data/dataset_sources/mixed/pv_pd_v4"
EXTERNAL_LINK_DIR = "external_v4"
EXT_SOURCES = "data/external/sources"
EXT_STATUS = "data/label_review_ext/status.json"
QUARANTINE_STATUS = "data/label_review/status.json"
OWN_MANIFEST = "data/own_photos/manifest.jsonl"
DHASH_CACHE = "data/external/dhash_cache.npz"
DUP_HAMMING = 6
GROUP_MERGE_HAMMING = 4
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.80), ("calibration", 0.90), ("legacy_diagnostic_test", 1.00))
SPLIT_FILES = ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine")
NUM_CLIENTS = 5


# --------------------------------------------------------------------------- phần thuần (có test)
def split_for_group(group_id: str, seed: int) -> str:
    """70/10/10/10 theo mã băm cố định của nhóm; cùng nhóm luôn cùng split."""
    h = int(hashlib.sha256(f"{seed}|{group_id}".encode()).hexdigest()[:8], 16) / 2**32
    for name, upper in SPLIT_FRACTIONS:
        if h < upper:
            return name
    return "legacy_diagnostic_test"


def min_hamming(queries: np.ndarray, refs: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Khoảng cách Hamming nhỏ nhất từ mỗi phần tử `queries` (uint64) tới tập `refs`; 64 nếu refs rỗng."""
    if len(refs) == 0:
        return np.full(len(queries), 64, dtype=np.int64)
    out = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        out[start:start + chunk] = np.bitwise_count(np.bitwise_xor(block[:, None], refs[None, :])).min(axis=1)
    return out


def merge_near_duplicate_groups(hashes: np.ndarray, groups: list[str], max_hamming: int) -> list[str]:
    """Union-find: ảnh có Hamming <= max_hamming (bất kể nguồn) gộp nhóm. Trả group_id mới (đại diện = nhóm nhỏ nhất theo thứ tự chữ)."""
    parent = {g: g for g in set(groups)}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(hashes), 256):
        block = hashes[start:start + 256]
        dist = np.bitwise_count(np.bitwise_xor(block[:, None], hashes[None, :]))
        rows, cols = np.nonzero(dist <= max_hamming)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i == c:
                continue
            a, b = find(groups[i]), find(groups[c])
            if a != b:
                keep, drop = sorted((a, b))
                parent[drop] = keep
    return [find(g) for g in groups]


def conflicting_groups(classes: list[int], groups: list[str]) -> set[str]:
    """Nhóm gần trùng có ảnh mang hơn một nhãn: hai ảnh gần như giống hệt mà khác nhãn thì ít nhất một nhãn sai."""
    seen: dict[str, set[int]] = defaultdict(set)
    for class_id, group in zip(classes, groups):
        seen[group].add(int(class_id))
    return {g for g, c in seen.items() if len(c) > 1}


def groups_by_primary_class(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Gom ảnh theo nhóm rồi xếp trọn mỗi nhóm vào lớp chiếm đa số (hòa thì lớp nhỏ hơn): bộ chia theo lớp không cắt một nhóm cho nhiều client."""
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        members[r["group_id"]].append(r)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for gid, items in sorted(members.items()):
        counts = Counter(int(i["class_id"]) for i in items)
        primary = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        by_class[primary].append({"group_id": gid, "items": items})
    return dict(by_class)


SCENE_SPLIT_NAME = {"train": "train", "val": "validation", "calibration": "calibration", "legacy_diagnostic_test": "test", "quarantine": "quarantine"}
SCENE_ROLE = {"train": "supervised_train", "validation": "supervised_val", "calibration": "post_selection_calibration",
              "test": "legacy_diagnostic", "quarantine": "quarantine"}


def build_scene_groups(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Một dòng mỗi group_id theo định dạng v3 (split dùng tên 'validation'/'test'). Nhóm vừa ở quarantine vừa ở split khác (một cảnh có lá được nhận,
    lá khác còn cách ly) tính theo split không phải quarantine để group_id không lặp (bộ kiểm toàn vẹn của runner yêu cầu duy nhất)."""
    chosen: dict[str, str] = {}
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("quarantine", "train", "val", "calibration", "legacy_diagnostic_test"):  # quarantine trước để bị ghi đè
        for r in rows_by_split[split]:
            chosen[r["group_id"]] = SCENE_SPLIT_NAME[split]
            members[r["group_id"]].append(r)
    out = []
    for gid in sorted(chosen):
        sources = Counter(r["source_domain"] for r in members[gid])
        out.append({"evaluation_role": SCENE_ROLE[chosen[gid]], "group_id": gid, "num_samples": len(members[gid]), "sources": dict(sources), "split": chosen[gid]})
    return out


def cap_per_class(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """Giữ tối đa `cap` dòng mỗi lớp, chọn theo mã băm cố định (không phụ thuộc thứ tự)."""
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    kept: list[dict[str, Any]] = []
    for cid, items in by_class.items():
        items = sorted(items, key=lambda r: hashlib.sha256(f"{seed}|{r['sample_id']}".encode()).hexdigest())
        kept.extend(items[:cap])
    return sorted(kept, key=lambda r: r["sample_id"])


# --------------------------------------------------------------------------- I/O
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def accepted_set(path: str, key: str = "accepted") -> set[str]:
    p = reg.abs_path(path)
    return set(json.loads(p.read_text(encoding="utf-8")).get(key, [])) if p.is_file() else set()


def mirrored_hash(path: Path) -> int:
    from PIL import Image, ImageOps

    from scripts.build_external_eval_set import dhash64

    with Image.open(path) as image:
        return dhash64(ImageOps.mirror(image.convert("RGB")))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_reference_hashes(v3_rows: list[dict[str, Any]], own_test: list[dict[str, Any]], workers: int) -> tuple[np.ndarray, dict[str, int]]:
    """dHash của mọi ảnh phải được tách biệt: PlantVillage, cảnh PlantDoc, lá PlantDoc v3, PlantWild và ảnh tự chụp phần thử."""
    from PIL import Image

    from scripts.build_external_eval_set import dhash64, hash_many

    cache_path = reg.abs_path(DHASH_CACHE)
    if not cache_path.is_file():
        raise SystemExit("thiếu data/external/dhash_cache.npz: chạy scripts/build_external_eval_set.py trước")
    data = np.load(cache_path, allow_pickle=False)
    cache = {str(k): int(v) for k, v in zip(data["paths"], data["hashes"])}
    base = reg.abs_path(reg.DATASET_SOURCES_DIR)
    crop_paths = [base / r["relative_path"] for r in v3_rows if r["source_domain"] == "plantdoc"]
    hash_many(crop_paths, cache, workers)
    np.savez_compressed(cache_path, paths=np.array(list(cache)), hashes=np.array(list(cache.values()), dtype=np.uint64))
    hashes = np.array(list(cache.values()), dtype=np.uint64)
    own = []
    for r in own_test:
        with Image.open(reg.abs_path("data") / r["relative_path"]) as image:
            own.append(dhash64(image))
    if own:
        hashes = np.concatenate([hashes, np.array(own, dtype=np.uint64)])
    return hashes, {"reference_images": int(len(hashes)), "own_test_images": len(own)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pv-cap", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-link", action="store_true", help="không tạo liên kết ảnh (chỉ thử dựng manifest)")
    args = parser.parse_args()

    from scripts.audit_release_leakage import audit_overlaps, audit_partition
    from src.data.dirichlet_split import partition

    v3 = reg.abs_path(reg.RELEASE_V3_DIR)
    v4 = reg.abs_path(V4_DIR)
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]

    # ---- v3
    v3_rows = {name: read_jsonl(v3 / "manifests" / f"{name}.jsonl") for name in SPLIT_FILES}
    # quarantine.jsonl của v3 chứa 23 dòng lặp y hệt nhau (cùng sample_id): giữ một bản để sample_id duy nhất trong cả release
    seen_ids: set[str] = set()
    unique_quarantine = []
    for r in v3_rows["quarantine"]:
        if r["sample_id"] not in seen_ids:
            seen_ids.add(r["sample_id"])
            unique_quarantine.append(r)
    v3_rows["quarantine"] = unique_quarantine
    own_all = read_jsonl(reg.abs_path(OWN_MANIFEST))
    own_test = [r for r in own_all if r["split"] == "test"]
    own_train = [r for r in own_all if r["split"] == "train"]

    # ---- ảnh ngoài đồng sẵn sàng
    ext_ok = accepted_set(EXT_STATUS)
    ext_samples_ok = accepted_set(EXT_STATUS, "accepted_samples")
    ext_rows: list[dict[str, Any]] = []
    not_ready: Counter[str] = Counter()
    for manifest in sorted(reg.abs_path(EXT_SOURCES).glob("*/manifest.jsonl")):
        for r in read_jsonl(manifest):
            if r["label_review_status"] == "trusted" or f"{r['source_dataset']}|{r['target_class']}" in ext_ok or r["sample_id"] in ext_samples_ok:
                ext_rows.append(r)
            else:
                not_ready[f"{r['source_dataset']}|{r['target_class']}"] += 1
    print(f"[v4] ảnh ngoài đồng sẵn sàng: {len(ext_rows)}; chờ kiểm nhãn: {sum(not_ready.values())}", flush=True)

    # ---- loại trùng với các tập đã khóa
    ref_hashes, ref_info = load_reference_hashes(
        [r for name in ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine") for r in v3_rows[name]], own_test, args.workers)
    ext_base = reg.abs_path(EXT_SOURCES)
    paths = [ext_base / r["relative_path"] for r in ext_rows]
    direct = np.array([int(r["dhash64"], 16) for r in ext_rows], dtype=np.uint64)
    flipped = np.array([int(r["dhash64_flip"], 16) if r.get("dhash64_flip") else mirrored_hash(p) for r, p in zip(ext_rows, paths)], dtype=np.uint64)
    d_direct, d_flip = min_hamming(direct, ref_hashes), min_hamming(flipped, ref_hashes)
    is_dup = (np.minimum(d_direct, d_flip) <= DUP_HAMMING)
    dup_by_source = Counter(r["source_dataset"] for r, d in zip(ext_rows, is_dup) if d)
    kept_rows = [r for r, d in zip(ext_rows, is_dup) if not d]
    kept_hashes = direct[~is_dup]
    print(f"[v4] loại {int(is_dup.sum())} ảnh gần trùng với tập khóa/PlantVillage: {dict(dup_by_source)}", flush=True)

    # ---- gộp nhóm gần trùng giữa các nguồn, chia split theo nhóm
    merged = merge_near_duplicate_groups(kept_hashes, [r["group_id"] for r in kept_rows], GROUP_MERGE_HAMMING)
    # Hai ảnh gần như giống hệt nhau mà mang hai nhãn khác nhau: ít nhất một nhãn sai -> bỏ cả nhóm (nhãn mâu thuẫn), không đoán nhãn nào đúng.
    conflict_groups = conflicting_groups([int(r["class_id"]) for r in kept_rows], merged)
    conflict_by_source = Counter(r["source_dataset"] for r, g in zip(kept_rows, merged) if g in conflict_groups)
    print(f"[v4] bỏ {sum(conflict_by_source.values())} ảnh thuộc {len(conflict_groups)} nhóm gần trùng có nhãn mâu thuẫn: {dict(conflict_by_source)}", flush=True)
    ext_out: list[dict[str, Any]] = []
    for r, group in ((r, g) for r, g in zip(kept_rows, merged) if g not in conflict_groups):
        split = split_for_group(group, args.seed)
        link_rel = f"{EXTERNAL_LINK_DIR}/{r['source_dataset']}/{r['target_class']}/{r['byte_sha256'][:16]}.jpg"
        if not args.no_link:
            link_or_copy(ext_base / r["relative_path"], dataset_root / link_rel)
        ext_out.append({
            "sample_id": r["sample_id"], "relative_path": link_rel, "original_path": r.get("original_file", ""), "byte_sha256": r["byte_sha256"],
            "class_id": int(r["class_id"]), "target_class": r["target_class"], "source_dataset": r["source_dataset"], "source_domain": "plantdoc",
            "group_id": group, "parent_scene_id": group, "image_id": r["sample_id"], "annotation_id": "", "crop_policy_version": "whole_leaf_canonical",
            "mapping_status": r["mapping_status"], "quarantine_status": "clean", "supervised_eligible": True, "split": split,
            "license": r.get("license"), "label_trust": r["label_trust"], "capture_note": r.get("capture_note", ""),
        })

    # ---- lá PlantDoc cách ly đã được nhận
    q_ok = accepted_set(QUARANTINE_STATUS)
    promoted: list[dict[str, Any]] = []
    still_quarantined: list[dict[str, Any]] = []
    for r in v3_rows["quarantine"]:
        usable = (r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean" and r.get("target_class")
                  and int(r.get("class_id", -1)) >= 0 and r["target_class"] in q_ok and r.get("split") in ("train", "val", "calibration", "legacy_diagnostic_test"))
        if usable:
            promoted.append({**r, "supervised_eligible": True, "label_trust": "reviewed_quarantine"})
        else:
            still_quarantined.append(r)

    # ---- ảnh tự chụp
    own_rows = [{"sample_id": r["sample_id"], "relative_path": r["relative_path"], "original_path": r.get("original_file", ""), "byte_sha256": r["byte_sha256"],
                 "class_id": r["class_id"], "target_class": r["target_class"], "source_dataset": "own_photos", "source_domain": "plantdoc",
                 "group_id": r["group_id"], "parent_scene_id": r["group_id"], "image_id": r["sample_id"], "annotation_id": "",
                 "crop_policy_version": "whole_leaf_canonical", "mapping_status": "exact_match", "quarantine_status": "clean", "supervised_eligible": True,
                 "split": "train" if r["split"] == "train" else "legacy_diagnostic_test", "license": "own", "label_trust": r.get("label_trust", "self")}
                for r in own_all]
    for r in own_rows:  # ảnh tự chụp nằm ngoài dataset-root: nối cứng vào external_v4 như các nguồn khác
        link_rel = f"{EXTERNAL_LINK_DIR}/own_photos/{r['target_class']}/{r['byte_sha256'][:16]}.jpg"
        if not args.no_link:
            link_or_copy(reg.abs_path("data") / r["relative_path"], dataset_root / link_rel)
        r["relative_path"] = link_rel

    # ---- gộp theo split
    out: dict[str, list[dict[str, Any]]] = {name: list(v3_rows[name]) for name in ("train", "val", "calibration", "legacy_diagnostic_test")}
    pv_before = sum(1 for r in out["train"] if r["source_domain"] == "plantvillage")
    pv_train = [r for r in out["train"] if r["source_domain"] == "plantvillage"]
    out["train"] = [r for r in out["train"] if r["source_domain"] != "plantvillage"] + cap_per_class(pv_train, args.pv_cap, args.seed)
    for r in [*ext_out, *promoted, *own_rows]:
        out[r["split"]].append(r)
    out["quarantine"] = still_quarantined
    for name in out:
        out[name] = sorted(out[name], key=lambda r: r["sample_id"])

    # ---- kiểm rò rỉ (cùng khóa với release v3) + bốn tập thử giữ nguyên
    overlaps = audit_overlaps({k: out[k] for k in ("train", "val", "calibration", "legacy_diagnostic_test")})
    total_overlap = sum(v for pair in overlaps.values() for v in pair.values())
    if total_overlap:
        print(f"[v4] RÒ RỈ: {overlaps}", file=sys.stderr)
        return 2
    ids = Counter(r["sample_id"] for name in out for r in out[name])
    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        print(f"[v4] sample_id trùng: {dup_ids[:5]}", file=sys.stderr)
        return 2

    # ---- ghi release
    (v4 / "manifests").mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {name: write_jsonl(v4 / "manifests" / f"{name}.jsonl", out[name]) for name in SPLIT_FILES}
    scene_groups = build_scene_groups(out)
    groups = [(g["group_id"], g["split"]) for g in scene_groups]
    hashes["scene_groups"] = write_jsonl(v4 / "manifests" / "scene_groups.jsonl", scene_groups)
    for name in ("taxonomy.json", "label_mapping.csv"):
        shutil.copy2(v3 / name, v4 / name)
    (v4 / "annotations").mkdir(exist_ok=True)
    shutil.copy2(v3 / "annotations" / "detection.json", v4 / "annotations" / "detection.json")
    (v4 / "audits").mkdir(exist_ok=True)
    shutil.copy2(v3 / "audits" / "release_repair_ledger.json", v4 / "audits" / "release_repair_ledger.json")

    # ---- phân vùng 5 client trên train (theo nhóm)
    shard_hashes: dict[str, dict[str, str]] = {}
    audits: dict[str, Any] = {}
    train_ids = {r["sample_id"] for r in out["train"]}
    for scheme, scenario, alpha in (("label_alpha_0_1", "label_skew", 0.1), ("iid", "iid", 1.0)):
        # Bộ chia theo lớp: một nhóm có ảnh thuộc nhiều lớp (vài cảnh PlantDoc của v3) được xếp trọn vào lớp chiếm đa số để không bị chia cho nhiều client.
        groups_by_class = groups_by_primary_class(out["train"])
        shards, diag = partition(groups_by_class, scenario, NUM_CLIENTS, alpha=alpha, seed=42)
        shard_hashes[scheme] = {}
        shard_rows: dict[str, list[dict[str, Any]]] = {}
        for client in range(NUM_CLIENTS):
            rows = sorted(shards[client], key=lambda r: r["sample_id"])
            name = f"client_{client:02d}"
            shard_hashes[scheme][name] = write_jsonl(v4 / "partitions" / scheme / f"{name}.jsonl", rows)
            shard_rows[name] = rows
        audits[scheme] = {**audit_partition(shard_rows, train_ids), "diagnostics": diag,
                          "n_k": [len(shard_rows[f"client_{c:02d}"]) for c in range(NUM_CLIENTS)]}
    for scheme, a in audits.items():
        if not (a["disjoint"] and a["union_equals_train"] and a["groups_split_across_clients"] == 0):
            print(f"[v4] phân vùng {scheme} không đạt: {a}", file=sys.stderr)
            return 2

    v3_manifest = json.loads((v3 / "release_manifest.json").read_text(encoding="utf-8"))
    counts = {"train_samples": len(out["train"]), "validation_samples": len(out["val"]), "calibration_samples": len(out["calibration"]),
              "legacy_diagnostic_test_samples": len(out["legacy_diagnostic_test"]), "num_scene_groups": len(scene_groups),
              "detection_images": v3_manifest["counts"]["detection_images"], "detection_annotations": v3_manifest["counts"]["detection_annotations"],
              "total_classification_samples": sum(len(out[n]) for n in out)}
    by_source = {n: dict(Counter(r["source_dataset"] if r["source_domain"] == "plantdoc" and r["source_dataset"] not in ("PlantDoc", "plantdoc") else r["source_domain"]
                                 for r in out[n])) for n in ("train", "val", "calibration", "legacy_diagnostic_test")}
    manifest = {
        "release_name": RELEASE_NAME, "schema_version": "3.0.0", "num_classes": 38, "crop_policy": v3_manifest["crop_policy"],
        "preprocessing_version": v3_manifest["preprocessing_version"], "derived_from_release": v3_manifest["release_name"],
        "derived_from_release_manifest_sha256": sha256_file(v3 / "release_manifest.json"),
        "counts": {**counts, "quarantined_samples": len(out["quarantine"]), "external_images_added": len(ext_out), "own_photos": len(own_rows),
                   "plantdoc_quarantine_promoted": len(promoted), "plantvillage_train_before_cap": pv_before, "pv_cap_per_class": args.pv_cap},
        "manifest_hashes": hashes, "partition_shard_hashes": shard_hashes, "partitions_verified": sorted(audits),
        "taxonomy_sha256": sha256_file(v4 / "taxonomy.json"), "mapping_sha256": sha256_file(v4 / "label_mapping.csv"),
        "detection_json_sha256": sha256_file(v4 / "annotations" / "detection.json"), "group_graph_sha256": hashes["scene_groups"],
        "repair_ledger_sha256": sha256_file(v4 / "audits" / "release_repair_ledger.json"),
        "split_sha256": sha256_bytes(json.dumps(groups, sort_keys=True).encode()),
        "rules": {"dedup_hamming_max": DUP_HAMMING, "group_merge_hamming_max": GROUP_MERGE_HAMMING, "external_split": dict(SPLIT_FRACTIONS), "seed": args.seed,
                  "locked_tests": "PlantVillage/PlantDoc v3 (val, legacy_diagnostic_test), PlantWild, ảnh tự chụp phần thử"},
    }
    (v4 / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (v4 / "dataset_card.md").write_text(
        "# Mixed PlantVillage–PlantDoc v4\n\nv3 cộng ảnh ngoài đồng đã kiểm nhãn, lá PlantDoc cách ly đã được nhận và ảnh tự chụp. "
        "Xem docs/DATASET_V4.md. Ảnh ngoài đồng nối cứng vào external_v4/; source_domain=plantdoc nghĩa là \"ngoài đồng\".\n", encoding="utf-8")
    report = {
        "release": RELEASE_NAME, "release_manifest_sha256": sha256_file(v4 / "release_manifest.json"), "counts": manifest["counts"],
        "reference": ref_info, "removed_near_duplicates": {"total": int(is_dup.sum()), "by_source": dict(dup_by_source)},
        "dropped_label_conflict": {"groups": len(conflict_groups), "images": sum(conflict_by_source.values()), "by_source": dict(conflict_by_source)},
        "waiting_label_review": dict(not_ready), "overlaps": overlaps, "partitions": audits, "by_source": by_source,
        "by_class": {c: {n: sum(1 for r in out[n] if r["target_class"] == c) for n in ("train", "val", "calibration", "legacy_diagnostic_test")} for c in classes},
    }
    reg.abs_path(reg.RELEASE_V4_REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[v4] xong: {counts}; ảnh ngoài đồng thêm {len(ext_out)}, lá PlantDoc cách ly nhận lại {len(promoted)}, ảnh tự chụp {len(own_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
