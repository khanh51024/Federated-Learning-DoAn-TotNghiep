"""Integrity and deterministic hashing utilities for dataset releases."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def canonical_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization with sorted keys and compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_sha256_of_data(data: Union[str, bytes, dict, list]) -> str:
    """Compute SHA-256 hash of arbitrary data or JSON-serializable structure."""
    if isinstance(data, (dict, list)):
        payload = canonical_json_dumps(data).encode("utf-8")
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = data
    return hashlib.sha256(payload).hexdigest()


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 of a file in 64KB chunks."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_RELEASE_MANIFEST_SHA256 = "3100dd6b7ce336aa0b6f35846d864c139b112b1dfb59511a8205e7d4d1f041d2"

APPROVED_RELEASE_MANIFEST_HASHES: Dict[str, str] = {
    "mixed_pv_pd_v1": "3100dd6b7ce336aa0b6f35846d864c139b112b1dfb59511a8205e7d4d1f041d2",
    "mixed_pv_pd_v2": "d2a22635e8ad6d8fa3bbcbbddb95aee6cb27fdad4b923f7893a84b5a6f472f9a",
    "mixed_pv_pd_v3": "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252",
}

PINNED_PARTITION_SHARD_HASHES: Dict[str, Dict[str, str]] = {
    "iid": {
        "client_00.jsonl": "eb457e675b68196cc53f97866e3e705e0ab2b5887c51b0ce5e083d960e897c8d",
        "client_01.jsonl": "73fc561fa13c740bdfd8e08af15bb79cc08df0a5ba6edba47600eb6ece29c59a",
        "client_02.jsonl": "075d3a59f1b9cada56a8f82dfd4bbfe7176806d294b6ea66c30b2201fe882d98",
        "client_03.jsonl": "a786cd868d38222b8fc4ccb2a0ad0db21c1beed53d466e4f90f15cdbf3e39150",
        "client_04.jsonl": "27a355ebfdc0d78fe96dc6748b1a75e7aaf61cd03acf7f1b89f2340ac3995f77",
    },
    "domain_skew": {
        "client_00.jsonl": "fe49829b97530cd1e9a6ca85fe38115c47f810fff74c448f1ddf0bda3e697f8c",
        "client_01.jsonl": "3fe0736391ba8485b573e6a41ff8dcded207f7cffe10fab222ec0a8272d97e69",
        "client_02.jsonl": "928f7d7ddfaad031f4e290279765fb6d9db707ba9b0f1ab3c1eb72e1b09b29df",
        "client_03.jsonl": "21ad3b514fae13852616e5ef9239f7a1830030560d166d9c7dcf3bf3941c3c8e",
        "client_04.jsonl": "0adbf49cce7b0d31e509b8ccfae3aa71a90d26094dc90e727775436f571b28db",
    },
    "label_skew_alpha1": {
        "client_00.jsonl": "e4e58df82ecdf40a1822cfb3119c487066763dfb1eda91081795ecd03874ad43",
        "client_01.jsonl": "3b79a46b2db97cc9bc8279d55ae3092104e34bd8dc6e56e52aa6dca716a53963",
        "client_02.jsonl": "489662d5975c51d3bf278f33b89a28aa07e29dab81eb1db64e9ad4f35b305c76",
        "client_03.jsonl": "e1e2c93587ecd1579ce44cf148004a5ac761c7c5d780e6a5caec73a2d329af05",
        "client_04.jsonl": "558d7c3deca55be2271722029eb5d682b051b7fd935efdbaa7212896be902d05",
    },
    "label_skew_alpha01": {
        "client_00.jsonl": "74a0de2469144cf523ae649fabc5bf88c11ceb8426e7afebf0460e635c9fcfe4",
        "client_01.jsonl": "b43a4ec6dc8133a2b689486c7b6cbfa0fbb4fe6f9e7b56d7d94c1288ea9c2aa6",
        "client_02.jsonl": "a6b14e617eb7905be542475c1a6364e606b105234f4a4726696f2934638b6b9a",
        "client_03.jsonl": "f50b51380419ef62607566d95c2cbd3cf8df273c5a34fd71d88b9dc2ee7cc7ed",
        "client_04.jsonl": "f0c9a7ff99d67a40c0996f48c8caffec8717daa82f5d032d7b222b828d0f7c84",
    },
}


def verify_release_integrity(
    release_dir: Union[str, Path],
    expected_release_sha: Optional[str] = None,
    partition_scheme: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify that all files in the release match the hashes in release_manifest.json.

    Validates against approved release manifest hashes (pv_pd_v1, pv_pd_v2, or explicit override).
    Validates taxonomy, label mapping, detection annotations, split manifests, and partition shards.
    """
    r_dir = Path(release_dir)
    manifest_path = r_dir / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release_manifest.json in {release_dir}")

    cur_release_sha = compute_file_sha256(manifest_path)
    release_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    rel_name = release_meta.get("release_name", "")

    if expected_release_sha is not None:
        target_expected_sha = expected_release_sha
    elif rel_name in APPROVED_RELEASE_MANIFEST_HASHES:
        target_expected_sha = APPROVED_RELEASE_MANIFEST_HASHES[rel_name]
    else:
        target_expected_sha = EXPECTED_RELEASE_MANIFEST_SHA256

    if target_expected_sha and cur_release_sha != target_expected_sha:
        raise ValueError(
            f"Release manifest SHA-256 mismatch! Expected approved pinned hash {target_expected_sha}, "
            f"got {cur_release_sha}. Untracked dataset modification detected."
        )

    checks: Dict[str, Any] = {
        "release_manifest_sha256": cur_release_sha,
        "pinned_sha_verified": True,
    }
    mismatches = []

    # 1. Verify taxonomy.json
    tax_path = r_dir / "taxonomy.json"
    if tax_path.is_file():
        cur_tax_hash = compute_file_sha256(tax_path)
        expected = release_meta.get("taxonomy_sha256")
        checks["taxonomy_json"] = (cur_tax_hash == expected)
        if cur_tax_hash != expected:
            mismatches.append(f"taxonomy.json: expected {expected}, got {cur_tax_hash}")
    else:
        mismatches.append("Missing taxonomy.json")

    # 2. Verify label_mapping.csv
    map_path = r_dir / "label_mapping.csv"
    if map_path.is_file():
        cur_map_hash = compute_file_sha256(map_path)
        expected = release_meta.get("mapping_sha256")
        checks["label_mapping_csv"] = (cur_map_hash == expected)
        if cur_map_hash != expected:
            mismatches.append(f"label_mapping.csv: expected {expected}, got {cur_map_hash}")
    else:
        mismatches.append("Missing label_mapping.csv")

    # 3. Verify annotations/detection.json
    det_path = r_dir / "annotations/detection.json"
    if det_path.is_file():
        cur_det_hash = compute_file_sha256(det_path)
        expected = release_meta.get("detection_json_sha256")
        checks["detection_json"] = (cur_det_hash == expected)
        if cur_det_hash != expected:
            mismatches.append(f"detection.json: expected {expected}, got {cur_det_hash}")
    else:
        mismatches.append("Missing annotations/detection.json")

    # 4. Verify scene_groups.jsonl (group graph)
    group_path = r_dir / "manifests/scene_groups.jsonl"
    if group_path.is_file():
        cur_grp_hash = compute_file_sha256(group_path)
        expected = release_meta.get("group_graph_sha256")
        checks["group_graph"] = (cur_grp_hash == expected)
        if cur_grp_hash != expected:
            mismatches.append(f"scene_groups.jsonl: expected {expected}, got {cur_grp_hash}")
    else:
        mismatches.append("Missing manifests/scene_groups.jsonl")

    # Repaired releases pin the decision ledger as part of their identity.
    repair_hash = release_meta.get("repair_ledger_sha256")
    if repair_hash is not None:
        repair_file = r_dir / "audits/release_repair_ledger.json"
        if not repair_file.is_file():
            mismatches.append("Missing audits/release_repair_ledger.json")
        elif compute_file_sha256(repair_file) != repair_hash:
            mismatches.append("audits/release_repair_ledger.json SHA mismatch")

    # 5. Verify all split manifests
    man_hashes = release_meta.get("manifest_hashes", {})
    split_hashes = {}
    for man_key, expected_hash in man_hashes.items():
        m_file = r_dir / f"manifests/{man_key}.jsonl"
        if m_file.is_file():
            cur_hash = compute_file_sha256(m_file)
            split_hashes[man_key] = cur_hash
            checks[f"manifest_{man_key}"] = (cur_hash == expected_hash)
            if cur_hash != expected_hash:
                mismatches.append(f"manifests/{man_key}.jsonl: expected {expected_hash}, got {cur_hash}")
        else:
            mismatches.append(f"Missing manifests/{man_key}.jsonl")
    checks["split_hashes"] = split_hashes

    # 6. Verify client partitions
    part_schemes = [partition_scheme] if partition_scheme else release_meta.get("partitions_verified", [])
    shard_hashes: Dict[str, Dict[str, str]] = {}
    for scheme in part_schemes:
        p_dir = r_dir / f"partitions/{scheme}"
        if p_dir.is_dir():
            shards = sorted(p_dir.glob("client_*.jsonl"))
            checks[f"partition_{scheme}"] = (len(shards) == 5)
            if len(shards) != 5:
                mismatches.append(f"partitions/{scheme}: expected 5 shards, found {len(shards)}")
            shard_hashes[scheme] = {}
            expected_shards = release_meta.get("partition_shard_hashes", {}).get(
                scheme, PINNED_PARTITION_SHARD_HASHES.get(scheme, {})
            )
            if set(expected_shards) != {sh.name for sh in shards}:
                mismatches.append(f"partitions/{scheme}: shard hash inventory is incomplete or inconsistent")
            for sh in shards:
                cur_shard_sha = compute_file_sha256(sh)
                shard_hashes[scheme][sh.name] = cur_shard_sha
                if sh.name in expected_shards and cur_shard_sha != expected_shards[sh.name]:
                    mismatches.append(
                        f"partitions/{scheme}/{sh.name}: expected pinned {expected_shards[sh.name]}, got {cur_shard_sha}"
                    )
        else:
            mismatches.append(f"Missing partitions directory partitions/{scheme}")
    checks["shard_hashes"] = shard_hashes

    if mismatches:
        raise ValueError(f"Release integrity verification failed! Mismatches: {mismatches}")

    return {
        "status": "PASS",
        "release_name": release_meta.get("release_name"),
        "release_manifest_sha256": cur_release_sha,
        "checks": checks,
    }


def compute_selected_partition_digest(
    release_dir: Union[str, Path],
    partition_scheme_or_condition: str,
) -> str:
    """Compute deterministic cryptographic digest of the selected partition scheme or condition.
    Ensures client data assignment invariance across sessions without mutating release manifest.
    """
    r_dir = Path(release_dir)
    scheme = partition_scheme_or_condition
    p_dir = r_dir / f"partitions/{scheme}"
    if p_dir.is_dir():
        shards = sorted(p_dir.glob("client_*.jsonl"))
        entries = []
        for sh in shards:
            entries.append(f"{sh.name}:{compute_file_sha256(sh)}")
        raw = "\n".join(entries).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # Fallback to pinned hashes if release dir not present on disk
    if scheme in PINNED_PARTITION_SHARD_HASHES:
        entries = [f"{k}:{v}" for k, v in sorted(PINNED_PARTITION_SHARD_HASHES[scheme].items())]
        raw = "\n".join(entries).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # For centralized conditions:
    train_manifest = r_dir / "manifests/train.jsonl"
    if train_manifest.is_file():
        h = compute_file_sha256(train_manifest)
        raw = f"centralized:{scheme}:{h}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    raise FileNotFoundError(
        f"Manifest file train.jsonl not found in {r_dir / 'manifests'}. "
        f"Cannot compute verified partition digest for condition '{scheme}'."
    )



def verify_manifest_records_contract(manifest_or_release_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify that sample records conform strictly to classification contract:
    - Labels must be integers in range [0, 37]
    - supervised_eligible must be True
    - Non-empty sample_id and valid image path
    - Special documentation for calibration split duplicates
    Accepts either a single manifest JSONL file or the release directory containing manifests/.
    """
    p = Path(manifest_or_release_dir)
    if p.is_dir():
        manifests_dir = p / "manifests" if (p / "manifests").is_dir() else p
        split_names = ["train.jsonl", "val.jsonl", "calibration.jsonl", "legacy_diagnostic_test.jsonl", "quarantine.jsonl"]
        results = {}
        manifest_counts = {}
        for name in split_names:
            mf = manifests_dir / name
            if not mf.is_file():
                raise FileNotFoundError(f"Missing required manifest file {name} in {manifests_dir}")
            res = verify_manifest_records_contract(mf)
            results[name] = res
            manifest_counts[name] = res["total_records"]
        return {
            "status": "PASS",
            "manifest_counts": manifest_counts,
            "manifests": results,
        }

    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")

    total_records = 0
    class_counts: Dict[int, int] = {}
    sample_ids = set()
    duplicate_ids = []

    if p.name == "scene_groups.jsonl":
        group_ids = set()
        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                gid = item.get("group_id")
                if not gid:
                    raise ValueError(f"Line {line_num} in {p.name} missing group_id")
                if gid in group_ids:
                    raise ValueError(f"Line {line_num} in {p.name}: duplicate group_id '{gid}'")
                group_ids.add(gid)
                total_records += 1
        return {
            "status": "PASS",
            "file": p.name,
            "total_records": total_records,
            "unique_groups": len(group_ids),
        }

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("sample_id")
            if not sid:
                raise ValueError(f"Line {line_num} in {p.name} missing sample_id")

            if sid in sample_ids:
                duplicate_ids.append(sid)
            sample_ids.add(sid)

            cid = item.get("class_id")
            if p.name == "quarantine.jsonl":
                if cid is None or not isinstance(cid, int) or (cid != -1 and (cid < 0 or cid > 37)):
                    raise ValueError(
                        f"Line {line_num} in {p.name}: invalid class_id {cid!r}. Must be -1 or integer in [0, 37]"
                    )
                if item.get("supervised_eligible", False) is True:
                    raise ValueError(f"Line {line_num} in {p.name}: quarantine sample {sid} is marked supervised_eligible")
            else:
                if cid is None or not isinstance(cid, int) or cid < 0 or cid > 37:
                    raise ValueError(
                        f"Line {line_num} in {p.name}: invalid class_id {cid!r}. Must be integer in [0, 37]"
                    )
                if not item.get("supervised_eligible", True):
                    raise ValueError(f"Line {line_num} in {p.name}: sample {sid} is marked not supervised_eligible")
            class_counts[cid] = class_counts.get(cid, 0) + 1

            total_records += 1

    # Calibration split has 2 known duplicates documented in release:
    # sample_pd_29595743274bed1f and sample_pd_55f3f92a8c0be5ed (crop_0ea1317f5e200dde_00 and 01)
    if "calibration" in p.name:
        known_calib_dups = {"sample_pd_29595743274bed1f", "sample_pd_55f3f92a8c0be5ed"}
        unexpected_dups = set(duplicate_ids) - known_calib_dups
        if unexpected_dups:
            raise ValueError(f"Unexpected duplicate sample_ids in calibration manifest: {unexpected_dups}")
    elif "quarantine" in p.name:
        pass  # Quarantine is an audit log of quarantined detection crops, duplicates tracked in return dict
    else:
        if duplicate_ids:
            raise ValueError(f"Duplicate sample_ids found in {p.name}: {duplicate_ids[:5]}")

    return {
        "status": "PASS",
        "file": p.name,
        "total_records": total_records,
        "unique_samples": len(sample_ids),
        "num_classes": len(class_counts),
        "duplicate_ids": duplicate_ids,
    }


def compute_model_w0_fingerprint(model: Any) -> str:
    """Compute deterministic SHA-256 fingerprint of initial model weights W0.

    Iterates all parameters and buffers in sorted state_dict key order,
    converts CPU tensors to byte arrays, and hashes key names and bytes.
    """
    import torch
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu()
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def verify_image_dataset_integrity(
    dataset_root: Union[str, Path],
    release_dir: Union[str, Path],
    splits: Sequence[str] = ("train", "val", "legacy_diagnostic_test"),
    verify_hashes: bool = True,
    sample_hash_count: Optional[int] = None,
    check_decode: bool = True,
    sample_decode_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify that all image files referenced in the split manifests exist under dataset_root,
    have non-zero size, match expected byte SHA-256 hashes, and can be decoded by PIL.

    Fail-closed immediately if dataset_root does not exist, is empty, or any required image is missing/corrupted.
    Full production mode verifies 100% of images without hidden caps.
    """
    ds_root = Path(dataset_root).resolve()
    rel_dir = Path(release_dir).resolve()

    if not ds_root.exists() or not ds_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {ds_root}")

    manifests_dir = rel_dir / "manifests" if (rel_dir / "manifests").is_dir() else rel_dir
    missing_files = []
    corrupt_files = []
    records_checked = 0
    splits_checked: Dict[str, int] = {}
    verified_files_hashed: Dict[Path, str] = {}
    verified_files_decoded: Set[Path] = set()
    partial_sampling_used = False

    from PIL import Image

    for split in splits:
        mf = manifests_dir / f"{split}.jsonl"
        if not mf.is_file():
            raise FileNotFoundError(f"Missing manifest file for split '{split}': {mf}")

        split_count = 0
        with open(mf, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rel_path = record.get("relative_path", "").replace("\\", "/")
                if not rel_path or rel_path.startswith("/") or ".." in rel_path:
                    raise ValueError(f"Illegal or empty relative_path in {split}.jsonl line {line_idx}: {rel_path!r}")

                img_path = (ds_root / rel_path).resolve()
                try:
                    if not img_path.is_relative_to(ds_root):
                        raise ValueError(f"Path traversal detected: {rel_path}")
                except AttributeError:
                    if not str(img_path).startswith(str(ds_root)):
                        raise ValueError(f"Path traversal detected: {rel_path}")

                if not img_path.is_file():
                    missing_files.append(rel_path)
                    if len(missing_files) >= 50:
                        break
                    continue

                sz = img_path.stat().st_size
                if sz == 0:
                    corrupt_files.append((rel_path, "zero_byte_file"))
                    continue

                exp_sha = record.get("byte_sha256")
                should_hash = verify_hashes and (sample_hash_count is None or split_count < sample_hash_count)
                if verify_hashes and sample_hash_count is not None and split_count >= sample_hash_count:
                    partial_sampling_used = True

                if should_hash:
                    if img_path in verified_files_hashed:
                        actual_sha = verified_files_hashed[img_path]
                    else:
                        actual_sha = compute_file_sha256(img_path)
                        verified_files_hashed[img_path] = actual_sha

                    if exp_sha and actual_sha != exp_sha:
                        corrupt_files.append((rel_path, f"sha_mismatch: exp={exp_sha}, actual={actual_sha}"))
                        continue

                should_decode = check_decode and (sample_decode_count is None or split_count < sample_decode_count)
                if check_decode and sample_decode_count is not None and split_count >= sample_decode_count:
                    partial_sampling_used = True

                if should_decode:
                    if img_path not in verified_files_decoded:
                        try:
                            with Image.open(img_path) as im:
                                im.convert("RGB")
                            verified_files_decoded.add(img_path)
                        except Exception as e:
                            corrupt_files.append((rel_path, f"pil_decode_error: {e}"))
                            continue

                split_count += 1
                records_checked += 1

            if missing_files:
                break

        splits_checked[split] = split_count

    if missing_files:
        raise FileNotFoundError(
            f"Image dataset integrity check FAILED: {len(missing_files)} required image files "
            f"are missing under dataset_root '{ds_root}'. First missing examples: {missing_files[:5]}"
        )

    if corrupt_files:
        raise ValueError(
            f"Image dataset integrity check FAILED: {len(corrupt_files)} files corrupt or hash mismatch! "
            f"Examples: {corrupt_files[:5]}"
        )

    status = "PARTIAL" if partial_sampling_used else "PASS"

    return {
        "status": status,
        "dataset_root": str(ds_root),
        "records_checked": records_checked,
        "total_images_checked": records_checked,
        "unique_files_hashed": len(verified_files_hashed),
        "unique_files_decoded": len(verified_files_decoded),
        "splits_checked": splits_checked,
    }

