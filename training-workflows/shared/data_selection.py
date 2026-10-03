"""Select existing, audited views of one canonical dataset at launch time.

The release owns the train/validation/calibration/test split. A workflow may
choose the full train manifest or one audited client partition; it never moves
images or changes the release manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RELEASE_NAME = "pv_pd_v3"
RELEASE_SHA256 = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def inspect_dataset(dataset_root: Path, partition_scheme: str | None = None,
                    expected_clients: int = 5, expected_sha: str = RELEASE_SHA256) -> dict:
    """Return the run's dataset view after verifying the immutable release.

    A selected partition is checked against the train union and release audit.
    This work is read-only; physical image bytes remain under ``dataset_root``.
    """
    dataset_root = dataset_root.expanduser().resolve()
    release = dataset_root / "mixed" / RELEASE_NAME
    release_manifest = release / "release_manifest.json"
    if not release_manifest.is_file():
        raise FileNotFoundError(f"Canonical release not found: {release_manifest}")
    actual_sha = sha256(release_manifest)
    if actual_sha != expected_sha:
        raise ValueError(f"Release SHA mismatch: {actual_sha} != {expected_sha}")
    train = release / "manifests" / "train.jsonl"
    validation = release / "manifests" / "val.jsonl"
    calibration = release / "manifests" / "calibration.jsonl"
    diagnostic = release / "manifests" / "legacy_diagnostic_test.jsonl"
    view = {
        "dataset_root": str(dataset_root),
        "release_dir": str(release),
        "release_manifest_sha256": actual_sha,
        "train_manifest": str(train),
        "train_count": _count_jsonl(train),
        "validation_manifest": str(validation),
        "validation_count": _count_jsonl(validation),
        "calibration_manifest": str(calibration),
        "calibration_count": _count_jsonl(calibration),
        "diagnostic_test_manifest": str(diagnostic),
        "diagnostic_test_count": _count_jsonl(diagnostic),
        "partition_scheme": partition_scheme,
        "client_shards": {},
    }
    if partition_scheme is None:
        return view
    if not partition_scheme or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in partition_scheme):
        raise ValueError("Partition scheme must be a lowercase release directory name")
    if expected_clients < 2 or expected_clients > 100:
        raise ValueError("expected_clients must be in [2, 100]")
    spec_path = release / "partition_spec_v4.json"
    partition_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if partition_spec.get("release_manifest_sha256") != actual_sha:
        raise ValueError("Partition audit belongs to a different release")
    if partition_spec.get("num_clients") != expected_clients:
        raise ValueError("Partition client count differs from selected job")
    scenario = partition_spec.get("scenarios", {}).get(partition_scheme)
    if partition_scheme != "iid" and (not isinstance(scenario, dict) or scenario.get("status") != "PASS"):
        raise ValueError(f"Partition {partition_scheme!r} has no PASS audit")
    if isinstance(scenario, dict) and scenario.get("metadata", {}).get("skew_type") == "feature_skew_dirichlet":
        if float(scenario.get("mean_js_divergence", 1.0)) > 0.05:
            raise ValueError("Feature skew partition also changes label priors too much")
    shard_dir = release / "partitions" / partition_scheme
    expected_names = {f"client_{i:02d}.jsonl" for i in range(expected_clients)}
    actual_names = {path.name for path in shard_dir.glob("client_*.jsonl")}
    if actual_names != expected_names:
        raise ValueError(f"Partition files mismatch: {sorted(actual_names)}")
    audit_hashes = scenario.get("shard_hashes", {}) if isinstance(scenario, dict) else {}
    for name in sorted(expected_names):
        shard = shard_dir / name
        count = _count_jsonl(shard)
        if name in audit_hashes and sha256(shard) != audit_hashes[name]:
            raise ValueError(f"Partition audit SHA mismatch: {shard}")
        view["client_shards"][name.removesuffix(".jsonl")] = {"manifest": str(shard), "count": count}
    if sum(item["count"] for item in view["client_shards"].values()) != view["train_count"]:
        raise ValueError("Selected partition does not cover the full train count")
    return view


def print_plan(view: dict, command: list[str] | None = None) -> None:
    print(json.dumps({"dataset_selection": view, "command": command}, ensure_ascii=False, indent=2))
