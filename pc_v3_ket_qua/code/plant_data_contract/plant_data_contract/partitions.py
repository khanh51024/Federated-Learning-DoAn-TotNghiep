"""Partition validation and verification helpers for Federated Learning."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


def verify_client_partitions(
    client_shards: Dict[str, List[Dict[str, Any]]],
    centralized_train_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Strictly verify group-disjointness, union completeness, and client sample counts.

    Raises:
        ValueError if any FedAvg invariant is violated.
    """
    if not client_shards:
        raise ValueError("At least one client shard is required")

    central_records: Dict[str, Dict[str, Any]] = {}
    for s in centralized_train_samples:
        sid = s.get("sample_id")
        if not sid:
            raise ValueError("Centralized train record missing sample_id")
        if sid in central_records:
            raise ValueError(f"Duplicate sample_id '{sid}' in centralized train split!")
        central_records[sid] = s

    central_ids: Set[str] = set(central_records.keys())
    union_ids: Set[str] = set()
    group_to_client: Dict[str, str] = {}
    client_metrics: Dict[str, Any] = {}

    for cid, samples in client_shards.items():
        if not samples:
            raise ValueError(f"Client {cid} has empty sample partition!")

        c_sample_ids: Set[str] = set()
        c_groups: Set[str] = set()
        c_domains: Counter = Counter()
        c_classes: Counter = Counter()

        for s in samples:
            sid = s.get("sample_id")
            gid = s.get("group_id")

            if not sid:
                raise ValueError(f"Client {cid} sample missing sample_id!")
            if not gid:
                raise ValueError(f"Client {cid} sample '{sid}' missing group_id!")

            if sid not in central_records:
                raise ValueError(f"Client {cid} contains sample '{sid}' not present in centralized train split!")

            ref = central_records[sid]
            # Record-level equality: prevent stealth tampering with group_id, class_id, relative_path, source_domain, byte_sha256
            if s.get("group_id") != ref.get("group_id"):
                raise ValueError(
                    f"Tampered group_id for sample '{sid}' in client {cid}: "
                    f"shard has {s.get('group_id')!r}, train has {ref.get('group_id')!r}"
                )
            if s.get("class_id") != ref.get("class_id"):
                raise ValueError(
                    f"Tampered class_id for sample '{sid}' in client {cid}: shard has {s.get('class_id')!r}, train has {ref.get('class_id')!r}"
                )
            if s.get("relative_path") != ref.get("relative_path"):
                raise ValueError(
                    f"Tampered relative_path for sample '{sid}' in client {cid}: shard has {s.get('relative_path')!r}, train has {ref.get('relative_path')!r}"
                )
            if s.get("source_domain") != ref.get("source_domain"):
                raise ValueError(
                    f"Tampered source_domain for sample '{sid}' in client {cid}: shard has {s.get('source_domain')!r}, train has {ref.get('source_domain')!r}"
                )
            if "byte_sha256" in ref and s.get("byte_sha256") != ref.get("byte_sha256"):
                raise ValueError(
                    f"Tampered byte_sha256 for sample '{sid}' in client {cid}: shard has {s.get('byte_sha256')!r}, train has {ref.get('byte_sha256')!r}"
                )

            # Strict class_id validation: type must be int, not bool!
            cid_val = s.get("class_id")
            if isinstance(cid_val, bool):
                raise ValueError(
                    f"Client {cid} sample '{sid}' has boolean class_id {cid_val!r}. "
                    "Must be strict integer in [0, 37]"
                )
            if cid_val is None or type(cid_val) is not int or cid_val < 0 or cid_val > 37:
                raise ValueError(f"Client {cid} sample '{sid}' has invalid class_id {cid_val!r}. Must be strict integer in [0, 37]")

            # Strict supervised_eligible validation: must be present, must be strictly boolean True
            if "supervised_eligible" not in s or type(s.get("supervised_eligible")) is not bool or not s.get("supervised_eligible"):
                raise ValueError(
                    f"Client {cid} sample '{sid}' missing or invalid 'supervised_eligible' field!"
                )
            if s.get("supervised_eligible") != ref.get("supervised_eligible"):
                raise ValueError(
                    f"Tampered supervised_eligible for sample '{sid}' in client {cid}: shard has {s.get('supervised_eligible')!r}, train has {ref.get('supervised_eligible')!r}"
                )

            if sid in c_sample_ids:
                raise ValueError(f"Duplicate sample {sid} within client {cid}")
            c_sample_ids.add(sid)

            # Check cross-client group disjointness
            if gid in group_to_client and group_to_client[gid] != cid:
                raise ValueError(
                    f"Group-disjointness violation! Group '{gid}' assigned to both {group_to_client[gid]} and {cid}"
                )
            group_to_client[gid] = cid
            c_groups.add(gid)
            c_domains[s.get("source_domain", "plantvillage")] += 1
            c_classes[s["class_id"]] += 1

        # Check cross-client sample overlap
        intersection = union_ids.intersection(c_sample_ids)
        if intersection:
            raise ValueError(f"Client {cid} shares {len(intersection)} samples with other clients: {list(intersection)[:5]}")

        union_ids.update(c_sample_ids)
        client_metrics[cid] = {
            "n_k": len(samples),
            "num_groups": len(c_groups),
            "domains": dict(c_domains),
            "num_classes": len(c_classes),
        }

    # Verify union equals centralized train
    if union_ids != central_ids:
        missing_in_clients = central_ids - union_ids
        extra_in_clients = union_ids - central_ids
        raise ValueError(
            f"Union mismatch! Missing in clients: {len(missing_in_clients)}, Extra in clients: {len(extra_in_clients)}"
        )

    return {
        "status": "PASS",
        "num_clients": len(client_shards),
        "total_union_samples": len(union_ids),
        "total_groups": len(group_to_client),
        "client_metrics": client_metrics,
    }


def verify_partition_directory(
    release_dir: Union[str, Path],
    partition_scheme: str,
    client_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Verify partition directory against centralized train manifest.

    Loads shards from release_dir/partitions/{partition_scheme}/{cid}.jsonl
    and verifies union completeness, record equality, and group-disjointness against
    release_dir/manifests/train.jsonl.
    """
    rel = Path(release_dir)
    train_manifest = rel / "manifests/train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing centralized train manifest at {train_manifest}")

    train_samples = []
    with open(train_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                train_samples.append(json.loads(line))

    part_dir = rel / f"partitions/{partition_scheme}"
    if not part_dir.is_dir():
        raise FileNotFoundError(f"Missing partition directory at {part_dir}")

    available = sorted(path.stem for path in part_dir.glob("client_*.jsonl"))
    cids = list(client_ids) if client_ids is not None else available
    if not cids or len(cids) != len(set(cids)):
        raise ValueError("Client IDs must be nonempty and unique")
    if set(cids) != set(available):
        raise ValueError(
            f"Partition client IDs mismatch: expected {sorted(cids)}, found {available}"
        )
    shards: Dict[str, List[Dict[str, Any]]] = {}

    for cid in cids:
        shard_file = part_dir / f"{cid}.jsonl"
        if not shard_file.is_file():
            raise FileNotFoundError(f"Missing shard file for {cid} at {shard_file}")
        c_samples = []
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c_samples.append(json.loads(line))
        shards[cid] = c_samples

    result = verify_client_partitions(client_shards=shards, centralized_train_samples=train_samples)
    result["partition_scheme"] = partition_scheme
    result["total_samples"] = result["total_union_samples"]
    return result


def verify_partition_audit(
    release_dir: Union[str, Path], partition_scheme: str, client_ids: Sequence[str],
    release_manifest_sha256: str,
) -> Dict[str, Any]:
    """Check a versioned non-IID scenario before treating it as a valid experiment."""
    rel = Path(release_dir)
    spec_path = rel / "partition_spec_v4.json"
    if not spec_path.is_file():
        return {"status": "LEGACY_NO_AUDIT"}
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("release_manifest_sha256") != release_manifest_sha256:
        raise ValueError("Partition audit release hash differs from the selected release")
    if spec.get("num_clients") != len(client_ids):
        raise ValueError("Partition audit client count differs from training configuration")
    if partition_scheme == "iid":
        return {"status": "IID_VERIFIED_BY_UNION"}
    scenario = spec.get("scenarios", {}).get(partition_scheme)
    if not isinstance(scenario, dict):
        raise ValueError(f"Partition audit missing scenario {partition_scheme!r}")
    if scenario.get("status") != "PASS":
        raise ValueError(
            f"Partition scenario {partition_scheme!r} is {scenario.get('status')!r}; "
            "do not train or report it as a validated non-IID axis"
        )
    if (scenario.get("metadata", {}).get("skew_type") == "feature_skew_dirichlet"
            and float(scenario.get("mean_js_divergence", 1.0)) > 0.05):
        raise ValueError(
            f"MIXED_FEATURE_LABEL_SKEW: {partition_scheme} changes label priors "
            "too much to serve as an isolated feature-skew condition"
        )
    expected_hashes = scenario.get("shard_hashes", {})
    if set(expected_hashes) != {f"{cid}.jsonl" for cid in client_ids}:
        raise ValueError("Partition audit shard list differs from training configuration")
    for name, expected in expected_hashes.items():
        shard_path = rel / "partitions" / partition_scheme / name
        if hashlib.sha256(shard_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Partition audit shard SHA mismatch: {name}")
    return {"status": "PASS", "scenario": partition_scheme}
