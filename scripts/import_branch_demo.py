"""Import the three pinned, published validation runs for the web demo.

The mixed PV/PD image release is not published in these branches. This importer
checks the published bytes and shared protocol; it does not reproduce metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "centralized": ("train-centralized-v6", "2294de21c93e28d611013a75413995a128b303b7"),
    "fedavg": ("train-fedavg-non-iid-v7", "b37488fbaf99d94cc0d66e66a9649c99dd9e1257"),
    "local": ("train-local-only-a01-v7", "6de4c6b99805399673dc0a41ab97aeefd7099943"),
}
RELEASE_SHA = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"
VALIDATION_SHA = "5094123650efd6d47e7922e0111401ad8bc60bd6b20ee39d619d4105f9bd9f6f"
W0 = "eaf9197b42fb39163186a7285f7f1ed4f1e728bd83522260fdc5f0f51c110850"
CONTRACT_FILES = (
    "plant_data_contract/plant_data_contract/taxonomy.py",
    "plant_data_contract/plant_data_contract/transforms.py",
    "plant_data_contract/plant_data_contract/models.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classes_from_taxonomy(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PLANTVILLAGE_38_CLASSES":
                classes = ast.literal_eval(node.value)
                if len(classes) != 38 or len(set(classes)) != 38:
                    raise ValueError("Taxonomy must contain 38 distinct classes")
                return classes
    raise ValueError("PLANTVILLAGE_38_CLASSES was not found")


def verify_branch(source: Path, name: str, branch: str, commit: str) -> tuple[dict, dict]:
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source branch: {source}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != commit:
        raise ValueError(f"{branch}: commit {actual_commit} differs from pinned {commit}")
    source_manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if source_manifest["branch"] != branch or source_manifest["release_manifest_sha256"] != RELEASE_SHA:
        raise ValueError(f"{branch}: source manifest identity mismatch")
    for relative in CONTRACT_FILES:
        if sha256(source / relative) != source_manifest["files"].get(relative):
            raise ValueError(f"{branch}: contract file hash mismatch: {relative}")

    result_dir = source / "results" / "20261003"
    manifest = json.loads((result_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    if (manifest["branch"] != branch or manifest["release_manifest_sha256"] != RELEASE_SHA
            or manifest["validation_manifest_sha256"] != VALIDATION_SHA):
        raise ValueError(f"{branch}: result manifest identity mismatch")
    for relative, expected in manifest["files"].items():
        file = result_dir / relative
        if not file.is_file() or file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
            raise ValueError(f"{branch}: missing or corrupted result: {relative}")

    summary_name = {
        "centralized": "summary.json",
        "fedavg": "validation_metrics.json",
        "local": "combined_summary.json",
    }[name]
    summary = result_dir / summary_name
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("release_manifest_sha256") != RELEASE_SHA:
        raise ValueError(f"{branch}: summary release mismatch")
    if name in {"centralized", "local"} and (
        payload.get("seed") != 42 or payload.get("w0_fingerprint") != W0
    ):
        raise ValueError(f"{branch}: seed or W0 mismatch")
    if name == "local" and payload.get("partition_scheme") != "label_alpha_0_1":
        raise ValueError(f"{branch}: partition protocol mismatch")
    if name == "fedavg":
        validation = payload.get("validation", {})
        if (payload.get("checkpoint_sha256") != manifest["files"]["checkpoint_best.pt"]["sha256"]
                or validation.get("manifest_sha256") != VALIDATION_SHA
                or validation.get("by_source", {}).get("plantdoc", {}).get("count") != 269
                or validation.get("by_source", {}).get("plantvillage", {}).get("count") != 4344):
            raise ValueError(f"{branch}: checkpoint or validation split mismatch")
    return manifest, source_manifest


def verify_comparison(path: Path, manifests: dict[str, dict]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 14:
        raise ValueError("Expected seven checkpoints × two validation sources")
    expected_models = {"Centralized v6", "FedAvg v7", *(f"Local {i:02d}" for i in range(5))}
    if {r["model"] for r in rows} != expected_models:
        raise ValueError("Unexpected model set in validation comparison")
    checkpoint_shas = {
        "Centralized v6": manifests["centralized"]["files"]["checkpoint_best.pt"]["sha256"],
        "FedAvg v7": manifests["fedavg"]["files"]["checkpoint_best.pt"]["sha256"],
        **{f"Local {i:02d}": manifests["local"]["files"][f"client_{i:02d}/checkpoint_best.pt"]["sha256"]
           for i in range(5)},
    }
    seen = set()
    for row in rows:
        key = (row["model"], row["source"])
        if key in seen or row["source"] not in {"plantdoc", "plantvillage"}:
            raise ValueError("Duplicate or unknown comparison row")
        seen.add(key)
        if int(row["count"]) != (269 if row["source"] == "plantdoc" else 4344):
            raise ValueError(f"Unexpected validation count: {key}")
        if (row["checkpoint_sha256"] != checkpoint_shas[row["model"]]
                or row["release_manifest_sha256"] != RELEASE_SHA
                or row["validation_manifest_sha256"] != VALIDATION_SHA):
            raise ValueError(f"Mismatched checkpoint or split: {key}")
        for metric in ("top1_accuracy", "macro_f1_supported", "crop_accuracy"):
            if not 0.0 <= float(row[metric]) <= 1.0:
                raise ValueError(f"Invalid {metric}: {key}")


def import_results(sources_root: Path, target: Path) -> Path:
    if target.exists():
        raise FileExistsError(f"Target exists; keep published import immutable: {target}")
    manifests: dict[str, dict] = {}
    source_manifests: dict[str, dict] = {}
    for name, (branch, commit) in SOURCES.items():
        manifests[name], source_manifests[name] = verify_branch(sources_root / name, name, branch, commit)
    comparison = sources_root / "fedavg/results/20261003/validation_comparison.csv"
    if len({m["files"]["validation_comparison.csv"]["sha256"] for m in manifests.values()}) != 1:
        raise ValueError("Branches do not publish the same comparison file")
    verify_comparison(comparison, manifests)
    contract_hashes = {relative: source_manifests["fedavg"]["files"][relative] for relative in CONTRACT_FILES}
    if any(sm["files"][relative] != expected for sm in source_manifests.values()
           for relative, expected in contract_hashes.items()):
        raise ValueError("Training branches use different model, taxonomy or transform code")
    classes = classes_from_taxonomy(sources_root / "fedavg" / CONTRACT_FILES[0])

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="branch_demo_", dir=target.parent) as staging_name:
        staging = Path(staging_name)
        for name in SOURCES:
            shutil.copytree(sources_root / name / "results" / "20261003", staging / name)
        (staging / "contract").mkdir()
        for relative in CONTRACT_FILES:
            shutil.copy2(sources_root / "fedavg" / relative, staging / "contract" / Path(relative).name)
        metadata = {
            "protocol_id": "mixed_pv_pd_v3_pretrained_seed42_validation",
            "evidence_level": "published_artifact_hash_verified_not_locally_reevaluated",
            "release_manifest_sha256": RELEASE_SHA,
            "validation_manifest_sha256": VALIDATION_SHA,
            "w0_fingerprint": W0,
            "seed": 42,
            "partition_scheme": "label_alpha_0_1",
            "num_classes": 38,
            "validation_counts": {"plantdoc": 269, "plantvillage": 4344},
            "transform": "canonical_v1",
            "contract_hashes": contract_hashes,
            "classes": classes,
            "sources": {name: {"branch": branch, "commit": commit} for name, (branch, commit) in SOURCES.items()},
        }
        (staging / "import_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, default=ROOT / ".tmp-agent/branch-demo-import/sources")
    parser.add_argument("--target", type=Path, default=ROOT / "artifacts/branch_demo/20261003")
    args = parser.parse_args()
    print(import_results(args.sources_root, args.target))


if __name__ == "__main__":
    main()
