"""Script đóng gói phát hành Stage 2 Scratch FL v4 (FedAvg-only).

Gói nén fl_package_v4.zip và manifest release_manifest_v4.json bao gồm:
- Toàn bộ mã nguồn FedAvg-only: stage2_scratch, stage2_matched, src, fl_training, tests
- Bộ phân hoạch sạch v3: data/partitions_stage2_scratch_v3 (từ repo hoặc training-data ngoài)
- Review và verification fixtures: data/duplicate_review.json, data/visually_verified_pairs.json, data/four_visually_verified_pairs.json
- Notebook chạy Kaggle v4: kaggle_stage2_scratch_v4.ipynb (hoặc notebooks/fedavg_stage2_scratch_v4.ipynb)
- Cấu hình & phụ thuộc: pyproject.toml, requirements-stage1.txt, requirements-stage2-matched.txt

Loại trừ:
- .venv, __pycache__, .git, .pytest_cache, logs, runs, weights/checkpoints (.pth, .pt)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import tempfile
import zipfile
import csv
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_training.package_verify import compute_file_sha256, verify_package_manifest

EXPECTED_CSV_COLUMNS = [
    "relative_path", "label", "class_name", "client_id", "split",
    "group_id", "leaf_group_id", "scenario", "alpha", "quantity_alpha",
    "feature_skew", "seed", "domain_id",
]

INCLUDE_DIRS = [
    "stage2_scratch",
    "stage2_matched",
    "stage1_compat",
    "src",
    "fl_training",
    "tests",
]

INCLUDE_FILES = [
    "data/duplicate_review.json",
    "data/visually_verified_pairs.json",
    "data/four_visually_verified_pairs.json",
    "requirements-stage1.txt",
    "requirements-stage2-matched.txt",
    "pyproject.toml",
]

EXCLUDE_PARTS = {
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".git",
    ".runtime",
    ".stage2_matched_runtime",
    ".stage2_scratch_runtime",
    "runs",
    "output",
}

EXCLUDE_FILES = {
    "stage2_matched/__main__.py",
    "stage2_matched/experiment.py",
    "fl_training/cli.py",
    "fl_training/baselines.py",
    "stage1_compat/dataset_identity.json",
    "stage1_compat/cli.py",
    "stage1_compat/__main__.py",
    "stage1_compat/package_stage1.py",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pt",
    ".pth",
    ".log",
}


def should_exclude(path: Path) -> bool:
    try:
        rel_posix = path.relative_to(ROOT).as_posix()
        if rel_posix in EXCLUDE_FILES:
            return True
    except ValueError:
        pass
    for part in path.parts:
        if part in EXCLUDE_PARTS:
            return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False

REQUIRED_CONDITIONS = (
    "feature01",
    "feature100",
    "label01",
    "label1",
    "label100",
    "label_quantity01",
    "quantity01",
    "quantity100",
)


def _canonical_digest(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_csv_rows(csv_path: Path) -> list[dict]:
    """Đọc dữ liệu CSV, xác thực 13 cột bắt buộc và ép kiểu int cho label, client_id, domain_id."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty required CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_CSV_COLUMNS:
            raise ValueError(
                f"Invalid CSV header in {csv_path}. Expected {EXPECTED_CSV_COLUMNS}, got {reader.fieldnames}"
            )
        rows = list(reader)
    for r in rows:
        for k in ("label", "client_id", "domain_id"):
            r[k] = int(r[k])
    return rows


def _semantic_rows_digest(rows: list[dict], with_domain: bool = False) -> str:
    """Tính semantic digest SHA-256 từ các bộ giá trị định danh mẫu."""
    keys = ("relative_path", "label", "class_name", "group_id") + (("domain_id",) if with_domain else ())
    tuples = [tuple(r[k] for k in keys) for r in sorted(rows, key=lambda x: x["relative_path"])]
    return _canonical_digest(tuples)


def _compute_condition_manifest_hash(cond_dir: Path) -> str:
    """Tính digest SHA-256 của tập file trong condition directory theo đúng chuẩn stage2_matched."""
    files = list(cond_dir.glob("*.csv")) + list((cond_dir / "clients").glob("*.csv"))
    files += [cond_dir / "partition_config.json", cond_dir / "image_content.json"]
    files = [p for p in sorted(files) if p.is_file()]
    entries = {p.relative_to(cond_dir).as_posix(): compute_file_sha256(p) for p in files}
    return _canonical_digest(entries)


def _validate_csv_file(csv_path: Path, expected_rows: int | None = None) -> int:
    """Xác thực định dạng CSV, 13 cột bắt buộc và số dòng dữ liệu."""
    rows = _read_csv_rows(csv_path)
    row_count = len(rows)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"Row count mismatch in {csv_path}: expected {expected_rows}, got {row_count}"
        )
    return row_count


def validate_suite_schema_and_contents(suite_dir: Path | str) -> dict:
    """Strictly validate schema and mandatory files of the partition suite before release packaging.
    
    Requirements F03, R2-03, R3-01 and R4-01: 100% aligned with runtime preflight.
    Mandatory complete schema: protocol must be 'stage2_scratch_clean_v3' (reject runner protocol 'stage2_scratch_fedavg_v4'),
    num_clients=5, split_seed=42, target_ratios=[0.72, 0.08, 0.20], sample counts matching 38984/4400/10921,
    38 class_names, 8 conditions matching CONDITIONS, 'common' semantic identity with 5 mandatory digests,
    'manifest_sha256' required for each condition (reject missing/empty/wrong type/hash mismatch),
    13 CSV columns, matching union of 38,984 samples across 5 clients with no duplicates, holdout domain_id == -1,
    image_content.json 54,305 valid SHA-256 hex hashes, verification fixtures, and bidirectional preflight cross-check.
    """
    s_dir = Path(suite_dir).resolve()
    if not s_dir.is_dir():
        raise FileNotFoundError(f"Suite path does not exist or is not a directory: {s_dir}")

    suite_json_file = s_dir / "suite.json"
    if not suite_json_file.is_file():
        raise FileNotFoundError(f"Missing suite.json in {s_dir}")

    try:
        suite_meta = json.loads(suite_json_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Malformed suite.json in {s_dir}: {e}")

    if not isinstance(suite_meta, dict):
        raise ValueError(f"suite.json must contain a JSON object, got {type(suite_meta)}")

    # Mandatory protocol is stage2_scratch_clean_v3 (R4-01: reject runner protocol)
    protocol = suite_meta.get("protocol")
    if protocol != "stage2_scratch_clean_v3":
        raise ValueError(f"Invalid suite protocol '{protocol}'. Expected 'stage2_scratch_clean_v3'")

    if suite_meta.get("num_clients") != 5:
        raise ValueError(f"Suite must specify num_clients=5, got {suite_meta.get('num_clients')}")

    if suite_meta.get("split_seed") != 42:
        raise ValueError(f"Suite must specify split_seed=42, got {suite_meta.get('split_seed')}")

    if suite_meta.get("target_ratios") != [0.72, 0.08, 0.20]:
        raise ValueError(f"Suite target_ratios must be [0.72, 0.08, 0.20], got {suite_meta.get('target_ratios')}")

    counts = suite_meta.get("counts")
    if not isinstance(counts, dict) or counts.get("train") != 38984 or counts.get("val") != 4400 or counts.get("test") != 10921:
        raise ValueError(f"suite.json counts must match exactly {{'train': 38984, 'val': 4400, 'test': 10921}}, got {counts}")

    if sum(counts.values()) != 54305:
        raise ValueError(f"Suite counts sum must be 54305, got {sum(counts.values())}")

    class_names = suite_meta.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != 38 or any(not isinstance(c, str) for c in class_names):
        raise ValueError("suite.json class_names must be a list of 38 class name strings")

    hex_pattern = re.compile(r"^[a-f0-9]{64}$")

    # Mandatory common semantic identity mapping (R4-01: reject if missing common or invalid digest)
    if "common" not in suite_meta or not isinstance(suite_meta["common"], dict):
        raise ValueError("suite.json missing required 'common' semantic identity mapping")
    common = suite_meta["common"]
    required_common_keys = ("train", "val", "test", "inventory", "classes")
    for k in required_common_keys:
        digest_val = common.get(k)
        if not isinstance(digest_val, str) or not hex_pattern.match(digest_val):
            raise ValueError(f"suite.json common missing or invalid required digest '{k}': got {digest_val!r}")

    conditions_meta = suite_meta.get("conditions")
    if not isinstance(conditions_meta, dict):
        raise ValueError("suite.json missing conditions mapping dictionary")

    if set(conditions_meta.keys()) != set(REQUIRED_CONDITIONS):
        raise ValueError(f"suite.json conditions mismatch: expected {sorted(REQUIRED_CONDITIONS)}, got {sorted(conditions_meta.keys())}")

    # Check verification fixtures
    dup_review = s_dir / "duplicate_review.json"
    vis_pairs = s_dir / "visually_verified_pairs.json"
    if not dup_review.is_file():
        raise FileNotFoundError(f"Missing duplicate_review.json in suite: {s_dir}")
    if not vis_pairs.is_file():
        raise FileNotFoundError(f"Missing visually_verified_pairs.json in suite: {s_dir}")

    try:
        dup_data = json.loads(dup_review.read_text(encoding="utf-8"))
        if not isinstance(dup_data, list) or len(dup_data) == 0:
            raise ValueError("duplicate_review.json must be a non-empty list")
    except Exception as e:
        raise ValueError(f"Malformed duplicate_review.json: {e}")

    try:
        vis_data = json.loads(vis_pairs.read_text(encoding="utf-8"))
        if not isinstance(vis_data, list) or len(vis_data) < 6:
            raise ValueError("visually_verified_pairs.json must contain at least 6 verified pairs")
    except Exception as e:
        raise ValueError(f"Malformed visually_verified_pairs.json: {e}")

    # Verify hashes of verification fixtures if pinned in suite.json
    if "duplicate_review_sha256" in suite_meta:
        dup_sha = hashlib.sha256(dup_review.read_bytes()).hexdigest()
        if dup_sha != suite_meta["duplicate_review_sha256"]:
            raise ValueError(f"duplicate_review.json hash mismatch: expected {suite_meta['duplicate_review_sha256']}, got {dup_sha}")

    if "visually_verified_pairs_sha256" in suite_meta:
        vis_sha = hashlib.sha256(vis_pairs.read_bytes()).hexdigest()
        if vis_sha != suite_meta["visually_verified_pairs_sha256"]:
            raise ValueError(f"visually_verified_pairs.json hash mismatch: expected {suite_meta['visually_verified_pairs_sha256']}, got {vis_sha}")

    if "leaf_map_sha256" in suite_meta:
        leaf_file = s_dir / "leaf_map.json"
        if leaf_file.is_file():
            leaf_sha = hashlib.sha256(leaf_file.read_bytes()).hexdigest()
            if leaf_sha != suite_meta["leaf_map_sha256"]:
                raise ValueError(f"leaf_map.json hash mismatch: expected {suite_meta['leaf_map_sha256']}, got {leaf_sha}")

    from stage2_matched.data import CONDITIONS as RUNTIME_CONDITIONS
    from src.data.transforms import create_deterministic_client_profile

    # Validate each of the 8 non-IID conditions
    for cond in REQUIRED_CONDITIONS:
        cond_dir = s_dir / cond
        if not cond_dir.is_dir():
            raise FileNotFoundError(f"Missing required condition directory: {cond_dir}")

        c_meta = conditions_meta[cond]
        if not isinstance(c_meta, dict):
            raise ValueError(f"Condition '{cond}' entry in suite.json must be a dict")

        # R4-01: manifest_sha256 is strictly mandatory and must be valid 64-char hex
        expected_manifest_sha = c_meta.get("manifest_sha256")
        if not isinstance(expected_manifest_sha, str) or not hex_pattern.match(expected_manifest_sha):
            raise ValueError(f"Condition '{cond}' missing or invalid required 'manifest_sha256' (got {expected_manifest_sha!r})")

        # R4-01: Verify condition metadata against RUNTIME_CONDITIONS
        exp_cond = RUNTIME_CONDITIONS[cond]
        for k, v in exp_cond.items():
            if c_meta.get(k) != v:
                raise ValueError(f"Condition '{cond}' metadata mismatch for '{k}': expected {v}, got {c_meta.get(k)}")

        cfg_file = cond_dir / "partition_config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Missing partition_config.json in {cond_dir}")

        try:
            cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if not isinstance(cfg_data, dict):
                raise ValueError("partition_config.json must contain a JSON object")
            if cfg_data.get("condition") != cond:
                raise ValueError(f"Condition mismatch in {cfg_file}: expected '{cond}', got '{cfg_data.get('condition')}'")
            if cfg_data.get("num_clients") != 5:
                raise ValueError(f"num_clients mismatch in {cfg_file}")
            if not cfg_data.get("content_aware") or not cfg_data.get("group_aware"):
                raise ValueError(f"partition_config.json in {cond_dir} must have content_aware=True and group_aware=True")
            if cfg_data.get("class_names") != class_names:
                raise ValueError(f"class_names mismatch in {cfg_file}")
            for k, v in exp_cond.items():
                if cfg_data.get(k) != v:
                    raise ValueError(f"partition_config.json in {cond_dir} mismatch for '{k}': expected {v}, got {cfg_data.get(k)}")
        except Exception as e:
            raise ValueError(f"Malformed partition_config.json in {cond_dir}: {e}")

        # 1. Validate 13 columns and exact row counts of holdout CSVs
        central_rows = _read_csv_rows(cond_dir / "centralized_train.csv")
        val_rows = _read_csv_rows(cond_dir / "global_val.csv")
        test_rows = _read_csv_rows(cond_dir / "global_test.csv")

        if len(central_rows) != 38984:
            raise ValueError(f"Row count mismatch in {cond_dir / 'centralized_train.csv'}: expected 38984, got {len(central_rows)}")
        if len(val_rows) != 4400:
            raise ValueError(f"Row count mismatch in {cond_dir / 'global_val.csv'}: expected 4400, got {len(val_rows)}")
        if len(test_rows) != 10921:
            raise ValueError(f"Row count mismatch in {cond_dir / 'global_test.csv'}: expected 10921, got {len(test_rows)}")

        # Holdout samples must retain domain_id == -1
        if any(r["domain_id"] != -1 for r in val_rows + test_rows):
            raise ValueError(f"Holdout samples must have domain_id == -1 in condition '{cond}'")

        clients_dir = cond_dir / "clients"
        if not clients_dir.is_dir():
            raise FileNotFoundError(f"Missing clients directory in {cond_dir}")

        n_k = c_meta.get("n_k")
        if not isinstance(n_k, list) or len(n_k) != 5 or any(not isinstance(x, int) or x <= 0 for x in n_k) or sum(n_k) != 38984:
            raise ValueError(f"Condition '{cond}' has invalid n_k in suite.json: {n_k}")
        if cfg_data.get("n_k") != n_k:
            raise ValueError(f"Condition '{cond}' n_k in partition_config.json mismatch with suite.json: {cfg_data.get('n_k')} vs {n_k}")

        client_total = 0
        client_rows_map = {}
        for cid in range(5):
            client_csv = clients_dir / f"client_{cid:02d}.csv"
            c_rows = _read_csv_rows(client_csv)
            if len(c_rows) != n_k[cid]:
                raise ValueError(f"Row count mismatch in {client_csv}: expected {n_k[cid]}, got {len(c_rows)}")
            client_total += len(c_rows)
            client_rows_map[cid] = c_rows

        if client_total != 38984:
            raise ValueError(f"Sum of client samples in {cond_dir} is {client_total}, expected 38984")

        # 2. Validate image_content.json: must contain exactly 54,305 images with valid SHA-256 hex
        img_content = cond_dir / "image_content.json"
        if not img_content.is_file():
            raise FileNotFoundError(f"Missing image_content.json in {cond_dir}")
        try:
            img_data = json.loads(img_content.read_text(encoding="utf-8"))
            if not isinstance(img_data, dict) or len(img_data) != 54305:
                raise ValueError(
                    f"image_content.json in {cond_dir} must contain exactly 54,305 entries, "
                    f"got {len(img_data) if isinstance(img_data, dict) else type(img_data)}"
                )
        except Exception as e:
            raise ValueError(f"Malformed image_content.json in {cond_dir}: {e}")

        for rel_p, img_hash in img_data.items():
            if not isinstance(img_hash, str) or not hex_pattern.match(img_hash):
                raise ValueError(
                    f"Invalid SHA-256 hash in image_content.json in {cond_dir}: '{img_hash}' for image '{rel_p}'"
                )

        # 3. R3-01 and R4-01: Validate condition directory manifest SHA-256 against pinned expected hash
        computed_manifest_sha = _compute_condition_manifest_hash(cond_dir)
        if computed_manifest_sha != expected_manifest_sha:
            raise ValueError(
                f"Condition '{cond}' manifest_sha256 mismatch: expected {expected_manifest_sha}, got {computed_manifest_sha}"
            )

        # 4. R3-01 and R4-01: Validate semantic identity against mandatory suite_meta['common']
        if _semantic_rows_digest(central_rows) != common.get("train"):
            raise ValueError(f"Semantic rows mismatch for centralized_train in condition '{cond}'")
        if _semantic_rows_digest(val_rows) != common.get("val"):
            raise ValueError(f"Semantic rows mismatch for global_val in condition '{cond}'")
        if _semantic_rows_digest(test_rows) != common.get("test"):
            raise ValueError(f"Semantic rows mismatch for global_test in condition '{cond}'")
        if _canonical_digest(img_data) != common.get("inventory"):
            raise ValueError(f"Inventory digest mismatch in condition '{cond}'")
        if _canonical_digest(cfg_data.get("class_names")) != common.get("classes"):
            raise ValueError(f"Class names digest mismatch in condition '{cond}'")

        # Validate client partition union and integrity against centralized_train
        by_path = {r["relative_path"]: r for r in central_rows}
        client_paths = set()
        for cid in range(5):
            for cr in client_rows_map[cid]:
                rp = cr["relative_path"]
                if rp in client_paths:
                    raise ValueError(f"Duplicate sample '{rp}' assigned across multiple clients in {cond}")
                client_paths.add(rp)
                if rp not in by_path:
                    raise ValueError(f"Client sample '{rp}' not found in centralized_train in {cond}")
                if cr != by_path[rp]:
                    raise ValueError(
                        f"Client sample '{rp}' data mismatch with centralized_train in {cond} "
                        f"(e.g., label/domain/group inconsistency)"
                    )

        if len(client_paths) != 38984:
            raise ValueError(f"Client partition union size mismatch in {cond_dir}: got {len(client_paths)}, expected 38984")

        # Validate all partition samples exist in image_content inventory
        for r in central_rows + val_rows + test_rows:
            if r["relative_path"] not in img_data:
                raise ValueError(f"Sample '{r['relative_path']}' in {cond} is missing from image_content.json inventory")

        # Check feature dirichlet vs non-feature scenarios
        if exp_cond["scenario"] == "feature_dirichlet":
            if any(not 0 <= int(r["domain_id"]) < 5 for r in central_rows):
                raise ValueError(f"Missing/invalid domain ID in centralized_train for {cond}")
            profiles = [create_deterministic_client_profile(d, 42, "moderate").to_dict() for d in range(5)]
            if cfg_data.get("domain_profiles") != profiles:
                raise ValueError(f"Feature strength differs from preregistered moderate profiles in {cond}")
        else:
            if any(r["domain_id"] != -1 for r in central_rows):
                raise ValueError(f"Unexpected feature transform domain_id in centralized_train for {cond}")
            if cfg_data.get("domain_profiles"):
                raise ValueError(f"Unexpected domain_profiles in partition_config for {cond}")

    # R4-01: Call offline preflight to ensure 100% bidirectional runtime compatibility
    from stage2_matched.data import preflight
    preflight(s_dir, dataset_root=None)

    return suite_meta


def build_release_v4(
    suite_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    notebook_path: Path | str | None = None,
) -> dict:
    out_dir = Path(output_dir).resolve() if output_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    output_zip = out_dir / "fl_package_v4.zip"
    output_manifest = out_dir / "release_manifest_v4.json"

    # Resolve suite directory
    if suite_dir is not None:
        resolved_suite = Path(suite_dir).resolve()
    elif (ROOT / "data/partitions_stage2_scratch_v3").is_dir():
        resolved_suite = ROOT / "data/partitions_stage2_scratch_v3"
    else:
        candidates = [
            ROOT.parents[1] / "training-data/stage2/partitions_stage2_scratch_v3",
            ROOT.parent / "training-data/stage2/partitions_stage2_scratch_v3",
            Path("training-data/stage2/partitions_stage2_scratch_v3").resolve(),
        ]
        resolved_suite = None
        for cand in candidates:
            if cand.is_dir():
                resolved_suite = cand
                break
        if resolved_suite is None:
            raise FileNotFoundError(
                "partitions_stage2_scratch_v3 not found. Provide --suite <path>."
            )

    # Validate suite schema and mandatory files strictly before packaging (F03)
    print(f"[0/4] Validating partition suite schema and completeness: {resolved_suite}...")
    validate_suite_schema_and_contents(resolved_suite)
    print(f"  Suite validation passed: all 8 conditions, fixtures, and manifests verified.")

    # Resolve notebook
    if notebook_path is not None:
        resolved_nb = Path(notebook_path).resolve()
    elif (ROOT / "notebooks/fedavg_stage2_scratch_v4.ipynb").is_file():
        resolved_nb = ROOT / "notebooks/fedavg_stage2_scratch_v4.ipynb"
    elif (ROOT / "kaggle_stage2_scratch_v4.ipynb").is_file():
        resolved_nb = ROOT / "kaggle_stage2_scratch_v4.ipynb"
    else:
        raise FileNotFoundError("Canonical notebook not found. Provide --notebook <path>.")

    print(f"[1/4] Scanning files to package from {ROOT}...")
    print(f"  External suite: {resolved_suite}")
    print(f"  Notebook: {resolved_nb}")
    print(f"  Target output: {out_dir}")

    # Map of relative path in package -> source Path on disk
    package_map: dict[str, Path] = {}

    for d in INCLUDE_DIRS:
        dir_path = ROOT / d
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
        for p in dir_path.rglob("*"):
            if p.is_file() and not should_exclude(p):
                rel = p.relative_to(ROOT).as_posix()
                package_map[rel] = p

    for f in INCLUDE_FILES:
        file_path = ROOT / f
        if not file_path.is_file():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        rel = file_path.relative_to(ROOT).as_posix()
        package_map[rel] = file_path

    # Add notebook mapped to kaggle_stage2_scratch_v4.ipynb
    package_map["kaggle_stage2_scratch_v4.ipynb"] = resolved_nb

    # Add suite files mapped to data/partitions_stage2_scratch_v3/...
    suite_files = [p for p in resolved_suite.rglob("*") if p.is_file() and not should_exclude(p)]
    if len(suite_files) == 0:
        raise FileNotFoundError(f"No files found in suite directory: {resolved_suite}")
    for p in suite_files:
        rel = f"data/partitions_stage2_scratch_v3/{p.relative_to(resolved_suite).as_posix()}"
        package_map[rel] = p

    sorted_entries = sorted(package_map.items(), key=lambda item: item[0])
    print(f"  Found {len(sorted_entries)} valid files to package.")

    # Generate file-level manifest
    print("[2/4] Computing SHA256 hashes for manifest...")
    manifest_entries = {}
    total_uncompressed = 0
    for rel, p in sorted_entries:
        size = p.stat().st_size
        sha = compute_file_sha256(p)
        manifest_entries[rel] = {
            "size_bytes": size,
            "sha256": sha,
        }
        total_uncompressed += size

    manifest_data = {
        "protocol": "stage2_scratch_fedavg_v4",
        "scope": "fedavg_only",
        "total_files": len(sorted_entries),
        "uncompressed_bytes": total_uncompressed,
        "files": manifest_entries,
    }

    output_manifest.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Manifest written to {output_manifest}")

    # Build zip archive
    print(f"[3/4] Creating zip package {output_zip}...")
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include manifest inside the zip
        zf.write(output_manifest, "release_manifest_v4.json")
        for rel, p in sorted_entries:
            zf.write(p, rel)

    zip_size = output_zip.stat().st_size
    zip_sha = compute_file_sha256(output_zip)

    # Verification: extract to clean temp directory
    print("[4/4] Verifying zip integrity in clean temp directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(output_zip, "r") as zf:
            zf.extractall(temp_path)

        verify_res = verify_package_manifest(temp_path, temp_path / "release_manifest_v4.json")
        print(f"  Verified {verify_res['verified_files']} files with manifest {verify_res['manifest_sha256'][:16]}...")

    summary = {
        "output_zip": str(output_zip),
        "output_manifest": str(output_manifest),
        "file_count": len(sorted_entries),
        "uncompressed_bytes": total_uncompressed,
        "uncompressed_mb": round(total_uncompressed / (1024 * 1024), 2),
        "zip_bytes": zip_size,
        "zip_mb": round(zip_size / (1024 * 1024), 2),
        "zip_sha256": zip_sha,
        "verification": "PASSED",
    }
    print("\n--- PACKAGE RELEASE V4 SUCCESSFUL ---")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Package Stage 2 Scratch FedAvg release")
    parser.add_argument("--suite", default=None, help="Path to partitions_stage2_scratch_v3 suite")
    parser.add_argument("--output-dir", default=None, help="Directory to save release zip and manifest")
    parser.add_argument("--notebook", default=None, help="Path to canonical notebook")
    args = parser.parse_args()

    build_release_v4(
        suite_dir=args.suite,
        output_dir=args.output_dir,
        notebook_path=args.notebook,
    )


if __name__ == "__main__":
    main()
