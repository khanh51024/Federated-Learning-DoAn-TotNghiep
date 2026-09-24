"""Unit tests for package_release_v4 negative cases and schema validation (F03, R2-03, R3-01, R4-01)."""
import hashlib
import json
import re
import shutil
import pytest
from pathlib import Path

from scripts.package_release_v4 import (
    validate_suite_schema_and_contents,
    build_release_v4,
    _compute_condition_manifest_hash,
    _semantic_rows_digest,
    _canonical_digest,
    REQUIRED_CONDITIONS,
    EXPECTED_CSV_COLUMNS,
)
from stage2_matched.data import CONDITIONS
from src.data.transforms import create_deterministic_client_profile


def generate_synthetic_valid_suite(s_dir: Path) -> Path:
    """Generate a fully valid synthetic partition suite with 54,305 images, 38 classes,
    8 conditions, metadata, hashes, and semantic digests, achieving 100% pass on
    validate_suite_schema_and_contents and runtime preflight.
    """
    s_dir.mkdir(parents=True, exist_ok=True)
    
    # 6 visual pairs in client 0 (first 12 images)
    vis_pairs = [{"a": f"class_0/{2*i:06d}.JPG", "b": f"class_0/{2*i+1:06d}.JPG"} for i in range(6)]
    dup_review = [{"a": "class_0/000000.JPG", "b": "class_0/000001.JPG", "decision": "accepted"}]
    
    dup_content = json.dumps(dup_review).encode("utf-8")
    vis_content = json.dumps(vis_pairs).encode("utf-8")
    (s_dir / "duplicate_review.json").write_bytes(dup_content)
    (s_dir / "visually_verified_pairs.json").write_bytes(vis_content)

    counts = {"train": 38984, "val": 4400, "test": 10921}
    class_names = [f"class_{i}" for i in range(38)]
    
    # 54,305 unique image paths and hashes across 38 classes
    img_data = {}
    for i in range(12):
        img_data[f"class_0/{i:06d}.JPG"] = hashlib.sha256(f"img_{i}".encode()).hexdigest()
    for i in range(12, 54305):
        img_data[f"class_{i%38}/{i:06d}.JPG"] = hashlib.sha256(f"img_{i}".encode()).hexdigest()
    img_json_bytes = json.dumps(img_data).encode("utf-8")

    header_line = ",".join(EXPECTED_CSV_COLUMNS) + "\n"
    
    # Client slice ranges for n_k = [7796, 7797, 7797, 7797, 7797]
    n_k = [7796, 7797, 7797, 7797, 7797]
    client_ranges = []
    curr = 0
    for k in n_k:
        client_ranges.append((curr, curr + k))
        curr += k

    conditions_meta = {}
    
    # Precompute CSV text lines to make generation very fast
    val_lines = [header_line]
    val_rows_list = []
    for i in range(4400):
        idx = 38984 + i
        lbl = idx % 38
        cname = f"class_{lbl}"
        p = f"{cname}/{idx:06d}.JPG"
        val_lines.append(f"{p},{lbl},{cname},-1,val,p::{p},l::{p},none,0.1,0.1,none,42,-1\n")
        val_rows_list.append({
            "relative_path": p, "label": lbl, "class_name": cname,
            "client_id": -1, "split": "val", "group_id": f"p::{p}",
            "leaf_group_id": f"l::{p}", "scenario": "none", "alpha": 0.1,
            "quantity_alpha": 0.1, "feature_skew": "none", "seed": 42,
            "domain_id": -1
        })
    val_bytes = "".join(val_lines).encode("utf-8")

    test_lines = [header_line]
    test_rows_list = []
    for i in range(10921):
        idx = 38984 + 4400 + i
        lbl = idx % 38
        cname = f"class_{lbl}"
        p = f"{cname}/{idx:06d}.JPG"
        test_lines.append(f"{p},{lbl},{cname},-1,test,p::{p},l::{p},none,0.1,0.1,none,42,-1\n")
        test_rows_list.append({
            "relative_path": p, "label": lbl, "class_name": cname,
            "client_id": -1, "split": "test", "group_id": f"p::{p}",
            "leaf_group_id": f"l::{p}", "scenario": "none", "alpha": 0.1,
            "quantity_alpha": 0.1, "feature_skew": "none", "seed": 42,
            "domain_id": -1
        })
    test_bytes = "".join(test_lines).encode("utf-8")

    # Non-feature central rows and bytes
    nf_central_lines = [header_line]
    nf_central_rows = []
    for cid, (st, ed) in enumerate(client_ranges):
        for idx in range(st, ed):
            lbl = 0 if idx < 12 else (idx % 38)
            cname = f"class_{lbl}"
            p = f"{cname}/{idx:06d}.JPG"
            gid = f"p::pair_{idx//2}" if idx < 12 else f"p::{p}"
            lgid = f"l::pair_{idx//2}" if idx < 12 else f"l::{p}"
            nf_central_lines.append(f"{p},{lbl},{cname},{cid},train,{gid},{lgid},none,0.1,0.1,none,42,-1\n")
            nf_central_rows.append({
                "relative_path": p, "label": lbl, "class_name": cname,
                "client_id": cid, "split": "train", "group_id": gid,
                "leaf_group_id": lgid, "scenario": "none", "alpha": 0.1,
                "quantity_alpha": 0.1, "feature_skew": "none", "seed": 42,
                "domain_id": -1
            })
    nf_central_bytes = "".join(nf_central_lines).encode("utf-8")

    # Feature central rows and bytes
    f_central_lines = [header_line]
    f_central_rows = []
    for cid, (st, ed) in enumerate(client_ranges):
        for idx in range(st, ed):
            lbl = 0 if idx < 12 else (idx % 38)
            cname = f"class_{lbl}"
            p = f"{cname}/{idx:06d}.JPG"
            gid = f"p::pair_{idx//2}" if idx < 12 else f"p::{p}"
            lgid = f"l::pair_{idx//2}" if idx < 12 else f"l::{p}"
            f_central_lines.append(f"{p},{lbl},{cname},{cid},train,{gid},{lgid},feature_dirichlet,0.1,0.1,none,42,{cid}\n")
            f_central_rows.append({
                "relative_path": p, "label": lbl, "class_name": cname,
                "client_id": cid, "split": "train", "group_id": gid,
                "leaf_group_id": lgid, "scenario": "feature_dirichlet", "alpha": 0.1,
                "quantity_alpha": 0.1, "feature_skew": "none", "seed": 42,
                "domain_id": cid
            })
    f_central_bytes = "".join(f_central_lines).encode("utf-8")

    # Create condition files
    for cond in REQUIRED_CONDITIONS:
        c_dir = s_dir / cond
        c_dir.mkdir(parents=True, exist_ok=True)
        (c_dir / "image_content.json").write_bytes(img_json_bytes)
        (c_dir / "global_val.csv").write_bytes(val_bytes)
        (c_dir / "global_test.csv").write_bytes(test_bytes)

        is_feature = (CONDITIONS[cond]["scenario"] == "feature_dirichlet")
        if is_feature:
            (c_dir / "centralized_train.csv").write_bytes(f_central_bytes)
            use_lines = f_central_lines
        else:
            (c_dir / "centralized_train.csv").write_bytes(nf_central_bytes)
            use_lines = nf_central_lines

        clients_dir = c_dir / "clients"
        clients_dir.mkdir(parents=True, exist_ok=True)
        for cid, (st, ed) in enumerate(client_ranges):
            c_bytes = (header_line + "".join(use_lines[1 + st: 1 + ed])).encode("utf-8")
            (clients_dir / f"client_{cid:02d}.csv").write_bytes(c_bytes)

        cfg = {
            "condition": cond,
            "num_clients": 5,
            "content_aware": True,
            "group_aware": True,
            "class_names": class_names,
            "n_k": n_k,
        }
        cfg.update(CONDITIONS[cond])
        if is_feature:
            cfg["domain_profiles"] = [create_deterministic_client_profile(d, 42, "moderate").to_dict() for d in range(5)]
        else:
            cfg["domain_profiles"] = []
        (c_dir / "partition_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        cond_sha = _compute_condition_manifest_hash(c_dir)
        cond_entry = dict(CONDITIONS[cond])
        cond_entry["n_k"] = n_k
        cond_entry["manifest_sha256"] = cond_sha
        conditions_meta[cond] = cond_entry

    common = {
        "train": _semantic_rows_digest(nf_central_rows),
        "val": _semantic_rows_digest(val_rows_list),
        "test": _semantic_rows_digest(test_rows_list),
        "inventory": _canonical_digest(img_data),
        "classes": _canonical_digest(class_names),
    }

    suite_meta = {
        "protocol": "stage2_scratch_clean_v3",
        "smoke": False,
        "num_clients": 5,
        "split_seed": 42,
        "target_ratios": [0.72, 0.08, 0.20],
        "counts": counts,
        "class_names": class_names,
        "common": common,
        "conditions": conditions_meta,
        "duplicate_review_sha256": hashlib.sha256(dup_content).hexdigest(),
        "visually_verified_pairs_sha256": hashlib.sha256(vis_content).hexdigest(),
    }
    (s_dir / "suite.json").write_text(json.dumps(suite_meta), encoding="utf-8")
    return s_dir


@pytest.fixture(scope="module")
def base_valid_suite(tmp_path_factory):
    """Generate valid synthetic suite once for the entire test module."""
    base_dir = tmp_path_factory.mktemp("base_valid_suite") / "suite"
    return generate_synthetic_valid_suite(base_dir)


@pytest.fixture
def schema_valid_suite(tmp_path, base_valid_suite):
    """Fixture for rapid copying from base suite allowing each test to tamper independently."""
    test_suite = tmp_path / "valid_suite"
    shutil.copytree(base_valid_suite, test_suite)
    return test_suite


def test_validate_suite_valid(schema_valid_suite):
    meta = validate_suite_schema_and_contents(schema_valid_suite)
    assert meta["protocol"] == "stage2_scratch_clean_v3"
    assert meta["num_clients"] == 5


def test_validate_suite_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty_suite"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing suite.json"):
        validate_suite_schema_and_contents(empty_dir)


def test_validate_suite_malformed_json(tmp_path):
    s_dir = tmp_path / "bad_json"
    s_dir.mkdir()
    (s_dir / "suite.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed suite.json"):
        validate_suite_schema_and_contents(s_dir)


def test_validate_suite_invalid_protocol(tmp_path):
    s_dir = tmp_path / "bad_proto"
    s_dir.mkdir()
    (s_dir / "suite.json").write_text(json.dumps({"protocol": "old_protocol_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid suite protocol"):
        validate_suite_schema_and_contents(s_dir)


def test_validate_suite_rejects_runner_protocol(schema_valid_suite):
    """R4-01: Builder must reject suite specifying runner protocol (stage2_scratch_fedavg_v4)."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    meta["protocol"] = "stage2_scratch_fedavg_v4"
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid suite protocol 'stage2_scratch_fedavg_v4'"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_manifest_sha256(schema_valid_suite):
    """R4-01: Reproduce Codex probe: deleting manifest_sha256 in feature01 must be rejected."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    del meta["conditions"]["feature01"]["manifest_sha256"]
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid required 'manifest_sha256'"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_empty_manifest_sha256(schema_valid_suite):
    """R4-01: Empty manifest_sha256 must be rejected."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    meta["conditions"]["feature01"]["manifest_sha256"] = ""
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or invalid required 'manifest_sha256'"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_mismatched_manifest_sha256(schema_valid_suite):
    """R4-01: Mismatched condition manifest_sha256 must be rejected."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    meta["conditions"]["feature01"]["manifest_sha256"] = "a" * 64
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_common(schema_valid_suite):
    """R4-01: Suite missing common semantic identity mapping must be rejected."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    del meta["common"]
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required 'common'"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_common_key(schema_valid_suite):
    """R4-01: Suite common missing inventory digest field must be rejected."""
    s_json = schema_valid_suite / "suite.json"
    meta = json.loads(s_json.read_text(encoding="utf-8"))
    del meta["common"]["inventory"]
    s_json.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="common missing or invalid required digest 'inventory'"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_condition_directory(schema_valid_suite):
    shutil.rmtree(schema_valid_suite / "label100")
    with pytest.raises(FileNotFoundError, match="Missing required condition directory"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_client_csv(schema_valid_suite):
    """R2-04: Test with real condition label01 and verify file exists before unlinking."""
    client_csv = schema_valid_suite / "label01" / "clients" / "client_02.csv"
    assert client_csv.is_file(), "Client CSV must exist prior to deletion"
    client_csv.unlink()
    with pytest.raises(FileNotFoundError, match="Missing or empty required CSV"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_missing_image_content(schema_valid_suite):
    """R2-04: Test with real condition quantity01 and verify file exists before unlinking."""
    img_content = schema_valid_suite / "quantity01" / "image_content.json"
    assert img_content.is_file(), "image_content.json must exist prior to deletion"
    img_content.unlink()
    with pytest.raises(FileNotFoundError, match="Missing image_content.json"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_too_few_visual_pairs(schema_valid_suite):
    vis_pairs = schema_valid_suite / "visually_verified_pairs.json"
    vis_pairs.write_text(json.dumps([{"a": "1.jpg", "b": "2.jpg"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 6 verified pairs"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_tampered_csv_header(schema_valid_suite):
    """R2-03: Detect modified CSV header or missing required columns."""
    train_csv = schema_valid_suite / "label01" / "centralized_train.csv"
    train_csv.write_text("col1,col2\nrow1,row2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid CSV header"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_tampered_row_count(schema_valid_suite):
    """R2-03: Detect CSV row count mismatching expected counts."""
    train_csv = schema_valid_suite / "label01" / "centralized_train.csv"
    content = train_csv.read_text(encoding="utf-8").splitlines(keepends=True)
    # Remove 10 rows
    train_csv.write_text("".join(content[:-10]), encoding="utf-8")
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_tampered_image_hash(schema_valid_suite):
    """R2-03: Detect image hash in image_content.json not being a valid 64-char hex string."""
    img_json = schema_valid_suite / "label01" / "image_content.json"
    data = json.loads(img_json.read_text(encoding="utf-8"))
    first_key = next(iter(data))
    data[first_key] = "not_a_valid_sha256_hash"
    img_json.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid SHA-256 hash"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_tampered_fixture_hash(schema_valid_suite):
    """R2-03: Detect duplicate_review fixture tampered after pinned hash verification."""
    dup_file = schema_valid_suite / "duplicate_review.json"
    dup_file.write_text(json.dumps([{"tampered": True}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_review.json hash mismatch"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_validate_suite_tampered_condition_label(schema_valid_suite):
    """R3-01 and R4-01: Detect label tamper (0->1) in client_00.csv preserving header and row count."""
    target_csv = schema_valid_suite / "label01" / "clients" / "client_00.csv"
    lines = target_csv.read_text(encoding="utf-8").splitlines()
    first_row = lines[1].split(",")
    # Change label from 0 to 1
    orig_label = first_row[1]
    first_row[1] = "1" if orig_label != "1" else "2"
    lines[1] = ",".join(first_row)
    target_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        validate_suite_schema_and_contents(schema_valid_suite)


def test_build_release_v4_rejects_malformed_suite(tmp_path):
    bad_suite = tmp_path / "bad_suite"
    bad_suite.mkdir()
    (bad_suite / "suite.json").write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        build_release_v4(suite_dir=bad_suite, output_dir=out_dir)
    assert not (out_dir / "fl_package_v4.zip").exists()
    assert not (out_dir / "release_manifest_v4.json").exists()
