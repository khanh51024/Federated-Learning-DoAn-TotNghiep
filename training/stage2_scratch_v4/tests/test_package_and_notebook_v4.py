"""Regression tests for package and notebook verification logic (v4)."""
import json
import os
from pathlib import Path

import pytest

from fl_training.package_verify import compute_file_sha256, verify_package_manifest


@pytest.fixture
def mock_package_env(tmp_path):
    pkg_dir = tmp_path / "mock_pkg"
    pkg_dir.mkdir()

    # Create dummy files
    f1 = pkg_dir / "src/module.py"
    f1.parent.mkdir(parents=True)
    f1.write_text("print('hello world')", encoding="utf-8")

    f2 = pkg_dir / "config.json"
    f2.write_text('{"key": "value"}', encoding="utf-8")

    files_map = {
        "src/module.py": {
            "size_bytes": f1.stat().st_size,
            "sha256": compute_file_sha256(f1),
        },
        "config.json": {
            "size_bytes": f2.stat().st_size,
            "sha256": compute_file_sha256(f2),
        },
    }

    manifest = {
        "protocol": "stage2_scratch_fedavg_v4",
        "scope": "fedavg_only",
        "total_files": 2,
        "uncompressed_bytes": sum(m["size_bytes"] for m in files_map.values()),
        "files": files_map,
    }

    manifest_file = pkg_dir / "release_manifest_v4.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return pkg_dir, manifest_file


def test_verify_package_manifest_success(mock_package_env):
    pkg_dir, manifest_file = mock_package_env
    res = verify_package_manifest(pkg_dir, manifest_file)

    assert res["status"] == "VERIFIED"
    assert res["verified_files"] == 2
    assert res["total_bytes"] > 0
    assert len(res["manifest_sha256"]) == 64
    assert res["protocol"] == "stage2_scratch_fedavg_v4"


def test_verify_package_manifest_fails_on_tampered_byte(mock_package_env):
    pkg_dir, manifest_file = mock_package_env
    # Modify 1 byte in module.py while keeping size or changing content
    target = pkg_dir / "src/module.py"
    content = target.read_bytes()
    # Change first character 'p' to 'P' (same length)
    tampered = b"P" + content[1:]
    target.write_bytes(tampered)

    with pytest.raises(ValueError, match="File SHA-256 hash mismatch"):
        verify_package_manifest(pkg_dir, manifest_file)


def test_verify_package_manifest_fails_on_missing_file(mock_package_env):
    pkg_dir, manifest_file = mock_package_env
    (pkg_dir / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="Manifest file missing on disk"):
        verify_package_manifest(pkg_dir, manifest_file)


def test_verify_package_manifest_fails_on_size_mismatch(mock_package_env):
    pkg_dir, manifest_file = mock_package_env
    target = pkg_dir / "config.json"
    target.write_text('{"key": "longer_value_added_here"}', encoding="utf-8")

    with pytest.raises(ValueError, match="File size mismatch"):
        verify_package_manifest(pkg_dir, manifest_file)


def test_verify_package_manifest_fails_on_missing_manifest(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No release manifest found"):
        verify_package_manifest(empty_dir)


def test_verify_package_manifest_fails_on_empty_files(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    manifest = {
        "protocol": "stage2_scratch_fedavg_v4",
        "scope": "fedavg_only",
        "total_files": 0,
        "files": {},
    }
    manifest_file = pkg_dir / "release_manifest_v4.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest 'files' dictionary is empty"):
        verify_package_manifest(pkg_dir, manifest_file)


def test_verify_package_manifest_fails_on_unauthorized_file(mock_package_env):
    pkg_dir, manifest_file = mock_package_env
    # Add an unexpected stray file
    stray = pkg_dir / "stray_malicious_code.py"
    stray.write_text("import os; print('bad')", encoding="utf-8")

    with pytest.raises(ValueError, match="Unauthorized file found in package directory"):
        verify_package_manifest(pkg_dir, manifest_file, strict_allowlist=True)


def test_validate_suite_rejects_tampered_condition_label(tmp_path):
    """R3-01 / R4-03 Regression: Portable unit test independent of author machine paths.
    Creates a synthetic valid fixture in tmp_path, then tampers one label (0 -> 1)
    in label01/clients/client_00.csv while preserving the 13-column header and row counts.
    The builder MUST reject the tampered suite due to manifest_sha256 mismatch.
    """
    from scripts.package_release_v4 import validate_suite_schema_and_contents
    from tests.test_package_release_negative import generate_synthetic_valid_suite

    t_suite = tmp_path / "synthetic_suite"
    generate_synthetic_valid_suite(t_suite)

    # 1. Verify original synthetic suite passes validation
    valid_res = validate_suite_schema_and_contents(t_suite)
    assert valid_res["protocol"] == "stage2_scratch_clean_v3"

    # 2. Tamper test: Modify exactly one label in label01/clients/client_00.csv
    target_csv = t_suite / "label01" / "clients" / "client_00.csv"
    lines = target_csv.read_text(encoding="utf-8").splitlines()
    first_row = lines[1].split(",")
    orig_label = first_row[1]
    first_row[1] = "1" if orig_label != "1" else "2"
    lines[1] = ",".join(first_row)
    target_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Validation MUST fail due to condition manifest_sha256 mismatch
    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        validate_suite_schema_and_contents(t_suite)


def test_validate_suite_real_dataset_integration():
    """R4-03 / R5-02 Integration: Validate real partition suite via explicit configuration.
    If FEDAVG_SUITE_DIR is explicitly configured:
      - If the directory does not exist, fail setup clearly with pytest.fail.
      - If valid, run validation and assert protocol.
    If FEDAVG_SUITE_DIR is not set:
      - Only check candidate local path root_dir / 'data/partitions_stage2_scratch_v3'.
      - If not found locally, skip cleanly without silent probing of author machine paths.
    """
    from scripts.package_release_v4 import validate_suite_schema_and_contents

    suite_env = os.environ.get("FEDAVG_SUITE_DIR")
    if suite_env:
        suite_path = Path(suite_env).resolve()
        if not suite_path.is_dir():
            pytest.fail(f"Configured FEDAVG_SUITE_DIR='{suite_env}' does not exist or is not a directory")
    else:
        root_dir = Path(__file__).resolve().parents[1]
        local_candidate = root_dir / "data" / "partitions_stage2_scratch_v3"
        if local_candidate.is_dir():
            suite_path = local_candidate
        else:
            pytest.skip(
                "Integration test skipped: FEDAVG_SUITE_DIR is not set and local "
                "data/partitions_stage2_scratch_v3 not found."
            )

    res = validate_suite_schema_and_contents(suite_path)
    assert res["protocol"] == "stage2_scratch_clean_v3"
