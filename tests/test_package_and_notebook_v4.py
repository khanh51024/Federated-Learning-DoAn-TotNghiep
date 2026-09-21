"""Regression tests for package and notebook verification logic (v4)."""
import json
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
