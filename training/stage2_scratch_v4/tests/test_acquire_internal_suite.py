"""Comprehensive regression test suite for scripts/acquire_internal_suite.py (v4 / r9).

Guarantees 100% portable safety, zero mutation on real suites, and complete coverage of
the R7 review findings (R7-01, R7-02, R7-03) across the full A-I subcase matrix:
- No hardcoded absolute path to real suite in tests.
- Unit path guards run entirely on isolated, minimal fixtures in tmp_path.
- Integration test with real suite is strictly opt-in via FEDAVG_SUITE_DIR, strictly read-only,
  and creates private working copies in tmp_path.
- Negative tests NEVER pass real research paths as source, target, or report.
- OS link / Windows junction tests create real junctions in isolated test areas,
  reporting NOT_RUN with mock fallback if OS privileges deny link creation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
import pytest

import scripts.acquire_internal_suite as acq_mod
from scripts.acquire_internal_suite import (
    acquire_suite,
    validate_paths_pre_mutation,
    compute_directory_tree_hash,
    compute_file_sha256,
    is_same_or_descendant,
    is_link_or_reparse,
    check_path_and_ancestors_for_links,
    check_for_links_in_tree,
    PINNED_TREE_HASH,
    EXPECTED_FILE_COUNT,
    EXPECTED_TOTAL_BYTES,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "acquire_internal_suite.py"


def get_real_suite_path() -> Optional[Path]:
    """Retrieve explicit real suite path from environment, failing clearly if set but missing."""
    env_path = os.environ.get("FEDAVG_SUITE_DIR")
    if not env_path:
        return None
    p = Path(env_path)
    if not p.is_dir():
        pytest.fail(f"Configured FEDAVG_SUITE_DIR does not exist or is not a directory: {env_path}")
    return p


def run_cli(*args: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Execute acquire_internal_suite.py via subprocess and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, "-B", str(SCRIPT_PATH), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr


@pytest.fixture
def small_fixture(tmp_path: Path) -> Path:
    """Create a minimal, self-contained suite fixture for unit path and containment testing."""
    src = tmp_path / "small_fixture_source"
    src.mkdir(parents=True, exist_ok=True)
    (src / "suite.json").write_text('{"fixture": true}', encoding="utf-8")
    (src / "file_a.txt").write_text("alpha", encoding="utf-8")
    cond_dir = src / "cond1"
    cond_dir.mkdir(parents=True, exist_ok=True)
    (cond_dir / "data.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
    return src


def _make_verified_fixture(base_dir: Path, monkeypatch) -> Path:
    """Create a fixture matching monkeypatched pins so verification succeeds."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "suite.json").write_text('{"valid": true}', encoding="utf-8")
    (base_dir / "duplicate_review.json").write_text("{}", encoding="utf-8")
    (base_dir / "visually_verified_pairs.json").write_text("[]", encoding="utf-8")

    c_dir = base_dir / "feature01"
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "centralized_train.csv").write_text("id,val\n1,10\n", encoding="utf-8")
    (c_dir / "partition_config.json").write_text("{}", encoding="utf-8")
    (c_dir / "image_content.json").write_text("{}", encoding="utf-8")

    tree_hash, file_count, total_bytes, _ = compute_directory_tree_hash(base_dir)
    cond_hash = acq_mod.compute_condition_manifest_hash(c_dir)

    monkeypatch.setattr(acq_mod, "PINNED_TREE_HASH", tree_hash)
    monkeypatch.setattr(acq_mod, "EXPECTED_FILE_COUNT", file_count)
    monkeypatch.setattr(acq_mod, "EXPECTED_TOTAL_BYTES", total_bytes)
    monkeypatch.setattr(acq_mod, "PINNED_ROOT_FILES", {
        "suite.json": compute_file_sha256(base_dir / "suite.json"),
        "duplicate_review.json": compute_file_sha256(base_dir / "duplicate_review.json"),
        "visually_verified_pairs.json": compute_file_sha256(base_dir / "visually_verified_pairs.json"),
    })
    monkeypatch.setattr(acq_mod, "PINNED_CONDITION_MANIFESTS", {
        "feature01": cond_hash,
    })
    return base_dir


# ===========================================================================
# Test Case A: Valid COPY to new target, report outside tree (Integration)
# ===========================================================================
def test_case_a_valid_copy_and_report_outside(tmp_path: Path):
    real_suite = get_real_suite_path()
    if real_suite is None:
        pytest.skip("FEDAVG_SUITE_DIR not configured; skipping real-suite acquisition test.")

    target_dir = tmp_path / "acquired_suite_a"
    report_file = tmp_path / "report_case_a.json"

    # Compute source state before copy (read-only audit)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(real_suite)

    ret, stdout, stderr = run_cli(
        "--source", str(real_suite),
        "--target-dir", str(target_dir),
        "--json-out", str(report_file),
    )

    assert ret == 0, f"Acquire COPY failed: {stderr}"
    assert "[Acquire] SUCCESS:" in stdout
    assert report_file.is_file()

    # Verify source was NOT modified
    after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(real_suite)
    assert before_tree == after_tree
    assert before_count == after_count
    assert before_bytes == after_bytes

    # Verify target contents
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["status"] == "VERIFIED"
    assert data["mode"] == "COPY"
    assert data["file_count"] == EXPECTED_FILE_COUNT
    assert data["total_bytes"] == EXPECTED_TOTAL_BYTES
    assert data["tree_fingerprint_sha256"] == PINNED_TREE_HASH


# ===========================================================================
# Test Case B: VERIFY-ONLY Mode & Source Independence (Integration)
# ===========================================================================
def test_case_b_verify_only_untouched(tmp_path: Path):
    real_suite = get_real_suite_path()
    if real_suite is None:
        pytest.skip("FEDAVG_SUITE_DIR not configured; skipping real-suite verify-only test.")

    # Create a private working copy in tmp_path so the real suite is never targeted directly
    suite_copy = tmp_path / "private_suite_b"
    shutil.copytree(real_suite, suite_copy)

    report_file = tmp_path / "report_case_b.json"
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(suite_copy)

    # Sub-case B1: Verify-only with report outside
    ret, stdout, stderr = run_cli(
        "--verify-only",
        "--target-dir", str(suite_copy),
        "--json-out", str(report_file),
    )

    assert ret == 0, f"Verify-only failed: {stderr}"
    assert "[Acquire] SUCCESS:" in stdout
    assert report_file.is_file()

    after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(suite_copy)
    assert before_tree == after_tree
    assert before_count == after_count
    assert before_bytes == after_bytes

    # Sub-case B2: Verify-only allows target == source (read-only audit)
    report_b2 = tmp_path / "report_case_b2.json"
    ret_b2, stdout_b2, stderr_b2 = run_cli(
        "--verify-only",
        "--source", str(suite_copy),
        "--target-dir", str(suite_copy),
        "--json-out", str(report_b2),
    )
    assert ret_b2 == 0, f"Verify-only allowing target==source failed: {stderr_b2}"
    assert "[Acquire] SUCCESS:" in stdout_b2
    assert report_b2.is_file()


# ===========================================================================
# Test Case C: Report Collision & Path Containment Rejection (Unit Guard)
# ===========================================================================
def test_case_c_report_collision_rejection(tmp_path: Path, small_fixture: Path):
    """Ensure report cannot be inside target, inside source, or overwrite existing files."""
    suite_json = small_fixture / "suite.json"
    suite_json_sha_before = compute_file_sha256(suite_json)

    # Sub-case C1: Report path collides with suite.json (R6-01 collision bug)
    ret1, stdout1, stderr1 = run_cli(
        "--verify-only",
        "--target-dir", str(small_fixture),
        "--json-out", str(suite_json),
    )
    assert ret1 != 0
    assert "[Acquire] SUCCESS:" not in stdout1
    assert "already exists" in stderr1 or "cannot be inside target" in stderr1
    assert compute_file_sha256(suite_json) == suite_json_sha_before

    # Sub-case C2: Report path is a new file inside target suite
    new_report_inside = small_fixture / "new_report.json"
    ret2, stdout2, stderr2 = run_cli(
        "--verify-only",
        "--target-dir", str(small_fixture),
        "--json-out", str(new_report_inside),
    )
    assert ret2 != 0
    assert "[Acquire] SUCCESS:" not in stdout2
    assert not new_report_inside.exists(), "Report inside target suite must not be created!"

    # Sub-case C3: Report path is inside source directory in COPY mode
    target_c3 = tmp_path / "target_c3"
    report_in_source = small_fixture / "report_in_src.json"
    ret3, stdout3, stderr3 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_c3),
        "--json-out", str(report_in_source),
    )
    assert ret3 != 0
    assert "[Acquire] SUCCESS:" not in stdout3
    assert not target_c3.exists(), "Target must not be created when report is inside source!"
    assert not report_in_source.exists()

    # Sub-case C4: Report path inside declared source in VERIFY-ONLY mode even when source DOES NOT EXIST
    absent_source = tmp_path / "non_existent_source_c4"
    report_in_absent = absent_source / "report.json"
    ret4, stdout4, stderr4 = run_cli(
        "--verify-only",
        "--source", str(absent_source),
        "--target-dir", str(small_fixture),
        "--json-out", str(report_in_absent),
    )
    assert ret4 != 0
    assert "[Acquire] SUCCESS:" not in stdout4
    assert not absent_source.exists(), "Absent source must not be created!"
    assert not report_in_absent.exists()

    # Sub-case C5: Report path is an ancestor of target directory
    ancestor_report = tmp_path / "ancestor_report.json"
    sub_target = ancestor_report / "sub_target"
    ret5, stdout5, stderr5 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(sub_target),
        "--json-out", str(ancestor_report),
    )
    assert ret5 != 0
    assert "[Acquire] SUCCESS:" not in stdout5

    # Sub-case C6: Report path is an ancestor of source directory
    ancestor_src_report = tmp_path / "ancestor_src_report.json"
    sub_src = ancestor_src_report / "sub_src"
    sub_target6 = tmp_path / "sub_target6"
    ret6, stdout6, stderr6 = run_cli(
        "--source", str(sub_src),
        "--target-dir", str(sub_target6),
        "--json-out", str(ancestor_src_report),
    )
    assert ret6 != 0
    assert "[Acquire] SUCCESS:" not in stdout6


# ===========================================================================
# Test Case D: Target Validation & Invalid Report Parent (Unit Guard)
# ===========================================================================
def test_case_d_target_validation_and_invalid_parent(tmp_path: Path, small_fixture: Path):
    # Sub-case D1: Target does not exist in VERIFY-ONLY
    missing_target = tmp_path / "missing_target_d1"
    ret1, stdout1, stderr1 = run_cli(
        "--verify-only",
        "--target-dir", str(missing_target),
    )
    assert ret1 != 0
    assert "[Acquire] SUCCESS:" not in stdout1
    assert not missing_target.exists()

    # Sub-case D2: Target is a regular file in VERIFY-ONLY
    file_target = tmp_path / "file_target_d2.txt"
    file_target.write_text("regular file", encoding="utf-8")
    ret2, stdout2, stderr2 = run_cli(
        "--verify-only",
        "--target-dir", str(file_target),
    )
    assert ret2 != 0
    assert "[Acquire] SUCCESS:" not in stdout2

    # Sub-case D3: Target is a regular file in COPY mode
    ret3, stdout3, stderr3 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(file_target),
    )
    assert ret3 != 0
    assert "[Acquire] SUCCESS:" not in stdout3
    assert file_target.read_text(encoding="utf-8") == "regular file"

    # Sub-case D4: Report parent is a regular file -> must fail BEFORE copy begins!
    blocker_file = tmp_path / "blocker_file.txt"
    blocker_file.write_text("I am a file, not a directory", encoding="utf-8")
    invalid_report_path = blocker_file / "report.json"
    target_d4 = tmp_path / "target_should_not_exist_d4"

    ret4, stdout4, stderr4 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_d4),
        "--json-out", str(invalid_report_path),
    )
    assert ret4 != 0
    assert "[Acquire] SUCCESS:" not in stdout4
    assert not target_d4.exists(), "Target must NOT be created when report parent is an invalid file!"
    assert blocker_file.read_text(encoding="utf-8") == "I am a file, not a directory"

    # Sub-case D5: Pre-existing report file -> rejects before copy
    existing_report = tmp_path / "existing_report_d5.json"
    existing_report.write_text("PRE_EXISTING_CONTENT", encoding="utf-8")
    target_d5 = tmp_path / "target_d5"

    ret5, stdout5, stderr5 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_d5),
        "--json-out", str(existing_report),
    )
    assert ret5 != 0
    assert "[Acquire] SUCCESS:" not in stdout5
    assert not target_d5.exists()
    assert existing_report.read_text(encoding="utf-8") == "PRE_EXISTING_CONTENT"


# ===========================================================================
# Test Case E: Overlap & Existence Rejection in COPY Mode (Unit Guard)
# ===========================================================================
def test_case_e_target_overlap_and_existence(tmp_path: Path, small_fixture: Path):
    """Ensure COPY mode strictly prevents target overwrite and directory overlap."""
    # Sub-case E1: Target already exists with content
    existing_target = tmp_path / "existing_target_e1"
    existing_target.mkdir(parents=True)
    (existing_target / "file.txt").write_text("data")
    ret1, _, stderr1 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(existing_target),
    )
    assert ret1 != 0
    assert "already exists" in stderr1
    assert (existing_target / "file.txt").read_text() == "data"

    # Sub-case E2: Source == Target in COPY mode (both isolated in tmp_path)
    ret2, _, stderr2 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(small_fixture),
    )
    assert ret2 != 0
    assert "already exists" in stderr2 or "cannot be inside or identical" in stderr2

    # Sub-case E3: Target inside Source (both isolated in tmp_path)
    target_in_source = small_fixture / "sub_target_e3"
    ret3, _, stderr3 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_in_source),
    )
    assert ret3 != 0
    assert "cannot be inside or identical" in stderr3
    assert not target_in_source.exists()

    # Sub-case E4: Source inside Target
    outer_target = tmp_path / "outer_target_e4"
    inner_source = outer_target / "inner_source"
    inner_source.mkdir(parents=True)
    (inner_source / "f.txt").write_text("content")

    ret4, _, stderr4 = run_cli(
        "--source", str(inner_source),
        "--target-dir", str(outer_target),
    )
    assert ret4 != 0
    assert "cannot be inside" in stderr4 or "already exists" in stderr4


# ===========================================================================
# Test Case F: Cryptographic Tamper & Integrity Rejection (Unit on Fixture)
# ===========================================================================
def test_case_f_tampered_suite_rejection(tmp_path: Path, monkeypatch):
    """Ensure tamper, missing files, or hash mismatches fail acquisition verification."""
    fixture_dir = _make_verified_fixture(tmp_path / "tamper_fixture", monkeypatch)
    c_dir = fixture_dir / "feature01"

    # Sub-case F0: Untampered fixture passes
    rep = acq_mod.acquire_suite(source_dir=None, target_dir=fixture_dir, verify_only=True)
    assert rep["status"] == "VERIFIED"

    # Sub-case F1: Corrupt file content
    (c_dir / "centralized_train.csv").write_text("CORRUPTED_CONTENT", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        acq_mod.acquire_suite(source_dir=None, target_dir=fixture_dir, verify_only=True)

    # Restore
    (c_dir / "centralized_train.csv").write_text("id,val\n1,10\n", encoding="utf-8")

    # Sub-case F2: Missing root file
    (fixture_dir / "suite.json").unlink()
    with pytest.raises(ValueError, match="Root file 'suite.json' SHA mismatch|count mismatch"):
        acq_mod.acquire_suite(source_dir=None, target_dir=fixture_dir, verify_only=True)


# ===========================================================================
# Test Case G: VERIFY-ONLY Source Independence (Unit Guard)
# ===========================================================================
def test_case_g_verify_only_source_independence(tmp_path: Path, small_fixture: Path):
    non_existent_source = tmp_path / "does_not_exist_source_g"
    report_file = tmp_path / "report_g.json"

    # In verify-only, source non-existence should NOT block verifying a target directory
    assert not non_existent_source.exists()
    assert small_fixture.is_dir()

    # Pre-mutation validation directly
    validate_paths_pre_mutation(
        source_dir=non_existent_source,
        target_dir=small_fixture,
        json_out=report_file,
        verify_only=True,
    )
    # Target and missing source must be completely untouched
    assert not non_existent_source.exists()
    assert not report_file.exists()


# ===========================================================================
# Test Case H: Empty Target, Obsolete Flag, Sibling Paths & OS Links
# ===========================================================================
def test_case_h_empty_target_flag_sibling_and_links(tmp_path: Path, small_fixture: Path):
    # Sub-case H1: Empty target directory already exists in COPY mode -> must fail
    empty_target = tmp_path / "empty_target_h1"
    empty_target.mkdir()
    ret1, _, stderr1 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(empty_target),
    )
    assert ret1 != 0
    assert "already exists" in stderr1

    # Sub-case H2: Passing obsolete --allow-overwrite flag must be rejected
    ret2, _, stderr2 = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(tmp_path / "target_h2"),
        "--allow-overwrite",
    )
    assert ret2 != 0
    assert "unrecognized arguments: --allow-overwrite" in stderr2

    # Sub-case H3: Sibling path sharing prefix (e.g. source_sibling vs source)
    sibling_target = tmp_path / f"{small_fixture.name}_sibling"
    # Pre-mutation validation must succeed for sibling paths
    validate_paths_pre_mutation(
        source_dir=small_fixture,
        target_dir=sibling_target,
        json_out=None,
        verify_only=False,
    )
    assert not is_same_or_descendant(sibling_target, small_fixture)
    assert not is_same_or_descendant(small_fixture, sibling_target)

    # Sub-case H4: OS Link / Junction Guards
    link_created = False
    junc_dir = tmp_path / "suite_junction_h4"
    if sys.platform == "win32":
        try:
            import _winapi
            _winapi.CreateJunction(str(small_fixture), str(junc_dir))
            link_created = True
        except Exception as exc:
            print(f"[NOTE] OS junction creation failed: {exc}")

    if link_created:
        try:
            # Root is junction -> must be rejected
            assert is_link_or_reparse(junc_dir)
            with pytest.raises(ValueError, match="reparse point|symlink"):
                check_path_and_ancestors_for_links(junc_dir, label="Target suite")

            # Check inside tree rejection
            with pytest.raises(ValueError, match="reparse point|symlink"):
                check_for_links_in_tree(junc_dir, label="Target suite")
        finally:
            os.rmdir(str(junc_dir))
    else:
        # If OS denies link creation in the environment, verify rejection via mock and record NOT_RUN
        print("[NOT_RUN] OS link creation not supported by current environment permissions; verifying logic via mock.")
        mock_path = tmp_path / "mock_reparse"
        mock_path.mkdir(exist_ok=True)
        orig_is_link = acq_mod.is_link_or_reparse
        try:
            acq_mod.is_link_or_reparse = lambda p: True if str(p) == str(mock_path) else orig_is_link(p)
            with pytest.raises(ValueError, match="reparse point|symlink"):
                acq_mod.check_path_and_ancestors_for_links(mock_path)
        finally:
            acq_mod.is_link_or_reparse = orig_is_link


# ===========================================================================
# Test Case I: Mid-stream Fault Injection & Atomic Publication
# ===========================================================================

class _FailingFileWrapper:
    def __init__(self, wrapped, fail_on=""):
        self._wrapped = wrapped
        self._fail_on = fail_on

    def write(self, s):
        return self._wrapped.write(s)

    def flush(self):
        if self._fail_on == "flush":
            raise OSError("Simulated flush failure after write")
        return self._wrapped.flush()

    def fileno(self):
        return self._wrapped.fileno()

    def close(self):
        if self._fail_on == "close":
            raise OSError("Simulated close failure after write")
        return self._wrapped.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def test_fault_i1_copy_midstream_failure(tmp_path: Path, small_fixture: Path, monkeypatch):
    """Test mid-stream copy failure: partial copy retained, source untouched, no SUCCESS."""
    target_dir1 = tmp_path / "target_fault_copy_i1"
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(small_fixture)

    def mock_copytree_fail(src, dst, *args, **kwargs):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "partial_copy.tmp").write_text("PARTIAL_CONTENT", encoding="utf-8")
        raise IOError("Simulated hardware I/O error during copy")

    monkeypatch.setattr(acq_mod.shutil, "copytree", mock_copytree_fail)
    try:
        with pytest.raises(IOError, match="Simulated hardware I/O error"):
            acq_mod.acquire_suite(
                source_dir=small_fixture,
                target_dir=target_dir1,
                verify_only=False,
            )

        # Partial output is preserved for diagnosis, not claimed as success
        assert (target_dir1 / "partial_copy.tmp").is_file()
        # Source suite is 100% untouched
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(small_fixture)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_fault_i2a_dump_write_failure_partial(tmp_path: Path, monkeypatch):
    """Test write failure mid-stream: partial bytes written to staging, suite untouched, no report."""
    fx_i2a = _make_verified_fixture(tmp_path / "fx_i2a", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fx_i2a)
    report_i2a = tmp_path / "report_i2a.json"

    def mock_dump_fail_partial(obj, fp, *args, **kwargs):
        fp.write('{"status": "PARTIAL_UNFINISHED"')
        fp.flush()
        raise IOError("Simulated disk full during report write mid-stream")

    monkeypatch.setattr(acq_mod.json, "dump", mock_dump_fail_partial)
    try:
        with pytest.raises(IOError, match="Simulated disk full"):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=fx_i2a,
                verify_only=True,
                json_out=report_i2a,
            )

        # Valid report must NOT have been published
        assert not report_i2a.exists()

        # Pre-existing suite bytes, count, and tree fingerprint are 100% untouched
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fx_i2a)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree

        # Check that staging file was retained for diagnostics with partial content
        staging_files = list(tmp_path.glob(".tmp_report_*.json"))
        assert len(staging_files) >= 1
        assert "PARTIAL_UNFINISHED" in staging_files[0].read_text(encoding="utf-8")
    finally:
        monkeypatch.undo()


def test_fault_i2b_flush_failure_partial(tmp_path: Path, monkeypatch):
    """Test flush failure after write: staging file has written bytes, suite untouched, no report."""
    fx_i2b = _make_verified_fixture(tmp_path / "fx_i2b", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fx_i2b)
    report_i2b = tmp_path / "report_i2b.json"

    import builtins
    orig_open = builtins.open

    def mock_open_flush(file, mode="r", *args, **kwargs):
        f = orig_open(file, mode, *args, **kwargs)
        if "w" in mode or "a" in mode:
            return _FailingFileWrapper(f, fail_on="flush")
        return f

    monkeypatch.setattr(builtins, "open", mock_open_flush)
    try:
        with pytest.raises(OSError, match="Simulated flush failure after write"):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=fx_i2b,
                verify_only=True,
                json_out=report_i2b,
            )
        assert not report_i2b.exists()
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fx_i2b)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_fault_i2c_fsync_failure_partial(tmp_path: Path, monkeypatch):
    """Test fsync failure after write/flush: staging file has bytes, suite untouched, no report."""
    fx_i2c = _make_verified_fixture(tmp_path / "fx_i2c", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fx_i2c)
    report_i2c = tmp_path / "report_i2c.json"

    def mock_fsync_fail(*args, **kwargs):
        raise OSError("Simulated fsync failure after write")

    monkeypatch.setattr(acq_mod.os, "fsync", mock_fsync_fail)
    try:
        with pytest.raises(OSError, match="Simulated fsync failure after write"):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=fx_i2c,
                verify_only=True,
                json_out=report_i2c,
            )
        assert not report_i2c.exists()
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fx_i2c)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_fault_i2d_close_failure_partial(tmp_path: Path, monkeypatch):
    """Test close failure after write: suite untouched, no report."""
    fx_i2d = _make_verified_fixture(tmp_path / "fx_i2d", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fx_i2d)
    report_i2d = tmp_path / "report_i2d.json"

    import builtins
    orig_open = builtins.open

    def mock_open_close(file, mode="r", *args, **kwargs):
        f = orig_open(file, mode, *args, **kwargs)
        if "w" in mode or "a" in mode:
            return _FailingFileWrapper(f, fail_on="close")
        return f

    monkeypatch.setattr(builtins, "open", mock_open_close)
    try:
        with pytest.raises(OSError, match="Simulated close failure after write"):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=fx_i2d,
                verify_only=True,
                json_out=report_i2d,
            )
        assert not report_i2d.exists()
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fx_i2d)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_fault_i2e_mkdir_failure(tmp_path: Path, monkeypatch):
    """Test mkdir failure for report parent: suite untouched, no report."""
    fx_i2e = _make_verified_fixture(tmp_path / "fx_i2e", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fx_i2e)
    report_in_sub = tmp_path / "sub_new" / "report.json"

    def mock_mkdir_fail(*args, **kwargs):
        raise PermissionError("Simulated mkdir permission denied for report parent")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir_fail)
    try:
        with pytest.raises(PermissionError, match="Simulated mkdir permission denied"):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=fx_i2e,
                verify_only=True,
                json_out=report_in_sub,
            )
        assert not report_in_sub.exists()
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fx_i2e)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_fault_i3_publish_collision(tmp_path: Path, monkeypatch):
    """Test publish collision: pre-existing file preserved, suite untouched, FileExistsError raised."""
    valid_fixture3 = _make_verified_fixture(tmp_path / "valid_fixture_i3", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(valid_fixture3)
    report_i3 = tmp_path / "report_i3.json"

    orig_rename = acq_mod.os.rename

    def mock_rename_collision(src, dst):
        Path(dst).write_text("PRE_EXISTING_FOREIGN_REPORT", encoding="utf-8")
        return orig_rename(src, dst)

    monkeypatch.setattr(acq_mod.os, "rename", mock_rename_collision)
    try:
        with pytest.raises(FileExistsError):
            acq_mod.acquire_suite(
                source_dir=None,
                target_dir=valid_fixture3,
                verify_only=True,
                json_out=report_i3,
            )

        assert report_i3.read_text(encoding="utf-8") == "PRE_EXISTING_FOREIGN_REPORT"
        after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(valid_fixture3)
        assert after_count == before_count
        assert after_bytes == before_bytes
        assert after_tree == before_tree
    finally:
        monkeypatch.undo()


def test_case_i_fault_injection_handling(tmp_path: Path, small_fixture: Path, monkeypatch):
    """Composite runner for I1, I2a-e, I3 fault injection subcases."""
    test_fault_i1_copy_midstream_failure(tmp_path / "sub_i1", small_fixture, monkeypatch)
    test_fault_i2a_dump_write_failure_partial(tmp_path / "sub_i2a", monkeypatch)
    test_fault_i2b_flush_failure_partial(tmp_path / "sub_i2b", monkeypatch)
    test_fault_i2c_fsync_failure_partial(tmp_path / "sub_i2c", monkeypatch)
    test_fault_i2d_close_failure_partial(tmp_path / "sub_i2d", monkeypatch)
    test_fault_i2e_mkdir_failure(tmp_path / "sub_i2e", monkeypatch)
    test_fault_i3_publish_collision(tmp_path / "sub_i3", monkeypatch)


# ===========================================================================
# Test Case J: R8-01 Regression: Relative Report When CWD Is Beneath Junction
# ===========================================================================
def test_case_j_regression_r8_01_cwd_under_junction(tmp_path: Path):
    """Verify that when CWD is an ordinary directory beneath a junction, relative report is rejected."""
    if sys.platform != "win32":
        pytest.skip("Windows junction test; Linux native marked NOT_RUN")

    import _winapi
    real_storage = tmp_path / "real_storage_j"
    real_storage.mkdir(parents=True, exist_ok=True)
    real_subdir = real_storage / "subdir"
    real_subdir.mkdir(parents=True, exist_ok=True)

    alias_junction = tmp_path / "alias_junction_j"
    _winapi.CreateJunction(str(real_storage), str(alias_junction))

    target_suite = tmp_path / "target_suite_j"
    target_suite.mkdir(parents=True, exist_ok=True)
    (target_suite / "suite.json").write_text('{"protocol": "stage2_scratch_clean_v3"}')

    cwd_under = alias_junction / "subdir"
    cmd = [sys.executable, "-B", str(SCRIPT_PATH), "--verify-only", "--target-dir", str(target_suite), "--json-out", "relative-out.json"]
    proc = subprocess.run(cmd, cwd=str(cwd_under), capture_output=True, text=True, encoding="utf-8")

    assert proc.returncode != 0
    assert "Symlinks and junctions are strictly prohibited" in proc.stderr
    assert not (real_subdir / "relative-out.json").exists()


# ===========================================================================
# Test Case K: R8-01 Regression: Path Traversal '..' Prohibited
# ===========================================================================
def test_case_k_regression_r8_01_path_traversal_dot_dot(tmp_path: Path, small_fixture: Path):
    target = tmp_path / "target_k"
    report_dot_dot = tmp_path / "sub" / ".." / "report_k.json"

    ret, stdout, stderr = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target),
        "--json-out", str(report_dot_dot),
    )
    assert ret != 0
    assert "strictly prohibited" in stderr
    assert not target.exists()


# ===========================================================================
# Test Case L: R8-01 Regression: Report Destination Is Hardlink
# ===========================================================================
def test_case_l_regression_r8_01_report_hardlink(tmp_path: Path, small_fixture: Path):
    target = tmp_path / "target_l"
    orig_file = tmp_path / "orig_file_l.json"
    orig_file.write_text("orig")
    hardlink_report = tmp_path / "hardlink_report_l.json"
    os.link(str(orig_file), str(hardlink_report))

    ret, stdout, stderr = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target),
        "--json-out", str(hardlink_report),
    )
    assert ret != 0
    assert "hardlink" in stderr or "already exists" in stderr
    assert not target.exists()


# ===========================================================================
# Test Case M: R8-01 Regression: Fail-Closed On Unexpected Lstat Error
# ===========================================================================
def test_case_m_regression_r8_01_fail_closed_on_lstat_error(tmp_path: Path):
    from unittest.mock import patch
    with patch("os.lstat", side_effect=PermissionError("Simulated access denied")):
        with pytest.raises(ValueError, match="Cannot verify path attributes"):
            acq_mod.is_link_or_reparse(tmp_path / "test_file.txt")


# ===========================================================================
# Test Case N: Missing FEDAVG_SUITE_DIR Path Fails Loudly
# ===========================================================================
def test_case_n_missing_env_path_fails(tmp_path: Path, monkeypatch):
    invalid_path = tmp_path / "definitely_nonexistent_suite_path_99999"
    monkeypatch.setenv("FEDAVG_SUITE_DIR", str(invalid_path))
    with pytest.raises(pytest.fail.Exception, match="does not exist or is not a directory"):
        get_real_suite_path()


# ===========================================================================


# ===========================================================================
# Test Case O: R9-01 Regression: Windows Extended Prefix CLI Containment
# ===========================================================================
def test_case_o_r9_01_extended_prefix_containment_cli(tmp_path: Path):
    r"""Reproduces the reviewer's exact CLI test on an ordinary fixture copy.
    
    Verifies that --target-dir <plain> --json-out \\?\plain\unexpected.json
    exits nonzero, does NOT write unexpected.json into the target, and leaves
    suite file count and tree hash strictly unchanged.
    """
    fixture = tmp_path / "ext_prefix_fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "suite.json").write_text('{"protocol": "stage2_scratch_clean_v3"}', encoding="utf-8")
    (fixture / "file1.txt").write_text("data1", encoding="utf-8")

    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fixture)
    report_extended = "\\\\?\\" + str(fixture / "unexpected.json")

    cmd = [
        sys.executable, "-B", str(SCRIPT_PATH),
        "--verify-only",
        "--target-dir", str(fixture),
        "--json-out", report_extended,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    assert p.returncode != 0, f"Expected nonzero exit, got {p.returncode}. stdout: {p.stdout}"
    assert "[Acquire] SUCCESS:" not in p.stdout
    assert "cannot be inside" in p.stderr
    assert not (fixture / "unexpected.json").exists(), "unexpected.json was created inside suite!"

    after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fixture)
    assert after_count == before_count
    assert after_bytes == before_bytes
    assert after_tree == before_tree


# ===========================================================================
# Test Case P: R9-01 Regression: Two-Way Plain/Extended Permutation Matrix
# ===========================================================================
def test_case_p_r9_01_two_way_plain_extended_matrix(tmp_path: Path):
    """Tests all 4 target/report containment combinations and source overlap."""
    fixture = tmp_path / "matrix_fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "suite.json").write_text('{"valid": true}', encoding="utf-8")

    plain_tgt = str(fixture)
    ext_tgt = "\\\\?\\" + plain_tgt

    combos = [
        ("plain target, extended report inside", plain_tgt, "\\\\?\\" + str(fixture / "rep1.json")),
        ("extended target, plain report inside", ext_tgt, str(fixture / "rep2.json")),
        ("extended target, extended report inside", ext_tgt, "\\\\?\\" + str(fixture / "rep3.json")),
        ("plain target, plain report inside", plain_tgt, str(fixture / "rep4.json")),
    ]

    for label, tgt, rep in combos:
        ret, stdout, stderr = run_cli(
            "--verify-only",
            "--target-dir", tgt,
            "--json-out", rep,
        )
        assert ret != 0, f"{label}: Expected failure but got {ret}"
        assert "[Acquire] SUCCESS:" not in stdout
        assert "cannot be inside" in stderr

    # Negative: report inside source (plain source, extended report)
    src_dir = tmp_path / "source_tree"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "suite.json").write_text('{"src": true}', encoding="utf-8")
    out_target = tmp_path / "target_never_created"

    ret_src, stdout_src, stderr_src = run_cli(
        "--source", str(src_dir),
        "--target-dir", str(out_target),
        "--json-out", "\\\\?\\" + str(src_dir / "report_in_source.json"),
    )
    assert ret_src != 0
    assert "[Acquire] SUCCESS:" not in stdout_src
    assert "cannot be inside declared source" in stderr_src
    assert not out_target.exists()


# ===========================================================================
# Test Case Q: R9-01 Regression: Extended Prefix Positive Outside Tree
# ===========================================================================
def test_case_q_r9_01_extended_prefix_positive_cli(tmp_path: Path, monkeypatch):
    """Tests that positive verification with extended report OUTSIDE suite passes."""
    fixture = _make_verified_fixture(tmp_path / "fixture_pos_q", monkeypatch)
    before_tree, before_count, before_bytes, _ = compute_directory_tree_hash(fixture)
    report_outside_ext = Path("\\\\?\\" + str(tmp_path / "outside_rep_q.json"))

    rep = acq_mod.acquire_suite(
        source_dir=None,
        target_dir=fixture,
        verify_only=True,
        json_out=report_outside_ext,
    )
    assert rep["status"] == "VERIFIED"
    assert Path(tmp_path / "outside_rep_q.json").exists()

    after_tree, after_count, after_bytes, _ = compute_directory_tree_hash(fixture)
    assert after_count == before_count
    assert after_bytes == before_bytes
    assert after_tree == before_tree

    real_suite = get_real_suite_path()
    if real_suite is not None:
        report_real_ext = "\\\\?\\" + str(tmp_path / "outside_real_q.json")
        cmd = [
            sys.executable, "-B", str(SCRIPT_PATH),
            "--verify-only",
            "--target-dir", str(real_suite),
            "--json-out", report_real_ext,
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert p.returncode == 0
        assert "[Acquire] SUCCESS:" in p.stdout
        assert Path(tmp_path / "outside_real_q.json").exists()


# ===========================================================================
# Test Case R: R9-01 Regression: Unsupported Namespaces & Devices Rejection
# ===========================================================================
def test_case_r_unsupported_namespaces_and_devices(tmp_path: Path, small_fixture: Path):
    """Rejects device namespace (\\.\\), NT namespace (\\??\\), DOS device names, trailing dots and spaces."""
    # 1. Device namespace (\\.\\)
    cmd1 = [sys.executable, "-B", str(SCRIPT_PATH), "--target-dir", r"\\.\PhysicalDrive0", "--verify-only"]
    p1 = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8")
    assert p1.returncode != 0
    assert "Win32 device namespace" in p1.stderr

    # 2. NT device namespace (\\??\\)
    cmd_nt = [sys.executable, "-B", str(SCRIPT_PATH), "--target-dir", r"\??\D:\fake_dev", "--verify-only"]
    p_nt = subprocess.run(cmd_nt, capture_output=True, text=True, encoding="utf-8")
    assert p_nt.returncode != 0
    assert "NT object manager namespace" in p_nt.stderr

    # 3. Reserved DOS device names: CON, PRN, AUX, NUL, COM1, LPT1
    dos_devices = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
    for dev in dos_devices:
        cmd_dos = [sys.executable, "-B", str(SCRIPT_PATH), "--target-dir", str(small_fixture), "--verify-only", "--json-out", dev]
        p_dos = subprocess.run(cmd_dos, capture_output=True, text=True, encoding="utf-8")
        assert p_dos.returncode != 0
        assert "Reserved Windows device name" in p_dos.stderr

    # 4. Trailing dot in report component
    cmd_dot = [sys.executable, "-B", str(SCRIPT_PATH), "--target-dir", str(small_fixture), "--verify-only", "--json-out", str(tmp_path / "report_dot.")]
    p_dot = subprocess.run(cmd_dot, capture_output=True, text=True, encoding="utf-8")
    assert p_dot.returncode != 0
    assert "trailing dot or space" in p_dot.stderr

    # 5. Trailing space in report component
    cmd_space = [sys.executable, "-B", str(SCRIPT_PATH), "--target-dir", str(small_fixture), "--verify-only", "--json-out", str(tmp_path / "report_space ")]
    p_space = subprocess.run(cmd_space, capture_output=True, text=True, encoding="utf-8")
    assert p_space.returncode != 0
    assert "trailing dot or space" in p_space.stderr


def test_case_s_source_target_report_ancestor_matrix(tmp_path: Path, small_fixture: Path):
    """Validates ancestor and containment checks across relative and absolute forms,
    including source/target/report containment, relative CWD resolution, and
    NTFS junction ancestors/children.
    """
    # Record initial fixture state
    src_before_tree, src_before_count, src_before_bytes, _ = compute_directory_tree_hash(small_fixture)

    # --- Part 1: String & Lexical Ancestor Hierarchy ---
    base = tmp_path / "base_tree"
    base.mkdir(parents=True, exist_ok=True)
    child = base / "child_suite"
    child.mkdir(parents=True, exist_ok=True)
    grandchild = child / "deep"
    grandchild.mkdir(parents=True, exist_ok=True)

    assert is_same_or_descendant(child, base)
    assert is_same_or_descendant(grandchild, base)
    assert is_same_or_descendant(grandchild, child)
    assert not is_same_or_descendant(base, child)
    assert not is_same_or_descendant(base, grandchild)

    sibling = tmp_path / "base_tree_sibling"
    sibling.mkdir(parents=True, exist_ok=True)
    assert not is_same_or_descendant(sibling, base)
    assert not is_same_or_descendant(base, sibling)

    # --- Part 2: Source / Target / Report Containment Rejections (Absolute & Relative) ---
    work_dir = tmp_path / "work_rel"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Case 2.1: Target inside Source (Absolute & Relative)
    # Absolute API
    nested_target = small_fixture / "nested_target"
    with pytest.raises(ValueError, match="cannot be inside or identical to source"):
        acq_mod.validate_paths_pre_mutation(
            source_dir=small_fixture,
            target_dir=nested_target,
            json_out=tmp_path / "outside_21.json",
            verify_only=False,
        )
    # Absolute CLI
    ret_21a, stdout_21a, stderr_21a = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(nested_target),
        "--json-out", str(tmp_path / "outside_21.json"),
    )
    assert ret_21a != 0
    assert "[Acquire] SUCCESS:" not in stdout_21a
    assert "cannot be inside or identical to source" in stderr_21a
    assert not nested_target.exists()
    assert not (tmp_path / "outside_21.json").exists()

    # Relative CLI
    rel_src = work_dir / "rel_src"
    shutil.copytree(small_fixture, rel_src)
    rel_src_before_tree, rel_src_before_count, rel_src_before_bytes, _ = compute_directory_tree_hash(rel_src)

    ret_21r, stdout_21r, stderr_21r = run_cli(
        "--source", "rel_src",
        "--target-dir", "rel_src/nested_tgt",
        "--json-out", "outside_rel_21.json",
        cwd=work_dir,
    )
    assert ret_21r != 0
    assert "[Acquire] SUCCESS:" not in stdout_21r
    assert "cannot be inside or identical to source" in stderr_21r
    assert not (rel_src / "nested_tgt").exists()
    assert not (work_dir / "outside_rel_21.json").exists()
    rel_src_after_tree, rel_src_after_count, rel_src_after_bytes, _ = compute_directory_tree_hash(rel_src)
    assert rel_src_after_tree == rel_src_before_tree
    assert rel_src_after_count == rel_src_before_count
    assert rel_src_after_bytes == rel_src_before_bytes

    # Case 2.2: Source inside Target (Absolute & Relative)
    # Note: In COPY mode, if source exists inside target, then target necessarily already exists.
    # The COPY contract strictly rejects existing targets first via FileExistsError.
    target_sup = tmp_path / "target_sup"
    src_nested = target_sup / "src_nested"
    src_nested.mkdir(parents=True, exist_ok=True)
    (src_nested / "suite.json").write_text('{"nested": true}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists. In COPY mode, target must not exist prior to acquisition."):
        acq_mod.validate_paths_pre_mutation(
            source_dir=src_nested,
            target_dir=target_sup,
            json_out=tmp_path / "outside_22.json",
            verify_only=False,
        )
    ret_22a, stdout_22a, stderr_22a = run_cli(
        "--source", str(src_nested),
        "--target-dir", str(target_sup),
        "--json-out", str(tmp_path / "outside_22.json"),
    )
    assert ret_22a != 0
    assert "[Acquire] SUCCESS:" not in stdout_22a
    assert "already exists. In COPY mode, target must not exist prior to acquisition." in stderr_22a
    assert not (tmp_path / "outside_22.json").exists()

    # Relative CLI
    rel_tgt_sup = work_dir / "rel_tgt_sup"
    rel_nested_src = rel_tgt_sup / "rel_nested_src"
    rel_nested_src.mkdir(parents=True, exist_ok=True)
    (rel_nested_src / "suite.json").write_text('{"nested_rel": true}', encoding="utf-8")
    ret_22r, stdout_22r, stderr_22r = run_cli(
        "--source", "rel_tgt_sup/rel_nested_src",
        "--target-dir", "rel_tgt_sup",
        "--json-out", "outside_rel_22.json",
        cwd=work_dir,
    )
    assert ret_22r != 0
    assert "[Acquire] SUCCESS:" not in stdout_22r
    assert "already exists. In COPY mode, target must not exist prior to acquisition." in stderr_22r
    assert not (work_dir / "outside_rel_22.json").exists()

    # Case 2.3: Report inside Target (Absolute & Relative)
    target_valid = tmp_path / "target_valid"
    rep_inside_target = target_valid / "sub" / "report.json"
    with pytest.raises(ValueError, match="cannot be inside target suite directory"):
        acq_mod.validate_paths_pre_mutation(
            source_dir=small_fixture,
            target_dir=target_valid,
            json_out=rep_inside_target,
            verify_only=False,
        )
    ret_23a, stdout_23a, stderr_23a = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_valid),
        "--json-out", str(rep_inside_target),
    )
    assert ret_23a != 0
    assert "[Acquire] SUCCESS:" not in stdout_23a
    assert "cannot be inside target suite directory" in stderr_23a
    assert not target_valid.exists()
    assert not rep_inside_target.exists()

    # Relative report inside target via CLI
    ret_23r, stdout_23r, stderr_23r = run_cli(
        "--source", str(small_fixture),
        "--target-dir", "rel_target_23",
        "--json-out", "rel_target_23/sub_rep.json",
        cwd=work_dir,
    )
    assert ret_23r != 0
    assert "[Acquire] SUCCESS:" not in stdout_23r
    assert "cannot be inside target suite directory" in stderr_23r
    assert not (work_dir / "rel_target_23").exists()

    # Case 2.4: Report inside Source (Absolute & Relative)
    rep_inside_source = small_fixture / "report_inside_src.json"
    target_valid2 = tmp_path / "target_valid2"
    with pytest.raises(ValueError, match="cannot be inside declared source directory"):
        acq_mod.validate_paths_pre_mutation(
            source_dir=small_fixture,
            target_dir=target_valid2,
            json_out=rep_inside_source,
            verify_only=False,
        )
    ret_24a, stdout_24a, stderr_24a = run_cli(
        "--source", str(small_fixture),
        "--target-dir", str(target_valid2),
        "--json-out", str(rep_inside_source),
    )
    assert ret_24a != 0
    assert "[Acquire] SUCCESS:" not in stdout_24a
    assert "cannot be inside declared source directory" in stderr_24a
    assert not target_valid2.exists()
    assert not rep_inside_source.exists()

    # Relative report inside source via CLI
    ret_24r, stdout_24r, stderr_24r = run_cli(
        "--source", "rel_src",
        "--target-dir", "rel_target_24",
        "--json-out", "rel_src/inside_rep.json",
        cwd=work_dir,
    )
    assert ret_24r != 0
    assert "[Acquire] SUCCESS:" not in stdout_24r
    assert "cannot be inside declared source directory" in stderr_24r
    assert not (work_dir / "rel_target_24").exists()
    assert not (rel_src / "inside_rep.json").exists()

    # Case 2.5: Report is ancestor of target or source
    rep_parent = tmp_path / "rep_ancestor_parent"
    target_sub = rep_parent / "sub_target"
    with pytest.raises(ValueError, match="cannot be an ancestor of target directory"):
        acq_mod.validate_paths_pre_mutation(
            source_dir=small_fixture,
            target_dir=target_sub,
            json_out=rep_parent,
            verify_only=False,
        )

    # Verify small_fixture was NOT mutated across Part 2
    src_after_tree, src_after_count, src_after_bytes, _ = compute_directory_tree_hash(small_fixture)
    assert src_after_tree == src_before_tree
    assert src_after_count == src_before_count
    assert src_after_bytes == src_before_bytes

    # --- Part 3: NTFS Junction / Reparse Ancestor Matrix ---
    if sys.platform == "win32":
        import _winapi
        real_store = tmp_path / "real_storage_s"
        real_store.mkdir(parents=True, exist_ok=True)
        (real_store / "data.txt").write_text("storage", encoding="utf-8")
        junction_root = tmp_path / "junc_root_s"

        # Catch OSError ONLY immediately at CreateJunction
        try:
            _winapi.CreateJunction(str(real_store), str(junction_root))
        except OSError as exc:
            pytest.skip(f"NOT_RUN: Environment does not support NTFS junction creation: {exc}")

        # Subcase S_J1: Target has junction ancestor
        junc_target = junction_root / "target_inside_junc"
        with pytest.raises(ValueError, match="Symlinks and junctions are strictly prohibited"):
            acq_mod.validate_paths_pre_mutation(
                source_dir=small_fixture,
                target_dir=junc_target,
                json_out=tmp_path / "rep_safe_j1.json",
                verify_only=False,
            )
        ret_j1, stdout_j1, stderr_j1 = run_cli(
            "--source", str(small_fixture),
            "--target-dir", str(junc_target),
            "--json-out", str(tmp_path / "rep_safe_j1.json"),
        )
        assert ret_j1 != 0
        assert "[Acquire] SUCCESS:" not in stdout_j1
        assert "Symlinks and junctions are strictly prohibited" in stderr_j1
        assert not junc_target.exists()
        assert not (tmp_path / "rep_safe_j1.json").exists()

        # Subcase S_J2: Report has junction ancestor
        junc_report = junction_root / "rep_inside_junc.json"
        target_j2 = tmp_path / "target_plain_j2"
        with pytest.raises(ValueError, match="Symlinks and junctions are strictly prohibited"):
            acq_mod.validate_paths_pre_mutation(
                source_dir=small_fixture,
                target_dir=target_j2,
                json_out=junc_report,
                verify_only=False,
            )
        ret_j2, stdout_j2, stderr_j2 = run_cli(
            "--source", str(small_fixture),
            "--target-dir", str(target_j2),
            "--json-out", str(junc_report),
        )
        assert ret_j2 != 0
        assert "[Acquire] SUCCESS:" not in stdout_j2
        assert "Symlinks and junctions are strictly prohibited" in stderr_j2
        assert not target_j2.exists()
        assert not junc_report.exists()

        # Subcase S_J3: Source has junction ancestor
        junc_source = junction_root / "src_inside_junc"
        junc_source.mkdir(parents=True, exist_ok=True)
        (junc_source / "suite.json").write_text('{"valid": true}', encoding="utf-8")
        target_j3 = tmp_path / "target_plain_j3"
        with pytest.raises(ValueError, match="Symlinks and junctions are strictly prohibited"):
            acq_mod.validate_paths_pre_mutation(
                source_dir=junc_source,
                target_dir=target_j3,
                json_out=tmp_path / "rep_safe_j3.json",
                verify_only=False,
            )
        ret_j3, stdout_j3, stderr_j3 = run_cli(
            "--source", str(junc_source),
            "--target-dir", str(target_j3),
            "--json-out", str(tmp_path / "rep_safe_j3.json"),
        )
        assert ret_j3 != 0
        assert "[Acquire] SUCCESS:" not in stdout_j3
        assert "Symlinks and junctions are strictly prohibited" in stderr_j3
        assert not target_j3.exists()
        assert not (tmp_path / "rep_safe_j3.json").exists()

        # Subcase S_J4: Target itself is junction root
        with pytest.raises(ValueError, match="Symlinks and junctions are strictly prohibited"):
            acq_mod.validate_paths_pre_mutation(
                source_dir=small_fixture,
                target_dir=junction_root,
                json_out=tmp_path / "rep_safe_j4.json",
                verify_only=False,
            )
        ret_j4, stdout_j4, stderr_j4 = run_cli(
            "--source", str(small_fixture),
            "--target-dir", str(junction_root),
            "--json-out", str(tmp_path / "rep_safe_j4.json"),
        )
        assert ret_j4 != 0
        assert "[Acquire] SUCCESS:" not in stdout_j4
        assert "Symlinks and junctions are strictly prohibited" in stderr_j4
        assert not (tmp_path / "rep_safe_j4.json").exists()

    # Final assertion: small_fixture remains completely untouched
    final_tree, final_count, final_bytes, _ = compute_directory_tree_hash(small_fixture)
    assert final_tree == src_before_tree
    assert final_count == src_before_count
    assert final_bytes == src_before_bytes
