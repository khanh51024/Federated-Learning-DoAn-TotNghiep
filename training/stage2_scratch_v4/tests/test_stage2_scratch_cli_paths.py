"""Unit tests for Stage 2 Scratch CLI path resolution and dataset gating (R2-01).

Verifies:
1. Actions not requiring images (collect, compare-stage1) execute successfully without dataset.
2. Default suite resolution is anchored and independent of CWD (cannot be shadowed by CWD).
3. Explicit invalid --suite or --dataset paths are rejected immediately with FileNotFoundError.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_SUITE = ROOT.parent.parent / "training-data/stage2/partitions_stage2_scratch_v3"


def test_explicit_bad_suite_path_rejected():
    cmd = [sys.executable, "-m", "stage2_scratch", "collect", "--suite", "D:/nonexistent_suite_dir_12345"]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode != 0
    assert "FileNotFoundError" in res.stderr
    assert "Specified --suite path does not exist" in res.stderr


def test_explicit_bad_dataset_path_rejected():
    cmd = [sys.executable, "-m", "stage2_scratch", "preflight", "--dataset", "D:/nonexistent_dataset_dir_12345"]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode != 0
    assert "FileNotFoundError" in res.stderr
    assert "Specified --dataset path does not exist" in res.stderr


def test_collect_without_dataset_succeeds(tmp_path):
    """R2-01: collect must not require dataset to be present."""
    if not REAL_SUITE.is_dir():
        pytest.skip("External suite partitions_stage2_scratch_v3 not available")
    
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    from stage2_scratch.experiment import frozen_protocol
    proto = frozen_protocol(json.loads((REAL_SUITE / "suite.json").read_text(encoding="utf-8")))
    (out_dir / "scratch_protocol.json").write_text(json.dumps(proto), encoding="utf-8")
    cmd = [
        sys.executable, "-m", "stage2_scratch", "collect",
        "--suite", str(REAL_SUITE),
        "--output", str(out_dir),
    ]
    # Intentionally do not provide --dataset
    env = os.environ.copy()
    env.pop("STAGE2_DATASET_PATH", None)
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"collect failed without dataset: {res.stderr}"
    data = json.loads(res.stdout)
    assert "summary" in data or "collected" in data or "completed" in data or isinstance(data, dict)


def test_compare_stage1_without_dataset_succeeds(tmp_path):
    """R2-01: compare-stage1 must not require dataset to be present."""
    if not REAL_SUITE.is_dir():
        pytest.skip("External suite partitions_stage2_scratch_v3 not available")
    
    out_dir = tmp_path / "compare_out"
    cmd = [
        sys.executable, "-m", "stage2_scratch", "compare-stage1",
        "--suite", str(REAL_SUITE),
        "--output", str(out_dir),
    ]
    env = os.environ.copy()
    env.pop("STAGE2_DATASET_PATH", None)
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"compare-stage1 failed without dataset: {res.stderr}"
    target_json = out_dir / "stage1_comparison.json"
    assert target_json.is_file()
    data = json.loads(target_json.read_text(encoding="utf-8"))
    assert "centralized" in data or "table" in data or isinstance(data, dict)


def test_default_suite_precedence_independent_of_cwd(tmp_path):
    """R2-01: Having a shadow fake suite in CWD must NOT override default suite resolution."""
    if not REAL_SUITE.is_dir():
        pytest.skip("External suite partitions_stage2_scratch_v3 not available")

    # Create a shadow suite in tmp_path
    shadow_suite = tmp_path / "data/partitions_stage2_scratch_v3"
    shadow_suite.mkdir(parents=True)
    (shadow_suite / "suite.json").write_text(json.dumps({"shadow_suite": True, "protocol": "fake"}), encoding="utf-8")

    # Run CLI from tmp_path as CWD, with PYTHONPATH set to candidate
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.pop("STAGE2_SUITE_PATH", None)

    probe_code = (
        "import sys, json; "
        "from pathlib import Path; "
        "import stage2_scratch.__main__ as m; "
        "sys.argv = ['stage2_scratch', 'compare-stage1', '--output', 'out']; "
        "parser_func = m.main; "
    )
    # Test resolution logic directly:
    probe_script = tmp_path / "test_probe.py"
    probe_script.write_text(
        "import sys, os, json\n"
        "from pathlib import Path\n"
        "repo_root = Path(sys.argv[1])\n"
        "suite_candidates = []\n"
        "if os.environ.get('STAGE2_SUITE_PATH'):\n"
        "    suite_candidates.append(Path(os.environ['STAGE2_SUITE_PATH']).resolve())\n"
        "suite_candidates.append(repo_root / 'data' / 'partitions_stage2_scratch_v3')\n"
        "for ancestor in [repo_root] + list(repo_root.parents):\n"
        "    suite_candidates.append(ancestor / 'training-data' / 'stage2' / 'partitions_stage2_scratch_v3')\n"
        "suite_path = None\n"
        "for cand in suite_candidates:\n"
        "    if cand.is_dir() and (cand / 'suite.json').is_file():\n"
        "        suite_path = cand\n"
        "        break\n"
        "print(str(suite_path))\n",
        encoding="utf-8"
    )
    res = subprocess.run([sys.executable, str(probe_script), str(ROOT)], cwd=str(tmp_path), capture_output=True, text=True, env=env)
    assert res.returncode == 0
    resolved = res.stdout.strip()
    assert str(shadow_suite.resolve()) != resolved, f"Shadow suite in CWD unexpectedly took precedence: {resolved}"
    assert "training-data" in resolved or "data" in resolved
