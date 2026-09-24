"""Regression tests for optional Flower import behavior (R3-02).

Verifies that when flower (flwr) is absent or masked:
1. Importing stage2_scratch.flower_adapter does NOT raise AttributeError: 'NoneType' object has no attribute 'common'.
2. Calling run_flower_fedavg raises an informative ImportError mentioning 'pip install .[flower]'.
3. CLI action 'flower_verify' fails with error directing user to install .[flower].
These tests use subprocesses to guarantee isolation and are NOT skipped by pytest.importorskip.
"""

import os
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flower_adapter_import_without_flower():
    """Importing flower_adapter with flwr masked must succeed without AttributeError."""
    code = (
        "import sys\n"
        "sys.modules['flwr'] = None\n"
        "sys.modules['flwr.common'] = None\n"
        "sys.modules['flwr.server'] = None\n"
        "from stage2_scratch.flower_adapter import HAS_FLWR, VerifiedFedAvgStrategy, FlowerPlantClient\n"
        "assert not HAS_FLWR, 'HAS_FLWR should be False when flwr is masked'\n"
        "print('IMPORT_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, f"Import failed with stderr: {res.stderr}"
    assert "IMPORT_OK" in res.stdout


def test_run_flower_fedavg_raises_informative_importerror_without_flower():
    """Calling run_flower_fedavg without flower must raise ImportError mentioning .[flower]."""
    code = (
        "import sys\n"
        "sys.modules['flwr'] = None\n"
        "from stage2_scratch.flower_adapter import run_flower_fedavg\n"
        "try:\n"
        "    run_flower_fedavg(None, None, None, None, None, None, None, None)\n"
        "except ImportError as e:\n"
        "    msg = str(e)\n"
        "    assert '.[flower]' in msg, f'Unexpected message: {msg}'\n"
        "    print('GUARD_OK')\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, f"Guard check failed: {res.stderr}"
    assert "GUARD_OK" in res.stdout


def test_flower_verify_cli_without_flower(tmp_path):
    """CLI action 'flower_verify' without flower must exit with error directing user to install .[flower]."""
    dummy_suite = tmp_path / "dummy_suite"
    dummy_suite.mkdir()
    (dummy_suite / "suite.json").write_text("{}", encoding="utf-8")
    dummy_data = tmp_path / "dummy_data"
    dummy_data.mkdir()

    code = (
        "import sys\n"
        "sys.modules['flwr'] = None\n"
        "import stage2_scratch.__main__\n"
        f"sys.argv = ['stage2_scratch', 'flower_verify', '--suite', r'{dummy_suite}', '--dataset', r'{dummy_data}']\n"
        "try:\n"
        "    stage2_scratch.__main__.main()\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception as e:\n"
        "    msg = str(e)\n"
        "    assert '.[flower]' in msg, f'Unexpected error: {msg}'\n"
        "    print('CLI_GUARD_OK')\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert res.returncode == 0, f"CLI action check failed: {res.stderr}"
    assert "CLI_GUARD_OK" in res.stdout
