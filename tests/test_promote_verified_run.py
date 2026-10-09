from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_verified_run.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(run_dir: Path, preflight: Path, verified_root: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(run_dir),
            "--scenario",
            "label1",
            "--preflight",
            str(preflight),
            "--verified-root",
            str(verified_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_promote_refuses_smoke_even_with_passing_preflight(tmp_path: Path):
    run_dir = tmp_path / "smoke_run"
    _write_json(run_dir / "plan.json", {"mode": "smoke", "seed": 42})
    _write_json(run_dir / "summary.json", {"mode": "smoke", "seed": 42, "scientific_accuracy_valid": False})
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, {"ok": True})

    proc = _run(run_dir, preflight, tmp_path / "verified")

    assert proc.returncode != 0
    assert "run is not mode=real" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "verified").exists()


def test_promote_refuses_real_run_when_scientific_flag_false(tmp_path: Path):
    run_dir = tmp_path / "real_but_invalid"
    _write_json(run_dir / "plan.json", {"mode": "real", "seed": 42})
    _write_json(run_dir / "summary.json", {"mode": "real", "seed": 42, "scientific_accuracy_valid": False})
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, {"ok": True})

    proc = _run(run_dir, preflight, tmp_path / "verified")

    assert proc.returncode != 0
    assert "scientific_accuracy_valid is false" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "verified").exists()


def test_promote_refuses_failed_preflight(tmp_path: Path):
    run_dir = tmp_path / "real_run"
    _write_json(run_dir / "plan.json", {"mode": "real", "seed": 42})
    _write_json(run_dir / "summary.json", {"mode": "real", "seed": 42, "scientific_accuracy_valid": True})
    preflight = tmp_path / "preflight.json"
    _write_json(preflight, {"ok": False})

    proc = _run(run_dir, preflight, tmp_path / "verified")

    assert proc.returncode != 0
    assert "preflight did not pass" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "verified").exists()
