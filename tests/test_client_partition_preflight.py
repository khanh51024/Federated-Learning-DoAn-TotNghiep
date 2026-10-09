from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_client_partition.py"


def _write_client_manifest(path: Path, relative_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "label", "class_name"])
        writer.writeheader()
        writer.writerow({"relative_path": relative_path, "label": 0, "class_name": "ClassA"})


def test_client_preflight_requires_only_local_partition_images(tmp_path: Path):
    dataset_root = tmp_path / "local_data"
    (dataset_root / "ClassA").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(dataset_root / "ClassA" / "a.jpg")

    partition = tmp_path / "label1"
    _write_client_manifest(partition / "clients" / "client_00.csv", "ClassA/a.jpg")
    (partition / "partition_config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "preflight.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dataset_root),
            "--partition-dir",
            str(partition),
            "--client-id",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["num_examples"] == 1
    assert payload["missing_count"] == 0
    assert len(payload["client_manifest_sha256"]) == 64


def test_client_preflight_fails_when_local_manifest_image_is_missing(tmp_path: Path):
    dataset_root = tmp_path / "local_data"
    dataset_root.mkdir()
    partition = tmp_path / "label1"
    _write_client_manifest(partition / "clients" / "client_00.csv", "ClassA/missing.jpg")
    output = tmp_path / "preflight.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dataset_root),
            "--partition-dir",
            str(partition),
            "--client-id",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["missing_count"] == 1
