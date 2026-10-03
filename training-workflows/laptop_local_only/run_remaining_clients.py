"""Run audited local-only clients 01-04 sequentially, reusing completed client 00.

This is the five-client label-alpha-0.1 baseline matching the FedAvg v7 spec.
It never aggregates weights and never starts duplicate client runs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = ROOT / "training-workflows/dist/fedavg-v7-non-iid-20261003/preflight_extract/quality_spec_v4_FEDAVG_FULL.json"
CLIENT_00_RUN = HERE / "runs/local_only_label_a01_client00_full_cuda_20261003"
DEFAULT_OUTPUT_ROOT = HERE / "runs/local_only_label_a01_clients01to04_full_cuda_20261003"
EXPECTED_RELEASE_SHA = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"
EXPECTED_W0 = "eaf9197b42fb39163186a7285f7f1ed4f1e728bd83522260fdc5f0f51c110850"


def read_completed_summary(path: Path, client_id: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "backend": "local_only",
        "device": "cuda:0",
        "partition_scheme": "label_alpha_0_1",
        "partial_batches": None,
        "plantdoc_sampling_ratio": "0.25",
        "rounds": 60,
        "seed": 42,
        "release_manifest_sha256": EXPECTED_RELEASE_SHA,
        "w0_fingerprint": EXPECTED_W0,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"{path}: {key} differs from the five-client campaign")
    if set(data.get("clients", {})) != {client_id}:
        raise ValueError(f"{path}: expected only {client_id}")
    client_dir = path.parent / client_id
    if not (client_dir / "checkpoint_best.pt").is_file() or not (client_dir / "checkpoint_last.pt").is_file():
        raise FileNotFoundError(f"{path}: completed checkpoints are missing")
    return data


@contextmanager
def output_lock(output_root: Path):
    """Hold a Windows file lock while dispatching all four clients."""
    import msvcrt

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".runner.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Another local-only client queue is active: {output_root}") from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def execute(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    first_summary = CLIENT_00_RUN / "summary.json"
    if not first_summary.is_file():
        raise FileNotFoundError(f"Completed client_00 baseline missing: {first_summary}")
    read_completed_summary(first_summary, "client_00")
    if not SPEC.is_file():
        raise FileNotFoundError(SPEC)

    summaries = {"client_00": first_summary}
    plan = []
    for number in range(1, 5):
        client_id = f"client_{number:02d}"
        run_dir = output_root / client_id
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / client_id / "checkpoint_last.pt"
        if summary_path.is_file():
            read_completed_summary(summary_path, client_id)
            action = "skip completed"
        elif checkpoint_path.is_file():
            action = "resume"
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Unknown partial output; inspect before continuing: {run_dir}")
        else:
            action = "new"
        summaries[client_id] = summary_path
        plan.append((client_id, run_dir, action))

    for client_id, run_dir, action in plan:
        print(f"{client_id}: {action} -> {run_dir}", flush=True)
    if args.dry_run:
        return 0

    launcher = HERE / "launch_local_only.py"
    for client_id, run_dir, action in plan:
        if action == "skip completed":
            continue
        command = [
            sys.executable, "-u", str(launcher),
            "--partition-scheme", "label_alpha_0_1",
            "--client-id", client_id,
            "--full", "--device", "cuda",
            "--spec-file", str(SPEC),
            "--output-dir", str(run_dir),
        ]
        if action == "resume":
            command.append("--resume")
        print(f"Starting {client_id} ({action})", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        read_completed_summary(run_dir / "summary.json", client_id)

    combined = {
        "backend": "local_only",
        "partition_scheme": "label_alpha_0_1",
        "rounds_per_client": 60,
        "seed": 42,
        "release_manifest_sha256": EXPECTED_RELEASE_SHA,
        "w0_fingerprint": EXPECTED_W0,
        "clients": {},
    }
    for client_id, path in summaries.items():
        combined["clients"][client_id] = read_completed_summary(path, client_id)["clients"][client_id]
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "combined_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    temporary.replace(target)
    print(f"Five-client local-only baseline complete: {target}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return execute(args)
    with output_lock(args.output_root.resolve()):
        return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
