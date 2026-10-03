"""Kaggle FedAvg launcher; selects audited client shards from one dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

WORKFLOWS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS))
from shared.data_selection import inspect_dataset, print_plan  # noqa: E402


def _default(code_root: Path, packaged: str, local: str) -> Path:
    return code_root / packaged if (code_root / packaged).exists() else code_root / local


def build_command(args: argparse.Namespace, view: dict) -> list[str]:
    root = args.code_root.resolve()
    runner = args.runner or _default(root, "fedavg_mixed_runner.py", "train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1/fedavg_mixed_runner.py")
    spec = args.spec_file or _default(root, "quality_spec_v4_FEDAVG_FULL.json", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/quality_spec_v4_FEDAVG_FULL.json")
    weights = args.pretrained_weights or _default(root, "mobilenet_v3_small-047dcff4.pth", "training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth")
    summary = args.incumbent_summary or _default(root, "incumbent/fedavg_summary.json", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/incumbent/fedavg_summary.json")
    checkpoint = args.incumbent_checkpoint or _default(root, "incumbent/fedavg_checkpoint_best.pt", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/incumbent/fedavg_checkpoint_best.pt")
    for path in (runner, spec, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = json.loads(spec.read_text(encoding="utf-8"))
    jobs = {job["job_id"]: job for job in data.get("kaggle_fedavg_jobs", [])}
    if args.job_id not in jobs:
        raise ValueError(f"Unknown job ID: {args.job_id}")
    if jobs[args.job_id]["partition_scheme"] != args.partition_scheme:
        raise ValueError("Selected partition differs from signed job spec; supply a matching approved spec")
    if data.get("release_manifest_sha256") != view["release_manifest_sha256"]:
        raise ValueError("Job spec belongs to a different dataset release")
    if args.mode == "full":
        for path in (summary, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
    command = [sys.executable, str(runner), "--mode", args.mode, "--job-id", args.job_id,
               "--dataset-root", view["dataset_root"], "--release-dir", view["release_dir"],
               "--expected-release-sha", view["release_manifest_sha256"], "--spec-file", str(spec),
               "--init", "imagenet_v1", "--pretrained-weights", str(weights),
               "--plantdoc-sampling-ratio", "0.25", "--sampler-repeat-policy", args.sampler_repeat_policy,
               "--transform-version", args.transform_version, "--output-dir", str(args.output_dir.resolve())]
    if args.mode == "full":
        command += ["--incumbent-summary", str(summary), "--incumbent-checkpoint", str(checkpoint)]
    if args.mode == "smoke":
        command += ["--rounds", "2"]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.stop_after_round is not None:
        command += ["--stop-after-round", str(args.stop_after_round)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="One shared canonical dataset directory")
    parser.add_argument("--code-root", type=Path, default=WORKFLOWS.parent)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--spec-file", type=Path)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--incumbent-summary", type=Path)
    parser.add_argument("--incumbent-checkpoint", type=Path)
    parser.add_argument("--job-id", default="fed_mixed_iid_baseline_s42")
    parser.add_argument("--partition-scheme", default="iid")
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--transform-version", choices=("canonical_v1", "light_augment_v2"), default="canonical_v1")
    parser.add_argument("--sampler-repeat-policy", choices=("with_replacement", "cycle_without_replacement"), default="with_replacement")
    parser.add_argument("--stop-after-round", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the selected job; default only prints the plan")
    args = parser.parse_args()
    if args.mode == "full" and (args.transform_version != "canonical_v1" or args.sampler_repeat_policy != "with_replacement"):
        raise ValueError("Experimental full requires a separately approved READY_FULL spec and launcher")
    view = inspect_dataset(args.dataset_root, args.partition_scheme)
    command = build_command(args, view)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.execute:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    print_plan(view, command)
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
