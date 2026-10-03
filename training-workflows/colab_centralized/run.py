"""Centralized Colab launcher; selects the full canonical train manifest."""

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
    runner = args.runner or _default(root, "train_production.py", "train-tap-trung/scripts/train_production.py")
    spec = args.spec_file or _default(root, "central_spec_v3.json", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/central_spec_v3.json")
    weights = args.pretrained_weights or _default(root, "mobilenet_v3_small-047dcff4.pth", "training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth")
    summary = args.incumbent_summary or _default(root, "incumbent/central_summary.json", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/incumbent/central_summary.json")
    checkpoint = args.incumbent_checkpoint or _default(root, "incumbent/central_checkpoint_best.pt", "training-artifacts/cloud-v4-full-ready-20261001/preflight_extract/incumbent/central_checkpoint_best.pt")
    for path in (runner, spec, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.mode == "full":
        for path in (summary, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
    command = [sys.executable, str(runner), "--mode", args.mode, "--condition", "mixed",
               "--seed", "42", "--dataset-root", view["dataset_root"],
               "--release-dir", view["release_dir"], "--expected-release-sha", view["release_manifest_sha256"],
               "--spec-file", str(spec), "--init", "imagenet_v1", "--pretrained-weights", str(weights),
               "--plantdoc-sampling-ratio", "0.25", "--sampler-repeat-policy", args.sampler_repeat_policy,
               "--transform-version", args.transform_version, "--output-dir", str(args.output_dir.resolve())]
    if args.mode == "full":
        spec_data = json.loads(spec.read_text(encoding="utf-8"))
        hp = spec_data["hyperparameters"]["centralized"]
        if spec_data.get("status") != "READY_FULL":
            raise ValueError("Full central training requires an approved READY_FULL spec")
        command += ["--job-id", "cent_mixed_quality_s42", "--epochs", str(hp["max_epochs"]),
                    "--patience", str(hp["early_stopping_patience"]),
                    "--incumbent-summary", str(summary), "--incumbent-checkpoint", str(checkpoint)]
        if args.sync_dir is not None:
            command += ["--colab-sync-dir", str(args.sync_dir.resolve())]
    if args.mode == "pilot":
        if not args.pilot_manifest:
            raise ValueError("Pilot requires --pilot-manifest")
        if not args.pilot_manifest.is_file():
            raise FileNotFoundError(args.pilot_manifest)
        command += ["--pilot-train-manifest", str(args.pilot_manifest.resolve()), "--epochs", str(args.epochs)]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.stop_after_epoch is not None:
        command += ["--stop-after-epoch", str(args.stop_after_epoch)]
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
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--transform-version", choices=("canonical_v1", "light_augment_v2"), default="canonical_v1")
    parser.add_argument("--sampler-repeat-policy", choices=("with_replacement", "cycle_without_replacement"), default="with_replacement")
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--sync-dir", type=Path, help="Drive checkpoint directory; recommended for Colab full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the selected job; default only prints the plan")
    args = parser.parse_args()
    if args.mode == "full" and (args.transform_version != "canonical_v1" or args.sampler_repeat_policy != "with_replacement"):
        raise ValueError("Experimental full requires a separately approved READY_FULL spec and launcher")
    if args.epochs < 1 or args.epochs > 5:
        raise ValueError("Pilot epochs must be in [1, 5]")
    view = inspect_dataset(args.dataset_root)
    command = build_command(args, view)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.execute:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    print_plan(view, command)
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
