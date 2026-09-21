"""python -m stage2_matched: prepare, preflight, run, collect."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="FedAvg GĐ2 Dirichlet, matched to the stage-1 trainer")
    parser.add_argument("action", choices=["prepare", "preflight", "run", "collect", "baseline"])
    parser.add_argument("--suite", default="data/partitions_stage2_matched_v5")
    parser.add_argument("--dataset", default="../PlantVillage-Dataset/raw/color")
    parser.add_argument("--leaf-map", default="../PlantVillage-Dataset/leaf-map.json")
    parser.add_argument("--output", default="runs/stage2_matched_v5")
    parser.add_argument("--smoke", action="store_true", help="prepare ONLY: small whole-group dataset, 2 rounds, no pretrained")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--session-minutes", type=float, default=420)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["centralized", "local-only", "both"], default="centralized",
                        help="Baseline mode: centralized, local-only, or both")
    args = parser.parse_args()
    if args.smoke and args.action != "prepare":
        parser.error("--smoke is a prepare flag; run automatically reads frozen suite.smoke")
    # Keep new runtime/cache files beside the project, off the Windows C drive.
    runtime = Path(__file__).resolve().parents[1] / ".stage2_matched_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME"):
        os.environ[key] = str(runtime / ("torch" if key == "TORCH_HOME" else "tmp"))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from .data import prepare, preflight
    from .experiment import run, collect, run_baseline
    if args.action == "prepare":
        result = prepare(args.suite, args.dataset, args.leaf_map, args.smoke)
    elif args.action == "preflight":
        suite, audits = preflight(args.suite, args.dataset)
        result = {"protocol": suite["protocol"], "smoke": suite["smoke"], "counts": suite["counts"], "audits": audits}
    elif args.action == "run":
        result = run(args.suite, args.dataset, args.output, args.conditions, args.seeds, args.session_minutes, args.device)
    elif args.action == "baseline":
        result = run_baseline(args.suite, args.dataset, args.output, args.mode, args.conditions, args.seeds, args.device)
    else:
        result = collect(args.output, args.suite)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
