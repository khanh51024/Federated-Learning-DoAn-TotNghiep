"""Inject interruption after a committed round; compare resumed real-image weights."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--dataset", default="../PlantVillage-Dataset/raw/color", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output for the interruption test")
    runtime = ROOT / ".stage2_matched_runtime" / "tmp"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(runtime), TMP=str(runtime), TMPDIR=str(runtime))
    import torch
    torch.set_num_threads(4)
    from stage2_matched.experiment import run, collect
    from stage1_compat import runner
    from stage1_compat.budget import BudgetExhausted
    from stage1_compat.checkpoint import load_checkpoint
    original = runner.save_atomic_checkpoint

    def interrupt_after_commit(*a, **kw):
        result = original(*a, **kw)
        if kw["server_round"] == 1:
            raise BudgetExhausted("Test interruption AFTER checkpoint commit, BEFORE ledger update")
        return result

    runner.save_atomic_checkpoint = interrupt_after_commit
    try:
        interrupted = run(args.suite, args.dataset, args.output, ["feature01"], session_minutes=20, device="cpu")
        assert interrupted[-1]["status"] == "PAUSED_DEADLINE"
    finally:
        runner.save_atomic_checkpoint = original
    resumed = run(args.suite, args.dataset, args.output, ["feature01"], session_minutes=20, device="cpu")
    assert resumed[0]["status"] == "COMPLETED"
    filename = "feature01_seed42/checkpoints/feature01_seed42_checkpoint.pth"
    left, right = load_checkpoint(args.reference / filename), load_checkpoint(args.output / filename)
    assert left["identity"] == right["identity"]
    for field in ("global_model_state", "best_model_state"):
        assert all(torch.equal(v, right[field][k]) for k, v in left[field].items()), field
    report = collect(args.output, args.suite)
    result = {"interrupted_after_round": 1, "resumed_round": right["server_round"],
              "tensor_entries": len(left["global_model_state"]), "max_abs_difference": 0,
              "best_weights_equal": True, "smoke": True,
              "scope": "Real images, synthetic feature domains; CPU, 2 rounds, no pretrained",
              "test": report["completed_jobs"][0]["test"]}
    (args.output / "resume_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
