"""Compare v4 FedAvg checkpoints with the matched centralized baseline per seed.

Reads only val and legacy_diagnostic_test predictions, never the sealed split.
Old v4b initialization hashes were not recorded: reconstructed W0 hashes are
explicitly distinguished from hashes observed during the original training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.compare_model_weights import grouped_distance
from scripts.eval_sweep_v4 import recipe_of
from scripts.train_v4 import build_model
from src.model.mobilenetv3 import compute_model_hash, get_parameters
from src.training.data import sha256_file


def prediction_agreement(left: Path, right: Path) -> dict:
    result = {}
    with np.load(left / "logits_best.npz", allow_pickle=False) as a, np.load(right / "logits_best.npz", allow_pickle=False) as b:
        for split in ("val", "legacy_diagnostic_test"):
            ids_a, ids_b = a[f"{split}_sample_ids"], b[f"{split}_sample_ids"]
            if len(set(ids_a.tolist())) != len(ids_a) or len(set(ids_b.tolist())) != len(ids_b):
                raise ValueError(f"duplicate sample ids: {left}, {right}, {split}")
            order_a, order_b = np.argsort(ids_a), np.argsort(ids_b)
            if not np.array_equal(ids_a[order_a], ids_b[order_b]):
                raise ValueError(f"different evaluation samples: {left}, {right}, {split}")
            pred_a, pred_b = a[split][order_a].argmax(1), b[split][order_b].argmax(1)
            result[split] = {"n": len(pred_a), "same": int((pred_a == pred_b).sum()),
                             "agreement": float((pred_a == pred_b).mean())}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    compared, initializations, excluded = [], {}, []
    for base in args.runs:
        for run in sorted(base.iterdir()):
            if not (run / "DONE").is_file():
                continue
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if config.get("method") != "fedavg" or config.get("limit"):
                continue
            seed = int(config["seed"])
            central = args.baseline_runs / f"centralized_s{seed}"
            central_config = json.loads((central / "config.json").read_text(encoding="utf-8"))
            if recipe_of(config) != recipe_of(central_config) or config.get("release_manifest_sha256") != central_config.get("release_manifest_sha256"):
                excluded.append({"run": str(run), "reason": "recipe or release differs from centralized"})
                continue
            if seed not in initializations:
                initializations[seed] = compute_model_hash(get_parameters(build_model(seed)))
            # Do not access any sealed metrics in final.json.
            final = json.loads((run / "final.json").read_text(encoding="utf-8"))
            central_final = json.loads((central / "final.json").read_text(encoding="utf-8"))
            hashes = {"reconstructed_from_seed": initializations[seed], "fedavg_recorded": final.get("w0_sha256"),
                      "centralized_recorded": central_final.get("w0_sha256")}
            for key in ("fedavg_recorded", "centralized_recorded"):
                if hashes[key] is not None and hashes[key] != hashes["reconstructed_from_seed"]:
                    raise ValueError(f"W0 mismatch: {run}, {key}")
            left = torch.load(central / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
            right = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
            if left.keys() != right.keys() or any(left[k].shape != right[k].shape or left[k].dtype != right[k].dtype for k in left):
                raise ValueError(f"incompatible state dictionaries: {run}")
            if any(not torch.isfinite(value).all() for state in (left, right) for value in state.values() if value.is_floating_point()):
                raise ValueError(f"non-finite checkpoint: {run}")
            compared.append({"run": str(run), "centralized": str(central), "seed": seed, "scheme": config["scheme"],
                             "runtime": config.get("runtime", "simulation"), "w0": hashes,
                             "checkpoint_sha256": {"centralized": sha256_file(central / "checkpoint_best.pt"),
                                                   "fedavg": sha256_file(run / "checkpoint_best.pt")},
                             "weight_distance": grouped_distance({k: v.numpy() for k, v in left.items()}, {k: v.numpy() for k, v in right.items()}),
                             "prediction_agreement": prediction_agreement(central, run)})
            del left, right
    result = {"evidence": "matched trained v4 checkpoints and saved predictions; no additional training or sealed evaluation",
              "w0_caveat": "reconstructed W0 verifies the seed recipe; it cannot prove the actual initialization of an old run without a recorded hash",
              "initialization_reconstruction_torch": torch.__version__, "comparisons": compared, "excluded": excluded}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"compared {len(compared)} runs; excluded {len(excluded)}; wrote {args.out}")
    return 0 if compared and not excluded else 1


if __name__ == "__main__":
    raise SystemExit(main())
