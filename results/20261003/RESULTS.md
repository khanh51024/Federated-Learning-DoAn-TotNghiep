# Training result: train-fedavg-non-iid-v7

These are the best checkpoints from the completed full run on the mixed PlantDoc/PlantVillage release (seed 42).
The table below uses one shared validation manifest: PlantDoc 269 images and PlantVillage 4,344 images.
Validation is not an independent estimate for new Internet photos; no external-photo accuracy is claimed.

| Model | Source | Top-1 | Supported macro-F1 | Crop accuracy | Best checkpoint SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| FedAvg v7 | PlantDoc | 47.2% | 46.9% | 75.5% | `eb5b4e70c17c5e1be70fc7c493e6e8ddb32149150f81496176332e6fe62d0c5c` |
| FedAvg v7 | PlantVillage | 93.1% | 90.2% | 98.7% | `eb5b4e70c17c5e1be70fc7c493e6e8ddb32149150f81496176332e6fe62d0c5c` |

## Artifacts and interpretation

- `validation_metrics.json` contains per-source metrics and prominent class confusions; local-only results are inside each `client_XX/` directory.
- `validation_*.png` compare the centralized, FedAvg and five local-only checkpoints on the same validation split.
- `validation_comparison.csv` contains exact values, validation manifest SHA and checkpoint SHA for all seven models.
- `artifact_manifest.json` records SHA-256 and file size for the files in this result package.
- `training-artifacts/evaluations/20261003/evaluate_three_workflows.py` can reproduce the validation calculation when the shared dataset root is supplied.
- The checkpoints are PyTorch training checkpoints. Load only from this trusted project source; checkpoint metadata and source release SHA are checked during evaluation.
- The selected global checkpoint is round 27; the Kaggle run completed at round 32 with early convergence.
