# Training result: train-centralized-v6

These are the best checkpoints from the completed full run on the mixed PlantDoc/PlantVillage release (seed 42).
The table below uses one shared validation manifest: PlantDoc 269 images and PlantVillage 4,344 images.
Validation is not an independent estimate for new Internet photos; no external-photo accuracy is claimed.

| Model | Source | Top-1 | Supported macro-F1 | Crop accuracy | Best checkpoint SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Centralized v6 | PlantDoc | 53.9% | 53.5% | 78.8% | `7fa1a91f20fac2f37526366d7d11256d409f23e45f0ced5dd89e837ade7bb603` |
| Centralized v6 | PlantVillage | 99.2% | 99.0% | 99.7% | `7fa1a91f20fac2f37526366d7d11256d409f23e45f0ced5dd89e837ade7bb603` |

## Artifacts and interpretation

- `validation_metrics.json` contains per-source metrics and prominent class confusions; local-only results are inside each `client_XX/` directory.
- `validation_*.png` compare the centralized, FedAvg and five local-only checkpoints on the same validation split.
- `validation_comparison.csv` contains exact values, validation manifest SHA and checkpoint SHA for all seven models.
- `artifact_manifest.json` records SHA-256 and file size for the files in this result package.
- `training-artifacts/evaluations/20261003/evaluate_three_workflows.py` can reproduce the validation calculation when the shared dataset root is supplied.
- The checkpoints are PyTorch training checkpoints. Load only from this trusted project source; checkpoint metadata and source release SHA are checked during evaluation.
- The selected checkpoint is epoch 22; training stopped early after epoch 83. See `centralized_training_curve.png`.
