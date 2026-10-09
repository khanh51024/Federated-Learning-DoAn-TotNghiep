# Training result: train-local-only-a01-v7

These are the best checkpoints from the completed full run on the mixed PlantDoc/PlantVillage release (seed 42).
The table below uses one shared validation manifest: PlantDoc 269 images and PlantVillage 4,344 images.
Validation is not an independent estimate for new Internet photos; no external-photo accuracy is claimed.

| Model | Source | Top-1 | Supported macro-F1 | Crop accuracy | Best checkpoint SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Local 00 | PlantDoc | 25.7% | 20.1% | 59.1% | `55f2b0d476cd18bda957293fd00869a95cb9ab67050d4bab83c5edff54ce86ae` |
| Local 00 | PlantVillage | 44.9% | 26.8% | 79.2% | `55f2b0d476cd18bda957293fd00869a95cb9ab67050d4bab83c5edff54ce86ae` |
| Local 01 | PlantDoc | 25.3% | 11.7% | 57.6% | `0391db294a87c634a5e6ca9d81a90252f35cc3c17d92c1bae4607d56fc091cd1` |
| Local 01 | PlantVillage | 52.0% | 34.9% | 76.2% | `0391db294a87c634a5e6ca9d81a90252f35cc3c17d92c1bae4607d56fc091cd1` |
| Local 02 | PlantDoc | 29.0% | 21.0% | 60.6% | `ba2d936eb74e3de33500c8106fbb1edfef7a59fdbe2627211c479c01639be7c4` |
| Local 02 | PlantVillage | 55.3% | 34.3% | 82.6% | `ba2d936eb74e3de33500c8106fbb1edfef7a59fdbe2627211c479c01639be7c4` |
| Local 03 | PlantDoc | 26.8% | 25.6% | 50.2% | `9d94ab2f7452c8940c1b74118a81e934efb96c09920bba5244bb5f91e2764f76` |
| Local 03 | PlantVillage | 55.5% | 33.9% | 80.3% | `9d94ab2f7452c8940c1b74118a81e934efb96c09920bba5244bb5f91e2764f76` |
| Local 04 | PlantDoc | 21.2% | 18.6% | 52.4% | `7ff7a833777dd195b9cdf922ccba088aa706edea8dc4e9d56a9a744dea8eccb0` |
| Local 04 | PlantVillage | 55.2% | 37.5% | 79.9% | `7ff7a833777dd195b9cdf922ccba088aa706edea8dc4e9d56a9a744dea8eccb0` |

## Artifacts and interpretation

- `validation_metrics.json` contains per-source metrics and prominent class confusions; local-only results are inside each `client_XX/` directory.
- `validation_*.png` compare the centralized, FedAvg and five local-only checkpoints on the same validation split.
- `validation_comparison.csv` contains exact values, validation manifest SHA and checkpoint SHA for all seven models.
- `artifact_manifest.json` records SHA-256 and file size for the files in this result package.
- `training-artifacts/evaluations/20261003/evaluate_three_workflows.py` can reproduce the validation calculation when the shared dataset root is supplied.
- The checkpoints are PyTorch training checkpoints. Load only from this trusted project source; checkpoint metadata and source release SHA are checked during evaluation.
- Five local-only checkpoints are independent models trained on non-IID shards. They were not averaged into a global model. Use the FedAvg branch for the aggregated global model.
