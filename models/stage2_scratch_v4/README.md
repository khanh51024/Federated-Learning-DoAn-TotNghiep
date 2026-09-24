# Stage 2 Scratch-v4 Federated Learning Models (FedAvg MobileNetV3-Small)

This directory contains the global aggregated models and intermediate checkpoints produced by the Stage 2 Scratch FedAvg training experiments (`stage2_scratch_fedavg_v4`) on the PlantVillage dataset (38 classes).

## 1. Architectural & Training Specification

- **Base Architecture**: MobileNetV3-Small (38 output classes).
- **Initialization**: Strict random initialization (`weights=None`, `pretrained=False`).
- **Federated Algorithm**: Canonical FedAvg with sample-size weighting ($w = \sum_{k} \frac{n_k}{N} w_k$).
- **Local Training**: 5 clients, 1 local epoch per round, batch size 32, SGD optimizer (resolved learning rate 0.01).
- **Input Pipeline**: 3-channel RGB, resized to $224 \times 224$, normalized using ImageNet standard (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Partition Suite**: Partitioned from PlantVillage with leaf-group holdout (~20% global test, 8% validation, 72% train across 5 clients). Partition suite tracked in `data/partitions_stage2_scratch_v3` (84 files, 264,037,275 bytes).

---

## 2. Completed Global Models (4 Conditions, Seed 42)

The 4 models below represent completed runs where the global model was selected by validation accuracy across rounds:

| Job ID / Condition | Seed | Best Round | Selection Metric | Test Accuracy | Test Loss | Best Model Path | File SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`label01_seed42`** | 42 | Round 5 | `validation_accuracy` | **25.5013%** | 2.6569 | `label01_seed42/best_model.pth` | `04bb4db39f046c8e3aa786c23ae2418e5ff903c7ea019ee65b73f7528e1694f5` |
| **`label100_seed42`** | 42 | Round 10 | `validation_accuracy` | **53.7222%** | 1.6366 | `label100_seed42/best_model.pth` | `c01e858dbf436894c4892c902dbe2c448fe97f59d4c72ba18819e99a8dcfe194` |
| **`label1_seed42`** | 42 | Round 10 | `validation_accuracy` | **54.8576%** | 1.5841 | `label1_seed42/best_model.pth` | `03d9972b22ec78b4d8d1796c9751e70e53a25d2b6b55365511b8b8eb8db6b1bb` |
| **`quantity100_seed42`**| 42 | Round 10 | `validation_accuracy` | **52.4311%** | 1.7454 | `quantity100_seed42/best_model.pth` | `96f53e34b9d09c25da7cf0a68d89a23c34825946fa5a0f678e715dfd5c95c80a` |

*All 4 best model binary files are 6,364,874 bytes each and tracked via Git LFS.*

### Model Verification Status:
Each model has been verified with:
1. File SHA-256 digest match against run ledger.
2. Embedded `state_sha256` match with `compute_state_dict_sha256(checkpoint['model_state'])`.
3. Safe loading with `torch.load(..., weights_only=True, map_location='cpu')`.
4. Strict state dict loading into `create_mobilenetv3_stage1(num_classes=38, pretrained=False)`.
5. Finite forward pass producing logits with shape `[1, 38]`.

---

## 3. Resume Checkpoints (5 Checkpoint Files)

Intermediate state checkpoints saved during training:

| Condition / Job | Round | Status | Checkpoint Path | File SHA-256 |
| :--- | :---: | :--- | :--- | :--- |
| **`quantity01_seed42`** | 5 | `PAUSED_QUOTA` | `quantity01_seed42/checkpoints/round_0005_paused.pth` | `12389148b3b7280695ee91a329d89201991207137f758baad7eeaf69fe5a57f5` |
| **`label01_seed42`** | 10 | `COMPLETED_JOB_RESUME_SNAPSHOT` | `label01_seed42/checkpoints/round_0010.pth` | `f1a8335b1d7d08ba28489808bfd7e98a3c5d64843b01859bf54e565980072e29` |
| **`label100_seed42`** | 10 | `COMPLETED_JOB_RESUME_SNAPSHOT` | `label100_seed42/checkpoints/round_0010.pth` | `bc8ea83bb516e53063aa8ba9aa74619a97120df055ea2ae3d8dd22b64d30a848` |
| **`label1_seed42`** | 10 | `COMPLETED_JOB_RESUME_SNAPSHOT` | `label1_seed42/checkpoints/round_0010.pth` | `8b6719b3879a9d7ddad097e3612dcf6e5ca90a3ea7cbfbe0dbeceb9f71c469f6` |
| **`quantity100_seed42`**| 10 | `COMPLETED_JOB_RESUME_SNAPSHOT` | `quantity100_seed42/checkpoints/round_0010.pth` | `b324c4d5bc89c7247ca884022a15c3ee42a5c8e239eb14532b2f6efbdf66750b` |

> [!WARNING]
> `quantity01_seed42` was paused mid-training due to Kaggle GPU quota exhaustion at round 5; it has **no final or best model**.
> Checkpoints contain full optimizer and RNG states and are tracked for provenance. They have been verified by file SHA-256 only. Never unpickle resume checkpoints with `weights_only=False` without isolating the execution environment.

---

## 4. Scientific Constraints & Boundary Conditions

1. **`scientific_stage2_complete = false`**: Only 4 non-IID conditions with a single seed (seed 42) have been run to completion. Other Dirichlet configurations (`label_quantity01`, `feature100`, `feature01`) and multi-seed replications are pending.
2. **No Model Averaging Across Conditions**: Each model was trained under a distinct data distribution ($\alpha \in \{0.1, 1.0, 100.0\}$ or quantity skew). Averaging their weights across conditions is scientifically invalid.
3. **No Cross-Condition "Winner"**: Test accuracies across different non-IID conditions reflect partition difficulty, not model quality. Label skew $\alpha=0.1$ is fundamentally harder than $\alpha=100.0$.
4. **Distinction from Legacy Pretrained**: Historical files in `results/fedavg/` (accuracy ~99%) are transfer-learned from ImageNet weights under an older Stage 1 protocol. Do not conflate them with scratch training.

---

## 5. Usage & Inference

Run prediction using `scripts/predict_stage2.py`:

```bash
python scripts/predict_stage2.py \
  --model models/stage2_scratch_v4/label1_seed42/best_model.pth \
  --image path/to/leaf_image.jpg \
  --top-k 5
```
