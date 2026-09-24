# Kaggle Scratch-v4 Training Source Provenance

This directory contains the original Python source package (`scratch_package`) deployed to Kaggle to train the Stage 2 Scratch-v4 MobileNetV3-Small FedAvg models on 2026-09-21.

## 1. Provenance Fingerprints

- **Training Run Source Fingerprint**:
  `6f2b6e8c1d3dd3a03dff0127b90aaf73b9dfc83c8c667eaa90d2e8f1a914a3bd`
  *This matches the `source_sha256` field in `scratch_protocol.json` and the `scratch_source_sha256` in all completed checkpoint identity headers.*
- **Current Operational Codebase (R12 in `training/stage2_scratch_v4/`)**:
  `192e40760869a3a00c3da056da67a1b3ef47273d8d9600ba57976d8fd8a572c7`
- **Partition Suite (`data/partitions_stage2_scratch_v3`)**:
  `eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15` (84 files, 264,037,275 bytes).

## 2. Integrity and Honest Attribution

1. **Why Two Codebases Exist**:
   - The code in this `provenance/` directory is the exact historical closure deployed during the Kaggle training run.
   - The code in `training/stage2_scratch_v4/` is the latest Candidate R12 codebase with enhanced packaging, CLI helpers, and clean namespace management.
   - To maintain scientific integrity, the model identity headers have **not** been rewritten to point to R12. Both snapshots are preserved with their true provenance.
2. **Line Endings and Raw Byte Preservation**:
   - 30 of the 99 Python files in this snapshot contain CRLF line endings from the original development environment.
   - To prevent Git from normalizing line endings and altering the raw SHA-256 digest, `provenance/**` is pinned with `-text` in `.gitattributes`.
3. **Reproducibility**:
   - To replicate training with this exact historical package, pass `--suite data/partitions_stage2_scratch_v3` pointing to the pre-partitioned dataset suite tracked in this repository.
