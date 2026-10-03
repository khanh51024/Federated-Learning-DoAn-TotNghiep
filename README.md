# FedAvg non-IID training v7

This branch contains the exact FedAvg runner and data-contract source used by the
Kaggle label-Dirichlet α=0.1 full run. `SOURCE_MANIFEST.json` records a SHA-256
for every copied source file. The canonical release is `mixed/pv_pd_v3` with
release SHA-256 `6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252`.

Data images, partition manifests, pretrained weights, W0, incumbent checkpoint,
run outputs and the Kaggle upload notebook are external to this code branch.
Supply them from the audited release and the pinned bundle; the launcher checks
release and spec hashes before execution. Run from this repository root:

```powershell
python training-workflows/kaggle_fedavg/run.py `
  --dataset-root <canonical-dataset-root> `
  --runner train-gd-2/kaggle-workspace/campaign-mixed-pv-pd-v1/fedavg_mixed_runner.py `
  --spec-file configs/quality_spec_v4_FEDAVG_FULL.json `
  --pretrained-weights <mobilenet_v3_small-047dcff4.pth> `
  --incumbent-summary <fedavg_summary.json> `
  --incumbent-checkpoint <fedavg_checkpoint_best.pt> `
  --job-id fed_mixed_label_a01_s42 --partition-scheme label_alpha_0_1 `
  --mode full --output-dir <new-output-dir> --preflight-only --execute
```

Remove `--preflight-only` for a new full run. Keep its output
outside the repository. Use the same data release and W0 for comparisons with
the central and local-only branches.
