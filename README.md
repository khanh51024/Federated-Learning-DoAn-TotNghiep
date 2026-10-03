# Centralized training v6

This branch contains the exact central runner and data-contract source used by
the Colab full run. `SOURCE_MANIFEST.json` records SHA-256 for every copied source
file. The canonical release is `mixed/pv_pd_v3`, SHA-256
`6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252`.

The selected checkpoint, summary, validation metrics and charts are in
[results/20261003/RESULTS.md](results/20261003/RESULTS.md). Data images,
pretrained weights, W0, incumbent checkpoint, full run outputs and the Colab
upload notebook remain external. Run from this repository root:

```powershell
python training-workflows/colab_centralized/run.py `
  --dataset-root <canonical-dataset-root> `
  --runner train-tap-trung/scripts/train_production.py `
  --spec-file configs/central_spec_v3.json `
  --pretrained-weights <mobilenet_v3_small-047dcff4.pth> `
  --incumbent-summary <central_summary.json> `
  --incumbent-checkpoint <central_checkpoint_best.pt> `
  --mode full --output-dir <new-output-dir> --preflight-only --execute
```

Remove `--preflight-only` for a new full run. On Colab,
provide `--sync-dir <Drive-output-dir>` for durable checkpoint sync. Keep outputs
outside the repository.
