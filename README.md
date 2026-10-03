# Local-only training, five non-IID clients

This branch contains the exact local-only launcher, runner and data-contract
source used for the CUDA label-Dirichlet α=0.1 baseline. Each client trains an
independent model; no weights are aggregated. `SOURCE_MANIFEST.json` records a
SHA-256 for every copied source file. The canonical release SHA-256 is
`6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252`.

Data images, shard manifests, pretrained weights, W0 and run outputs are external.
Set `WORKSPACE_ROOT` to this repository root so the launcher finds its runners,
and pass the audited inputs explicitly:

```powershell
$env:WORKSPACE_ROOT = (Get-Location).Path
python training-workflows/laptop_local_only/launch_local_only.py `
  --dataset-root <canonical-dataset-root> `
  --spec-file configs/quality_spec_v4_FEDAVG_FULL.json `
  --pretrained-weights <mobilenet_v3_small-047dcff4.pth> `
  --partition-scheme label_alpha_0_1 --client-id client_00 `
  --output-dir <new-output-dir> --device cuda --dry-run
```

Replace `--dry-run` with `--full` for a new run. Use a separate output directory
for each client. `run_remaining_clients.py` is a campaign-specific queue that
refers to the original workspace and completed client_00; it is preserved as
provenance, not a portable launcher.
