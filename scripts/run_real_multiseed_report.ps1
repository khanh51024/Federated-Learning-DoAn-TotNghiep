param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [int]$CentralizedEpochs = 10,
    [int]$FederatedRounds = 10,
    [int[]]$Seeds = @(42, 123, 2026)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { throw "Không tìm thấy virtualenv Python tại $Python" }

Write-Host "=== PlantVillage preflight ==="
& $Python "scripts/preflight_plantvillage.py" $DatasetRoot --full-path-check
if ($LASTEXITCODE -ne 0) { throw "Dataset preflight failed. Dừng toàn bộ real experiment." }

foreach ($Seed in $Seeds) {
    Write-Host ""
    Write-Host "=== REAL experiment seed=$Seed ==="
    & $Python "scripts/run_report_experiments.py" `
        --mode real `
        --dataset-root $DatasetRoot `
        --epochs $CentralizedEpochs `
        --rounds $FederatedRounds `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "Experiment seed=$Seed failed." }
}

Write-Host ""
Write-Host "=== Aggregate mean/std across seeds ==="
$seedArgs = @()
foreach ($Seed in $Seeds) { $seedArgs += "$Seed" }
& $Python "scripts/summarize_multiseed.py" --mode real --seeds @seedArgs
if ($LASTEXITCODE -ne 0) { throw "Multi-seed aggregation failed." }

Write-Host ""
Write-Host "REAL multi-seed protocol completed."
Write-Host "Summary: artifacts/report_experiments/multiseed_real_summary.json"
Write-Host "CSV:     artifacts/report_experiments/multiseed_real_summary.csv"
