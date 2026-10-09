param(
    [Parameter(Mandatory=$true)][int]$ClientId,
    [Parameter(Mandatory=$true)][string]$Server,
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [string]$PartitionDir = "data/partitions_stage2_scratch_v3/label100",
    [int]$LocalEpochs = 1,
    [int]$Seed = 42,
    [ValidateSet("normal","no_train","random_weights","scaled_update","noise_update")]
    [string]$FaultMode = "normal"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { throw "Không tìm thấy virtualenv Python tại $Python" }

Write-Host "Preflight local partition for Client $ClientId at: $DatasetRoot"
& $Python "scripts/preflight_client_partition.py" $DatasetRoot `
    --partition-dir $PartitionDir `
    --client-id $ClientId
if ($LASTEXITCODE -ne 0) { throw "Local client partition preflight failed. Client sẽ không khởi động." }

Write-Host "Starting REAL Flower client $ClientId -> $Server"
Write-Host "Partition: $PartitionDir"
Write-Host "Fault mode: $FaultMode"

& $Python -m src.federated.client_app `
    --client-id $ClientId `
    --server $Server `
    --partition-dir $PartitionDir `
    --dataset-root $DatasetRoot `
    --local-epochs $LocalEpochs `
    --seed $Seed `
    --fault-mode $FaultMode `
    --require-real-data

exit $LASTEXITCODE
