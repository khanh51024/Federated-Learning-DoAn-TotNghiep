param(
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [ValidateSet("label100","label1","label01","quantity01","label_quantity01")]
    [string]$Scenario = "label100",
    [int]$Rounds = 5,
    [int]$LocalEpochs = 1,
    [int]$Seed = 42,
    [int]$Port = 8089,
    [double]$MaxUpdateNorm = 5.0,
    [switch]$RejectSuspicious,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { throw "Không tìm thấy virtualenv Python tại $Python" }

$PartitionDir = "data/partitions_stage2_scratch_v3/$Scenario"
$Metrics = "artifacts/metrics/network_flower_history.json"
$Checkpoint = "artifacts/checkpoints/federated_network_global.pt"
$LogDir = "artifacts/network_run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$serverArgs = @(
    "-m", "src.federated.server_app",
    "--mode", "deployment",
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--num-clients", "5",
    "--rounds", "$Rounds",
    "--local-epochs", "$LocalEpochs",
    "--seed", "$Seed",
    "--require-real-clients",
    "--evaluation-dataset-root", "$DatasetRoot",
    "--evaluation-partition-dir", "$PartitionDir",
    "--require-real-evaluation",
    "--max-update-norm", "$MaxUpdateNorm",
    "--metrics-file", "$Metrics",
    "--checkpoint-file", "$Checkpoint"
)
if ($RejectSuspicious) { $serverArgs += "--reject-suspicious-updates" } else { $serverArgs += "--observe-only" }

$clientCommands = @()
for ($cid = 0; $cid -lt 5; $cid++) {
    $clientCommands += ,@(
        "-m", "src.federated.client_app",
        "--client-id", "$cid",
        "--server", "127.0.0.1:$Port",
        "--partition-dir", "$PartitionDir",
        "--dataset-root", "$DatasetRoot",
        "--local-epochs", "$LocalEpochs",
        "--seed", "$Seed",
        "--fault-mode", "normal",
        "--require-real-data"
    )
}

Write-Host "=== Scientific 5-client Flower Network ==="
Write-Host "Dataset: $DatasetRoot"
Write-Host "Scenario: $Scenario | Rounds: $Rounds | LocalEpochs: $LocalEpochs | Seed: $Seed"
Write-Host "Flower: 127.0.0.1:$Port | Probe: $($Port + 1)"

if ($DryRun) {
    Write-Host "DRY-RUN: không khởi động process."
    Write-Host ("SERVER: `"{0}`" {1}" -f $Python, ($serverArgs -join " "))
    for ($cid = 0; $cid -lt 5; $cid++) {
        Write-Host ("CLIENT {0}: `"{1}`" {2}" -f $cid, $Python, ($clientCommands[$cid] -join " "))
    }
    exit 0
}

& $Python scripts/preflight_plantvillage.py $DatasetRoot --full-path-check
if ($LASTEXITCODE -ne 0) { throw "Dataset preflight failed." }

$serverOut = Join-Path $LogDir "server.out.log"
$serverErr = Join-Path $LogDir "server.err.log"
$server = Start-Process -FilePath $Python -ArgumentList $serverArgs -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr -PassThru
Start-Sleep -Seconds 3
if ($server.HasExited) { throw "Flower server exited early. Xem $serverErr" }

$clients = @()
for ($cid = 0; $cid -lt 5; $cid++) {
    $out = Join-Path $LogDir ("client_{0:D2}.out.log" -f $cid)
    $err = Join-Path $LogDir ("client_{0:D2}.err.log" -f $cid)
    $clients += Start-Process -FilePath $Python -ArgumentList $clientCommands[$cid] -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
}

try {
    foreach ($p in $clients) { $p.WaitForExit() }
    $server.WaitForExit()

    foreach ($p in $clients) {
        if ($p.ExitCode -ne 0) { throw "Một Flower client thất bại (PID=$($p.Id), exit=$($p.ExitCode)). Xem $LogDir" }
    }
    if ($server.ExitCode -ne 0) { throw "Flower server thất bại (exit=$($server.ExitCode)). Xem $serverErr" }

    & $Python scripts/validate_network_artifact.py --metrics $Metrics --checkpoint $Checkpoint --expected-clients 5 --expected-rounds $Rounds
    if ($LASTEXITCODE -ne 0) { throw "Network artifact validation failed." }

    & $Python scripts/promote_verified_network_run.py --metrics $Metrics --checkpoint $Checkpoint --expected-clients 5 --expected-rounds $Rounds --tag $Scenario
    if ($LASTEXITCODE -ne 0) { throw "Verified network promotion failed." }

    Write-Host "PASS: 1 server + 5 real clients + held-out evaluation + validation + verified promotion."
}
finally {
    foreach ($p in $clients) { if (!$p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }
    if (!$server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
}
