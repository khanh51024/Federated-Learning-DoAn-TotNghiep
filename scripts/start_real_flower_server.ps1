param(
    [int]$Rounds = 5,
    [int]$NumClients = 5,
    [int]$LocalEpochs = 1,
    [int]$Seed = 42,
    [int]$Port = 8089,
    [double]$MaxUpdateNorm = 5.0,
    [string]$EvaluationDatasetRoot = "",
    [switch]$RejectSuspicious
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { throw "Không tìm thấy virtualenv Python tại $Python" }

$argsList = @(
    "-m", "src.federated.server_app",
    "--mode", "deployment",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--num-clients", "$NumClients",
    "--rounds", "$Rounds",
    "--local-epochs", "$LocalEpochs",
    "--seed", "$Seed",
    "--require-real-clients",
    "--max-update-norm", "$MaxUpdateNorm",
    "--metrics-file", "artifacts/metrics/network_flower_history.json",
    "--checkpoint-file", "artifacts/checkpoints/federated_network_global.pt"
)
if ($EvaluationDatasetRoot) {
    Write-Host "Held-out evaluator enabled at: $EvaluationDatasetRoot"
    Write-Host "Server will verify every global_val/global_test manifest path before training."
    $argsList += @(
        "--evaluation-dataset-root", $EvaluationDatasetRoot,
        "--require-real-evaluation"
    )
}
if ($RejectSuspicious) { $argsList += "--reject-suspicious-updates" }

Write-Host "Starting REAL Flower server on 0.0.0.0:$Port (probe port $($Port + 1))"
Write-Host "Require real clients: ON"
Write-Host "Reject suspicious: $($RejectSuspicious.IsPresent)"
& $Python @argsList
exit $LASTEXITCODE
