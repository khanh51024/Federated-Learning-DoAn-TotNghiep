param(
    [int]$Rounds = 5,
    [int]$NumClients = 5,
    [int]$LocalEpochs = 1,
    [int]$Seed = 42,
    [int]$Port = 8089,
    [double]$MaxUpdateNorm = 5.0,
    [string]$EvaluationDatasetRoot = "",
    [string]$BindHost = "127.0.0.1",
    [switch]$RejectSuspicious,
    [switch]$ObserveOnly,
    [switch]$AllowRemoteClients
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { throw "Không tìm thấy virtualenv Python tại $Python" }

# Mặc định là chính sách AN TOÀN (loại update dị thường). Ablation observe-only phải bật tường minh,
# vì ở chế độ đó server cố ý vẫn trung bình các cập nhật bị nghi ngờ.
if ($ObserveOnly -and $RejectSuspicious) { throw "-ObserveOnly và -RejectSuspicious loại trừ nhau." }
$reject = -not $ObserveOnly
if ($AllowRemoteClients) { $BindHost = "0.0.0.0" }

$argsList = @(
    "-m", "src.federated.server_app",
    "--mode", "deployment",
    "--host", $BindHost,
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
if ($reject) { $argsList += "--reject-suspicious-updates" } else { $argsList += "--observe-only" }
if ($EvaluationDatasetRoot) {
    Write-Host "Held-out evaluator enabled at: $EvaluationDatasetRoot"
    Write-Host "Server will verify every global_val/global_test manifest path before training."
    $argsList += @(
        "--evaluation-dataset-root", $EvaluationDatasetRoot,
        "--require-real-evaluation"
    )
}

Write-Host "Starting REAL Flower server on ${BindHost}:$Port (probe port $($Port + 1))"
Write-Host "Require real clients: ON"
Write-Host "Suspicious updates: $(if ($reject) { 'REJECTED (chinh sach an toan)' } else { 'OBSERVE ONLY - van trung binh, chi ghi nhan' })"
if ($BindHost -eq "0.0.0.0") {
    Write-Warning "Cổng Flower đang mở ra mạng công cộng, không TLS và không xác thực client. Chi dung -AllowRemoteClients khi tren cung may da chot security group/firewall."
}
& $Python @argsList
exit $LASTEXITCODE
