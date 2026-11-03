# Mo tam cong tuong lua cho demo 2 laptop, chi nhan may cung mang, roi tu dong dong.
# Chay voi quyen Administrator (DO_2_MAY.bat tu xin). Dong khi co StopFile hoac het Minutes.
param(
    [Parameter(Mandatory = $true)][string]$StopFile,
    [int[]]$Ports = @(18089, 18090),
    [int]$Minutes = 30
)

# Ten rieng cho moi lan chay, de lan cu het han khong dong cong cua lan moi
$ruleName = "FL demo 2 laptop $PID"
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Ports -RemoteAddress LocalSubnet -Action Allow -Profile Any | Out-Null
try {
    $deadline = (Get-Date).AddMinutes($Minutes)
    while (-not (Test-Path $StopFile) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
    }
}
finally {
    Remove-NetFirewallRule -DisplayName $ruleName
}
