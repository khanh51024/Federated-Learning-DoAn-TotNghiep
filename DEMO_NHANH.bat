@echo off
chcp 65001 >nul
rem Demo nhanh, nhấp đúp trên laptop 1: mở máy chủ trên VM, mở trang web, rồi mở 5 client của máy này.
rem   DEMO_NHANH.bat            : 7 client, 3 vòng (laptop 2 nhấp đúp LAPTOP_2_DEMO_NHANH.bat)
rem   DEMO_NHANH.bat 5 1        : 5 client, 1 vòng, chỉ cần laptop này
rem Cần: khóa SSH (mặc định trên ổ F, đổi bằng biến DEMO_SSH_KEY) và rule Oracle mở cổng 18089-18090.
cd /d "%~dp0"
set CLIENTS=7
set ROUNDS=3
if not "%~1"=="" set CLIENTS=%~1
if not "%~2"=="" set ROUNDS=%~2
set VM=149.118.51.118
set "KEY=F:\khóa pem\ssh-key-2026-10-14.key"
if defined DEMO_SSH_KEY set "KEY=%DEMO_SSH_KEY%"
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not exist "%KEY%" (
  echo [LỖI] Không thấy khóa SSH: %KEY%
  echo Cắm ổ chứa khóa, hoặc đặt biến DEMO_SSH_KEY trỏ tới khóa.
  goto :fail
)
echo [1/3] Kiểm tra nối tới máy chủ VM...
ssh -i "%KEY%" -o BatchMode=yes -o ConnectTimeout=15 ubuntu@%VM% "echo VM-OK"
if errorlevel 1 (
  echo [LỖI] Không SSH được vào VM. Kiểm tra mạng và khóa.
  goto :fail
)

echo [2/3] Mở máy chủ: %CLIENTS% client, %ROUNDS% vòng. Cửa sổ "Máy chủ trên VM" hiện từng vòng.
start "Máy chủ trên VM" cmd %KEEP% ssh -t -i "%KEY%" ubuntu@%VM% "bash $(systemctl show cropfed-demo -p WorkingDirectory --value)/scripts/remote/run_network_server.sh --public --clients %CLIENTS% --rounds %ROUNDS%"

set /a WAITED=0
:wait_port
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $r = $c.BeginConnect('%VM%', 18089, $null, $null); if ($r.AsyncWaitHandle.WaitOne(3000) -and $c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 } finally { $c.Close() }"
if not errorlevel 1 goto :port_open
set /a WAITED+=5
if %WAITED% GEQ 120 (
  echo [LỖI] Sau 2 phút chưa nối được cổng 18089. Xem cửa sổ "Máy chủ trên VM", và kiểm tra rule Oracle 18089-18090.
  goto :fail
)
ping -n 3 127.0.0.1 >nul
goto :wait_port

:port_open
start "" "https://datn.tranngochung.id.vn/#flower"
set /a LOCAL=5
if %CLIENTS% LSS 5 set /a LOCAL=%CLIENTS%
if %CLIENTS% GTR %LOCAL% (
  echo.
  echo [3/3] Máy chủ đã sẵn sàng. BÂY GIỜ nhấp đúp LAPTOP_2_DEMO_NHANH.bat trên laptop 2.
) else (
  echo [3/3] Máy chủ đã sẵn sàng.
)
echo.
call deploy\lan_client\CHAY_CLIENT.bat 0 %LOCAL% %VM%
exit /b %errorlevel%

:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
