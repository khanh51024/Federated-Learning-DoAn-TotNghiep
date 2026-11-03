@echo off
chcp 65001 >nul
rem Laptop 2: nối vào máy chủ trên laptop 1 và tham gia học với vai trò client 1.
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PORT=18089

rem Gói chạy ngay có sẵn Python trong thư mục python; gói nhỏ thì dùng môi trường từ CAI_DAT.bat
set "CLIENT_PY=venv\Scripts\python.exe"
if exist python\python.exe set "CLIENT_PY=python\python.exe"
set PYTHONNOUSERSITE=1
if not exist "%CLIENT_PY%" (
  echo [LỖI] Chưa cài đặt. Chạy CAI_DAT.bat trước.
  goto :fail
)

if defined DEMO_SERVER_IP (
  set "IP=%DEMO_SERVER_IP%"
  goto :check
)

:ask
set IP=
set /p IP=Nhập địa chỉ IP hiện trên màn hình laptop 1:
if not defined IP goto :ask

:check
echo Đang thử kết nối tới %IP% cổng %PORT%...
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $r = $c.BeginConnect('%IP%', %PORT%, $null, $null); if ($r.AsyncWaitHandle.WaitOne(5000) -and $c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 } finally { $c.Close() }"
if not errorlevel 1 goto :connected
echo.
echo [KHÔNG NỐI ĐƯỢC] Kiểm tra lần lượt:
echo   1. Laptop 1 đã chạy DO_2_MAY.bat và đang chờ laptop 2 chưa?
echo   2. IP có gõ đúng không?
echo   3. Hai laptop có cùng một wifi không?
echo   4. Wifi trường hay công ty thường chặn: cho cả hai laptop bắt điểm phát từ điện thoại,
echo      rồi chạy lại DO_2_MAY.bat vì IP sẽ đổi.
if defined DEMO_SERVER_IP goto :fail
goto :ask

:connected
echo Đã nối được. Bắt đầu học...
"%CLIENT_PY%" -m src.federated.client_app --client-id 1 --server %IP%:%PORT% --partition-dir partition --dataset-root images_client_01 --require-real-data --local-epochs 1 --seed 42
if errorlevel 1 (
  echo [LỖI] Client dừng vì có lỗi. Xem dòng lỗi ở trên.
  goto :fail
)
echo.
echo XONG. Xem kết quả trên laptop 1.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 0

:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
