@echo off
chcp 65001 >nul
rem Mở các client Flower cùng một lúc trên máy này và nối vào máy chủ.
rem   CHAY_CLIENT.bat                    : một client số 1 (như bản cũ của phép đo 2 laptop)
rem   CHAY_CLIENT.bat 0 5                : 5 client, số 0 đến 4
rem   CHAY_CLIENT.bat 5 5 149.118.51.118 : 5 client, số 5 đến 9, máy chủ ở địa chỉ đã cho (không hỏi lại)
rem   CHAY_CLIENT.bat 0 10 149.118.51.118 tmp\mini10 : thêm thư mục dữ liệu (bộ ảnh nhẹ)
rem Máy chủ chỉ bắt đầu học khi ĐỦ client của cả hai máy đã nối; client nào nối trước sẽ đợi, không ai bị lệch vòng.
cd /d "%~dp0"
set PYTHONUTF8=1
if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=1
set PORT=18089
if defined DEMO_PORT set PORT=%DEMO_PORT%

rem Chạy từ thư mục gốc của dự án (có .venv, ảnh ở tmp\net10) hoặc từ gói đã giải nén (có src ngay cạnh file này)
if not exist src\ if exist ..\..\src\ cd /d "%~dp0..\.."
set DATA=.
if not exist partition\ if exist tmp\net10\partition\ set DATA=tmp\net10
if not "%~4"=="" set "DATA=%~4"
if not exist "%DATA%\partition\" (
  echo [LỖI] Không thấy dữ liệu ở %DATA%\partition.
  goto :fail
)

rem Gói chạy ngay có sẵn Python trong thư mục python; gói nhỏ thì dùng môi trường từ CAI_DAT.bat; trong dự án thì dùng .venv
set "CLIENT_PY="
if exist python\python.exe set "CLIENT_PY=python\python.exe"
if not defined CLIENT_PY if exist venv\Scripts\python.exe set "CLIENT_PY=venv\Scripts\python.exe"
if not defined CLIENT_PY if exist .venv\Scripts\python.exe set "CLIENT_PY=.venv\Scripts\python.exe"
set PYTHONNOUSERSITE=1
if not defined CLIENT_PY (
  echo [LỖI] Chưa cài đặt. Chạy CAI_DAT.bat trước.
  goto :fail
)

set FIRST=1
set COUNT=1
if not "%~1"=="" set FIRST=%~1
if not "%~2"=="" set COUNT=%~2
set /a LAST=FIRST+COUNT-1
if %COUNT% LSS 1 (
  echo [LỖI] Số client phải từ 1 trở lên.
  goto :fail
)
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not "%~3"=="" (
  set "IP=%~3"
  goto :check
)
if defined DEMO_SERVER_IP (
  set "IP=%DEMO_SERVER_IP%"
  goto :check
)

:ask
set IP=
set /p IP=Nhập địa chỉ IP của máy chủ:
if not defined IP goto :ask

:check
echo Đang thử kết nối tới %IP% cổng %PORT%...
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $r = $c.BeginConnect('%IP%', %PORT%, $null, $null); if ($r.AsyncWaitHandle.WaitOne(5000) -and $c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 } finally { $c.Close() }"
if not errorlevel 1 goto :connected
echo.
echo [KHÔNG NỐI ĐƯỢC] Kiểm tra lần lượt:
echo   1. Máy chủ đã chạy và đang chờ client chưa?
echo   2. Địa chỉ IP có gõ đúng không?
echo   3. Cổng %PORT% đã được mở cho IP của máy này chưa? (máy chủ ở xa thường chặn cổng lạ)
echo   4. Wifi trường hay công ty hay chặn: thử bắt điểm phát từ điện thoại.
if not "%~3"=="" goto :fail
if defined DEMO_SERVER_IP goto :fail
goto :ask

:connected
echo Đã nối được. Mở %COUNT% client, số %FIRST% đến %LAST%...
for /l %%i in (%FIRST%,1,%LAST%) do call :launch %%i
echo.
echo Các client đã mở trong %COUNT% cửa sổ riêng. Máy chủ sẽ bắt đầu khi đủ client.
echo Cửa sổ này đợi tới khi các client học xong (tối đa 3 giờ).
set /a WAITED=0
:wait_done
set /a DONE=0
for /l %%i in (%FIRST%,1,%LAST%) do if exist done_client_%%i.flag set /a DONE+=1
if %DONE% GEQ %COUNT% goto :all_done
if %WAITED% GEQ 10800 (
  echo [LỖI] Sau 3 giờ vẫn chưa xong. Xem lỗi trong các cửa sổ client.
  goto :fail
)
ping -n 6 127.0.0.1 >nul
set /a WAITED+=5
goto :wait_done

:all_done
set /a BAD=0
for /l %%i in (%FIRST%,1,%LAST%) do findstr /b /c:"fail" done_client_%%i.flag >nul 2>&1 && set /a BAD+=1
for /l %%i in (%FIRST%,1,%LAST%) do del done_client_%%i.flag >nul 2>&1
if %BAD% GTR 0 (
  echo [LỖI] %BAD% client dừng vì có lỗi. Xem dòng lỗi trong các cửa sổ client.
  goto :fail
)
echo.
echo XONG. Cả %COUNT% client đã học hết các vòng. Xem kết quả trên máy chủ.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 0

:launch
set "ID=0%1"
set "ID=%ID:~-2%"
if exist done_client_%1.flag del done_client_%1.flag
start "Client %1" cmd %KEEP% ""%CLIENT_PY%" -m src.federated.client_app --client-id %1 --server %IP%:%PORT% --partition-dir %DATA%\partition --dataset-root %DATA%\images_client_%ID% --require-real-data --local-epochs 1 --seed 42 && (echo ok> done_client_%1.flag) || (echo fail> done_client_%1.flag & timeout /t 120)"
exit /b 0

:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
