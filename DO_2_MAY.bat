@echo off
chcp 65001 >nul
rem Laptop 1: máy chủ Flower + client 0, chờ laptop 2 nối vào, rồi in bảng tốc độ. Hướng dẫn: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
set PORT=18089
set OUT=tmp\demo_lan
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not exist "%PY%" (
  echo [LỖI] Không thấy %PY%. File này phải nằm ở thư mục gốc của dự án.
  goto :fail
)
if not exist tmp\speedtest\partition (
  echo [LỖI] Chưa có ảnh demo. Chạy: .\.venv\Scripts\python.exe -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10
  goto :fail
)
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  echo [LỖI] Cổng %PORT% đang bận: còn cửa sổ máy chủ Flower cũ. Đóng các cửa sổ cũ rồi chạy lại.
  goto :fail
)
if not exist %OUT% mkdir %OUT%
if exist %OUT%\history.json del %OUT%\history.json
if exist %OUT%\stop.flag del %OUT%\stop.flag

echo ============================================================
echo  ĐO TỐC ĐỘ GIỮA 2 LAPTOP: máy này là máy chủ và client 0
echo  Laptop 2 là client 1. Hai máy phải cùng wifi hoặc cùng điểm phát.
echo ============================================================
echo.
if defined DEMO_NOFIREWALL goto :server
echo [1/4] Windows sẽ hỏi quyền để mở tạm cổng. Chọn YES.
echo       Cổng chỉ nhận máy cùng mạng và tự đóng sau khi đo xong, lâu nhất 30 phút.
powershell -NoProfile -Command "try { Start-Process powershell -Verb RunAs -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%CD%\scripts\open_demo_port.ps1','-StopFile','%CD%\%OUT%\stop.flag' -ErrorAction Stop } catch { exit 1 }"
if errorlevel 1 (
  echo [LỖI] Chưa mở được cổng vì Windows không được cấp quyền. Chạy lại và chọn YES.
  goto :fail
)
powershell -NoProfile -Command "for ($i = 0; $i -lt 15; $i++) { if (Get-NetFirewallRule -DisplayName 'FL demo 2 laptop*' -ErrorAction SilentlyContinue) { exit 0 }; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
  echo [LỖI] Không thấy cổng được mở. Chạy lại và chọn YES.
  goto :fail
)

:server
echo [2/4] Mở máy chủ Flower...
start "Máy chủ Flower" cmd %KEEP% "%PY% -m src.federated.server_app --mode deployment --host 0.0.0.0 --port %PORT% --rounds 3 --num-clients 2 --local-epochs 1 --seed 42 --pretrained-init --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file %OUT%/history.json --checkpoint-file %OUT%/global.pt"
set /a WAITED=0
:wait_server
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 goto :server_ready
if %WAITED% GEQ 120 (
  echo [LỖI] Sau 2 phút máy chủ chưa sẵn sàng. Xem lỗi trong cửa sổ "Máy chủ Flower".
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_server

:server_ready
echo [3/4] Mở client 0 trên máy này...
start "Client 0" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 0 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42"
echo.
echo ============================================================
echo  BÂY GIỜ SANG LAPTOP 2: chạy CHAY_CLIENT.bat và nhập IP này:
powershell -NoProfile -Command "Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | ForEach-Object { '      ' + $_.IPv4Address.IPAddress + '   (' + $_.InterfaceAlias + ')' }"
echo ============================================================
echo.
echo [4/4] Đang chờ laptop 2 và đang học, chờ tối đa 15 phút...
set /a WAITED=0
:wait_done
if exist %OUT%\history.json goto :done
if %WAITED% GEQ 900 (
  echo [LỖI] Sau 15 phút chưa xong. Xem phần xử lý lỗi trong hướng dẫn.
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_done

:done
rem Máy chủ có thể còn đang ghi tệp: chờ thêm một chút rồi mới đọc
ping -n 3 127.0.0.1 >nul
if not exist %OUT%\stop.flag type nul > %OUT%\stop.flag
echo.
echo ================= KẾT QUẢ 2 LAPTOP =================
"%PY%" scripts/demo_show.py speed --history %OUT%/history.json --names "0=Laptop 1 (cùng máy chủ),1=Laptop 2"
if errorlevel 1 (
  echo [LỖI] Không đọc được kết quả.
  goto :fail
)
echo.
echo Độ trễ và tốc độ mạng thật là dòng Laptop 2. Laptop 1 nằm cùng máy với máy chủ nên số mạng rất lớn.
echo.
"%PY%" scripts/demo_show.py flower --history %OUT%/history.json
if errorlevel 1 (
  echo [LỖI] Không đọc được kết quả.
  goto :fail
)
if defined DEMO_NOFIREWALL goto :finish
"%PY%" -m scripts.summarize_network_speedtest --run lan=%OUT%/history.json --machines 0=laptop_1_may_chu,1=laptop_2 --label lan_2_laptop --local-epochs 1 --notes "Hai laptop cùng mạng wifi hoặc điểm phát điện thoại; client 0 cùng máy với máy chủ; 380 ảnh mỗi client, 1 epoch mỗi vòng, 3 vòng, seed 42. Chỉ đo tốc độ." >nul
if errorlevel 1 echo [CẢNH BÁO] Chưa lưu được bản tóm tắt, nhưng kết quả trên vẫn đúng.

:finish
echo.
echo Xong. Cổng tường lửa đã được đóng.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 0

:fail
if exist %OUT% if not exist %OUT%\stop.flag type nul > %OUT%\stop.flag
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
