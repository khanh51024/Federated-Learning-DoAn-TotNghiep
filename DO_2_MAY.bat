@echo off
rem Laptop 1: may chu Flower + client 0, cho laptop 2 noi vao, roi in bang toc do. Huong dan: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
set PORT=18089
set OUT=tmp\demo_lan
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not exist "%PY%" (
  echo [LOI] Khong thay %PY%. File nay phai nam o thu muc goc cua du an.
  goto :fail
)
if not exist tmp\speedtest\partition (
  echo [LOI] Chua co anh demo. Chay: .\.venv\Scripts\python.exe -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10
  goto :fail
)
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  echo [LOI] Cong %PORT% dang ban: con cua so may chu Flower cu. Dong cac cua so cu roi chay lai.
  goto :fail
)
if not exist %OUT% mkdir %OUT%
if exist %OUT%\history.json del %OUT%\history.json
if exist %OUT%\stop.flag del %OUT%\stop.flag

echo ============================================================
echo  DO TOC DO GIUA 2 LAPTOP: may nay la may chu va client 0
echo  Laptop 2 la client 1. Hai may phai cung wifi hoac cung diem phat.
echo ============================================================
echo.
if defined DEMO_NOFIREWALL goto :server
echo [1/4] Windows se hoi quyen de mo tam cong. Chon YES.
echo       Cong chi nhan may cung mang va tu dong sau khi do xong, lau nhat 30 phut.
powershell -NoProfile -Command "try { Start-Process powershell -Verb RunAs -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%CD%\scripts\open_demo_port.ps1','-StopFile','%CD%\%OUT%\stop.flag' -ErrorAction Stop } catch { exit 1 }"
if errorlevel 1 (
  echo [LOI] Chua mo duoc cong vi Windows khong duoc cap quyen. Chay lai va chon YES.
  goto :fail
)
powershell -NoProfile -Command "for ($i = 0; $i -lt 15; $i++) { if (Get-NetFirewallRule -DisplayName 'FL demo 2 laptop' -ErrorAction SilentlyContinue) { exit 0 }; Start-Sleep -Seconds 1 }; exit 1"
if errorlevel 1 (
  echo [LOI] Khong thay cong duoc mo. Chay lai va chon YES.
  goto :fail
)

:server
echo [2/4] Mo may chu Flower...
start "May chu Flower" cmd %KEEP% "%PY% -m src.federated.server_app --mode deployment --host 0.0.0.0 --port %PORT% --rounds 3 --num-clients 2 --local-epochs 1 --seed 42 --pretrained-init --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file %OUT%/history.json --checkpoint-file %OUT%/global.pt"
set /a WAITED=0
:wait_server
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 goto :server_ready
if %WAITED% GEQ 120 (
  echo [LOI] Sau 2 phut may chu chua san sang. Xem loi trong cua so "May chu Flower".
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_server

:server_ready
echo [3/4] Mo client 0 tren may nay...
start "Client 0" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 0 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42"
echo.
echo ============================================================
echo  BAY GIO SANG LAPTOP 2: chay CHAY_CLIENT.bat va nhap IP nay:
powershell -NoProfile -Command "Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | ForEach-Object { '      ' + $_.IPv4Address.IPAddress + '   (' + $_.InterfaceAlias + ')' }"
echo ============================================================
echo.
echo [4/4] Dang cho laptop 2 va dang hoc, cho toi da 15 phut...
set /a WAITED=0
:wait_done
if exist %OUT%\history.json goto :done
if %WAITED% GEQ 900 (
  echo [LOI] Sau 15 phut chua xong. Xem phan xu ly loi trong huong dan.
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_done

:done
rem May chu co the con dang ghi tep: cho them mot chut roi moi doc
ping -n 3 127.0.0.1 >nul
if not exist %OUT%\stop.flag type nul > %OUT%\stop.flag
echo.
echo ================= KET QUA 2 LAPTOP =================
"%PY%" scripts/demo_show.py speed --history %OUT%/history.json --names "0=Laptop 1 (cung may chu),1=Laptop 2"
if errorlevel 1 (
  echo [LOI] Khong doc duoc ket qua.
  goto :fail
)
echo.
echo Do tre va toc do mang that la dong Laptop 2. Laptop 1 nam cung may voi may chu nen so mang rat lon.
echo.
"%PY%" scripts/demo_show.py flower --history %OUT%/history.json
if errorlevel 1 (
  echo [LOI] Khong doc duoc ket qua.
  goto :fail
)
if defined DEMO_NOFIREWALL goto :finish
"%PY%" -m scripts.summarize_network_speedtest --run lan=%OUT%/history.json --machines 0=laptop_1_may_chu,1=laptop_2 --label lan_2_laptop --local-epochs 1 --notes "Hai laptop cung mang wifi hoac diem phat dien thoai; client 0 cung may voi may chu; 380 anh moi client, 1 epoch moi vong, 3 vong, seed 42. Chi do toc do." >nul
if errorlevel 1 echo [CANH BAO] Chua luu duoc ban tom tat, nhung ket qua tren van dung.

:finish
echo.
echo Xong. Cong tuong lua da duoc dong.
if not defined DEMO_NOPAUSE pause
exit /b 0

:fail
if exist %OUT% if not exist %OUT%\stop.flag type nul > %OUT%\stop.flag
if not defined DEMO_NOPAUSE pause
exit /b 1
