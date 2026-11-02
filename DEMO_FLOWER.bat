@echo off
rem Demo Flower mot lan bam: 1 may chu va 2 client tren may nay, roi in bang ket qua. Huong dan: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
set PORT=18089
set ROUNDS=2
rem Mo cua so giu lai de xem log; khi kiem thu tu dong dat DEMO_NOPAUSE=1 de cua so tu dong
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not exist "%PY%" (
  echo [LOI] Khong thay %PY%. File nay phai nam o thu muc goc cua du an.
  goto :fail
)
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  echo [LOI] Cong %PORT% dang ban: con cua so may chu Flower cu dang chay.
  echo Hay dong cac cua so "May chu" va "Client" cu roi chay lai.
  goto :fail
)
if not exist tmp\speedtest\partition (
  echo Dang chuan bi anh demo, lan dau mat khoang 1 phut...
  "%PY%" -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10
  if errorlevel 1 (
    echo [LOI] Khong chuan bi duoc anh demo.
    goto :fail
  )
)
if not exist tmp\demo mkdir tmp\demo
if exist tmp\demo\history.json del tmp\demo\history.json

echo ============================================================
echo  DEMO HOC LIEN KET TREN FLOWER: 1 may chu, 2 client, %ROUNDS% vong
echo  Moi client chi hoc tren anh cua minh va chi gui trong so.
echo ============================================================
echo.
echo [1/3] Mo may chu Flower...
start "May chu Flower" cmd %KEEP% "%PY% -m src.federated.server_app --mode deployment --host 127.0.0.1 --port %PORT% --rounds %ROUNDS% --num-clients 2 --local-epochs 1 --seed 42 --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file tmp/demo/history.json --checkpoint-file tmp/demo/global.pt"

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
echo [2/3] Mo 2 client...
start "Client 0" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 0 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42"
start "Client 1" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 1 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_01 --require-real-data --local-epochs 1 --seed 42"

echo [3/3] Dang hoc, thuong mat khoang 1 phut. Theo doi o 3 cua so vua mo...
set /a WAITED=0
:wait_done
if exist tmp\demo\history.json goto :done
if %WAITED% GEQ 300 (
  echo [LOI] Sau 5 phut chua xong. Xem loi trong cac cua so vua mo; thieu RAM thi dong bot Chrome, game roi chay lai.
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_done

:done
rem May chu co the con dang ghi tep: cho them mot chut roi moi doc
ping -n 3 127.0.0.1 >nul
echo.
echo ================= KET QUA =================
"%PY%" scripts/demo_show.py flower --history tmp/demo/history.json
if errorlevel 1 (
  echo [LOI] Khong doc duoc ket qua.
  goto :fail
)
echo.
echo Xong. Dong "KHOP" nghia la trong so den may chu van nguyen ven, khong sai bit nao.
if not defined DEMO_NOPAUSE pause
exit /b 0

:fail
if not defined DEMO_NOPAUSE pause
exit /b 1
