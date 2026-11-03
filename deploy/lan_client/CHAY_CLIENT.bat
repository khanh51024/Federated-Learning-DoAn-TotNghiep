@echo off
rem Laptop 2: noi vao may chu tren laptop 1 va tham gia hoc voi vai tro client 1.
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PORT=18089

rem Goi chay ngay co san Python trong thu muc python; goi nho thi dung moi truong tu CAI_DAT.bat
set "CLIENT_PY=venv\Scripts\python.exe"
if exist python\python.exe set "CLIENT_PY=python\python.exe"
set PYTHONNOUSERSITE=1
if not exist "%CLIENT_PY%" (
  echo [LOI] Chua cai dat. Chay CAI_DAT.bat truoc.
  goto :fail
)

if defined DEMO_SERVER_IP (
  set "IP=%DEMO_SERVER_IP%"
  goto :check
)

:ask
set IP=
set /p IP=Nhap dia chi IP hien tren man hinh laptop 1:
if not defined IP goto :ask

:check
echo Dang thu ket noi toi %IP% cong %PORT%...
powershell -NoProfile -Command "$c = New-Object Net.Sockets.TcpClient; try { $r = $c.BeginConnect('%IP%', %PORT%, $null, $null); if ($r.AsyncWaitHandle.WaitOne(5000) -and $c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 } finally { $c.Close() }"
if not errorlevel 1 goto :connected
echo.
echo [KHONG NOI DUOC] Kiem tra lan luot:
echo   1. Laptop 1 da chay DO_2_MAY.bat va dang cho laptop 2 chua?
echo   2. IP co go dung khong?
echo   3. Hai laptop co cung mot wifi khong?
echo   4. Wifi truong hay cong ty thuong chan: cho ca hai laptop bat diem phat tu dien thoai,
echo      roi chay lai DO_2_MAY.bat vi IP se doi.
if defined DEMO_SERVER_IP goto :fail
goto :ask

:connected
echo Da noi duoc. Bat dau hoc...
chcp 65001 >nul
"%CLIENT_PY%" -m src.federated.client_app --client-id 1 --server %IP%:%PORT% --partition-dir partition --dataset-root images_client_01 --require-real-data --local-epochs 1 --seed 42
if errorlevel 1 (
  echo [LOI] Client dung vi co loi. Xem dong loi o tren.
  goto :fail
)
echo.
echo XONG. Xem ket qua tren laptop 1.
if not defined DEMO_NOPAUSE pause
exit /b 0

:fail
if not defined DEMO_NOPAUSE pause
exit /b 1
