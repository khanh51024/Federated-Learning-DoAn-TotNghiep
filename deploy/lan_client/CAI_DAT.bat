@echo off
rem Laptop 2: cai moi truong, chi can chay mot lan va can co internet.
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYBOOT=

py -3.11 --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=py -3.11"
if defined PYBOOT goto :found
py -3.12 --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=py -3.12"
if defined PYBOOT goto :found
py -3.13 --version >nul 2>nul
if not errorlevel 1 set "PYBOOT=py -3.13"
if defined PYBOOT goto :found
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 set "PYBOOT=python"
if defined PYBOOT goto :found

echo [LOI] May nay chua co Python 3.11, 3.12 hoac 3.13.
echo Vao https://www.python.org/downloads/release/python-3119/ tai "Windows installer (64-bit)".
echo Khi cai nho tick o "Add python.exe to PATH". Cai xong chay lai file nay.
goto :fail

:found
echo [1/3] Tao moi truong bang %PYBOOT%...
if not exist venv\Scripts\python.exe %PYBOOT% -m venv venv
if errorlevel 1 (
  echo [LOI] Khong tao duoc moi truong Python.
  goto :fail
)
echo [2/3] Tai va cai thu vien, khoang 5 den 10 phut, can khoang 2 GB dia...
venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo [LOI] Khong tai duoc torch. Kiem tra internet roi chay lai.
  goto :fail
)
venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check flwr==1.36.0 grpcio==1.84.0 numpy==2.4.6 pillow==12.3.0 pyyaml==6.0.3 scikit-learn==1.9.1 scipy==1.17.1 pandas==3.0.6 matplotlib==3.11.2
if errorlevel 1 (
  echo [LOI] Khong cai duoc thu vien. Kiem tra internet roi chay lai.
  goto :fail
)
echo [3/3] Kiem tra...
venv\Scripts\python.exe -c "import src.federated.client_app; print('Moi truong OK')"
if errorlevel 1 (
  echo [LOI] Da cai xong nhung chua chay duoc client. Chup man hinh gui lai.
  goto :fail
)
echo.
echo XONG. Lan sau chi can chay CHAY_CLIENT.bat.
if not defined DEMO_NOPAUSE pause
exit /b 0

:fail
if not defined DEMO_NOPAUSE pause
exit /b 1
