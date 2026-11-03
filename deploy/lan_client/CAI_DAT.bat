@echo off
chcp 65001 >nul
rem Laptop 2, gói nhỏ: cài môi trường, chỉ cần chạy một lần và cần có internet. Gói chạy ngay thì không cần file này.
cd /d "%~dp0"
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

echo [LỖI] Máy này chưa có Python 3.11, 3.12 hoặc 3.13.
echo Vào https://www.python.org/downloads/release/python-3119/ tải "Windows installer (64-bit)".
echo Khi cài nhớ tick ô "Add python.exe to PATH". Cài xong chạy lại file này.
goto :fail

:found
echo [1/3] Tạo môi trường bằng %PYBOOT%...
if not exist venv\Scripts\python.exe %PYBOOT% -m venv venv
if errorlevel 1 (
  echo [LỖI] Không tạo được môi trường Python.
  goto :fail
)
echo [2/3] Tải và cài thư viện, khoảng 6 phút, cần khoảng 1,2 GB đĩa...
venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo [LỖI] Không tải được torch. Kiểm tra internet rồi chạy lại.
  goto :fail
)
venv\Scripts\python.exe -m pip install --quiet --disable-pip-version-check flwr==1.36.0 grpcio==1.84.0 numpy==2.4.6 pillow==12.3.0 pyyaml==6.0.3 scikit-learn==1.9.1 scipy==1.17.1 pandas==3.0.6 matplotlib==3.11.2
if errorlevel 1 (
  echo [LỖI] Không cài được thư viện. Kiểm tra internet rồi chạy lại.
  goto :fail
)
echo [3/3] Kiểm tra...
venv\Scripts\python.exe -c "import src.federated.client_app; print('Môi trường OK')"
if errorlevel 1 (
  echo [LỖI] Đã cài xong nhưng chưa chạy được client. Chụp màn hình gửi lại.
  goto :fail
)
echo.
echo XONG. Lần sau chỉ cần chạy CHAY_CLIENT.bat.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 0

:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
