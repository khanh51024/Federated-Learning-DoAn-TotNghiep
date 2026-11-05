@echo off
chcp 65001 >nul
rem Chạy MỘT LẦN trên mỗi laptop. Sau đó nút "Mở client trên máy này" trên trang web mở được client của máy này.
rem Chỉ ghi vào phần đăng ký của tài khoản đang dùng, không cần quyền quản trị. Muốn gỡ thì chạy GO_NUT_WEB.bat.
rem Lệnh đăng ký không nhận gì từ trang web: bấm nút chỉ mở đúng các client đã ghi ở đây, nối vào đúng máy chủ đã ghi ở đây.
rem Còn cài client_agent.py chạy nền, tự chạy lại mỗi lần đăng nhập: máy tự hiện trên trang và tự mở client khi quản trị viên chọn máy này.
rem Web chỉ giao được số client đầu và số lượng, máy chủ đo và thư mục ảnh vẫn là các giá trị ghi ở đây.
set VM=149.118.51.118
if not defined CROPFED_WEB set "CROPFED_WEB=https://datn.tranngochung.id.vn"
if exist "%~dp0src\" (
  set FIRST=5
  set COUNT=2
  set ROLE=Laptop 2: client 5 và 6
  set "LIGHT=mini10"
) else (
  set FIRST=0
  set COUNT=5
  set ROLE=Laptop 1: client 0 đến 4
  set "LIGHT=tmp\mini10"
)
if not exist "%~dp0CHAY_CLIENT.bat" (
  echo [LỖI] Phải đặt file này cạnh CHAY_CLIENT.bat.
  goto :fail
)
set "BASE=HKCU\Software\Classes\cropfed-demo"
reg add "%BASE%" /ve /t REG_SZ /d "URL:CropFed demo" /f >nul || goto :reg_fail
reg add "%BASE%" /v "URL Protocol" /t REG_SZ /d "" /f >nul || goto :reg_fail
reg add "%BASE%\shell\open\command" /ve /t REG_SZ /d "cmd /c \"\"%~dp0CHAY_CLIENT.bat\" %FIRST% %COUNT% %VM%\"" /f >nul || goto :reg_fail
rem Thêm một nút: 5 client bằng bộ ảnh nhẹ, chỉ cần một laptop.
reg add "%BASE%-nhe" /ve /t REG_SZ /d "URL:CropFed demo bo anh nhe" /f >nul || goto :reg_fail
reg add "%BASE%-nhe" /v "URL Protocol" /t REG_SZ /d "" /f >nul || goto :reg_fail
reg add "%BASE%-nhe\shell\open\command" /ve /t REG_SZ /d "cmd /c \"\"%~dp0CHAY_CLIENT.bat\" 0 5 %VM% %LIGHT%\"" /f >nul || goto :reg_fail
rem Python không có cửa sổ để chạy nền: trong gói nằm ở python, trong dự án nằm ở .venv.
set "AGENT_DIR=%~dp0python"
if not exist "%AGENT_DIR%\pythonw.exe" set "AGENT_DIR=%~dp0..\..\.venv\Scripts"
if not exist "%AGENT_DIR%\pythonw.exe" goto :no_python
set PYTHONNOUSERSITE=1
"%AGENT_DIR%\python.exe" "%~dp0client_agent.py" --install --web %CROPFED_WEB% --server %VM% --data %LIGHT%
if errorlevel 1 goto :fail
echo Đã cài xong. %ROLE%.
echo Từ giờ máy tự nối với trang mỗi lần đăng nhập Windows.
echo Lần đo bộ ảnh nhẹ: quản trị viên chọn máy này thì client tự mở, không hỏi lại. Lần đo khác: bấm "Mở client trên máy này" trên trang.
echo Muốn tắt hẳn: chạy "%~dp0GO_NUT_WEB.bat"
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
exit /b 0

:no_python
echo [LỖI] Không thấy Python để chạy nền ở "%AGENT_DIR%".
goto :fail

:reg_fail
echo [LỖI] Không ghi được vào phần đăng ký của Windows.
:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
exit /b 1
