@echo off
chcp 65001 >nul
rem Chạy MỘT LẦN trên mỗi laptop. Sau đó nút "Mở client trên máy này" trên trang web mở được client của máy này.
rem Chỉ ghi vào phần đăng ký của tài khoản đang dùng, không cần quyền quản trị. Muốn gỡ thì chạy GO_NUT_WEB.bat.
rem Lệnh đăng ký không nhận gì từ trang web: bấm nút chỉ mở đúng các client đã ghi ở đây, nối vào đúng máy chủ đã ghi ở đây.
set VM=149.118.51.118
if exist "%~dp0src\" (
  set FIRST=5
  set COUNT=2
  set ROLE=Laptop 2: client 5 và 6
) else (
  set FIRST=0
  set COUNT=5
  set ROLE=Laptop 1: client 0 đến 4
)
if not exist "%~dp0CHAY_CLIENT.bat" (
  echo [LỖI] Phải đặt file này cạnh CHAY_CLIENT.bat.
  goto :fail
)
set "BASE=HKCU\Software\Classes\cropfed-demo"
reg add "%BASE%" /ve /t REG_SZ /d "URL:CropFed demo" /f >nul || goto :reg_fail
reg add "%BASE%" /v "URL Protocol" /t REG_SZ /d "" /f >nul || goto :reg_fail
reg add "%BASE%\shell\open\command" /ve /t REG_SZ /d "cmd /c \"\"%~dp0CHAY_CLIENT.bat\" %FIRST% %COUNT% %VM%\"" /f >nul || goto :reg_fail
echo Đã cài xong. %ROLE%.
echo Từ giờ bấm "Mở client trên máy này" trên trang web, trình duyệt hỏi thì chọn Mở.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
exit /b 0

:reg_fail
echo [LỖI] Không ghi được vào phần đăng ký của Windows.
:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
exit /b 1
