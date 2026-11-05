@echo off
chcp 65001 >nul
rem Gỡ phần CAI_NUT_WEB.bat đã cài và dừng tiến trình nền. Chạy sau khi demo xong.
reg delete "HKCU\Software\Classes\cropfed-demo" /f >nul 2>&1
reg delete "HKCU\Software\Classes\cropfed-demo-nhe" /f >nul 2>&1
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v CropFedClient /f >nul 2>&1
if exist "%LOCALAPPDATA%\CropFedClient\agent.lock" type nul > "%LOCALAPPDATA%\CropFedClient\agent.stop"
echo Đã gỡ nút mở client của máy này. Máy sẽ biến khỏi trang sau khoảng 20 giây.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
