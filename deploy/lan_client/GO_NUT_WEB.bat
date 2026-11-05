@echo off
chcp 65001 >nul
rem Gỡ phần CAI_NUT_WEB.bat đã cài. Chạy sau khi demo xong.
reg delete "HKCU\Software\Classes\cropfed-demo" /f >nul 2>&1
reg delete "HKCU\Software\Classes\cropfed-demo-nhe" /f >nul 2>&1
echo Đã gỡ nút mở client của máy này.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để đóng... & pause >nul)
