@echo off
chcp 65001 >nul
rem Demo nhanh: laptop 2 chỉ mở client 5 và 6 (7 client cả hai máy, 3 vòng) rồi nối vào máy chủ trên VM.
call "%~dp0CHAY_CLIENT.bat" 5 2 149.118.51.118
