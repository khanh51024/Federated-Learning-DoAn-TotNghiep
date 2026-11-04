@echo off
chcp 65001 >nul
rem Laptop 2 trong lần đo 10 client: nhấp đúp file này để mở client 5 đến 9 và nối vào máy chủ trên VM.
call "%~dp0CHAY_CLIENT.bat" 5 5 149.118.51.118
