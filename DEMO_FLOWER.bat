@echo off
chcp 65001 >nul
rem Demo Flower một lần bấm: 1 máy chủ và 2 client trên máy này, rồi in bảng kết quả. Hướng dẫn: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
set PORT=18089
set ROUNDS=3
rem Mở cửa sổ giữ lại để xem log; khi kiểm thử tự động đặt DEMO_NOPAUSE=1 để cửa sổ tự đóng
set KEEP=/k
if defined DEMO_NOPAUSE set KEEP=/c

if not exist "%PY%" (
  echo [LỖI] Không thấy %PY%. File này phải nằm ở thư mục gốc của dự án.
  goto :fail
)
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  echo [LỖI] Cổng %PORT% đang bận: còn cửa sổ máy chủ Flower cũ đang chạy.
  echo Hãy đóng các cửa sổ "Máy chủ" và "Client" cũ rồi chạy lại.
  goto :fail
)
if not exist tmp\speedtest\partition (
  echo Đang chuẩn bị ảnh demo, lần đầu mất khoảng 1 phút...
  "%PY%" -m scripts.prepare_network_speedtest --out tmp/speedtest --per-class 10
  if errorlevel 1 (
    echo [LỖI] Không chuẩn bị được ảnh demo.
    goto :fail
  )
)
if not exist tmp\demo mkdir tmp\demo
if exist tmp\demo\history.json del tmp\demo\history.json

echo ============================================================
echo  DEMO HỌC LIÊN KẾT TRÊN FLOWER: 1 máy chủ, 2 client, %ROUNDS% vòng
echo  Mỗi client chỉ học trên ảnh của mình và chỉ gửi trọng số.
echo ============================================================
echo.
echo [1/3] Mở máy chủ Flower...
start "Máy chủ Flower" cmd %KEEP% "%PY% -m src.federated.server_app --mode deployment --host 127.0.0.1 --port %PORT% --rounds %ROUNDS% --num-clients 2 --local-epochs 1 --seed 42 --pretrained-init --require-real-clients --reject-suspicious-updates --max-update-norm 500 --evaluation-partition-dir tmp/speedtest/partition --probe-payload-bytes 6275944 --metrics-file tmp/demo/history.json --checkpoint-file tmp/demo/global.pt"

set /a WAITED=0
:wait_server
netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 goto :server_ready
if %WAITED% GEQ 120 (
  echo [LỖI] Sau 2 phút máy chủ chưa sẵn sàng. Xem lỗi trong cửa sổ "Máy chủ Flower".
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_server

:server_ready
echo [2/3] Mở 2 client...
start "Client 0" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 0 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_00 --require-real-data --local-epochs 1 --seed 42"
start "Client 1" cmd %KEEP% "%PY% -m src.federated.client_app --client-id 1 --server 127.0.0.1:%PORT% --partition-dir tmp/speedtest/partition --dataset-root tmp/speedtest/images_client_01 --require-real-data --local-epochs 1 --seed 42"

echo [3/3] Đang học, thường mất khoảng 1,5 phút. Theo dõi ở 3 cửa sổ vừa mở...
set /a WAITED=0
:wait_done
if exist tmp\demo\history.json goto :done
if %WAITED% GEQ 300 (
  echo [LỖI] Sau 5 phút chưa xong. Xem lỗi trong các cửa sổ vừa mở; thiếu RAM thì đóng bớt Chrome, game rồi chạy lại.
  goto :fail
)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
goto :wait_done

:done
rem Máy chủ có thể còn đang ghi tệp: chờ thêm một chút rồi mới đọc
ping -n 3 127.0.0.1 >nul
echo.
echo ================= KẾT QUẢ =================
"%PY%" scripts/demo_show.py flower --history tmp/demo/history.json
if errorlevel 1 (
  echo [LỖI] Không đọc được kết quả.
  goto :fail
)
echo.
echo Xong. Dòng "KHỚP" nghĩa là trọng số đến máy chủ vẫn nguyên vẹn, không sai bit nào.
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 0

:fail
if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
exit /b 1
