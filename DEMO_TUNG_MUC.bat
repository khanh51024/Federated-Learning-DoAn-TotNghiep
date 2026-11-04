@echo off
chcp 65001 >nul
rem Menu demo: bấm số, máy tự chạy lệnh và in kết quả. Giải thích từng mục: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [LỖI] Không thấy %PY%. File này phải nằm ở thư mục gốc của dự án.
  if not defined DEMO_NOPAUSE (echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
  exit /b 1
)
if defined DEMO_ITEM (
  set "ITEM=%DEMO_ITEM%"
  goto :dispatch
)

:top
cls
echo ================= DEMO ĐỀ TÀI HỌC LIÊN KẾT =================
echo  A. 9 mục của thầy
echo  B. File Word, theo từng nhiệm vụ
echo  0. Thoát
echo ============================================================
set PART=
set /p PART=Chọn A, B hoặc 0 rồi Enter:
if /i "%PART%"=="A" goto :menu_a
if /i "%PART%"=="B" goto :menu_b
if "%PART%"=="0" exit /b 0
goto :top

:menu_a
set BACK=menu_a
cls
echo ==================== A. 9 MỤC CỦA THẦY =====================
echo  1. Gửi trọng số có bị sai không
echo  2. So với học tập trung có lệch không
echo  3. Phương pháp kiểm tra: cố tình làm sai xem có bắt được không
echo  4. Chạy nhiều client: xem 5 client trong một vòng
echo  5. Thông số Flower: chạy thật 1 máy chủ + 2 client trên máy này
echo  6. Mô hình tổng so với mô hình client
echo  7. Mở code: phần tập trung và phần phân tán
echo  T. Chạy thử cả hai cách học với ít ảnh, khoảng 2 phút
echo  8. Tốc độ 2 máy: số đã đo
echo  L. Tốc độ 2 máy: đo trực tiếp với laptop 2
echo  9. Khác biệt giữa phân tán và tập trung
echo  0. Quay lại
echo ============================================================
set ITEM=
set /p ITEM=Bấm số rồi Enter:
if "%ITEM%"=="0" goto :top
goto :dispatch

:menu_b
set BACK=menu_b
cls
echo =================== B. FILE WORD THEO NHIỆM VỤ ==============
echo  Đợt 1  1.1 Tập trung SGD và AdamW     1.2 Flower so với bản tự cài
echo         1.3 Nhật ký từng vòng          1.4 Tập kiểm thử niêm phong
echo  Đợt 2  2.1 Quét mức lệch, biểu đồ     2.2 Các dạng lệch đã chia
echo         2.3 Đo theo từng client
echo  Đợt 3  3.1 Code và test 4 thuật toán  3.1T Chạy thử FedProx, FedBN, 2,5 phút
echo         3.3 Hướng cải tiến
echo  Đợt 4  4.1 Lượng tử hóa               4.2 Nén truyền thông
echo         4.3 Quyền riêng tư             4.4 Demo 2 client thật
echo  Đợt 5  5.1 Bảng và biểu đồ cốt lõi    5.2 Năm trục đánh giá
echo         5.4 Tái lập kết quả
echo  Chưa làm: 3.2, 3.4, 5.3, 5.5. Xem hướng dẫn để biết cách nói.
echo  0. Quay lại
echo ============================================================
set ITEM=
set /p ITEM=Gõ mã nhiệm vụ, ví dụ 2.2, rồi Enter:
if "%ITEM%"=="0" goto :top
goto :dispatch

:dispatch
if "%ITEM%"=="1" goto :item1
if "%ITEM%"=="2" goto :item2
if "%ITEM%"=="3" goto :item3
if "%ITEM%"=="4" goto :item4
if "%ITEM%"=="5" goto :item5
if "%ITEM%"=="6" goto :item6
if "%ITEM%"=="7" goto :item7
if /i "%ITEM%"=="T" goto :itemT
if "%ITEM%"=="8" goto :item8
if /i "%ITEM%"=="L" goto :itemL
if "%ITEM%"=="9" goto :item9
if "%ITEM%"=="1.1" goto :n11
if "%ITEM%"=="1.2" goto :n12
if "%ITEM%"=="1.3" goto :n13
if "%ITEM%"=="1.4" goto :n14
if "%ITEM%"=="2.1" goto :n21
if "%ITEM%"=="2.4" goto :n21
if "%ITEM%"=="2.2" goto :n22
if "%ITEM%"=="2.3" goto :n23
if "%ITEM%"=="3.1" goto :n31
if /i "%ITEM%"=="3.1T" goto :n31t
if "%ITEM%"=="3.3" goto :n33
if "%ITEM%"=="4.1" goto :n41
if "%ITEM%"=="4.2" goto :n41
if "%ITEM%"=="4.3" goto :n43
if "%ITEM%"=="4.4" goto :itemL
if "%ITEM%"=="5.1" goto :n21
if "%ITEM%"=="5.2" goto :n52
if "%ITEM%"=="5.4" goto :n54
if defined DEMO_ITEM (
  echo [LỖI] Không có mục %ITEM%.
  exit /b 1
)
goto :%BACK%

:item1
echo.
echo Mỗi dòng là một lần chạy 40 vòng với 5 client. 400/400 nghĩa là cả 400 lần gửi đều giống hệt.
for %%R in (fedavg_s42_label_alpha_0_1_flower fedavg_s123_label_alpha_0_1_flower fedavg_s2026_label_alpha_0_1_flower fedavg_s42_iid_flower fedavg_s123_iid_flower fedavg_s2026_iid_flower) do (
  "%PY%" scripts/demo_show.py run --run runs/v4_sweep/%%R --every 100 | findstr /C:"/400"
)
echo 6 lần chạy x 400 lần gửi = 2400 lần kiểm tra.
goto :end

:item2
echo.
echo [Bước 1] Hai bên có xuất phát từ cùng một mô hình không:
"%PY%" scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
if errorlevel 1 goto :error
echo.
echo [Bước 2] Cho cùng điều kiện, dòng A; rồi cho mỗi client học nhiều bước, dòng B. Khoảng 1 phút:
"%PY%" scripts/verify_fedsgd_equivalence.py --quick --device cpu
if errorlevel 1 goto :error
echo.
echo [Bước 3] So hai mô hình đã học xong:
"%PY%" scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42
if errorlevel 1 goto :error
goto :end

:item3
echo.
echo Test này cố tình đổi trọng số từ 2.0 thành 2.0000002. Hệ thống phải báo lỗi và dừng lại.
echo PASSED / 1 passed = hệ thống đã bắt được.
"%PY%" -m pytest tests/test_flower_app.py -k changed_weights -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:item4
echo.
echo Vòng đầu và vòng cuối của lần chạy 40 vòng trên GPU, mỗi vòng đủ 5 client:
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 100
if errorlevel 1 goto :error
goto :end

:item5
call "%~dp0DEMO_FLOWER.bat"
goto :end_nopause

:item6
echo.
echo "sai số gộp": sai số của phép gộp. "cách mô hình chung": client khác mô hình tổng bao nhiêu.
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
if errorlevel 1 goto :error
goto :end

:item7
echo.
echo Mở 4 chỗ trong code:
echo   scripts/train_v4.py dòng 189         - tập trung: lấy toàn bộ ảnh
echo   src/federated/flower_app.py dòng 206 - phân tán: gửi trọng số, nhận về, kiểm tra
echo   src/federated/strategy.py dòng 133   - gộp trọng số theo số ảnh
echo   src/training/loop.py dòng 67         - hàm học dùng chung cho cả hai cách
if defined DEMO_NOPAUSE goto :end
where code >nul 2>nul
if errorlevel 1 (
  start "" notepad scripts\train_v4.py
  start "" notepad src\federated\flower_app.py
  goto :end
)
call code -g scripts/train_v4.py:189 -g src/federated/flower_app.py:206 -g src/federated/strategy.py:133 -g src/training/loop.py:67
goto :end

:itemT
echo.
if exist tmp\demo\runs rmdir /s /q tmp\demo\runs
echo [1/2] Học tập trung: gom ảnh lại một chỗ
"%PY%" scripts/train_v4.py --method centralized --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs
if errorlevel 1 goto :error
echo.
echo [2/2] Học phân tán trên Flower: 5 client, cùng lệnh, chỉ khác cách chạy
"%PY%" scripts/train_v4.py --method fedavg --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag flower
if errorlevel 1 goto :error
echo.
echo Đây chỉ là chạy thử với rất ít ảnh để thấy hai cách chạy như thế nào, không dùng số này để báo cáo.
goto :end

:item8
echo.
echo Lần đo 02/11, laptop và máy thuê GPU, qua đường hầm SSH:
"%PY%" scripts/demo_show.py speed --history artifacts/network_speedtest/20261102T143913Z_wan_may_thue/wan_history.json --names "0=Laptop,1=Máy thuê RTX 3090"
if errorlevel 1 goto :error
echo.
echo Lần đo 20/10, laptop và máy Oracle, qua Internet:
"%PY%" scripts/demo_show.py speed --history artifacts/network_speedtest/20261020T170306Z_wan_vm_oracle/wan_history.json --names "0=Laptop,1=Máy Oracle"
if errorlevel 1 goto :error
echo.
echo Máy còn lại nằm cùng máy với máy chủ, nên độ trễ và tốc độ mạng thật là dòng Laptop.
goto :end

:itemL
call "%~dp0DO_2_MAY.bat"
goto :end_nopause

:item9
echo.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Dung lượng mạng: gửi ảnh một lần 1524 MB, phân tán khoảng 2510 MB, 40 vòng, cả đi lẫn về.
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\gap_by_alpha.png
goto :end

:n11
echo.
echo N1.1: hai dòng đầu là Tập trung SGD và AdamW. Chọn SGD làm trần vì cùng cách học với client.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
goto :end

:n12
echo.
echo N1.2: so dòng "Phân tán tự cài" và "Phân tán trên Flower", rất lệch.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Chênh lệch: toàn bộ ảnh 0,11 điểm, ảnh ngoài đồng 0,96 điểm. Chưa thể nói tương đương trên mọi nhóm.
goto :end

:n13
echo.
echo N1.3: nhật ký từng vòng của phân tán:
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 100
if errorlevel 1 goto :error
echo.
echo Nhật ký từng vòng của một client tự học, Local-only đợt cũ chưa có số byte:
"%PY%" scripts/demo_show.py run --run runs/v4b/local_c0_s42 --every 20
if errorlevel 1 goto :error
goto :end

:n14
echo.
echo N1.4: chỉ kiểm tra đủ lần chạy, KHÔNG mở độ chính xác của tập niêm phong:
"%PY%" scripts/run_sealed_test.py --runs runs/v4b --cells "label_alpha_0_1|fedavg" --dry-run
if errorlevel 1 goto :error
echo.
echo Test: tập niêm phong chỉ chạy được một lần, không có đường nào khác đọc được số:
"%PY%" -m pytest tests/test_run_sealed_test.py -k "runs_once or no_other_path" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:n21
echo.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Local-only mới có ở mức rất lệch, α = 0,1.
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\curves_by_alpha.png
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\gap_by_alpha.png
goto :end

:n22
echo.
"%PY%" scripts/demo_show.py partitions
if errorlevel 1 goto :error
echo.
echo Test kiểm tra cách chia:
"%PY%" -m pytest tests/test_extra_partitions.py -k "quantity_scheme or source_skew_gives" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
echo.
echo Hai dạng lệch số lượng và lệch nguồn ảnh mới chia xong dữ liệu, CHƯA train.
goto :end

:n23
echo.
"%PY%" scripts/demo_show.py fairness --split val
if errorlevel 1 goto :error
echo.
"%PY%" scripts/demo_show.py drift
if errorlevel 1 goto :error
goto :end

:n31
echo.
echo N3.1: FedProx, FedBN, SCAFFOLD, MOON nằm trong src/federated/algorithms.py. Chạy test:
"%PY%" -m pytest tests/test_fed_algorithms.py -q --no-header -p no:cacheprovider
if errorlevel 1 goto :error
echo.
echo Mới có code và test, CHƯA có thí nghiệm đầy đủ 3 lần chạy cho các thuật toán này.
if not defined DEMO_NOPAUSE call code -g src/federated/algorithms.py
goto :end

:n31t
echo.
if exist tmp\demo\runs rmdir /s /q tmp\demo\runs
echo [1/2] FedProx trên Flower với rất ít ảnh
"%PY%" scripts/train_v4.py --method fedavg --algo fedprox --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag fedprox
if errorlevel 1 goto :error
echo.
echo [2/2] FedBN trên Flower với rất ít ảnh
"%PY%" scripts/train_v4.py --method fedavg --algo fedbn --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag fedbn
if errorlevel 1 goto :error
echo.
echo Chạy thử để thấy thuật toán chạy được, không dùng số này để báo cáo.
goto :end

:n33
echo.
echo N3.3: hai hướng đã có code: kết hợp FedBN + FedProx, và tinh chỉnh riêng cho từng client.
"%PY%" -m pytest tests/test_compression_privacy.py -k "fedbnprox or finetuning" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:n41
echo.
"%PY%" scripts/demo_show.py compression
if errorlevel 1 goto :error
goto :end

:n43
echo.
echo N4.3 là tùy chọn. Đã có code thêm nhiễu và tính epsilon, CHƯA chạy đường epsilon - độ chính xác:
"%PY%" -m pytest tests/test_compression_privacy.py -k "rdp or clip or noise" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:n52
echo.
echo [1] Độ chính xác và [2] độ bền khi dữ liệu lệch:
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo [3] Chi phí truyền thông:
"%PY%" scripts/demo_show.py compression
if errorlevel 1 goto :error
echo.
echo [4] Công bằng giữa các client:
"%PY%" scripts/demo_show.py fairness --split val
if errorlevel 1 goto :error
echo.
echo [5] Riêng tư: ảnh không rời client, chỉ gửi trọng số. Chưa có bảo đảm DP.
goto :end

:n54
echo.
echo N5.4: mỗi lần chạy lưu cấu hình, seed, phiên bản mã và mã băm dữ liệu:
type runs\v4_sweep\fedavg_s42_label_alpha_0_1_flower\config.json
echo.
echo.
echo Dựng lại mô hình xuất phát từ seed, phải khớp với lần chạy đã lưu:
"%PY%" scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
if errorlevel 1 goto :error
goto :end

:error
echo.
echo [LỖI] Lệnh vừa rồi bị lỗi. Xem dòng lỗi ở trên.
if defined DEMO_ITEM exit /b 1

:end
echo.
if defined DEMO_ITEM exit /b 0
(echo Bấm phím bất kỳ để tiếp tục... & pause >nul)
goto :%BACK%

:end_nopause
if defined DEMO_ITEM exit /b 0
goto :%BACK%
