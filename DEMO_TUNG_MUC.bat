@echo off
rem Menu demo: bam so, may tu chay lenh va in ket qua. Giai thich tung muc: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [LOI] Khong thay %PY%. File nay phai nam o thu muc goc cua du an.
  if not defined DEMO_NOPAUSE pause
  exit /b 1
)
if defined DEMO_ITEM (
  set "ITEM=%DEMO_ITEM%"
  goto :dispatch
)

:top
chcp 437 >nul
cls
echo ================= DEMO DE TAI HOC LIEN KET =================
echo  A. 9 muc cua thay
echo  B. File Word, theo tung nhiem vu
echo  0. Thoat
echo ============================================================
set PART=
set /p PART=Chon A, B hoac 0 roi Enter:
if /i "%PART%"=="A" goto :menu_a
if /i "%PART%"=="B" goto :menu_b
if "%PART%"=="0" exit /b 0
goto :top

:menu_a
chcp 437 >nul
set BACK=menu_a
cls
echo ==================== A. 9 MUC CUA THAY =====================
echo  1. Gui trong so co bi sai khong
echo  2. So voi hoc tap trung co lech khong
echo  3. Phuong phap kiem tra: co tinh lam sai xem co bat duoc khong
echo  4. Chay nhieu client: xem 5 client trong mot vong
echo  5. Thong so Flower: chay that 1 may chu + 2 client tren may nay
echo  6. Mo hinh tong so voi mo hinh client
echo  7. Mo code: phan tap trung va phan phan tan
echo  T. Chay thu ca hai cach hoc voi it anh, khoang 2 phut
echo  8. Toc do 2 may: so da do
echo  L. Toc do 2 may: do truc tiep voi laptop 2
echo  9. Khac biet giua phan tan va tap trung
echo  0. Quay lai
echo ============================================================
set ITEM=
set /p ITEM=Bam so roi Enter:
if "%ITEM%"=="0" goto :top
goto :dispatch

:menu_b
chcp 437 >nul
set BACK=menu_b
cls
echo =================== B. FILE WORD THEO NHIEM VU ==============
echo  Dot 1  1.1 Tap trung SGD va AdamW     1.2 Flower so voi ban tu cai
echo         1.3 Nhat ky tung vong          1.4 Tap kiem thu niem phong
echo  Dot 2  2.1 Quet muc lech, bieu do     2.2 Cac dang lech da chia
echo         2.3 Do theo tung client
echo  Dot 3  3.1 Code va test 4 thuat toan  3.1T Chay thu FedProx, FedBN, 2 phut
echo         3.3 Huong cai tien
echo  Dot 4  4.1 Luong tu hoa               4.2 Nen truyen thong
echo         4.3 Quyen rieng tu             4.4 Demo 2 client that
echo  Dot 5  5.1 Bang va bieu do cot loi    5.2 Nam truc danh gia
echo         5.4 Tai lap ket qua
echo  Chua lam: 3.2, 3.4, 5.3, 5.5. Xem huong dan de biet cach noi.
echo  0. Quay lai
echo ============================================================
set ITEM=
set /p ITEM=Go ma nhiem vu, vi du 2.2, roi Enter:
if "%ITEM%"=="0" goto :top
goto :dispatch

:dispatch
chcp 65001 >nul
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
  echo [LOI] Khong co muc %ITEM%.
  exit /b 1
)
goto :%BACK%

:item1
echo.
echo Moi dong la mot lan chay 40 vong voi 5 client. 400/400 nghia la ca 400 lan gui deu giong het.
for %%R in (fedavg_s42_label_alpha_0_1_flower fedavg_s123_label_alpha_0_1_flower fedavg_s2026_label_alpha_0_1_flower fedavg_s42_iid_flower fedavg_s123_iid_flower fedavg_s2026_iid_flower) do (
  "%PY%" scripts/demo_show.py run --run runs/v4_sweep/%%R --every 100 | findstr /C:"/400"
)
echo 6 lan chay x 400 lan gui = 2400 lan kiem tra.
goto :end

:item2
echo.
echo [Buoc 1] Hai ben co xuat phat tu cung mot mo hinh khong:
"%PY%" scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
if errorlevel 1 goto :error
echo.
echo [Buoc 2] Cho cung dieu kien (dong A), roi cho moi client hoc nhieu buoc (dong B). Khoang 1 phut:
"%PY%" scripts/verify_fedsgd_equivalence.py --quick --device cpu
if errorlevel 1 goto :error
echo.
echo [Buoc 3] So hai mo hinh da hoc xong:
"%PY%" scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42
if errorlevel 1 goto :error
goto :end

:item3
echo.
echo Test nay co tinh doi trong so tu 2.0 thanh 2.0000002. He thong phai bao loi va dung lai.
echo PASSED / 1 passed = he thong da bat duoc.
"%PY%" -m pytest tests/test_flower_app.py -k changed_weights -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:item4
echo.
echo Vong dau va vong cuoi cua lan chay 40 vong tren GPU, moi vong du 5 client:
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 100
if errorlevel 1 goto :error
goto :end

:item5
call "%~dp0DEMO_FLOWER.bat"
goto :end_nopause

:item6
echo.
echo "sai so gop": sai so cua phep gop. "cach mo hinh chung": client khac mo hinh tong bao nhieu.
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 20
if errorlevel 1 goto :error
goto :end

:item7
echo.
echo Mo 4 cho trong code:
echo   scripts/train_v4.py dong 189         - tap trung: lay toan bo anh
echo   src/federated/flower_app.py dong 206 - phan tan: gui trong so, nhan ve, kiem tra
echo   src/federated/strategy.py dong 133   - gop trong so theo so anh
echo   src/training/loop.py dong 67         - ham hoc dung chung cho ca hai cach
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
echo [1/2] Hoc tap trung: gom anh lai mot cho
"%PY%" scripts/train_v4.py --method centralized --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs
if errorlevel 1 goto :error
echo.
echo [2/2] Hoc phan tan tren Flower: 5 client, cung lenh, chi khac cach chay
"%PY%" scripts/train_v4.py --method fedavg --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag flower
if errorlevel 1 goto :error
echo.
echo Day chi la chay thu voi rat it anh de thay hai cach chay nhu the nao, khong dung so nay de bao cao.
goto :end

:item8
echo.
echo Lan do 02/11, laptop va may thue GPU, qua duong ham SSH:
"%PY%" scripts/demo_show.py speed --history artifacts/network_speedtest/20261102T143913Z_wan_may_thue/wan_history.json --names "0=Laptop,1=May thue RTX 3090"
if errorlevel 1 goto :error
echo.
echo Lan do 20/10, laptop va may Oracle, qua Internet:
"%PY%" scripts/demo_show.py speed --history artifacts/network_speedtest/20261020T170306Z_wan_vm_oracle/wan_history.json --names "0=Laptop,1=May Oracle"
if errorlevel 1 goto :error
echo.
echo May con lai nam cung may voi may chu, nen do tre va toc do mang that la dong Laptop.
goto :end

:itemL
call "%~dp0DO_2_MAY.bat"
goto :end_nopause

:item9
echo.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Dung luong mang: gui anh mot lan 1524 MB, phan tan khoang 2510 MB (40 vong, ca di lan ve).
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\gap_by_alpha.png
goto :end

:n11
echo.
echo N1.1: hai dong dau la Tap trung SGD va AdamW. Chon SGD lam tran vi cung cach hoc voi client.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
goto :end

:n12
echo.
echo N1.2: so dong "Phan tan tu cai" va "Phan tan tren Flower", rat lech.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Chenh lech: toan bo anh 0.11 diem, anh ngoai dong 0.96 diem. Chua the noi tuong duong tren moi nhom.
goto :end

:n13
echo.
echo N1.3: nhat ky tung vong cua phan tan:
"%PY%" scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower --every 100
if errorlevel 1 goto :error
echo.
echo Nhat ky tung vong cua mot client tu hoc (Local-only, dot cu chua co so byte):
"%PY%" scripts/demo_show.py run --run runs/v4b/local_c0_s42 --every 20
if errorlevel 1 goto :error
goto :end

:n14
echo.
echo N1.4: chi kiem tra du run, KHONG mo do chinh xac cua tap niem phong:
"%PY%" scripts/run_sealed_test.py --runs runs/v4b --cells "label_alpha_0_1|fedavg" --dry-run
if errorlevel 1 goto :error
echo.
echo Test: tap niem phong chi chay duoc mot lan, khong co duong nao khac doc duoc so:
"%PY%" -m pytest tests/test_run_sealed_test.py -k "runs_once or no_other_path" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:n21
echo.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Local-only moi co o muc rat lech (alpha 0,1).
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\curves_by_alpha.png
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\gap_by_alpha.png
goto :end

:n22
echo.
"%PY%" scripts/demo_show.py partitions
if errorlevel 1 goto :error
echo.
echo Test kiem tra cach chia:
"%PY%" -m pytest tests/test_extra_partitions.py -k "quantity_scheme or source_skew_gives" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
echo.
echo Hai dang lech so luong va lech nguon anh moi chia xong du lieu, CHUA train.
goto :end

:n23
echo.
"%PY%" scripts/demo_show.py fairness
if errorlevel 1 goto :error
goto :end

:n31
echo.
echo N3.1: FedProx, FedBN, SCAFFOLD, MOON nam trong src/federated/algorithms.py. Chay test:
"%PY%" -m pytest tests/test_fed_algorithms.py -q --no-header -p no:cacheprovider
if errorlevel 1 goto :error
echo.
echo Moi co code va test, CHUA co thi nghiem day du 3 lan chay cho cac thuat toan nay.
if not defined DEMO_NOPAUSE call code -g src/federated/algorithms.py
goto :end

:n31t
echo.
if exist tmp\demo\runs rmdir /s /q tmp\demo\runs
echo [1/2] FedProx tren Flower voi rat it anh
"%PY%" scripts/train_v4.py --method fedavg --algo fedprox --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag fedprox
if errorlevel 1 goto :error
echo.
echo [2/2] FedBN tren Flower voi rat it anh
"%PY%" scripts/train_v4.py --method fedavg --algo fedbn --runtime flower --seed 42 --limit 200 --budget 2 --workers 0 --device cpu --batch-size 16 --out tmp/demo/runs --tag fedbn
if errorlevel 1 goto :error
echo.
echo Chay thu de thay thuat toan chay duoc, khong dung so nay de bao cao.
goto :end

:n33
echo.
echo N3.3: hai huong da co code: ket hop FedBN + FedProx, va tinh chinh rieng cho tung client.
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
echo N4.3 la tuy chon. Da co code them nhieu va tinh epsilon, CHUA chay duong epsilon - do chinh xac:
"%PY%" -m pytest tests/test_compression_privacy.py -k "rdp or clip or noise" -v --no-header -p no:cacheprovider
if errorlevel 1 goto :error
goto :end

:n52
echo.
echo [1] Do chinh xac va [2] do ben khi du lieu lech:
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo [3] Chi phi truyen thong:
"%PY%" scripts/demo_show.py compression
if errorlevel 1 goto :error
echo.
echo [4] Cong bang giua cac client:
"%PY%" scripts/demo_show.py fairness
if errorlevel 1 goto :error
echo.
echo [5] Rieng tu: anh khong roi client, chi gui trong so. Chua co bao dam DP.
goto :end

:n54
echo.
echo N5.4: moi lan chay luu cau hinh, seed, phien ban ma va ma bam du lieu:
type runs\v4_sweep\fedavg_s42_label_alpha_0_1_flower\config.json
echo.
echo.
echo Dung lai mo hinh xuat phat tu seed, phai khop voi lan chay da luu:
"%PY%" scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower
if errorlevel 1 goto :error
goto :end

:error
echo.
echo [LOI] Lenh vua roi bi loi. Xem dong loi o tren.
if defined DEMO_ITEM exit /b 1

:end
echo.
if defined DEMO_ITEM exit /b 0
pause
goto :%BACK%

:end_nopause
if defined DEMO_ITEM exit /b 0
goto :%BACK%
