@echo off
rem Menu demo 9 muc: bam so, may tu chay lenh va in ket qua. Giai thich tung muc: docs\HUONG_DAN_DEMO_THU_CONG.md
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set OMP_NUM_THREADS=2
set PY=.\.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [LOI] Khong thay %PY%. File nay phai nam o thu muc goc cua du an.
  if not defined DEMO_NOPAUSE pause
  exit /b 1
)

:menu
if defined DEMO_ITEM (
  set ITEM=%DEMO_ITEM%
  goto :dispatch
)
cls
echo ==================== DEMO 9 MUC ====================
echo  1. Gui trong so co bi sai khong
echo  2. So voi hoc tap trung co lech khong
echo  3. Phuong phap kiem tra: co tinh lam sai xem co bat duoc khong
echo  4. Chay nhieu client: xem 5 client trong mot vong
echo  5. Thong so Flower: chay that 1 may chu + 2 client
echo  6. Mo hinh tong so voi mo hinh client
echo  7. Mo code: phan tap trung va phan phan tan
echo  T. Chay thu ca hai cach hoc voi it anh, khoang 2 phut
echo  8. Toc do giua 2 may: so da do ngay 02/11
echo  9. Khac biet giua phan tan va tap trung
echo  0. Thoat
echo ====================================================
set ITEM=
set /p ITEM=Bam so roi Enter: 

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
if "%ITEM%"=="9" goto :item9
if "%ITEM%"=="0" exit /b 0
goto :menu

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
echo Ket qua do that ngay 02/11 giua laptop (client 0) va may thue RTX 3090 (client 1):
"%PY%" scripts/demo_show.py flower --history artifacts/network_speedtest/20261102T143913Z_wan_may_thue/wan_history.json
if errorlevel 1 goto :error
echo.
echo Client 1 nam cung may voi may chu, nen RTT va toc do cua no khong phai duong truyen that.
goto :end

:item9
echo.
"%PY%" scripts/demo_show.py sweep
if errorlevel 1 goto :error
echo.
echo Dung luong mang: gui anh mot lan 1524 MB, phan tan khoang 2510 MB (40 vong, ca di lan ve).
if not defined DEMO_NOPAUSE start "" artifacts\sweep_v4\gap_by_alpha.png
goto :end

:error
echo.
echo [LOI] Lenh vua roi bi loi. Xem dong loi o tren.
if defined DEMO_ITEM exit /b 1

:end
echo.
if defined DEMO_ITEM exit /b 0
pause
goto :menu

:end_nopause
if defined DEMO_ITEM exit /b 0
goto :menu
