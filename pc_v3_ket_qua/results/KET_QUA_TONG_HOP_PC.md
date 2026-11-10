# Ket qua tong hop PC (goi web r3, 09-10/11/2026)

Moi so la trung binh ± do lech chuan (SD, ddof 1) qua cac seed neu co seed; lay thang tu file ket qua da duoc TU VAN PASS (khong go tay).
Tap test: PD-172 / PD-463 / PV-legacy la **legacy test da dung 3 lan (Q5 / Q6 / Q11b)**; EXT-FINAL = 440 anh, 7 lop, tap ngoai doc nhat.
Chi so chinh khong doi lua chon: P6 chi bao cao, khong doi lua chon (SCAFFOLD, A1 0/4, S5).

## Cac cau bat buoc
- (i) Lua chon trien khai dua tren validation; tren EXT-FINAL, aug2_lap (0.145) va bal_lap (0.141) cao hon r3data_pc (0.105) co y nghia sau Holm (p_holm 0.045); lua chon khong doi.
- (ii) EXT-FINAL (anh hien truong, ngoai mien): accuracy moi nhom (>= 3 seed) 0.10-0.17 (cat xuong 2 chu so), tuc mo hinh KHONG tong quat hoa tot ra anh hien truong. Nhom mo ta 1 seed: d4_descriptive|fedavg|0.1 0.130; d4_descriptive|p1|0.1 0.132; d4_descriptive|p2|0.1 0.127; d4_descriptive|p3|0.1 0.080; d4_descriptive|scaffold|0.1 0.120.
- (iii) PV-legacy la 'legacy test da dung 3 lan (Q5 / Q6 / Q11b)'; PD-172 va PD-463 cung la tap legacy da lo.
- (iv) int8 (chi lop Linear): |delta accuracy| <= 0.006 tren moi tap so voi fp32 CPU cung mo hinh.
- Giai phap duoc chon: SCAFFOLD (nhanh b), G = 0.9008 ± 0.0157 (3 seed, alpha 0,1) so voi FedAvg 0.8829 ± 0.0034; dG = +0.0179.
- E2 (PD val 269 anh): khong bac bo H0 sau Holm (min p_holm 0.135); khong viet 'khong co khac biet'.
- FedAvg 40 vong truyen 3.1x dung luong anh tho (2510 MB so voi 806 MB); khong phai tiet kiem bang thong, ma vi rieng tu.
- R-DIAG (chan doan mo ta, 3 seed, khong kiem dinh): o alpha 100, lay mau PD theo ti le tu nhien thay vi 0,25 doi PD macro-F1 +0.014 ± 0.016 (2/3 seed duong); FedBN (BN cuc bo) doi -0.003 ± 0.009 (runner q18 so voi baseline runner q17).
- FLW-A: 3/3 seed bit-exact (40 round + checkpoint cuoi); Flower 1.42.0 trong venv rieng; flower_exact = cong tuong duong cua dieu phoi va truyen tai; harness giao DataLoader, RNG va GradScaler (khong phai trien khai Flower doc lap).
- int8 dong chi lop Linear: file TorchScript 4,36 so voi 6,24 MB (-30%), macro-F1 val thay doi <= 0,0015, do tre CPU tuong duong.

## P6 - test cuoi (accuracy, TB ± SD qua seed; EXT-FINAL = 7 lop)
| nhom (phuong phap, alpha) | n seed | PD-172 | PD-463 | PV-legacy | EXT-FINAL |
|---|---|---|---|---|---|
| central|adamw|None | 5 | 0.551 ± 0.036 | 0.523 ± 0.015 | 0.995 ± 0.000 | 0.115 ± 0.016 |
| central|sgd|None | 5 | 0.538 ± 0.019 | 0.527 ± 0.012 | 0.991 ± 0.003 | 0.150 ± 0.011 |
| d4_descriptive|fedavg|0.1 | 1 | 0.436 | 0.408 | 0.939 | 0.130 |
| d4_descriptive|p1|0.1 | 1 | 0.529 | 0.508 | 0.915 | 0.132 |
| d4_descriptive|p2|0.1 | 1 | 0.488 | 0.464 | 0.918 | 0.127 |
| d4_descriptive|p3|0.1 | 1 | 0.436 | 0.428 | 0.928 | 0.080 |
| d4_descriptive|scaffold|0.1 | 1 | 0.465 | 0.449 | 0.934 | 0.120 |
| fl_phase_d|fedavg|qs | 3 | 0.504 ± 0.013 | 0.513 ± 0.019 | 0.981 ± 0.001 | 0.139 ± 0.018 |
| fl_phase_d|fedavg|ss | 3 | 0.525 ± 0.009 | 0.505 ± 0.008 | 0.982 ± 0.002 | 0.145 ± 0.008 |
| fl|fedavg_fp16_uplink|0.1 | 3 | 0.492 ± 0.033 | 0.438 ± 0.020 | 0.941 ± 0.004 | 0.106 ± 0.009 |
| fl|fedavg|0.1 | 5 | 0.477 ± 0.028 | 0.438 ± 0.019 | 0.940 ± 0.004 | 0.112 ± 0.009 |
| fl|fedavg|0.5 | 5 | 0.456 ± 0.023 | 0.493 ± 0.013 | 0.973 ± 0.002 | 0.163 ± 0.012 |
| fl|fedavg|1.0 | 3 | 0.486 ± 0.027 | 0.515 ± 0.009 | 0.978 ± 0.001 | 0.170 ± 0.010 |
| fl|fedavg|10.0 | 3 | 0.502 ± 0.007 | 0.512 ± 0.009 | 0.980 ± 0.001 | 0.141 ± 0.018 |
| fl|fedavg|100.0 | 3 | 0.525 ± 0.027 | 0.528 ± 0.016 | 0.981 ± 0.001 | 0.161 ± 0.019 |
| fl|fedbn|0.1 | 5 | 0.453 ± 0.022 | 0.418 ± 0.007 | 0.917 ± 0.005 | 0.110 ± 0.018 |
| fl|fedbn|0.5 | 5 | 0.456 ± 0.024 | 0.493 ± 0.018 | 0.966 ± 0.001 | 0.155 ± 0.010 |
| fl|fedprox|0.1 | 5 | 0.484 ± 0.022 | 0.440 ± 0.016 | 0.941 ± 0.002 | 0.111 ± 0.008 |
| fl|fedprox|0.5 | 5 | 0.463 ± 0.025 | 0.497 ± 0.018 | 0.974 ± 0.002 | 0.163 ± 0.012 |
| fl|moon|0.1 | 5 | 0.484 ± 0.019 | 0.448 ± 0.023 | 0.931 ± 0.011 | 0.109 ± 0.007 |
| fl|moon|0.5 | 5 | 0.481 ± 0.029 | 0.511 ± 0.014 | 0.970 ± 0.004 | 0.152 ± 0.016 |
| fl|scaffold_bn|0.1 | 3 | 0.469 ± 0.007 | 0.443 ± 0.011 | 0.914 ± 0.006 | 0.107 ± 0.008 |
| fl|scaffold_bn|0.5 | 3 | 0.436 ± 0.010 | 0.500 ± 0.013 | 0.960 ± 0.006 | 0.167 ± 0.014 |
| fl|scaffold|0.1 | 5 | 0.487 ± 0.025 | 0.464 ± 0.021 | 0.944 ± 0.026 | 0.122 ± 0.012 |
| fl|scaffold|0.5 | 5 | 0.484 ± 0.013 | 0.498 ± 0.021 | 0.981 ± 0.002 | 0.159 ± 0.011 |
| rdiag_descriptive|fedavg_pd_natural|100.0 | 3 | 0.541 ± 0.029 | 0.539 ± 0.028 | 0.987 ± 0.003 | 0.128 ± 0.007 |
| rdiag_descriptive|fedbn|100.0 | 3 | 0.500 ± 0.036 | 0.522 ± 0.014 | 0.979 ± 0.002 | 0.155 ± 0.026 |

Deploy (EXT-FINAL, Holm 4 phep; p tho / p_holm): r3data_pc vs Q5 ref 0.229 / 0.459; vs R4 per-leaf (n=86) 0.453 / 0.459;
vs aug2_lap 0.0114 / 0.0454; vs bal_lap 0.0139 / 0.0454. Bang chi tiet: results/p6/P6_RESULT.json.

## R-DIAG (validation, alpha 100, 3 seed, mo ta)
| nhom | endpoint | PD macro-F1 | PD accuracy | d endpoint vs baseline | d PD macro-F1 |
|---|---|---|---|---|---|
| FedAvg alpha 100 (baseline q17) | 0.9403 ± 0.0007 | 0.4840 ± 0.0158 | 0.4738 ± 0.0168 | | |
| PD lay mau tu nhien | 0.9495 ± 0.0013 | 0.4984 ± 0.0116 | 0.4899 ± 0.0080 | +0.0092 ± 0.0018 | +0.0144 ± 0.0161 |
| FedBN (mo hinh server, runner q18) | 0.9362 ± 0.0015 | 0.4811 ± 0.0067 | 0.4655 ± 0.0050 | -0.0041 ± 0.0013 | -0.0029 ± 0.0092 |

## E2 / E3 / E1 / E4 / E5 / FLW
- E2, E3: results/e2_e3/ (E2_RESULT.json, E3_RESULT.json, E3_confusion_top10.csv). Truyen thong: 6,275,944 B/client/huong/vong; SCAFFOLD them 6,227,224 B moi huong.
- E1 (cong bang giua client, alpha curve) va E1 phase D: results/e1/, results/e1_phase_d/ (cot mean_over_seeds, sd_over_seeds).
- E4 (drift, theo round, D5): results/e4/.
- E5 (int8): results/e5/E5_RESULT.json; static PTQ khong dua len web (macro-F1 0,03-0,06); mo hinh int8 trong int8_dynamic_linear/.
- FLW: FLW-A: 3/3 seed bit-exact (40 round + checkpoint cuoi); Flower 1.42.0 trong venv rieng; flower_exact = cong tuong duong cua dieu phoi va truyen tai; harness giao DataLoader, RNG va GradScaler (khong phai trien khai Flower doc lap).

## Pham vi ket luan
Test cuoi la mo ta; mo hinh KHONG tong quat hoa tot ra anh hien truong (EXT-FINAL). 24/38 lop khong co anh train PlantDoc. Khong claim chi tiet detector (human_GT = 0).
Khong co anh nao, ten file anh EXT, hay logit test trong goi nay.
