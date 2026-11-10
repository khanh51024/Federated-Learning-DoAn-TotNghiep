# FLW_SPEC addendum A1 (PC, 2026-11-09 ~09:55) - gate path after G_SMOKE_CPU att2 FAIL; base FLW_SPEC r2 1e9872a7 (unchanged)

Decided by TU VAN PC, registered BEFORE any FLW-A data exists. ASCII. FLW_SPEC r2 is not edited.

## 1. Record of the CPU gate
- G_SMOKE_CPU att1 FAIL (flw/smoke_cpu/G_SMOKE_CPU.json de931a98; harness flw_exact.py 7ef110a3): first diff round 1 client_00 delta_l2.
  Cause found by the thread diagnostic (THREADS_ref cdbdebab, THREADS_flower 5cd53089): Ray actor 2 intra-op threads + OMP_NUM_THREADS=2
  vs 10 / unset on ref and server.
- Fix (one change, TU VAN PASS): flw_exact_r3.py 978c143e hands the server thread config to the client. Docstring erratum: it cites
  "THREADS_flower cdbdebab"; correct shas are THREADS_ref cdbdebab, THREADS_flower 5cd53089 (file not edited).
- G_SMOKE_CPU att2 FAIL (flw/smoke_cpu_att2/G_SMOKE_CPU_att2.json afa3658d): threads applied 10/10, OMP unset; round 1 delta_l2 and
  loss equal at printed precision for all 5 clients, weight fingerprint differs for clients 0-3 (equal for client 4); round 2 delta_l2
  ~1e-5; round 3 |d val_macro_f1_raw| 0.00136; checkpoint max abs diff 0.00448. ref att1 vs ref att2 bit-identical
  (REF_att1_vs_att2_descriptive.json f3948b70).
- No further CPU debugging (time-box). Report wording (mandatory): "duong Flower tren CPU khong bit-exact giua cac tien trinh; sau khi
  sua thread van con sai lech thu tu float o muc bit, nguyen nhan chua xac dinh; khong run bao cao nao dung CPU". Never "CPU gate PASS".

## 2. Why FLW-A still opens
In FLW_SPEC r2, G_SMOKE_CPU is a preliminary harness gate, not a conclusion rule. The extra condition "CPU att2 PASS" was TU VAN's own
(message item 4), not in the spec; it is dropped here, with this reason, before any FLW-A data. FLW-A runs on GPU; G_SMOKE_GPU att1
(harness 7ef110a3, diagnostic) gave rows 1-3 equal vs archived pc_fedavg_a05_s42 (f7dd01a2). The FLW-A rule of FLW_SPEC r2 s4
(bit-exact / equivalent 1e-6 / NOT EQUIVALENT) is unchanged and is reported as it comes out. No claim rests on the CPU gate.

## 3. Order (one detached window, SLOTS 1, scheduler idle)
a) G_SMOKE_GPU att2 with harness r3 978c143e, same command as att1 (receipt command of pc_fedavg_a05_s42 + --stop-after-round 3).
   PASS iff history rows 1-3 equal vs archived s42 (flw_compare).
b) PASS -> FLW-A s42 -> s43 -> s44 with r3.
c) FAIL -> FLW-A s42 -> s43 -> s44 with flw_exact.py 7ef110a3 (the harness that passed G_SMOKE_GPU att1). Both attempts reported.
   No other fallback.
d) Per seed: exact receipt command of pc_fedavg_a05_s<seed> with only --output-dir replaced (flw/flwa_s<seed>/run), ETA rule
   (now + 2 h <= 2026-11-10 20:00) and stop flag checked before each START, flw_compare vs archived after each seed. rc != 0 -> stop,
   no retry.

## 4. Cut order C after FLW-A (~5 h, done ~15:00-15:30)
A3 (~0.3 h) next. flower_stock and FLW-B only if time remains before 2026-11-10 08:00, and only after the P6 r2 spec has TU VAN PASS.
