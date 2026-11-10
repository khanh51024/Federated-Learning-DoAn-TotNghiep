# P0.6 PROBE SPEC (PC, r1, 2026-11-01) - author: DIEU PHOI PC; needs TU VAN PC PASS before any run

Purpose: wall-time probe on the PC (RTX 2060 SUPER 8 GiB, i5-13400F, 15.8 GiB RAM) to fix ONCE, before the first measured run:
GPU duty, DataLoader num_workers (nw), SLOTS; and to estimate hours for P1-P3. Probe outputs are NEVER numbers of the study.

## 1. What runs (stage A: pinned code, nw = 0, no code change)
Harness: plans/cai-thien-train-20261015/pc_r9/p06_probe.py (sha in the PASS request). It does NOT call driver main(); it loads the
pinned driver modules with importlib, re-checks every pin the driver main() checks, and builds the command with the driver's OWN
command builder, then only (i) redirects --output-dir to <PC_PATH>/q15proto\probe_pc\<job>\run and (ii) appends one stop flag.
- A1 Central: r3_laptop_driver_pinned.cmd_for("data", out, smoke=False) + ["--stop-after-epoch", "1"]
  (R3 spec + EXTRA_R3DATA = the first PC deploy run; batch 32, full images, AMP, same sampler/transform).
- A2 FedAvg: q17_laptop_kaggle.fed_cmd("fedavg_a100_s42", out) with the q17 driver's throttle insertion + ["--stop-after-round", "2"]
  (full mode, --r1-seed-override, partition root v5; Q17_LOCAL_TEST unset).
- A3 = A1 and A2 started together (2 slots), new out dirs.
Order: A1, A2, A3. Each job: through p1lo_throttle_thermal.py (pinned f7103fb9), env Q5_GPU_DUTY=1.0, Q5_GPU_MEM_FRACTION=0.5,
thermal monitor = thermal_monitor_v2.MonitorV2 (pinned b1b622d5, the drivers' own pause/stop rule), BelowNormal priority.
Out root <PC_PATH>/q15proto\probe_pc (create-once). Real out dirs (r3l, r3xl, q15l, q16l, q17l, q18l, n11l, lo17l) must not exist after.

## 2. What is measured (sampled every 2 s for the whole job)
nvidia-smi: utilization.gpu, memory.used (whole GPU), temperature.gpu. psutil: system available RAM; RSS of each job's process tree
(parent + all children, recursive). From run.log: per-epoch "Time: Xs" (Central) and per-round "Time: Xs" (FedAvg); job wall time; rc.
util_mean(job) = mean of utilization.gpu over samples from the first sample with util >= 10 to job exit (startup excluded).

## 3. Decision rules (registered before measuring)
- R-VALID: a job is valid iff run/progress.json status == "PAUSED_CHECKPOINTED" AND the log has epoch 1 (A1) / rounds 1 and 2 (A2)
  AND no thermal STOP flag was raised (rc is recorded, not used: the trainers' exit code on a budget pause is not registered). Any invalid job -> no decision, report to TU VAN (one infra rerun allowed per PLAN_PC_R9 section 4).
- R-DUTY: if max GPU temperature over A1, A2, A3 < 80 C -> duty 1.0 for every PC run; else 0.85.
- R-NW: if min(util_mean(A1), util_mean(A2)) >= 90 -> nw = 0 is final (a gain >= 10 % is impossible when the GPU is busy >= 90 % of
  the time), stage B is NOT run, no code changes, G-REPRO not needed (default path untouched). Else -> stage B (separate spec, PASS
  needed): builder copies in <PC_PATH>/q15proto with nw in {2, 4}; choose nw > 0 only if t(nw) <= 0.9 * t(0) AND 2-slot available RAM min
  > 3 GiB; smallest such nw; then G-REPRO (nw 0 byte-equal original) + determinism (chosen nw twice, tensor-equal) on 1 Central and
  1 FL driver.
- R-SLOTS: SLOTS = 2 iff in A3 available RAM min > 3 GiB AND GPU memory.used peak < 7168 MiB; else SLOTS = 1. SLOTS 3 is not decided
  here (not measured).
- Estimate (not a rule): Central run h = 35 x epoch_s(A1); FL run h = 40 x mean(round_s rounds 1-2 of A2); 2-slot factor =
  A3 times / alone times. Local-only: "unknown" (not probed). Written to PROBE_PC.json and STATE_PC.json.

## 4. Output
<PC_PATH>/q15proto\probe_pc\PROBE_PC.json (create-once, utf-8, read back): per job cmd, pins, rc, wall, epoch/round times, util_mean,
peaks (VRAM, temp, tree RSS), RAM available min, and the decisions R-DUTY / R-NW / R-SLOTS computed by the harness from section 3.
Last stdout line: {"status", "gates_passed", "receipt"}.
