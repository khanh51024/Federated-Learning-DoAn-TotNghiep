# P0.6B SPEC - num_workers x SLOTS grid + gates (PC, r4, 2026-11-01) - author: DIEU PHOI PC; needs TU VAN PC PASS before any run

ORDER (r2): gates of section 4 run FIRST, for both nw 2 and nw 4 (CPU, also proves Windows spawn works with runpy); the grid of
section 2 runs only if ALL_PASS, so the grid only chooses among already-gated options. A failing gate for one nw removes that nw
from the grid (cells with that nw are not run); G_REPRO failing stops everything.

R4 CHANGES (after the r3 gate run <PC_PATH>/q15proto\gates_b, GATES_B f5250f67, kept as evidence; grid never started):
- r3 result: Central: model_state tensor-equal in ALL pairs (orig vs _pc no-env, nw2 x2, nw4 x2); history_equal false ONLY because
  best_validation_cache paths point into each run's own dir (+ duration). FL nw 2 / 4: rc 1, "Can't pickle local object
  'main.<locals>.client_collate_fn'" (Windows spawn). FL G_REPRO passed. RAM available min 0.93 GiB in all 12 runs while the
  1 GiB x 3-sample safety never fired -> a single-sample reading, cause still unknown.
- Code under test is now pc_r9/p1lo_throttle_thermal_pc_r3.py (b5c34e05; builder build_throttle_pc_r3.py; v1 5fc7e83e and r2
  d8bcd10b kept, never integrated). Only when PC_NUM_WORKERS > 0 and a loader's collate_fn is LOCAL: it is swapped for the
  module-level twin pc_r9/pc_r9_collate.client_collate_fn (5a0bdf00, same source as the q15 / q17 / q18 runners) iff
  closure-free, no defaults, same torch module, and the code object is equal recursively (bytecode, consts incl. inner
  comprehensions, names, varnames, arg counts, co_flags with bit 0x10 CO_NESTED masked - r2 refused the real collate for that bit
  alone). Otherwise that loader keeps num_workers 0 (skipped_unpicklable). Checked on the 3 runners' real code objects (True) and
  3 negatives (changed index, added dtype, closure: refused).
- Gates (now <PC_PATH>/q15proto\gates_b2): history comparison maps per-run output paths to <OUTPUT_PATH> (not results); FL G_DET also
  needs PC_RUNTIME stats collate_swapped >= 1 and skipped_unpicklable == 0 in both runs (the FL train loaders really use workers).
- Grid reads <PC_PATH>/q15proto\gates_b2\GATES_B.json. Rules of section 3 unchanged.

Why: stage A (P0_6_PROBE_SPEC 591f41da) showed GPU util ~20 % at nw 0 (data loading bound). Stage A numbers are REFERENCE ONLY:
its harness set BelowNormal but did not turn EcoQoS off (the scheduler does). This spec replaces the stage-A R-NW rule
("smallest nw with t <= 0.9 t(0)") by a whole-machine throughput rule; that rule was never applied to any data. Performance only:
nothing here touches a study number, gate, endpoint, seed or hyperparameter.

## 1. Code under test (no pinned file changes during the grid)
- pc_r9/p1lo_throttle_thermal_pc.py (built by pc_r9/build_throttle_pc.py from p1lo_throttle_thermal.py f7103fb9, which stays
  unchanged). Only when env PC_NUM_WORKERS is present AND a loader's num_workers is exactly 0 (default or passed) it becomes
  PC_NUM_WORKERS; loaders with num_workers != 0 are kept; sampler / generator / pin_memory / persistent_workers untouched (default
  False). Env absent = no DataLoader patch. Adds: first-batch latency per iterator (spawn overhead, summed), sidecar PC_RUNTIME.json
  in the parent of --output-dir (nw, persistent_workers, own sha, duty, mem fraction, stats).
- Commands: same builders as stage A (r3 cmd_for("data") / q17 kaggle fed_cmd("fedavg_a100_s42")), with the throttle path (cmd[2])
  asserted == pinned f7103fb9 path and replaced by the _pc path. Central: --stop-after-epoch 2; FL: --stop-after-round 2.

## 2. Grid (harness pc_r9/p06b_grid.py, out <PC_PATH>/q15proto\probe_b, create-once)
Cells in this order: (nw, slots) = (0,2) (2,2) (4,2) (0,3) (2,3) (4,3).
slots 2 = 1 Central + 1 FL; slots 3 = 1 Central + 2 FL (two FL jobs, separate out dirs). All jobs of a cell start together.
nw 0 = env PC_NUM_WORKERS ABSENT (default path); nw 2 / 4 = env set. Duty 1.0, mem fraction 0.5, MonitorV2 per job, BelowNormal,
and the scheduler's own eco_optout (laptop_sched_v2s4.eco_optout) applied to every process of every job tree at every sample.
Sampling every 2 s: GPU util / memory.used / temperature, system available RAM, per-job tree RSS, and the time each new
"Epoch NN" / "Round NN" line appears in each run.log (unit end timestamps, +-2 s).
Diagnostic only (stage A had RAM available min 0.93 GiB in all 3 phases, cause unknown, idle 9.09 GiB): RAM timeline per sample and,
at each new minimum, the top-8 processes by PRIVATE bytes; per-job tree private bytes (peak + timeline) and system commit
(GetPerformanceInfo CommitTotal / CommitLimit). Not part of the rule.

## 3. Rule (registered before measuring)
- Cell valid iff every job: progress status PAUSED_CHECKPOINTED, units 1 and 2 present, no thermal STOP, util samples present,
  AND overlap: every job's unit-1 end timestamp <= the earliest job exit of the cell.
- throughput(cell) = 3600 x sum over jobs of 1 / unit_s(unit 1)   [units per hour, whole machine; 1 unit = 1 Central epoch or 1 FL
  round, each one pass over the training data]. unit_s = "Time:" of unit 1; wall-based values are recorded too.
- feasible(cell), 2 slots = valid AND RAM available min > 3 GiB AND GPU memory.used peak < 7168 MiB AND GPU temperature max < 85 C;
  3 slots = valid AND RAM available min > 4 GiB AND GPU memory.used peak < 6656 MiB AND GPU temperature max < 85 C (PLAN_PC_R9 s4).
- SAFETY (r3, gates and grid): RAM available < 1 GiB in 3 consecutive 2-s samples -> kill every job tree; the cell is ABORT_RAM
  (not valid, not feasible), the gate run is ABORT_RAM (its pair fails). No feasible cell -> stop and report; thresholds never relaxed.
- Choice: best = max throughput over feasible cells; candidates = feasible cells with throughput >= 0.95 x best; choose the one with
  the smallest nw, then the smallest slots. Duty = 1.0 if the chosen cell's GPU max < 80 C, else 0.85.
- If the chosen nw is 0 -> no patch is used: drivers keep the pinned throttle, sections 4-5 are skipped, SLOTS = chosen slots.
- No feasible cell -> no decision, report to TU VAN.

## 4. Gates for nw 2 and nw 4, run BEFORE the grid (harness pc_r9/p06b_gates.py, CPU: CUDA_VISIBLE_DEVICES=-1, OMP/MKL 2 threads, at most 2 smokes in parallel, nw 4 runs sequential, out <PC_PATH>/q15proto\gates_b)
Smoke commands from the same builders (Central: r3 cmd_for("data", out, smoke=True); FL: fed_cmd with --mode smoke --rounds 2 and
without --r1-seed-override, like q17_gates).
- G_REPRO: pinned throttle vs _pc with PC_NUM_WORKERS absent: every tensor in the checkpoint_last.pt model state equal, history
  metrics in summary.json equal, resolved_config equal except keys containing "output". Central and FL.
  PC_RUNTIME.json of the env-absent run must show loaders_patched 0.
- G_DET_2 / G_DET_4: _pc with PC_NUM_WORKERS = 2 / 4, each run twice: same equalities. Central and FL. PC_RUNTIME.json must show
  loaders_patched >= 1 (the patch really acted). History comparison drops timing keys (duration / vram / telemetry / time).
- ALL_PASS required. nw > 0 changes the RNG stream vs nw 0 (worker seeds come from the main RNG): allowed for PC tables
  (P40-PC option, PLAN_PC_R9 section 2) but must be deterministic (G_DET).

## 5. Integration after ALL_PASS (builder, PC_CHANGES, TU VAN re-check)
- 9 queue drivers: THROTTLE tuple path + sha -> _pc file (exact anchor, count 1). Then grep the OLD sha of each changed driver in all
  code / locks / queue_append / scheduler; every hit updated by the builder and listed in PC_CHANGES.
- Scheduler v2s4: env.update line gets PC_NUM_WORKERS = chosen nw and Q5_GPU_DUTY = chosen duty (one anchor). SLOTS.txt = chosen slots.
- During the campaign nw / duty are fixed; SLOTS follows PLAN_PC_R9 section 4 (up to 3 when VRAM peak < 6.5 GiB and RAM available
  > 4 GiB; down on guard), checked ~30 min, logged in STATE_PC.

## 6. Output
<PC_PATH>/q15proto\probe_b\PROBE_B.json (create-once, utf-8, read back): per cell jobs (cmd, rc, unit_s, unit end ts, wall, util_mean,
tree RSS peak, PC_RUNTIME stats), cell peaks, valid / feasible / throughput, choice, duty. Last stdout line
{"status", "gates_passed", "receipt"}. Gates: <PC_PATH>/q15proto\gates_b\GATES_B.json, same last line.
