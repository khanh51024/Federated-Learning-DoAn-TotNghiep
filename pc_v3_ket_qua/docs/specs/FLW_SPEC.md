# FLW_SPEC (PC, r2, 2026-11-09; r1 9598ac63 + TU VAN review items 1-5) - FedAvg on Flower (N1.2) + A3 (FedProx / FedBN on Flower) - DIEU PHOI PC r3; TU VAN PASS needed

Source: PROMPT_DIEU_PHOI_R3 3fc66ff8 s3 + s6 step 9, PROMPT_DIEU_PHOI_R3_ADDENDUM_FLW (FLW-A mandatory, FLW-B recommended),
PROMPT_DIEU_PHOI_R3_ADDENDUM_3_YEUCAU A3 + C (cut order R-DIAG -> flower_stock -> A3 -> A2 -> S5 s46 -> FLW-B; never FLW-A).
User decision 2026-11-07: new separate venv, flwr 1.42.0 + Ray. Web app untouched; no Flower checkpoint is packaged.

## 0. Deviation from the prompt (stated first)
The prompt names p4_draft/p4c_flower.py 0ff6a082 + p4c_driver.py a3469de4. They are NOT on this PC (laptop-only). The copy here,
r8_draft/p4c_flower_q15.py d526d1ee, reseeds every (round, client) with 42 + 1000 r + k, so by its own docstring it can NOT
reproduce the runner trajectory, and the FLW-A gate (flower_exact == pc_fedavg_a05_s42-44, bit-exact or |d endpoint| <= 1e-6)
needs exactly that. Hence a new harness, pc_r9/flw_exact.py (section 2), same Message-API design as P4-C.

## 1. Environment (built 2026-11-09 04:27, ledger)
experiments/train-improve-20261015/flw/venv_flwr: Python 3.11.9 venv, pip install flwr[simulation]==1.42.0 numpy==2.4.6
pillow==12.3.0 (ray 2.55.1 pulled = P4-C pin). torch 2.6.0+cu124 / torchvision 0.21.0+cu124 = the main .venv build, reached
READ-ONLY by flw/venv_flwr/Lib/site-packages/zz_main_venv_torch.pth (one line: main .venv site-packages; added after pip, so
the venv's own packages win). Receipts: flw/pip_freeze_flw.txt, flw/pyvenv_flw.cfg. No other venv touched. GPU RTX 2060 SUPER.

## 2. Harness flw_exact.py (flower_exact)
Server process = p1lo_throttle_thermal_pc_r3.py b5c34e05 executed unchanged (its DataLoader patches: PC_NUM_WORKERS 2, collate
swap, CUDA memory fraction 0.5, thermal flags), its final runpy.run_path(runner) intercepted; the runner (same file as the
compared run) is loaded as module fedavg_mixed_runner and its main() runs INSIDE a Flower ServerApp (run_simulation, Ray,
5 SuperNodes, client_resources num_gpus 1.0 -> clients train one after another on the one GPU). Only the module-level
train_client is replaced, by a dispatcher that sends one Flower Message per client (order client_00..04), waits for the reply
and returns it. Aggregation (aggregate_weights float64), evaluation, checkpoints, history, quality gate: runner code, unchanged.
Message down: ArrayRecord(global state) = the FL payload. Message up: ArrayRecord(client weights). Plus a pickled CONTROL blob
both ways (harness-only, bytes logged separately, NOT counted as FL payload): the client DataLoader object (never iterated on
the server), the GradScaler state, the torch CPU / CUDA / numpy / python RNG states, the other train_client kwargs
(progress_callback dropped -> progress.json only); back: telemetry, RNG + scaler states, sampler draw statistics.
Why: the runner uses ONE global RNG stream across server eval and the 5 clients (worker base seeds, RandomSampler, CUDA dropout)
and a per-client GradScaler; handing the stream over is what makes Flower reproduce the in-process trajectory. Report label:
"RNG / scaler state handed over by the harness (simulation device for the equivalence gate; a real deployment seeds clients
independently)".
LABEL r2 (mandatory, verbatim in report): "flower_exact = equivalence gate of orchestration / transport. The client DataLoader
(dataset + sampler), the RNG and the GradScaler are created by the harness on the server side and passed in the control blob.
The client does not hold its own shard / independent seed. Only the ArrayRecord is the FL payload." The report must NOT say
"FedAvg runs truly distributed on Flower" on the basis of FLW-A; the more realistic part is FLW-B (Deployment Engine).
Client process: same runner module, cudnn deterministic True / benchmark False (the runner's set_seed flags),
CUDA memory fraction 0.5 (= throttle), then the runner's own train_client.
Logs (N1.3 schema, create-once, <out>/flw/FLW_EVENTS.jsonl): client_round_execution per (round, client): n_k, train_loss,
delta_l2 (float64, ||w_k - g||) + layer_delta_norms, client_compute_wall_ms, message_rtt_ms, payload_down/up_bytes (Array.data
bytes Flower serialises, npy header incl.), tensor_bytes_down/up (numel x element size), harness_ctl bytes, client pid, peak
VRAM; FLW_SUMMARY.json: rounds (clients wall), total wall, status, versions, shas, env. Server round wall / aggregate ms / val
metrics / RAM-VRAM peaks: the runner's own fedavg_telemetry_events.jsonl + summary + THERMAL / builtin_log (RAM).
Communication-cost table (N1.3): payload_down/up_bytes = ArrayRecord only; harness_ctl bytes are logged separately and NEVER
enter the cost table; extra column "bytes / round / client = tensor_bytes" for the theoretical check (numel x 4 for float32).
Compare: pc_r9/flw_compare.py r2: history row per round equal after dropping timing LEAF keys whose name contains one of
TIMING = ("duration", "vram", "telemetry", "time", "_sec", "elapsed", "updated_at", "utc", "_ms") (registered here; asserted equal
to p22_gates.py 1c5b2bd1 TIMING); containers (dicts / lists, e.g. client_telemetry) are compared inside; a stripped key starting
with val_ / plantdoc_ / train_ / loss / acc / macro aborts; the full stripped-key list is written to the output (archived run vs
itself: aggregate_ms, client_compute_wall_ms, client_load_ms, client_serialize_ms, compute_wall_ms, dispatch_ms, duration_sec,
load_ms, network_one_way_ms, network_rtt_ms, peak_vram_mb, receive_ms, round_wall_ms, serialize_ms, server_prepare_ms, timestamp;
s42 vs s43 first diff round 1 client_telemetry[0].delta_l2 -> the compare is sensitive). Plus |d val_macro_f1_raw| per round,
checkpoint_last model_state tensor-equal when both end at the same round, endpoint r38-40 difference.

## 3. Gates (in this order; each result to TU VAN before the next GPU step)
G_SMOKE_CPU  CUDA hidden, runner --mode smoke --rounds 3 (5 batches), job fed_mixed_label_a05_q10_s42, q15 runner, PC_NUM_WORKERS 2,
             WITHOUT --r1-seed-override (the runner rejects it outside --mode full; both arms identical, so the gate is unaffected):
             ref (throttle r3 + runner, venv_flwr) vs flower_exact: all history rows equal AND checkpoint_last tensor-equal.
             Runs in the exclusive window only (2026-11-09 04:34 attempt beside 2 GPU runs: ram_guard abort, fx peak RSS 3.97 GiB;
             outputs .kill1 / .kill2, ledger).
G_DET_GPU    SLOTS 2->1 window, queue empty: plain ref (throttle r3 + q15 runner, exact receipt command of pc_fedavg_a05_s42,
             --stop-after-round 3, new output dir) vs the archived pc_fedavg_a05_s42 run: history rows 1-3 equal. This tells whether
             the archived run is reproducible at all on this PC (it ran at SLOTS 2 next to another run).
G_SMOKE_GPU  same window: flower_exact with the same command (--stop-after-round 3) vs the archived run, rows 1-3 equal; also
             VRAM / RAM / s per round recorded (Ray + server + client process).
If G_DET_GPU fails (fallback registered now, before any result; no other fallback):
 (i) report the first differing key to TU VAN; (ii) registered back-up comparator ref_fresh = plain runner, exact receipt command,
 SLOTS 1, 40 rounds, seed 42 only; FLW-A s42 is compared with ref_fresh under the same rule; s43 / s44 are compared with the
 archived runs and, if they fail, reported "NOT EQUIVALENT vs archived" with |d| (no further ref run); (iii) archived vs ref_fresh
 reported descriptively. Cost ~ +1.6 h; ETA still before 2026-11-10 08:00.

## 4. Runs
FLW-A (mandatory): flower_exact x seeds 42 / 43 / 44, alpha 0.5 (job fed_mixed_label_a05_q10_s<seed>, q16 spec, q15 runner,
  exact receipt command of pc_fedavg_a05_s<seed> except --output-dir), 40 rounds, exclusive (SLOTS 1, scheduler idle), launched
  detached via WMI, thermal logger on. Output E/flw/flwa_s<seed>/.
FLW-A rule (registered): per seed, "bit-exact" iff all 40 history rows equal AND checkpoint_last tensor-equal vs pc_fedavg_a05_s<seed>;
  else "equivalent (1e-6)" iff |d endpoint| <= 1e-6; else "NOT EQUIVALENT" (reported as is, first differing round / key).
  Table per round (40 rows x 3 seeds: val macro-F1 raw, acc, PD acc, PD macro-F1 supported, |d|).
flower_stock (optional, cut first): same ClientApp replies aggregated by stock flwr FedAvg (float32, every array averaged incl.
  num_batches_tracked) on 1 seed (42); rule |d endpoint| <= 0.005 vs pc_fedavg_a05_s42 -> "close", else "differs"; descriptive.
A3 (cut before A2): flower_exact for --fl-method fedprox --fedprox-mu 0.01 and --fl-method fedbn, alpha 0.1 seed 42, q18 runner,
  exact receipt command of c_m_fedprox_a01_s42 / c_m_fedbn_a01_s42 + --stop-after-round 5; rule = rows 1-5 equal (bit-exact) else
  max |d val_macro_f1_raw| <= 1e-6. Label for FedBN: "local BN kept by the runner on the server side and shipped to the client in
  the message (simulation); the transport does not realise FedBN's BN locality".
FLW-B (recommended): Deployment Engine on localhost (1 SuperLink + 5 SuperNodes via WMI), 5 rounds, seed 42, alpha 0.5, nw 0,
  RAM smoke first (>= 3 GiB free left, else concurrency 1 or NOT_RUN_RAM); separate spec addendum before it runs.

## 5. Incident note (2026-11-09 04:34)
Scheduler GUARD_PAUSE of pc_q22_a1_scaffold_bn_a05_s43 for 20 s (21:35:04Z-21:35:24Z) caused by the FLW smoke beside 2 GPU runs.
At its EXIT: report the real pause (not "pause 0"), run fl_exit_check as usual, validity by the registered rule only; report text
"scheduler suspend 20 s caused by the parallel FLW smoke; no change to any computation".

## 6. ETA (hard rule: no START with ETA > 2026-11-10 20:00; target all runs by 2026-11-10 08:00)
Queue (Q22) empties ~2026-11-09 10:30-11:00. G_DET_GPU + G_SMOKE_GPU ~0.5 h, then FLW-A 3 x ~1.6-2.0 h exclusive (runner 1.42 h
at SLOTS 2; + Ray / message overhead measured at G_SMOKE_GPU) -> ~2026-11-09 17:00-18:00; A3 ~0.3 h; FLW-B after its addendum.
