# E4_SPEC (PC, r1, 2026-11-07) - server / client weight divergence (addendum E4 0bf34974) - author: DIEU PHOI PC r3; TU VAN PASS needed

Compute-only (no training, no inference), CPU BelowNormal, read-only, sequential (one torch process, via ram_guard_launcher, in a
SLOTS 2->1 window or when RAM available >= 3 GiB). Formulas registered BEFORE any number is computed. The TU VAN feasibility numbers
(FedAvg s42) are a tool check only and are not reported.

## 0. Correction to the addendum (inventory 2026-11-07, coordinator)
Snapshots are NOT written at a fixed set 1-6, 12, 40: client_server_snapshots_round_<r>.pt exists exactly at the is_best rounds of the
run (checked: q15l fedavg_s44 snapshots == its is_best rounds 1, 2, 3, 5, 6, 9, 11, 14, 26, 27, 29, 32, 33, 40). Rounds present in
EVERY candidate run (27 runs: FedAvg 5 alpha x 3 seeds, SCAFFOLD and C-M FedBN alpha 0.1 / 0.5 x 3 seeds) = {1, 2, 3, 40}.
Therefore: REGISTERED snapshot rounds R = {1, 2, 3, 40} (comparable across every run). All other available snapshot rounds of a run
are reported per run as "descriptive, run-specific rounds (is_best)" and never averaged across runs. D2 uses every round (telemetry).

## 1. Notation
Parameter vector w = all float tensors of the state dict EXCEPT BN buffers (running_mean, running_var, num_batches_tracked); BN buffers
form a separate vector b (reported separately, same formulas). ||.|| = L2 over the concatenated float64 vector. For round r of a run:
g0 = server_state_before, g1 = server_state_after, w_k = client_states[k] (after local training), Delta_k = w_k - g0, G = g1 - g0.
p_k, n_k from the snapshot (checked == telemetry raw n_k).

## 2. Metrics
D1_k(r) = ||w_k - g1|| / ||g1||                        r in R
D2_k(r) = ||Delta_k||, every round r = 1..40, from fedavg_telemetry_events.jsonl: sqrt(sum over non-buffer layers of layer_delta_norms^2)
          (telemetry delta_l2 also reported raw, with a note whether it includes buffers - decided by gate G_D2)
D3_k(r) = cos(Delta_k, G) = <Delta_k, G> / (||Delta_k|| ||G||)       r in R; ||G|| == 0 -> "undefined"
D4_k,l(r) = ||w_k,l - g1_l|| / ||g1_l|| per layer l       r in R; reported per layer and per group: classifier (classifier.*), BN affine
          (weight / bias of a BN layer = a layer that has running_mean), other conv / linear; top 5 layers by mean_k D4 at r = 40
D5 = ||w_FedAvg - w_Central|| / ||w_Central||: FedAvg checkpoint_last (round 40) vs Central SGD checkpoint_last (epoch 40,
          n11l/sgd_s<seed>), SAME seed, alpha in {100, 10, 1, 0.5, 0.1} x seeds 42-44 (Zhao et al. 2018 weight divergence, here vs the
          SGD central model: same W0 eaf9197b, same optimizer family). Scale (descriptive): ||w_FedAvg - W0|| and ||w_Central - W0|| with
          W0 = the round-1 g0 of the same FedAvg run. BN buffers: D5_b separately.

## 3. Aggregation and output
Per run and r: mean_k, SD_k (n - 1 = 4), max_k (client id, n_k), n_k-weighted mean. Per method x alpha over seeds 42-44: mean +- SD
(n - 1 = 2). Methods: FedAvg 5 alpha; SCAFFOLD, FedBN at alpha 0.1 / 0.5 now; FedProx / MOON C-M when their runs exist (same harness,
appended rows, separate PASS). FedBN: D1 / D3 / D4 on non-BN parameters only (its BN affine are local by design - reported "local").
Outputs (create-once, E/e4/): E4_RESULT.json, E4_D1_D3_D4.csv, E4_D2_rounds.csv, E4_D5.csv; figures E4_D2_by_round_<alpha>.png
(mean_k D2 vs round, one line per seed + mean) and E4_D1_D3_by_alpha.png (r = 40 and r = 1). Harness pc_r9/e4_divergence.py (sha).
Labels "validation not used", "PC", "descriptive (no test)". No claim beyond the numbers; no rule / gate on the outcome.

## 4. Gates (fail -> that run's metric "unknown")
G_SNAP  required rounds R present; snapshot keys {server_state_before, server_state_after, client_states (5), p_k, n_k}; state-dict key
        sets of g0 / g1 / w_k equal (244); n_k == telemetry raw n_k; sum p_k == 1 (|.| <= 1e-12).
G_AGG   FedAvg-family (FedAvg, FedProx, MOON): ||g1 - (g0 + sum_k p_k Delta_k)|| / ||G|| <= 1e-5 on w (checks snapshot semantics; not
        applied to SCAFFOLD (server step uses control variates) or FedBN (BN local) - recorded "n/a").
G_D2    at each r in R: |D2_telemetry - ||Delta_k||_snapshot| <= 1e-4 * ||Delta_k|| + 2e-6 * sqrt(L) (telemetry layer norms are rounded
        to 6 decimals; L = number of layers); also which of {params only, params + buffers} the raw delta_l2 matches.
G_D5    FedAvg and Central summaries w0_fingerprint == eaf9197b...; checkpoint key sets equal; Central epoch == 40, FedAvg round == 40.
G_SELF  toy: g0 = [0, 0], w_1 = [1, 0], w_2 = [0, 1], p = [0.5, 0.5] -> g1 = [0.5, 0.5], D1_1 = sqrt(0.5) / sqrt(0.5) = 1.0,
        D3_1 = 0.5 / (1 * sqrt(0.5)) = 0.707107.
