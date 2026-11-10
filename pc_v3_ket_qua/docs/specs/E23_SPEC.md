# E23_SPEC r1 (PC, 2026-11-09) - E3 (recovery + N1.3 logger tables) and E2 (PlantDoc diagnosis alpha 100, parts a, d)

Source: PROMPT_DIEU_PHOI_R3_20261107.md s6 items 6-7 + ADDENDUM_3_YEUCAU section B (E3 additions). Formulas registered BEFORE any
number is computed. Validation only (manifests/val.jsonl 4,613 = PV 4,344 + PD 269). No test, no internal_holdout, no EXT. ASCII.

## 0. Shared inference (one CPU job)
pc_r9/e23_predict.py (sha in the cover message). Probe (2 models, 80 val images, scratchpad) OK. Output experiments/train-improve-20261015/
e23/ (create-once): logits/<model>.npz (float32 logits 4613x38, targets, plantdoc mask), logits_pd_last3/ (Central, epochs E-2, E-1, PD
only), E23_PREDICT.json. Models (21, all checkpoint_last = last epoch / round 40):
central_adamw_s42 (r3l/ctrl), _s43/_s44 (n11l); central_sgd_s42-44 (n11l); fedavg_a01 (q15l), a05 (q16l), a1 / a10 / a100 (q17l) x s42-44.
Dataset / transform / collate / evaluate_metrics imported from the q15 runner file 7c29f6f6. CPU 4 threads, batch 64.
Gate G_REPRO per model: |recomputed val_macro_f1_raw - the run's own history value at that step| <= 2e-3; Central also on
plantdoc_macro_f1_supported at E-2, E-1, E. FAIL -> that model's rows "unknown", nothing re-tuned.
Run: one detached WMI window through ram_guard_launcher (start >= 3 GiB free, kill < 2 GiB), process priority BelowNormal, beside
R-DIAG (TU VAN condition). Expected ~20-30 min.

## E3.1 Client drift by alpha (FedAvg 5 alpha x 3 seeds; telemetry fedavg_telemetry_events.jsonl, event_type client_round_execution)
Per run, round r, client k: L_rk = delta_l2; C_rk = sqrt(sum of squares of layer_delta_norms over keys starting "classifier.")
(layer_delta_norms is a Python-dict string -> ast.literal_eval).
D_r = mean_k L_rk; CV_r = sd_k(L_rk, ddof 0) / D_r; S_r = mean_k (C_rk / L_rk).
Report per run: D_1, D_20, D_40, Dbar = mean_{r=1..40} D_r, CVbar = mean_r CV_r, Sbar = mean_r S_r.
Also parameters only (delta_l2 includes BN running stats): P_rk = sqrt(sum of squares of layer_delta_norms over keys NOT ending
running_mean / running_var / num_batches_tracked); DP_r = mean_k P_rk; report DP_1, DP_40, DPbar. layer_delta_norms may be a dict
or its string form (both read). Per alpha: mean +- SD (ddof 1) over 3 seeds.
Descriptive; note that delta_l2 also shrinks with the LR schedule (same schedule policy for all alpha).

## E3.2 Communication (from telemetry + code, no guess)
Per client per round: payload_down_bytes, payload_up_bytes (telemetry; must be constant over rounds/clients, else report the set).
Per round = sum over the 5 clients of (down + up); per run = 40 rounds. Units: bytes, MB = 1e6 B, GiB = 2^30 B, all three printed.
SCAFFOLD (from code q8r2_methods.scaffold_init / scaffold_client_update / scaffold_server_update): server sends c, client returns
c_i+ - c_i, one float32 tensor per trainable parameter key -> extra bytes per client per round per direction = 4 x sum numel over the
q8 parameter keys (computed from the model, key list as built by the runner; cite the runner line). Simulation counts are not network
measurements (telemetry says "in_process_simulation").
Raw images: total bytes on disk of the 37,238 files listed in pv_pd_v3 manifests/train.jsonl (os.stat of the resolved path); ratio
FedAvg bytes per 40-round run / raw image bytes.

## E3.3 Rounds to 95 % / 99 % of the endpoint (FedAvg 5 alpha x 3 seeds, history val_macro_f1_raw)
E = mean r38-40; R95 = min r with val_macro_f1_raw_r >= 0.95 E; R99 = same with 0.99 E; none -> "not reached". Per run + per alpha
mean +- SD. (SCAFFOLD / other methods at alpha 0.1, 0.5 also reported per run if their history is present; same formula.)

## E3.4 Time
Per run: mean over r of mean_k client_compute_wall_ms, and total = sum over r, k (seconds). Per alpha mean +- SD. Label: "PC, simulation
sequential trong mot tien trinh; SLOTS 2 dung chung GPU; khong phai thoi gian mang".

## E3.5 Error analysis (confusion matrices, logits from s0)
38 x 38 count matrix (rows true, cols pred) per model; family sum over seeds 42-44 for central_adamw and fedavg_a{01,05,1,10,100}.
Top-10 off-diagonal (true -> pred) pairs by summed count per family, with class names from pv_pd_v3 taxonomy.json class_to_idx and
the share of that true class. Also the same restricted to the PD subset. CSV per family + the JSON.

## E3.6 PlantDoc with bootstrap CI 95 % (21 models, logits from s0, the 269 PD val images)
Point: PD accuracy; PD macro-F1 over the classes present among the PD targets (= runner raw_macro_f1_supported on the PD source).
Bootstrap: B = 2000, rng = numpy.random.default_rng(20261109), ONE index matrix (2000 x 269, with replacement) shared by all models
(paired); per resample macro-F1 over the classes present in that resample's targets; CI = 2.5 / 97.5 percentiles. Per family: mean of
the 3 per-seed points, and the seed range.

## E2 (a) Central "mean of last 3 epochs", PlantDoc
Per Central seed (AdamW s42-44: epochs 33-35; SGD s42-44: 38-40): PD macro-F1 supported = mean over the 3 epochs of history
plantdoc_macro_f1_supported; PD accuracy = mean over the 3 epochs of e23 PD accuracy (Central history has no plantdoc_accuracy).
Per family mean +- SD (ddof 1) over seeds. Same for FedAvg alpha 100 r38-40 from its history (plantdoc_macro_f1_supported,
plantdoc_accuracy). Paired gap per seed: Central - FedAvg alpha 100, mean +- SD.

## E2 (d) McNemar exact, paired, PD val (269)
Correct/incorrect per image from the argmax of checkpoint_last logits (s0). Pairs, per seed s in 42-44:
(central_adamw_s, fedavg_a100_s) and (central_sgd_s, fedavg_a100_s) -> 6 tests. b = Central correct & FedAvg wrong, c = reverse.
p = min(1, 2 * P(X <= min(b, c))), X ~ Binomial(b + c, 0.5) (exact, scipy-free, math.comb); b + c = 0 -> p = 1.
Holm over the 6 declared tests (step-down, monotone). Report b, c, p, p_holm, accuracies. Descriptive conclusion only per registered
alpha 0.05 on p_holm; no extra tests added.

## Analysis code
pc_r9/e23_analysis.py (written after this spec, sha sent before it runs); reads only e23/ outputs, run summaries / telemetry / partitions /
manifests listed above; writes e23/E3_RESULT.json, e23/E2_RESULT.json + CSVs, create-once. CPU seconds.

## Wording
All numbers "tren validation (4.613 anh), mo ta". E2(d): "McNemar exact ghep cap tren 269 anh PD val; Holm 6 phep thu".
