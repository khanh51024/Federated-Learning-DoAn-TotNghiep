# A1A2_SPEC (PC, r1, 2026-11-07) - A1 SCAFFOLD-BN (N3.3, team proposal) + A2 float16 uplink (N4.2) - DIEU PHOI PC r3; TU VAN PASS needed

Source: PROMPT_DIEU_PHOI_R3_ADDENDUM_3_YEUCAU 2736303a sections A1, A2 (user decision 2026-11-07). Registered BEFORE any A1 / A2 code
is built. Protocol P40 unchanged (SGD 0.01, m0, wd 1e-4, batch 32, AMP, 5 clients, local_epochs 1, 40 rounds, PD sampler 0.25,
canonical_v1, partition_seed 42, W0 eaf9197b, nw 2, throttle r3). Validation only.

## 1. Code: sandbox q22 = byte copy of the q18 sandbox (Q18 lock 3a5bfdfe, every file checked), then builder edits with exact anchors
in fedavg_mixed_runner.py and q8r2_methods.py ONLY (everything else byte-equal, own Q22 lock):
A1 --fl-method scaffold_bn:
  - BN handling == fedbn: every client keeps its own BN parameters AND buffers (fedbn_client_start(global, local_bn[k]) each round);
    the server never sends averaged BN to a client. The server-side BN placeholder (n_k-average, exactly as fedbn) is used only for the
    descriptive server-model endpoint.
  - SCAFFOLD (option II, unchanged functions) on the NON-BN trainable parameters only: c, c_i initialised on q8_param_keys minus BN
    keys; the correction is added to non-BN gradients only (scaffold_apply skips parameters without a correction entry - for the
    existing scaffold path every parameter has an entry, so behaviour is unchanged, checked by G_NEUTRAL).
  - Personalised evaluation at round 40 with fedbn_round_eval (same call as fedbn).
A2 --uplink-dtype {float32 (default), float16}: the client update Delta_k = w_k - g (every floating tensor of the state dict) is cast
  to float16 and back before aggregation (w_k' = g + float(half(Delta_k))); integer tensors unchanged; aggregation float64 as now.
  payload_up_bytes = bytes of the floating tensors at 2 B + integer tensors at their size (measured from the tensors, not assumed).
  float32 path = no cast at all (code path skipped).
Drivers: pc_r9/q22_laptop_kaggle.py / q22_laptop_driver.py from the q19m pair (same validity checks; RUNS = the 9 runs; output q22l).

## 2. Gates (CPU unless stated; ALL must pass before append; Q22_GATES.json)
G_LOCK     q22 sandbox: every file except the 2 edited == Q18 lock sha; the 2 edited files == builder receipt; Q22 lock written.
G_NEUTRAL  smoke 2 rounds (job fed_mixed_label_a05_q10_s42, CPU): q22 runner == q18 runner tensor-equal (checkpoint_last model_state)
           and history-equal (timing keys dropped) for fedavg, scaffold, fedbn, fedprox (mu 0.01) and moon (mu 1.0 tau 0.5), with
           --uplink-dtype absent and = float32.
G_BN       scaffold_bn smoke 2 rounds: (a) for every client and round 2 the starting BN tensors == that client's round-1 BN after local
           training (never the server placeholder); (b) c / c_i contain no BN key and contain every non-BN parameter; (c) with
           --q8r2-neutral (BN shared, c = 0) scaffold_bn == fedavg tensor-equal (neutral switch defined for the gate only).
G_FP16     fedavg --uplink-dtype float16 smoke 2 rounds: payload_up_bytes == expected from the tensors (float tensors x 2 B + int tensors)
           and ~ 0.5 x float32 (ratio printed); max |w_k' - w_k| <= 2^-11 * max|Delta_k| per tensor (half rounding); run != float32 run.
G_ARGS     scaffold_bn refuses --fedprox-mu / --moon-mu; --uplink-dtype only in {float32, float16}; the 9 commands carry exactly the
           registered args.
G_JOBS     --preflight-only of the 9 commands, rc 0 + PREFLIGHT CHECK OK.
GPU smoke  2 rounds of one scaffold_bn command through the driver in a SLOTS 2->1 window (VRAM, s / round recorded).

## 3. Runs (9, appended right after S5; names pc_q22_*)
A1: scaffold_bn x alpha {0.1, 0.5} x seeds 42-44 (jobs fed_mixed_label_a01_s<s> / fed_mixed_label_a05_q10_s<s>, q18 spec).
A2: fedavg --uplink-dtype float16, alpha 0.1, seeds 42-44 (job fed_mixed_label_a01_s<s>).

## 4. Rules (registered now)
A1 primary metric = CDR at round 40 (PHASE_C s3 c, fedbn_compare function: personalised models, p_k-weighted recall, mean over the 5
clients). Comparators, same alpha and seed: FedAvg (PC anchors; CDR from fedbn_compare on checkpoint_last) and SCAFFOLD (C-S runs; CDR
from fedbn_compare on its checkpoint_last, global model for every client). 4 cells = {vs FedAvg, vs SCAFFOLD} x alpha {0.1, 0.5}.
delta_s = CDR(scaffold_bn, s) - CDR(comparator, s), s = 42-44, one-sided paired t (df 2, same implementation as S5 s3 b), Holm over
the 4 cells (family alpha 0.05). WINNER iff p_Holm <= 0.05 AND dbar >= 0.02 AND pd_drop <= 0.02 (pd_drop = comparator r40 PD acc
minus scaffold_bn uniform-mean PD acc of the 5 personalised models, mean over seeds). Else "not shown". Self-computed critical
one-sided t (df 2): Holm step 1 (p <= 0.0125) t >= 6.205, step 2 (0.016667) 5.339, step 3 (0.025) 4.303, step 4 (0.05) 2.920.
Descriptive: server-model endpoint (val_macro_f1_raw r38-40) and PD acc; per-seed deltas; signs 3/3. Report label: "SCAFFOLD-BN is
the team's proposal; 3 seeds only; descriptive server-model endpoint is not the primary metric".
A2 descriptive only (no winner rule): MB up per round (measured), ratio to float32, endpoint paired by seed vs pc_fedavg_a01_s42-44
(delta_s and mean), MB up until rounds_to_095 / rounds_to_099 (E3 definition), trade-off MB vs accuracy.

## 5. ETA
A1 6 x ~1.5 h (SCAFFOLD 1.45 h + local BN eval at r40), A2 3 x ~1.42 h -> ~13.3 h-slot / 2 x 1.05 ~ 7 h after S5 (~2026-11-09 01:00)
-> ~2026-11-09 08:00. Hard rule: no START with ETA > 2026-11-10 20:00; cut order per addendum 3 C (A2 before S5 seed 46; A1 never).
