# S5_SPEC (PC, r1, 2026-11-07) - Dot 3: 5-seed method analysis - author: DIEU PHOI PC r3; TU VAN PC PASS needed

Registered BEFORE any seed-45 run starts (PROMPT_DIEU_PHOI_R3 3fc66ff8 s4 + s6 step 8; TU VAN addendum 2026-11-07 "thong nhat seed").
This is a NEW analysis requested by the lecturer for Dot 3. It does NOT replace, re-score or re-label Phase C: the Phase C 3-seed
cells (PHASE_C_SPEC_PC r2 a8326c06, rule R1, "one of 8 tests, no multiple-comparison correction") are reported unchanged, side by side.

## 0. Seed roles (write this verbatim in the spec and in the report)
- seed 41: tuning / screening only (D4, C-T). Deliberately separated from the evaluation seeds. Never enters any table below.
- seeds 42 / 43 / 44: MAIN tables (Central, FedAvg, Local-only, Phase C cells) - unchanged.
- seeds 42-46: METHOD analysis of this spec (5 seeds). Local-only is NOT run for 45 / 46 (main tables only).
- every seed: partition_seed 42 (same shards, same registry digests), W0 eaf9197b; the seed changes ONLY the data order, the sampler
  and the augmentation (training_seed). Protocol P40 unchanged (SGD 0.01 constant, m0, wd 1e-4, batch 32, AMP, 5 clients,
  local_epochs 1, 40 rounds, PD sampler 0.25, canonical_v1, nw 2, throttle r3, duty 1.0).

## 1. Runs (24, all full mode, create-once, no resume, cap 10 h)
FL (20) on the q18 runner (the gated sandbox that holds fedavg / scaffold / fedprox / moon / fedbn; Q18 lock unchanged), alpha
{0.1 -> job fed_mixed_label_a01_s<seed>, 0.5 -> fed_mixed_label_a05_q10_s<seed>}, seeds {45, 46}, --r1-seed-override:
  FedAvg (--fl-method fedavg) | FedProx (--fedprox-mu 0.01 = mu*) | FedBN | SCAFFOLD (--fl-method scaffold) | MOON (--moon-mu 1.0
  --moon-tau 0.5 = mu*). mu* from Q19l/scores/q19_tuning_20261107.json 1eec7a02 (TU VAN PASS 2026-11-07); no new tuning.
Central (4) on the n11 sandbox (trainer 35540d10, unchanged): sgd_s45 / sgd_s46 (job cent_mixed_sgd_s<seed>, 40 epochs, recipe of
  sgd_s42-44) and adamw_s45 / adamw_s46 (job cent_mixed_r3c_s<seed>, R3-ctrl recipe, N 35, as adamw_s43 / s44), --seed-override-floor.
Names: pc_s5_<method>_<a01|a05>_s<seed> (FL), pc_s5_<sgd|adamw>_s<seed> (Central). Output roots q21l (FL), n11l (Central; new names).

Runner lineage (stated in the report): FedAvg / SCAFFOLD seeds 42-44 ran on the q15 runner (q15 / q16 drivers), seeds 45 / 46 run on
the q18 runner; the default fedavg / scaffold path is tensor-equal by the PC gates q15 == q17 (q17_gates G_DEFAULT_FED) and q17 == q18
(q18_gates G_DEFAULT), the same argument Phase C already uses for its cross-runner pairs.

## 2. Code (built after this spec PASS; builders with exact anchors, sources pinned and unchanged, create-once outputs)
- E/q21/specs/fedavg_spec_q21.json = byte copy of q18/specs/fedavg_spec_q18.json (812682ff) + 4 jobs (a01 / a05_q10 x s45 / s46),
  each a copy of the s44 job of the same scheme with only job_id and seed changed (partition_seed 42 kept). The runner uses the spec
  file only for the job entry and provenance (campaign_spec_sha256); the Q18 lock is not touched.
- pc_r9/q21_laptop_kaggle.py from pc_r9/q19m_laptop_kaggle.py 9a6a7529: SPEC -> q21 spec, RUNS = the 20 S5 runs, Q18_PARAMS for them,
  validity unchanged (40 rounds, status, digest == registry, training_seed, R1 status PASS_SEED_OVERRIDE_R1, method config).
- pc_r9/q21_laptop_driver.py from pc_r9/q19m_laptop_driver.py 13f8b835: kaggle pin -> q21, RUNS -> 20 names, output q21l.
- n11 Central: E/n11s5/specs/central_spec_n11s5{,_adamw}.json = byte copies of the n11 specs (f9394482 / 029243d9) + the 2 jobs each
  (copy of the s44 job, job_id + seed changed); r8_draft-derived pc_r9/n11s5_laptop_driver.py from r8_draft/n11_laptop_driver.py:
  RUNS -> 4 runs, spec paths + PINS -> the new specs, everything else unchanged (sandbox pin 35540d10, throttle r3, monitor v2).
- Gates (CPU, sequential, ram_guard_launcher), Q21_GATES.json: G_BUILD (all shas; sources unchanged), G_SPEC (new spec JSON ==
  source + exactly the listed jobs; every other key byte-equal after json load), G_ARGS (each run: method args / params exactly as s1;
  job / scheme equal to the s44 run of the same method and alpha with the seed swapped), G_JOBS (--preflight-only of the 20 FL
  commands rc 0 + PREFLIGHT CHECK OK; Central: the trainer's own dry / preflight path if it has one, else "not available" recorded and
  the first Central run is watched to round 1). No GPU smoke: no code path changes (same runners, same drivers modulo tables).
- pc_r9/s5_score.py (CPU, read-only) + its self-test (section 4 vectors) before it reads any S5 run.

## 3. Decision rules (registered now, before seed 45 starts)
Cells: 8 = {SCAFFOLD, FedProx, MOON, FedBN} x alpha {0.1, 0.5}. Comparator: FedAvg of the same alpha and seed (42-44 = the PC anchors
q15l fedavg_s<seed> / q16l fedavg_a05_s<seed>; 45 / 46 = q21l s5 FedAvg). Metric per cell as Phase C: endpoint = mean val_macro_f1_raw
rounds 38-40 (SCAFFOLD / FedProx / MOON); FedBN = CDR (PHASE_C s3 c, same fedbn_compare function on FedAvg checkpoint_last, round 40).
pd_acc as Phase C (FedBN: uniform mean PD acc of the 5 personalised models at round 40 vs FedAvg r40 PD acc).
a) delta_s = metric(M, s) - metric(FedAvg, s), s = 42..46 (paired by seed). n = 5. dbar = mean, sd = sample SD (n - 1).
b) Test: one-sided paired t, H1: dbar > 0, t = dbar / (sd / sqrt(n)), df = n - 1, p = P(T_df > t) (regularized incomplete beta; no
   scipy - implementation self-tested against the df 4 closed form and t_0.95,4 = 2.131847, t_0.99,4 = 3.746947, t_0.95,3 = 2.353363).
   sd == 0 -> p = 0 if dbar > 0 else 1.
c) Holm over the 8 declared cells, family alpha 0.05 (adjusted p_(i) = max_{j<=i} min(1, (8 - j + 1) p_(j))). A cell with an
   invalid / missing run is "unknown" and STAYS in the family with p = 1 (the family size is fixed at 8).
d) WINNER5 iff Holm-adjusted p <= 0.05 AND dbar >= 0.02 AND pd_drop5 (mean over the 5 seeds) <= 0.02. Otherwise "not shown".
   The 0.02 effect floor and the 0.02 PD rule are the Phase C values, unchanged. Self-computed critical t (df 4) at the Holm steps:
   4.315 / 4.148 / 3.961 / 3.747 / 3.495 / 3.186 / 2.776 / 2.132 (smallest p first). No other test, no other alpha, no one-cell
   re-test, no "best method" across cells.
e) Descriptive only (no rule): per-seed deltas, signs k/5, exact sign-flip p (2^5 = 32, smallest one-sided 1/32 = 0.03125, so it can
   never pass Holm at 8 - reported for transparency only), 95% t-interval of dbar, gap to Central with 5 seeds (Central AdamW and SGD
   s42-46, mean +- SD; adamw_s42 = R3-ctrl r3l/ctrl_out as in the main table), gap-narrowing ratio as PHASE_C s3 b.
f) Reported side by side with the Phase C 3-seed cell (R1, unchanged) under the label "Phase C (3 seeds, R1, uncorrected)" vs
   "S5 (5 seeds, paired t + Holm 8)". Disagreement is reported as is; neither overrides the other.
g) Seed-46 cut (PROMPT r3 s6 "Neu tre"): allowed ONLY by the ETA rule (a run that cannot finish before 2026-11-10 20:00 local is not
   started), decided from the queue clock, never from any seed-45 result. If cut: n = 4 (seeds 42-45), df 3, same rule, label
   "4 seeds (seed 46 cut for time before results)"; critical t df 3: 5.392 / 5.138 / 4.857 / 4.541 / 4.177 / 3.740 / 3.182 / 2.353.
h) No EXT / test data read; val only (P6 is the only final test).

## 4. Scorer self-test vectors (s5_score must reproduce before reading runs)
- deltas [0.031, 0.024, 0.028, 0.019, 0.035]: dbar 0.027400, sd 0.006189, t 9.900022, p_one 2.921e-04.
- Holm on p = [0.001, 0.004, 0.006, 0.02, 0.03, 0.2, 0.5, 0.04] -> adjusted [0.008, 0.028, 0.036, 0.1, 0.12, 0.4, 0.5, 0.12].

## 5. Queue order and ETA
Order (lecturer order FedProx -> FedBN -> SCAFFOLD -> MOON, each with the FedAvg of the same seed; seed-major so that a seed-46 cut
leaves a complete 4-seed set): seed 45: FedAvg a01, a05, FedProx a01, a05, FedBN a01, a05, SCAFFOLD a01, a05, MOON a01, a05, Central
sgd, adamw; then seed 46 in the same order. Appended after C-M (append-only, separate append script with dry run, TU VAN PASS).
Measured PC wall: FL 40-round ~1.42-1.47 h (C-T MOON == FedProx 0.73-0.75 h / 20 rounds), SCAFFOLD ~1.45 h, Central SGD 1.05-1.25 h,
AdamW 0.91-0.92 h -> 16 x 1.46 + 2 x 1.15 + 2 x 0.92 ~ 27.5 h-slot / 2 slots x 1.05 ~ 14.5 h. Queue ahead: Phase D 12 (~20 h-slot) +
C-M 12 (~17.5 h-slot) ~ 19.5 h -> S5 ~ 2026-11-08 10:00 -> 2026-11-09 01:00; FLW (exclusive) after. Windows Update pause to
2026-11-20T15:21Z: no constraint.
