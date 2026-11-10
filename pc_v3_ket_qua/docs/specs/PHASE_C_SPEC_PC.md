# PHASE_C_SPEC (PC, r2, 2026-11-02) - non-IID methods on the PC (PLAN_PC_R9 P4) - author: DIEU PHOI PC; TU VAN PC PASS needed

Registered BEFORE any phase C run. Platform: PC RTX 2060 SUPER, nw 2 (gated throttle r3), duty 1.0, SLOTS 2. Protocol P40 unchanged
(SGD 0.01 constant, m0, wd 1e-4, batch 32, AMP, 5 clients, local_epochs 1, 40 rounds, PD sampler 0.25, canonical_v1, W0 eaf9197b,
partition_seed 42, endpoint = mean val_macro_f1_raw rounds 38-40, pd_acc = mean plantdoc_accuracy rounds 38-40; q15_score definitions).
D4 = NOT_ESTABLISHED (ledger) -> NO D4 confirmation runs in phase C.

## 1. Runs
C-S  SCAFFOLD x alpha {0.1, 0.5} x seeds 42-44 (6 runs) with the EXISTING pinned drivers (no new code):
     q15_laptop_driver scaffold_s42 / s43 / s44 (alpha 0.1, job fed_mixed_label_a01_s<seed>) and
     q16_laptop_driver scaffold_a05_s42 / s43 / s44 (alpha 0.5, job fed_mixed_label_a05_q10_s<seed>).
C-T  Tuning, seed 41, alpha 0.1, job fed_mixed_label_a01_d4_fedavg_s41 (q18 spec; the job only carries scheme / seed), 20 rounds
     (--stop-after-round 20, status PAUSED_CHECKPOINTED; lr is constant so this equals a 20-round budget), same budget per config:
     FedProx mu in {0.001, 0.01, 0.1}; MOON mu in {0.1, 1.0, 5.0} with tau 0.5 fixed. 6 runs.
     Tuning score T = mean val_macro_f1_raw rounds 18-20. mu* = argmax T per method; tie -> the smaller mu. A tuning run is valid iff
     rc 0, rounds 1..20 present, status PAUSED_CHECKPOINTED, no thermal stop. Fewer than 2 valid configs for a method -> that method is
     NOT run in C-M and is reported "tuning failed".
C-M  Main: FedProx(mu*), MOON(mu*, tau 0.5), FedBN x alpha {0.1, 0.5} x seeds 42-44 (18 runs), jobs fed_mixed_label_a01_s<seed> /
     fed_mixed_label_a05_q10_s<seed> (q18 spec), 40 rounds, full mode, --r1-seed-override as q15 / q16.
     mu* enters C-M through an append-only builder step AFTER C-T is scored and TU VAN PASS.

Runner lineage: the FedAvg / SCAFFOLD anchors run on the q15 runner (q15 / q16 drivers), C-T / C-M on the q18 runner; the default
path is tensor-equal by the PC gates q15 == q17 (q17_gates G_DEFAULT_FED) and q17 == q18 (q18_gates G_DEFAULT, fedavg + scaffold),
so paired comparisons across the two runners are on the same code path for FedAvg / SCAFFOLD.

## 2. Code (built after this spec PASS; builders with exact anchors, originals unchanged)
- pc_r9/q19_laptop_kaggle.py from r8_draft/q18_laptop_kaggle.py: RUNS table for C-T and C-M (q18 sandbox = the gated runner that
  contains fedprox / moon / fedbn; G_DEFAULT q17 == q18 already ALL_PASS), output root q19l, tuning validity branch
  (PAUSED_CHECKPOINTED + 20 rounds) ONLY for the c_t_* runs; every other run uses the q18 validity unchanged.
- pc_r9/q19_laptop_driver.py from r8_draft/q18_laptop_driver.py: Q19 names, throttle r3 (as the integrated q18 driver), lock pins.
- Gates before queueing (CPU, as q18_gates): G_ARGS (fedprox needs --fedprox-mu; moon needs --moon-mu + --moon-tau; fedbn none);
  G_JOBS = --preflight-only of all 24 new commands; G_NEUTRAL (fedprox mu 0 == fedavg, moon mu 0 == fedavg, tensor-equal, smoke 2
  rounds); GPU smoke of one moon and one fedbn command through the driver (VRAM / time per round recorded).
- pc_r9/q19_score.py (CPU, read-only) and pc_r9/fedbn_compare.py (CPU eval of FedAvg checkpoint_last, see section 3 c).

## 3. Decision rules (registered now)
a) SCAFFOLD, FedProx, MOON vs FedAvg, per alpha, paired by seed over 42-44, FedAvg = the PC anchors of the same alpha
   (q15 fedavg_s<seed> for 0.1, q16 fedavg_a05_s<seed> for 0.5): R1 unchanged (q15_score):
   delta_s = endpoint(M, s) - endpoint(FedAvg, s); delta = mean_s delta_s; pd_drop = mean_s pd_acc(FedAvg, s) - pd_acc(M, s);
   WINNER iff delta >= max(0.02, 2 r) = 0.02 (r = 0.0028218398420888358) AND all three delta_s > 0 AND pd_drop <= 0.02.
   Any of the 6 runs of a (M, alpha) cell invalid -> "unknown" for that cell. QUALITY_GATE_FAILED counts as valid (L6).
b) Gap narrowing ratio (descriptive, per method and alpha): (mean_s endpoint(M) - mean_s endpoint(FedAvg)) /
   (ceiling - mean_s endpoint(FedAvg)), ceiling = PC 3-seed main ceiling C0 (PHASE_A_SPEC rule applied to the PC anchors);
   reported for both ceilings, label "PC".
c) FedBN (no single global model). Primary metric (registered, PLAN_R8 item 4): CDR = mean over the 5 clients of client_dist_recall
   = sum_c p_k(c) * recall_k(c) on the common val (p_k = label share of client k's train shard, classes without val support
   excluded, mass reported), at the FINAL round 40 (the runner evaluates the personalised models only at the last round).
   Comparator: FedAvg CDR of the same alpha / seed from fedavg_compare on FedAvg checkpoint_last (round 40), SAME function
   (q8r2_methods.class_share + per-class recall, global model for every client), CPU. Rule: FedBN better on CDR iff
   mean_s (CDR_FedBN - CDR_FedAvg) >= 0.02 AND all three seeds > 0 AND PD accuracy drop (FedAvg r40 PD acc minus FedBN uniform-mean PD
   acc of the 5 personalised models, mean over seeds) <= 0.02. Otherwise "not better". Descriptive (no rule): (a) uniform mean raw
   macro-F1 of the 5 personalised models; the standard endpoint of the FedBN run ("BN as in the global state").
d) Communication (descriptive): bytes per round = 5 clients x 2 directions x payload; payload 6,275,944 B fp32 (FedAvg / FedProx /
   MOON), x2 for SCAFFOLD; FedBN = payload minus its BN tensors (computed from the state dict). rounds_to_085 as q15_score.
e) No claim outside these cells; seed 41 only tunes mu; EXT / test never read in phase C.

## 4. Queue (appended after P3, append-only) and ETA
Order: C-T (6) -> C-S (6) -> [score C-T, builder step mu*, TU VAN PASS] -> C-M by seed 42, 43, 44 (per seed: fedprox a01, a05,
moon a01, a05, fedbn a01, a05).
ETA (h-slot, PC times measured: FedAvg-type run 1.37-1.59 h; tuning = 20 rounds ~0.7 h; MOON unknown, assumed <= 2x FedAvg until
the smoke measures it): C-T 6 x 0.7-1.4 = 4-8; C-S 6 x 1.45 = 8.7; C-M 12 x 1.5 + 6 x 3.0 (MOON worst case) = 36 -> 49-53 h-slot
/ 2 slots x 1.05 ~ 26-28 h after P3 ends (~03-11 14:00) -> ~04-11 16:00-18:00, far before the Windows Update limit
2026-11-20T12:00Z. Any run whose start + 10 h would pass 2026-11-20T12:00Z without a renewed pause is not started.

## 5. r2 additions (TU VAN conditional PASS of r1 932cd57c)
1. G_CDR (gate of fedbn_compare.py, CPU): fedbn_compare does NOT re-implement anything: it imports class_share and evaluate_metrics and
   the shard / loader builders of the q18 runner (pinned sha). On checkpoint_last of q15l fedavg_s42 and q16l fedavg_a05_s42 it must
   show: (i) re-evaluated val_macro_f1_raw and plantdoc_accuracy equal history round 40 (|diff| <= 1e-9); (ii) n_k of the 5 shards
   equal the client sample counts of the run; (iii) sum_c p_k(c) = 1 for every k. Gate fail -> both FedBN cells "unknown".
2. G_NEUTRAL also: fedbn with --q8r2-neutral == fedavg (tensor-equal, smoke 2 rounds).
3. C-T: a PAUSED_CHECKPOINTED tuning run is NEVER resumed (create-once); its selected_partition_digest must equal the label_alpha_0_1
   registry digest 2b9d5c4a254a53f7b3a88f8535023212a6cfaf2580e27393bb6463fc32f13563, else that C-T run is invalid. Report label:
   "mu chosen on val, seed 41, alpha 0.1 only; the same mu is used for alpha 0.5".
4. Digest check for every C-S / C-M run: selected_partition_digest == registry of its scheme (label_alpha_0_1 2b9d5c4a254a53f7...,
   label_alpha_0_5 d5a1e3dfffe96e6a783e8f5b1fb5c9581c680d48cb004e837e95a2a3da9b7fd4). Mismatch -> run invalid -> its cell "unknown".
5. Reporting: 8 cells = {SCAFFOLD, FedProx, MOON, FedBN} x alpha {0.1, 0.5}, each reported on its own; NO "best method" chosen across
   cells. A WINNER cell is reported with "one of 8 tests, no multiple-comparison correction".
