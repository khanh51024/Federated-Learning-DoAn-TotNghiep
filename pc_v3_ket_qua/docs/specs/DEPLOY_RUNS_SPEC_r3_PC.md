# DEPLOY_RUNS_SPEC r3 (PC) - deploy candidates across machines, EXT-SEL selection (2026-11-02) - author: DIEU PHOI PC; TU VAN PC PASS needed

Replaces DEPLOY_RUNS_SPEC r2 sections 2-3 for the PC (r2 = laptop, unchanged file). Source of the candidate list: PC_HANDOVER section 4
("deploy: choose on EXT-SEL by MODEL regardless of machine; effects of bg / r3c relative to R3-data PC"). Registered BEFORE any
EXT-SEL evaluation on the PC. Detector / R4 (r2 section 4) unchanged; done after this step with laptop_runs/detl (not retrained).

## 1. Candidates (fixed now; each evaluated on EXT-SEL exactly ONCE, in one ext_sel_eval_pc call)
| label | platform | run dir (experiments/train-improve-20261015/) | checkpoint_best sha256 | nw | lr_drop_epochs | best_epoch | selectable |
|---|---|---|---|---|---|---|---|
| r3data_pc | PC RTX 2060S | r3l/data_out | a274bb9e713728ab5578198661b93419d4ff8d5050cf643416346736997f58a4 | 2 | 22 26 30 34 | 35 | yes (BASELINE) |
| bg_pc | PC | r3xl/bg_out | fad07c39666acb2dee20e7627adc0f5a3443c6b4f4adc7c516e5a8f17aac0d33 | 0 | 21 26 30 34 | 22 | yes |
| r3c_pc | PC | r3xl/r3c_out | 8751d8f10cab78f2e7394c95cb3091e4643cada5c31b2b371942d0f65e75a344 | 0 | 19 25 29 33 | 21 | yes |
| r3data_lap | laptop RTX 2050 | laptop_runs/r3l/data_out | 905043dde38f14901148a342e886195f9a7f48eca1fddfd826712ca90648f450 | 0 | 19 24 32 | 35 | yes |
| aug2_lap | laptop | laptop_runs/r3xl/aug2_out | 0463eb3eefc1811a18ea2186472c172b439098f2c4ae90e283167a79fe54c308 | 0 | 19 23 30 34 | 26 | yes |
| bal_lap | laptop | laptop_runs/r3xl/bal_out | 8f9336dadbc8a9e2a6fef80b9d12166a807ebb0cde2c8a996dc866cb1e46bdc9 | 0 | 19 23 27 31 35 | 20 | yes |
| q5central | Colab T4 (AdamW) | q5/colab/received_20261021 | c45fa9937b636d815ef99f4bd21f078cad86521cf67a5b118f7c64e0a7809998 | - | - | - | no (reference) |
Platform / nw notes: nw 2 changes the RNG stream of the train flips vs nw 0 (one different draw); lr plateau drops are data dependent.
These are recorded, not corrected.

## 2. Validity (deploy_gates_pc; r2 section 2 rules, per platform)
- PC candidates: receipt rc 0; status MAX_BUDGET_REACHED or QUALITY_GATE_FAILED; epochs 1..35; summary best_checkpoint_sha256 == file
  sha == the table; resolved_config vs r3data_pc differs only by the registered keys (bg: r3x_variant, r3x_ext_sha256,
  r3x_mask_index_sha256, r3x_bg_bank_sha256; r3c: r3_extra_train_manifest_sha256); keys with "output" ignored.
- Laptop candidates: the same rules vs r3data_lap (aug2 / bal: r3x_variant, r3x_ext_sha256) - exactly the r2 check.
- Cross-machine: r3data_pc vs r3data_lap resolved_config after normalising absolute paths (PC root -> laptop root, PC_HANDOVER s4 c):
  only path-valued keys may differ; recorded (TU VAN found: spec_file_path only).
- Guardrails (val, registered in r2, absolute, UNCHANGED, all candidates): PV acc >= 0.9759, PD acc >= 0.5339, PD disease macro-F1
  >= 0.541545 (from best_validation_predictions.csv, r2 definition). Failing -> not selectable, still evaluated and reported.

## 3. EXT-SEL selection (split_v1 lock 15601109, EXT-SEL 9a3d0e3f, 403 items; label "selection, biased")
- Score S = unweighted mean of per-class recall over the 7 EXT-SEL classes (exact fractions) - r2 definition, same image path.
- Winner = argmax S over valid + guardrail-passing selectable candidates (all platforms); it must exceed S(r3data_pc) by >= 0.02,
  otherwise the winner is r3data_pc. Ties -> earlier row of the table. q5central reference only.
- Descriptive effects (not a selection rule): bg_pc - r3data_pc, r3c_pc - r3data_pc (PC pairs); aug2_lap - r3data_lap,
  bal_lap - r3data_lap (laptop pairs); r3data_pc - r3data_lap ("reproduction on a second machine"). Every row carries its platform.
- Secondary (reported, not used): mean recall over unseen-field classes (Potato healthy, Apple rust, Apple healthy; r3c: Potato
  healthy only). Per-class recall with Wilson 95 % CI, per-source accuracy.
- No second round on EXT-SEL (P1b round 2 needs its own spec).

## 4. Tools (builders, originals unchanged) and resources
- pc_r9/ext_sel_eval_pc.py built by pc_r9/build_ext_sel_eval_pc.py from r8_draft/ext_sel_eval.py (b38020ad) by 12 exact anchors
  (candidate table + PLATFORM, baseline label r3data -> r3data_pc, IN_SOURCE key r3c_pc, output ext_test/sel_eval_pc_v1 create-once,
  torch.set_num_threads 2 -> 4). Selection logic, image path, S, Wilson, manifest / byte checks: unchanged code.
- pc_r9/deploy_gates_pc.py: new file that imports the UNCHANGED r2 functions of r8_draft/deploy_gates.py (pd_disease_f1, src_acc,
  guard, q5_reference, flat, sha, GUARD, EXPECT_DIFF) and only re-implements the candidate loop for two baselines + the cross-machine
  check + the detector path laptop_runs/detl. Real gates file: experiments/train-improve-20261015/deploy_pc/gates_v1 (create-once).
- pc_r9/ram_guard_launcher.py runs the evaluation under the resource rule below.
- Before the real call: selftest mode of ext_sel_eval (non-EXT manifest, outside ext_test) with all 7 checkpoints.
- Resources (TU VAN 2026-11-02): one evaluation process, CPU only, DataLoader nw 0, 4 threads, BelowNormal; start only when RAM
  available >= 3.0 GiB; RAM logged every 30 s; stop itself if < 2.0 GiB. GPU stays with the queue.
- Never reads EXT_FINAL; EXT-SEL bytes sha-checked per image (r2 code).

## 5. Addendum (TU VAN PASS 2026-11-02): failure before EXT_SEL_RESULT.json
If RAM_ABORT, a crash or rc != 0 happens before EXT_SEL_RESULT.json exists: keep ext_test/sel_eval_pc_v1.tmp untouched (not read,
not deleted without the user's consent), write an INCIDENT; ONE rerun is allowed through a builder anchor REAL_OUT ->
ext_test/sel_eval_pc_v1b (same evaluation: nobody has seen any result). A second failure -> stop and ask the user.
