# P6_FINAL_TEST_SPEC r2 (PC, 2026-11-09) - the single final test, run on the PC

Base: plans/cai-thien-train-20261015/P6_FINAL_TEST_SPEC.md r1 (laptop draft, not edited). This r2 replaces it on the PC. Changes vs r1
(PROMPT_DIEU_PHOI_R3_20261107 s6 item 12 + TU VAN 2026-11-09): model list = the current PC campaign; web abstain / tau section REMOVED;
no v4b model (release v4 has Plant Pathology 2021 images); legacy sets labelled "exposed: used 3 times before (Q5 / Q6 / Q11b)"; FedBN and
SCAFFOLD-BN = server model only; E5 int8 dynamic rows added (no static PTQ); P6 does NOT change any selection (deploy EXT-SEL winner,
NONIID_DEPLOY_SELECT r1 bfa12b54 result SCAFFOLD, A1 0/4, S5 decisions) - it only reports test numbers. ASCII. Nothing evaluated yet.
User consent for running P6 + EXT-FINAL once (~10-11/11) after this spec is PASSED: given 2026-11-07 (prompt s3).

## 1. Once
Every checkpoint of s2 is evaluated ONCE on every set of s3 in ONE call of pc_r9/p6_final_test.py (built after this PASS, sha-sealed,
create-once output experiments/train-improve-20261015/p6/). No re-run with changes. A crash before P6_RESULT.json is written may be re-run
once with identical inputs (INCIDENT in the ledger). Every test number of the report comes from that one result file.

## 2. Checkpoints (full sha list frozen BEFORE the call)
- PC list: P6_CHECKPOINTS_PC.json built by pc_r9/build_p6_list.py (sha in the cover message), create-once at
  experiments/train-improve-20261015/p6_inputs/P6_CHECKPOINTS_PC.json, built AFTER the 6 R-DIAG runs exit (R-DIAG enters only if VALID).
  Dry build now: 201 entries, 0 excluded except the 6 R-DIAG runs not yet finished. Families:
  central_p40 (AdamW s42-46: s42 = r3l ctrl, epoch 35; SGD s42-46, epoch 40) 10; fl_alpha_curve FedAvg alpha 0.1 / 0.5 / 1 / 10 / 100 x
  s42-44 15; fl_methods alpha 0.1 / 0.5: SCAFFOLD, FedProx (mu 0.01), MOON (mu 1, tau 0.5), FedBN (server model) x s42-46 (s45/46 = S5,
  q21l) + fl_s5_fedavg s45/46; fl_methods_a1 SCAFFOLD-BN (server model) x 2 alpha x s42-44; fl_a2_fp16 3; fl_phase_d FedAvg qs / ss x
  s42-44; d4_descriptive 5 (seed 41); local_only 5 alpha x s42-44 x 5 clients = 75 and local_only_phase_d qs / ss = 30; deploy_candidate
  r3data_pc (a274bb9e, EXT-SEL winner), bg_pc, r3c_pc (guardrail FAIL: evaluated, not selectable); e5_int8_dynamic 4 (web models, TorchScript);
  rdiag_descriptive up to 6.
- Laptop list: r8_draft/p6/P6_CHECKPOINTS_LAPTOP_r1.json e118897a (26 entries; on the PC all 26 pc_path files exist and match their sha,
  checked 2026-11-09). Laptop rows = "replication on a second machine", never paired across machines. lap_detector 20cfd1a5 = the
  R4 detector.
- Every entry is resolved ONLY by its pc_path, sha256 asserted before loading; missing / mismatch stops the call before any evaluation.
- Selection rule unchanged from r1 2.1: P40 FL / Central / Local-only = checkpoint_last; deploy = checkpoint_best (QC); prediction =
  argmax of the 38 raw logits, no temperature. FedBN / SCAFFOLD-BN: the model_state of checkpoint_last (server / averaged BN) only; the
  personalised local-BN numbers stay validation-only (PHASE_C_SPEC_PC) and are never put next to these test numbers as the same quantity.

## 3. Test sets
- PD-172 = the 172 PlantDoc rows of manifests/legacy_diagnostic_test.jsonl; PD-463 = PD-172 + the 291 PlantDoc rows of
  manifests/calibration.jsonl (main PD set); PV-legacy = the 10,843 PlantVillage rows of legacy_diagnostic_test.jsonl.
  Label everywhere: "exposed: legacy test, used 3 times before (Q5 / Q6 / Q11b)". (calibration.jsonl is not used for any fitting here:
  the E5 dynamic int8 has no calibration and the static PTQ is not evaluated.)
- EXT-FINAL = ext_test/split_v1/EXT_FINAL_MANIFEST.jsonl (440 rows, 7 classes: FieldPlant / Potato / PP2020-test); unpacked ONCE with
  /c/pc_transfer_r2/unpack_on_pc_r2a.py from pc_bundle_ext_final_sealed.zip after this PASS; manifest sha 81271f2b checked; crops from
  crop_xyxy after EXIF transpose (ext_sel_eval geometry). Only clean external set.
- R4 per-leaf (lap_detector + r3data_pc) ONLY on the EXT-FINAL FieldPlant images; r4_sel_eval geometry (score >= 0.40, 8 % margin, highest
  classifier prob, fallback whole image).
- internal_holdout stays sealed. val.jsonl is used only by the dry run (s6).

## 4. Statistics (registered)
- Wilson 95 % for every accuracy / recall.
- McNemar exact two-sided binomial on discordant pairs, same items.
- Deploy family (Holm, alpha 0.05, 4 tests, EXT-FINAL item accuracy), unchanged from r1: r3data_pc vs Q5 reference (laptop ref_q5_central);
  r3data_pc vs R4 per-leaf (FieldPlant images); r3data_pc vs aug2_lap; r3data_pc vs bal_lap. Raw and Holm p; platform next to each model.
- FL / Central / Local-only: per set mean +- SD over seeds (5 seeds where S5 exists: FedAvg, SCAFFOLD, FedProx, MOON, FedBN at alpha 0.1 /
  0.5, Central AdamW / SGD; 3 seeds otherwise) + McNemar per seed (FedAvg vs method, same seed, same alpha, PC only). DESCRIPTIVE: no
  Holm family, no new claim; the registered decisions were taken on validation and are not revised by P6.
- Gap table (the thesis axis): Central AdamW / SGD vs FedAvg per alpha vs Local-only client mean, per set, mean +- SD.
- Local-only: client mean and per-client (smallest client named).
- E5 int8: delta int8 - fp32 of the same web model per set, with the s6 E5 wording.

## 5. (r1 section 5, web uncertain threshold tau) REMOVED in r2. No TAU.json, no abstain numbers.

## 6. Build, gates and dry run (before the call)
p6_final_test.py: inputs = P6_CHECKPOINTS_PC.json + the laptop list + manifest shas; INPUTS.json lists every sha (checkpoints, manifests,
script). Preprocessing per set into a temporary uint8 cache (224, canonical_v1 resize; EXT crops per ext_sel_eval) outside p6/, never
copied, then forward passes (GPU allowed when the scheduler has no open run, else CPU; logits identical up to device, argmax compared in
the dry run). Gates: unit tests of Wilson / McNemar exact / Holm on toy data with known values; dry run = the same script on val.jsonl only
(no test / EXT file opened; asserted by path) for 6 models (pc_central_adamw_s42, pc_fedavg_a01_s42, pc_scaffold_a01_s42, pc_local_a01_s42
client_00, pc_deploy_r3data, pc_int8_dynamic_fedavg) -> val numbers equal the known ones (E23 / E5 / run histories) within 2e-3.
TU VAN PASS of build + gates + dry run, then EXT-FINAL unpack, then the one call.

## 7. Output (create-once, on the PC)
p6/P6_RESULT.json (every model x set: n, accuracy, macro-F1, per-class recall with Wilson, per-source accuracy; McNemar tables; Holm;
gap table) + per-model prediction CSV / logits NPZ + p6/INPUTS.json. No image bytes in any output (EXT-FINAL has PP2020 images): ids,
relative paths, labels, predictions, logits, box coordinates only. The images and the cache never leave the PC.

## 8. Scope of conclusions
EXT-FINAL = 7 classes only. 24 / 38 classes have no PlantDoc train image. Legacy sets are exposed. Test numbers are reported next to, never
instead of, the validation numbers that made the decisions.

## 9. Schedule
Spec PASS tonight; build + gates + val dry run 10/11 morning (after R-DIAG + E23, ~09:00-11:00); EXT-FINAL unpack + the one call 10/11
~11:00-13:00 (GPU, queue idle); TU VAN PASS of the result; report + web r3 numbers 10/11 afternoon / evening.
