# E5_SPEC r1 (PC, 2026-11-09) - N4.1 int8 quantisation of the 4 web models (CPU only, descriptive)

Source: PROMPT_DIEU_PHOI_R3_ADDENDUM_3_YEUCAU.md section B. Registered BEFORE any val number. ASCII.
Script: pc_r9/e5_int8.py sha256 27c1a905435ca27a139d481de48011bc49748f24d9c083ca2970ab3ed5d946ff (LF).
Output: experiments/train-improve-20261015/e5/ (create-once, script asserts it does not exist) -> e5/E5_RESULT.json + per model
e5/<role>/model_{fp32,static_ptq,dynamic_linear}_torchscript.pt.

## 1. Models (web package r2, <PC_PATH>/pc_transfer_r2/web_pc_v3_a01_s42_r2, alpha 0.1, s42, last epoch/round)
centralized_adamw (0a8293a3), centralized_sgd, fedavg (5f923b6d), scaffold (4817fbac). sha256 of each checkpoint_best.pt is asserted
against its model_info.json. MobileNetV3-Small 38 classes, key model_state, strict load.

## 2. Arms
- static_ptq (asked by the addendum): torch.ao FX prepare_fx/convert_fx, get_default_qconfig_mapping("x86"), engine x86
  (per-channel int8 weights, HistogramObserver activations). Calibration: 512 TRAIN images, manifests/train.jsonl,
  random.Random(42).sample(range(37238), 512), canonical_v1 eval transform, batch 32. No val/test in calibration.
- dynamic_linear (added, see s5): torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, qint8). No calibration.
- fp32 reference: same checkpoint, eager and frozen TorchScript.
Every arm is exported as frozen TorchScript (torch.jit.trace + freeze), reloaded, and the reload check (argmax equal on 8 calib
images, max abs logit diff) is recorded. All val metrics of the int8 arms are computed on the RELOADED TorchScript file (what the web
would load).

## 3. Measures
- Size: torch.save(state_dict) bytes per arm, TorchScript file bytes per arm, plus the web checkpoint file bytes (descriptive; the
  AdamW file holds extra state).
- Val: manifests/val.jsonl (4,613 images), dataset class, eval transform, collate and evaluate_metrics imported from the q15 runner
  file (sha 7c29f6f6, not copied), CPU, batch 64. Reported: raw_macro_f1 (38 classes), raw_accuracy, loss, PlantDoc per-source
  numbers; delta arm - fp32.
- Latency: CPU i5-13400F (name recorded from Win32_Processor), batch 1, fixed 1x3x224x224 randn(seed 0), preprocessing excluded,
  20 warm-up + 200 timed forward passes, median, p90, min, max; torch threads 1 and 4; fp32 eager, fp32 TorchScript, both int8
  TorchScript. The like-for-like latency comparison is TorchScript vs TorchScript.
- No test set, no internal_holdout, no EXT. int8 rows go to P6 only if P6 r2 lists them.

## 4. Gate (pipeline check, not a result rule)
G_FP32_REPRO per model: |fp32 CPU raw_macro_f1 - model_info val_macro_f1_at_checkpoint| <= 2e-3 (GPU training-run eval vs CPU eval).
FAIL on a model -> its int8 rows are reported as "unknown (pipeline mismatch)", nothing re-tuned. No winner rule: E5 is descriptive.

## 5. Why dynamic_linear was added (train-only probe, no val touched)
Probe (fedavg, 32 calib + 64 other TRAIN images, scratchpad e5_probe1/e5_probe2): static_ptq argmax agreement with fp32 1/64,
accuracy 0.95 -> 0.02. Diagnosis on random input with the ImageNet weights mobilenet_v3_small-047dcff4: logits rel error 0.96, top-1
agreement 0; error by depth features[:1] 0.08, features[:2] 0.63; same with engines x86, fbgemm and onednn, and with torchvision's
eager QuantizableMobileNetV3 (fuse_model + prepare/convert). A toy conv net quantises fine (rel err ~2%). Activations of the trained
FedAvg model reach |a| 97 after the first BN (q99.9 ~29). Conclusion: plain static PTQ does not work for MobileNetV3-Small in this setup
(torchvision's own int8 MobileNetV3 weights are QAT); no further tuning (time-box, and tuning on val is not allowed).
dynamic_linear on the probe: agreement 64/64, state_dict 4.47 MB vs fp32 6.36 MB (static 1.86 MB).

## 6. Report wording (mandatory)
- static_ptq: report the numbers as measured; if the val macro-F1 collapses, write "PTQ static int8 (torch FX, x86, calib 512 anh train)
  lam sap MobileNetV3-Small: macro-F1 val <x> so voi <y> fp32; khong dua len web; can QAT (ngoai pham vi)".
- dynamic_linear: "int8 dong, chi lop Linear (classifier); conv van fp32" - never "mo hinh int8" without that qualifier.
- Latency is "tren CPU i5-13400F cua PC, batch 1, median 200 lan, khong gom tien xu ly"; not the web server's latency.

## 7. Run
One detached WMI window, after TU VAN PASS, only when the scheduler has no open run (last event of each run is EXIT) and no FLW
window is active (so latency is not disturbed). Expected wall ~20-30 min, RAM < 2 GiB. rc != 0 -> stop, no retry.

## 8. Web (for TU VAN, r3)
int8 files are TorchScript: loader torch.jit.load(path, map_location="cpu"), input 1x3x224x224 canonical_v1, output 38 logits in the
taxonomy.py order. Needs a CPU torch build with quantized engine x86/fbgemm (x86_64). The web repo is not on the PC, so the web backend
(torch version, CPU arch, loader path for TorchScript) is NOT checked here and must be checked by TU VAN before r3.
