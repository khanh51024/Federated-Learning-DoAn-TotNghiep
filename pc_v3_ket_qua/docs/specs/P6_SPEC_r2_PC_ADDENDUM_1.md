# P6_SPEC r2 PC - ADDENDUM 1 (2026-11-09, after the P6 dry run 1) - ASCII

Finding (dry run 1, TU VAN diagnosis from the dry logits): pc_int8_dynamic_fedavg gave all_pipe_equal = False. int8 VAL[PD] vs VAL_PD_EXTPIPE:
1 / 269 argmax differ, max |delta logit| 0.277. fp32 cpu_parent through the same two pipelines: 0 / 269 differ, max 3.8e-6. So preprocessing is
identical; the cause is that dynamic int8 (quantize_dynamic on Linear) computes the activation scale PER BATCH, so the output depends on the
batch composition (VAL mixes PV + PD, EXTPIPE has PD only, both BATCH 128).

Change (measurement, not a gate): P6 evaluates every int8 entry (format torchscript_int8_dynamic_linear, the 4 e5_int8_dynamic models) with
batch size 1 = one image at a time, as the web app infers; the result is then deterministic per image. fp32 entries keep BATCH 128. The
script records "int8_batch": 1 and "fp32_batch": 128 in INPUTS.json and EVAL_RAW.json. The CPU fp32 parent used for the int8 delta stays
BATCH 128 (fp32 does not depend on batch composition: 0 / 269 differ). The pipe gate and the 2e-3 known-value gate are UNCHANGED.
If dry run 2 shows any int8 known value off by > 2e-3 or all_pipe_equal False: STOP and report, no further change.

Script: pc_r9/p6_final_test_v3.py = p6_final_test.py v2 (4e9ed572) with exactly 4 changes (forward(bs), int8 batch 1 at the call, 2 records).
No other line changed.
