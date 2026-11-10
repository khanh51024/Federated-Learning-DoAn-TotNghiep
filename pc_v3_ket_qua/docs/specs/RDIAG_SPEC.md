# RDIAG_SPEC r1 (PC, 2026-11-09) - R-DIAG: PlantDoc gap at alpha 100, parts (b) and (c) (descriptive diagnostic)

Source: PROMPT_DIEU_PHOI_R3_20261107.md s6 item 11 (lowest priority, first in the cut order). Registered BEFORE any R-DIAG data. ASCII.
TU VAN conditions (msg 2026-11-09 ~15:00): START only after E5 EXIT; must not delay E3 / E2 / P6 r2 (20:00); no PASS before 18:00 ->
cut; every run must end before 2026-11-10 08:00 (ETA from real s/round at SLOTS 2); E2/E3 beside R-DIAG only with ram_guard >= 3 GiB free.

## 1. Runs (6, P40: 40 rounds, 5 clients, alpha 100 = label_alpha_100_0 of partitions_v5, partition spec 3cd41f7c, digest c77a0030)
| run | job | runner | method | PlantDoc sampling |
|---|---|---|---|---|
| rd_fedavg_pdnat_a100_s{42,43,44} | fed_mixed_label_a100_p40_s<seed> | q17 sandbox (c310f114, Q17 lock 2f71e6ff) = the baseline's runner | fedavg | natural (--plantdoc-sampling-ratio none) |
| rd_fedbn_a100_s{42,43,44} | fed_mixed_label_a100_p40_s<seed> | q18 sandbox (47b31c6a, Q18 lock 3a5bfdfe) = the C-M FedBN runner | fedbn | 0.25 (as baseline) |
(b) = "PD sampled at its global proportion": at alpha 100 every client holds ~ the global PD share, so plain shuffling (no domain-balanced
sampler) = the global proportion. It needs the runner's READY_FULL gate to accept ratio 0, which it does only for selected_variant
"pretrained": spec copy rdiag/specs/fedavg_spec_rdiag_pdnat.json (d8e8268c) = q17 spec 1f8a3958 with ONLY that one value changed.
Everything else (seed, W0, lr, rounds, AMP, quality gate, incumbent, partitions) = baseline.

## 2. Code (built, not run)
build_rdiag.py bc964bf5 -> rdiag_laptop_kaggle.py d3a2f6d1 (from q22 kaggle d40e5c5f: + runner q17, RUNS = 6, digest + label_alpha_100_0,
per-run spec / ratio; validity unchanged), rdiag_laptop_driver.py 04198894 (from q22 driver 5dc21278: + Q17 lock + R-DIAG lock checks,
output experiments/train-improve-20261015/rdiagl), lock rdiag/RDIAG_STAGE_LOCK.json 6564779c. Queue append = copy of
queue_append_q22.py with the 6 names (dry run first), order seed-major: pdnat s42, fedbn s42, pdnat s43, fedbn s43, pdnat s44, fedbn s44.

## 3. Pre-run check (after E5 EXIT, before append)
Runner --preflight-only in full mode for the exact commands of rd_fedavg_pdnat_a100_s42 and rd_fedbn_a100_s42 (no output dir is created
in preflight). PASS iff "PREFLIGHT CHECK OK". FAIL of (b) -> (b) dropped and reported, (c) goes alone. No code change to rescue it.

## 4. Baseline (existing, not rerun)
pc_q17_fedavg_a100_s42/43/44 (q17l): endpoint val macro-F1 r38-40 0.9408 / 0.9406 / 0.9395; PD macro-F1 supported r38-40
0.4775 / 0.4724 / 0.5020; wall 1.37-1.42 h at SLOTS 2.

## 5. Measures (descriptive only; no test, no winner rule, no p-value)
Per run (RESULT.json per_round): endpoint = mean r38-40 of val_macro_f1_raw; PD = mean r38-40 of plantdoc_macro_f1_supported and
PD accuracy (summary history plantdoc accuracy, same rounds). Per arm: mean +- SD over 3 seeds, paired d vs the same-seed baseline,
count of seeds with d > 0. FedBN additionally: the q8r2 round-40 block (personalised CDR / PD uniform) as in C-M, labelled personalised,
never placed next to the server-model numbers as the same quantity. Validity = the kaggle validity (unchanged); invalid run -> its row
"unknown", no rerun.

## 6. Report wording (mandatory)
"R-DIAG (chan doan mo ta, 3 seed, khong kiem dinh): o alpha 100, lay mau PD theo ti le tu nhien thay vi 0,25 doi PD macro-F1 <d +- sd>;
FedBN (BN cuc bo) doi <...>". Never "explains the gap" / "fixes the gap".

## 7. Schedule and cut
Append right after the s3 check (target ~16:00-17:00), SLOTS 2, normal queue. 6 x ~1.45 h in 3 waves of 2 -> ~4.5 h. Before append:
now + 4.6 h must be <= 2026-11-10 08:00, else append only what fits, by the order above (ETA only). Spec not PASSED by 18:00 -> whole
R-DIAG cut. queue_watch watches as usual.
