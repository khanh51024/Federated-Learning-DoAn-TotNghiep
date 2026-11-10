# E1_SPEC (PC, r1, 2026-11-07) - N2.3 per-client fairness (lecturer, Dot 2 acceptance) - author: DIEU PHOI PC r3; TU VAN PASS needed

Compute-only (no training, no checkpoint inference), CPU BelowNormal, read-only on run dirs. Formula registered BEFORE any number is
computed. Validation only (4,613 images = PD 269 + PV 4,344); no EXT / test / internal_holdout read.

## 1. Inputs (existing PC runs, all TU VAN PASS)
FedAvg (global model), 5 alpha x seeds 42-44: alpha 0.1 q15l/fedavg_s<s>, 0.5 q16l/fedavg_a05_s<s>, 1 / 10 / 100
q17l/fedavg_a<1|10|100>_s<s> (run dir <run>_out/<run>/run/summary.json).
Local-only (5 personalised models), same alpha x seeds: lo17l/lo_a<01|05|1|10|100>_s<s>_out/run/client_0k/history.json.
Shards: dataset/mixed/pv_pd_v3/partitions/<scheme>/client_0k.jsonl (label_alpha_0_1 / 0_5 / 1_0 / 10_0); alpha 100 from
E/partitions_v5/partitions/label_alpha_100_0 (rooted, partition spec 3cd41f7c). Loaded with the pinned q18 runner dataset class (as fedbn_compare,
Q18 lock checked), class shares with q8r2_methods.class_share (the function already gated by G_CDR). No re-implementation.

## 2. Formula
For client k, p_k(c) = share of class c in client k's TRAIN shard (class_share, 38 classes, sum 1). recall_c = recall of class c on the
common validation set. Classes with zero validation support are excluded and the excluded mass m_k = sum of their p_k(c) is reported
(expected 0: every class has val support).
  acc_k = sum_c p_k(c) * recall_c           (expected accuracy of the model on client k's own label distribution)
- FedAvg: recall_c of the GLOBAL model, per round r in {38, 39, 40}, then acc_k averaged over the 3 rounds (acc_k is linear in recall,
  so this equals acc_k of the averaged recall). recall_c per round = (sum over sources of support_c,src * recall_c,src) / support_c from
  history[r].quality.by_source.{plantdoc, plantvillage}.per_class_recall (exact: recall = correct / support).
- Local-only: recall_c of client k's OWN model (client_0k/history.json per_class_recall, rounds 38-40), same averaging.
- Class index <-> name: index = position in the sorted class-name list (contract order); checked by gate G_MAP below.
- Weighted macro-F1 (lecturer "neu du du lieu"): needs per-class precision, the logs carry recall only -> reported "unknown". No
  checkpoint re-inference in E1.

## 3. Statistics (per method x alpha x seed, over the 5 clients)
mean_k acc_k (uniform over clients), SD_k (sample SD, n - 1 = 4), worst client (argmin acc_k: client id, n_k, acc_k), range max - min,
n_k-weighted mean (descriptive). Then per method x alpha over the 3 seeds: mean +- SD (n - 1 = 2) of each statistic, and min / max over
seeds. Also the per-client gap FedAvg - Local-only (acc_k^FedAvg - acc_k^LO, same k, same seed): mean, SD, number of clients with gap > 0.
All numbers raw (no rounding in JSON / CSV; tables round to 4 decimals). No test, no claim: E1 is descriptive (no gate on the outcome).

## 4. Gates (before the numbers are used; any fail -> that cell "unknown")
G_MAP   sorted-name support vector of FedAvg (PD + PV summed) == Local-only index support vector (38 entries, total 4,613), every run.
G_REC   FedAvg: sum_c support_c * recall_c / 4,613 == history val_accuracy (|diff| <= 5e-5, history is rounded to 4 decimals);
        Local-only: same against the client history val_accuracy (|diff| <= 1e-9).
G_PK    sum_c p_k(c) == 1 (|.| <= 1e-12); n_k of the shards == the registry n_k of the run (FedAvg telemetry round 40 raw n_k;
        Local-only ledger n_k); partition digest of the scheme == registry where the run records it.
G_SELF  self-test: a 3-class toy (p = [0.5, 0.3, 0.2], recall = [1.0, 0.5, 0.0]) -> acc = 0.65; SD of [0.6, 0.7, 0.8, 0.9, 1.0] = 0.158114.

## 5. Output
E/e1/E1_RESULT.json (create-once, sha in ledger) + E1_per_client.csv + E1_summary.csv, harness pc_r9/e1_fairness.py (sha), labels
"validation", "PC", "TB round 38-40". Spec PASS -> run -> TU VAN PASS of the result.
