# FLW_SPEC addendum A3 (PC, 2026-11-09 ~12:4x) - A3 FedBN verdict; base FLW_SPEC r2 1e9872a7 s4 A3 (unchanged)

Decided by TU VAN PC (option ii), registered BEFORE any A3 data. ASCII. Window <PC_PATH>/q15proto/flw_a3_window.sh 46ff4d6f (not edited).

## 1. Reason
q18 runner (q18/sandbox/fedavg_mixed_runner.py):
- line 1474: `q8r2_last_round = args.stop_after_round if args.stop_after_round is not None else num_rounds`
- line 1689-1692: `"q8r2": (q8r2_methods.fedbn_round_eval(...) if (q8r2_local_bn and r == q8r2_last_round) else {"scaffold_c_norm": ...})`
With --stop-after-round 5 the FedBN round-end eval (CDR + PD uniform) runs at round 5 of the Flower run; the archived 40-round
run has `{"scaffold_c_norm": None}` at round 5. That key has no counterpart in the archived run, so keeping it makes "bit-exact"
impossible whatever the data. The eval is built after val_res of round 5 (round 5 metrics unaffected) and the run stops at round 5
(the RNG it consumes affects no later row). Dropping it is not a loosened gate.

## 2. Official verdict
FedBN (c_m_fedbn_a01_s42 vs flower_exact r3, rows 1-5):
- "bit-exact" iff rows 1-5 equal after the registered timing strip, dropping ONLY key q8r2 of row 5 on both sides
  (rows 1-4 compared including q8r2);
- else "equivalent (1e-6)" iff max |d val_macro_f1_raw| over rows 1-5 <= 1e-6;
- else "NOT EQUIVALENT".
FedProx: rule of FLW_SPEC r2 s4 unchanged (row 5 compared in full).
The descriptive plain-ref arm runs when the official verdict is NOT EQUIVALENT (same max |d| mapping as the window's v, so the
window triggers it correctly). The ledger records the official verdict from v (FedProx) / v2 = verdict_excl_q8r2_r5 (FedBN) in
the window log; the script is not changed.

## 3. Descriptive only (not in the verdict), both methods
model_state of run/checkpoints/gen_000005/checkpoint_last.pt (Flower) vs the archived gen_000005/checkpoint_last.pt (round 5):
torch.equal per tensor + max abs diff. The Flower q8r2 block of round 5 is descriptive only and is NEVER placed next to the
S5 FedBN round-40 numbers.

## 4. Report wording (FedBN, if met)
"bit-exact tren 5 round (tru khoi danh gia FedBN q8r2, von chi sinh o round dung)" + the FedBN label of FLW_SPEC r2 s4 verbatim:
"local BN kept by the runner on the server side and shipped to the client in the message (simulation); the transport does not
realise FedBN's BN locality".
