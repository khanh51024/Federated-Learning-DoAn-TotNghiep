# NONIID_DEPLOY_SELECT_SPEC r1 (TU VAN PC, 2026-11-07) - choosing ONE non-IID solution for the web app

Registered BEFORE any S5 seed-45 run, any A1 run, or any C-M FedProx/MOON result. Requested by the user on 2026-11-07:
"one final proposed solution, decided after the comparison data". ASCII.

This is a DEPLOYMENT choice, made on validation only. It is separate from the scientific claims:
- S5 is ruled by WINNER5 (5 seeds, Holm 8).
- A1 is ruled by WINNER (3 seeds, Holm 4).
- Phase C is ruled by R1.
This rule never re-labels those results. The test set (P6) is used only once, AFTER the choice, to report the chosen model.

## 1. Candidates (alpha 0.1 = the web demo setting)
SCAFFOLD, FedProx (mu* 0.01), MOON (mu* 1.0, tau 0.5), FedBN, SCAFFOLD-BN, and the fallback FedAvg.

## 2. Deployable model
The web serves ONE global model to unknown users, so every candidate is compared on its GLOBAL / server model:
- metric G = mean val_macro_f1_raw over rounds 38-40, alpha 0.1;
- G is computed over seeds 42-44, the seeds common to every candidate.
For FedBN and SCAFFOLD-BN the global model is the server model with the n_k-average BN placeholder. Their personalised CDR is reported
next to G but is not used here: it needs a client identity that the web does not have.
PD rule: pd_drop = PD acc(FedAvg) - PD acc(candidate), both global models, rounds 38-40, mean over seeds 42-44.

## 3. Rule (applied in this order; first match wins)
a) If at least one candidate is a registered statistical winner over FedAvg at alpha 0.1 (S5 WINNER5, or A1 WINNER vs FedAvg), choose
   among those winners the one with the largest G.
b) Else, choose the candidate with the largest G, provided that BOTH hold:
   - G(candidate) - G(FedAvg) >= 0.005 (0.5 point, a deploy-relevance floor);
   - pd_drop <= 0.02.
   Its label is "deployment choice by validation; NOT shown to beat FedAvg with statistical significance".
c) Else FedAvg is the deployed model, with the label "no non-IID method beat FedAvg on validation".
Ties in G at 4 decimals: prefer the larger alpha-0.5 G; if still tied, the simpler method in this order:
FedProx < FedBN < SCAFFOLD < MOON < SCAFFOLD-BN.

## 4. What is shipped
The chosen method's alpha-0.1 seed-42 checkpoint_last (round 40). Seed 42 is fixed in advance and never the "best seed". It goes into
web package r3 under the label "giai phap chong non-IID duoc chon (theo luat NONIID_DEPLOY_SELECT r1)". The label carries:
- which branch (a / b / c) applied;
- G and its SD over 3 seeds for the chosen method and for FedAvg;
- the P6 test numbers of the chosen model (reported once, never used to change the choice).

## 5. When
After S5 and A1 have both finished and passed TU VAN review (planned ~2026-11-09 08:00):
1. TU VAN computes the choice with a script.
2. The coordinator rechecks it.
3. P6 runs on 2026-11-10.
4. Web package r3 ships on the evening of 2026-11-10.
If A1 is late, it is still a candidate as soon as its 3 seeds have passed. The choice waits for it only until 2026-11-10 08:00.
After that time, A1 is excluded and this is written in the label.
