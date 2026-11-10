"""E1 per-client fairness (pc_r9/E1_SPEC.md r1 d28d8ce6). CPU, read-only on run dirs, no inference. acc_k = sum_c p_k(c) * recall_c,
p_k = q8r2_methods.class_share of client k's train shard (loaded with the pinned q18 runner dataset class through fedbn_compare.load_runner,
Q18 lock checked), recall_c from history rounds 38-40 (FedAvg global: PD + PV per-class recall merged by support; Local-only: client k's
own model). Class index = position in the contract class list (sorted names), checked by G_MAP.
Writes <out_dir>/E1_RESULT.json + E1_per_client.csv + E1_summary.csv (create-once).
Usage: python -I -B e1_fairness.py <out_dir> [--selftest-only]
"""
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
H = WS / "plans/cai-thien-train-20261015/pc_r9"
REL = WS / "dataset/mixed/pv_pd_v3"
SEEDS = (42, 43, 44)
ROUNDS = (38, 39, 40)
ALPHA = {"0.1": ("label_alpha_0_1", "a01"), "0.5": ("label_alpha_0_5", "a05"), "1": ("label_alpha_1_0", "a1"),
         "10": ("label_alpha_10_0", "a10"), "100": ("label_alpha_100_0", "a100")}
VAL_N = 4613


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fedavg_dir(al, s):
    if al == "0.1":
        return E / f"q15l/fedavg_s{s}_out/fedavg_s{s}/run"
    if al == "0.5":
        return E / f"q16l/fedavg_a05_s{s}_out/fedavg_a05_s{s}/run"
    t = ALPHA[al][1]
    return E / f"q17l/fedavg_{t}_s{s}_out/fedavg_{t}_s{s}/run"


def lo_dir(al, s):
    return E / f"lo17l/lo_{ALPHA[al][1]}_s{s}_out/run"


def shard_dir(al):
    sch = ALPHA[al][0]
    return E / f"partitions_v5/partitions/{sch}" if al == "100" else REL / f"partitions/{sch}"


def acc_k(p, rec, sup):
    keep = [c for c in range(len(p)) if sup[c] > 0]
    return sum(p[c] * rec[c] for c in keep), sum(p[c] for c in range(len(p)) if sup[c] == 0)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0


def selftest():
    a, m = acc_k([0.5, 0.3, 0.2], [1.0, 0.5, 0.0], [1, 1, 1])
    assert abs(a - 0.65) < 1e-15 and m == 0.0, a
    assert abs(sd([0.6, 0.7, 0.8, 0.9, 1.0]) - 0.158114) < 1e-6
    return {"toy_acc": a, "toy_sd": sd([0.6, 0.7, 0.8, 0.9, 1.0]), "pass": True}


def fedavg_recall(hist_row, classes):
    sup, cor = {}, {}
    for src, d in hist_row["quality"]["by_source"].items():
        for n, x in d["per_class_recall"].items():
            sup[n] = sup.get(n, 0) + x["support"]
            cor[n] = cor.get(n, 0.0) + x["support"] * x["recall"]
    assert set(sup) <= set(classes), set(sup) - set(classes)
    s = [sup.get(n, 0) for n in classes]
    r = [cor[n] / sup[n] if sup.get(n, 0) else 0.0 for n in classes]
    return r, s


def lo_recall(row):
    v = row["per_class_recall"]
    s = [v[str(i)]["support"] for i in range(38)]
    r = [v[str(i)]["recall"] for i in range(38)]
    return r, s


def main(argv):
    out = Path(argv[0])
    st = selftest()
    if "--selftest-only" in argv:
        print(json.dumps(st))
        return 0
    assert not out.exists(), out
    s_ = importlib.util.spec_from_file_location("fedbn_compare", H / "fedbn_compare.py")
    F = importlib.util.module_from_spec(s_)
    s_.loader.exec_module(F)
    m, q = F.load_runner()
    sys.path.insert(0, str(E / "q18/sandbox/plant_data_contract"))
    from plant_data_contract.taxonomy import PLANTVILLAGE_38_CLASSES as CL
    classes = list(CL)
    assert len(classes) == 38 and classes == sorted(classes)
    pk, nk = {}, {}
    for al in ALPHA:
        pk[al], nk[al] = [], []
        for k in range(5):
            cds = m.CanonicalClassificationDataset.from_manifest(shard_dir(al) / f"client_{k:02d}.jsonl", dataset_root=WS / "dataset",
                                                                 transform=None, return_metadata=True)
            pk[al].append(q.class_share(cds.targets, 38))
            nk[al].append(len(cds))
    rows, gates, cells = [], {}, {}
    for al in ALPHA:
        for s in SEEDS:
            for meth in ("fedavg", "local_only"):
                key = f"{meth}|{al}|{s}"
                g = {"G_PK_sum": all(abs(sum(p) - 1) <= 1e-12 for p in pk[al])}
                per_round = []
                if meth == "fedavg":
                    d = fedavg_dir(al, s)
                    hist = json.loads((d / "summary.json").read_text(encoding="utf-8"))["history"]
                    by_r = {h["round"]: h for h in hist}
                    tel = sorted(by_r[40]["client_telemetry"], key=lambda c: c["client_id"])
                    g["G_PK_nk"] = [c["raw_images_n_k"] for c in tel] == nk[al]
                    g_map, g_rec = True, True
                    for r in ROUNDS:
                        rec, sup = fedavg_recall(by_r[r], classes)
                        g_rec = g_rec and sum(sup) == VAL_N and abs(sum(a * b for a, b in zip(sup, rec)) / VAL_N - by_r[r]["val_accuracy"]) <= 5e-5
                        per_round.append([(rec, sup)] * 5)
                    lo0 = json.loads((lo_dir(al, s) / "client_00/history.json").read_text(encoding="utf-8"))
                    g_map = per_round[0][0][1] == lo_recall([x for x in lo0 if x["round"] == 40][0])[1]
                    g["G_MAP"], g["G_REC"] = g_map, g_rec
                    src = str(d)
                else:
                    d = lo_dir(al, s)
                    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
                    g["G_PK_nk"] = [summ["clients"][f"client_{k:02d}"]["n_k"] for k in range(5)] == nk[al]
                    hs = [{x["round"]: x for x in json.loads((d / f"client_{k:02d}/history.json").read_text(encoding="utf-8"))} for k in range(5)]
                    g_rec, g_map = True, True
                    for r in ROUNDS:
                        rr = []
                        for k in range(5):
                            rec, sup = lo_recall(hs[k][r])
                            g_rec = g_rec and sum(sup) == VAL_N and abs(sum(a * b for a, b in zip(sup, rec)) / VAL_N - hs[k][r]["val_accuracy"]) <= 1e-9
                            rr.append((rec, sup))
                        per_round.append(rr)
                    g["G_MAP"], g["G_REC"] = g_map, g_rec
                    src = str(d)
                accs, mass = [], []
                for k in range(5):
                    v = [acc_k(pk[al][k], per_round[i][k][0], per_round[i][k][1]) for i in range(len(ROUNDS))]
                    accs.append(sum(a for a, _ in v) / len(v))
                    mass.append(v[0][1])
                ok = all(g.values())
                gates[key] = g
                for k in range(5):
                    rows.append({"method": meth, "alpha": al, "seed": s, "client": f"client_{k:02d}", "n_k": nk[al][k], "acc_k": accs[k],
                                 "excluded_mass": mass[k], "valid": ok})
                w = min(range(5), key=lambda k: accs[k])
                cells[key] = {"valid": ok, "source": src, "mean": statistics.mean(accs), "sd": sd(accs), "min": accs[w],
                              "worst_client": f"client_{w:02d}", "worst_n_k": nk[al][w], "max": max(accs), "range": max(accs) - min(accs),
                              "nk_weighted_mean": sum(a * n for a, n in zip(accs, nk[al])) / sum(nk[al]), "acc_k": accs}
    summary = {}
    for meth in ("fedavg", "local_only"):
        for al in ALPHA:
            cs = [cells[f"{meth}|{al}|{s}"] for s in SEEDS]
            if not all(c["valid"] for c in cs):
                summary[f"{meth}|{al}"] = "unknown"
                continue
            summary[f"{meth}|{al}"] = {st_: {"mean": statistics.mean(c[st_] for c in cs), "sd": sd([c[st_] for c in cs]),
                                             "min": min(c[st_] for c in cs), "max": max(c[st_] for c in cs)}
                                       for st_ in ("mean", "sd", "min", "range", "nk_weighted_mean")}
    gap = {}
    for al in ALPHA:
        for s in SEEDS:
            a, b = cells[f"fedavg|{al}|{s}"], cells[f"local_only|{al}|{s}"]
            dk = [x - y for x, y in zip(a["acc_k"], b["acc_k"])]
            gap[f"{al}|{s}"] = {"mean": statistics.mean(dk), "sd": sd(dk), "n_pos": sum(x > 0 for x in dk), "valid": a["valid"] and b["valid"]}
    doc = {"stage": "E1 per-client fairness (E1_SPEC r1)", "spec_sha256": sha(H / "E1_SPEC.md"), "harness_sha256": sha(__file__),
           "fedbn_compare_sha256": sha(H / "fedbn_compare.py"), "labels": ["validation", "PC", "TB round 38-40"],
           "weighted_macro_f1": "unknown (logs carry per-class recall only)", "selftest": st, "gates": gates, "cells": cells,
           "summary": summary, "gap_fedavg_minus_lo": gap, "p_k": pk, "n_k": nk}
    out.mkdir(parents=True)
    b = json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")
    (out / "E1_RESULT.json").write_bytes(b)
    assert (out / "E1_RESULT.json").read_bytes() == b
    with open(out / "E1_per_client.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "E1_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "alpha", "stat", "mean_over_seeds", "sd_over_seeds", "min_over_seeds", "max_over_seeds"])
        for k, v in summary.items():
            meth, al = k.split("|")
            if v == "unknown":
                wr.writerow([meth, al, "unknown", "", "", "", ""])
                continue
            for st_, x in v.items():
                wr.writerow([meth, al, st_, x["mean"], x["sd"], x["min"], x["max"]])
    print(json.dumps({"all_gates": all(all(g.values()) for g in gates.values()), "sha256": sha(out / "E1_RESULT.json"),
                      "summary_mean": {k: (v if v == "unknown" else round(v["mean"]["mean"], 4)) for k, v in summary.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
