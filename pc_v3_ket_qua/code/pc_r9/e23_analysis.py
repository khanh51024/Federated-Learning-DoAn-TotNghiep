"""E3 + E2 analysis (pc_r9/E23_SPEC.md r1). Reads e23/ (from e23_predict.py) + run summaries / telemetry / manifests. CPU only.
Writes e23/E3_RESULT.json, e23/E2_RESULT.json, e23/E3_confusion_top10.csv (create-once).
Usage: python -I -B e23_analysis.py
"""
import ast
import csv
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

E = Path(r"experiments\train-improve-20261015")
DS = Path(r"dataset")
REL = DS / "mixed/pv_pd_v3"
O = E / "e23"
sys.path.insert(0, r"plans\cai-thien-train-20261015\pc_r9")
import e23_predict as P  # MODELS map only (no torch work at import)

ALPHAS = ("01", "05", "1", "10", "100")
SEEDS = (42, 43, 44)
B, BOOT_SEED = 2000, 20261109


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def msd(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": st.mean(xs) if xs else None, "sd": st.stdev(xs) if len(xs) > 1 else None, "n": len(xs)}


def hist(rdir):
    return json.load(open(Path(rdir) / "summary.json", encoding="utf-8"))["history"]


def write_once(p, obj):
    assert not p.exists(), p
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    p.with_suffix(".tmp").write_text(s, encoding="utf-8")
    os.replace(p.with_suffix(".tmp"), p)
    assert "\ufffd" not in p.read_text(encoding="utf-8") and json.loads(p.read_text(encoding="utf-8")) is not None
    return sha256(p)


def macro_f1_present(t, p):
    f = []
    for c in np.unique(t):
        tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


BUF = ("running_mean", "running_var", "num_batches_tracked")


def ldn(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        out[i] = run
    return out


def class_names():
    t = json.load(open(REL / "taxonomy.json", encoding="utf-8"))["class_to_idx"]
    m = {int(v): k for k, v in t.items()}
    assert sorted(m) == list(range(38)), sorted(m)[:5]
    return m


def main():
    pred = json.load(open(O / "E23_PREDICT.json", encoding="utf-8"))
    assert pred["probe"] is False
    ok = {n: v["G_REPRO"]["pass"] and v.get("G_REPRO_pd_last3_pass", True) for n, v in pred["models"].items()}
    lg = {n: np.load(O / "logits" / f"{n}.npz") for n in P.MODELS}
    names = class_names()
    e3 = {"spec": "pc_r9/E23_SPEC.md r1", "predict_sha256": sha256(O / "E23_PREDICT.json"), "script_sha256": sha256(__file__),
          "G_REPRO": ok}
    fl = {f"fedavg_a{a}_s{s}": P.MODELS[f"fedavg_a{a}_s{s}"][1] for a in ALPHAS for s in SEEDS}
    # E3.1 drift, E3.2 comm, E3.4 time
    drift, comm, tim = {}, {"per_run": {}}, {}
    for n, rd in fl.items():
        ev = [json.loads(l) for l in open(rd / "fedavg_telemetry_events.jsonl", encoding="utf-8")]
        ev = [e for e in ev if e.get("event_type") == "client_round_execution"]
        rounds = sorted({e["round"] for e in ev})
        assert rounds == list(range(1, 41)) and len(ev) == 200, (n, len(ev))
        D, CV, S, CW, DP = {}, [], [], [], []
        for r in rounds:
            er = [e for e in ev if e["round"] == r]
            L = [e["delta_l2"] for e in er]
            LN = [ldn(e["layer_delta_norms"]) for e in er]
            C = [math.sqrt(sum(v * v for k, v in d.items() if k.startswith("classifier."))) for d in LN]
            PL = [math.sqrt(sum(v * v for k, v in d.items() if not k.endswith(BUF))) for d in LN]
            DP.append(st.mean(PL))
            D[r] = st.mean(L)
            CV.append(st.pstdev(L) / D[r])
            S.append(st.mean(c / l for c, l in zip(C, L)))
            CW.append(st.mean(e["client_compute_wall_ms"] for e in er))
        drift[n] = {"D_1": D[1], "D_20": D[20], "D_40": D[40], "Dbar": st.mean(D.values()), "CVbar": st.mean(CV), "Sbar": st.mean(S),
                    "DPbar_params_only": st.mean(DP), "DP_1": DP[0], "DP_40": DP[-1]}
        tim[n] = {"mean_client_compute_ms": st.mean(CW), "total_compute_s": sum(e["client_compute_wall_ms"] for e in ev) / 1000.0}
        comm["per_run"][n] = {"down": sorted({e["payload_down_bytes"] for e in ev}), "up": sorted({e["payload_up_bytes"] for e in ev}),
                              "transport_mode": sorted({e.get("transport_mode") for e in ev})}
    e3["E3_1_drift"] = {"per_run": drift, "per_alpha": {a: {k: msd([drift[f"fedavg_a{a}_s{s}"][k] for s in SEEDS])
                                                           for k in ("D_1", "D_20", "D_40", "Dbar", "CVbar", "Sbar", "DPbar_params_only", "DP_1", "DP_40")} for a in ALPHAS}}
    e3["E3_4_time"] = {"per_run": tim, "per_alpha": {a: {k: msd([tim[f"fedavg_a{a}_s{s}"][k] for s in SEEDS])
                                                        for k in ("mean_client_compute_ms", "total_compute_s")} for a in ALPHAS},
                       "label": "PC, simulation sequential trong mot tien trinh; SLOTS 2 dung chung GPU; khong phai thoi gian mang"}
    downs = {x for v in comm["per_run"].values() for x in v["down"]}
    ups = {x for v in comm["per_run"].values() for x in v["up"]}
    assert len(downs) == 1 and len(ups) == 1, (downs, ups)
    d1, u1 = downs.pop(), ups.pop()
    per_round = 5 * (d1 + u1)
    per_run = 40 * per_round
    import torch  # model param count for SCAFFOLD control variates (code: q8r2_methods.scaffold_init over q8 parameter keys)
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 38)
    cv_bytes = 4 * sum(p.numel() for _, p in m.named_parameters())
    raw = 0
    nimg = 0
    for l in open(REL / "manifests/train.jsonl", encoding="utf-8"):
        rec = json.loads(l)
        raw += (DS / rec["relative_path"]).stat().st_size
        nimg += 1
    assert nimg == 37238, nimg
    fmt = lambda b: {"bytes": b, "MB": b / 1e6, "GiB": b / 2 ** 30}
    comm.update({"per_client_round_down": fmt(d1), "per_client_round_up": fmt(u1), "per_round_5_clients": fmt(per_round),
                 "per_run_40_rounds": fmt(per_run),
                 "scaffold_extra_per_client_round_per_direction": fmt(cv_bytes),
                 "scaffold_per_run_40_rounds": fmt(per_run + 40 * 5 * 2 * cv_bytes),
                 "scaffold_note": "c (down) and c_i+ - c_i (up), float32 per trainable parameter (q8r2_methods.scaffold_init/"
                                  "scaffold_client_update/scaffold_server_update); not in the simulation telemetry",
                 "raw_train_images": {"n": nimg, **fmt(raw)}, "ratio_fedavg_run_over_raw": per_run / raw})
    del comm["per_run"]
    e3["E3_2_comm"] = comm
    # E3.3 rounds to 95 / 99 %
    r9 = {}
    for n, rd in fl.items():
        h = {x["round"]: x["val_macro_f1_raw"] for x in hist(rd)}
        end = st.mean(h[r] for r in (38, 39, 40))
        r9[n] = {"endpoint": end, **{f"R{int(q * 100)}": next((r for r in sorted(h) if h[r] >= q * end), "not reached")
                                     for q in (0.95, 0.99)}}
    e3["E3_3_rounds_to"] = {"per_run": r9, "per_alpha": {a: {k: msd([r9[f"fedavg_a{a}_s{s}"][k] for s in SEEDS
                                                                    if isinstance(r9[f"fedavg_a{a}_s{s}"][k], int)])
                                                            for k in ("R95", "R99")} for a in ALPHAS}}
    # E3.5 confusion
    fams = {"central_adamw": [f"central_adamw_s{s}" for s in SEEDS], **{f"fedavg_a{a}": [f"fedavg_a{a}_s{s}" for s in SEEDS] for a in ALPHAS}}
    conf, rows = {}, []
    for fam, ms in fams.items():
        for sub in ("all", "plantdoc"):
            M = np.zeros((38, 38), dtype=np.int64)
            for n in ms:
                z = lg[n]
                sel = np.ones(len(z["targets"]), bool) if sub == "all" else z["plantdoc"]
                np.add.at(M, (z["targets"][sel], z["logits"][sel].argmax(1)), 1)
            off = [(int(M[i, j]), i, j) for i in range(38) for j in range(38) if i != j and M[i, j]]
            top = sorted(off, reverse=True)[:10]
            conf[f"{fam}_{sub}"] = {"matrix": M.tolist(), "top10": [{"true": names[i], "pred": names[j], "count": c,
                                                                     "share_of_true": c / M[i].sum()} for c, i, j in top]}
            rows += [[fam, sub, k + 1, names[i], names[j], c, round(c / M[i].sum(), 4)] for k, (c, i, j) in enumerate(top)]
    e3["E3_5_confusion"] = conf
    # E3.6 PD bootstrap (paired index matrix)
    any_m = next(iter(lg.values()))
    pd_mask = any_m["plantdoc"]
    npd = int(pd_mask.sum())
    assert npd == 269
    for n in lg:
        assert np.array_equal(lg[n]["plantdoc"], pd_mask) and np.array_equal(lg[n]["targets"], any_m["targets"])
    idx = np.random.default_rng(BOOT_SEED).integers(0, npd, size=(B, npd))
    tpd = any_m["targets"][pd_mask]
    boot = {}
    for n in P.MODELS:
        ppd = lg[n]["logits"][pd_mask].argmax(1)
        acc = [float(np.mean(ppd[ix] == tpd[ix])) for ix in idx]
        f1 = [macro_f1_present(tpd[ix], ppd[ix]) for ix in idx]
        boot[n] = {"pd_accuracy": float(np.mean(ppd == tpd)), "pd_accuracy_ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
                   "pd_macro_f1": macro_f1_present(tpd, ppd), "pd_macro_f1_ci95": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
                   "G_REPRO": ok[n]}
    fam_all = {**fams, "central_sgd": [f"central_sgd_s{s}" for s in SEEDS]}
    e3["E3_6_pd_bootstrap"] = {"B": B, "seed": BOOT_SEED, "paired_index_matrix": True, "per_model": boot,
                               "per_family": {f: {k: {"mean": st.mean(boot[n][k] for n in ms), "min": min(boot[n][k] for n in ms),
                                                      "max": max(boot[n][k] for n in ms)} for k in ("pd_accuracy", "pd_macro_f1")}
                                              for f, ms in fam_all.items()}}
    # E2 (a)
    e2 = {"spec": "pc_r9/E23_SPEC.md r1", "predict_sha256": e3["predict_sha256"], "script_sha256": e3["script_sha256"]}
    a = {}
    for fam in ("central_adamw", "central_sgd"):
        for s in SEEDS:
            n = f"{fam}_s{s}"
            l3 = pred["models"][n]["central_pd_last3"]
            hh = {x["epoch"]: x for x in hist(P.MODELS[n][1])}
            a[n] = {"epochs": [x["epoch"] for x in l3], "pd_macro_f1": st.mean(hh[x["epoch"]]["plantdoc_macro_f1_supported"] for x in l3),
                    "pd_accuracy": st.mean(x["pd_accuracy"] for x in l3), "G_REPRO": ok[n]}
    for s in SEEDS:
        n = f"fedavg_a100_s{s}"
        hh = {x["round"]: x for x in hist(P.MODELS[n][1])}
        a[n] = {"rounds": [38, 39, 40], "pd_macro_f1": st.mean(hh[r]["plantdoc_macro_f1_supported"] for r in (38, 39, 40)),
                "pd_accuracy": st.mean(hh[r]["plantdoc_accuracy"] for r in (38, 39, 40)), "G_REPRO": ok[n]}
    fam_a = {f: {k: msd([a[f"{f}_s{s}"][k] for s in SEEDS]) for k in ("pd_macro_f1", "pd_accuracy")}
             for f in ("central_adamw", "central_sgd", "fedavg_a100")}
    gaps = {f: {k: msd([a[f"{f}_s{s}"][k] - a[f"fedavg_a100_s{s}"][k] for s in SEEDS]) for k in ("pd_macro_f1", "pd_accuracy")}
            for f in ("central_adamw", "central_sgd")}
    e2["a_last3"] = {"per_run": a, "per_family": fam_a, "paired_gap_central_minus_fedavg_a100": gaps}
    # E2 (d)
    tests = []
    for f in ("central_adamw", "central_sgd"):
        for s in SEEDS:
            x = lg[f"{f}_s{s}"]["logits"][pd_mask].argmax(1) == tpd
            y = lg[f"fedavg_a100_s{s}"]["logits"][pd_mask].argmax(1) == tpd
            b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
            tests.append({"pair": f"{f}_s{s} vs fedavg_a100_s{s}", "b_central_only": b, "c_fedavg_only": c,
                          "acc_central": float(x.mean()), "acc_fedavg": float(y.mean()), "p": mcnemar(b, c),
                          "G_REPRO": ok[f"{f}_s{s}"] and ok[f"fedavg_a100_s{s}"]})
    for t, ph in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = ph
        t["reject_0_05"] = ph < 0.05
    e2["d_mcnemar"] = {"n_pd": npd, "tests": tests, "holm_family": 6}
    assert not (O / "E3_confusion_top10.csv").exists()
    with open(O / "E3_confusion_top10.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "subset", "rank", "true", "pred", "count", "share_of_true"])
        w.writerows(rows)
    print(json.dumps({"E3": write_once(O / "E3_RESULT.json", e3), "E2": write_once(O / "E2_RESULT.json", e2),
                      "csv": sha256(O / "E3_confusion_top10.csv"), "G_REPRO_all": all(ok.values())}))


if __name__ == "__main__":
    main()
