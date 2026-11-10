"""E4 server / client weight divergence (pc_r9/E4_SPEC.md r1 444382ea). CPU only, read-only, no inference; one snapshot in memory at a time.
D1 = ||w_k - g1|| / ||g1||, D3 = cos(w_k - g0, g1 - g0), D4 = per-layer D1 at the registered rounds R = {1, 2, 3, 40}; D2 = ||w_k - g0|| at
every round from telemetry (non-buffer layer norms); D5 = ||w_FedAvg - w_CentralSGD|| / ||w_CentralSGD|| (checkpoint_last, same seed).
w = float tensors without BN buffers (running_mean / running_var / num_batches_tracked); BN buffers reported separately; FedBN: BN affine
excluded too (local by design). Other snapshot rounds of a run (is_best rounds) are written as run-specific descriptive rows only.
Writes <out_dir>/E4_RESULT.json, E4_D1_D3_D4.csv, E4_D2_rounds.csv, E4_D5.csv, E4_D2_by_round_<alpha>.png, E4_D1_D3_by_alpha.png.
Usage: python -I -B e4_divergence.py <out_dir> [--selftest-only]
"""
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
H = WS / "plans/cai-thien-train-20261015/pc_r9"
R = (1, 2, 3, 40)
SEEDS = (42, 43, 44)
W0FP = "eaf9197b42fb39163186a7285f7f1ed4f1e728bd83522260fdc5f0f51c110850"
BUF = ("running_mean", "running_var", "num_batches_tracked")
A_TAG = {"0.1": "a01", "0.5": "a05", "1": "a1", "10": "a10", "100": "a100"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs():
    out = []
    for s in SEEDS:
        out.append(("fedavg", "0.1", s, E / f"q15l/fedavg_s{s}_out/fedavg_s{s}/run"))
        out.append(("fedavg", "0.5", s, E / f"q16l/fedavg_a05_s{s}_out/fedavg_a05_s{s}/run"))
        for al in ("1", "10", "100"):
            t = A_TAG[al]
            out.append(("fedavg", al, s, E / f"q17l/fedavg_{t}_s{s}_out/fedavg_{t}_s{s}/run"))
        out.append(("scaffold", "0.1", s, E / f"q15l/scaffold_s{s}_out/scaffold_s{s}/run"))
        out.append(("scaffold", "0.5", s, E / f"q16l/scaffold_a05_s{s}_out/scaffold_a05_s{s}/run"))
        for al, t in (("0.1", "a01"), ("0.5", "a05")):
            out.append(("fedbn", al, s, E / f"q19l/c_m_fedbn_{t}_s{s}_out/c_m_fedbn_{t}_s{s}/run"))
    return out


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0


def selftest(torch):
    g0 = torch.zeros(2, dtype=torch.float64)
    w = [torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.0, 1.0], dtype=torch.float64)]
    g1 = 0.5 * w[0] + 0.5 * w[1]
    d1 = float((w[0] - g1).norm() / g1.norm())
    d3 = float(torch.dot(w[0] - g0, g1 - g0) / ((w[0] - g0).norm() * (g1 - g0).norm()))
    assert abs(d1 - 1.0) < 1e-12 and abs(d3 - 0.707107) < 1e-6, (d1, d3)
    return {"D1": d1, "D3": d3, "pass": True}


def groups(keys):
    bn = {k[:-len(".running_mean")] for k in keys if k.endswith(".running_mean")}
    buf = [k for k in keys if k.endswith(BUF)]
    bn_aff = [k for k in keys if k.rsplit(".", 1)[0] in bn and k.endswith((".weight", ".bias"))]
    return buf, bn_aff


def main(argv):
    import torch
    st = selftest(torch)
    if "--selftest-only" in argv:
        print(json.dumps(st))
        return 0
    out = Path(argv[0])
    assert not out.exists(), out
    torch.set_num_threads(2)

    def vec(sd_, names):
        return torch.cat([sd_[n].detach().to(torch.float64).reshape(-1) for n in names])

    rows, d2rows, runinfo, extra = [], [], {}, []
    for meth, al, s, d in runs():
        key = f"{meth}|{al}|{s}"
        info = {"dir": str(d), "gates": {}}
        snaps = {int(re.search(r"(\d{4})", f.name).group(1)): f for f in d.glob("client_server_snapshots_round_*.pt")}
        info["snapshot_rounds"] = sorted(snaps)
        miss = [r for r in R if r not in snaps]
        tel = [json.loads(l) for l in (d / "fedavg_telemetry_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        tel = [t for t in tel if t.get("event_type") == "client_round_execution"]
        telmap = {(t["round"], t["client_id"]): t for t in tel}
        g_snap, g_agg, g_d2, raw_match, g_agg_old = not miss, [], [], set(), []
        for r in sorted(snaps):
            sn = torch.load(snaps[r], map_location="cpu", weights_only=False)
            g0, g1, cs = sn["server_state_before"], sn["server_state_after"], sn["client_states"]
            keys = list(g0)
            buf, bn_aff = groups(keys)
            fl = [k for k in keys if k not in buf and g0[k].is_floating_point()]
            par = [k for k in fl if not (meth == "fedbn" and k in bn_aff)]
            bufv = [k for k in buf if g0[k].is_floating_point()]
            cids = sorted(cs)
            ok = (len(cids) == 5 and all(set(cs[c]) == set(keys) for c in cids) and set(g1) == set(keys)
                  and abs(sum(sn["p_k"]) - 1) <= 1e-12
                  and [telmap.get((r, c), {}).get("raw_images_n_k") for c in cids] == list(sn["n_k"]))
            if r in R:
                g_snap = g_snap and ok
            G0, G1 = vec(g0, par), vec(g1, par)
            G = G1 - G0
            B1 = vec(g1, bufv)
            deltas = []
            for i, c in enumerate(cids):
                Wk = vec(cs[c], par)
                Dk = Wk - G0
                deltas.append(Dk)
                dn = float(Dk.norm())
                d1 = float((Wk - G1).norm() / G1.norm())
                d3 = float(torch.dot(Dk, G) / (Dk.norm() * G.norm())) if float(G.norm()) > 0 and dn > 0 else None
                d1b = float((vec(cs[c], bufv) - B1).norm() / B1.norm()) if bufv else None
                row = {"method": meth, "alpha": al, "seed": s, "round": r, "client": c, "n_k": sn["n_k"][i], "p_k": sn["p_k"][i],
                       "D1": d1, "D3": d3, "D1_bn_buffers": d1b, "delta_norm_snapshot": dn, "registered_round": r in R}
                if r in R:
                    lay = {k: float((cs[c][k].double() - g1[k].double()).norm() / g1[k].double().norm()) if float(g1[k].double().norm()) > 0
                           else None for k in par}
                    row["D4"] = lay
                    t = telmap.get((r, c))
                    if t:
                        ln = t["layer_delta_norms"]
                        tp = math.sqrt(sum(ln[k] ** 2 for k in par if k in ln))
                        g_d2.append(abs(tp - dn) <= 1e-4 * dn + 2e-6 * math.sqrt(len(par)))
                        full = math.sqrt(dn ** 2 + float((vec(cs[c], bufv) - vec(g0, bufv)).norm()) ** 2) if bufv else dn
                        if abs(t["delta_l2"] - dn) <= 1e-4 * dn + 1e-5:
                            raw_match.add("params_only")
                        if abs(t["delta_l2"] - full) <= 1e-4 * full + 1e-5:
                            raw_match.add("params_plus_buffers")
                    else:
                        g_d2.append(False)
                    rows.append(row)
                else:
                    extra.append(row)
            if meth == "fedavg" and r in R:
                agg = G0 + sum(float(p) * dk for p, dk in zip(sn["p_k"], deltas))
                g_agg_old.append(float((G1 - agg).norm() / G.norm()))
                g_agg.append(float((G1 - agg).norm()) <= 2 ** -24 * float(G1.norm()))
            del sn, g0, g1, cs, deltas
        info["gates"] = {"G_SNAP": g_snap, "missing_R": miss, "G_AGG": (all(g_agg) if g_agg else "n/a"), "G_AGG_old_rel_max": (max(g_agg_old) if g_agg_old else None), "G_D2": all(g_d2) and bool(g_d2),
                         "raw_delta_l2_matches": sorted(raw_match)}
        layer_names = sorted(telmap[(1, "client_00")]["layer_delta_norms"]) if (1, "client_00") in telmap else []
        nonbuf = [k for k in layer_names if not k.endswith(BUF)]
        for t in tel:
            ln = t["layer_delta_norms"]
            d2rows.append({"method": meth, "alpha": al, "seed": s, "round": t["round"], "client": t["client_id"],
                           "D2": math.sqrt(sum(ln[k] ** 2 for k in nonbuf if k in ln)), "delta_l2_raw": t["delta_l2"]})
        runinfo[key] = info
        print(json.dumps({"run": key, **{k: v for k, v in info["gates"].items()}}), flush=True)

    def valid(meth, al, s):
        g = runinfo[f"{meth}|{al}|{s}"]["gates"]
        return g["G_SNAP"] and g["G_D2"] and g["G_AGG"] in (True, "n/a")

    per_run, summary = {}, {}
    for meth, al, s, _ in runs():
        for r in R:
            rs = [x for x in rows if x["method"] == meth and x["alpha"] == al and x["seed"] == s and x["round"] == r]
            if not valid(meth, al, s) or len(rs) != 5:
                per_run[f"{meth}|{al}|{s}|{r}"] = "unknown"
                continue
            d1, d3 = [x["D1"] for x in rs], [x["D3"] for x in rs]
            w = max(range(5), key=lambda i: d1[i])
            l4 = {k: statistics.mean(x["D4"][k] for x in rs) for k in rs[0]["D4"] if all(x["D4"][k] is not None for x in rs)}
            grp = {"classifier": [v for k, v in l4.items() if k.startswith("classifier.")]}
            per_run[f"{meth}|{al}|{s}|{r}"] = {
                "D1_mean": statistics.mean(d1), "D1_sd": sd(d1), "D1_max": d1[w], "D1_max_client": rs[w]["client"], "D1_max_n_k": rs[w]["n_k"],
                "D1_nk_weighted": sum(a * x["n_k"] for a, x in zip(d1, rs)) / sum(x["n_k"] for x in rs),
                "D3_mean": statistics.mean(d3) if None not in d3 else None, "D3_sd": sd(d3) if None not in d3 else None,
                "D3_min": min(d3) if None not in d3 else None,
                "D4_top5": sorted(l4.items(), key=lambda kv: -kv[1])[:5],
                "D4_classifier_mean": statistics.mean(grp["classifier"]) if grp["classifier"] else None}
    for meth in ("fedavg", "scaffold", "fedbn"):
        for al in (("0.1", "0.5", "1", "10", "100") if meth == "fedavg" else ("0.1", "0.5")):
            for r in R:
                cs_ = [per_run[f"{meth}|{al}|{s}|{r}"] for s in SEEDS]
                if any(c == "unknown" for c in cs_):
                    summary[f"{meth}|{al}|{r}"] = "unknown"
                    continue
                summary[f"{meth}|{al}|{r}"] = {m_: {"mean": statistics.mean(c[m_] for c in cs_), "sd": sd([c[m_] for c in cs_])}
                                              for m_ in ("D1_mean", "D1_max", "D3_mean", "D4_classifier_mean") if all(c[m_] is not None for c in cs_)}
    d2sum = {}
    for meth, al, s, _ in runs():
        for r in range(1, 41):
            xs = [x["D2"] for x in d2rows if x["method"] == meth and x["alpha"] == al and x["seed"] == s and x["round"] == r]
            if len(xs) == 5:
                d2sum[f"{meth}|{al}|{s}|{r}"] = {"mean": statistics.mean(xs), "sd": sd(xs), "max": max(xs)}
    # D5
    d5 = {}
    for al in ("0.1", "0.5", "1", "10", "100"):
        for s in SEEDS:
            fd = [d for m_, a_, s_, d in runs() if m_ == "fedavg" and a_ == al and s_ == s][0]
            cd = E / f"n11l/sgd_s{s}_out/run"
            fs = json.loads((fd / "summary.json").read_text(encoding="utf-8"))
            csum = json.loads((cd / "summary.json").read_text(encoding="utf-8"))
            fck = torch.load(fd / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            cck = torch.load(cd / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            fsd, csd = fck["model_state"], cck["model_state"]
            w0 = torch.load(fd / "client_server_snapshots_round_0001.pt", map_location="cpu", weights_only=False)["server_state_before"]
            g = (fs.get("w0_fingerprint") == W0FP and csum.get("w0_fingerprint") == W0FP and set(fsd) == set(csd) == set(w0)
                 and cck.get("epoch") == 40 and fck.get("round") == 40)
            keys = list(fsd)
            buf, _ = groups(keys)
            par = [k for k in keys if k not in buf and fsd[k].is_floating_point()]
            bufv = [k for k in buf if fsd[k].is_floating_point()]
            F, C, W = vec(fsd, par), vec(csd, par), vec(w0, par)
            Fb, Cb = vec(fsd, bufv), vec(csd, bufv)
            d5[f"{al}|{s}"] = {"G_D5": g, "D5": float((F - C).norm() / C.norm()) if g else "unknown",
                               "D5_bn_buffers": float((Fb - Cb).norm() / Cb.norm()) if g else "unknown",
                               "fedavg_minus_w0": float((F - W).norm()), "central_minus_w0": float((C - W).norm()),
                               "fedavg_ckpt_sha": sha(fd / "checkpoint_last.pt"), "central_ckpt_sha": sha(cd / "checkpoint_last.pt")}
            del fck, cck, w0
            print(json.dumps({"d5": f"{al}|{s}", "gate": g}), flush=True)
    out.mkdir(parents=True)
    doc = {"stage": "E4 weight divergence (E4_SPEC r1 + D-4 / PC_CHANGES C13, harness r2)", "spec_sha256": sha(H / "E4_SPEC.md"), "harness_sha256": sha(__file__),
           "registered_rounds": list(R), "labels": ["validation not used", "PC", "descriptive (no test)"], "selftest": st,
           "runs": runinfo, "per_run": per_run, "summary": summary, "D2_per_round": d2sum, "D5": d5,
           "note_extra_rounds": "rows with registered_round false are run-specific is_best rounds, never averaged across runs"}
    b = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    (out / "E4_RESULT.json").write_bytes(b)
    assert (out / "E4_RESULT.json").read_bytes() == b
    with open(out / "E4_D1_D3_D4.csv", "w", newline="", encoding="utf-8") as f:
        flds = ["method", "alpha", "seed", "round", "registered_round", "client", "n_k", "p_k", "D1", "D3", "D1_bn_buffers", "delta_norm_snapshot"]
        wr = csv.DictWriter(f, fieldnames=flds, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows + extra)
    with open(out / "E4_D2_rounds.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(d2rows[0]))
        wr.writeheader()
        wr.writerows(d2rows)
    with open(out / "E4_D5.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["alpha", "seed", "D5", "D5_bn_buffers", "fedavg_minus_w0", "central_minus_w0", "gate"])
        for k, v in d5.items():
            wr.writerow(k.split("|") + [v["D5"], v["D5_bn_buffers"], v["fedavg_minus_w0"], v["central_minus_w0"], v["G_D5"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for al in ("0.1", "0.5", "1", "10", "100"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for s in SEEDS:
            ys = [d2sum.get(f"fedavg|{al}|{s}|{r}", {}).get("mean") for r in range(1, 41)]
            ax.plot(range(1, 41), ys, lw=1, alpha=0.6, label=f"seed {s}")
        ax.set_xlabel("round")
        ax.set_ylabel("mean_k ||w_k - g0|| (D2)")
        ax.set_title(f"FedAvg alpha {al} (PC)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"E4_D2_by_round_{A_TAG[al]}.png", dpi=150)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    al_list = ("100", "10", "1", "0.5", "0.1")
    for ax, m_ in zip(axs, ("D1_mean", "D3_mean")):
        for r in (1, 40):
            ys = [summary.get(f"fedavg|{a}|{r}", {}) for a in al_list]
            mu = [y[m_]["mean"] if isinstance(y, dict) and m_ in y else float("nan") for y in ys]
            er = [y[m_]["sd"] if isinstance(y, dict) and m_ in y else 0 for y in ys]
            ax.errorbar(range(len(al_list)), mu, yerr=er, marker="o", capsize=3, label=f"round {r}")
        ax.set_xticks(range(len(al_list)))
        ax.set_xticklabels(al_list)
        ax.set_xlabel("alpha")
        ax.set_title(m_.replace("_mean", ""))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "E4_D1_D3_by_alpha.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"sha256": sha(out / "E4_RESULT.json"), "runs_valid": sum(valid(m_, a_, s_) for m_, a_, s_, _ in runs()),
                      "n_runs": len(runs()), "d5_valid": sum(1 for v in d5.values() if v["G_D5"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
