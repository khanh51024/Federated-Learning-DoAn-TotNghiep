"""S5 scorer (S5_SPEC r1 89b24b36 sections 3 + 4). CPU, read-only on run dirs. Self-test (section 4 vectors + t constants) runs
FIRST and must pass before any run is read.
  cells  : 8 = {scaffold, fedprox, moon, fedbn} x alpha {a01, a05}; comparator FedAvg same alpha + seed, seeds 42-46 (n = 5).
           endpoint = mean val_macro_f1_raw rounds 38-40, pd_acc = mean plantdoc_accuracy rounds 38-40 (= q19_score load_run);
           FedBN metric = CDR at round 40 (FedBN: history r40 q8r2 block; FedAvg: fedbn_compare on checkpoint_last, same function
           as q19_score fedbn_cell); FedBN pd_drop = FedAvg r40 PD acc - FedBN uniform-mean PD acc of the 5 personalised models.
  test   : one-sided paired t (df n-1, regularized incomplete beta, no scipy); Holm over the 8 cells (family 0.05); a cell with a
           missing / invalid run is "unknown", p = 1, stays in the family. WINNER5 iff p_holm <= 0.05 AND dbar >= 0.02 AND
           pd_drop5 <= 0.02, else "not shown".
  descr. : per-seed deltas, signs k/5, exact sign-flip p (1/32 floor), 95% t-interval, gap to Central (AdamW + SGD s42-46, mean
           +- SD; adamw_s42 = r3l/ctrl_out), gap-narrowing ratio (PHASE_C s3 b form, both ceilings), Phase C 3-seed cell side by side.
Usage: python -I -B s5_score.py <out_json> [--self-test-only] [--no-fedbn]    (out_json create-once)
"""
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

E = Path(r"experiments\train-improve-20261015")
H = Path(r"plans\cai-thien-train-20261015\pc_r9")
SPEC_SHA = "89b24b36c2dd1012e58da86bf8ef0747f9771c5bb5809d1e7c6e428d56479ced"
PHASE_C_SCORE = E / "Q19l/scores/q19_phase_c_full_20261108.json"
SEEDS = (42, 43, 44, 45, 46)
FAMILY, ALPHA_F, EFFECT, PD_RULE = 8, 0.05, 0.02, 0.02
DIGEST = {"label_alpha_0_1": "2b9d5c4a254a53f7b3a88f8535023212a6cfaf2580e27393bb6463fc32f13563",
          "label_alpha_0_5": "d5a1e3dfffe96e6a783e8f5b1fb5c9581c680d48cb004e837e95a2a3da9b7fd4"}
SCHEME = {"a01": "label_alpha_0_1", "a05": "label_alpha_0_5"}
METHODS = ("scaffold", "fedprox", "moon", "fedbn")
CENTRAL_EPOCHS = {"sgd": 40, "adamw": 35}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- statistics (no scipy)
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            return h
    raise RuntimeError("betacf did not converge")


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T_df > t)."""
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def t_quantile_upper(p, df):
    """t with P(T_df > t) = p, 0 < p < 0.5 (bisection)."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_sf(mid, df) > p else (lo, mid)
    return (lo + hi) / 2


def paired_t(deltas):
    n = len(deltas)
    dbar = sum(deltas) / n
    sd = math.sqrt(sum((x - dbar) ** 2 for x in deltas) / (n - 1))
    if sd == 0:
        return {"n": n, "dbar": dbar, "sd": 0.0, "t": None, "p_one": 0.0 if dbar > 0 else 1.0}
    t = dbar / (sd / math.sqrt(n))
    return {"n": n, "dbar": dbar, "sd": sd, "t": t, "p_one": t_sf(t, n - 1)}


def holm(ps):
    """Adjusted p in the ORIGINAL order: adj_(i) = max_{j<=i} min(1, (m - j + 1) p_(j))."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [None] * m, 0.0
    for j, i in enumerate(order, start=1):
        run = max(run, min(1.0, (m - j + 1) * ps[i]))
        adj[i] = run
    return adj


def sign_flip_p(deltas):
    n, dbar = len(deltas), sum(deltas) / len(deltas)
    a = [abs(x) for x in deltas]
    hits = sum(1 for sg in itertools.product((1, -1), repeat=n) if sum(s * x for s, x in zip(sg, a)) / n >= dbar - 1e-15)
    return hits / 2 ** n


def self_test():
    res = {}
    for t, df, p in ((2.131847, 4, 0.05), (3.746947, 4, 0.01), (2.353363, 3, 0.05)):
        got = t_sf(t, df)
        assert abs(got - p) < 1e-6, ("t constant", t, df, got)
        res[f"sf({t},{df})"] = got
    for t in (0.3, 1.0, 2.5, 6.0, 12.0):     # df 4 closed form
        u = t * t / (1 + t * t / 4)
        cf = 0.5 - 0.375 * (t / math.sqrt(1 + t * t / 4)) * (1 - u / 12)
        assert abs(t_sf(t, 4) - cf) < 1e-12, ("df4 closed form", t, t_sf(t, 4), cf)
    res["df4_closed_form"] = "ok (5 points, 1e-12)"
    r = paired_t([0.031, 0.024, 0.028, 0.019, 0.035])
    assert round(r["dbar"], 6) == 0.0274 and round(r["sd"], 6) == 0.006189 and round(r["t"], 6) == 9.900022, r
    assert abs(r["p_one"] - 2.921e-04) < 5e-08, r
    res["vector_t"] = r
    adj = holm([0.001, 0.004, 0.006, 0.02, 0.03, 0.2, 0.5, 0.04])
    want = [0.008, 0.028, 0.036, 0.1, 0.12, 0.4, 0.5, 0.12]
    assert all(abs(x - y) < 1e-12 for x, y in zip(adj, want)), adj
    res["vector_holm"] = adj
    for df, crit in ((4, (4.315, 4.148, 3.961, 3.747, 3.495, 3.186, 2.776, 2.132)),
                     (3, (5.392, 5.138, 4.857, 4.541, 4.177, 3.740, 3.182, 2.353))):
        got = [round(t_quantile_upper(ALPHA_F / (FAMILY - j), df), 3) for j in range(FAMILY)]
        assert all(abs(g - c) <= 0.0011 for g, c in zip(got, crit)), (df, got, crit)
        res[f"holm_crit_df{df}"] = got
    assert sign_flip_p([0.01] * 5) == 1 / 32 and sign_flip_p([-0.01] * 5) == 1.0
    res["sign_flip"] = "ok"
    return res


# ---------------------------------------------------------------- run loading
def run_dir(method, a, s):
    if s >= 45:
        r = f"s5_{method}_{a}_s{s}"
        return E / f"q21l/{r}_out/{r}"
    if method in ("fedavg", "scaffold"):
        base = "q15l" if a == "a01" else "q16l"
        name = f"{method}_s{s}" if a == "a01" else f"{method}_a05_s{s}"
        return E / base / f"{name}_out/{name}"
    r = f"c_m_{method}_{a}_s{s}"
    return E / f"q19l/{r}_out/{r}"


def load_run(d, scheme, rounds=40):
    """-> None (missing) or dict(valid, endpoint, pd_acc, r_last). Same definitions as q19_score.load_run."""
    if not (d / "RESULT.json").is_file():
        return None
    res = json.loads((d / "RESULT.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "run/summary.json").read_text(encoding="utf-8"))
    summ.pop("legacy_diagnostic_test", None)
    cfg = json.loads((d / "run/resolved_config.json").read_text(encoding="utf-8"))
    h = {x["round"]: x for x in summ["history"]}
    ok = bool(res["valid"]) and sorted(h) == list(range(1, rounds + 1)) and cfg.get("selected_partition_digest") == DIGEST[scheme]
    out = {"dir": str(d), "valid": ok, "result_sha256": sha(d / "RESULT.json"), "outcome": res.get("outcome")}
    if ok:
        last = (rounds - 2, rounds - 1, rounds)
        out["endpoint"] = sum(h[i]["val_macro_f1_raw"] for i in last) / 3
        out["pd_acc"] = sum(h[i]["plantdoc_accuracy"] for i in last) / 3
        out["r_last"] = h[rounds]
    return out


def central(kind, s):
    d = E / "r3l/ctrl_out" if (kind, s) == ("adamw", 42) else E / f"n11l/{kind}_s{s}_out"
    rec = d / ("R3_RUN_RECEIPT.json" if (kind, s) == ("adamw", 42) else "N11_RUN_RECEIPT.json")
    if not (d / "run/summary.json").is_file() or not rec.is_file():
        return {"dir": str(d), "present": False}
    summ = json.loads((d / "run/summary.json").read_text(encoding="utf-8"))
    rc = json.loads(rec.read_text(encoding="utf-8")).get("rc")
    h = summ["history"]
    ok = (rc == 0 and summ.get("status") == "MAX_BUDGET_REACHED" and summ.get("seed") == s
          and [e["epoch"] for e in h] == list(range(1, CENTRAL_EPOCHS[kind] + 1)))
    out = {"dir": str(d), "present": True, "valid": ok, "summary_sha256": sha(d / "run/summary.json")}
    if ok:
        out["endpoint"] = sum(e["val_macro_f1_raw"] for e in h[-3:]) / 3
    return out


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None)


# ---------------------------------------------------------------- cells
def endpoint_cell(method, a):
    fa = {s: load_run(run_dir("fedavg", a, s), SCHEME[a]) for s in SEEDS}
    mm = {s: load_run(run_dir(method, a, s), SCHEME[a]) for s in SEEDS}
    bad = [str(run_dir(k, a, s)) for s in SEEDS for k, v in (("fedavg", fa[s]), (method, mm[s])) if v is None or not v["valid"]]
    if bad:
        return {"status": "unknown", "p_one": 1.0, "missing_or_invalid": bad}
    d = [mm[s]["endpoint"] - fa[s]["endpoint"] for s in SEEDS]
    out = {"metric": "endpoint (mean val_macro_f1_raw r38-40)", **paired_t(d),
           "delta_s": dict(zip(SEEDS, d)), "pd_drop5": sum(fa[s]["pd_acc"] - mm[s]["pd_acc"] for s in SEEDS) / len(SEEDS),
           "mean_method": sum(mm[s]["endpoint"] for s in SEEDS) / len(SEEDS),
           "mean_fedavg": sum(fa[s]["endpoint"] for s in SEEDS) / len(SEEDS),
           "endpoint_method": {s: mm[s]["endpoint"] for s in SEEDS}, "endpoint_fedavg": {s: fa[s]["endpoint"] for s in SEEDS},
           "result_sha256": {s: [fa[s]["result_sha256"], mm[s]["result_sha256"]] for s in SEEDS}}
    out["status"] = "scored"
    return out


def fedbn_cell(a, F, m, q):
    fb = {s: load_run(run_dir("fedbn", a, s), SCHEME[a]) for s in SEEDS}
    fa = {s: load_run(run_dir("fedavg", a, s), SCHEME[a]) for s in SEEDS}
    bad = [str(run_dir(k, a, s)) for s in SEEDS for k, v in (("fedavg", fa[s]), ("fedbn", fb[s])) if v is None or not v["valid"]]
    if bad:
        return {"status": "unknown", "p_one": 1.0, "missing_or_invalid": bad}
    rows = {}
    for s in SEEDS:
        ev = fb[s]["r_last"].get("q8r2") or {}
        if (len(ev.get("fedbn_endpoint_b_client_dist_recall") or {}) != 5
                or ev.get("fedbn_endpoint_a_uniform_mean_plantdoc_accuracy") is None):
            return {"status": "unknown", "p_one": 1.0, "reason": f"seed {s}: FedBN round-40 q8r2 block incomplete"}
        cdr_fb = sum(ev["fedbn_endpoint_b_client_dist_recall"].values()) / 5
        c = F.compute(Path(fa[s]["dir"]) / "run", SCHEME[a], m, q)
        rows[s] = {"cdr_fedbn": cdr_fb, "cdr_fedavg": c["cdr_mean"], "diff": cdr_fb - c["cdr_mean"],
                   "pd_fedavg_r40": fa[s]["r_last"]["plantdoc_accuracy"],
                   "pd_fedbn_uniform_mean": ev["fedbn_endpoint_a_uniform_mean_plantdoc_accuracy"],
                   "server_endpoint_fedbn": fb[s]["endpoint"], "server_endpoint_fedavg": fa[s]["endpoint"],
                   "fedavg_checkpoint_last_sha256": c["checkpoint_last_sha256"],
                   "fedavg_reeval_equal_history": abs(c["reeval_val_macro_f1_raw"] - fa[s]["r_last"]["val_macro_f1_raw"]) <= 1e-9}
        print(f"fedbn {a} s{s}: cdr_fb {cdr_fb:.5f} cdr_fa {c['cdr_mean']:.5f}", flush=True)
    if not all(r["fedavg_reeval_equal_history"] for r in rows.values()):
        return {"status": "unknown", "p_one": 1.0, "reason": "FedAvg re-eval != history r40", "rows": rows}
    d = [rows[s]["diff"] for s in SEEDS]
    return {"metric": "CDR at round 40 (fedbn_compare)", "status": "scored", **paired_t(d), "delta_s": dict(zip(SEEDS, d)),
            "pd_drop5": sum(r["pd_fedavg_r40"] - r["pd_fedbn_uniform_mean"] for r in rows.values()) / len(SEEDS), "rows": rows,
            "mean_method": sum(r["cdr_fedbn"] for r in rows.values()) / len(SEEDS),
            "mean_fedavg": sum(r["cdr_fedavg"] for r in rows.values()) / len(SEEDS)}


def main(argv):
    out = Path(argv[0])
    assert not out.exists(), out
    st = self_test()
    print("SELF-TEST OK", flush=True)
    if "--self-test-only" in argv:
        print(json.dumps(st, default=str))
        return
    assert sha(H / "S5_SPEC.md") == SPEC_SHA, "S5_SPEC sha"
    cells = {f"{mt}_{a}": endpoint_cell(mt, a) for mt in ("scaffold", "fedprox", "moon") for a in ("a01", "a05")}
    if "--no-fedbn" in argv:
        cells.update({f"fedbn_{a}": {"status": "unknown", "p_one": 1.0, "reason": "--no-fedbn"} for a in ("a01", "a05")})
    else:
        s_ = importlib.util.spec_from_file_location("fedbn_compare", H / "fedbn_compare.py")
        F = importlib.util.module_from_spec(s_)
        s_.loader.exec_module(F)
        m, q = F.load_runner()
        cells.update({f"fedbn_{a}": fedbn_cell(a, F, m, q) for a in ("a01", "a05")})
    keys = [f"{mt}_{a}" for mt in METHODS for a in ("a01", "a05")]
    assert len(keys) == FAMILY
    adj = holm([cells[k]["p_one"] for k in keys])
    cen = {k: {s: central(k, s) for s in SEEDS} for k in CENTRAL_EPOCHS}
    ceil = {}
    for k, rows in cen.items():
        if all(r.get("valid") for r in rows.values()):
            mu, sdv = mean_sd([rows[s]["endpoint"] for s in SEEDS])
            ceil[k] = {"mean": mu, "sd": sdv}
        else:
            ceil[k] = {"status": "pending / invalid", "bad": [r["dir"] for r in rows.values() if not r.get("valid")]}
    try:
        pc = json.loads(PHASE_C_SCORE.read_text(encoding="utf-8"))["cells"]
        pc_sha = sha(PHASE_C_SCORE)
    except FileNotFoundError:
        pc, pc_sha = {}, None
    t975 = t_quantile_upper(0.025, len(SEEDS) - 1)
    for k, p_h in zip(keys, adj):
        c = cells[k]
        c["p_holm"] = p_h
        if c["status"] != "scored":
            c["verdict"] = "unknown"
        else:
            win = p_h <= ALPHA_F and c["dbar"] >= EFFECT and c["pd_drop5"] <= PD_RULE
            c["verdict"] = "WINNER5" if win else "not shown"
            dl = list(c["delta_s"].values())
            c["descriptive"] = {"signs_pos": f"{sum(x > 0 for x in dl)}/{len(dl)}", "sign_flip_p_one": sign_flip_p(dl),
                                "ci95": [c["dbar"] - t975 * c["sd"] / math.sqrt(len(dl)), c["dbar"] + t975 * c["sd"] / math.sqrt(len(dl))]}
            if not k.startswith("fedbn"):
                for ck, cv in ceil.items():
                    if "mean" in cv:
                        c["descriptive"][f"gap_to_central_{ck}_method"] = cv["mean"] - c["mean_method"]
                        c["descriptive"][f"gap_to_central_{ck}_fedavg"] = cv["mean"] - c["mean_fedavg"]
                        c["descriptive"][f"gap_narrowing_ratio_{ck}"] = (c["mean_method"] - c["mean_fedavg"]) / (cv["mean"] - c["mean_fedavg"])
        pck = k
        c["phase_c_3seed_R1"] = {"label": "Phase C (3 seeds, R1, uncorrected)", "status": (pc.get(pck) or {}).get("status")}
        c["label"] = "S5 (5 seeds, paired t + Holm 8)"
    doc = {"stage": "S5 scores (S5_SPEC r1)", "spec_sha256": SPEC_SHA, "harness_sha256": sha(__file__),
           "fedbn_compare_sha256": sha(H / "fedbn_compare.py"), "phase_c_score_sha256": pc_sha, "seeds": list(SEEDS),
           "family": FAMILY, "family_alpha": ALPHA_F, "effect_floor": EFFECT, "pd_rule": PD_RULE, "t_0975_df4": t975,
           "self_test": st, "cells": cells, "central": cen, "central_ceiling_5seed": ceil,
           "notes": ["seed 41 tuning only, never in this table", "Central PD column n/a (different measurement field)",
                     "FedBN metric = CDR (personalised models); no gap-narrowing ratio for FedBN",
                     "seeds 42-44 FedAvg / SCAFFOLD on the q15 runner, 45 / 46 on q18 (tensor-equal default path by PC gates)",
                     "val only, no EXT / test / internal_holdout read"]}
    b = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b)
    assert out.read_bytes() == b
    print(json.dumps({"cells": {k: [cells[k]["status"], cells[k].get("verdict"), cells[k].get("dbar"), cells[k]["p_holm"]] for k in keys},
                      "sha256": sha(out)}, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
