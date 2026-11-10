"""Q19 scorer (PHASE_C_SPEC_PC r2 a8326c06 section 3 + 5). CPU, read-only on run dirs. Definitions = q15_score (endpoint = mean
val_macro_f1_raw rounds 38-40, pd_acc = mean plantdoc_accuracy rounds 38-40; R1 THRESH = max(0.02, 2 r)).
  tuning : C-T T = mean val_macro_f1_raw rounds 18-20 per valid run; mu* = argmax T per method, tie -> smaller mu; < 2 valid -> failed.
  cells  : R1 per (method in scaffold / fedprox / moon, alpha in 0.1 / 0.5) vs the PC FedAvg anchors, paired by seed 42-44.
  fedbn  : CDR rule per alpha (FedBN CDR from history round 40 q8r2; FedAvg CDR from fedbn_compare on FedAvg checkpoint_last).
A cell with a missing run is "pending"; with an invalid run (RESULT invalid or digest != registry) "unknown". 8 cells reported
separately, never a "best method"; a WINNER is labelled "one of 8 tests, no multiple-comparison correction".
Usage: python -I -B q19_score.py <out_json> [--tuning-only] [--no-fedbn]    (out_json create-once)
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

E = Path(r"experiments\train-improve-20261015")
H = Path(r"plans\cai-thien-train-20261015\pc_r9")
R = 0.0028218398420888358
THRESH = max(0.02, 2 * R)
SEEDS = (42, 43, 44)
DIGEST = {"label_alpha_0_1": "2b9d5c4a254a53f7b3a88f8535023212a6cfaf2580e27393bb6463fc32f13563",
          "label_alpha_0_5": "d5a1e3dfffe96e6a783e8f5b1fb5c9581c680d48cb004e837e95a2a3da9b7fd4"}
SCHEME = {"a01": "label_alpha_0_1", "a05": "label_alpha_0_5"}
TUNE = {"fedprox": [("c_t_fedprox_mu0p001_s41", 0.001), ("c_t_fedprox_mu0p01_s41", 0.01), ("c_t_fedprox_mu0p1_s41", 0.1)],
        "moon": [("c_t_moon_mu0p1_s41", 0.1), ("c_t_moon_mu1_s41", 1.0), ("c_t_moon_mu5_s41", 5.0)]}
assert THRESH == 0.02


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_dir(method, a, s):
    if method == "fedavg":
        return E / ("q15l" if a == "a01" else "q16l") / (f"fedavg_s{s}_out/fedavg_s{s}" if a == "a01" else f"fedavg_a05_s{s}_out/fedavg_a05_s{s}")
    if method == "scaffold":
        return E / ("q15l" if a == "a01" else "q16l") / (f"scaffold_s{s}_out/scaffold_s{s}" if a == "a01" else f"scaffold_a05_s{s}_out/scaffold_a05_s{s}")
    r = f"c_m_{method}_{a}_s{s}"
    return E / f"q19l/{r}_out/{r}"


def load_run(d, scheme, rounds):
    """-> None (missing) or dict(valid, endpoint, pd_acc, history)."""
    if not (d / "RESULT.json").is_file():
        return None
    res = json.loads((d / "RESULT.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "run/summary.json").read_text(encoding="utf-8"))
    summ.pop("legacy_diagnostic_test", None)
    cfg = json.loads((d / "run/resolved_config.json").read_text(encoding="utf-8"))
    h = {x["round"]: x for x in summ["history"]}
    ok = bool(res["valid"]) and sorted(h) == list(range(1, rounds + 1)) and cfg.get("selected_partition_digest") == DIGEST[scheme]
    out = {"dir": str(d), "valid": ok, "result_sha256": sha(d / "RESULT.json"), "digest_ok": cfg.get("selected_partition_digest") == DIGEST[scheme]}
    if ok:
        last = (rounds - 2, rounds - 1, rounds)
        out["endpoint"] = sum(h[i]["val_macro_f1_raw"] for i in last) / 3
        out["pd_acc"] = sum(h[i]["plantdoc_accuracy"] for i in last) / 3
        out["r_last"] = h[rounds]
    return out


def tuning():
    out = {}
    for m, grid in TUNE.items():
        rows = []
        for run, mu in grid:
            r = load_run(E / f"q19l/{run}_out/{run}", "label_alpha_0_1", 20)
            rows.append({"run": run, "mu": mu, "present": r is not None, "valid": bool(r and r["valid"]),
                         "T": r.get("endpoint") if r and r["valid"] else None})
        valid = [x for x in rows if x["valid"]]
        if any(not x["present"] for x in rows):
            out[m] = {"rows": rows, "mu_star": None, "status": "pending"}
        elif len(valid) < 2:
            out[m] = {"rows": rows, "mu_star": None, "status": "tuning failed"}
        else:
            best = max(valid, key=lambda x: (x["T"], -x["mu"]))
            out[m] = {"rows": rows, "mu_star": best["mu"], "status": "ok",
                      "label": "mu chosen on val, seed 41, alpha 0.1 only; the same mu is used for alpha 0.5"}
    return out


def r1_cell(method, a):
    fa = {s: load_run(run_dir("fedavg", a, s), SCHEME[a], 40) for s in SEEDS}
    mm = {s: load_run(run_dir(method, a, s), SCHEME[a], 40) for s in SEEDS}
    if any(v is None for v in list(fa.values()) + list(mm.values())):
        return {"status": "pending", "present": {s: [fa[s] is not None, mm[s] is not None] for s in SEEDS}}
    if not all(v["valid"] for v in list(fa.values()) + list(mm.values())):
        return {"status": "unknown", "invalid": [v["dir"] for v in list(fa.values()) + list(mm.values()) if not v["valid"]]}
    d = {s: mm[s]["endpoint"] - fa[s]["endpoint"] for s in SEEDS}
    delta = sum(d.values()) / 3
    pd_drop = sum(fa[s]["pd_acc"] - mm[s]["pd_acc"] for s in SEEDS) / 3
    win = delta >= THRESH and all(v > 0 for v in d.values()) and pd_drop <= 0.02
    return {"status": "WINNER" if win else "not established", "delta": delta, "delta_s": d, "pd_drop": pd_drop,
            "endpoint_method": {s: mm[s]["endpoint"] for s in SEEDS}, "endpoint_fedavg": {s: fa[s]["endpoint"] for s in SEEDS},
            "mean_method": sum(mm[s]["endpoint"] for s in SEEDS) / 3, "mean_fedavg": sum(fa[s]["endpoint"] for s in SEEDS) / 3,
            "label": "one of 8 tests, no multiple-comparison correction" if win else None}


def fedbn_cell(a, F=None, m=None, q=None):
    fb = {s: load_run(run_dir("fedbn", a, s), SCHEME[a], 40) for s in SEEDS}
    fa = {s: load_run(run_dir("fedavg", a, s), SCHEME[a], 40) for s in SEEDS}
    if any(v is None for v in list(fb.values()) + list(fa.values())):
        return {"status": "pending"}
    if not all(v["valid"] for v in list(fb.values()) + list(fa.values())):
        return {"status": "unknown"}
    rows = {}
    for s in SEEDS:
        ev = fb[s]["r_last"].get("q8r2") or {}
        if (len(ev.get("fedbn_endpoint_b_client_dist_recall") or {}) != 5
                or ev.get("fedbn_endpoint_a_uniform_mean_plantdoc_accuracy") is None):   # TU VAN: never divide a short CDR block
            return {"status": "unknown", "reason": f"seed {s}: FedBN round-40 q8r2 block incomplete (need 5 CDR + PD uniform mean)"}
        cdr_fb = sum(ev["fedbn_endpoint_b_client_dist_recall"].values()) / 5
        c = F.compute(Path(fa[s]["dir"]) / "run", SCHEME[a], m, q)
        rows[s] = {"cdr_fedbn": cdr_fb, "cdr_fedavg": c["cdr_mean"], "diff": cdr_fb - c["cdr_mean"],
                   "pd_fedavg_r40": fa[s]["r_last"]["plantdoc_accuracy"], "pd_fedbn_uniform_mean": ev["fedbn_endpoint_a_uniform_mean_plantdoc_accuracy"],
                   "fedbn_uniform_mean_raw_macro_f1": ev["fedbn_endpoint_a_uniform_mean_raw_macro_f1"],
                   "fedavg_reeval_equal_history": abs(c["reeval_val_macro_f1_raw"] - fa[s]["r_last"]["val_macro_f1_raw"]) <= 1e-9}
    md = sum(r["diff"] for r in rows.values()) / 3
    pd_drop = sum(r["pd_fedavg_r40"] - r["pd_fedbn_uniform_mean"] for r in rows.values()) / 3
    better = md >= 0.02 and all(r["diff"] > 0 for r in rows.values()) and pd_drop <= 0.02
    if not all(r["fedavg_reeval_equal_history"] for r in rows.values()):
        return {"status": "unknown", "reason": "FedAvg re-eval != history r40", "rows": rows}
    return {"status": "better on CDR" if better else "not better", "mean_diff": md, "pd_drop": pd_drop, "rows": rows,
            "label": "one of 8 tests, no multiple-comparison correction" if better else None}


def main(argv):
    out = Path(argv[0])
    assert not out.exists(), out
    doc = {"stage": "Q19 scores (PHASE_C_SPEC_PC r2)", "spec_sha256": sha(H / "PHASE_C_SPEC_PC.md"), "harness_sha256": sha(__file__),
           "thresh": THRESH, "tuning": tuning()}
    if "--tuning-only" not in argv:
        doc["cells"] = {f"{m}_{a}": r1_cell(m, a) for m in ("scaffold", "fedprox", "moon") for a in ("a01", "a05")}
        if "--no-fedbn" not in argv:
            s_ = importlib.util.spec_from_file_location("fedbn_compare", H / "fedbn_compare.py")
            F = importlib.util.module_from_spec(s_)
            s_.loader.exec_module(F)
            m, q = F.load_runner()
            doc["cells"].update({f"fedbn_{a}": fedbn_cell(a, F, m, q) for a in ("a01", "a05")})
    b = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b)
    assert out.read_bytes() == b
    print(json.dumps({"tuning": {k: v["status"] for k, v in doc["tuning"].items()},
                      "cells": {k: v["status"] for k, v in doc.get("cells", {}).items()}, "sha256": sha(out)}))


if __name__ == "__main__":
    main(sys.argv[1:])
