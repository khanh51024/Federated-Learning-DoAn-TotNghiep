"""Q22 scorer (A1A2_SPEC r1 section 4). CPU, read-only on run dirs. Statistics = s5_score 61188a9e functions (imported by path,
its self-test re-run here) + the df 2 Holm critical values of A1A2_SPEC s4, asserted BEFORE any run is read.
A1 (SCAFFOLD-BN, team proposal): 4 cells = {vs FedAvg, vs SCAFFOLD} x alpha {a01, a05}, seeds 42-44 (n 3, df 2).
   metric = CDR at round 40: scaffold_bn = its history r40 q8r2 block (personalised models, 5 clients); comparator = fedbn_compare
   (fa229ea5) on the comparator's checkpoint_last (global model for every client), same function as PHASE_C s3 c / S5.
   delta_s = CDR(scaffold_bn, s) - CDR(comp, s); one-sided paired t; Holm over the 4 cells (family 0.05);
   pd_drop = mean_s [comp r40 PD acc - scaffold_bn uniform-mean PD acc of the 5 personalised models].
   WINNER iff p_holm <= 0.05 AND dbar >= 0.02 AND pd_drop <= 0.02, else "not shown". Missing / invalid run -> "unknown", p = 1.
   descriptive: server-model endpoint (mean val_macro_f1_raw r38-40) + PD acc of scaffold_bn and comparators, per-seed deltas,
   signs k/3. Label: "SCAFFOLD-BN is the team's proposal; 3 seeds only; descriptive server-model endpoint is not the primary metric".
A2 (fedavg --uplink-dtype float16, alpha 0.1, seeds 42-44), descriptive only: per round MB up (sum of the 5 clients'
   payload_up_bytes from fedavg_telemetry_events, MB = 2^20 bytes as the E3 table), ratio to the float32 anchor pc_fedavg_a01_s<s>
   (q15l fedavg_s<s>), endpoint delta paired by seed + mean, rounds_to_095 / rounds_to_099 = first round with val_macro_f1_raw >=
   0.95 / 0.99 x own endpoint (E3 definition), cumulative MB up to those rounds and over 40 rounds, for both arms.
Usage: python -I -B q22_score.py <out_json> [--self-test-only] [--no-cdr]    (out_json create-once)
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

E = Path(r"experiments\train-improve-20261015")
H = Path(r"plans\cai-thien-train-20261015\pc_r9")
SPEC = (H / "A1A2_SPEC.md", "5824f9deda181672a84276f2a8d0f952952b2266d7974e75be88412ba6685abd")
S5 = (H / "s5_score.py", "61188a9eb04091c0b0cdc581bdd3ab79b986cbaa008df774cdca0414d32cb744")
SEEDS = (42, 43, 44)
SCHEME = {"a01": "label_alpha_0_1", "a05": "label_alpha_0_5"}
FAMILY, ALPHA_F, EFFECT, PD_RULE = 4, 0.05, 0.02, 0.02
MB = 2 ** 20


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def s5mod():
    assert sha(S5[0]) == S5[1], "s5_score sha"
    s_ = importlib.util.spec_from_file_location("s5_score", S5[0])
    m = importlib.util.module_from_spec(s_)
    s_.loader.exec_module(m)
    return m


def self_test(S):
    res = {"s5_self_test": S.self_test()}
    got = [round(S.t_quantile_upper(ALPHA_F / (FAMILY - j), 2), 3) for j in range(FAMILY)]
    want = (6.205, 5.339, 4.303, 2.920)
    assert all(abs(g - w) <= 0.0011 for g, w in zip(got, want)), got
    res["holm_crit_df2"] = got
    adj = S.holm([0.01, 0.04, 0.03, 0.2])
    assert all(abs(x - y) < 1e-12 for x, y in zip(adj, [0.04, 0.09, 0.09, 0.2])), adj
    res["holm4_vector"] = adj
    return res


def run_dir(kind, a, s):
    if kind == "scaffold_bn":
        r = f"a1_scaffold_bn_{a}_s{s}"
        return E / f"q22l/{r}_out/{r}"
    if kind == "fedavg_fp16":
        r = f"a2_fedavg_fp16_a01_s{s}"
        return E / f"q22l/{r}_out/{r}"
    base = "q15l" if a == "a01" else "q16l"
    name = f"{kind}_s{s}" if a == "a01" else f"{kind}_a05_s{s}"
    return E / base / f"{name}_out/{name}"


def a1_cell(S, comp, a, F, m, q):
    sb = {s: S.load_run(run_dir("scaffold_bn", a, s), SCHEME[a]) for s in SEEDS}
    cp = {s: S.load_run(run_dir(comp, a, s), SCHEME[a]) for s in SEEDS}
    bad = [str(run_dir(k, a, s)) for s in SEEDS for k, v in (("scaffold_bn", sb[s]), (comp, cp[s])) if v is None or not v["valid"]]
    if bad:
        return {"status": "unknown", "p_one": 1.0, "missing_or_invalid": bad}
    rows = {}
    for s in SEEDS:
        ev = sb[s]["r_last"].get("q8r2") or {}
        if (len(ev.get("fedbn_endpoint_b_client_dist_recall") or {}) != 5
                or ev.get("fedbn_endpoint_a_uniform_mean_plantdoc_accuracy") is None):
            return {"status": "unknown", "p_one": 1.0, "reason": f"seed {s}: scaffold_bn round-40 q8r2 block incomplete"}
        cdr_sb = sum(ev["fedbn_endpoint_b_client_dist_recall"].values()) / 5
        if F is None:
            return {"status": "unknown", "p_one": 1.0, "reason": "--no-cdr"}
        c = F.compute(Path(cp[s]["dir"]) / "run", SCHEME[a], m, q)
        rows[s] = {"cdr_scaffold_bn": cdr_sb, "cdr_comp": c["cdr_mean"], "diff": cdr_sb - c["cdr_mean"],
                   "pd_comp_r40": cp[s]["r_last"]["plantdoc_accuracy"],
                   "pd_scaffold_bn_uniform_mean": ev["fedbn_endpoint_a_uniform_mean_plantdoc_accuracy"],
                   "server_endpoint_scaffold_bn": sb[s]["endpoint"], "server_endpoint_comp": cp[s]["endpoint"],
                   "server_pd_acc_scaffold_bn": sb[s]["pd_acc"], "server_pd_acc_comp": cp[s]["pd_acc"],
                   "comp_checkpoint_last_sha256": c["checkpoint_last_sha256"],
                   "comp_reeval_equal_history": abs(c["reeval_val_macro_f1_raw"] - cp[s]["r_last"]["val_macro_f1_raw"]) <= 1e-9,
                   "result_sha256": [sb[s]["result_sha256"], cp[s]["result_sha256"]]}
        print(f"A1 {comp} {a} s{s}: cdr_sb {cdr_sb:.5f} cdr_comp {c['cdr_mean']:.5f}", flush=True)
    if not all(r["comp_reeval_equal_history"] for r in rows.values()):
        return {"status": "unknown", "p_one": 1.0, "reason": "comparator re-eval != history r40", "rows": rows}
    d = [rows[s]["diff"] for s in SEEDS]
    out = {"status": "scored", "metric": "CDR at round 40", **S.paired_t(d), "delta_s": dict(zip(SEEDS, d)),
           "pd_drop": sum(r["pd_comp_r40"] - r["pd_scaffold_bn_uniform_mean"] for r in rows.values()) / len(SEEDS), "rows": rows,
           "signs_pos": f"{sum(x > 0 for x in d)}/{len(d)}",
           "server_endpoint_delta_s": {s: rows[s]["server_endpoint_scaffold_bn"] - rows[s]["server_endpoint_comp"] for s in SEEDS}}
    return out


def telemetry_up(run):
    f = Path(run) / "run/fedavg_telemetry_events.jsonl"
    per = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("event_type") == "client_round_execution":
            per.setdefault(e["round"], []).append(e["payload_up_bytes"])
    assert sorted(per) == list(range(1, 41)) and all(len(v) == 5 for v in per.values()), "telemetry rounds / clients"
    return {r: sum(v) for r, v in per.items()}


def rounds_to(hist, frac, endpoint):
    for r in sorted(hist):
        if hist[r]["val_macro_f1_raw"] >= frac * endpoint:
            return r
    return None


def a2(S):
    out, deltas = {}, []
    for s in SEEDS:
        fp = S.load_run(run_dir("fedavg_fp16", "a01", s), SCHEME["a01"])
        fa = S.load_run(run_dir("fedavg", "a01", s), SCHEME["a01"])
        if fp is None or fa is None or not (fp["valid"] and fa["valid"]):
            out[s] = {"status": "unknown (missing / invalid)"}
            continue
        row = {"status": "scored", "endpoint_fp16": fp["endpoint"], "endpoint_fp32": fa["endpoint"],
               "endpoint_delta": fp["endpoint"] - fa["endpoint"], "pd_acc_fp16": fp["pd_acc"], "pd_acc_fp32": fa["pd_acc"]}
        deltas.append(row["endpoint_delta"])
        for tag, r in (("fp16", fp), ("fp32", fa)):
            up = telemetry_up(r["dir"])
            h = {x["round"]: x for x in json.loads((Path(r["dir"]) / "run/summary.json").read_text(encoding="utf-8"))["history"]}
            cum = {}
            acc = 0
            for k in range(1, 41):
                acc += up[k]
                cum[k] = acc
            r95, r99 = rounds_to(h, 0.95, r["endpoint"]), rounds_to(h, 0.99, r["endpoint"])
            row[tag] = {"mb_up_per_round": sorted(set(round(v / MB, 4) for v in up.values())), "mb_up_40": cum[40] / MB,
                        "rounds_to_095": r95, "rounds_to_099": r99, "mb_up_to_095": cum[r95] / MB if r95 else None,
                        "mb_up_to_099": cum[r99] / MB if r99 else None}
        row["up_ratio_fp16_over_fp32"] = row["fp16"]["mb_up_40"] / row["fp32"]["mb_up_40"]
        out[s] = row
    mean = sum(deltas) / len(deltas) if deltas else None
    return {"rows": out, "endpoint_delta_mean": mean, "n": len(deltas), "label": "descriptive only (A1A2_SPEC s4 A2: no winner rule)"}


def main(argv):
    out = Path(argv[0])
    assert not out.exists(), out
    S = s5mod()
    st = self_test(S)
    print("SELF-TEST OK", flush=True)
    if "--self-test-only" in argv:
        print(json.dumps(st, default=str))
        return
    assert sha(SPEC[0]) == SPEC[1], "A1A2_SPEC sha"
    F = m = q = None
    if "--no-cdr" not in argv:
        s_ = importlib.util.spec_from_file_location("fedbn_compare", H / "fedbn_compare.py")
        F = importlib.util.module_from_spec(s_)
        s_.loader.exec_module(F)
        m, q = F.load_runner()
    keys = [(c, a) for c in ("fedavg", "scaffold") for a in ("a01", "a05")]
    cells = {f"scaffold_bn_vs_{c}_{a}": a1_cell(S, c, a, F, m, q) for c, a in keys}
    adj = S.holm([cells[k]["p_one"] for k in cells])
    for k, p in zip(cells, adj):
        c = cells[k]
        c["p_holm"] = p
        c["verdict"] = ("WINNER" if (c["status"] == "scored" and p <= ALPHA_F and c["dbar"] >= EFFECT and c["pd_drop"] <= PD_RULE)
                        else ("not shown" if c["status"] == "scored" else "unknown"))
    doc = {"stage": "Q22 scores (A1A2_SPEC r1)", "spec_sha256": sha(SPEC[0]), "harness_sha256": sha(__file__), "s5_score_sha256": S5[1],
           "fedbn_compare_sha256": sha(H / "fedbn_compare.py"), "seeds": list(SEEDS), "family": FAMILY, "self_test": st,
           "a1_cells": cells, "a1_label": "SCAFFOLD-BN is the team's proposal; 3 seeds only; descriptive server-model endpoint is "
                                         "not the primary metric", "a2": a2(S),
           "notes": ["val only, no EXT / test / internal_holdout read", "MB = 2^20 bytes (E3 table convention)"]}
    b = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b)
    assert out.read_bytes() == b
    print(json.dumps({"a1": {k: [v["status"], v["verdict"], v.get("dbar"), v["p_holm"], v.get("pd_drop")] for k, v in cells.items()},
                      "a2_mean_delta": doc["a2"]["endpoint_delta_mean"], "sha256": sha(out)}, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
