"""Report data builder (TU VAN PC 2026-11-10). Reads ONLY existing run histories / compare files (no model is run) and writes, into <repo>/pc_v3_ket_qua/results/:
  histories/per_round.csv     one row per (run, round|epoch[, client]) : run, method, alpha, seed, scheme, optimizer, client, round, val_macro_f1_raw, val_accuracy, plantdoc_macro_f1
  histories/payload_bytes.csv payload down/up bytes per client per round (FedAvg vs fp16 uplink, from telemetry)
  flw/*.json, flw/FLW_SUMMARY.csv, flw/LABEL_r2.txt
Runs = the entries of p6_inputs/P6_CHECKPOINTS_PC.json (families central_p40, fl_alpha_curve, fl_methods, fl_methods_a1, fl_a2_fp16, fl_phase_d, fl_s5_fedavg,
local_only, local_only_phase_d, rdiag_descriptive). Absolute paths are removed from every written file.
Usage: python -I -B build_report_data.py <pc_v3_ket_qua dir>
"""
import csv
import json
import re
import sys
from pathlib import Path

WS = Path("C:/do-an-tot-nghiep")
E = WS / "experiments/train-improve-20261015"
OUT = Path(sys.argv[1]) / "results"
assert not (OUT / "histories").exists() and not (OUT / "flw").exists(), "create-once"
SEP = r"(?:\\\\|\\|/)"


def clean(s):
    s = re.sub(r"[Cc]:" + SEP + r"do-an-tot-nghiep" + SEP, "", s)
    s = re.sub(r"[Cc]:" + SEP + r"do-an-tot-nghiep", "<WORKSPACE_ROOT>", s)
    return re.sub(r"(?<![A-Za-z])[A-Za-z]:" + SEP + r"(?!/)", "<PC_PATH>/", s)


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


L = J(E / "p6_inputs/P6_CHECKPOINTS_PC.json")["checkpoints"]
FAMS = {"central_p40", "fl_alpha_curve", "fl_methods", "fl_methods_a1", "fl_a2_fp16", "fl_phase_d", "fl_s5_fedavg", "local_only", "local_only_phase_d", "rdiag_descriptive"}
rows = []


LO_TAG = {"01": 0.1, "05": 0.5, "1": 1.0, "10": 10.0, "100": 100.0}


def add(e, hist, client="", epoch_key="round"):
    scheme = e.get("scheme", "")
    if e["family"] in ("local_only", "local_only_phase_d"):
        m = re.match(r"pc_local_(?:a(\d+)|(qs|ss))_s\d+_client_\d+$", e["name"])
        e = dict(e)
        if m.group(1):
            e["alpha"] = LO_TAG[m.group(1)]
        else:
            scheme = m.group(2)
    for h in hist:
        rows.append({"run": e["name"].replace("_client_0%s" % client[-1], "") if client else e["name"], "method": e.get("method", e["family"]), "alpha": e.get("alpha", ""),
                     "seed": e.get("seed", ""), "scheme": scheme, "optimizer": e.get("optimizer", ""), "client": client, "family": e["family"],
                     "round": h[epoch_key], "val_macro_f1_raw": h.get("val_macro_f1_raw"), "val_accuracy": h.get("val_accuracy"),
                     "plantdoc_macro_f1": h.get("plantdoc_macro_f1_supported")})


for e in L:
    if e["family"] not in FAMS:
        continue
    ck = WS / e["pc_path"]
    if e["family"] == "central_p40":
        add(e, J(ck.parent / "summary.json")["history"], epoch_key="epoch")
        for r in rows[-len(J(ck.parent / "summary.json")["history"]):]:
            r["method"] = "central_" + e["optimizer"]
    elif e["family"] in ("local_only", "local_only_phase_d"):
        add(e, J(ck.parent / "history.json"), client="client_%02d" % e["client"])
        for r in rows[-40:]:
            r["method"] = "local_only"
    else:
        add(e, J(ck.parent / "summary.json")["history"])
cols = ["run", "method", "family", "alpha", "seed", "scheme", "optimizer", "client", "round", "val_macro_f1_raw", "val_accuracy", "plantdoc_macro_f1"]
(OUT / "histories").mkdir(parents=True)
with open(OUT / "histories/per_round.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows({c: r.get(c, "") for c in cols} for r in rows)
print("rows", len(rows), "runs", len({(r["run"], r["client"]) for r in rows}))

# payload bytes (telemetry): FedAvg vs fp16 uplink, alpha 0.1 seed 42
pb = []
for name, p in (("fedavg_a01_s42", E / "q15l/fedavg_s42_out/fedavg_s42/run/fedavg_telemetry_events.jsonl"),
                ("fedavg_fp16_uplink_a01_s42", E / "q22l/a2_fedavg_fp16_a01_s42_out/a2_fedavg_fp16_a01_s42/run/fedavg_telemetry_events.jsonl")):
    seen = set()
    for l in p.read_text(encoding="utf-8").splitlines():
        ev = json.loads(l)
        if ev.get("event_type") == "client_round_execution" and "payload_down_bytes" in ev:
            pb.append({"run": name, "client": ev["client_id"], "payload_down_bytes": ev["payload_down_bytes"], "payload_up_bytes": ev["payload_up_bytes"]})
            seen.add(ev["client_id"])
with open(OUT / "histories/payload_bytes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["run", "client", "payload_down_bytes", "payload_up_bytes"])
    w.writeheader()
    w.writerows(pb[:0] or [dict(t) for t in {tuple(sorted(x.items())) for x in pb}])

# FLW
(OUT / "flw").mkdir(parents=True)
LEDGER = [json.loads(l) for l in (E / "pc_runs_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ex = {e["run"]: e for e in LEDGER if e.get("kind") == "exit" and str(e.get("run", "")).startswith(("flwa_s", "a3_fed"))}
items = [("flwa_s42", "flw/flwa_s42/FLWA_s42_compare.json"), ("flwa_s43", "flw/flwa_s43/FLWA_s43_compare.json"), ("flwa_s44", "flw/flwa_s44/FLWA_s44_compare.json"),
         ("a3_fedprox_a01_s42", "flw/a3_fedprox_a01_s42/A3_fedprox_compare.json"), ("a3_fedbn_a01_s42", "flw/a3_fedbn_a01_s42/A3_fedbn_compare.json")]
srows = []
for run, rel in items:
    src = E / rel
    (OUT / "flw" / src.name).write_text(clean(src.read_text(encoding="utf-8")), encoding="utf-8")
    c = J(src)
    e = ex[run]
    srows.append({"run": run, "verdict": e.get("verdict_by_rule") or e.get("official_verdict"), "compare_sha256": e["compare_sha256"], "max_abs_d_val_macro_f1_raw": c.get("max_abs_d_val_macro_f1_raw"),
                  "endpoint_abs_diff": c.get("endpoint_abs_diff", ""), "checkpoint_equal": c.get("checkpoint_last_tensor_equal", e.get("descriptive_gen5_checkpoint", "")),
                  "total_wall_s": e.get("total_wall_s", ""), "rtt_ms_median": e.get("rtt_ms_median", ""),
                  "payload_bytes_per_client_round": e.get("payload_bytes_per_client_round") or (e.get("payload_down_up_bytes") or [""])[0], "ram": e.get("ram", "")})
with open(OUT / "flw/FLW_SUMMARY.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0]))
    w.writeheader()
    w.writerows(srows)
LABEL_R2 = """LABEL r2 (mandatory, verbatim in report): "flower_exact = equivalence gate of orchestration / transport. The client DataLoader
(dataset + sampler), the RNG and the GradScaler are created by the harness on the server side and passed in the control blob.
The client does not hold its own shard / independent seed. Only the ArrayRecord is the FL payload." The report must NOT say
"FedAvg runs truly distributed on Flower" on the basis of FLW-A; the more realistic part is FLW-B (Deployment Engine).
"""
(OUT / "flw/LABEL_r2.txt").write_text(LABEL_R2, encoding="utf-8")
print("flw", [s["run"] for s in srows])
