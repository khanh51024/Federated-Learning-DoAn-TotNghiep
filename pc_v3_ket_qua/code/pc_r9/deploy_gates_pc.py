"""Deploy gates for the PC (DEPLOY_RUNS_SPEC r3 PC section 2) -> <out_dir>/DEPLOY_GATES.json (create-once), input of ext_sel_eval_pc.py.
Reuses the UNCHANGED r2 functions of r8_draft/deploy_gates.py (pd_disease_f1, src_acc, guard, q5_reference, flat, sha; guardrails and
EXPECT_DIFF values) and applies the r2 validity rules per platform: PC candidates vs r3data_pc, laptop candidates vs r3data_lap.
Cross-machine: r3data_pc vs r3data_lap resolved_config after mapping the PC root to the laptop root (PC_HANDOVER s4 c); recorded.
Detector (used by R4 later): laptop_runs/detl/det_out (not retrained). Never reads test keys.
Usage: python -I -B deploy_gates_pc.py <out_dir> [--dry]
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PL = Path(r"plans\cai-thien-train-20261015")
SRC = PL / "r8_draft/deploy_gates.py"
_s = importlib.util.spec_from_file_location("deploy_gates_r2", SRC)
DG = importlib.util.module_from_spec(_s)
_s.loader.exec_module(DG)
WS, E = DG.WS, DG.E
PC_ROOT, LAP_ROOT = str(WS), r"<PC_PATH>/university\do-an-tot-nghiep"
CANDS = {  # label: (run dir, platform, pinned checkpoint sha, baseline label, expected-diff key in DG.EXPECT_DIFF or None)
    "r3data_pc": (E / "r3l/data_out", "PC", "a274bb9e713728ab5578198661b93419d4ff8d5050cf643416346736997f58a4", "r3data_pc", None),
    "bg_pc": (E / "r3xl/bg_out", "PC", "fad07c39666acb2dee20e7627adc0f5a3443c6b4f4adc7c516e5a8f17aac0d33", "r3data_pc", "bg"),
    "r3c_pc": (E / "r3xl/r3c_out", "PC", "8751d8f10cab78f2e7394c95cb3091e4643cada5c31b2b371942d0f65e75a344", "r3data_pc", "r3c"),
    "r3data_lap": (E / "laptop_runs/r3l/data_out", "laptop", "905043dde38f14901148a342e886195f9a7f48eca1fddfd826712ca90648f450", "r3data_lap", None),
    "aug2_lap": (E / "laptop_runs/r3xl/aug2_out", "laptop", "0463eb3eefc1811a18ea2186472c172b439098f2c4ae90e283167a79fe54c308", "r3data_lap", "aug2"),
    "bal_lap": (E / "laptop_runs/r3xl/bal_out", "laptop", "8f9336dadbc8a9e2a6fef80b9d12166a807ebb0cde2c8a996dc866cb1e46bdc9", "r3data_lap", "bal"),
}


def cfg(out):
    return DG.flat(json.loads((out / "run/resolved_config.json").read_text(encoding="utf-8")))


def run(label):
    out, platform, pinned, base, exp = CANDS[label]
    rec = json.loads((out / "R3_RUN_RECEIPT.json").read_text(encoding="utf-8"))
    s = json.loads((out / "run/summary.json").read_text(encoding="utf-8"))
    s = {k: v for k, v in s.items() if "test" not in k.lower()}
    epochs = [h["epoch"] for h in s["history"]]
    best_sha = DG.sha(out / "run/checkpoint_best.pt")
    v = {"rc0": rec["rc"] == 0, "status": s["status"] in ("MAX_BUDGET_REACHED", "QUALITY_GATE_FAILED"),
         "epochs_1_35": epochs == list(range(1, 36)), "best_sha_matches_summary": s["best_checkpoint_sha256"] == best_sha,
         "best_sha_pinned": best_sha == pinned}
    ref = cfg(CANDS[base][0])
    c = cfg(out)
    diff = {k: c.get(k) for k in sorted(set(c) | set(ref)) if c.get(k) != ref.get(k) and "output" not in k.lower()}
    v["cfg_diff_registered"] = diff == {} if exp is None else {k.split(".")[-1]: x for k, x in diff.items()} == DG.EXPECT_DIFF[exp]
    ps = s["best_checkpoint_validation"]["per_source"]
    pc = out / "run/best_validation_predictions.csv"
    f1, n = DG.pd_disease_f1(pc)
    pv, pd = DG.src_acc(pc, "plantvillage"), DG.src_acc(pc, "plantdoc")
    v["csv_acc_matches_summary_4dp"] = abs(pv - ps["plantvillage"]["accuracy"]) < 5.1e-5 and abs(pd - ps["plantdoc"]["accuracy"]) < 5.1e-5
    g = DG.guard(pv, pd, f1)
    return {"platform": platform, "checkpoint": str((out / "run/checkpoint_best.pt").relative_to(WS)).replace("\\", "/"),
            "checkpoint_sha256": best_sha, "best_epoch": s["best_epoch"], "status": s["status"], "validity": v, "valid": all(v.values()),
            "baseline": base, "cfg_diff": diff, "val": g, "guardrail_pass": all(g["pass"].values()), "pd_disease_classes": n,
            "lr_drop_epochs": [h["epoch"] for h in s["history"] if h.get("lr_dropped")],
            "receipt_sha256": DG.sha(out / "R3_RUN_RECEIPT.json")}


def cross_machine():
    a, b = cfg(CANDS["r3data_pc"][0]), cfg(CANDS["r3data_lap"][0])
    norm = lambda x: x.replace(PC_ROOT, LAP_ROOT).replace(PC_ROOT.replace("\\", "/"), LAP_ROOT.replace("\\", "/")) if isinstance(x, str) else x
    raw = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and "output" not in k.lower())
    after = sorted(k for k in raw if norm(a.get(k)) != norm(b.get(k)))
    return {"diff_keys_raw": raw, "diff_keys_after_root_normalisation": after}


def detector():
    p = E / "laptop_runs/detl/det_out/run/DET_RESULT.json"
    if not p.is_file():
        return {"present": False, "path": str(p)}
    d = json.loads(p.read_text(encoding="utf-8"))
    v = {"done": d["status"] == "DONE", "epochs_12": d["config"]["epochs"] == 12 and not d["config"]["smoke"],
         "exif_16_2": d["exif_rotated_skipped"] == {"train": 16, "validation": 2}, "cuda_note": "not bit-deterministic" in d["note"]}
    return {"present": True, "platform": "laptop", "validity": v, "valid": all(v.values()), "checkpoint_sha256": d["checkpoint_sha256"],
            "min_size": d["config"]["min_size"], "max_size": d["config"]["max_size"], "result_sha256": DG.sha(p)}


def main(argv):
    out_dir, dry = Path(argv[0]), "--dry" in argv
    cands = {l: run(l) for l in CANDS}
    q = DG.q5_reference()
    q["guardrail_pass"] = all(q["val"]["pass"].values())
    q["platform"] = "Colab T4"
    cands["q5central"] = q
    res = {"stage": "deploy gates PC (DEPLOY_RUNS_SPEC r3 PC section 2)", "r2_functions_from": str(SRC), "r2_sha256": DG.sha(SRC),
           "spec_sha256": DG.sha(PL / "pc_r9/DEPLOY_RUNS_SPEC_r3_PC.md"), "guardrails": DG.GUARD, "candidates": cands,
           "cross_machine_r3data": cross_machine(), "detector": detector()}
    text = json.dumps(res, ensure_ascii=False, indent=1)
    summary = {"valid": {k: c.get("valid") for k, c in cands.items()}, "guardrail_pass": {k: c["guardrail_pass"] for k, c in cands.items()},
               "cross": res["cross_machine_r3data"], "detector_valid": res["detector"].get("valid")}
    if dry:
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "DEPLOY_GATES.json").write_bytes(text.encode("utf-8"))
    summary.update(gates=str(out_dir / "DEPLOY_GATES.json"), sha256=DG.sha(out_dir / "DEPLOY_GATES.json"))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
