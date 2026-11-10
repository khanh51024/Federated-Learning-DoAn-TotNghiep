"""Read-only EXIT check of one FL run (q15l / q16l / q17l / q18l <run>_out): RESULT outcome / valid / status / rounds, endpoint = mean
val_macro_f1_raw rounds 38-40, PD accuracy + PD macro-F1 supported rounds 38-40, thermal summary, PC_RUNTIME stats, shas.
Usage: python -I -B fl_exit_check.py <outdir_name> <run>      e.g. q18l d4_scaffold_s41
"""
import hashlib
import json
import sys
from pathlib import Path

E = Path(r"experiments\train-improve-20261015")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(od, run):
    top = E / od / f"{run}_out"
    D = top / run
    r = json.loads((D / "RESULT.json").read_text(encoding="utf-8"))
    s = json.loads((D / "run/summary.json").read_text(encoding="utf-8"))
    pr = {x["round"]: x for x in r["per_round"]}
    h = {x["round"]: x for x in s["history"]}
    last = (38, 39, 40)
    ok3 = all(i in pr and i in h for i in last)
    th = json.loads((top / "THERMAL_SUMMARY.json").read_text(encoding="utf-8"))
    rt = D / "PC_RUNTIME.json"
    out = {"run": run, "outcome": r["outcome"], "valid": r["valid"], "invalid_reasons": r["invalid_reasons"], "status": r["status"],
           "stopped_round": r["stopped_round"], "n_rounds": len(pr), "job": r["job"], "scheme": r["scheme"], "method_args": r["method_args"],
           "best_round": r["best_round"], "best_ckpt": r["best_checkpoint_sha256"][:8],
           "endpoint_r38_40": round(sum(pr[i]["val_macro_f1_raw"] for i in last) / 3, 4) if ok3 else None,
           "endpoint_last3": [round(pr[i]["val_macro_f1_raw"], 5) for i in last] if ok3 else None,
           "pd_acc_r38_40": round(sum(h[i]["plantdoc_accuracy"] for i in last) / 3, 4) if ok3 else None,
           "pd_mf1s_r38_40": round(sum(pr[i]["plantdoc_macro_f1_supported"] for i in last) / 3, 4) if ok3 else None,
           "gpu_max_c": th["gpu_C"]["max"], "gpu_mean_c": round(th["gpu_C"]["mean"], 1), "pause_s": th["pause_s"], "stop": th["stop_reason"],
           "wall_h": round(r["wall_s"] / 3600, 3), "pc_runtime": json.loads(rt.read_text(encoding="utf-8"))["stats"] if rt.exists() else None,
           "sha": {f: sha(D / f) for f in ("RESULT.json", "RECEIPT.jsonl", "run/summary.json")}}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
