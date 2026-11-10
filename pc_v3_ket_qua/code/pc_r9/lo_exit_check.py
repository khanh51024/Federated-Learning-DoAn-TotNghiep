"""Read-only EXIT check of Local-only runs (lo17l/<run>_out): receipt valid / rc / invalid_reasons, run endpoint (mean over the 5
clients of mean val_macro_f1_raw rounds 38-40, as lo17_laptop_driver), per-client endpoint, PD accuracy (client mean of rounds 38-40),
rounds per client, lr constant, thermal summary, PC_RUNTIME, shas, partition digest if recorded.
Usage: python -I -B lo_exit_check.py <run> [<run> ...]
"""
import hashlib
import json
import sys
from pathlib import Path

E = Path(r"experiments\train-improve-20261015")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(run):
    D = E / "lo17l" / f"{run}_out"
    r = json.loads((D / "LO17_RUN_RECEIPT.json").read_text(encoding="utf-8"))
    s = json.loads((D / "run/summary.json").read_text(encoding="utf-8"))
    cl = s["clients"]
    rounds = {c: [x["round"] for x in v["history"]] for c, v in cl.items()}
    lr = sorted({x.get("lr") for v in cl.values() for x in v["history"]})
    pd = {c: sum(x["plantdoc_accuracy"] for x in v["history"][-3:]) / 3 for c, v in cl.items()}
    th = json.loads((D / "THERMAL_SUMMARY.json").read_text(encoding="utf-8"))
    digest = {k: v for k, v in s.items() if "digest" in k}
    e = r["endpoint_val_macro_f1_raw"] or {}
    return {"run": run, "rc": r["rc"], "valid": r["valid"], "invalid_reasons": r["invalid_reasons"], "wall_h": round(r["wall_s"] / 3600, 3),
            "endpoint": round(e.get("run_endpoint", float("nan")), 6), "per_client": {k: round(v, 4) for k, v in (e.get("per_client_mean_last3") or {}).items()},
            "pd_acc_client_mean": round(sum(pd.values()) / len(pd), 4), "rounds_ok": all(v == list(range(1, 41)) for v in rounds.values()),
            "n_clients": len(cl), "lr_set": lr, "gpu_max_c": th["gpu_C"]["max"], "pause_s": th["pause_s"], "stop": th["stop_reason"],
            "digest": digest or "not recorded in summary", "sha_receipt": sha(D / "LO17_RUN_RECEIPT.json"), "sha_summary": sha(D / "run/summary.json")}


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(json.dumps(one(run), ensure_ascii=False))
