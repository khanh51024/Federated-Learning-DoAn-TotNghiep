"""Q5 local: collect the finished Local-only runs (validation-side facts only; no test set is read).
Writes <run_root>/LOCAL_RESULT.json create-once (atomic). Refuses unless 3 schemes x 5 clients reached round 60.
Usage: python -B q5_local_collect.py [run_root]     default q5/local/runs_rtx2050
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCHEMES = ("iid", "label_alpha_1_0", "label_alpha_0_1")
CLIENTS = [f"client_{i:02d}" for i in range(5)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(root):
    root = Path(root)
    out = root / "LOCAL_RESULT.json"
    assert not out.exists(), f"create-once: {out}"
    res = {"run_root": str(root), "schemes": {}}
    for s in SCHEMES:
        d = root / "runs" / f"local_{s}"
        rec = [json.loads(l) for l in (d / "RECEIPT.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        sch = {"gpu_pin": json.loads((d / "GPU_PIN.json").read_text(encoding="utf-8")),
               "receipt_sha256": sha(d / "RECEIPT.jsonl"),
               "gpu_s_logged": round(sum(r["wall_s"] for r in rec if r["event"] == "segment_end"), 1),
               "gpu_mismatch": any(r["event"] == "GPU_MISMATCH_STOP" for r in rec), "clients": {}}
        for c in CLIENTS:
            cd = d / "clients" / c
            last = torch.load(cd / c / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            assert int(last["round"]) == 60, (s, c, last["round"])
            summ = json.loads((cd / "summary.json").read_text(encoding="utf-8"))["clients"][c]
            starts = sum(r["event"] == "segment_start" and r["label"] == c for r in rec)
            sch["clients"][c] = {"n_k": summ["n_k"], "best_validation": summ["best_validation"],
                                 "class_coverage_train_labels": summ["class_coverage"]["train_label_count"],
                                 "checkpoint_best_sha256": sha(cd / c / "checkpoint_best.pt"),
                                 "segment_starts": starts,
                                 "flag": "RESUMED_NOT_BIT_EXACT" if starts > 1 else None}
        res["schemes"][s] = sch
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=True, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({"written": str(out), "sha256": sha(out)}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "runs_rtx2050")
