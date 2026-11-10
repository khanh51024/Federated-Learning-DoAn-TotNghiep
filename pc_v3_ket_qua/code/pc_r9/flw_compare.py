"""FLW equivalence compare (draft r1). CPU, read-only. Compares two runner output dirs (the runner --output-dir, i.e. .../run):
  per round r in 1..N (common rounds): history row equal after dropping timing keys (p22_gates TIMING list), and
  |val_macro_f1_raw A - B|; checkpoint_last model_state tensor-equal when both runs ended at the same round;
  endpoint (mean val_macro_f1_raw r38-40) difference when both have 40 rounds.
  r2 (TU VAN 2026-11-09): TIMING asserted equal to p22_gates.TIMING (source sha pinned); a stripped key starting with
  val_ / plantdoc_ / train_ / loss / acc / macro aborts; the full list of stripped key names is written to the output.
Usage: python -I -B flw_compare.py <run_dir_A> <run_dir_B> <out_json>     (out_json create-once)
"""
import hashlib
import json
import sys
from pathlib import Path

TIMING = ("duration", "vram", "telemetry", "time", "_sec", "elapsed", "updated_at", "utc", "_ms")   # == p22_gates.TIMING (asserted)
P22 = (Path(__file__).resolve().parent / "p22_gates.py", "1c5b2bd1dc2af84ceab7781ac259d3aa3eac2eadae8e5e76c8c84cd955a89e76")
PROTECTED = ("val_", "plantdoc_", "train_", "loss", "acc", "macro")
STRIPPED = set()


def check_timing_list():
    src = P22[0].read_text(encoding="utf-8")
    assert hashlib.sha256(P22[0].read_bytes()).hexdigest() == P22[1], "p22_gates sha"
    line = next(x for x in src.splitlines() if x.startswith("TIMING = "))
    assert eval(line.split("=", 1)[1].strip()) == TIMING, "TIMING list differs from p22_gates"


def strip(h, path=""):
    if isinstance(h, dict):
        out = {}
        for k, v in h.items():
            if any(t in k for t in TIMING) and not isinstance(v, (dict, list)):   # containers are compared inside (stricter)
                assert not k.startswith(PROTECTED), ("protected key would be stripped", path + "." + k)
                STRIPPED.add(k)
            else:
                out[k] = strip(v, path + "." + k)
        return out
    if isinstance(h, list):
        return [strip(x, path) for x in h]
    return h


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return path or "<root>"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else path


def load(d):
    s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
    return {x["round"]: x for x in s["history"]}


def main(a, b, out):
    import torch
    out = Path(out)
    assert not out.exists(), out
    check_timing_list()
    ha, hb = load(a), load(b)
    common = sorted(set(ha) & set(hb))
    rows = []
    for r in common:
        d = first_diff(strip(ha[r]), strip(hb[r]))
        rows.append({"round": r, "history_equal": d is None, "first_diff": d,
                     "abs_d_val_macro_f1_raw": abs(ha[r]["val_macro_f1_raw"] - hb[r]["val_macro_f1_raw"])})
    res = {"A": str(a), "B": str(b), "rounds_A": max(ha), "rounds_B": max(hb), "common_rounds": len(common), "rows": rows,
           "all_history_equal": all(x["history_equal"] for x in rows),
           "max_abs_d_val_macro_f1_raw": max(x["abs_d_val_macro_f1_raw"] for x in rows),
           "timing_list": list(TIMING), "timing_list_source": f"p22_gates.py {P22[1]}", "protected_prefixes": list(PROTECTED),
           "stripped_keys": sorted(STRIPPED)}
    if max(ha) == max(hb):
        ca = torch.load(Path(a) / "checkpoint_last.pt", map_location="cpu", weights_only=False)["model_state"]
        cb = torch.load(Path(b) / "checkpoint_last.pt", map_location="cpu", weights_only=False)["model_state"]
        res["checkpoint_last_tensor_equal"] = set(ca) == set(cb) and all(torch.equal(ca[k], cb[k]) for k in ca)
        fl = [k for k in ca if ca[k].is_floating_point()]
        res["checkpoint_last_max_abs_diff"] = max(float((ca[k].double() - cb[k].double()).abs().max()) for k in fl)
    if 40 in ha and 40 in hb:
        ea = sum(ha[i]["val_macro_f1_raw"] for i in (38, 39, 40)) / 3
        eb = sum(hb[i]["val_macro_f1_raw"] for i in (38, 39, 40)) / 3
        res.update(endpoint_A=ea, endpoint_B=eb, endpoint_abs_diff=abs(ea - eb))
    b_ = json.dumps(res, ensure_ascii=False, indent=1).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b_)
    print(json.dumps({k: res[k] for k in res if k != "rows"}), hashlib.sha256(b_).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
