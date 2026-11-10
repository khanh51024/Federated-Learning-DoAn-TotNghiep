"""E2 / E3 shared val inference (pc_r9/E23_SPEC.md). CPU only, descriptive. Create-once output dir.
For every model of MODELS: checkpoint_last.pt (key model_state), MobileNetV3-Small 38, eval on manifests/val.jsonl (4,613) with the q15
runner's dataset class / canonical_v1 eval transform / collate (imported from the runner file, sha pinned), batch 64, CPU, 4 threads.
Saves logits (float32), targets, plantdoc mask, sample ids per model + metrics from the runner's evaluate_metrics, and the gate
G_REPRO (|val_macro_f1_raw recomputed - the run's own history value at that round/epoch| <= 2e-3).
Central only: PD-only passes (the 269 PlantDoc val images) for the last 3 epoch checkpoints (gen_<e>/checkpoint_last.pt) -> PD accuracy
of the last 3 epochs (Central history has no plantdoc_accuracy), gate on plantdoc_macro_f1_supported of that epoch.
Usage: python -I -B e23_predict.py <out_dir> [--probe]   (probe: 2 models, 64 val images, writes nothing outside out_dir)
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import mobilenet_v3_small

E = Path(r"experiments\train-improve-20261015")
RUNNER = E / "q15/sandbox/fedavg_mixed_runner.py"
RUNNER_SHA = "7c29f6f6a3b164743231be4ef78075cda328d6b2b2cbb960d72a529741d0d462"
DS_ROOT = Path(r"dataset")
REL = DS_ROOT / "mixed/pv_pd_v3"
G_TOL = 2e-3
MODELS = {}
for s in (42, 43, 44):
    MODELS[f"central_adamw_s{s}"] = ("central", E / ("r3l/ctrl_out/run" if s == 42 else f"n11l/adamw_s{s}_out/run"))
    MODELS[f"central_sgd_s{s}"] = ("central", E / f"n11l/sgd_s{s}_out/run")
    MODELS[f"fedavg_a01_s{s}"] = ("fl", E / f"q15l/fedavg_s{s}_out/fedavg_s{s}/run")
    MODELS[f"fedavg_a05_s{s}"] = ("fl", E / f"q16l/fedavg_a05_s{s}_out/fedavg_a05_s{s}/run")
    for a in ("1", "10", "100"):
        MODELS[f"fedavg_a{a}_s{s}"] = ("fl", E / f"q17l/fedavg_a{a}_s{s}_out/fedavg_a{a}_s{s}/run")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_runner():
    assert sha256(RUNNER) == RUNNER_SHA, "runner sha"
    sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("q15_runner", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(ck):
    st = torch.load(ck, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(st["model_state"], strict=True)
    return model.eval(), st.get("epoch", st.get("round"))


def infer(model, loader):
    lo, ta, dom, ids = [], [], [], []
    with torch.inference_mode():
        for x, y, metas in loader:
            lo.append(model(x).float().numpy())
            ta.append(y.numpy())
            dom += [m.get("source_domain", "unknown") for m in metas]
            ids += [m.get("sample_id") or m.get("annotation_id") for m in metas]
    return np.concatenate(lo), np.concatenate(ta), np.array(dom), ids


def main():
    out = Path(sys.argv[1])
    probe = "--probe" in sys.argv[2:]
    assert not out.exists(), f"create-once: {out}"
    torch.set_num_threads(4)
    R = load_runner()
    tf = R.get_transforms("canonical_v1", is_train=False)
    val = R.CanonicalClassificationDataset.from_manifest(REL / "manifests/val.jsonl", dataset_root=DS_ROOT, transform=tf,
                                                        return_metadata=True)
    assert len(val) == 4613
    pd_idx = [i for i, s in enumerate(val.samples) if s.get("source_domain") == "plantdoc"]
    assert len(pd_idx) == 269, len(pd_idx)
    idx = list(range(64)) + pd_idx[:16] if probe else list(range(len(val)))
    full = DataLoader(Subset(val, idx), batch_size=64, shuffle=False, num_workers=0, collate_fn=R.eval_collate_fn)
    pdl = DataLoader(Subset(val, pd_idx[:16] if probe else pd_idx), batch_size=64, shuffle=False, num_workers=0,
                     collate_fn=R.eval_collate_fn)
    (out / "logits").mkdir(parents=True)
    (out / "logits_pd_last3").mkdir()
    res = {"spec": "pc_r9/E23_SPEC.md", "script_sha256": sha256(__file__), "runner_sha256": RUNNER_SHA, "probe": probe,
           "torch": torch.__version__, "n_eval": len(idx), "models": {}}
    names = ["central_adamw_s42", "fedavg_a100_s42"] if probe else list(MODELS)
    for name in names:
        t0 = time.time()
        kind, rdir = MODELS[name]
        ck = rdir / "checkpoint_last.pt"
        model, step = build(ck)
        hist = json.load(open(rdir / "summary.json", encoding="utf-8"))["history"]
        key = "epoch" if kind == "central" else "round"
        h = {int(x[key]): x for x in hist}
        last = max(h)
        assert step == last, (name, step, last)
        r = R.evaluate_metrics(model, full, torch.device("cpu"))
        lo, ta, dom, ids = infer(model, full)
        pred = lo.argmax(1)
        assert abs(float((pred == ta).mean()) - r["raw_accuracy"]) < 1e-9, "logits vs evaluate_metrics"
        np.savez_compressed(out / "logits" / f"{name}.npz", logits=lo, targets=ta, plantdoc=(dom == "plantdoc"),
                            ids_sha256=np.array(hashlib.sha256("\n".join(ids).encode()).hexdigest()))
        ref = h[last]["val_macro_f1_raw"]
        row = {"checkpoint": str(ck), "checkpoint_sha256": sha256(ck), "step": step,
               "raw_macro_f1": r["raw_macro_f1"], "raw_accuracy": r["raw_accuracy"],
               "plantdoc": {k: v for k, v in (r["per_source"].get("plantdoc") or {}).items() if isinstance(v, (int, float))},
               "G_REPRO": {"ref_history_val_macro_f1_raw": ref, "abs_d": abs(r["raw_macro_f1"] - ref), "tol": G_TOL,
                           "pass": None if probe else bool(abs(r["raw_macro_f1"] - ref) <= G_TOL)},
               "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
        if kind == "central":
            last3 = []
            for e in (last - 2, last - 1):
                m_e, st_e = build(rdir / f"checkpoints/gen_{e:06d}/checkpoint_last.pt")
                assert st_e == e, (name, e, st_e)
                r_e = R.evaluate_metrics(m_e, pdl, torch.device("cpu"))
                pdm = r_e["per_source"]["plantdoc"]
                last3.append({"epoch": e, "pd_accuracy": pdm["raw_accuracy"], "pd_macro_f1_supported": pdm["raw_macro_f1_supported"],
                              "G_REPRO_pd_abs_d": abs(pdm["raw_macro_f1_supported"] - h[e]["plantdoc_macro_f1_supported"])})
                lo_e, ta_e, _, _ = infer(m_e, pdl)
                np.savez_compressed(out / "logits_pd_last3" / f"{name}_ep{e}.npz", logits=lo_e, targets=ta_e)
            pdm = r["per_source"]["plantdoc"]
            last3.append({"epoch": last, "pd_accuracy": pdm["raw_accuracy"], "pd_macro_f1_supported": pdm["raw_macro_f1_supported"],
                          "G_REPRO_pd_abs_d": abs(pdm["raw_macro_f1_supported"] - h[last]["plantdoc_macro_f1_supported"])})
            row["central_pd_last3"] = last3
            row["G_REPRO_pd_last3_pass"] = None if probe else all(x["G_REPRO_pd_abs_d"] <= G_TOL for x in last3)
        row["wall_s"] = time.time() - t0
        res["models"][name] = row
        print(name, json.dumps({"G": row["G_REPRO"], "pd": row.get("G_REPRO_pd_last3_pass"), "wall_s": round(row["wall_s"])}), flush=True)
    res["all_G_REPRO"] = None if probe else all(v["G_REPRO"]["pass"] and v.get("G_REPRO_pd_last3_pass", True)
                                                for v in res["models"].values())
    p = out / "E23_PREDICT.json"
    p.with_suffix(".tmp").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(p.with_suffix(".tmp"), p)
    print("WROTE", p, sha256(p))


if __name__ == "__main__":
    main()
