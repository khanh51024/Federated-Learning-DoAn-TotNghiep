"""E5 / N4.1 - static post-training quantisation (int8) of the 4 web-package models (CPU only).

Spec: pc_r9/E5_SPEC.md. Descriptive measurement, no winner rule.
Models: web package r2 (<PC_PATH>/pc_transfer_r2/web_pc_v3_a01_s42_r2): centralized_adamw, centralized_sgd, fedavg, scaffold
(alpha 0.1, s42, last epoch/round), MobileNetV3-Small 38 classes, key model_state.
PTQ: torch.ao FX graph mode, get_default_qconfig_mapping("x86"), engine x86, calibration on 512 train images
(release pv_pd_v3 manifests/train.jsonl, random.Random(42).sample, eval transform canonical_v1), no val/test in calibration.
Metrics: val (manifests/val.jsonl, 4,613 images) with the q15 runner's own dataset class, eval transform, collate and
evaluate_metrics (imported from the runner file, not copied). Latency: CPU, batch 1, 1x3x224x224 fixed tensor,
20 warm-up + 200 timed forward passes, median and p90, at torch threads 1 and 4.
Arms (registered before any val number): static_ptq (as asked) and dynamic_linear (quantize_dynamic, nn.Linear only, qint8),
added because the train-only probe showed static PTQ collapses on MobileNetV3-Small (also on the ImageNet weights).
fp32 is also exported as frozen TorchScript so latency is compared like for like (frozen TorchScript vs frozen TorchScript).
Modes: probe (plumbing only: fedavg, 32 calib + 64 other TRAIN images, 20 timed passes; no val) | full.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import importlib.util
import io
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping, quantize_dynamic
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Subset
from torchvision.models import mobilenet_v3_small

RUNNER = Path(r"experiments\train-improve-20261015\q15\sandbox\fedavg_mixed_runner.py")
RUNNER_SHA = "7c29f6f6a3b164743231be4ef78075cda328d6b2b2cbb960d72a529741d0d462"
DS_ROOT = Path(r"dataset")
REL = DS_ROOT / "mixed" / "pv_pd_v3"
PKG = Path(r"<PC_PATH>/pc_transfer_r2\web_pc_v3_a01_s42_r2")
MODELS = {
    "centralized_adamw": "0a8293a34cd3c52b0993d63a9cd9646e30672c2e5f1d21928af1c50495902815",
    "centralized_sgd": None,
    "fedavg": None,
    "scaffold": "4817fbaceff086a64d3af79484c761743baa64b917c5e595131070944927bb05",
}
N_CALIB, CALIB_SEED, CALIB_BS = 512, 42, 32
WARMUP, TIMED, THREADS = 20, 200, (1, 4)
G_FP32_TOL = 2e-3


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_runner():
    assert sha256(RUNNER) == RUNNER_SHA, "runner sha"
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("q15_runner", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_model(role):
    info = json.load(open(PKG / role / "model_info.json", encoding="utf-8"))
    ck = PKG / role / "checkpoint_best.pt"
    s = sha256(ck)
    assert s == info["sha256"], (role, s, info["sha256"])
    if MODELS[role]:
        assert s == MODELS[role], (role, s)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    state = torch.load(ck, map_location="cpu", weights_only=False)[info["checkpoint_key"]]
    model.load_state_dict(state, strict=True)
    return model.eval(), info, s


def state_bytes(m):
    b = io.BytesIO()
    torch.save(m.state_dict(), b)
    return len(b.getvalue())


def latency(m, threads, n_timed):
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 224, 224, generator=g)
    with torch.inference_mode():
        for _ in range(WARMUP):
            m(x)
        t = []
        for _ in range(n_timed):
            a = time.perf_counter()
            m(x)
            t.append((time.perf_counter() - a) * 1000.0)
    t.sort()
    return {"threads": threads, "n": n_timed, "median_ms": statistics.median(t),
            "p90_ms": t[int(0.9 * (len(t) - 1))], "min_ms": t[0], "max_ms": t[-1]}


def metrics(R, model, loader):
    torch.set_num_threads(4)
    r = R.evaluate_metrics(model, loader, torch.device("cpu"))
    pd = r.get("per_source", {}).get("plantdoc", {})
    keep = {"macro_f1": r["raw_macro_f1"], "accuracy": r["raw_accuracy"], "loss": r["raw_loss"],
            "macro_f1_supported": r["raw_macro_f1_supported"], "total_samples": r["total_samples"]}
    keep["all_keys"] = sorted(r.keys())
    keep["plantdoc"] = {k: v for k, v in pd.items() if isinstance(v, (int, float))}
    return keep, r


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "full"])
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir)
    assert not out.exists(), f"create-once: {out} exists"
    probe = a.mode == "probe"
    torch.backends.quantized.engine = "x86"
    R = load_runner()
    tf = R.get_transforms("canonical_v1", is_train=False)
    train_ds = R.CanonicalClassificationDataset.from_manifest(REL / "manifests/train.jsonl", dataset_root=DS_ROOT,
                                                             transform=tf, return_metadata=True)
    n_train = len(train_ds)
    assert n_train == 37238, n_train
    pick = random.Random(CALIB_SEED).sample(range(n_train), N_CALIB + (64 if probe else 0))
    calib_idx = sorted(pick[:32] if probe else pick[:N_CALIB])
    calib_ids = [train_ds.samples[i].get("sample_id", train_ds.samples[i].get("annotation_id")) for i in calib_idx]
    assert all(calib_ids), "calib ids"
    calib_loader = DataLoader(Subset(train_ds, calib_idx), batch_size=CALIB_BS, shuffle=False, num_workers=0,
                              collate_fn=R.eval_collate_fn)
    if probe:
        eval_loader = DataLoader(Subset(train_ds, sorted(pick[N_CALIB:])), batch_size=64, shuffle=False,
                                 num_workers=0, collate_fn=R.eval_collate_fn)
        eval_name = "probe_train_64 (NOT val)"
    else:
        val_ds = R.CanonicalClassificationDataset.from_manifest(REL / "manifests/val.jsonl", dataset_root=DS_ROOT,
                                                               transform=tf, return_metadata=True)
        assert len(val_ds) == 4613, len(val_ds)
        eval_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=R.eval_collate_fn)
        eval_name = "val 4613 (manifests/val.jsonl)"
    out.mkdir(parents=True)
    res = {"spec": "pc_r9/E5_SPEC.md", "mode": a.mode, "script_sha256": sha256(__file__), "runner_sha256": RUNNER_SHA,
           "plant_data_contract": sys.modules[R.CanonicalClassificationDataset.__module__].__file__,
           "torch": torch.__version__, "quant_engine": torch.backends.quantized.engine, "arms": {"static_ptq": "prepare_fx/convert_fx, get_default_qconfig_mapping('x86'), 512 train calib", "dynamic_linear": "quantize_dynamic({nn.Linear}, qint8), no calibration"},
           "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "eval_set": eval_name,
           "calib": {"n": len(calib_idx), "seed": CALIB_SEED, "source": "manifests/train.jsonl",
                     "ids_sha256": hashlib.sha256("\n".join(calib_ids).encode("utf-8")).hexdigest()},
           "latency_protocol": {"batch": 1, "input": "1x3x224x224 randn(seed 0), preprocessing excluded", "warmup": WARMUP,
                                "timed": 20 if probe else TIMED, "threads": list(THREADS)},
           "models": {}}
    roles = ["fedavg"] if probe else list(MODELS)
    for role in roles:
        t0 = time.time()
        fp32, info, ck_sha = build_model(role)
        ex = (torch.zeros(1, 3, 224, 224),)
        prepared = prepare_fx(build_model(role)[0], get_default_qconfig_mapping("x86"), ex)
        torch.set_num_threads(4)
        with torch.inference_mode():
            for images, _, _ in calib_loader:
                prepared(images)
        arms = {"static_ptq": convert_fx(prepared).eval(),
                "dynamic_linear": quantize_dynamic(build_model(role)[0], {nn.Linear}, dtype=torch.qint8).eval()}
        rdir = out / role
        rdir.mkdir()
        xb = next(iter(calib_loader))[0][:8]
        tsm = {}
        for name, mod in [("fp32", fp32)] + list(arms.items()):
            with torch.inference_mode():
                ts = torch.jit.freeze(torch.jit.trace(mod, ex).eval())
            path = rdir / f"model_{name}_torchscript.pt"
            torch.jit.save(ts, str(path))
            ts2 = torch.jit.load(str(path), map_location="cpu").eval()
            with torch.inference_mode():
                l_m, l_ts = mod(xb), ts2(xb)
            tsm[name] = {"module": ts2, "file": path, "reload_check": {
                "argmax_equal": bool(torch.equal(l_m.argmax(1), l_ts.argmax(1))),
                "max_abs_logit_diff": float((l_m - l_ts).abs().max())}}
        m32, _ = metrics(R, fp32, eval_loader)
        ref = info["metrics_validation_pv_pd_v3"]["val_macro_f1_at_checkpoint"]
        g_fp32 = None if probe else bool(abs(m32["macro_f1"] - ref) <= G_FP32_TOL)
        n_t = res["latency_protocol"]["timed"]
        row = {"checkpoint": str(PKG / role / "checkpoint_best.pt"), "checkpoint_sha256": ck_sha,
               "checkpoint_file_bytes": (PKG / role / "checkpoint_best.pt").stat().st_size,
               "G_FP32_REPRO": {"ref_val_macro_f1_at_checkpoint": ref, "fp32_cpu": m32["macro_f1"],
                                "abs_d": abs(m32["macro_f1"] - ref), "tol": G_FP32_TOL, "pass": g_fp32},
               "fp32": {"metrics": m32, "state_dict_bytes": state_bytes(fp32),
                        "torchscript_file_bytes": tsm["fp32"]["file"].stat().st_size,
                        "torchscript_sha256": sha256(tsm["fp32"]["file"]), "reload_check": tsm["fp32"]["reload_check"],
                        "latency_eager": [latency(fp32, t, n_t) for t in THREADS],
                        "latency_torchscript": [latency(tsm["fp32"]["module"], t, n_t) for t in THREADS]}}
        for name, mod in arms.items():
            m8, _ = metrics(R, tsm[name]["module"], eval_loader)
            row[name] = {"metrics": m8, "state_dict_bytes": state_bytes(mod),
                         "torchscript_file_bytes": tsm[name]["file"].stat().st_size,
                         "torchscript_sha256": sha256(tsm[name]["file"]), "reload_check": tsm[name]["reload_check"],
                         "latency_torchscript": [latency(tsm[name]["module"], t, n_t) for t in THREADS],
                         "delta_minus_fp32": {"macro_f1": m8["macro_f1"] - m32["macro_f1"],
                                              "accuracy": m8["accuracy"] - m32["accuracy"]}}
        row["wall_s"] = time.time() - t0
        res["models"][role] = row
        print(role, json.dumps({"G": row["G_FP32_REPRO"], **{n: {"d": row[n]["delta_minus_fp32"], "bytes": row[n]["torchscript_file_bytes"],
              "lat_ms": [x["median_ms"] for x in row[n]["latency_torchscript"]]} for n in arms},
              "fp32": {"bytes": row["fp32"]["torchscript_file_bytes"], "lat_ms": [x["median_ms"] for x in row["fp32"]["latency_torchscript"]]},
              "wall_s": round(row["wall_s"])}), flush=True)
    res["all_G_FP32_REPRO"] = None if probe else all(v["G_FP32_REPRO"]["pass"] for v in res["models"].values())
    p = out / ("E5_PROBE.json" if probe else "E5_RESULT.json")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    print("WROTE", p, sha256(p))


if __name__ == "__main__":
    main()
