"""N11s5 = S5 Central (S5_SPEC r1 89b24b36), built by pc_r9/build_q21.py from r8_draft/n11_laptop_driver.py with ONLY: spec paths +
PINS -> E/n11s5/specs (n11 specs + jobs s45 / s46), RUNS -> sgd_s45 / sgd_s46 / adamw_s45 / adamw_s46. Original docstring follows.
N1.1 laptop driver (Central-SGD, P40): one Centralized run of the N1.1 sandbox trainer on the laptop RTX 2050, started through
p1lo_throttle_thermal.py (duty 0.7, memory fraction 0.5) with thermal monitor v2 (rule r5d). Same structure as
r3_laptop_driver_pinned.py (70af05f5); differences: N1.1 sandbox + spec, runs sgd_s42 / sgd_s43 / sgd_s44, job
cent_mixed_sgd_s<seed>, --seed-override-floor for seeds 43 / 44 only, output n11l/<run>_out. Create-once, no resume, cap 10 h.
r2 (D1): + runs adamw_s43 / adamw_s44 = R3-ctrl recipe (spec central_spec_n11_adamw.json, job cent_mixed_r3c_s<seed>, N 35,
default AdamW path of the same sandbox, --seed-override-floor). adamw_s42 = R3-ctrl (r3l/ctrl_out), not run here.
Usage: python -B n11_laptop_driver.py <sgd_s42|sgd_s43|sgd_s44|adamw_s43|adamw_s44>   env N11_SMOKE_ROOT=<dir> -> smoke into that dir
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(r"<WORKSPACE_ROOT>")
E = ROOT / "experiments/train-improve-20261015"
PL = ROOT / "plans/cai-thien-train-20261015"
N11 = E / "n11"
SANDBOX = N11 / "sandbox/train_production.py"
SPEC = E / "n11s5/specs/central_spec_n11s5.json"
SPEC_ADAMW = E / "n11s5/specs/central_spec_n11s5_adamw.json"
REL_SHA = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"
INCUMBENT = "training-workflows/dist/fedavg-v7-non-iid-20261003/preflight_extract/incumbent"
WEIGHTS = "training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth"
THROTTLE = (Path(r"plans\cai-thien-train-20261015\pc_r9\p1lo_throttle_thermal_pc_r3.py"), "b5c34e05de7bfbcd1792ad8dc69ede8aa673a7df9e348457b218f4c492c27534")   # PC (pc_r9): gated throttle r3, nw via PC_NUM_WORKERS; was f7103fb9
MONITOR_V2 = (PL / "tools_draft/thermal_monitor_v2.py", "b1b622d5f3080c265bdf6532d379b609573dc38f2f43c2fd7f94fdf2c75b4d2b")
PINS = {"sandbox": "35540d108ed9967e18e21e7343a94240b7b862c20e3c1549fad9a32a90d5daba",
        "spec": "fb4f60b9e411ee87070f8f58b33dc3118b367740982ad1be58c6855ea89ba4df",
        "spec_adamw": "f7a19fdc5167f58d10bdc746c07e8d2e41a731ba13ee3f8352142693013e37db"}  # real build = mirror (n11_build 77c955c5)
CAP_S = 10 * 3600.0
RUNS = {"sgd_s45": (45, "sgd"), "sgd_s46": (46, "sgd"), "adamw_s45": (45, "r3c"), "adamw_s46": (46, "r3c")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_for(run, out_dir, smoke):
    seed, kind = RUNS[run]
    c = [sys.executable, "-B", str(THROTTLE[0]), str(SANDBOX), "--mode", "smoke" if smoke else "full",
         "--job-id", f"cent_mixed_{kind}_s{seed}", "--spec-file", str(SPEC if kind == "sgd" else SPEC_ADAMW), "--workspace-root", str(ROOT),
         "--dataset-root", str(ROOT / "dataset"), "--release-dir", str(ROOT / "dataset/mixed/pv_pd_v3"),
         "--expected-release-sha", REL_SHA, "--init", "imagenet_v1", "--pretrained-weights", str(ROOT / WEIGHTS),
         "--plantdoc-sampling-ratio", "0.25", "--sampler-repeat-policy", "with_replacement", "--transform-version", "canonical_v1",
         "--incumbent-summary", str(ROOT / INCUMBENT / "central_summary.json"),
         "--incumbent-checkpoint", str(ROOT / INCUMBENT / "central_checkpoint_best.pt"), "--output-dir", str(out_dir)]
    if seed != 42 and not smoke:
        c.append("--seed-override-floor")
    assert "--label-smoothing" not in c
    return c


def main(argv):
    assert len(argv) == 1 and argv[0] in RUNS, argv
    run = argv[0]
    assert sha(THROTTLE[0]) == THROTTLE[1] and sha(MONITOR_V2[0]) == MONITOR_V2[1], "tools sha"
    assert sha(SANDBOX) == PINS["sandbox"], "N1.1 sandbox trainer sha"
    assert sha(SPEC) == PINS["spec"] and sha(SPEC_ADAMW) == PINS["spec_adamw"], "N1.1 spec sha"
    smoke = os.environ.get("N11_SMOKE_ROOT")
    out_root = Path(smoke) / f"{run}_out" if smoke else E / "n11l" / f"{run}_out"
    assert not out_root.exists(), ("create-once, no resume", out_root)
    out_root.mkdir(parents=True)
    mon = importlib.util.module_from_spec(importlib.util.spec_from_file_location("mv2", MONITOR_V2[0]))
    mon.__spec__.loader.exec_module(mon)
    m = mon.MonitorV2(out_root)
    env = dict(os.environ, PYTHONUNBUFFERED="1", P1LO_PAUSE_FLAG=str(m.pause), P1LO_STOP_FLAG=str(m.stop),
               WORKSPACE_ROOT=str(ROOT))
    env.setdefault("Q5_GPU_DUTY", "0.7")
    th = threading.Thread(target=m.loop, daemon=True)
    th.start()
    c = cmd_for(run, out_root / "run", bool(smoke))
    rec = {"run": run, "cmd": c, "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pins": PINS}
    print(json.dumps({"driver": "n11l", "run": run, "out_root": str(out_root)}), flush=True)
    t0 = time.time()
    reason = None
    with open(out_root / "run.log", "w", encoding="utf-8") as logf:
        p = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT), env=env)

        def pump():
            for line in p.stdout:
                sys.stdout.write(line)
                logf.write(line)
        rd = threading.Thread(target=pump, daemon=True)
        rd.start()
        while p.poll() is None:
            if time.time() - t0 >= CAP_S:
                reason = "CAP_REACHED"
                p.kill()
                break
            time.sleep(1.0)
        p.wait()
        rd.join(timeout=30)
    m.flag.set()
    th.join(timeout=5)
    rc = p.returncode if reason is None else 5
    if m.stop.exists():
        rc = 7
    summ = m.summary()
    rec.update({"rc": rc, "stop_reason": reason, "wall_s": time.time() - t0, "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "log_sha256": sha(out_root / "run.log"), "thermal": summ})
    (out_root / "N11_RUN_RECEIPT.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_root / "THERMAL_SUMMARY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
