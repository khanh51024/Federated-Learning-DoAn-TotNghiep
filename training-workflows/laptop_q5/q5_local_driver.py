"""Q5 local (laptop RTX 2050): run Local-only for ONE scheme, 5 clients, with the SAME orchestration as Colab
(q5/colab/q5_colab_v2.py: receipt, GPU pin, 11 h cap watchdog, skip finished, --resume of a cut client), only:
  * WS = this workspace (dataset/, training-artifacts/, q4a specs + fl_bundle read in place, not copied);
  * the runner is started through q5_throttle.py (GPU duty cycle Q5_GPU_DUTY, memory fraction 0.5);
  * run root = q5/local/runs_rtx2050/runs/local_<scheme>/ (separate from every Colab run).
Training command, data, hyperparameters, seed, 60 rounds: identical to Colab.
Usage: python -B q5_local_driver.py <scheme|all> [--smoke <scratch_root>]
Exit codes as q5_colab: 0 done, 4 GPU/torch/CUDA differs from the pin, 5 cap (14 h/scheme here), 6 incomplete.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"D:\university\do-an-tot-nghiep")
Q4A = ROOT / "experiments/train-improve-20261015/q4a"
HERE = Path(__file__).resolve().parent
V2 = ROOT / "experiments/train-improve-20261015/q5/colab/q5_colab_v2.py"
V2_SHA = "6d322f7c2ac04a8d1b5c07e204eabcd77ea9d7a1ba9b372d890c7fd6d4170e58"
THROTTLE = HERE / "q5_throttle.py"
SCHEMES = ("iid", "label_alpha_1_0", "label_alpha_0_1")
CLIENTS = [f"client_{i:02d}" for i in range(5)]


def main(argv):
    if argv[0] == "all":   # iid -> label_alpha_1_0 -> label_alpha_0_1, stop at the first non-zero exit
        for s in SCHEMES:
            rc = main([s] + argv[1:])
            if rc != 0:
                return rc
        print("ALL SCHEMES DONE", flush=True)
        return 0
    scheme = argv[0]
    assert scheme in SCHEMES, scheme
    smoke = len(argv) == 3 and argv[1] == "--smoke"
    assert len(argv) == 1 or smoke, argv
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == V2_SHA, "q5_colab_v2.py sha"
    spec = importlib.util.spec_from_file_location("q5_colab_v2", V2)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.WS, m.SPECS, m.FL = ROOT, Q4A / "specs", Q4A / "fl_bundle"
    m.CAP_S["local"] = 14 * 3600.0   # user-approved 2026-10-20: laptop ~11.7 h/scheme > registered 11 h
    orig = m.local_cmd

    def local_cmd(scheme_, client, output_dir, rounds=None):
        cmd = orig(scheme_, client, output_dir, rounds)
        assert cmd[1] == "-B" and cmd[2].endswith("local_only_mixed_runner.py"), cmd[:3]
        cmd = cmd[:2] + [str(THROTTLE)] + cmd[2:]
        if smoke:
            cmd += ["--rounds", "2", "--max-batches", "3"]
        return cmd
    m.local_cmd = local_cmd
    os.environ.setdefault("Q5_GPU_DUTY", "0.7")
    run_root = Path(argv[2]) if smoke else HERE / "runs_rtx2050"
    run_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"driver": "q5_local", "scheme": scheme, "run_root": str(run_root),
                      "duty": os.environ["Q5_GPU_DUTY"], "smoke": smoke}), flush=True)
    for c in CLIENTS:
        try:
            rc = m.main(["--drive-root", str(run_root), "--local-root", str(run_root / "_local"),
                         "run", "local", scheme, c])
        except SystemExit as e:
            rc = e.code
        if rc != 0:
            print(json.dumps({"stop": c, "exit": rc}), flush=True)
            return rc
    print(f"SCHEME DONE {scheme}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
