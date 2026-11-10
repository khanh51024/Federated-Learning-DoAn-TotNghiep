"""Q19 PC driver (PHASE_C_SPEC_PC r2), built by pc_r9/build_q19.py from r8_draft/q18_laptop_driver.py d7b34fc9: kaggle =
pc_r9/q19_laptop_kaggle.py (own sha), Q18 lock still checked (sandbox files; its kaggle pin = the q18 source a92e52e1),
output q19l, smoke env Q19_LOCAL_TEST + Q19_SMOKE_ROOT. Original docstring follows.
Q18 (PLAN r8 D4 screening) laptop / PC driver: run one of the 5 D4 runs with r8_draft/q18_laptop_kaggle.py, through
p1lo_throttle_thermal.py (duty 0.7, memory fraction 0.5) with thermal monitor v2 (rule r5d). Copy of r8_draft/q17_laptop_driver.py
(fb12829e) with ONLY: run driver file + sha, Q18 stage lock (pins the Q15 lock sha and the kaggle driver) instead of the Q17 lock,
output root experiments/train-improve-20261015/q18l/<run>_out, smoke env names (Q18_LOCAL_TEST + Q18_SMOKE_ROOT). No resume.
Usage: python -B q18_laptop_driver.py <run>
"""
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(r"<WORKSPACE_ROOT>")
E = ROOT / "experiments/train-improve-20261015"
Q14K = (ROOT / "plans/cai-thien-train-20261015/pc_r9/q19_laptop_kaggle.py", "6aa699263454529b6710f964894f57df4413e7beea94faf79069219d666cf05a")
Q18K_SRC_SHA = "a92e52e1e317b3de63e363744f71bdf29519dca56d404889259ddadc88556557"   # q18 kaggle = source of q19, pinned by the Q18 lock
Q15_ROOT = Path(r"experiments\train-improve-20261015\q15")
Q13_LOCK = (Q15_ROOT / "Q15_STAGE_LOCK.json", "98fb09d43197aabcb59d3377b9dff4d9aa8282d2ab8f59d303db6a06f82e2521")
Q18_ROOT = Path(r"experiments\train-improve-20261015\q18")
Q18_LOCK = (Q18_ROOT / "Q18_STAGE_LOCK.json", "3a5bfdfecf623e52a0173169ddb80f295de1635114d70f0f6d818fbf92091e27")
P1LO = ROOT / "plans/cai-thien-train-20261015/p1lo_draft"
THROTTLE = (Path(r"plans\cai-thien-train-20261015\pc_r9\p1lo_throttle_thermal_pc_r3.py"), "b5c34e05de7bfbcd1792ad8dc69ede8aa673a7df9e348457b218f4c492c27534")   # PC (pc_r9): gated throttle r3, nw via PC_NUM_WORKERS; was f7103fb9
MONITOR_SRC = (P1LO / "p1lo_driver.py", "6e1c95d44e1720077e9f51053450627068095cbbf2ab6dae00b16960473eb4fa")
MONITOR_V2 = (ROOT / "plans/cai-thien-train-20261015/tools_draft/thermal_monitor_v2.py", "b1b622d5f3080c265bdf6532d379b609573dc38f2f43c2fd7f94fdf2c75b4d2b")
RUNS = tuple([f"c_t_{m}_s41" for m in ("fedprox_mu0p001", "fedprox_mu0p01", "fedprox_mu0p1", "moon_mu0p1", "moon_mu1", "moon_mu5")]
             + [f"c_m_fedbn_{a}_s{s}" for s in (42, 43, 44) for a in ("a01", "a05")])
LAPTOP_CAP_H = 10.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv):
    assert len(argv) == 1 and argv[0] in RUNS, argv
    run = argv[0]
    assert sha(Q13_LOCK[0]) == Q13_LOCK[1], "Q15 lock sha"
    lock = json.loads(Q13_LOCK[0].read_text(encoding="utf-8"))
    bad = [k for k, v in lock["files"].items() if sha(Q15_ROOT / k) != v]
    assert not bad, ("Q15 lock mismatch", bad[:3])
    assert sha(Q18_LOCK[0]) == Q18_LOCK[1], "Q18 lock sha"
    lock18 = json.loads(Q18_LOCK[0].read_text(encoding="utf-8"))
    assert lock18["q15_stage_lock_sha256"] == Q13_LOCK[1] and lock18["kaggle_driver_sha256"] == Q18K_SRC_SHA, "Q18 lock pins"
    bad18 = [k for k, v in lock18["files"].items() if sha(Q18_ROOT / k) != v]
    assert not bad18, ("Q18 lock mismatch", bad18)
    assert sha(THROTTLE[0]) == THROTTLE[1] and sha(MONITOR_SRC[0]) == MONITOR_SRC[1], "p1lo tools sha"
    assert sha(Q14K[0]) == Q14K[1], "q15_laptop_kaggle sha"
    assert sha(MONITOR_V2[0]) == MONITOR_V2[1], "monitor v2 sha"
    os.environ["Q15_ROOT"] = str(Q15_ROOT)
    os.environ["Q18_ROOT"] = str(Q18_ROOT)
    k = load("q18_laptop_kaggle", Q14K[0])
    assert k.RUNS[run][2] == "q18" and k.Q15_ROOT == Q15_ROOT and k.Q18_ROOT == Q18_ROOT, run
    k.CAP_S = LAPTOP_CAP_H * 3600.0
    orig = k.fed_cmd

    def fed_cmd(run_, out_dir):
        cmd = orig(run_, out_dir)
        assert cmd[1] == "-B" and cmd[2].endswith("fedavg_mixed_runner.py"), cmd[:3]  # q4a and q8r2 runners share the name
        return cmd[:2] + [str(THROTTLE[0])] + cmd[2:]
    k.fed_cmd = fed_cmd
    smoke = os.environ.get("Q19_SMOKE_ROOT")   # self-test only, together with Q18_LOCAL_TEST=1 (set before start)
    assert (smoke is None) == (os.environ.get("Q19_LOCAL_TEST") != "1"), "smoke needs both Q19_SMOKE_ROOT and Q19_LOCAL_TEST=1"
    out_root = Path(smoke) / f"{run}_out" if smoke else E / "q19l" / f"{run}_out"
    assert not out_root.exists(), ("output exists; no resume in Q18", out_root)
    out_root.mkdir(parents=True)
    mon = load("thermal_monitor_v2", MONITOR_V2[0]).MonitorV2(out_root)
    os.environ["Q5_OUT"] = str(out_root)
    os.environ.setdefault("Q5_GPU_DUTY", "0.7")
    os.environ["P1LO_PAUSE_FLAG"] = str(mon.pause)
    os.environ["P1LO_STOP_FLAG"] = str(mon.stop)
    th = threading.Thread(target=mon.loop, daemon=True)
    th.start()
    print(json.dumps({"driver": "q19l", "run": run, "out_root": str(out_root), "cap_h": LAPTOP_CAP_H,
                      "duty": os.environ["Q5_GPU_DUTY"]}), flush=True)
    rc = 1
    try:
        try:
            rc = k.main([run])
        except SystemExit as e:
            rc = e.code
        except AssertionError as e:
            print(json.dumps({"assertion": str(e)}), flush=True)
            rc = 7 if mon.stop.exists() else 1
    finally:
        mon.flag.set()
        th.join(timeout=5)
        summ = mon.summary()
        summ.update({"run": run, "rc": rc, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        (out_root / "THERMAL_SUMMARY.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
