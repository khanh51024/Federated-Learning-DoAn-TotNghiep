"""Q4B/Q5 Colab orchestrator. Centralized runs on account A, Local-only on account B (D-ACCT); both use the
shared Drive folder MyDrive/q5/. Every notebook cell calls exactly one command of this file.

  setup    verify the extracted bundles, the contract source and the environment
  probe    central                one full-mode epoch (timing)            -> probe/central
           local-straight         client_00 iid, rounds=3, uninterrupted  -> probe/local_straight
           local-start            same run, killed right after round 2    -> probe/local_resume
           local-resume           fresh runtime: --resume to round 3, compare with local-straight
           local-val              time one val pass with q5_eval (val = selection set)
  plan     write Q5_PLAN.json from the probe timings (caps from STATE D-WALL)
  run      central | local <scheme> <client_id>
  status

Exit codes: 0 done/skip, 4 GPU/torch/CUDA differs from the run's pin -> STOP (G-GPU),
5 cap reached or would be exceeded -> STOP and ask the user (D-WALL), 6 run incomplete -> re-run in a new session.
"h GPU" = sum of (segment_end - segment_start) of the run's own receipt (D-WALL clarification 1).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
REL_SHA = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"
SCHEMES = ("iid", "label_alpha_1_0", "label_alpha_0_1")
CLIENTS = tuple(f"client_{i:02d}" for i in range(5))
CAP_S = {"central": 8 * 3600.0, "local": 11 * 3600.0}
CENTRAL_JOB = "cent_mixed_q5_s42"
PROBE_LOCAL_ROUNDS = 3
INCUMBENT = "training-workflows/dist/fedavg-v7-non-iid-20261003/preflight_extract/incumbent"
WEIGHTS = "training-artifacts/pretrained_weights/mobilenet_v3_small-047dcff4.pth"
SPECS = HERE / "specs"
FL = HERE / "fl_bundle"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def env_info():
    import torch
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}


def write_once_json(path, obj):
    path = Path(path)
    assert not path.exists(), f"create-once: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Run:
    """One run = one directory with its own receipt and GPU pin (no two machines append to one file)."""

    def __init__(self, drive, name):
        self.dir = Path(drive) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.receipt = self.dir / "RECEIPT.jsonl"
        self.pin = self.dir / "GPU_PIN.json"

    def log(self, event, **kw):
        rec = dict(kw, utc=utc(), event=event)
        with open(self.receipt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps(rec, ensure_ascii=True, sort_keys=True), flush=True)

    def events(self, event=None):
        if not self.receipt.is_file():
            return []
        rows = [json.loads(l) for l in self.receipt.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [r for r in rows if event is None or r["event"] == event]

    def spent(self):
        return sum(e["wall_s"] for e in self.events("segment_end"))

    def gpu_guard(self):
        cur = env_info()
        assert cur["cuda_available"], "no CUDA device in this runtime"
        key = {k: cur[k] for k in ("gpu", "torch", "cuda")}
        if self.pin.exists():
            pin = json.loads(self.pin.read_text(encoding="utf-8"))
            if {k: pin[k] for k in key} != key:
                self.log("GPU_MISMATCH_STOP", pinned=pin, current=key)
                sys.exit(4)
        else:
            write_once_json(self.pin, dict(key, pinned_utc=utc()))
        self.log("session", **key)
        return key


def segment(run, cmd, cap_s, label, kill_when=None):
    """Run one trainer process as one receipt segment. Watchdog: stop at the cap (exit 5) or when kill_when()."""
    run.gpu_guard()
    spent = run.spent()
    if spent >= cap_s:
        run.log("CAP_REACHED", spent_s=spent, cap_s=cap_s, label=label)
        sys.exit(5)
    log_dir = run.dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{label}_{utc().replace(':', '')}.log"
    run.log("segment_start", label=label, cmd=cmd, spent_before_s=spent)
    t0 = time.time()
    reason = None
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(WS),
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))

        def pump():
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while proc.poll() is None:
            if spent + (time.time() - t0) >= cap_s:
                reason = "CAP_REACHED"
            elif kill_when is not None and kill_when():
                reason = "PLANNED_KILL"
            if reason:
                proc.kill()
                break
            time.sleep(0.5)
        proc.wait()
        reader.join(timeout=30)
    wall = time.time() - t0
    run.log("segment_end", label=label, rc=proc.returncode, wall_s=wall, stop_reason=reason,
            log=log_path.name, log_sha256=sha256_file(log_path))
    if reason == "CAP_REACHED":
        run.log("CAP_REACHED", spent_s=spent + wall, cap_s=cap_s, label=label)
        sys.exit(5)
    return proc.returncode, reason


def central_cmd(output_dir, sync_dir, stop_after=None):
    cmd = [sys.executable, "-B", str(WS / "train-tap-trung/scripts/train_production.py"), "--mode", "full",
           "--job-id", CENTRAL_JOB, "--spec-file", str(SPECS / "central_spec_q5.json"),
           "--dataset-root", str(WS / "dataset"), "--release-dir", str(WS / "dataset/mixed/pv_pd_v3"),
           "--expected-release-sha", REL_SHA, "--init", "imagenet_v1", "--pretrained-weights", str(WS / WEIGHTS),
           "--plantdoc-sampling-ratio", "0.25", "--sampler-repeat-policy", "with_replacement",
           "--transform-version", "canonical_v1",
           "--incumbent-summary", str(WS / INCUMBENT / "central_summary.json"),
           "--incumbent-checkpoint", str(WS / INCUMBENT / "central_checkpoint_best.pt"),
           "--output-dir", str(output_dir), "--colab-sync-dir", str(sync_dir)]
    if stop_after is not None:
        cmd += ["--stop-after-epoch", str(stop_after)]
    if (Path(output_dir) / "checkpoint_last.pt").is_file():
        cmd += ["--resume", str(Path(output_dir) / "checkpoint_last.pt")]
    assert "--label-smoothing" not in cmd   # G-PIPE-2
    return cmd


def local_cmd(scheme, client, output_dir, rounds=None):
    cmd = [sys.executable, "-B", str(FL / "local_only_mixed_runner.py"), "--dataset-root", str(WS / "dataset"),
           "--spec-file", str(SPECS / "fedavg_spec_q5.json"), "--pretrained-weights", str(WS / WEIGHTS),
           "--partition-scheme", scheme, "--client-id", client, "--plantdoc-sampling-ratio", "0.25",
           "--device", "cuda", "--output-dir", str(output_dir)]
    if rounds is not None:
        cmd += ["--rounds", str(rounds)]
    if (Path(output_dir) / client / "checkpoint_last.pt").is_file():
        cmd.append("--resume")
    return cmd


def last_round(output_dir, client):
    p = Path(output_dir) / client / "checkpoint_last.pt"
    if not p.is_file():
        return 0
    import torch
    try:
        return int(torch.load(p, map_location="cpu", weights_only=False)["round"])
    except Exception:  # noqa: BLE001  (mid-replace)
        return -1


def cmd_setup(drive, args):
    man = json.loads((Path(args.bundle_dir) / "Q5_BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [m["path"] for m in man["code_bundle"]["members"] if sha256_file(WS / m["path"]) != m["sha256"]]
    assert not bad, ("code members differ", bad[:5])
    assert sha256_file(WS / "dataset/mixed/pv_pd_v3/release_manifest.json") == REL_SHA
    probe = ("import sys, hashlib, pathlib; ws = pathlib.Path(sys.argv[1]); "
             "[sys.path.insert(0, str(p)) for p in (ws / 'plant_data_contract', ws / 'train-tap-trung', ws)]; "
             "import plant_data_contract.dataset as d; f = pathlib.Path(d.__file__).resolve(); print(f)")
    out = subprocess.run([sys.executable, "-B", "-c", probe, str(WS)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    assert Path(out.stdout.split()[0]).parent == (WS / "plant_data_contract/plant_data_contract").resolve()
    env = env_info()
    assert env["cuda_available"], "Runtime > Change runtime type > GPU"
    print(json.dumps({"setup": "OK", "env": env, "code_bundle_sha256": man["code_bundle"]["sha256"],
                      "code_members": len(man["code_bundle"]["members"])}, sort_keys=True))
    return 0


def cmd_probe(drive, args):
    local = Path(args.local_root)
    if args.step == "central":
        run = Run(drive, "probe/central")
        if run.events("segment_end"):
            print(json.dumps({"skip": "probe central done"}))
            return 0
        rc, _ = segment(run, central_cmd(local / "q5_probe_central", run.dir / "sync", stop_after=1), 3600.0, "central_e1")
        assert rc == 0, rc
        return 0
    if args.step == "local-straight":
        run = Run(drive, "probe/local_straight")
        if last_round(run.dir / "out", "client_00") >= PROBE_LOCAL_ROUNDS:
            print(json.dumps({"skip": "local-straight done"}))
            return 0
        rc, _ = segment(run, local_cmd("iid", "client_00", run.dir / "out", PROBE_LOCAL_ROUNDS), 3600.0, "straight")
        assert rc == 0 and last_round(run.dir / "out", "client_00") == PROBE_LOCAL_ROUNDS
        return 0
    if args.step == "local-start":
        run = Run(drive, "probe/local_resume")
        if last_round(run.dir / "out", "client_00") >= 2:
            print(json.dumps({"skip": "local-start done"}))
            return 0
        rc, reason = segment(run, local_cmd("iid", "client_00", run.dir / "out", PROBE_LOCAL_ROUNDS), 3600.0,
                             "start", kill_when=lambda: last_round(run.dir / "out", "client_00") == 2)
        assert reason == "PLANNED_KILL" and last_round(run.dir / "out", "client_00") == 2, (rc, reason)
        print("NOW: Runtime > Disconnect and delete runtime, reconnect a GPU, then run CELL 1, CELL 2 and the local-resume cell.")
        return 0
    if args.step == "local-resume":
        run = Run(drive, "probe/local_resume")
        out = run.dir / "out"
        if last_round(out, "client_00") < PROBE_LOCAL_ROUNDS:
            assert last_round(out, "client_00") == 2, "run local-start first"
            rc, _ = segment(run, local_cmd("iid", "client_00", out, PROBE_LOCAL_ROUNDS), 3600.0, "resume")
            assert rc == 0 and last_round(out, "client_00") == PROBE_LOCAL_ROUNDS
        import torch
        a = torch.load(Path(drive) / "probe/local_straight/out/client_00/checkpoint_last.pt", map_location="cpu", weights_only=False)
        b = torch.load(out / "client_00/checkpoint_last.pt", map_location="cpu", weights_only=False)
        eq = {k: bool(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k])) for k in a["model_state_dict"]}
        res = {"bit_exact_model": all(eq.values()), "n_tensors": len(eq), "n_equal": sum(eq.values()),
               "round": [a["round"], b["round"]],
               "max_abs_diff": max(float((a["model_state_dict"][k].float() - b["model_state_dict"][k].float()).abs().max())
                                   for k in a["model_state_dict"])}
        run.log("resume_compare", **res)
        return 0
    if args.step == "local-val":
        run = Run(drive, "probe/local_val")
        if run.events("val_timed"):
            print(json.dumps({"skip": "local-val done"}))
            return 0
        run.gpu_guard()
        ck = Path(drive) / "probe/local_straight/out/client_00/checkpoint_best.pt"
        t0 = time.time()
        r = subprocess.run([sys.executable, "-B", str(HERE / "q5_eval.py"), str(WS), "probe_local", "local_only",
                            str(ck), "val", str(run.dir / "eval")], capture_output=True, text=True, cwd=str(WS))
        assert r.returncode == 0, r.stderr[-1500:]
        run.log("val_timed", wall_s=time.time() - t0, result=json.loads(r.stdout.strip().splitlines()[-1]))
        return 0
    raise SystemExit(f"unknown probe step {args.step}")


def cmd_plan(drive, args):
    path = Path(drive) / "Q5_PLAN_COLAB.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return 0
    import torch
    sync = Path(drive) / "probe/central/sync"
    m = json.loads((sync / "commit_manifest.json").read_text(encoding="utf-8"))
    hist = torch.load(sync / m["last_checkpoint_path"], map_location="cpu", weights_only=False)["history"]
    c_end = Run(drive, "probe/central").events("segment_end")[0]
    epoch_s = c_end["wall_s"]                      # one full epoch incl. val + startup: conservative
    straight = Run(drive, "probe/local_straight").events("segment_end")[0]
    round_s_client0 = straight["wall_s"] / PROBE_LOCAL_ROUNDS
    val_s = Run(drive, "probe/local_val").events("val_timed")[0]["wall_s"]
    sizes = {}
    for s in SCHEMES:
        rows = {c: sum(1 for l in open(WS / f"dataset/mixed/pv_pd_v3/partitions/{s}/{c}.jsonl", encoding="utf-8") if l.strip())
                for c in CLIENTS}
        sizes[s] = rows
    n0 = sizes["iid"]["client_00"]
    per_sample = max(0.0, round_s_client0 - val_s) / n0
    est_local = {s: 60 * sum(per_sample * n + val_s for n in sizes[s].values()) for s in SCHEMES}
    plan = {"central": {"epoch_s_probe": epoch_s, "estimate_s_60": 60 * epoch_s, "cap_s": CAP_S["central"],
                        "history_duration_s_epoch1": hist[0]["duration_sec"]},
            "local": {"round_s_client00_iid": round_s_client0, "val_s": val_s, "per_sample_s": per_sample,
                      "client_sizes": sizes, "estimate_s_60": est_local, "cap_s": CAP_S["local"]},
            "verdict": {"central": "OK" if 60 * epoch_s <= CAP_S["central"] else "OVER_CAP_ASK_USER",
                        **{f"local_{s}": "OK" if est_local[s] <= CAP_S["local"] else "OVER_CAP_ASK_USER" for s in SCHEMES}},
            "gpu": env_info()["gpu"], "partition_spec_sha256": sha256_file(WS / "dataset/mixed/pv_pd_v3/partition_spec_v4.json"),
            "note": "tier 2 estimate from one probe; Q5 caps enforced by the watchdog regardless"}
    write_once_json(path, plan)
    print(json.dumps(plan["verdict"], sort_keys=True))
    return 0 if all(v == "OK" for v in plan["verdict"].values()) else 5


def migrate_shared_out(run):
    """q5 patch v2: move a finished client_00 from the old shared out/ to clients/client_00 (once)."""
    old, new = run.dir / "out", run.dir / "clients"
    if not old.exists():
        return
    assert not new.exists(), "both out/ and clients/ exist: STOP, report"
    names = sorted(p.name for p in old.iterdir())
    assert names == ["client_00", "summary.json"], ("unexpected content of out/", names)
    s = json.loads((old / "summary.json").read_text(encoding="utf-8"))
    assert sorted(s["clients"]) == ["client_00"], ("summary clients", sorted(s["clients"]))
    assert last_round(old, "client_00") == 60, "client_00 not finished"
    new.mkdir()
    os.rename(old, new / "client_00")
    assert last_round(new / "client_00", "client_00") == 60
    run.log("layout_migrated", src="out", dst="clients/client_00", patch="q5_colab_v2")


def cmd_run(drive, args):
    local = Path(args.local_root)
    if args.kind == "central":
        run = Run(drive, "runs/central")
        sync = run.dir / "sync"
        m = json.loads((sync / "commit_manifest.json").read_text(encoding="utf-8")) if (sync / "commit_manifest.json").is_file() else None
        if (sync / "summary.json").is_file() and m and json.loads((sync / "summary.json").read_text(encoding="utf-8")).get("status") not in (None, "PAUSED_CHECKPOINTED"):
            print(json.dumps({"skip": "central finished", "generation": m["generation"]}))
            return 0
        out = local / "q5_central"
        rc, _ = segment(run, central_cmd(out, sync), CAP_S["central"], "central")
        summary = sync / "summary.json"
        status = json.loads(summary.read_text(encoding="utf-8")).get("status") if summary.is_file() else None
        if rc != 0 or status in (None, "PAUSED_CHECKPOINTED"):
            run.log("RUN_INCOMPLETE", rc=rc, status=status)
            sys.exit(6)
        run.log("run_done", status=status)
        return 0
    assert args.scheme in SCHEMES and args.client in CLIENTS, (args.scheme, args.client)
    run = Run(drive, f"runs/local_{args.scheme}")
    migrate_shared_out(run)
    out = run.dir / "clients" / args.client
    if last_round(out, args.client) >= 60:
        print(json.dumps({"skip": args.client, "round": 60}))
        return 0
    rc, _ = segment(run, local_cmd(args.scheme, args.client, out), CAP_S["local"], args.client)
    if rc != 0 or last_round(out, args.client) < 60:
        run.log("RUN_INCOMPLETE", client=args.client, rc=rc, round=last_round(out, args.client))
        sys.exit(6)
    run.log("client_done", client=args.client)
    return 0


def cmd_status(drive, args):
    st = {}
    for name in ["runs/central"] + [f"runs/local_{s}" for s in SCHEMES]:
        r = Run(drive, name)
        st[name] = {"spent_h": round(r.spent() / 3600, 3), "pin": json.loads(r.pin.read_text()) if r.pin.exists() else None}
        if name.startswith("runs/local_"):
            st[name]["rounds"] = {c: last_round(r.dir / "clients" / c, c) for c in CLIENTS}
    print(json.dumps(st, sort_keys=True))
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", required=True)
    ap.add_argument("--local-root", default="/content")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--bundle-dir", required=True)
    s = sub.add_parser("probe")
    s.add_argument("step", choices=("central", "local-straight", "local-start", "local-resume", "local-val"))
    sub.add_parser("plan")
    s = sub.add_parser("run")
    s.add_argument("kind", choices=("central", "local"))
    s.add_argument("scheme", nargs="?")
    s.add_argument("client", nargs="?")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    drive = Path(args.drive_root)
    assert drive.is_dir(), f"Drive root missing: {drive}"
    return {"setup": cmd_setup, "probe": cmd_probe, "plan": cmd_plan, "run": cmd_run, "status": cmd_status}[args.cmd](drive, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
