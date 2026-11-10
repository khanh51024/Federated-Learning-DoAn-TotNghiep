"""PC scheduler v2s4 = v2s3 bd45c131 with VRAM guard 7168 MiB (RTX 2060 8 GiB) and SLOTS.txt values 1 / 2 / 3.
v2s3 docstring: Laptop scheduler v2s3 = v2s2 cd2b96df where running external driver runs occupy slots (no WAIT_EXTERNAL at start).
v2s2 docstring: Laptop scheduler v2s2 = v2s 4137b793 with slots read every loop from laptop_sched/SLOTS.txt ("1" / "2", else 1; changes logged)
and EcoQoS opt-out for every process of every running item (user 29-10 via advisor 14: 2 processes again; "turn off efficiency").
v2s docstring: Laptop scheduler v2s = v2 072dc0ca with ONE slot and BelowNormal priority (user 29-10 via advisor 14: laptop sluggish ->
runs one at a time). Detects the n11 / q16 / r3x / det drivers as external (waits at start until they exit).
v2 docstring: Laptop scheduler v2 (PLAN r7c: R1 9 runs + R3 runs; user 26-10 "2 tien trinh"; advisor conditions 26-10 + T2 27-10).
Slot-based: at most 2 training drivers at once; the next queue item starts when a slot is free and every guard passes.
Queue = experiments/train-improve-20261015/laptop_sched/QUEUE.json, written only by the coordinator (atomic replace), items
only APPENDED: {"closed": bool, "items": [{"name", "cmd": [...], "log", "flags_dir"}]}. Items already seen must stay
byte-identical (else halt). An item whose flags_dir exists is skipped (no resume). The scheduler exits when the queue is
closed, every item is done and nothing runs. Drivers are started exactly as their launchers do (same files, same args,
env Q5_GPU_DUTY 0.7, Q5_GPU_MEM_FRACTION 0.5); each run keeps its own thermal monitor and PAUSE/STOP flags.
Guards every 10 s (only on runs started here):
  - start a run only if VRAM <= 3584 MiB, RAM available >= 2 GiB, D: free >= 5 GiB (T2);
  - 2 runs and (VRAM > VRAM_MAX_MIB or RAM available < 2 GiB) -> PAUSE.flag "pair_guard ..." in the YOUNGER run;
  - D: free < 2 GiB -> PAUSE.flag "pair_guard disk ..." in EVERY running run; released when D: free > 3 GiB (T2);
  - a guard only removes flags whose text starts with "pair_guard" (thermal pauses of the monitors are never touched);
  - STOP.flag in one run -> STOP.flag in the other, no new start; a run exiting rc != 0 -> no new start.
Log: experiments/train-improve-20261015/laptop_sched/SCHED.jsonl + SCHED_LATEST.json + SCHED_RESULT_<stamp>.json.
Usage: python -B laptop_sched_v2.py          self-test of the guard logic: python -B laptop_sched_v2.py --selftest <scratch>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(r"<WORKSPACE_ROOT>")
E = ROOT / "experiments/train-improve-20261015"
OUT = E / "laptop_sched"
QUEUE = OUT / "QUEUE.json"
VRAM_MAX_MIB = 7168
RAM_MIN_B = 2 * 2 ** 30
DISK_START_MIN_B = 5 * 2 ** 30
DISK_PAUSE_B = 2 * 2 ** 30
DISK_RESUME_B = 3 * 2 ** 30
TICK_S = 10
BELOW_NORMAL = 0x00004000
DRIVER_NAMES = ("p3lo_driver.py", "q14_laptop_driver.py", "p1lo_driver.py", "p3l_driver.py", "q15_laptop_driver.py",
                "r3_laptop_driver.py", "laptop_pair_sched.py",
                "n11_laptop_driver.py", "q16_laptop_driver.py", "r3x_laptop_driver.py", "det_laptop_driver.py")


SLOTS_FILE = OUT / "SLOTS.txt"


def read_slots():
    try:
        v = SLOTS_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return 1
    return int(v) if v in ("1", "2", "3") else 1


def eco_optout(pid):
    """EcoQoS off for one process (SetProcessInformation ProcessPowerThrottling, ControlMask 1, StateMask 0)."""
    import ctypes
    from ctypes import wintypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
    if not h:
        return False
    try:
        s = PPTS(1, 1, 0)
        return bool(k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))  # ProcessPowerThrottling = 4
    finally:
        k.CloseHandle(h)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(ev, **kw):
    rec = {"t": utc(), "event": ev, **kw}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "SCHED.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def vram_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def ram_avail():
    return psutil.virtual_memory().available


def disk_free():
    return shutil.disk_usage("<PC_PATH>/").free


def external_drivers(own):
    out = []
    for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
        cl = p.info["cmdline"] or []
        if p.info["pid"] not in own and any(any(c.endswith(d) for d in DRIVER_NAMES) for c in cl):
            out.append({"pid": p.info["pid"], "ppid": p.info["ppid"], "cmd": " ".join(cl[-2:])})
    return out


def tree(proc):
    try:
        return {proc.pid} | {c.pid for c in psutil.Process(proc.pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        return {proc.pid}


def start_blockers(vram, ram, free):
    why = []
    if vram is None or vram > VRAM_MAX_MIB:
        why.append(f"vram {vram} MiB > {VRAM_MAX_MIB}" if vram is not None else "vram unknown")
    if ram < RAM_MIN_B:
        why.append(f"ram available {ram / 2 ** 30:.2f} GiB < 2")
    if free < DISK_START_MIN_B:
        why.append(f"D: free {free / 2 ** 30:.2f} GiB < 5")
    return why


def set_guard(flags_dir, why):
    """why non-empty -> ensure a pair_guard PAUSE.flag; empty -> remove it if it is ours. Returns action or None."""
    pf = Path(flags_dir) / "PAUSE.flag"
    if not Path(flags_dir).is_dir():
        return None
    if why:
        if not pf.exists():
            pf.write_text("pair_guard " + "; ".join(why), encoding="utf-8")
            return "GUARD_PAUSE"
        return None
    if pf.exists():
        try:
            txt = pf.read_text(encoding="utf-8")
        except OSError:
            return None
        if txt.startswith("pair_guard"):
            pf.unlink(missing_ok=True)
            return "GUARD_RELEASE"
    return None


def cross_stop(running):
    src = [r for r in running if (Path(r["flags_dir"]) / "STOP.flag").exists()]
    if not src:
        return None
    for r in running:
        sf = Path(r["flags_dir"]) / "STOP.flag"
        if Path(r["flags_dir"]).is_dir() and not sf.exists():
            sf.write_text(f"pair_guard cross stop from {src[0]['name']}", encoding="utf-8")
    return src[0]["name"]


def read_queue(seen):
    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    for it in q["items"]:
        if it["name"] in seen:
            assert json.dumps(seen[it["name"]], sort_keys=True) == json.dumps(it, sort_keys=True), ("queue item changed", it["name"])
    names = [it["name"] for it in q["items"]]
    assert len(names) == len(set(names)), "duplicate names"
    return q


def launch(it):
    Path(it["log"]).parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("Q14_", "Q15_", "P1LO_", "R3_")) and k != "Q5_OUT"}
    env.update({"Q5_GPU_DUTY": "1.0", "Q5_GPU_MEM_FRACTION": "0.5", "PYTHONUNBUFFERED": "1", "PC_NUM_WORKERS": "2"})   # PC (pc_r9 P0.6B): duty + num_workers; laptop value was duty 0.7, no PC_NUM_WORKERS
    fh = open(it["log"], "ab")
    p = subprocess.Popen(it["cmd"], cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL)
    log("START", run=it["name"], pid=p.pid, log=it["log"])
    return {**it, "proc": p, "fh": fh, "t0": time.time(), "t_start": utc(), "guard_s": 0.0, "g0": None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ext0 = external_drivers(set())
    if ext0:
        log("EXTERNAL_AT_START", runs=sorted({d["cmd"] for d in ext0}))
    seen, done, running, results, halt = {}, set(), [], [], None
    disk_paused = False
    slots_now, eco_done = None, set()
    while True:
        SLOTS = read_slots()
        if SLOTS != slots_now:
            log("SLOTS", slots=SLOTS, previous=slots_now)
            slots_now = SLOTS
        for r in running:
            for pid in sorted(tree(r["proc"]) - eco_done):
                eco_done.add(pid)
                log("ECO_OPTOUT", run=r["name"], pid=pid, ok=eco_optout(pid))
        q = read_queue(seen)
        for it in q["items"]:
            seen.setdefault(it["name"], it)
        todo = [it for it in q["items"] if it["name"] not in done and it["name"] not in {r["name"] for r in running}]
        src = cross_stop(running)
        if src and not halt:
            halt = f"STOP.flag in {src}"
            log("CROSS_STOP", source=src, runs=[r["name"] for r in running])
        free = disk_free()
        if free < DISK_PAUSE_B and not disk_paused:
            disk_paused = True
            for r in running:
                if set_guard(r["flags_dir"], [f"disk D: free {free / 2 ** 30:.2f} GiB < 2"]) == "GUARD_PAUSE":
                    r["g0"] = time.time()
            log("DISK_PAUSE", free_gib=round(free / 2 ** 30, 2))
        elif disk_paused and free > DISK_RESUME_B:
            disk_paused = False
            for r in running:
                if set_guard(r["flags_dir"], []) == "GUARD_RELEASE" and r["g0"]:
                    r["guard_s"] += time.time() - r["g0"]
                    r["g0"] = None
            log("DISK_RESUME", free_gib=round(free / 2 ** 30, 2))
        if len(running) >= 2 and not disk_paused:
            young = max(running, key=lambda r: r["t0"])
            why = [w for w in start_blockers(vram_mib(), ram_avail(), DISK_START_MIN_B) if not w.startswith("D:")]
            act = set_guard(young["flags_dir"], why)
            if act == "GUARD_PAUSE":
                young["g0"] = time.time()
                log(act, run=young["name"], why=why)
            elif act == "GUARD_RELEASE":
                young["guard_s"] += time.time() - (young["g0"] or time.time())
                young["g0"] = None
                log(act, run=young["name"])
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["fh"].close()
            running.remove(r)
            done.add(r["name"])
            for o in running:
                if not disk_paused and set_guard(o["flags_dir"], []) == "GUARD_RELEASE":
                    o["guard_s"] += time.time() - (o["g0"] or time.time())
                    o["g0"] = None
                    log("GUARD_RELEASE", run=o["name"], reason="partner exited")
            res = {"run": r["name"], "rc": rc, "start": r["t_start"], "end": utc(), "wall_h": (time.time() - r["t0"]) / 3600,
                   "parallel_note": f"scheduler v2s2, slots {slots_now}, BelowNormal + EcoQoS off; wall depends on the partner run",
                   "guard_pause_s": r["guard_s"]}
            results.append(res)
            log("EXIT", **res)
            if rc != 0 and not halt:
                halt = f"{r['name']} exit {rc}"
        own_now = set().union(*[tree(r["proc"]) for r in running]) if running else set()
        n_ext = len({d["cmd"] for d in external_drivers(own_now)})
        if not halt and not disk_paused and todo and len(running) + n_ext < SLOTS:
            it = todo[0]
            if Path(it["flags_dir"]).exists():
                done.add(it["name"])
                log("SKIP_EXISTS", run=it["name"], flags_dir=it["flags_dir"])
            else:
                why = start_blockers(vram_mib(), ram_avail(), disk_free())
                if why:
                    log("WAIT_START", run=it["name"], why=why)
                else:
                    running.append(launch(it))
        (OUT / "SCHED_LATEST.json").write_text(json.dumps(
            {"t": utc(), "running": [r["name"] for r in running], "halt": halt, "disk_paused": disk_paused,
             "free_D_gib": round(disk_free() / 2 ** 30, 2), "vram_mib": vram_mib(), "ram_avail_gib": round(ram_avail() / 2 ** 30, 2),
             "queue_closed": q.get("closed"), "done": sorted(done), "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
        if not running and (halt or (q.get("closed") and not todo)):
            break
        time.sleep(TICK_S)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (OUT / f"SCHED_RESULT_{stamp}.json").write_text(json.dumps({"t": utc(), "halt": halt, "results": results}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")
    log("DONE", halt=halt, n=len(results))
    return 0 if not halt else 2


def selftest(scratch):
    d = Path(scratch)
    d.mkdir(parents=True)
    a, b = d / "a", d / "b"
    a.mkdir()
    b.mkdir()
    gib = 2 ** 30
    assert start_blockers(3000, 3 * gib, 10 * gib) == []
    assert len(start_blockers(7200, 1 * gib, 4 * gib)) == 3 and start_blockers(None, 3 * gib, 10 * gib)
    assert set_guard(b, ["x"]) == "GUARD_PAUSE" and (b / "PAUSE.flag").read_text(encoding="utf-8").startswith("pair_guard")
    assert set_guard(b, ["x"]) is None and set_guard(b, []) == "GUARD_RELEASE" and not (b / "PAUSE.flag").exists()
    (b / "PAUSE.flag").write_text("gpu", encoding="utf-8")
    assert set_guard(b, []) is None and (b / "PAUSE.flag").exists()
    (b / "PAUSE.flag").unlink()
    ra, rb = {"name": "A", "flags_dir": str(a)}, {"name": "B", "flags_dir": str(b)}
    assert cross_stop([ra, rb]) is None
    (a / "STOP.flag").write_text("GPU >= 90 C", encoding="utf-8")
    assert cross_stop([ra, rb]) == "A" and (b / "STOP.flag").read_text(encoding="utf-8").startswith("pair_guard")
    global QUEUE
    QUEUE = d / "QUEUE.json"
    it1 = {"name": "x", "cmd": ["a"], "log": "l", "flags_dir": "f"}
    QUEUE.write_text(json.dumps({"closed": False, "items": [it1]}), encoding="utf-8")
    seen = {}
    for it in read_queue(seen)["items"]:
        seen[it["name"]] = it
    QUEUE.write_text(json.dumps({"closed": True, "items": [it1, {**it1, "name": "y"}]}), encoding="utf-8")
    assert len(read_queue(seen)["items"]) == 2
    QUEUE.write_text(json.dumps({"closed": True, "items": [{**it1, "cmd": ["changed"]}]}), encoding="utf-8")
    try:
        read_queue(seen)
        raise SystemExit("changed item not refused")
    except AssertionError:
        pass
    print("SELFTEST_OK", d)


if __name__ == "__main__":
    sys.exit(selftest(sys.argv[2]) if sys.argv[1:2] == ["--selftest"] else main())
