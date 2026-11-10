"""EcoQoS watcher (INCIDENT I6, TU VAN 2026-11-02): scheduler v2s4 keeps eco_done as a set of PIDs, so a REUSED pid of a new training
process is never opted out and Windows may apply EcoQoS (BelowNormal + background) -> slower FL rounds. Until the end of the campaign
this tool replaces that part: every 5 s, for every descendant (recursive) of the running scheduler (python laptop_sched_v2s4.py), key
(pid, create_time); a key not seen yet -> the scheduler's OWN eco_optout(pid) (SetProcessInformation ProcessPowerThrottling,
ControlMask 1, StateMask 0), then read back with GetProcessInformation and log ctrl / state. Never kills, never changes priority,
touches no file of the runs. Log: experiments/train-improve-20261015/laptop_sched/eco_watch.jsonl (append).
Usage: python -I -B eco_watch.py        (stops when no scheduler process exists for 10 min)
"""
import ctypes
import importlib.util
import json
import time
from ctypes import wintypes
from pathlib import Path

import psutil

E = Path(r"experiments\train-improve-20261015")
LOG = E / "laptop_sched/eco_watch.jsonl"
SCHED_PY = Path(r"plans\cai-thien-train-20261015\tools_draft\laptop_sched_v2s4.py")
_s = importlib.util.spec_from_file_location("sched_v2s4", SCHED_PY)
SCH = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SCH)          # top level only defines constants / functions (no side effect)

K = ctypes.WinDLL("kernel32", use_last_error=True)
K.OpenProcess.restype = wintypes.HANDLE
K.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
K.GetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
K.GetProcessInformation.restype = wintypes.BOOL
K.CloseHandle.argtypes = (wintypes.HANDLE,)


class PPTS(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def read_state(pid):
    h = K.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        s = PPTS(1, 0, 0)
        ok = K.GetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))
        return [s.ControlMask, s.StateMask] if ok else None
    finally:
        K.CloseHandle(h)


def log(**kw):
    with open(LOG, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dict(kw, t=time.strftime("%Y-%m-%dT%H:%M:%S")), ensure_ascii=False) + "\n")


def schedulers():
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "laptop_sched_v2s4.py" in c and "eco_watch" not in c:
            out.append(p)
    return out


def main():
    seen = set()
    log(event="START")
    gone_since = None
    while True:
        sch = schedulers()
        if not sch:
            gone_since = gone_since or time.time()
            if time.time() - gone_since > 600:
                log(event="STOP_no_scheduler")
                return
        else:
            gone_since = None
        for root in sch:
            try:
                kids = root.children(recursive=True)
            except psutil.Error:
                continue
            for q in kids:
                try:
                    key = (q.pid, q.create_time())
                    if key in seen:
                        continue
                    seen.add(key)
                    before = read_state(q.pid)
                    ok = SCH.eco_optout(q.pid)
                    after = read_state(q.pid)
                    log(event="OPTOUT", pid=q.pid, ctime=key[1], cmd=" ".join(q.cmdline())[-90:], ok=ok, before=before, after=after)
                except psutil.Error:
                    pass
        time.sleep(5)


if __name__ == "__main__":
    main()
