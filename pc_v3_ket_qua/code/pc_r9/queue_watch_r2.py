"""Queue watcher r2 (TU VAN PASS 2026-11-07): copy of queue_watch.py 179c3874 with ONLY the RAM rule changed: RAM avail < 2.0 GiB in the
last 2 consecutive builtin_log.csv rows (was < 2.5 GiB in the last row; 2-slot steady state is 2.4-2.6 GiB). Original docstring follows.
Queue watcher (TU VAN 2026-11-02, usage saving): replaces the 30-min checks. Read-only. Exits (prints one JSON line) on the first of:
SCHED.jsonl new event EXIT / HALT / PAUSE / STOP / DISK* or any rc != 0; GPU temp >= 80 C or RAM avail < 2.5 GiB (thermal builtin_log.csv
last row); thermal builtin_warnings.jsonl appears / grows; eco_watch.jsonl ok false; SLOTS != 2 with no p19 / p20 smoke process;
scheduler or eco_watch process gone; heartbeat 3 h. Polls every 60 s.
Usage: python -I -B queue_watch.py
"""
import json
import time
from pathlib import Path

import psutil

E = Path(r"experiments\train-improve-20261015")
SCHED, SLOTS, ECO = E / "laptop_sched/SCHED.jsonl", E / "laptop_sched/SLOTS.txt", E / "laptop_sched/eco_watch.jsonl"
TLOG, TWARN = E / "thermal/builtin_log.csv", E / "thermal/builtin_warnings.jsonl"
KEYS = ("EXIT", "HALT", "PAUSE", "STOP", "DISK")


def size(p):
    return p.stat().st_size if p.exists() else 0


def new_lines(p, off):
    with open(p, "rb") as fh:
        fh.seek(off)
        b = fh.read()
    cut = b.rfind(b"\n") + 1
    return [l for l in b[:cut].decode("utf-8", "replace").splitlines() if l.strip()], off + cut


def procs():
    out = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            c = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "queue_watch.py" not in c:
            out.append(c)
    return out


def done(**kw):
    print(json.dumps(dict(kw, t=time.strftime("%Y-%m-%d %H:%M:%S")), ensure_ascii=False))
    raise SystemExit(0)


def main():
    t0, so, eo, w0 = time.time(), size(SCHED), size(ECO), size(TWARN)
    while True:
        lines, so = new_lines(SCHED, so)
        hits = []
        for l in lines:
            try:
                d = json.loads(l)
            except ValueError:
                continue
            ev = str(d.get("event", ""))
            if ev == "ECO_OPTOUT":
                if d.get("ok") is False:
                    hits.append(l[:200])
                continue
            if any(ev.startswith(k) for k in KEYS) or d.get("rc") not in (None, 0):
                hits.append(l[:200])
        if hits:
            done(reason="SCHED", events=hits)
        el, eo = new_lines(ECO, eo)
        bad = [l[:200] for l in el if '"ok": false' in l]
        if bad:
            done(reason="ECO_OK_FALSE", lines=bad)
        if size(TWARN) > w0:
            done(reason="THERMAL_WARNING", tail=TWARN.read_text(encoding="utf-8").splitlines()[-3:])
        try:
            rows = TLOG.read_bytes()[-800:].decode("utf-8", "replace").strip().splitlines()[-2:]
            last, prev = rows[-1].split(","), rows[-2].split(",")
            g, ram, ram_prev = float(last[1]), float(last[-1]), float(prev[-1])
            if g >= 80 or (ram < 2.0 and ram_prev < 2.0):
                done(reason="THERMAL_OR_RAM", gpu_c=g, ram_gib=ram, ram_prev_gib=ram_prev, row=last[0])
        except (OSError, ValueError, IndexError):
            pass
        cl = procs()
        if not any("laptop_sched_v2s4.py" in c for c in cl):
            done(reason="SCHEDULER_GONE")
        if not any("eco_watch.py" in c for c in cl):
            done(reason="ECO_WATCH_GONE")
        s = SLOTS.read_text(encoding="utf-8").strip()
        if s != "2" and not any(k in c for c in cl for k in ("p19_smoke_window", "p19_gpu_smoke", "p20_smoke_window", "p20_gpu_smoke")):
            done(reason="SLOTS_NOT_2_NO_SMOKE", slots=s)
        if time.time() - t0 > 3 * 3600:
            done(reason="HEARTBEAT_3H")
        time.sleep(60)


if __name__ == "__main__":
    main()
