"""Run ONE CPU evaluation command under the TU VAN resource rule (2026-11-02): start only when RAM available >= 3.0 GiB (waits, max 2 h),
log RAM every 30 s to <log_csv>, kill the whole process tree if available < 2.0 GiB (rc 99 = RAM_ABORT). BelowNormal priority.
Usage: python -I -B ram_guard_launcher.py <log_csv> <cwd> -- <cmd ...>   (extra env: KEY=VAL pairs before "--" after cwd)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

START_GIB, STOP_GIB, PERIOD = 3.0, 2.0, 30


def avail():
    return psutil.virtual_memory().available / 2**30


def main(argv):
    i = argv.index("--")
    log, cwd, envs, cmd = Path(argv[0]), argv[1], argv[2:i], argv[i + 1:]
    assert not log.exists(), log
    env = dict(os.environ)
    for kv in envs:
        k, v = kv.split("=", 1)
        env[k] = v
    t0 = time.time()
    while avail() < START_GIB:
        assert time.time() - t0 < 7200, "RAM never reached 3 GiB in 2 h"
        time.sleep(PERIOD)
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("t,ram_avail_gib,proc_rss_gib\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=env, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        ps = psutil.Process(p.pid)
        abort = False
        while p.poll() is None:
            a = avail()
            try:
                rss = sum(q.memory_info().rss for q in [ps] + ps.children(recursive=True))
            except psutil.Error:
                rss = 0
            fh.write(f"{time.strftime('%H:%M:%S')},{a:.2f},{rss / 2**30:.2f}\n")
            fh.flush()
            if a < STOP_GIB:
                abort = True
                for q in reversed([ps] + ps.children(recursive=True)):
                    try:
                        q.kill()
                    except psutil.Error:
                        pass
                break
            for _ in range(PERIOD):
                if p.poll() is not None:
                    break
                time.sleep(1)
        p.wait()
    rc = 99 if abort else p.returncode
    print(json.dumps({"rc": rc, "ram_abort": abort, "wall_s": round(time.time() - t0, 1)}))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
