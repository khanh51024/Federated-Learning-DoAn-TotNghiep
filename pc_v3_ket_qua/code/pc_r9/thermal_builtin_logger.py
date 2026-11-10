"""Built-in thermal / CPU logger for the PC (user 2026-11-01 via TU VAN PC: no HWiNFO, only built-in tools, no admin).
Separate tool: NOT pinned by any driver, does NOT touch MonitorV2 and never pauses / stops a run (auto pause / stop stays the
MonitorV2 GPU rule 85 / 90 C). Every 30 s it appends one row to experiments/train-improve-20261015/thermal/builtin_log.csv:
  nvidia-smi: GPU temperature, power, SM clock / max, util, memory.used, clocks_event_reasons hw / sw thermal slowdown, hw slowdown;
  typeperf:   \\Thermal Zone Information(*)\\Temperature (ACPI zones, Kelvin -> C; NOT a CPU core temperature),
              \\Processor Information(_Total)\\% Processor Performance, Processor Frequency, % Processor Utility.
WARN only (appended to thermal/builtin_warnings.jsonl, read by the coordinator at each ~30 min check, who tells the user):
  W_CPU_LIMITED: % Processor Utility > 50 AND % Processor Performance < 100 for >= 5 min continuously (10 samples);
  W_GPU_THERMAL: any GPU thermal slowdown reason Active.
Stops when thermal/STOP_LOGGER exists. One instance (thermal/LOGGER.pid holds the live pid).
Usage: python -I -B thermal_builtin_logger.py
"""
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

D = Path(r"experiments\train-improve-20261015\thermal")
LOG, WARN, STOP, PID = D / "builtin_log.csv", D / "builtin_warnings.jsonl", D / "STOP_LOGGER", D / "LOGGER.pid"
PERIOD_S = 30
CPU_WARN_N = 10
COUNTERS = [r"\Thermal Zone Information(*)\Temperature", r"\Processor Information(_Total)\% Processor Performance",
            r"\Processor Information(_Total)\Processor Frequency", r"\Processor Information(_Total)\% Processor Utility"]
GPU_Q = ("temperature.gpu,power.draw,clocks.sm,clocks.max.sm,utilization.gpu,memory.used,"
         "clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_slowdown")
GPU_COLS = ["gpu_temp_c", "gpu_power_w", "gpu_sm_mhz", "gpu_sm_max_mhz", "gpu_util", "gpu_mem_mib", "gpu_hw_thermal_slowdown",
            "gpu_sw_thermal_slowdown", "gpu_hw_slowdown"]


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=30).stdout.strip().split(",")
        return [x.strip() for x in o] if len(o) == len(GPU_COLS) else ["unknown"] * len(GPU_COLS)
    except Exception:
        return ["unknown"] * len(GPU_COLS)


def perf():
    """-> (dict of thermal zones C, perf %, freq MHz, utility %); 'unknown' where unreadable."""
    try:
        o = subprocess.run(["typeperf"] + COUNTERS + ["-sc", "1"], capture_output=True, text=True, timeout=60).stdout
        rows = list(csv.reader([l for l in o.splitlines() if l.startswith('"')]))
        head, val = rows[0], rows[1]
    except Exception:
        return {}, "unknown", "unknown", "unknown"
    zones, p, f, u = {}, "unknown", "unknown", "unknown"
    for h, v in zip(head[1:], val[1:]):
        try:
            x = float(v)
        except ValueError:
            continue
        if "Thermal Zone" in h:
            zones[h.split("(")[1].split(")")[0]] = round(x - 273.15, 1)
        elif h.endswith("% Processor Performance"):
            p = round(x, 1)
        elif h.endswith("Processor Frequency"):
            f = round(x)
        elif h.endswith("% Processor Utility"):
            u = round(x, 1)
    return zones, p, f, u


def warn(kind, **kw):
    with open(WARN, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "warn": kind, **kw}, ensure_ascii=False) + "\n")


def main():
    D.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            old = int(PID.read_text(encoding="utf-8").strip())
            assert not (psutil.pid_exists(old) and "python" in psutil.Process(old).name().lower()), ("logger already running", old)
        except ValueError:
            pass
    PID.write_text(str(os.getpid()), encoding="utf-8")
    new = not LOG.exists()
    cpu_run = 0
    gpu_warned = False
    with open(LOG, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["t"] + GPU_COLS + ["cpu_perf_pct", "cpu_freq_mhz", "cpu_utility_pct", "acpi_zones_c", "ram_avail_gib"])
            fh.flush()
        while not STOP.exists():
            t0 = time.time()
            g = gpu()
            zones, p, f, u = perf()
            ram = round(psutil.virtual_memory().available / 2**30, 2)
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S")] + g + [p, f, u, json.dumps(zones), ram])
            fh.flush()
            limited = isinstance(u, float) and isinstance(p, float) and u > 50 and p < 100
            cpu_run = cpu_run + 1 if limited else 0
            if cpu_run == CPU_WARN_N:
                warn("W_CPU_LIMITED", minutes=CPU_WARN_N * PERIOD_S / 60, cpu_utility_pct=u, cpu_perf_pct=p, acpi_zones_c=zones)
            hot = "Active" in (g[6], g[7])
            if hot and not gpu_warned:
                warn("W_GPU_THERMAL", gpu_temp_c=g[0], hw_thermal=g[6], sw_thermal=g[7])
            gpu_warned = hot
            time.sleep(max(1.0, PERIOD_S - (time.time() - t0)))
    # the pid file is left in place (no deletes); a stale pid is detected at the next start


if __name__ == "__main__":
    main()
