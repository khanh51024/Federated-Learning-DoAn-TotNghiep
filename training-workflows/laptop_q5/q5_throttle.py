"""Q5 local: run the UNCHANGED Local-only runner with a GPU duty-cycle limit (laptop RTX 2050).
Every DataLoader iteration is wrapped: after the consumer finished one batch (cuda.synchronize), sleep
busy * (1 - DUTY) / DUTY. Only wall time changes: no tensor, RNG, sampler or batch order is touched.
Also caps the CUDA caching allocator at MEM_FRACTION of the card.
Usage: python -B q5_throttle.py <runner.py> <runner args...>      env Q5_GPU_DUTY (default 0.5)
"""
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

DUTY = float(os.environ.get("Q5_GPU_DUTY", "0.5"))
MEM_FRACTION = float(os.environ.get("Q5_GPU_MEM_FRACTION", "0.5"))
assert 0.1 <= DUTY <= 1.0 and 0.1 <= MEM_FRACTION <= 1.0, (DUTY, MEM_FRACTION)
_orig_iter = DataLoader.__iter__


def _throttled_iter(self):
    for batch in _orig_iter(self):
        t0 = time.perf_counter()
        yield batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        busy = time.perf_counter() - t0
        if DUTY < 1.0:
            time.sleep(busy * (1.0 - DUTY) / DUTY)


DataLoader.__iter__ = _throttled_iter
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, 0)
runner = Path(sys.argv[1]).resolve()
sys.argv = [str(runner)] + sys.argv[2:]
sys.path.insert(0, str(runner.parent))
print(f"[q5_throttle] duty={DUTY} mem_fraction={MEM_FRACTION} runner={runner.name}", flush=True)
runpy.run_path(str(runner), run_name="__main__")
