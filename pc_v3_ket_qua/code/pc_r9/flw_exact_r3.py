"""FLW flower_exact harness r3 (DIEU PHOI PC r3, ADDENDUM_FLW FLW-A; r1 7ef110a3 2026-11-09 + thread fix, needs TU VAN PASS of FLW_SPEC).
Goal: the SAME runner trajectory as the PC FedAvg runs (pc_fedavg_a05_s42-44 = throttle r3 + q15 runner), but every client's
local training executes inside a Flower ClientApp (flwr 1.42.0 Message API, flwr.simulation.run_simulation, Ray backend,
5 SuperNodes, num_gpus 1.0 per client -> clients train one after another on the one GPU) and every model transfer is a
Flower ArrayRecord.

Server process: p1lo_throttle_thermal_pc_r3.py (b5c34e05) is executed unchanged to install its DataLoader patches (num_workers
override from PC_NUM_WORKERS, collate swap, CUDA memory fraction, thermal flags); its final runpy.run_path(runner) is
intercepted, the runner is loaded as module "fedavg_mixed_runner" (as p22_probe) and its main() runs INSIDE the Flower
ServerApp. The module-level train_client is replaced by a dispatcher that sends one Flower Message per client (order client_00
.. client_04, one at a time) and returns what the ClientApp sent back. aggregate_weights, evaluation, checkpoints, history,
RESULT inputs: the runner's own code, unchanged.

What the dispatcher ships (and why): the runner keeps ONE global RNG stream across server and clients (DataLoader worker
base seeds, RandomSampler seeds, CUDA dropout), a per-client GradScaler and the client DataLoader object. To keep the
in-process trajectory, the message carries (a) the global state as ArrayRecord (the FL payload; bytes counted), (b) a pickled
control blob (harness-only, bytes counted separately, NOT part of the FL payload): the client DataLoader (dataset + sampler +
num_workers + collate, never iterated on the server), the GradScaler state, the torch CPU / CUDA / numpy / python RNG states
and the remaining train_client kwargs (progress_callback dropped: progress.json only). The reply carries the client weights
(ArrayRecord), the telemetry, the updated RNG + scaler states and the sampler draw statistics; the server restores them.
This RNG hand-off is a simulation device for the equivalence gate; a real deployment would seed clients independently.
LABEL (FLW_SPEC r2, verbatim): flower_exact = equivalence gate of orchestration / transport. The client DataLoader (dataset +
sampler), the RNG and the GradScaler are created by the harness on the server side and passed in the control blob. The client
does not hold its own shard / independent seed. Only the ArrayRecord is the FL payload. harness_ctl bytes never enter the
communication-cost table.

Client process (Ray actor): same runner module (same file, same name), cudnn.deterministic True / benchmark False (= the
runner's set_seed flags), CUDA memory fraction = Q5_GPU_MEM_FRACTION (= throttle), then the runner's own train_client.

Bytes: payload_*_bytes = bytes of the Array.data buffers Flower serialises (npy header included); tensor_bytes_* = raw
numel x element_size. Logs (create-once, under <out_root>/flw/): FLW_EVENTS.jsonl (N1.3 schema: client_round_execution / server_round_aggregation
+ transport_mode "simulation"), FLW_SUMMARY.json. Runner outputs go to the runner --output-dir as usual.
r3 (2026-11-09, TU VAN; after G_SMOKE_CPU att1 FAIL de931a98 and diag THREADS_flower cdbdebab): the Ray actor ran torch with
2 intra-op threads and OMP_NUM_THREADS=2 (Ray sets it from num_cpus) while the server / ref process runs 10 threads with
OMP_NUM_THREADS unset -> different float reduction order on CPU. Fix of the measurement setup only: the server ships its
thread config (torch num_threads / num_interop_threads, OMP_NUM_THREADS, MKL_NUM_THREADS) in the ctl blob, the client applies
it before train_client (env restored too, so spawned DataLoader workers see the same env) and the applied values are logged
in FLW_EVENTS (client_threads). No rule, threshold, payload or runner change.
Usage (cwd = WS): <venv_flwr python> -B flw_exact.py <out_root> -- <runner.py> <runner args...>
"""
import hashlib
import importlib.util
import json
import os
import pickle
import random
import runpy
import sys
import time
from pathlib import Path

PC_R9 = Path(__file__).resolve().parent
THROTTLE = PC_R9 / "p1lo_throttle_thermal_pc_r3.py"
THROTTLE_SHA = "b5c34e05de7bfbcd1792ad8dc69ede8aa673a7df9e348457b218f4c492c27534"
N_CLIENTS = 5


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ shared helpers (server + client processes)
def capture_rng():
    import numpy as np
    import torch
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(s):
    import numpy as np
    import torch
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def load_runner(runner_path):
    if "fedavg_mixed_runner" in sys.modules:
        return sys.modules["fedavg_mixed_runner"]
    runner_path = Path(runner_path)
    if str(runner_path.parent) not in sys.path:
        sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location("fedavg_mixed_runner", runner_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fedavg_mixed_runner"] = m
    spec.loader.exec_module(m)
    return m


SAMPLER_ATTRS = ("epoch", "last_statistics", "_last_indices")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def capture_threads():
    import torch
    d = {"num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads()}
    d.update({k: os.environ.get(k) for k in THREAD_ENV})
    return d


def apply_threads(t):
    """Make this (client) process use the server / ref thread config; returns the config in force afterwards."""
    import torch
    for k in THREAD_ENV:
        if t[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = t[k]
    if torch.get_num_threads() != t["num_threads"]:
        torch.set_num_threads(t["num_threads"])
    if torch.get_num_interop_threads() != t["num_interop_threads"]:
        torch.set_num_interop_threads(t["num_interop_threads"])   # raises if too late: no silent mismatch
    now = capture_threads()
    assert now == t, (now, t)
    return now


# ------------------------------------------------------------------ client side
def client_setup(runner_path):
    import torch
    if str(PC_R9) not in sys.path:
        sys.path.append(str(PC_R9))          # pc_r9_collate (swapped collate) for unpickling + spawn workers
    R = load_runner(runner_path)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not getattr(client_setup, "_capped", False):
            torch.cuda.set_per_process_memory_fraction(float(os.environ.get("Q5_GPU_MEM_FRACTION", "0.5")), 0)
            client_setup._capped = True
    return R


def build_client_app():
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp

    app = ClientApp()

    @app.query()
    def query(msg: Message, context: Context) -> Message:
        return Message(content=RecordDict({"m": MetricRecord({"partition-id": int(context.node_config["partition-id"])})}),
                       reply_to=msg)

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        import torch
        from torch.amp import GradScaler
        t0 = time.perf_counter_ns()
        pkg = pickle.loads(msg.content["ctl"]["blob"])
        R = client_setup(pkg["runner"])
        threads_before = capture_threads()
        threads_applied = apply_threads(pkg["threads"])
        assert int(context.node_config["partition-id"]) == pkg["k"], (context.node_config, pkg["k"])
        g = msg.content["arrays"].to_torch_state_dict()
        restore_rng(pkg["rng"])
        scaler = GradScaler("cuda", enabled=pkg["scaler_enabled"], init_scale=2048.0)
        if pkg["scaler_state"]:
            scaler.load_state_dict(pkg["scaler_state"])
        loader = pkg["loader"]
        t1 = time.perf_counter_ns()
        w, tel = R.train_client(client_id=pkg["client_id"], global_state=g, loader=loader, device=torch.device(pkg["device"]),
                                scaler=scaler, **pkg["kwargs"])
        t2 = time.perf_counter_ns()
        smp = loader.sampler
        ret = {"telemetry": tel, "rng": capture_rng(), "scaler_state": scaler.state_dict() if pkg["scaler_enabled"] else None,
               "sampler": {a: getattr(smp, a) for a in SAMPLER_ATTRS if hasattr(smp, a)},
               "threads_before": threads_before, "threads_applied": threads_applied,
               "client_train_ms": (t2 - t1) / 1e6, "client_setup_ms": (t1 - t0) / 1e6, "pid": os.getpid(),
               "max_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0}
        content = RecordDict({"arrays": ArrayRecord(w), "ctl": ConfigRecord({"blob": pickle.dumps(ret)}),
                              "metrics": MetricRecord({"partition-id": pkg["k"], "num-examples": int(pkg["kwargs"]["raw_sample_count"])})})
        return Message(content=content, reply_to=msg)

    return app


# ------------------------------------------------------------------ server side
class Dispatcher:
    def __init__(self, out_dir, runner_path):
        self.out, self.runner_path = Path(out_dir), str(runner_path)
        self.grid, self.node_of, self.calls, self.round_rows = None, {}, 0, []
        self.events = self.out / "FLW_EVENTS.jsonl"
        assert not self.events.exists(), self.events
        self.t_round = None

    def log(self, ev):
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def map_nodes(self):
        from flwr.app import Message, RecordDict
        t_end = time.time() + 300                    # SuperNodes register asynchronously
        ids = list(self.grid.get_node_ids())
        while len(ids) < N_CLIENTS and time.time() < t_end:
            time.sleep(1.0)
            ids = list(self.grid.get_node_ids())
        assert len(ids) == N_CLIENTS, ids
        msgs = [Message(content=RecordDict({}), dst_node_id=n, message_type="query") for n in ids]
        for rep in self.grid.send_and_receive(msgs):
            assert not rep.has_error(), rep.error
            self.node_of[int(rep.content["m"]["partition-id"])] = rep.metadata.src_node_id
        assert sorted(self.node_of) == list(range(N_CLIENTS)), self.node_of

    def train_client(self, client_id, global_state, loader, device, scaler=None, progress_callback=None, **kw):
        import torch
        from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
        k = int(client_id.split("_")[1])
        rnd = self.calls // N_CLIENTS + 1
        assert self.calls % N_CLIENTS == k, (self.calls, client_id)   # clients in order 00..04 every round
        if k == 0:
            self.t_round = time.perf_counter_ns()
        self.calls += 1
        sc_en = bool(scaler is not None and scaler.is_enabled())
        pkg = {"runner": self.runner_path, "k": k, "client_id": client_id, "device": str(device), "loader": loader,
               "scaler_enabled": sc_en, "scaler_state": scaler.state_dict() if sc_en else None, "kwargs": kw, "rng": capture_rng(),
               "threads": capture_threads()}
        blob = pickle.dumps(pkg)
        down = ArrayRecord({n: t.detach().cpu() for n, t in global_state.items()})
        down_bytes = sum(len(a.data) for a in down.values())
        msg = Message(content=RecordDict({"arrays": down, "ctl": ConfigRecord({"blob": blob})}), dst_node_id=self.node_of[k],
                      message_type="train")
        t0 = time.perf_counter_ns()
        reps = list(self.grid.send_and_receive([msg]))
        t1 = time.perf_counter_ns()
        assert len(reps) == 1 and not reps[0].has_error(), [r.error for r in reps if r.has_error()]
        rep = reps[0]
        up = rep.content["arrays"]
        up_bytes = sum(len(a.data) for a in up.values())
        w = up.to_torch_state_dict()
        ret = pickle.loads(rep.content["ctl"]["blob"])
        restore_rng(ret["rng"])
        if sc_en:
            scaler.load_state_dict(ret["scaler_state"])
        for a, v in ret["sampler"].items():
            setattr(loader.sampler, a, v)
        assert set(w) == set(global_state) and all(w[n].dtype == global_state[n].dtype and w[n].shape == global_state[n].shape
                                                     for n in w)
        fl = [n for n in w if w[n].is_floating_point()]
        layer = {n: float((w[n].double() - global_state[n].double()).norm()) for n in fl}
        self.log({"event": "client_round_execution", "transport_mode": "simulation", "round": rnd, "client_id": client_id,
                  "sample_count_n_k": kw.get("raw_sample_count"), "train_loss": ret["telemetry"].get("loss"),
                  "delta_l2": sum(v * v for v in layer.values()) ** 0.5, "layer_delta_norms": layer,
                  "client_compute_wall_ms": ret["client_train_ms"], "client_setup_ms": ret["client_setup_ms"],
                  "message_rtt_ms": (t1 - t0) / 1e6, "payload_down_bytes": down_bytes, "payload_up_bytes": up_bytes,
                  "tensor_bytes_down": sum(t.numel() * t.element_size() for t in global_state.values()),
                  "tensor_bytes_up": sum(t.numel() * t.element_size() for t in w.values()),
                  "harness_ctl_down_bytes": len(blob), "harness_ctl_up_bytes": len(rep.content["ctl"]["blob"]),
                  "optimizer_steps_applied": ret["telemetry"].get("optimizer_steps_applied"), "client_pid": ret["pid"],
                  "client_max_vram_mb": ret["max_vram_mb"],
                  "client_threads": {"server": pkg["threads"], "client_before": ret["threads_before"],
                                     "client_applied": ret["threads_applied"]}})
        if k == N_CLIENTS - 1:
            self.round_rows.append({"round": rnd, "clients_wall_ms": (time.perf_counter_ns() - self.t_round) / 1e6})
        return w, ret["telemetry"]


def server_main(out_root, runner_args):
    import torch
    out = Path(out_root) / "flw"
    out.mkdir(parents=True, exist_ok=False)
    assert sha_file(THROTTLE) == THROTTLE_SHA, "throttle r3 sha"
    captured = {}
    real = runpy.run_path

    def hook(path, run_name=None):
        captured["runner"] = path

    runpy.run_path = hook
    sys.argv = [str(THROTTLE)] + list(runner_args)
    try:
        real(str(THROTTLE), run_name="__main__")      # installs the throttle patches; its final run_path -> hook
    finally:
        runpy.run_path = real
    runner_path = Path(captured["runner"])
    R = load_runner(runner_path)
    D = Dispatcher(out, runner_path)
    orig_train = R.train_client
    R.train_client = D.train_client

    from flwr.app import Context
    from flwr.serverapp import Grid, ServerApp
    from flwr.simulation import run_simulation
    server_app = ServerApp()
    status = {}

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        D.grid = grid
        D.map_nodes()
        t0 = time.time()
        try:
            R.main()
            status["rc"] = 0
        except SystemExit as e:
            status["rc"] = e.code
        except BaseException as e:          # recorded, then re-raised after the summary
            status["rc"], status["error"] = "exception", repr(e)
        status["server_wall_s"] = time.time() - t0

    meta = {"throttle_sha256": THROTTLE_SHA, "harness_sha256": sha_file(__file__), "runner": str(runner_path),
            "runner_sha256": sha_file(runner_path), "orig_train_client": orig_train.__qualname__, "argv": sys.argv,
            "env": {k: os.environ.get(k) for k in ("PC_NUM_WORKERS", "Q5_GPU_DUTY", "Q5_GPU_MEM_FRACTION", "CUDA_VISIBLE_DEVICES",
                                                    "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "server_threads": capture_threads(), "harness_rev": "r3 (thread config handed to client)",
            "torch": torch.__version__, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    import flwr
    import ray
    meta.update(flwr=flwr.__version__, ray=ray.__version__)
    use_gpu = torch.cuda.is_available()
    t0 = time.time()
    run_simulation(server_app=server_app, client_app=build_client_app(), num_supernodes=N_CLIENTS,
                   backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 1.0 if use_gpu else 0.0},
                                   "init_args": {"num_cpus": 4, "num_gpus": 1 if use_gpu else 0, "include_dashboard": False}})
    meta.update(status=status, total_wall_s=time.time() - t0, dispatch_calls=D.calls, rounds=D.round_rows,
                end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    b = json.dumps(meta, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    (out / "FLW_SUMMARY.json").write_bytes(b)
    print(json.dumps({"status": status, "calls": D.calls, "wall_s": meta["total_wall_s"]}, default=str), flush=True)
    if status.get("rc") != 0:
        sys.exit(3)


if __name__ == "__main__":
    a = sys.argv[1:]
    assert len(a) >= 3 and a[1] == "--", "usage: flw_exact.py <out_root> -- <runner.py> <runner args...>"
    server_main(a[0], a[2:])
