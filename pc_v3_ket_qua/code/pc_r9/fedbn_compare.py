"""FedBN comparator (PHASE_C_SPEC_PC r2 section 3 c + section 5.1). CPU, read-only. Re-implements NOTHING: it imports the pinned q18
runner (fedavg_mixed_runner: CanonicalClassificationDataset, get_transforms, eval_collate_fn, model_from_global_state, evaluate_metrics,
part_root) and q8r2_methods.fedbn_round_eval / class_share. For a FedAvg run, the FedAvg CDR is fedbn_round_eval called with an EMPTY
local BN for every client (fedbn_client_start(global, {}) == global), i.e. the same function the FedBN runs use at round 40.
Returns the re-evaluated val metrics (for the G_CDR equality to history round 40), per-client n_k and sum p_k, and the CDR.
Usage (library): compute(run_dir, scheme) ; CLI: python -I -B fedbn_compare.py <run_dir> <scheme>
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
WS = Path(r"<WORKSPACE_ROOT>")
SBX = WS / "experiments/train-improve-20261015/q18/sandbox"
PINS = {"fedavg_mixed_runner.py": None, "q8r2_methods.py": None}   # filled from the Q18 stage lock below
LOCK = (WS / "experiments/train-improve-20261015/q18/Q18_STAGE_LOCK.json", "3a5bfdfecf623e52a0173169ddb80f295de1635114d70f0f6d818fbf92091e27")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_runner():
    assert sha(LOCK[0]) == LOCK[1], "Q18 lock sha"
    files = json.loads(LOCK[0].read_text(encoding="utf-8"))["files"]
    for k in PINS:
        key = next(f for f in files if f.endswith(k))
        assert sha(SBX.parent / key) == files[key], ("q18 sandbox file sha", k)
        PINS[k] = files[key]
    sys.path.insert(0, str(SBX))
    os.chdir(SBX)
    s = importlib.util.spec_from_file_location("fedavg_mixed_runner_q18", SBX / "fedavg_mixed_runner.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    import q8r2_methods
    return m, q8r2_methods


def compute(run_dir, scheme, m=None, q=None):
    import torch
    from torch.utils.data import DataLoader
    if m is None:
        m, q = load_runner()
    torch.set_num_threads(4)
    run_dir = Path(run_dir)
    release_dir = WS / "dataset/mixed/pv_pd_v3"
    dataset_root = WS / "dataset"
    tf = m.get_transforms("canonical_v1", is_train=False)
    val_ds = m.CanonicalClassificationDataset.from_manifest(release_dir / "manifests/val.jsonl", dataset_root=dataset_root,
                                                            transform=tf, return_metadata=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=m.eval_collate_fn)
    ck = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    gstate = ck["model_state"]
    dev = torch.device("cpu")
    res = m.evaluate_metrics(m.model_from_global_state(gstate), val_loader, dev)
    pdir = Path(release_dir) / f"partitions/{scheme}"
    client_ids = [f"client_{i:02d}" for i in range(5)]
    loaders, n_k, sum_p = [], {}, {}
    for cid in client_ids:
        cds = m.CanonicalClassificationDataset.from_manifest(pdir / f"{cid}.jsonl", dataset_root=dataset_root, transform=None,
                                                             return_metadata=True)
        loaders.append(DataLoader(cds, batch_size=32))
        n_k[cid] = len(cds)
        sum_p[cid] = sum(q.class_share(cds.targets, 38))
    cdr = q.fedbn_round_eval(gstate, {cid: {} for cid in client_ids}, client_ids, loaders, m.model_from_global_state,
                             m.evaluate_metrics, val_loader, dev, None)
    per = cdr["fedbn_endpoint_b_client_dist_recall"]
    return {"run_dir": str(run_dir), "scheme": scheme, "checkpoint_last_sha256": sha(run_dir / "checkpoint_last.pt"),
            "reeval_val_macro_f1_raw": res["raw_macro_f1"],
            "reeval_plantdoc_accuracy": (res.get("per_source", {}).get("plantdoc") or {}).get("raw_accuracy"),
            "n_k": n_k, "sum_p_k": sum_p, "cdr_per_client": per, "cdr_mean": sum(per.values()) / len(per),
            "uniform_mean_raw_macro_f1": cdr["fedbn_endpoint_a_uniform_mean_raw_macro_f1"],
            "uniform_mean_plantdoc_accuracy": cdr["fedbn_endpoint_a_uniform_mean_plantdoc_accuracy"],
            "p_mass_without_val_support": {r["client"]: r["p_mass_without_val_support"] for r in cdr["fedbn_eval"]},
            "pins": dict(PINS)}


if __name__ == "__main__":
    print(json.dumps(compute(sys.argv[1], sys.argv[2]), ensure_ascii=False, default=str))
