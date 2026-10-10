"""Reuse the exact GĐ1 FedAvg engine; vary data allocation, not the trainer."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from stage1_compat.artifacts import validate_completed
from stage1_compat.budget import BudgetLedger, BudgetExhausted
from stage1_compat.config import JobConfig
from stage1_compat.integrity import atomic_json, digest, file_hash
from stage1_compat.runner import run_single_fedavg_job
from .data import ROOT, PROTOCOL, CONDITIONS, ManifestDataset, preflight, read_rows


def source_hash():
    # Includes the pinned upstream trainer as well as the adapter/partition code.
    files = [p for folder in ("stage2_matched", "stage1_compat", "src/data", "fl_training")
             for p in (ROOT / folder).rglob("*.py")]
    return digest({p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)})


def make_job(condition, seed, smoke=False):
    return JobConfig(job_id=f"{condition}_seed{seed}", seed=seed,
                     alpha=CONDITIONS[condition].get("alpha", 0.1),
                     rounds=2 if smoke else 10, pretrained=not smoke)


def trainer_contract(job):
    # Alpha/condition and random seed are experiment axes; all other settings fixed.
    return {k: v for k, v in job.to_dict().items() if k not in ("job_id", "alpha", "seed")}


def run(suite_dir, dataset_root, output, conditions=None, seeds=(42,), session_minutes=420, device="auto"):
    if not 2 <= session_minutes <= 450:
        raise ValueError("Session limit must be between 2 and 450 minutes")
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - CONDITIONS.keys():
        raise ValueError("Unknown/duplicate conditions")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    output = Path(output).resolve()
    target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    ledger = BudgetLedger(output, user_quota_hours=30)
    previous_deadline = os.environ.get("STAGE1_SOFT_DEADLINE")
    deadline = time.time() + session_minutes * 60
    if previous_deadline is not None:
        deadline = min(deadline, float(previous_deadline))
    os.environ["STAGE1_SOFT_DEADLINE"] = str(deadline)
    results = []
    started = time.monotonic()
    try:
        with ledger.get_lock():
            suite, audits = preflight(suite_dir, dataset_root)
            source = source_hash()
            protocol = {"protocol": PROTOCOL, "suite_sha256": digest(suite), "source_sha256": source,
                        "smoke": suite["smoke"], "common": suite["common"],
                        "trainer": trainer_contract(make_job("label100", 42, suite["smoke"]))}
            frozen = output / "matched_protocol.json"
            if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != protocol:
                raise ValueError("Output belongs to different data/code/trainer; choose a new output directory")
            atomic_json(frozen, protocol)
            atomic_json(output / "preflight.json", audits)
            for seed in seeds:
                for name in conditions:
                    path = Path(suite_dir) / name
                    meta = json.loads((path / "partition_config.json").read_text(encoding="utf-8"))
                    rows = read_rows(path / "centralized_train.csv")
                    train = ManifestDataset(rows, dataset_root, True, meta["domain_profiles"])
                    val = ManifestDataset(read_rows(path / "global_val.csv"), dataset_root)
                    test = ManifestDataset(read_rows(path / "global_test.csv"), dataset_root)
                    partitions = [[i for i, r in enumerate(rows) if r["client_id"] == cid] for cid in range(5)]
                    train.identity_context = {
                        "scope": PROTOCOL, "smoke": suite["smoke"], "condition": name,
                        "matched_source_sha256": source, "matched_suite_sha256": digest(suite),
                        "matched_manifest_sha256": suite["conditions"][name]["manifest_sha256"],
                        "common": suite["common"], "domain_kind": meta["domain_kind"],
                        "domain_assignment_sha256": meta["domain_assignment_sha256"],
                    }
                    job = make_job(name, seed, suite["smoke"])
                    result = run_single_fedavg_job(job, train, val, test, suite["class_names"],
                                                  partitions, output, ledger, target)
                    results.append({"job_id": job.job_id, "status": result["status"]})
                    if result["status"] != "COMPLETED":
                        return results
    except BudgetExhausted:
        results.append({"status": "PAUSED_DEADLINE", "resume": "Rerun the identical command/output"})
    finally:
        if previous_deadline is None:
            os.environ.pop("STAGE1_SOFT_DEADLINE", None)
        else:
            os.environ["STAGE1_SOFT_DEADLINE"] = previous_deadline
        # This is a bounded session, not an account-wide Kaggle quota monitor.
        atomic_json(output / f"session_{time.time_ns()}.json", {
            "elapsed_wall_seconds": time.monotonic() - started,
            "session_minutes": session_minutes, "quota_scope": "session_only", "jobs": results})
    return results


def comparison_key(metrics):
    identity, config = metrics["identity"], metrics["resolved_config"]
    ctx = identity["context"]
    if ctx.get("scope") != PROTOCOL or ctx.get("smoke") is not False:
        raise ValueError("Historical/smoke results are not scientific comparison inputs")
    name = ctx["condition"]
    expected = make_job(name, identity["seed"])
    if config != expected.to_dict():
        raise ValueError("Result does not match frozen stage-1 trainer settings")
    return {"seed": identity["seed"], "common": ctx["common"],
            "source": ctx["matched_source_sha256"], "suite": ctx["matched_suite_sha256"],
            "initialization": ctx["initialization_sha256"], "runtime": identity["runtime"],
            "train_device": ctx["train_device"], "evaluation_device": ctx["evaluation_device"],
            "trainer": trainer_contract(expected)}


def paired_delta(reference, target):
    if comparison_key(reference) != comparison_key(target):
        raise ValueError("Cannot compare different data, seed, initialization, runtime or trainer")
    return {k + "_delta_pp": 100 * (target["test_metrics"][k] - reference["test_metrics"][k])
            for k in ("accuracy", "macro_f1")}


def collect(output, suite_dir):
    output = Path(output)
    suite, _ = preflight(suite_dir)
    protocol = json.loads((output / "matched_protocol.json").read_text())
    if protocol["suite_sha256"] != digest(suite) or protocol["source_sha256"] != source_hash():
        raise ValueError("Collector source/suite differs from frozen run")
    results = {}
    for file in sorted(output.glob("*/fedavg_metrics.json")):
        result = validate_completed(file.parent)
        ctx = result["identity"]["context"]
        name, seed = ctx["condition"], result["seed"]
        if (ctx.get("scope") != PROTOCOL or ctx["smoke"] != suite["smoke"]
                or ctx["matched_suite_sha256"] != digest(suite)
                or ctx["matched_source_sha256"] != source_hash()
                or ctx["common"] != suite["common"]
                or ctx["matched_manifest_sha256"] != suite["conditions"][name]["manifest_sha256"]):
            raise ValueError("Result is not from this matched suite")
        if result["resolved_config"] != make_job(name, seed, suite["smoke"]).to_dict():
            raise ValueError("Unexpected trainer configuration")
        if (name, seed) in results:
            raise ValueError("Duplicate condition/seed result")
        results[name, seed] = result
    pairs, missing = [], []
    # Feature alpha100 -> alpha0.1 controls for the synthetic domain transform.
    comparisons = [(ref, target) for ref in ("label100", "label1") for target in ("label01", "label_quantity01")]
    comparisons += [("quantity100", "quantity01"), ("feature100", "feature01")]
    seeds = sorted({s for _, s in results})
    if not suite["smoke"]:
        for seed in seeds:
            for ref, target in comparisons:
                if (ref, seed) not in results or (target, seed) not in results:
                    missing.append({"seed": seed, "reference": ref, "target": target})
                    continue
                pairs.append({"seed": seed, "reference": ref, "target": target,
                              **paired_delta(results[ref, seed], results[target, seed])})
    baselines = {}
    for cent_file in sorted(output.glob("centralized_seed*/centralized_metrics.json")):
        cent_data = json.loads(cent_file.read_text(encoding="utf-8"))
        baselines[f"centralized_seed{cent_data['seed']}"] = cent_data
    for loc_file in sorted(output.glob("local_only_*_seed*/local_only_metrics.json")):
        loc_data = json.loads(loc_file.read_text(encoding="utf-8"))
        baselines[f"local_only_{loc_data.get('condition', 'unknown')}_seed{loc_data['seed']}"] = loc_data

    scientific_complete = bool(baselines and len(results) > 0 and not suite["smoke"])
    report = {"protocol": PROTOCOL, "smoke": suite["smoke"],
              "historical_stage1": "reference_only; old split must not enter paired deltas",
              "scientific_stage2_complete": scientific_complete,
              "remaining_stage2_requirements": [] if scientific_complete else ["matched centralized/local-only controls", "per-client fairness analysis"],
              "completed_jobs": [{"condition": n, "seed": s, "test": r["test_metrics"],
                                  "best_round": r["best_round"]} for (n, s), r in results.items()],
              "baselines": baselines,
              "paired_deltas": pairs, "missing_pairs": missing,
              "scope": "FedAvg bridge; smoke only checks execution" if suite["smoke"] else "Matched FedAvg comparison",
              "accuracy_policy": "Measured values are retained, including 100%; no clipping or target accuracy"}
    atomic_json(output / "matched_comparison.json", report)
    return report


def run_baseline(suite_dir, dataset_root, output, mode="centralized", conditions=None, seeds=(42,), device="auto"):
    """
    Huấn luyện baseline Centralized (tập trung) hoặc Local-only (cục bộ) trên split sạch v5.
    Dùng kiến trúc MobileNetV3-Small, AdamW(lr=1e-3, weight_decay=1e-4), batch_size=32, 10 epochs.
    """
    import copy
    import numpy as np
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from stage1_compat.models import create_mobilenetv3_stage1
    from stage1_compat.upstream_loader import get_upstream_evaluate

    suite_dir = Path(suite_dir)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    eval_mod = get_upstream_evaluate()
    criterion = nn.CrossEntropyLoss()

    suite, _ = preflight(suite_dir, dataset_root)
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if mode in ("centralized", "both"):
            cond_name = conditions[0]
            path = suite_dir / cond_name
            train_rows = read_rows(path / "centralized_train.csv")
            val_rows = read_rows(path / "global_val.csv")
            test_rows = read_rows(path / "global_test.csv")

            train_ds = ManifestDataset(train_rows, dataset_root, True)
            val_ds = ManifestDataset(val_rows, dataset_root, False)
            test_ds = ManifestDataset(test_rows, dataset_root, False)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

            model = create_mobilenetv3_stage1(len(suite["class_names"]), pretrained=not suite["smoke"]).to(target)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            best_acc, best_epoch, best_state = -1.0, 0, None
            history = []
            epochs = 2 if suite["smoke"] else 10
            print(f"[Centralized seed={seed}] Training {epochs} epochs on {len(train_ds)} samples...")

            for ep in range(1, epochs + 1):
                train_loss = eval_mod.train_one_epoch(model, train_loader, optimizer, criterion, target, f"Centralized Ep {ep}/{epochs}")
                val_metrics = eval_mod.evaluate(model, val_loader, criterion, target)
                history.append({
                    "epoch": ep, "train_loss": train_loss,
                    "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                })
                print(f"[Centralized Ep {ep}/{epochs}] val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['macro_f1']:.4f}")
                if val_metrics["accuracy"] > best_acc:
                    best_acc, best_epoch = val_metrics["accuracy"], ep
                    best_state = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            test_metrics = eval_mod.evaluate(model, test_loader, criterion, target)
            res = {
                "experiment": "centralized", "seed": seed, "epochs": epochs,
                "best_epoch": best_epoch, "best_validation_accuracy": best_acc,
                "test_metrics": test_metrics, "history": history,
                "samples": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
            }
            cent_dir = output / f"centralized_seed{seed}"
            cent_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(cent_dir / "centralized_metrics.json", res)
            results.append(res)
            print(f"[Centralized seed={seed}] Test Acc: {test_metrics['accuracy']:.4f}, Test Macro-F1: {test_metrics['macro_f1']:.4f}")

        if mode in ("local-only", "both"):
            for cond_name in conditions:
                path = suite_dir / cond_name
                meta = json.loads((path / "partition_config.json").read_text(encoding="utf-8"))
                val_ds = ManifestDataset(read_rows(path / "global_val.csv"), dataset_root, False)
                test_ds = ManifestDataset(read_rows(path / "global_test.csv"), dataset_root, False)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

                client_metrics = []
                epochs = 2 if suite["smoke"] else 10
                print(f"[Local-only {cond_name} seed={seed}] Training 5 clients, {epochs} epochs/client...")

                for cid in range(5):
                    c_rows = read_rows(path / "clients" / f"client_{cid:02d}.csv")
                    c_train_ds = ManifestDataset(c_rows, dataset_root, True, meta.get("domain_profiles"))
                    c_train_loader = DataLoader(c_train_ds, batch_size=32, shuffle=True, num_workers=0)

                    c_model = create_mobilenetv3_stage1(len(suite["class_names"]), pretrained=not suite["smoke"]).to(target)
                    c_opt = optim.AdamW(c_model.parameters(), lr=1e-3, weight_decay=1e-4)

                    c_best_acc, c_best_epoch, c_best_state = -1.0, 0, None
                    for ep in range(1, epochs + 1):
                        eval_mod.train_one_epoch(c_model, c_train_loader, c_opt, criterion, target)
                        val_m = eval_mod.evaluate(c_model, val_loader, criterion, target)
                        if val_m["accuracy"] > c_best_acc:
                            c_best_acc, c_best_epoch = val_m["accuracy"], ep
                            c_best_state = copy.deepcopy(c_model.state_dict())

                    c_model.load_state_dict(c_best_state)
                    c_test_m = eval_mod.evaluate(c_model, test_loader, criterion, target)
                    c_test_m["client_id"] = cid
                    c_test_m["train_samples"] = len(c_rows)
                    client_metrics.append(c_test_m)
                    print(f"  Client {cid}: val_acc={c_best_acc:.4f}, test_acc={c_test_m['accuracy']:.4f}")

                accs = [m["accuracy"] for m in client_metrics]
                f1s = [m["macro_f1"] for m in client_metrics]
                res = {
                    "experiment": "local_only", "condition": cond_name, "seed": seed, "epochs": epochs,
                    "average_accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                    "accuracy_min": float(np.min(accs)), "accuracy_max": float(np.max(accs)),
                    "average_macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                    "macro_f1_min": float(np.min(f1s)), "macro_f1_max": float(np.max(f1s)),
                    "clients": client_metrics,
                }
                loc_dir = output / f"local_only_{cond_name}_seed{seed}"
                loc_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(loc_dir / "local_only_metrics.json", res)
                results.append(res)
                print(f"[Local-only {cond_name} seed={seed}] Avg Acc: {res['average_accuracy']:.4f} (min={res['accuracy_min']:.4f}, max={res['accuracy_max']:.4f})")

    return results

