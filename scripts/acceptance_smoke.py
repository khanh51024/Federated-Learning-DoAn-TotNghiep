"""Bounded acceptance run on real images; no full-data training or image copying.

Run from the package root with its .venv Python. Outputs are isolated under docs.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "docs" / "acceptance_20260909" / f"integration_{time.strftime('%H%M%S')}"
    output.mkdir(parents=True)
    bundle = output / "bundle"
    (bundle / "clients").mkdir(parents=True)
    source = ROOT / "data/partitions_train_v1/smoke_bundle_2c"

    def read(path):
        with path.open(encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return reader.fieldnames, list(reader)

    def write(path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    pooled = []
    for cid, size in enumerate((16, 32)):
        fields, rows = read(source / "clients" / f"client_{cid:02d}.csv")
        rows = rows[:size]
        pooled.extend(rows)
        write(bundle / "clients" / f"client_{cid:02d}.csv", fields, rows)
    write(bundle / "centralized_train.csv", fields, pooled)
    index = json.loads((ROOT / "data/partitions_train_v1/index.json").read_text())
    reference = ROOT / next(iter(index["partitions"].values()))["relative_dir"]
    for split in ("global_val.csv", "global_test.csv"):
        fields, rows = read(reference / split)
        selected = {}
        for row in rows:
            selected.setdefault(row["label"], row)
        write(bundle / split, fields, list(selected.values()))
    source_meta = json.loads((reference / "fedavg_meta.json").read_text())
    (bundle / "fedavg_meta.json").write_text(json.dumps({
        "schema_version": 3, "smoke": True, "num_clients": 2,
        "class_names": source_meta["class_names"], "total_train_samples": 48,
        "client_sample_counts": {"client_00": 16, "client_01": 32},
        "client_aggregation_weights": {"client_00": 1/3, "client_01": 2/3},
    }, indent=2))

    raw = yaml.safe_load((ROOT / "configs/train_smoke.yaml").read_text())
    raw["data"]["bundle_path"] = str(bundle)
    raw["output"].update(root=str(output / "fedavg"), evaluate_test_after_train=True, progress=False)
    config = output / "train.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sweep = output / "sweep.yaml"
    sweep.write_text(yaml.safe_dump({
        "base_config": str(config), "output_root": str(output), "sweep_id": "comparison",
        "modes": ["centralized", "local-only", "fedavg"], "seeds": [42],
        "conditions": [{"scenario": "label_skew", "alpha": .1, "feature_skew": "none", "split_seed": 42}],
        "run_timeout_seconds": 180,
    }), encoding="utf-8")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(label, *args):
        start = time.monotonic()
        with (output / f"{label}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "fl_training.cli", *args], cwd=ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                raise
        record = {"label": label, "exit_code": code, "seconds": round(time.monotonic() - start, 2)}
        commands.append(record)
        (output / "commands.json").write_text(json.dumps(commands, indent=2))
        print(record, flush=True)
        if code:
            raise RuntimeError(f"{label} failed; see {output / (label + '.log')}")

    print(f"OUTPUT={output}", flush=True)
    run("sweep", "sweep", "--config", str(sweep), "--execute")
    comparison = json.loads((output / "comparison/comparison.json").read_text())
    continuous = Path(next(row["run_dir"] for row in comparison if row["mode"] == "fedavg"))
    raw["federation"]["max_rounds"] = 1
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run("first_round", "train", "--config", str(config), "--no-progress")
    first = next((output / "fedavg").iterdir())
    raw["federation"]["max_rounds"] = 2
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run("resume", "train", "--config", str(config), "--resume", str(first / "last.pt"), "--no-progress")
    resumed = max((output / "fedavg").iterdir(), key=lambda p: p.stat().st_mtime_ns)
    import torch
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    left, right = load(continuous / "last.pt"), load(resumed / "last.pt")
    maximum = max(float((v.double() - right["model_state_dict"][k].double()).abs().max())
                  for k, v in left["model_state_dict"].items())
    assert maximum == 0, maximum
    weights = [json.loads(line) for line in (continuous / "weights.jsonl").read_text().splitlines()]
    for row in weights:
        assert abs(row["aggregation_weight"] - ({0: 1/3, 1: 2/3}[row["client_id"]])) < 1e-12
    final = load(resumed / "model_final.pt")
    best = load(resumed / "best.pt")
    assert all(torch.equal(v, best["model_state_dict"][k]) for k, v in final["model_state_dict"].items())
    run("inference_evaluate", "evaluate", "--checkpoint", str(resumed / "model_final.pt"),
        "--output", str(output / "inference_eval"))
    if torch.cuda.is_available():
        raw["runtime"]["client_device"] = "cuda"
        raw["training"]["amp"] = True
        raw["output"]["root"] = str(output / "cuda/fedavg")
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        run("cuda_fedavg", "train", "--config", str(config), "--no-progress")
        run("cuda_baseline", "baseline", "--config", str(config), "--mode", "centralized")
    result = {"tensor_entries": len(left["model_state_dict"]), "resume_max_abs_diff": maximum,
              "all_model_final_weights_equal_best": True, "client_samples": [16, 32],
              "aggregation_weights": [1/3, 2/3], "validation_samples": 38, "test_samples": 38,
              "continuous_run": str(continuous), "resumed_run": str(resumed),
              "scientific_stage2_complete": False}
    (output / "acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
