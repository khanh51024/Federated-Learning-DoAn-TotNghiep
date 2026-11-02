"""Chạy tập kiểm thử niêm phong ĐÚNG MỘT LẦN cho cấu hình đã chốt (N1.4, N5.3 của yêu cầu đề tài; lý do chọn tập ở docs/EXPERIMENT_PROTOCOL.md mục 9).

Tập niêm phong là split `calibration` của release v4: 7.005 ảnh chưa từng dùng để học, để chọn checkpoint hay chọn cấu hình, và chưa báo cáo độ chính xác nào.
Việc duy nhất đã dùng nó là fit nhiệt độ T cho xác suất hiển thị trên web; T không đổi dự đoán top-1 nên không ảnh hưởng số đo ở đây.

Khóa: chỉ script này đọc logit `calibration` để tính độ chính xác; `train_v4.py` không còn tóm tắt split này và `eval_sweep_v4.py` không đọc nó.
Kết quả đã có thì script từ chối chạy lại, không có cờ ghi đè: chạy lại rồi chọn số đẹp hơn chính là việc niêm phong muốn ngăn.

    python scripts/run_sealed_test.py --runs runs/v4b runs/sweep --cells "label_alpha_0_1|fedavg" "label_alpha_0_1|fedprox(mu=0.01)" --dry-run   # chỉ kiểm đủ run và đủ hạt giống, không mở logit, không tính số nào
    python scripts/run_sealed_test.py --runs runs/v4b runs/sweep --cells "label_alpha_0_1|fedavg" "label_alpha_0_1|fedprox(mu=0.01)" --confirm-final

Centralized và Local-only của mỗi cách chia được thêm tự động để giữ đủ ba mốc. Mọi ô phải đủ tối thiểu 3 hạt giống.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_sweep_v4 import BOOTSTRAP_SEED, MIN_SEEDS, compare_to_fedavg, part_score, spread, variant_of  # noqa: E402
from scripts.report_statistics import _cluster_of, cluster_bootstrap_ci  # noqa: E402
from src import registry as reg  # noqa: E402
from src.training.data import NUM_CLIENTS, load_release, read_jsonl  # noqa: E402
from src.training.metrics import LAB_DOMAIN  # noqa: E402

SEALED_SPLIT = "calibration"
TEST_SPLIT = "legacy_diagnostic_test"


def part_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Các phần của tập niêm phong. PlantDoc tách riêng 97 ảnh lá cách ly được nhận lại vì nhãn của chúng chỉ qua sàng lọc bằng AI."""
    lab = np.array([r["source_domain"] == LAB_DOMAIN for r in rows])
    plantdoc = np.array([r["source_dataset"] in ("PlantDoc", "plantdoc") for r in rows])
    reviewed = np.array([r.get("label_trust") == "reviewed_quarantine" for r in rows])
    return {"sealed_all": np.ones(len(rows), dtype=bool), "plantvillage": lab, "plantdoc": plantdoc, "plantdoc_without_reviewed_quarantine": plantdoc & ~reviewed, "field_other": ~lab & ~plantdoc}


def find_runs(run_dirs: list[Path], release_sha: str) -> list[dict[str, Any]]:
    """Các run đã xong trên đúng release; chỉ đọc cấu hình, chưa đụng logit."""
    runs = []
    for base in run_dirs:
        for run_dir in sorted(p for p in base.iterdir() if (p / "DONE").is_file()):
            cfg = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))["config"]
            if not cfg.get("limit") and cfg.get("release_manifest_sha256") == release_sha:
                runs.append({"dir": run_dir, "name": run_dir.name, "cfg": cfg, "method": cfg["method"], "seed": int(cfg["seed"]), "client": cfg.get("client"),
                             "scheme": cfg.get("scheme") if cfg["method"] != "centralized" else None, "variant": variant_of(cfg)})
    return runs


def plan_cells(runs: list[dict[str, Any]], requested: list[str]) -> dict[tuple[str | None, str], list[dict[str, Any]]]:
    """Ô phải đo = ô yêu cầu + Centralized + FedAvg và Local-only của mỗi cách chia có mặt. Thiếu ô hoặc thiếu hạt giống thì dừng TRƯỚC khi đọc logit, để không phí lần chạy duy nhất."""
    wanted: set[tuple[str | None, str]] = {(None, "centralized")}
    for text in requested:
        if "|" not in text:
            raise SystemExit(f"ô phải có dạng 'cách chia|biến thể': {text}")
        scheme, variant = text.split("|", 1)
        wanted |= {(scheme, variant), (scheme, "fedavg"), (scheme, "local")}
    cells: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if (run["scheme"], run["variant"]) in wanted:
            cells[(run["scheme"], run["variant"])].append(run)
    problems = []
    for key in sorted(wanted, key=str):
        members = cells.get(key, [])
        seeds = {r["seed"] for r in members}
        if len(seeds) < MIN_SEEDS:
            problems.append(f"{key}: chỉ có {len(seeds)} hạt giống, cần ít nhất {MIN_SEEDS}")
        if key[1] == "local":
            incomplete = [s for s in seeds if sum(r["seed"] == s for r in members) != NUM_CLIENTS]
            if incomplete:
                problems.append(f"{key}: hạt giống {sorted(incomplete)} không đủ {NUM_CLIENTS} client")
    if problems:
        raise SystemExit("chưa chạy được, chưa tính số nào trên tập niêm phong:\n  " + "\n  ".join(problems))
    return dict(cells)


def _load_preds(run: dict[str, Any], split: str, expected_ids: list[str]) -> np.ndarray:
    data = np.load(run["dir"] / "logits_best.npz", allow_pickle=False)
    if data[f"{split}_sample_ids"].tolist() != expected_ids:
        raise SystemExit(f"{run['name']}: mẫu trong logits_best.npz khác manifest {split}")
    return data[split].argmax(axis=1)


def evaluate(cells: dict[tuple[str | None, str], list[dict[str, Any]]], sealed_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top-1 và macro-F1 trung bình ± độ lệch chuẩn qua hạt giống cho từng phần; thêm PlantDoc gộp (niêm phong + tập thử) để có mẫu lớn hơn 172 ảnh."""
    sealed_ids = [r["sample_id"] for r in sealed_rows]
    test_ids = [r["sample_id"] for r in test_rows]
    sealed_targets = np.array([int(r["class_id"]) for r in sealed_rows])
    masks = part_masks(sealed_rows)
    test_plantdoc = np.array([r["source_dataset"] in ("PlantDoc", "plantdoc") for r in test_rows])
    extended_targets = np.concatenate([sealed_targets[masks["plantdoc"]], np.array([int(r["class_id"]) for r in test_rows])[test_plantdoc]])
    extended_rows = [r for r, m in zip(sealed_rows, masks["plantdoc"]) if m] + [r for r, m in zip(test_rows, test_plantdoc) if m]
    clusters = {name: [_cluster_of(r) for r, m in zip(sealed_rows, mask) if m] for name, mask in masks.items()}
    clusters["plantdoc_extended"] = [_cluster_of(r) for r in extended_rows]
    counts = {name: int(mask.sum()) for name, mask in masks.items()} | {"plantdoc_extended": len(extended_rows)}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    computed: dict[tuple[str | None, str], dict[int, dict[str, dict[str, float]]]] = {}
    correct: dict[tuple[str | None, str], dict[str, dict[int, np.ndarray]]] = {}
    for key, members in cells.items():
        per_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for run in members:
            sealed_preds = _load_preds(run, SEALED_SPLIT, sealed_ids)
            test_preds = _load_preds(run, TEST_SPLIT, test_ids)
            parts = {name: (sealed_preds[mask], sealed_targets[mask]) for name, mask in masks.items() if mask.any()}
            parts["plantdoc_extended"] = (np.concatenate([sealed_preds[masks["plantdoc"]], test_preds[test_plantdoc]]), extended_targets)
            per_seed[run["seed"]].append({name: {"score": part_score(p, t), "correct": p == t} for name, (p, t) in parts.items()})
        computed[key] = {}
        correct[key] = defaultdict(dict)
        for seed, entries in per_seed.items():         # Local-only: trung bình 5 client trong một hạt giống
            computed[key][seed] = {name: {m: float(np.mean([e[name]["score"][m] for e in entries])) for m in ("top1", "macro_f1")} for name in entries[0]}
            for name in entries[0]:
                correct[key][name][seed] = np.mean([e[name]["correct"] for e in entries], axis=0)

    central = computed[(None, "centralized")]
    out = []
    for key in sorted(cells, key=lambda k: (k[0] or "", k[1])):
        block: dict[str, Any] = {"scheme": key[0], "variant": key[1], "runs": sorted(r["name"] for r in cells[key]), "seeds": sorted(computed[key]), "parts": {}}
        for name in counts:
            if name not in next(iter(computed[key].values())):
                continue
            entry: dict[str, Any] = {"n": counts[name]}
            for metric in ("top1", "macro_f1"):
                by_seed = {s: v[name][metric] for s, v in computed[key].items()}
                entry[metric] = spread(by_seed)
                if key != (None, "centralized"):
                    entry[f"gap_{metric}_points"] = spread({s: 100 * (central[s][name][metric] - v) for s, v in by_seed.items() if s in central})
            pooled = np.mean([correct[key][name][s] for s in sorted(computed[key])], axis=0)
            interval = cluster_bootstrap_ci(clusters[name], pooled, rng)
            entry["top1_ci95"] = None if interval["low"] is None else [interval["low"], interval["high"]]
            if key[1] not in ("fedavg", "local", "centralized") and (key[0], "fedavg") in correct:
                entry["vs_fedavg"] = compare_to_fedavg(correct[key][name], correct[(key[0], "fedavg")][name], clusters[name], rng)
            block["parts"][name] = entry
        out.append(block)
    return out


def run_sealed(run_dirs: list[Path], release_dir: Path, requested: list[str], result_path: Path, dry_run: bool) -> dict[str, Any] | None:
    """`result_path` đã có thì từ chối: kết quả niêm phong chỉ được sinh một lần."""
    if result_path.exists():
        raise SystemExit(f"{result_path} đã có: tập niêm phong đã được chạy. Không chạy lại; muốn đo thêm thì dùng tập thử thường (scripts/eval_sweep_v4.py).")
    release = load_release(release_dir, None)
    cells = plan_cells(find_runs(run_dirs, release["sha256"]), requested)
    if dry_run:
        return {"dry_run": True, "cells": {f"{k[0]}|{k[1]}": sorted(r["name"] for r in v) for k, v in cells.items()}}
    sealed_rows = read_jsonl(release_dir / "manifests" / f"{SEALED_SPLIT}.jsonl")
    test_rows = read_jsonl(release_dir / "manifests" / f"{TEST_SPLIT}.jsonl")
    report = {"evidence": "chạy một lần duy nhất trên split calibration, sau khi chốt cấu hình bằng tập val", "split": SEALED_SPLIT, "release_manifest_sha256": release["sha256"],
              "sealed_manifest_sha256": release["meta"]["manifest_hashes"].get(SEALED_SPLIT), "requested_cells": requested, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "cells": evaluate(cells, sealed_rows, test_rows)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--cells", nargs="+", required=True, help="các ô đã chốt, dạng 'cách chia|biến thể'; xem tên biến thể ở artifacts/sweep_v4/table_main.csv")
    parser.add_argument("--release-dir", type=Path, default=reg.abs_path(reg.RELEASE_DIR))
    parser.add_argument("--dry-run", action="store_true", help="chỉ kiểm đủ run và đủ hạt giống, không mở logit và không tính số nào trên tập niêm phong")
    parser.add_argument("--confirm-final", action="store_true", help="xác nhận đây là lần chạy chính thức duy nhất")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_final:
        parser.error("cần --confirm-final (chỉ chạy được một lần) hoặc --dry-run để kiểm trước")
    report = run_sealed(args.runs, args.release_dir, args.cells, reg.abs_path(reg.SEALED_TEST_JSON), args.dry_run)
    if args.dry_run:
        print("[sealed] kiểm trước đạt, chưa tính số nào trên tập niêm phong:")
        for cell, names in report["cells"].items():
            print(f"  {cell}: {len(names)} run")
        return 0
    print(f"[sealed] ghi {reg.SEALED_TEST_JSON}")
    for cell in report["cells"]:
        part = cell["parts"]["sealed_all"]
        print(f"  {cell['scheme']}|{cell['variant']}: top-1 {100 * part['top1']['mean']:.2f}% ± {100 * (part['top1']['std'] or 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
