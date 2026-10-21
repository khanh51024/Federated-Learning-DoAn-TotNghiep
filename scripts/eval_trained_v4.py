"""Đánh giá các run huấn luyện v4 sau khi tải về máy này, và kiểm các ngưỡng đặt trước (docs/NEXT_TRAINING_PLAN.md mục 5.4).

Đầu vào: thư mục runs (mỗi run có checkpoint_best.pt, logits_best.npz, final.json, DONE) do scripts/train_v4.py sinh trên máy GPU.
Việc làm ở máy này (PlantWild giấy phép CC BY-NC-ND không đưa lên máy thuê):
  1. logit của từng checkpoint trên PlantWild đã khóa (5.045 ảnh, `canonical_v1`), lưu cache cạnh run (plantwild_logits.npz);
  2. tách tập thử v4 theo phần: PlantVillage test, PlantDoc giữ riêng của v3 (đúng các ảnh mô hình cũ được đo), ngoài đồng cùng nguồn (nguồn mới);
  3. gộp theo cách (trung bình ± độ lệch chuẩn qua hạt giống; Local-only lấy trung bình 5 client trong một hạt giống trước);
  4. so với mô hình đang công bố (cùng tập, cùng cách đo) và báo từng ngưỡng: đạt / không đạt, kèm khoảng tin cậy bootstrap.
Ghi artifacts/train_v4_results.json.

Chạy:
    python scripts/eval_trained_v4.py --runs runs/v4
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

from src import registry as reg  # noqa: E402

RESULTS_JSON = "artifacts/train_v4_results.json"
V3_DIR = "data/dataset_sources/mixed/pv_pd_v3"
V4_DIR = "data/dataset_sources/mixed/pv_pd_v4"
BOOTSTRAP = 2000
OLD = {"plantwild_strict_top1": {"centralized": 0.331, "fedavg": 0.318}, "plantdoc_holdout_top1": {"centralized": 0.634, "fedavg": 0.500}}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bootstrap_ci(correct: np.ndarray, seed: int = 20261022) -> list[float]:
    if len(correct) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(BOOTSTRAP, len(correct)))].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def plantwild_logits(run_dir: Path, items: list[dict[str, Any]], threads: int) -> np.ndarray:
    cache = run_dir / "plantwild_logits.npz"
    keys = np.array([i["key"] for i in items])
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if np.array_equal(data["keys"], keys):
            return data["logits"]
    import torch
    from PIL import Image

    from scripts.evaluate_published_models import _load_model
    from src.common.image_io import to_rgb
    from src.data.robust_transforms import get_eval_transform

    torch.set_num_threads(max(1, threads))
    model = _load_model(run_dir / "checkpoint_best.pt")
    transform = get_eval_transform()
    out = np.zeros((len(items), 38), np.float32)
    with torch.inference_mode():
        for start in range(0, len(items), 64):
            batch = []
            for item in items[start:start + 64]:
                with Image.open(item["path"]) as handle:
                    batch.append(transform(to_rgb(handle)))
            out[start:start + len(batch)] = model(torch.stack(batch)).numpy()
    np.savez_compressed(cache, keys=keys, logits=out)
    return out


def part_masks(test_rows: list[dict[str, Any]], v3_test_ids: set[str]) -> dict[str, np.ndarray]:
    pv = np.array([r["source_domain"] == "plantvillage" for r in test_rows])
    pd_v3 = np.array([r["source_domain"] == "plantdoc" and r["source_dataset"] in ("PlantDoc", "plantdoc") and r["sample_id"] in v3_test_ids for r in test_rows])
    field_new = np.array([r["source_domain"] == "plantdoc" and r["source_dataset"] not in ("PlantDoc", "plantdoc") and r["source_dataset"] != "own_photos" for r in test_rows])
    own = np.array([r["source_dataset"] == "own_photos" for r in test_rows])
    return {"plantvillage_test": pv, "plantdoc_holdout_v3": pd_v3, "field_same_source_test": field_new, "own_photos_test": own}


def measure(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if len(targets) == 0:
        return {"n": 0}
    correct = (logits.argmax(1) == targets).astype(float)
    top3 = (np.argsort(-logits, axis=1)[:, :3] == targets[:, None]).any(1).astype(float)
    return {"n": int(len(targets)), "top1": round(float(correct.mean()), 4), "top1_ci95": bootstrap_ci(correct), "top3": round(float(top3.mean()), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--include-limited", action="store_true", help="tính cả run thử trên tập nhỏ (--limit), chỉ để kiểm script; ghi vào tmp/ thay vì artifacts/")
    args = parser.parse_args()

    from scripts.build_external_eval_set import EXTERNAL_DIR, MANIFEST

    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]
    healthy = np.array(["healthy" in c for c in classes])
    wild_rows = json.loads(reg.abs_path(MANIFEST).read_text(encoding="utf-8"))["samples"]
    base = reg.abs_path(EXTERNAL_DIR)
    wild_items = [{"key": r["byte_sha256"], "path": base / r["relative_path"]} for r in wild_rows]
    wild_targets = np.array([int(r["class_id"]) for r in wild_rows])
    strict = np.array([r["tier"] == "strict" for r in wild_rows])
    v3_test_ids = {r["sample_id"] for r in jsonl(reg.abs_path(V3_DIR) / "manifests" / "legacy_diagnostic_test.jsonl")}
    test_rows = {r["sample_id"]: r for r in jsonl(reg.abs_path(V4_DIR) / "manifests" / "legacy_diagnostic_test.jsonl")}
    ceilings = json.loads(reg.abs_path(reg.DATA_GAP_REPORT_JSON).read_text(encoding="utf-8")).get("local_only_ceiling_top1_percent", {})

    runs: dict[str, Any] = {}
    for run_dir in sorted(p for p in args.runs.iterdir() if (p / "DONE").is_file()):
        final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
        cfg = final["config"]
        if cfg.get("limit") and not args.include_limited:
            continue  # run thử trên tập nhỏ: không báo cáo
        data = np.load(run_dir / "logits_best.npz", allow_pickle=False)
        rows = [test_rows[s] for s in data["legacy_diagnostic_test_sample_ids"]]
        targets = np.array([int(r["class_id"]) for r in rows])
        masks = part_masks(rows, v3_test_ids)
        started = time.time()
        wild = plantwild_logits(run_dir, wild_items, args.threads)
        healthy_rows = ~strict & healthy[wild_targets]
        entry = {"method": cfg["method"], "seed": cfg["seed"], "client": cfg["client"], "best_step": final["best_step"],
                 "val_selection_score": final["selection_score"], "release_manifest_sha256": cfg["release_manifest_sha256"],
                 **{name: measure(data["legacy_diagnostic_test"][m], targets[m]) for name, m in masks.items()},
                 "plantwild_strict": measure(wild[strict], wild_targets[strict]),
                 "plantwild_healthy_as_disease": round(float((~healthy[wild[healthy_rows].argmax(1)]).mean()), 4) if healthy_rows.any() else None}
        runs[run_dir.name] = entry
        print(f"[eval] {run_dir.name}: PV {entry['plantvillage_test'].get('top1')} PD {entry['plantdoc_holdout_v3'].get('top1')} "
              f"ngoài đồng {entry['field_same_source_test'].get('top1')} PlantWild {entry['plantwild_strict'].get('top1')} ({time.time() - started:.0f}s)", flush=True)
    if not runs:
        print("[eval] không có run nào đã xong (DONE) và chạy thật", file=sys.stderr)
        return 1

    metrics = ("plantvillage_test", "plantdoc_holdout_v3", "field_same_source_test", "plantwild_strict")
    per_seed: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in runs.values():
        for m in metrics:
            if entry[m].get("n"):
                per_seed[entry["method"]][entry["seed"]][m].append(entry[m]["top1"])
        if entry["plantwild_healthy_as_disease"] is not None:
            per_seed[entry["method"]][entry["seed"]]["plantwild_healthy_as_disease"].append(entry["plantwild_healthy_as_disease"])
    summary: dict[str, Any] = {}
    for method, seeds in per_seed.items():
        summary[method] = {"seeds": sorted(seeds), "clients_per_seed": {s: len(v["plantvillage_test"]) for s, v in seeds.items()}}
        for m in (*metrics, "plantwild_healthy_as_disease"):
            values = [float(np.mean(v[m])) for v in seeds.values() if v[m]]   # Local-only: trung bình 5 client trong một hạt giống
            if values:
                summary[method][m] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None}

    def mean(method: str, m: str) -> float | None:
        return summary.get(method, {}).get(m, {}).get("mean")

    def ci_low(method: str, m: str) -> float | None:
        lows = [e[m]["top1_ci95"][0] for e in runs.values() if e["method"] == method and e[m].get("n")]
        return round(float(np.min(lows)), 4) if lows else None

    checks = []
    for method in ("centralized", "fedavg"):
        if method not in summary:
            continue
        checks += [
            {"id": f"A_{method}_plantvillage", "rule": ">= 0,90 top-1 PlantVillage test", "value": mean(method, "plantvillage_test"),
             "pass": (mean(method, "plantvillage_test") or 0) >= 0.90},
            {"id": f"A_{method}_field_same_source", "rule": ">= 0,90 top-1 ngoài đồng cùng nguồn", "value": mean(method, "field_same_source_test"),
             "pass": (mean(method, "field_same_source_test") or 0) >= 0.90},
            {"id": f"B_{method}_plantwild_65", "rule": ">= 0,65 top-1 PlantWild bệnh", "value": mean(method, "plantwild_strict"),
             "pass": (mean(method, "plantwild_strict") or 0) >= 0.65},
            {"id": f"web1_{method}_plantwild_plus10", "rule": f">= {OLD['plantwild_strict_top1'][method] + 0.10:.3f} và cận dưới CI mọi hạt giống > {OLD['plantwild_strict_top1'][method]}",
             "value": mean(method, "plantwild_strict"), "ci_low_min": ci_low(method, "plantwild_strict"),
             "pass": (mean(method, "plantwild_strict") or 0) >= OLD["plantwild_strict_top1"][method] + 0.10
                     and (ci_low(method, "plantwild_strict") or 0) > OLD["plantwild_strict_top1"][method]},
            {"id": f"web2_{method}_plantdoc_holdout", "rule": f">= {OLD['plantdoc_holdout_top1'][method] - 0.02:.3f} (không giảm quá 2 điểm)",
             "value": mean(method, "plantdoc_holdout_v3"), "pass": (mean(method, "plantdoc_holdout_v3") or 0) >= OLD["plantdoc_holdout_top1"][method] - 0.02},
            {"id": f"web4_{method}_healthy_as_disease", "rule": "< 0,60 (nhãn ngầm định, chỉ báo xu hướng)", "value": mean(method, "plantwild_healthy_as_disease"),
             "pass": (mean(method, "plantwild_healthy_as_disease") or 1) < 0.60},
            {"id": f"web5_{method}_three_seeds", "rule": "đủ 3 hạt giống", "value": len(summary[method]["seeds"]), "pass": len(summary[method]["seeds"]) >= 3},
        ]
    report = {
        "evidence": "đo trên checkpoint tốt nhất (chọn theo val) của từng run; PlantWild đo ở máy này; tập thử v4 đo trên máy GPU (logits_best.npz)",
        "runs": runs, "summary": summary, "checks": checks,
        "local_only_ceiling_top1_percent": ceilings,
        "not_checked_here": "Ngưỡng 3 (PlantVillage không giảm quá 0,5 điểm) cần số của mô hình cũ trên cùng tập: so bằng scripts/evaluate_published_models.py; "
                            "D (cổng từ chối) phải hiệu chỉnh lại cho mô hình mới bằng scripts/eval_ood_gate.py.",
    }
    out = reg.abs_path("tmp/train_v4_results_limited.json" if args.include_limited else RESULTS_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'cách':12s} {'PV':>12s} {'PD giữ riêng':>14s} {'ngoài đồng':>12s} {'PlantWild':>12s}")
    for method, s in summary.items():
        cell = lambda m: f"{s[m]['mean']:.3f}±{s[m]['std'] or 0:.3f}" if m in s else "-"
        print(f"{method:12s} {cell('plantvillage_test'):>12s} {cell('plantdoc_holdout_v3'):>14s} {cell('field_same_source_test'):>12s} {cell('plantwild_strict'):>12s}")
    for c in checks:
        print(f"[{'ĐẠT' if c['pass'] else 'KHÔNG'}] {c['id']}: {c['value']} ({c['rule']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
