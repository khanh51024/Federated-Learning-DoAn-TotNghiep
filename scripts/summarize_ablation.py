"""Tóm tắt các run ablation (scripts/remote/run_ablation.sh) trên tập val: không dùng tập thử, không dùng PlantWild để chọn công thức.

Mỗi run (thư mục có history.jsonl) cho một dòng: điểm chọn tốt nhất (macro-F1 trung bình giữa các nguồn val), epoch đạt, trung bình 5 epoch cuối
(ít nhiễu hơn một điểm đơn), macro-F1 ảnh ngoài đồng, top-1 PlantVillage, top-1 PlantDoc ở epoch tốt nhất, và số giây trung bình mỗi epoch.

    python scripts/summarize_ablation.py runs/ablation [--baseline runs/v4/centralized_s42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "history.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [r for r in rows if r["val"].get("selection_score") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda r: r["val"]["selection_score"])
    tail = scored[-5:]
    by_source = best["val"].get("by_source", {})
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if (run_dir / "config.json").is_file() else {}
    return {
        "run": run_dir.name, "epochs": f"{len(rows)}/{config.get('budget', '?')}", "best": best["val"]["selection_score"], "best_epoch": best["step"] + 1,
        "last5": round(sum(r["val"]["selection_score"] for r in tail) / len(tail), 4),
        "field_f1": best["val"].get("field_macro_f1"), "pv_top1": best["val"].get("plantvillage_top1"),
        "plantdoc_top1": by_source.get("PlantDoc", {}).get("top1"),
        "sec_per_epoch": round(sum(r["seconds"] for r in rows[1:]) / max(1, len(rows) - 1), 1) if len(rows) > 1 else rows[0]["seconds"],
        "recipe": f"{config.get('train_transform', '?').split(' ')[0]} {config.get('loss_weighting', '')} la={config.get('logit_adjust_tau', 0)} size={config.get('train_size', 224)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    dirs = [*args.baseline, *sorted(p for p in args.runs.iterdir() if p.is_dir())]
    out = [s for s in (summarize_run(d) for d in dirs) if s]
    cols = ("run", "epochs", "best", "best_epoch", "last5", "field_f1", "pv_top1", "plantdoc_top1", "sec_per_epoch", "recipe")
    print(" | ".join(cols))
    for s in out:
        print(" | ".join(str(s[c]) for c in cols))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
