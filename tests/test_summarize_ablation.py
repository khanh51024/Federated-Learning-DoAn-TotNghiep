"""scripts/summarize_ablation.py: một dòng tóm tắt cho mỗi run ablation, chỉ từ tập val."""
from __future__ import annotations

import json

from scripts.summarize_ablation import summarize_run


def _record(step, score, seconds=10.0):
    return {"step": step, "seconds": seconds, "val": {"selection_score": score, "field_macro_f1": score - 0.1, "plantvillage_top1": 0.98,
                                                       "by_source": {"PlantDoc": {"top1": 0.5}}}}


def test_summary_picks_the_best_epoch_and_averages_the_last_five(tmp_path):
    run = tmp_path / "centralized_s42_C"
    run.mkdir()
    scores = [0.70, 0.80, 0.85, 0.83, 0.84, 0.86, 0.82]
    (run / "history.jsonl").write_text("".join(json.dumps(_record(i, s, 30.0 if i == 0 else 12.0)) + "\n" for i, s in enumerate(scores)),
                                       encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"budget": 20, "train_transform": "gpu_v2 (x)", "loss_weighting": "w"}), encoding="utf-8")
    out = summarize_run(run)
    assert out["best"] == 0.86 and out["best_epoch"] == 6 and out["epochs"] == "7/20"
    assert out["last5"] == round(sum(scores[-5:]) / 5, 4)
    assert out["sec_per_epoch"] == 12.0                     # bỏ epoch đầu (khởi động, đọc ảnh lần đầu)
    assert summarize_run(tmp_path / "missing") is None
