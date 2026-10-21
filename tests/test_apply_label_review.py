"""scripts/apply_label_review.py và scripts/record_review_decisions.py: gộp quyết định, quy tắc nhận nhóm, nhóm kiểm từng ảnh."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.apply_label_review import evaluate, merge_decisions
from scripts.record_review_decisions import build_decisions

ROOT = Path(__file__).resolve().parents[1]


def _samples() -> dict:
    return {"accept_correct": 0.9, "accept_unsure_max": 0.2, "totals": {"G1": 100, "G2": 80},
            "classes": {"G1": [f"a{i}" for i in range(10)], "G2": [f"b{i}" for i in range(10)]}, "per_image_groups": ["G2"]}


def test_later_decision_file_overrides_earlier_and_records_reviewer():
    ai = {"reviewer": "ai_screening", "decisions": {"a0": "sai", "a1": "dung"}}
    human = {"reviewer": "human", "decisions": {"a0": "dung"}}
    merged, who = merge_decisions([ai, human])
    assert merged == {"a0": "dung", "a1": "dung"} and who == {"a0": "human", "a1": "ai_screening"}


def test_record_decisions_defaults_unmarked_cells_to_correct():
    samples = _samples()
    out = build_decisions(samples, {"G1": {"sai": [3], "khong_ro": [4]}})
    assert out["a2"] == "sai" and out["a3"] == "khong_ro" and out["a0"] == "dung"
    assert not any(k.startswith("b") for k in out)  # nhóm không có trong spec thì chưa kiểm


def test_evaluate_accepts_clean_group_and_rejects_group_with_too_many_unsure():
    samples = _samples()
    decisions = {f"a{i}": "dung" for i in range(10)} | {f"b{i}": ("khong_ro" if i < 3 else "dung") for i in range(10)}
    groups = evaluate(samples, decisions, {})
    assert groups["G1"]["accepted"] is True and groups["G2"]["accepted"] is False


def test_cli_marks_per_image_group_samples_individually(tmp_path):
    samples = _samples()
    (tmp_path / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    dec = {"reviewer": "ai_screening", "decisions": {**{f"a{i}": "dung" for i in range(10)}, **{f"b{i}": ("dung" if i % 2 == 0 else "sai") for i in range(10)}}}
    (tmp_path / "d.json").write_text(json.dumps(dec), encoding="utf-8")
    subprocess.run([sys.executable, "-X", "utf8", "-m", "scripts.apply_label_review", "--samples", str(tmp_path / "samples.json"),
                    "--decisions", str(tmp_path / "d.json")], check=True, cwd=ROOT, capture_output=True)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["accepted"] == ["G1"]  # G2 là nhóm kiểm từng ảnh: không nhận nguyên nhóm
    assert status["accepted_samples"] == [f"b{i}" for i in range(0, 10, 2)]
