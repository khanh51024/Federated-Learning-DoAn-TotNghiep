"""Áp quy tắc nhận nhóm lên kết quả kiểm nhãn và ghi trạng thái từng nhóm.

Đầu vào: `samples.json` do scripts/build_label_review.py sinh (nhóm -> danh sách sample_id đã lấy mẫu) và một hoặc nhiều tệp `decisions.json`
(`{"decisions": {sample_id: "dung" | "sai" | "khong_ro"}, ...}`): do người kiểm bằng trang HTML, hoặc do sàng lọc bằng mắt của AI
(`reviewer` trong tệp). Tệp đứng sau ghi đè tệp đứng trước (đặt tệp của người cuối cùng để người có quyền quyết định).

Quy tắc nhận nhóm (đặt trước khi xem kết quả, ở scripts/build_label_review.py): nhóm được nhận khi mọi ảnh lấy mẫu đã kiểm, "đúng" >= 90% và "không rõ" <= 20%.
Kết quả ghi vào `status.json` cạnh `samples.json`: với mỗi nhóm ghi số liệu, `accepted`, và các tệp/bên kiểm đã dùng. Release v4 chỉ nhận ảnh `unreviewed`
của nhóm có `accepted == true`.

Chạy:
    python scripts/apply_label_review.py --samples data/label_review_ext/samples.json --decisions data/label_review_ext/decisions_ai.json [--decisions .../decisions.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_label_review import classify  # noqa: E402


def merge_decisions(files: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """(quyết định gộp, bên kiểm của từng quyết định). Tệp sau ghi đè tệp trước."""
    merged: dict[str, str] = {}
    who: dict[str, str] = {}
    for payload in files:
        reviewer = str(payload.get("reviewer", "human"))
        for sample_id, verdict in payload.get("decisions", {}).items():
            if verdict in ("dung", "sai", "khong_ro"):
                merged[sample_id] = verdict
                who[sample_id] = reviewer
    return merged, who


def evaluate(samples: dict[str, Any], decisions: dict[str, str], who: dict[str, str]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, ids in samples["classes"].items():
        result = classify(decisions, ids)
        result["reviewers"] = sorted({who[i] for i in ids if i in who})
        result["group_total"] = samples.get("totals", {}).get(name)
        groups[name] = result
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--decisions", action="append", required=True)
    args = parser.parse_args()

    samples_path = Path(args.samples)
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.decisions]
    decisions, who = merge_decisions(payloads)
    groups = evaluate(samples, decisions, who)
    status = {
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "decision_files": [str(p) for p in args.decisions],
        "accept_rule": {"correct_min": samples["accept_correct"], "unsure_max": samples["accept_unsure_max"]},
        "groups": groups,
        "accepted": sorted(k for k, v in groups.items() if v["accepted"]),
        "rejected": sorted(k for k, v in groups.items() if v["complete"] and not v["accepted"]),
        "incomplete": sorted(k for k, v in groups.items() if not v["complete"]),
    }
    per_image = set(samples.get("per_image_groups", []))
    accepted_samples = sorted(i for g in per_image for i in samples["classes"].get(g, []) if decisions.get(i) == "dung")
    for g in per_image:
        if g in groups:
            groups[g]["per_image"] = True
            groups[g]["accepted_samples"] = sum(1 for i in samples["classes"][g] if decisions.get(i) == "dung")
            groups[g]["accepted"] = False  # nhóm kiểm từng ảnh không được nhận nguyên nhóm
    status["groups"] = groups
    status["accepted"] = sorted(k for k, v in groups.items() if v["accepted"])
    status["rejected"] = sorted(k for k, v in groups.items() if v["complete"] and not v["accepted"] and not v.get("per_image"))
    status["accepted_samples"] = accepted_samples
    out = samples_path.with_name("status.json")
    out.write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[label-review] nhận {len(status['accepted'])}, loại {len(status['rejected'])}, chưa xong {len(status['incomplete'])} nhóm -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
