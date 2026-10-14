"""Ghi lại kết quả SÀNG LỌC nhãn bằng mắt (do trợ lý AI thực hiện) cho 14 nhóm lá PlantDoc bị cách ly.

Đây KHÔNG phải kiểm định của chuyên gia nông nghiệp. Người sàng lọc là trợ lý AI (Claude) xem các bảng ảnh thu nhỏ do
`scripts/build_label_review.py` dựng (14 nhóm × 30 lá, seed cố định) và đánh dấu từng ảnh:
  dung      lá đúng loại cây và phù hợp nhãn (nhóm lá khỏe: không thấy tổn thương; nhóm bệnh: triệu chứng phù hợp);
  sai       thấy rõ là sai (tổn thương ở nhóm "khỏe", khác loại cây, hoặc triệu chứng của bệnh khác);
  khong_ro  không đủ căn cứ (ảnh quá nhỏ/mờ, bị cắt, màu bất thường, triệu chứng mơ hồ). Cố ý thiên về thận trọng.
Chỉ các ảnh bị đánh dấu `sai`/`khong_ro` được liệt kê dưới đây (số thứ tự 1..30 trên bảng ảnh); ảnh còn lại là `dung`.

Quy tắc nhận nhóm (đặt trước khi sàng lọc, trong build_label_review.py): ≥ 90% đúng và ≤ 20% không rõ.

Chạy:  python scripts/record_label_screening.py   # ghi artifacts/label_screening.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_label_review import ACCEPT_CORRECT, ACCEPT_UNSURE_MAX, classify  # noqa: E402
from src import registry as reg  # noqa: E402

REVIEWER = "ai_visual_screening_claude_2026-10-12"
# nhóm -> (chỉ số 1..30 bị đánh dấu sai, chỉ số bị đánh dấu không rõ)
FLAGGED: dict[str, tuple[set[int], set[int]]] = {
    "Apple___Cedar_apple_rust": ({2}, {3, 5, 6, 8, 9, 10, 14, 20, 21, 25, 27, 28}),
    "Apple___healthy": (set(), {1, 11, 13, 17, 22}),
    "Blueberry___healthy": ({6, 10, 22}, {1, 2, 3, 5, 8, 9, 18, 23, 28}),
    "Cherry_(including_sour)___healthy": (set(), {3, 7, 10, 12, 13, 18, 23}),
    "Corn_(maize)___Northern_Leaf_Blight": ({5, 16, 18, 29}, set(range(1, 31)) - {1, 3, 6, 10, 11, 21, 28} - {5, 16, 18, 29}),
    "Grape___healthy": (set(), {6, 17, 21}),
    "Peach___healthy": (set(), {4, 13, 20, 21, 27, 28, 29}),
    "Pepper,_bell___Bacterial_spot": (set(), set(range(1, 31)) - {2, 4, 8, 13, 17, 19, 22, 23, 25, 26, 28}),
    "Pepper,_bell___healthy": ({2, 3, 8}, {1, 9, 15, 17, 19, 22, 24, 26, 30}),
    "Raspberry___healthy": (set(), {5, 17, 18, 22, 30}),
    "Soybean___healthy": ({17, 23}, {8, 11, 13, 26, 28}),
    "Strawberry___healthy": (set(), {10, 15}),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (set(), {1, 4, 5, 8, 9, 11, 12, 13, 14, 15, 19, 23, 26, 29}),
    "Tomato___healthy": ({14, 22, 23, 24}, {1, 4, 5, 9, 13, 16, 17, 21, 25, 28, 30}),
}


def build(samples: dict[str, list[str]]) -> dict:
    decisions: dict[str, str] = {}
    classes = {}
    for name, ids in samples.items():
        wrong, unsure = FLAGGED[name]
        if wrong & unsure:
            raise ValueError(f"{name}: một ảnh không thể vừa sai vừa không rõ: {sorted(wrong & unsure)}")
        if any(not 1 <= i <= len(ids) for i in wrong | unsure):
            raise ValueError(f"{name}: chỉ số ngoài 1..{len(ids)}")
        for position, sample_id in enumerate(ids, start=1):
            decisions[sample_id] = "sai" if position in wrong else "khong_ro" if position in unsure else "dung"
        result = classify(decisions, ids)
        result["wrong_rate"] = round(result["sai"] / result["total"], 4)
        classes[name] = result
    return {"decisions": decisions, "classes": classes}


def main() -> int:
    samples_path = reg.abs_path("data/label_review/samples.json")
    if not samples_path.is_file():
        print("[screening] chạy scripts/build_label_review.py trước", file=sys.stderr)
        return 1
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    missing = set(samples["classes"]) ^ set(FLAGGED)
    if missing:
        print(f"[screening] nhóm lệch giữa mẫu và bảng đánh dấu: {sorted(missing)}", file=sys.stderr)
        return 1
    built = build(samples["classes"])
    accepted = sorted(n for n, c in built["classes"].items() if c["accepted"])
    report = {
        "reviewer": REVIEWER,
        "kind": "ai_visual_screening",
        "caveat": "Sàng lọc bằng mắt qua ảnh thu nhỏ bởi trợ lý AI, KHÔNG phải kiểm định của chuyên gia nông nghiệp; cố ý "
                  "thận trọng (ảnh mơ hồ tính là 'không rõ', không tính là đúng). Chỉ là bằng chứng sơ bộ về chất lượng nhãn.",
        "seed": samples["seed"],
        "per_class": samples["per_class"],
        "rule": {"accept_correct": ACCEPT_CORRECT, "accept_unsure_max": ACCEPT_UNSURE_MAX},
        "accepted_classes": accepted,
        "classes": built["classes"],
        "decisions": built["decisions"],
    }
    out = reg.abs_path(reg.LABEL_SCREENING_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, c in sorted(built["classes"].items(), key=lambda kv: -kv[1]["dung"]):
        print(f"{'NHẬN ' if c['accepted'] else 'không '} {name:42s} đúng {c['dung']:2d}  sai {c['sai']:2d}  không rõ {c['khong_ro']:2d}")
    print(f"[screening] {len(accepted)}/{len(built['classes'])} nhóm đạt quy tắc → {reg.LABEL_SCREENING_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
