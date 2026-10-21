"""Lọc ảnh iNaturalist: chỉ giữ ảnh chụp LÁ (loại ảnh quả, hoa, cả cây, thân, nấm, côn trùng, phong cảnh) bằng BioCLIP zero-shot.

iNaturalist là ảnh sinh vật tổng quát, nên ảnh tải theo loài cây trồng hoặc theo tên tác nhân gây bệnh phần lớn KHÔNG phải ảnh lá. Với mỗi ảnh, tính xác suất
softmax của BioCLIP trên một tập câu mô tả gồm câu "đạt" (lá, lá có vết bệnh) và câu "loại" (quả, hoa, cả cây, thân, nấm/khối u, côn trùng, phong cảnh); giữ ảnh khi tổng xác
suất của các câu "đạt" >= ngưỡng. Kết quả ghi vào <nguồn>/leaf_filter.json (điểm từng ảnh); scripts/import_external_source.py chỉ nhập ảnh được giữ.

Đây là bộ lọc thô để giảm số ảnh phải xem; nhãn bệnh/khỏe vẫn phải qua kiểm nhãn mẫu (scripts/build_label_review.py).

Cần môi trường có open_clip_torch (tmp/venv-fm):
    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/filter_leaf_photos.py --all
    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/filter_leaf_photos.py --source inat_apple___healthy --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SOURCES_DIR = "data/external/sources"
BIOCLIP = "hf-hub:imageomics/bioclip"
THRESHOLD = 0.55
POSITIVE = ["a close-up photo of a {leaf}", "a photo of {leaf} with spots or lesions", "a photo of several {leaf}s on a plant"]
NEGATIVE = [
    "a photo of fruit", "a photo of a flower", "a photo of a whole tree or shrub in a landscape", "a photo of a tree trunk or bark or a stem",
    "a photo of a mushroom or a fungus fruiting body", "a photo of a gall or swelling on a branch", "a photo of an insect or a mite",
    "a photo of a landscape or a field from far away", "a photo of a seed or a pod", "a photo of a person",
]


def prompts_for(leaf_phrase: str) -> tuple[list[str], list[str]]:
    return [p.format(leaf=leaf_phrase) for p in POSITIVE], list(NEGATIVE)


def leaf_probability(probs: np.ndarray, n_positive: int) -> np.ndarray:
    """Tổng xác suất của các câu 'đạt' (n_positive câu đầu) trên mỗi ảnh."""
    return probs[:, :n_positive].sum(axis=1)


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import open_clip
    import torch
    from PIL import Image

    torch.set_num_threads(max(1, args.threads))
    config = yaml.safe_load(reg.abs_path("configs/external_sources.yaml").read_text(encoding="utf-8"))["inaturalist"]["classes"]
    phrase_by_class = {c["target_class"]: c.get("leaf_prompt") for c in config}
    base = reg.abs_path(SOURCES_DIR)
    names = [d.name for d in sorted(base.glob("inat_*"))] if args.all else args.source
    model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP)
    model.eval()
    tokenizer = open_clip.get_tokenizer(BIOCLIP)
    for name in names:
        meta_path = base / name / "inat_metadata.jsonl"
        if not meta_path.is_file():
            continue
        rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        target = rows[0]["target_class"]
        pos, neg = prompts_for(phrase_by_class.get(target) or "plant leaf")
        with torch.inference_mode():
            text = model.encode_text(tokenizer(pos + neg))
            text = text / text.norm(dim=-1, keepdim=True)
        done_path = base / name / "leaf_filter.json"
        done: dict[str, Any] = json.loads(done_path.read_text(encoding="utf-8")) if done_path.is_file() else {}
        todo = [r for r in rows if r["file"] not in done.get("scores", {})]
        scores: dict[str, float] = dict(done.get("scores", {}))
        for start in range(0, len(todo), 32):
            chunk = todo[start:start + 32]
            images, ok = [], []
            for r in chunk:
                try:
                    with Image.open(base / name / "raw" / r["file"]) as handle:
                        images.append(preprocess(handle.convert("RGB")))
                    ok.append(r)
                except Exception:  # noqa: BLE001
                    scores[r["file"]] = 0.0
            if not images:
                continue
            with torch.inference_mode():
                feats = model.encode_image(torch.stack(images))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                probs = (100.0 * feats @ text.T).softmax(dim=-1).numpy()
            for r, p in zip(ok, leaf_probability(probs, len(pos))):
                scores[r["file"]] = round(float(p), 4)
        kept = [f for f, s in scores.items() if s >= args.threshold]
        done_path.write_text(json.dumps({"threshold": args.threshold, "model": BIOCLIP, "positive": pos, "negative": neg, "scores": scores,
                                         "kept": len(kept), "total": len(scores)}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
        print(f"[leaf] {name}: giữ {len(kept)}/{len(scores)} ảnh (ngưỡng {args.threshold})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
