"""Trích đặc trưng DINOv2-small (token CLS) cho mọi ảnh ngoài đồng đã nhập, để gom các ảnh chụp lại CÙNG MỘT lá vào cùng một nhóm.

Vì sao cần: dHash không nhận ra cùng một lá khi người chụp xoay lá trên nền giấy (bộ cà chua ruộng, ớt, nhiều bộ Mendeley): khoảng cách dHash của hai ảnh xoay
lớn như hai lá khác nhau, nên ảnh của một lá rơi vào cả train lẫn test. Đặc trưng DINOv2 ổn định hơn nhiều với xoay và đổi góc chụp; scripts/build_release_v4.py
gộp ảnh cùng nguồn, cùng lớp có độ tương tự cosine >= ngưỡng (`--feature-threshold`) vào cùng nhóm.

Chỉ cần cho nguồn lá đặt trên nền giấy đồng màu (`--sources`, mặc định 7 nguồn ở UNIFORM_BACKGROUND_SOURCES, khoảng 5,6 nghìn ảnh, vài phút); nguồn khác đã gom
tốt bằng dHash, tên tệp và tiền tố tăng cường. Kết quả: data/external/dino_features.npz (khóa `sample_id`, đặc trưng float16 [N, 384]); chạy lại chỉ tính ảnh chưa có, lưu định kỳ nên ngắt được.
Cần tmp/venv-fm (torch, transformers). Chạy một mình, không song song tác vụ nặng khác (khoảng 2 GB RAM).

    tmp/venv-fm/Scripts/python.exe -X utf8 scripts/extract_external_features.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = ROOT / "data" / "external" / "sources"
OUT = ROOT / "data" / "external" / "dino_features.npz"
DINO = "facebook/dinov2-small"
UNIFORM_BACKGROUND_SOURCES = ("mendeley_tomato_field_unlabeled", "mendeley_chilli_bd", "mendeley_tomato_bangladesh", "mendeley_orange_hlb",
                              "mendeley_maize_bd_healthy", "mendeley_grape_leaf_18k", "mendeley_peach_pakistan")


def load_cache() -> dict[str, np.ndarray]:
    if not OUT.is_file():
        return {}
    data = np.load(OUT, allow_pickle=False)
    return {str(k): v for k, v in zip(data["keys"], data["feats"])}


def save_cache(cache: dict[str, np.ndarray]) -> None:
    tmp = OUT.with_suffix(".tmp.npz")
    np.savez(tmp, keys=np.array(list(cache)), feats=np.stack(list(cache.values())).astype(np.float16))
    tmp.replace(OUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sources", nargs="*", default=list(UNIFORM_BACKGROUND_SOURCES), help="tên thư mục nguồn cần trích; rỗng = tất cả")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    spec = importlib.util.spec_from_file_location("image_io_standalone", ROOT / "src" / "common" / "image_io.py")
    image_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_io)

    cache = load_cache()
    todo = []
    for manifest in sorted(SOURCES.glob("*/manifest.jsonl")):
        if args.sources and manifest.parent.name not in args.sources:
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["sample_id"] not in cache:
                    todo.append((row["sample_id"], SOURCES / row["relative_path"]))
    print(f"[dino] có sẵn {len(cache)}, cần thêm {len(todo)} ảnh", flush=True)
    if not todo:
        return 0
    torch.set_num_threads(max(1, args.threads))
    processor = AutoImageProcessor.from_pretrained(DINO)
    model = AutoModel.from_pretrained(DINO).eval()
    started, last_save = time.time(), time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        images, ids = [], []
        for sample_id, path in chunk:
            try:
                with Image.open(path) as handle:
                    images.append(image_io.to_rgb(handle).copy())
                ids.append(sample_id)
            except Exception:  # noqa: BLE001 - ảnh hỏng: bỏ qua, nơi dùng coi như không có đặc trưng
                continue
        if not images:
            continue
        with torch.inference_mode():
            feats = model(**processor(images=images, return_tensors="pt")).last_hidden_state[:, 0].numpy()
        for sample_id, feat in zip(ids, feats):
            cache[sample_id] = feat
        done = start + len(chunk)
        if time.time() - last_save > 120:
            save_cache(cache)
            last_save = time.time()
        if done % (args.batch * 10) < args.batch or done == len(todo):
            rate = done / max(time.time() - started, 1e-9)
            print(f"[dino] {done}/{len(todo)} ({rate:.1f} ảnh/s, còn ~{(len(todo) - done) / max(rate, 1e-9) / 60:.1f} phút)", flush=True)
    save_cache(cache)
    print(f"[dino] xong, {len(cache)} đặc trưng -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
