"""Kiểm khô tăng cường ảnh mạnh của lần train sau: vẽ lưới ảnh và chạy thử một batch qua MobileNetV3-Small trên CPU.

KHÔNG train. Mục đích: nhìn bằng mắt rằng phép biến đổi giữ được vết bệnh trên lá, và chắc rằng pipeline (đọc ảnh, tăng cường,
chuẩn hóa, forward, loss) chạy hết một batch.

Chạy:
    python scripts/preview_augmentations.py --images web/static/samples/demo --out tmp/aug_preview.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="web/static/samples/demo")
    parser.add_argument("--out", default="tmp/aug_preview.png")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    from src.data.robust_transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform, get_robust_train_transform, seed_everything
    from src.common.image_io import to_rgb

    folder = (ROOT / args.images).resolve()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.count]
    if not files:
        print(f"[aug] không có ảnh trong {folder}", file=sys.stderr)
        return 1
    train_t, eval_t = get_robust_train_transform(), get_eval_transform()
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def to_pil(tensor):
        return Image.fromarray((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy())

    seed_everything(args.seed)
    rows, batch = [], []
    for path in files:
        with Image.open(path) as handle:
            img = to_rgb(handle)
        tiles = [to_pil(eval_t(img))] + [to_pil(train_t(img)) for _ in range(args.variants)]
        batch += [train_t(img) for _ in range(2)]
        rows.append(np.concatenate([np.asarray(t.resize((112, 112))) for t in tiles], axis=1))
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print(f"[aug] đã vẽ {len(files)} ảnh x (1 gốc + {args.variants} biến thể): {out}")

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    x = torch.stack(batch)
    loss = nn.functional.cross_entropy(model(x), torch.zeros(len(batch), dtype=torch.long))
    loss.backward()
    print(f"[aug] một batch {tuple(x.shape)} qua MobileNetV3-Small: loss={float(loss):.3f}, gradient hữu hạn="
          f"{all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
