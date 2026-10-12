"""Huấn luyện detector lá (một lớp "leaf") để cắt ảnh trước bước chẩn đoán.

Vì sao: bộ phân loại chỉ quen ảnh MỘT lá đã cắt gọn (crop_policy bbox_margin_8pct_v1). Ảnh người dùng tải lên thường
là cả cây / nhiều lá / nền rối. `scripts/eval_fullframe_plantdoc.py` cho thấy đưa cả khung vào làm tụt độ chính xác;
detector là bước đưa ảnh về đúng dạng mô hình đã học. Chạy trên máy có GPU (CPU chỉ để thử đường ống).

Quy tắc chống rò rỉ: CHỈ học trên ảnh có split == "train" trong detection.json của release. Mọi cảnh thuộc validation /
calibration / test / quarantine bị loại — nên các cảnh dùng để đo bộ phân loại không lọt vào detector.

Mô hình: torchvision Faster R-CNN MobileNetV3-Large FPN (BSD-3, không thêm thư viện), 2 lớp (nền + lá).

Chạy:
    python scripts/train_leaf_detector.py --download --epochs 12 --device cuda --pretrained-backbone
    python scripts/train_leaf_detector.py --limit 12 --epochs 1 --device cpu --min-size 320   # thử đường ống
Kết quả (không ghi vào artifacts/ khi --limit): data/leaf_detector/leaf_detector.pt + report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from scripts.eval_fullframe_plantdoc import (  # noqa: E402
    CACHE_DIR, download_missing, scene_path, sha256_file,
)

OUT_DIR = "data/leaf_detector"  # gitignored (data/*)
DETECTOR_FORMAT = "leaf_detector_v1"


# --------------------------------------------------------------------------- phần thuần (có test)
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp khung xyxy: (N,4) x (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_boxes(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou: float = 0.5, score_thr: float = 0.5) -> dict[str, int]:
    """Ghép tham lam theo điểm giảm dần: trả số khung đúng (tp), thừa (fp), bỏ sót (fn)."""
    keep = scores >= score_thr
    pred, scores = pred[keep], scores[keep]
    order = np.argsort(-scores)
    pred = pred[order]
    taken: set[int] = set()
    tp = 0
    matrix = iou_matrix(pred, gt)
    for row in range(len(pred)):
        candidates = [(matrix[row, j], j) for j in range(len(gt)) if j not in taken and matrix[row, j] >= iou]
        if candidates:
            taken.add(max(candidates)[1])
            tp += 1
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def training_image_ids(images: dict[str, dict[str, Any]]) -> list[str]:
    """Chỉ ảnh split == "train" (chống rò rỉ sang val/calibration/test/quarantine)."""
    return sorted(i for i, im in images.items() if im["split"] == "train")


# --------------------------------------------------------------------------- dữ liệu / mô hình
def build_model(pretrained_backbone: bool, min_size: int, max_size: int):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    kwargs: dict[str, Any] = {"weights": None, "num_classes": 2, "min_size": min_size, "max_size": max_size}
    if pretrained_backbone:
        kwargs["weights_backbone"] = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        kwargs["trainable_backbone_layers"] = 3
    else:
        kwargs["weights_backbone"] = None
    return fasterrcnn_mobilenet_v3_large_fpn(**kwargs)


def _load_rgb(path: Path):
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:  # noqa: BLE001
            pass
        return image.convert("RGB")


class SceneDataset:
    def __init__(self, image_ids: list[str], images: dict, boxes: dict[str, list], cache: Path, train: bool):
        self.ids, self.images, self.boxes, self.cache, self.train = image_ids, images, boxes, cache, train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        import torch
        import torchvision.transforms.functional as F

        image_id = self.ids[index]
        image = _load_rgb(scene_path(self.cache, self.images[image_id]))
        width, height = image.size
        boxes = np.array(self.boxes.get(image_id, []), dtype=np.float32).reshape(-1, 4)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        keep = (boxes[:, 2] - boxes[:, 0] >= 4) & (boxes[:, 3] - boxes[:, 1] >= 4)
        boxes = boxes[keep]
        tensor = F.to_tensor(image)
        if self.train:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
                boxes = boxes.copy()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            tensor = F.adjust_brightness(tensor, random.uniform(0.85, 1.15))
            tensor = F.adjust_contrast(tensor, random.uniform(0.85, 1.15))
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return tensor, target


def _collate(batch):
    return tuple(zip(*batch))


def evaluate_detector(model, dataset: SceneDataset, device: str, score_thr: float, iou: float = 0.5) -> dict[str, Any]:
    import torch

    model.eval()
    totals = {"tp": 0, "fp": 0, "fn": 0}
    with torch.inference_mode():
        for index in range(len(dataset)):
            tensor, target = dataset[index]
            out = model([tensor.to(device)])[0]
            result = match_boxes(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), target["boxes"].numpy(), iou, score_thr)
            for key in totals:
                totals[key] += result[key]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {"images": len(dataset), "score_threshold": score_thr, "iou": iou, **totals,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None, "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def load_detector(path: Path, device: str = "cpu"):
    """Nạp detector đã huấn luyện (kiểm định dạng; weights_only=True)."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != DETECTOR_FORMAT:
        raise ValueError(f"{path}: không phải {DETECTOR_FORMAT}")
    model = build_model(False, int(payload["min_size"]), int(payload["max_size"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="tải ảnh gốc PlantDoc còn thiếu (kiểm SHA-256)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--pretrained-backbone", action="store_true", help="khởi tạo backbone từ ImageNet (cần tải trọng số torchvision)")
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh học và N ảnh đo (thử đường ống)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    release = reg.abs_path(reg.DATASET_SOURCES_DIR) / "mixed" / "pv_pd_v3"
    detection_path = release / "annotations" / "detection.json"
    if not detection_path.is_file():
        print("[detector] thiếu data/dataset_sources/mixed/pv_pd_v3 (nhánh dataset-sources)", file=sys.stderr)
        return 1
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    images = {i["image_id"]: i for i in detection["images"]}
    boxes: dict[str, list] = {}
    for ann in detection["annotations"]:
        if ann.get("detection_eligible", True):
            boxes.setdefault(ann["image_id"], []).append(ann["bbox_xyxy"])
    train_ids = [i for i in training_image_ids(images) if i in boxes]
    eval_ids = sorted(i for i, im in images.items() if im["split"] in ("validation", "test") and i in boxes)
    if args.limit:
        train_ids, eval_ids = train_ids[: args.limit], eval_ids[: args.limit]
    assert not set(train_ids) & set(eval_ids), "rò rỉ: cảnh vừa học vừa đo"

    cache = reg.abs_path(CACHE_DIR)
    if args.download:
        print("[detector] ảnh gốc:", download_missing(cache, images, set(train_ids) | set(eval_ids)))
    missing = [i for i in train_ids + eval_ids if not scene_path(cache, images[i]).is_file()]
    if missing:
        print(f"[detector] thiếu {len(missing)} ảnh gốc; chạy với --download", file=sys.stderr)
        return 1
    bad = [i for i in train_ids + eval_ids if sha256_file(scene_path(cache, images[i])) != images[i]["byte_sha256"]]
    if bad:
        print(f"[detector] {len(bad)} ảnh gốc sai SHA-256: dừng", file=sys.stderr)
        return 1

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[detector] không có CUDA", file=sys.stderr)
        return 1
    model = build_model(args.pretrained_backbone, args.min_size, args.max_size).to(device)
    train_set = SceneDataset(train_ids, images, boxes, cache, train=True)
    eval_set = SceneDataset(eval_ids, images, boxes, cache, train=False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch_images, batch_targets in loader:
            batch_images = [im.to(device) for im in batch_images]
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = sum(model(batch_images, batch_targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
        history.append({"epoch": epoch, "loss": round(running / max(1, len(loader)), 4), "seconds": round(time.time() - started, 1)})
        print(f"[detector] epoch {epoch}/{args.epochs} loss {history[-1]['loss']} ({history[-1]['seconds']}s)", flush=True)

    metrics = evaluate_detector(model, eval_set, device, args.score_threshold)
    print("[detector] đo trên cảnh val+test (không học):", json.dumps(metrics))
    out_dir = reg.abs_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "format": DETECTOR_FORMAT,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "min_size": args.min_size,
        "max_size": args.max_size,
        "score_threshold": args.score_threshold,
        "crop_margin": 0.08,
        "seed": args.seed,
        "train_images": len(train_ids),
        "train_image_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "release_name": release_manifest.get("release_name"),
        "detection_json_sha256": release_manifest.get("detection_json_sha256"),
        "epochs": args.epochs,
        "pretrained_backbone": bool(args.pretrained_backbone),
        "history": history,
        "metrics_heldout_scenes": metrics,
    }
    checkpoint = out_dir / "leaf_detector.pt"
    torch.save(payload, checkpoint)
    report = {k: v for k, v in payload.items() if k != "state_dict"}
    report["checkpoint_sha256"] = sha256_file(checkpoint)
    report["torch"], report["torchvision"] = torch.__version__, __import__("torchvision").__version__
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[detector] đã ghi {checkpoint} (SHA-256 {report['checkpoint_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
