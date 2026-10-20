"""Suy luận độc lập bằng mô hình PlantVillage tham chiếu (TorchScript của `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`).

Mô hình này KHÔNG do đồ án huấn luyện; đồ án chỉ đánh giá lại nó (scripts/evaluate_plantvillage_reference.py). Tiền xử lý là đúng cấu hình
của tác giả (resize 256 bilinear, cắt giữa 224, chuẩn hóa mean = std = 0,5): chuẩn hóa ImageNet làm độ chính xác trên validation
tụt từ 91,6% xuống 51,4%. Xác suất là softmax thô: nhiệt độ học trên calibration làm ECE chung xấu đi nên không áp dụng.
"""
from functools import lru_cache
from pathlib import Path
import json

import torch
from PIL import Image, ImageOps
from src import registry as reg
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = reg.abs_path(reg.PLANTVILLAGE_TRACED_PT)
MANIFEST_FILE = reg.abs_path(reg.DATASET_MANIFEST_JSON)


@lru_cache(maxsize=1)
def _load():
    classes = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))["classes"]
    if len(classes) != 38 or len(set(classes)) != 38:
        raise ValueError("PlantVillage class taxonomy must have 38 unique labels")
    model = torch.jit.load(str(MODEL_FILE), map_location="cpu")
    model.eval()
    return model, classes


@lru_cache(maxsize=1)
def _transform():
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def predict_plantvillage(image: Image.Image) -> dict:
    model, classes = _load()
    preprocessing = _transform()
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    tensor = preprocessing(rgb).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        if tuple(logits.shape) != (1, len(classes)):
            raise ValueError("PlantVillage model output shape differs from taxonomy")
        probabilities = torch.softmax(logits[0], dim=0)
        values, indices = probabilities.topk(3)
    top3 = [{"class": classes[i.item()], "confidence": round(v.item() * 100, 2)}
            for v, i in zip(values, indices)]
    return {
        "ready": True,
        "model_name": "PlantVillage MobileNetV2 (Hugging Face)",
        "predicted_class": top3[0]["class"],
        "confidence_pct": top3[0]["confidence"],
        "top3": top3,
        "is_uncertain": bool(values[0].item() < 0.5 or (values[0] - values[1]).item() < 0.2),
        "confidence_calibrated": False,
        "checkpoint": MODEL_FILE.name,
        "scope": "PlantVillage 38 nhóm; chính xác trên ảnh trong phòng, kém trên ảnh ngoài đồng (xem báo cáo đánh giá lại)",
    }
