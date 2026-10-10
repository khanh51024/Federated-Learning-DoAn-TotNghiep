"""Standalone inference for the repo's trained PlantVillage TorchScript reference.

This is an independent reference, NOT a FedAvg checkpoint or scientifically
comparable validation run. Confidence is raw softmax, not calibrated accuracy.
"""
from functools import lru_cache
from pathlib import Path
import json

import torch
from PIL import Image, ImageOps
from src import registry as reg
from src.data.plantvillage import get_default_transform

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


def predict_plantvillage(image: Image.Image) -> dict:
    model, classes = _load()
    # Reuse exactly the held-out evaluation transform used by the training pipeline.
    preprocessing = get_default_transform(is_train=False)
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
        "model_name": "PlantVillage TorchScript (reference)",
        "predicted_class": top3[0]["class"],
        "confidence_pct": top3[0]["confidence"],
        "top3": top3,
        "is_uncertain": bool(values[0].item() < 0.5 or (values[0] - values[1]).item() < 0.2),
        "confidence_calibrated": False,
        "checkpoint": MODEL_FILE.name,
        "scope": "PlantVillage 38 labels; not verified on field photos",
    }
