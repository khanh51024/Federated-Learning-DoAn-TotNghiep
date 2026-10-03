"""Evaluate completed 38-class checkpoints on the same canonical manifests.

Validation is the shared comparison split. The legacy diagnostic test is reported
separately and must not be described as a clean unseen external holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "plant_data_contract"))
from plant_data_contract.dataset import CanonicalClassificationDataset  # noqa: E402
from plant_data_contract.quality_metrics import summarize_quality  # noqa: E402
from plant_data_contract.taxonomy import PLANTVILLAGE_38_CLASSES  # noqa: E402
from plant_data_contract.transforms import get_transforms  # noqa: E402

RELEASE_SHA = "6d2c6b406b329e5016b3244c0079d1ef35103af5e73efe5b0f39c7ce1943d252"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    # Checkpoints here are produced by our own training runs. The loader accepts
    # their Python metadata and verifies the dataset release before inference.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    if payload.get("release_manifest_sha256") != RELEASE_SHA:
        raise ValueError(f"Dataset release mismatch: {path}")
    state = next((payload[key] for key in ("model_state", "model_state_dict", "state_dict")
                  if key in payload), None)
    if not isinstance(state, dict):
        raise ValueError(f"Model state missing: {path}")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def collate(batch):
    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch]), [item[2] for item in batch]


def scores_for_group(targets: np.ndarray, preds: np.ndarray, conf: np.ndarray) -> dict:
    support = sorted(set(targets.tolist()))
    if not support:
        return {"count": 0}
    crops = [label.split("___", 1)[0] for label in PLANTVILLAGE_38_CLASSES]
    return {
        "count": len(targets),
        "supported_classes": len(support),
        "top1_accuracy": float((targets == preds).mean()),
        "macro_f1_supported": float(f1_score(targets, preds, labels=support, average="macro", zero_division=0)),
        "crop_accuracy": float(np.mean([crops[t] == crops[p] for t, p in zip(targets, preds)])),
        "high_confidence_wrong_count_0_9": int(((targets != preds) & (conf >= 0.9)).sum()),
    }


def evaluate(model: nn.Module, device: torch.device, dataset_root: Path, manifest: Path,
             batch_size: int) -> dict:
    data = CanonicalClassificationDataset.from_manifest(
        manifest, dataset_root, transform=get_transforms("canonical_v1", is_train=False),
        return_metadata=True,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    targets, preds, conf, domains = [], [], [], []
    with torch.inference_mode():
        for images, labels, metadata in loader:
            probabilities = model(images.to(device)).softmax(dim=1).cpu()
            top_conf, top_idx = probabilities.max(dim=1)
            targets.extend(labels.tolist())
            preds.extend(top_idx.tolist())
            conf.extend(top_conf.tolist())
            domains.extend(item.get("source_domain", "plantvillage") for item in metadata)
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    c = np.asarray(conf, dtype=np.float64)
    d = np.asarray(domains)
    wrong_pairs = Counter((int(t), int(pred)) for t, pred in zip(y, p) if t != pred)
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "overall": scores_for_group(y, p, c),
        "by_source": {source: scores_for_group(y[d == source], p[d == source], c[d == source])
                      for source in ("plantdoc", "plantvillage")},
        "quality": summarize_quality(y, p, d, c),
        "top_confusions": [
            {"target": PLANTVILLAGE_38_CLASSES[t], "predicted": PLANTVILLAGE_38_CLASSES[pred], "count": count}
            for (t, pred), count in wrong_pairs.most_common(20)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-legacy-diagnostic-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    release = args.dataset_root / "mixed/pv_pd_v3"
    if sha256(release / "release_manifest.json") != RELEASE_SHA:
        raise ValueError("Wrong canonical release")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    model, payload = load_model(args.checkpoint, device)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_round_or_epoch": payload.get("round", payload.get("epoch")),
        "release_manifest_sha256": RELEASE_SHA,
        "validation": evaluate(model, device, args.dataset_root, release / "manifests/val.jsonl", args.batch_size),
    }
    if args.include_legacy_diagnostic_test:
        result["legacy_diagnostic_test"] = evaluate(
            model, device, args.dataset_root, release / "manifests/legacy_diagnostic_test.jsonl", args.batch_size
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": result["checkpoint"], "validation": result["validation"]["by_source"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
