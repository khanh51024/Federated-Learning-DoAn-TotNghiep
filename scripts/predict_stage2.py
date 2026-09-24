"""
Inference script for published Stage 2 Scratch-v4 global FedAvg models.
Loads a MobileNetV3-Small checkpoint (38 PlantVillage classes) safely with weights_only=True,
applies the canonical EVAL_TRANSFORM (RGB, resize 224x224, ImageNet normalization),
and produces top-k predicted classes with softmax probabilities.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Predict with a published scratch-v4 global model (38 PlantVillage classes)."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the checkpoint (.pth) file",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to an input image (JPEG/PNG)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to repository root (defaults to parent of scripts/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top class predictions to output (1-38)",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    code_dir = repo_root / "training" / "stage2_scratch_v4"
    if not (code_dir / "stage1_compat" / "models.py").is_file():
        raise RuntimeError(f"Missing published training code at {code_dir}")

    # Ensure training code is prioritized and imported from namespaced stage2_scratch_v4
    sys.path.insert(0, str(code_dir))

    import torch
    import stage1_compat
    from PIL import Image
    from stage1_compat.models import create_mobilenetv3_stage1
    from stage1_compat.data import EVAL_TRANSFORM
    from stage1_compat.checkpoint import compute_state_dict_sha256

    # Verify import origin to prevent accidental fallback to legacy root modules
    stage1_file = Path(stage1_compat.__file__).resolve()
    if not stage1_file.is_relative_to(code_dir):
        raise RuntimeError(
            f"Unexpected import origin for stage1_compat: {stage1_file} (expected inside {code_dir})"
        )

    # Safe weights_only load
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    names = checkpoint.get("class_names", [])
    if len(names) != 38:
        raise ValueError(f"Expected 38 classes, found {len(names)} in checkpoint")
    if not (1 <= args.top_k <= len(names)):
        raise ValueError(f"top-k must be in range 1..{len(names)}, got {args.top_k}")

    # Validate state dict hash against embedded checkpoint metadata
    expected_state_sha = checkpoint.get("state_sha256")
    actual_state_sha = compute_state_dict_sha256(checkpoint["model_state"])
    if actual_state_sha != expected_state_sha:
        raise ValueError(
            f"State hash mismatch: actual {actual_state_sha} != expected {expected_state_sha}"
        )

    # Instantiate model architecture (MobileNetV3-Small scratch, 38 classes)
    model = create_mobilenetv3_stage1(num_classes=len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    # Preprocessing: convert to RGB and apply standard evaluation transforms (resize 224, normalize)
    with Image.open(args.image) as img:
        tensor = EVAL_TRANSFORM(img.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = logits.softmax(dim=1)[0]

    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite prediction encountered")

    values, indices = probabilities.topk(args.top_k)
    output = {
        "model": str(args.model),
        "best_round": checkpoint.get("best_round"),
        "selection_metric": checkpoint.get("selection_metric", "validation_accuracy"),
        "predictions": [
            {"class": names[idx], "probability": float(prob)}
            for prob, idx in zip(values.tolist(), indices.tolist())
        ],
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
