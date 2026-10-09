"""Standardized and versioned image transforms for training and evaluation.

Adheres strictly to Gate B specification:
  - Canonical square resize (224x224, bilinear, antialias=True)
  - Aspect-preserving letterbox padding to (224x224)
  - Controlled train-only augmentations: horizontal flip, mild rotation (+-12 deg), gentle brightness/contrast jitter (hue=0.0)
  - Zero stochastic transformations during validation/evaluation (100% deterministic)
  - Full metadata recording of geometry, interpolation, normalization, and compositing policy
"""

from io import BytesIO
import random
from typing import Any, Dict, Tuple, Union
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_RGB = tuple(round(value * 255) for value in IMAGENET_MEAN)


class ResizeKeepRatioPad:
    """Resize PIL Image preserving aspect ratio, then pad with black to (target_size, target_size)."""

    def __init__(self, target_size: int = IMAGE_SIZE, fill: Union[int, Tuple[int, int, int]] = 0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.target_size, self.target_size), self.fill)

        # Scale factor
        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize preserving ratio with antialiased bilinear interpolation
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Pad to target_size x target_size
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(resized, padding, fill=self.fill)

    def __repr__(self) -> str:
        return f"ResizeKeepRatioPad(target_size={self.target_size}, fill={self.fill})"


class RandomJpegCompression:
    """Apply low-probability JPEG recompression without changing geometry."""

    def __init__(self, probability: float = 0.08, quality: Tuple[int, int] = (82, 95)):
        self.probability = float(probability)
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        quality = random.randint(*self.quality)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

    def __repr__(self) -> str:
        return f"RandomJpegCompression(probability={self.probability}, quality={self.quality})"


TRANSFORM_METADATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "canonical_v1": {
        "version": "canonical_v1",
        "geometry": "direct_resize_square",
        "target_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "compositing_background": "RGB(0,0,0)",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "space": "ImageNet1K",
        },
        "train_augmentations": [
            {"type": "RandomHorizontalFlip", "p": 0.5},
        ],
        "eval_augmentations": [],
    },
    "aspect_pad_v1": {
        "version": "aspect_pad_v1",
        "geometry": "aspect_preserving_letterbox_pad",
        "target_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "compositing_background": "RGB(0,0,0)",
        "padding_fill": [0, 0, 0],
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "space": "ImageNet1K",
        },
        "train_augmentations": [
            {"type": "RandomHorizontalFlip", "p": 0.5},
        ],
        "eval_augmentations": [],
    },
    "aspect_pad_mean_v2": {
        "version": "aspect_pad_mean_v2",
        "geometry": "aspect_preserving_letterbox_pad",
        "target_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "compositing_background": "ImageNet_mean_RGB",
        "padding_fill": list(IMAGENET_MEAN_RGB),
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "space": "ImageNet1K"},
        "train_augmentations": [{"type": "RandomHorizontalFlip", "p": 0.5}],
        "eval_augmentations": [],
    },
    "moderate_augment_v1": {
        "version": "moderate_augment_v1",
        "geometry": "direct_resize_square",
        "target_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "compositing_background": "RGB(0,0,0)",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "space": "ImageNet1K",
        },
        "train_augmentations": [
            {"type": "RandomHorizontalFlip", "p": 0.5},
            {"type": "RandomRotation", "degrees": [-12, 12]},
            {"type": "ColorJitter", "brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.0},
        ],
        "eval_augmentations": [],
    },
    "light_augment_v2": {
        "version": "light_augment_v2",
        "geometry": "direct_resize_square",
        "target_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "compositing_background": "RGB(0,0,0)",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "space": "ImageNet1K",
        },
        "train_augmentations": [
            {"type": "RandomHorizontalFlip", "p": 0.5},
            {"type": "RandomRotation", "degrees": [-8, 8]},
            {"type": "ColorJitter", "brightness": 0.08, "contrast": 0.08, "saturation": 0.04, "hue": 0.0},
            {"type": "GaussianBlur", "kernel_size": 3, "sigma": [0.1, 0.6], "p": 0.08},
            {"type": "RandomJpegCompression", "quality": [82, 95], "p": 0.08},
        ],
        "eval_augmentations": [],
    },
}


def get_transform_metadata(transform_name: str = "canonical_v1") -> Dict[str, Any]:
    """Retrieve full metadata for a versioned transform pipeline."""
    if transform_name not in TRANSFORM_METADATA_REGISTRY:
        raise ValueError(
            f"Unknown transform version: {transform_name}. "
            f"Supported: {list(TRANSFORM_METADATA_REGISTRY.keys())}"
        )
    return TRANSFORM_METADATA_REGISTRY[transform_name].copy()


def get_transforms(
    transform_name: str = "canonical_v1",
    is_train: bool = False,
    image_size: int = IMAGE_SIZE,
) -> T.Compose:
    """Construct versioned transform pipeline."""
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if transform_name == "canonical_v1":
        if is_train:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                norm,
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToTensor(),
                norm,
            ])

    elif transform_name in ("aspect_pad_v1", "aspect_pad_mean_v2"):
        fill = 0 if transform_name == "aspect_pad_v1" else IMAGENET_MEAN_RGB
        pad_op = ResizeKeepRatioPad(target_size=image_size, fill=fill)
        if is_train:
            return T.Compose([
                pad_op,
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                norm,
            ])
        else:
            return T.Compose([
                pad_op,
                T.ToTensor(),
                norm,
            ])

    elif transform_name == "moderate_augment_v1":
        if is_train:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=(-12, 12)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
                T.ToTensor(),
                norm,
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.ToTensor(),
                norm,
            ])

    elif transform_name == "light_augment_v2":
        if is_train:
            return T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=(-8, 8)),
                T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.0),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.08),
                RandomJpegCompression(probability=0.08, quality=(82, 95)),
                T.ToTensor(),
                norm,
            ])
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            norm,
        ])

    else:
        raise ValueError(f"Unknown transform pipeline: {transform_name}")


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Backward compatibility evaluation transform."""
    return get_transforms("canonical_v1", is_train=False, image_size=image_size)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Backward compatibility training transform."""
    return get_transforms("canonical_v1", is_train=True, image_size=image_size)
