"""
Transform pipeline: MobileNetV3-ready preprocessing + per-client feature skew.

MobileNetV3 contract (torchvision `mobilenet_v3_small/large` defaults)
----------------------------------------------------------------------
  * input      : RGB, 224 x 224
  * layout     : CHW, float32
  * scaling    : /255
  * normalize  : mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
The global test set always uses this clean pipeline, so every baseline
(Centralized / Federated / Local-only) is scored on identical inputs.

Feature skew ("lệch đặc trưng", the third GĐ2 axis)
---------------------------------------------------
Each agricultural facility is modelled as having its own acquisition domain:
lighting, white balance, sensor quality. That is simulated by a deterministic
per-client photometric profile applied at load time.

Two properties matter for a thesis benchmark:

1. **Raw images are never modified.** The profile lives in the manifest and is
   applied on the fly, so the same partition can be replayed with feature skew
   on/off -- exactly what an ablation needs.
2. **Fully reproducible.** Every randomised step (noise, augmentation) draws
   from a seed derived from `(profile_seed, client_id, sample_index, epoch)`,
   never from the global `numpy.random` state. That keeps results identical
   across runs and safe under multi-worker DataLoaders, where a shared global
   RNG would either repeat or diverge between workers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:  # Pillow >= 9.1
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - older Pillow
    _RESAMPLE = Image.BILINEAR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MOBILENETV3_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Feature-skew intensity presets. Ranges are deliberately moderate: the point
# is to make clients distinguishable, not to destroy the lesion cues that
# MobileNetV3 has to learn.
FEATURE_SKEW_LEVELS: Dict[str, Dict[str, Any]] = {
    "none": {"enabled": False},
    "mild": {
        "enabled": True,
        "brightness": (0.92, 1.08),
        "contrast": (0.92, 1.08),
        "saturation": (0.90, 1.10),
        "temp_r": (0.96, 1.04),
        "temp_b": (0.96, 1.04),
        "blur_p": 0.3, "blur": (0.3, 0.5),
        "noise_p": 0.3, "noise": (1.0, 3.0),
    },
    "moderate": {
        "enabled": True,
        "brightness": (0.85, 1.15),
        "contrast": (0.85, 1.16),
        "saturation": (0.82, 1.20),
        "temp_r": (0.92, 1.08),
        "temp_b": (0.92, 1.08),
        "blur_p": 0.45, "blur": (0.3, 0.7),
        "noise_p": 0.45, "noise": (2.0, 5.0),
    },
    "strong": {
        "enabled": True,
        "brightness": (0.75, 1.28),
        "contrast": (0.75, 1.30),
        "saturation": (0.70, 1.35),
        "temp_r": (0.85, 1.15),
        "temp_b": (0.85, 1.15),
        "blur_p": 0.6, "blur": (0.4, 1.0),
        "noise_p": 0.6, "noise": (3.0, 8.0),
    },
}

# Facility archetypes give each client a human-readable name for the report.
FACILITY_ARCHETYPES = [
    "Bright Sunlight Open Field",
    "Overcast / Shaded Canopy",
    "Low-cost Mobile Camera",
    "Warm Greenhouse Sunset",
    "High-contrast Direct Lighting",
    "Morning Mist / High Moisture",
    "Diffused Plastic Tunnel",
    "Late Afternoon Golden Hour",
    "Standard Agronomic Field Station",
    "Overexposed Handheld Sensor",
]


class ClientFeatureProfile:
    """Photometric/acquisition characteristics of one facility (client)."""

    __slots__ = (
        "client_id", "name", "brightness_factor", "contrast_factor",
        "saturation_factor", "temp_shift_r", "temp_shift_b",
        "blur_radius", "noise_std", "level",
    )

    def __init__(
        self,
        client_id: int,
        name: str = "",
        brightness_factor: float = 1.0,
        contrast_factor: float = 1.0,
        saturation_factor: float = 1.0,
        temp_shift_r: float = 1.0,
        temp_shift_b: float = 1.0,
        blur_radius: float = 0.0,
        noise_std: float = 0.0,
        level: str = "moderate",
    ):
        self.client_id = client_id
        self.name = name or f"Facility_{client_id:02d}"
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.temp_shift_r = temp_shift_r
        self.temp_shift_b = temp_shift_b
        self.blur_radius = blur_radius
        self.noise_std = noise_std
        self.level = level

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.brightness_factor - 1.0) < 1e-3
            and abs(self.contrast_factor - 1.0) < 1e-3
            and abs(self.saturation_factor - 1.0) < 1e-3
            and abs(self.temp_shift_r - 1.0) < 1e-3
            and abs(self.temp_shift_b - 1.0) < 1e-3
            and self.blur_radius <= 0.1
            and self.noise_std <= 0.5
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": int(self.client_id),
            "name": self.name,
            "level": self.level,
            "brightness_factor": round(float(self.brightness_factor), 4),
            "contrast_factor": round(float(self.contrast_factor), 4),
            "saturation_factor": round(float(self.saturation_factor), 4),
            "temp_shift_r": round(float(self.temp_shift_r), 4),
            "temp_shift_b": round(float(self.temp_shift_b), 4),
            "blur_radius": round(float(self.blur_radius), 4),
            "noise_std": round(float(self.noise_std), 4),
        }

    def __repr__(self) -> str:
        return f"ClientFeatureProfile(id={self.client_id}, level='{self.level}', name='{self.name}')"


def create_deterministic_client_profile(
    client_id: int,
    seed: int = 42,
    level: str = "moderate",
) -> ClientFeatureProfile:
    """Deterministically derive a facility profile from (client_id, seed, level)."""
    level_key = str(level).lower()
    if level_key not in FEATURE_SKEW_LEVELS:
        raise ValueError(
            f"Unknown feature_skew level '{level}'. "
            f"Supported: {sorted(FEATURE_SKEW_LEVELS)}"
        )
    preset = FEATURE_SKEW_LEVELS[level_key]
    archetype = FACILITY_ARCHETYPES[client_id % len(FACILITY_ARCHETYPES)]
    name = f"{archetype} (Client {client_id:02d})"

    if not preset["enabled"]:
        return ClientFeatureProfile(client_id=client_id, name=name, level=level_key)

    rng = np.random.default_rng(seed * 1_000_003 + client_id * 1001)
    blur_radius = float(rng.uniform(*preset["blur"])) if rng.random() < preset["blur_p"] else 0.0
    noise_std = float(rng.uniform(*preset["noise"])) if rng.random() < preset["noise_p"] else 0.0

    return ClientFeatureProfile(
        client_id=client_id,
        name=name,
        brightness_factor=float(rng.uniform(*preset["brightness"])),
        contrast_factor=float(rng.uniform(*preset["contrast"])),
        saturation_factor=float(rng.uniform(*preset["saturation"])),
        temp_shift_r=float(rng.uniform(*preset["temp_r"])),
        temp_shift_b=float(rng.uniform(*preset["temp_b"])),
        blur_radius=blur_radius,
        noise_std=noise_std,
        level=level_key,
    )


class BaseTransform:
    """MobileNetV3 train/eval preprocessing returning CHW float32."""

    def __init__(
        self,
        image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE,
        normalize: bool = True,
        augment: bool = False,
        crop_scale: Tuple[float, float] = (0.6, 1.0),
        hflip_p: float = 0.5,
    ):
        self.image_size = tuple(image_size)
        self.normalize = normalize
        self.augment = augment
        self.crop_scale = crop_scale
        self.hflip_p = hflip_p
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __call__(self, img: Image.Image, rng_seed: Optional[int] = None) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")

        rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

        if self.augment and rng is not None:
            # RandomResizedCrop + horizontal flip: the standard MobileNetV3
            # training augmentation, made deterministic via rng_seed.
            w, h = img.size
            scale = float(rng.uniform(*self.crop_scale))
            aspect = float(np.exp(rng.uniform(np.log(0.75), np.log(1.3333))))
            crop_w = int(round(min(w, h * aspect) * (scale ** 0.5)))
            crop_h = int(round(min(h, w / aspect) * (scale ** 0.5)))
            crop_w = max(1, min(crop_w, w))
            crop_h = max(1, min(crop_h, h))
            left = int(rng.integers(0, w - crop_w + 1))
            top = int(rng.integers(0, h - crop_h + 1))
            img = img.crop((left, top, left + crop_w, top + crop_h))
            if rng.random() < self.hflip_p:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.augment:
            if img.size != self.image_size:
                img = img.resize(self.image_size, _RESAMPLE)
        else:
            # Torchvision MobileNetV3 weights: resize short side to 256, then
            # center-crop 224. Scale the 256 value for custom crop sizes.
            crop_w, crop_h = self.image_size
            target_short = int(round(256 * min(crop_w, crop_h) / 224))
            w, h = img.size
            scale = target_short / min(w, h)
            resized = (max(crop_w, int(round(w * scale))), max(crop_h, int(round(h * scale))))
            img = img.resize(resized, _RESAMPLE)
            left = max(0, (img.width - crop_w) // 2)
            top = max(0, (img.height - crop_h) // 2)
            img = img.crop((left, top, left + crop_w, top + crop_h))

        arr = np.asarray(img, dtype=np.float32)
        arr /= 255.0
        if self.normalize:
            arr = (arr - self.mean) / self.std
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


class ClientDomainTransform:
    """Per-client feature skew, then the standard MobileNetV3 pipeline."""

    def __init__(
        self,
        profile: ClientFeatureProfile,
        base_transform: Optional[Union[BaseTransform, Callable[..., Any]]] = None,
        image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE,
        augment: bool = False,
    ):
        self.profile = profile
        self.base_transform = base_transform or BaseTransform(
            image_size=image_size, augment=augment
        )

    def __call__(self, img: Image.Image, rng_seed: Optional[int] = None) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        p = self.profile

        if not p.is_identity:
            if abs(p.brightness_factor - 1.0) > 1e-3:
                img = ImageEnhance.Brightness(img).enhance(p.brightness_factor)
            if abs(p.contrast_factor - 1.0) > 1e-3:
                img = ImageEnhance.Contrast(img).enhance(p.contrast_factor)
            if abs(p.saturation_factor - 1.0) > 1e-3:
                img = ImageEnhance.Color(img).enhance(p.saturation_factor)

            if abs(p.temp_shift_r - 1.0) > 1e-3 or abs(p.temp_shift_b - 1.0) > 1e-3:
                r, g, b = img.split()
                if abs(p.temp_shift_r - 1.0) > 1e-3:
                    r = r.point(lambda i, k=p.temp_shift_r: min(255, int(i * k)))
                if abs(p.temp_shift_b - 1.0) > 1e-3:
                    b = b.point(lambda i, k=p.temp_shift_b: min(255, int(i * k)))
                img = Image.merge("RGB", (r, g, b))

            if p.blur_radius > 0.1:
                img = img.filter(ImageFilter.GaussianBlur(radius=p.blur_radius))

            if p.noise_std > 0.5:
                arr = np.asarray(img, dtype=np.float32)
                # Seeded per sample -> reproducible and DataLoader-worker safe.
                noise_rng = np.random.default_rng(
                    None if rng_seed is None else rng_seed + 7919
                )
                noise = noise_rng.normal(0.0, p.noise_std, arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
                img = Image.fromarray(arr)

        return self.base_transform(img, rng_seed=rng_seed)


# --------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------
def get_default_transform(
    image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE,
    augment: bool = False,
) -> BaseTransform:
    """Clean MobileNetV3 pipeline -- used for the global test set."""
    return BaseTransform(image_size=image_size, augment=augment)


def get_mobilenetv3_transforms(
    image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE,
) -> Dict[str, Callable]:
    """Convenience pair matching torchvision's MobileNetV3 train/eval recipes."""
    return {
        "train": BaseTransform(image_size=image_size, augment=True),
        "eval": BaseTransform(image_size=image_size, augment=False),
    }


def get_client_transform(
    client_id: int,
    seed: int = 42,
    image_size: Tuple[int, int] = MOBILENETV3_IMAGE_SIZE,
    level: str = "moderate",
    augment: bool = False,
    custom_profile: Optional[ClientFeatureProfile] = None,
) -> ClientDomainTransform:
    """Client-specific feature-skew transform."""
    profile = custom_profile or create_deterministic_client_profile(
        client_id, seed=seed, level=level
    )
    return ClientDomainTransform(
        profile=profile, image_size=image_size, augment=augment
    )


def profiles_from_config(
    profiles: Optional[list],
    num_clients: int,
    seed: int = 42,
    level: str = "moderate",
) -> list:
    """
    Rebuild profiles from the values stored in `partition_config.json` when
    available (so a replay is byte-identical to the recorded run), otherwise
    regenerate them deterministically.
    """
    if profiles:
        out = []
        for d in profiles:
            out.append(ClientFeatureProfile(
                client_id=int(d["client_id"]),
                name=d.get("name", ""),
                brightness_factor=float(d.get("brightness_factor", 1.0)),
                contrast_factor=float(d.get("contrast_factor", 1.0)),
                saturation_factor=float(d.get("saturation_factor", 1.0)),
                temp_shift_r=float(d.get("temp_shift_r", 1.0)),
                temp_shift_b=float(d.get("temp_shift_b", 1.0)),
                blur_radius=float(d.get("blur_radius", 0.0)),
                noise_std=float(d.get("noise_std", 0.0)),
                level=d.get("level", level),
            ))
        return out
    return [
        create_deterministic_client_profile(c, seed=seed, level=level)
        for c in range(num_clients)
    ]
