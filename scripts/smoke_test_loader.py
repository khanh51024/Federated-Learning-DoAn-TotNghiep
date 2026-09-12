#!/usr/bin/env python
"""
Smoke test: prove a partition directory can actually drive a FedAvg run with a
MobileNetV3-shaped input contract, for all three GĐ2 regimes.

Runs WITHOUT torch (pure numpy/PIL) and, when torch is installed, additionally
performs one CPU forward pass through `mobilenet_v3_small(weights=None)`.

  python scripts/smoke_test_loader.py
  python scripts/smoke_test_loader.py --partition-dir data/partitions_v3/label_skew/<name> --probe 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data import TORCH_AVAILABLE, FedAvgPartition  # noqa: E402
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

EXPECTED_SHAPE = (3, 224, 224)


def describe(batch) -> str:
    images, labels = batch
    if TORCH_AVAILABLE and hasattr(images, "shape") and hasattr(images, "dtype"):
        return f"images{tuple(images.shape)} {images.dtype} labels{tuple(labels.shape)} {labels.dtype}"
    return f"images{tuple(np.asarray(images).shape)} {np.asarray(images).dtype} labels{tuple(np.asarray(labels).shape)}"


def first_batch(loader):
    for batch in loader:
        return batch
    raise AssertionError("loader yielded no batches")


def check_range(name: str, arr) -> None:
    a = arr.numpy() if TORCH_AVAILABLE and hasattr(arr, "numpy") else np.asarray(arr)
    lo, hi = float(a.min()), float(a.max())
    # After /255 and ImageNet normalisation the range is roughly [-2.64, 2.64].
    assert -4.0 < lo < hi < 4.0, f"{name}: values out of normalised range [{lo:.2f}, {hi:.2f}]"


def run(partition_dir: Path, probe: int) -> int:
    print("=" * 74)
    print(f"  FEDAVG / MOBILENETV3 READINESS SMOKE TEST")
    print(f"  partition: {partition_dir}")
    print(f"  torch    : {'available' if TORCH_AVAILABLE else 'NOT installed (numpy path)'}")
    print("=" * 74)

    part = FedAvgPartition(partition_dir, augment_train=False)
    print(f"\n{part!r}")
    s = part.summary()
    print(f"  n_k      : {s['n_k']}")
    print(f"  weights  : {s['weights']}  (sum={sum(s['weights']):.4f})")

    report = part.readiness_report(check_images=0)
    print(f"\n--- readiness_report() ---")
    for k in ("num_clients", "num_classes", "total_train_samples", "weights_sum",
              "scenario", "alpha", "feature_skew", "group_aware"):
        print(f"  {k:<22} {report[k]}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  ! {w}")
    if report["problems"]:
        for p in report["problems"]:
            print(f"  x {p}")
        print("\nSMOKE TEST FAILED")
        return 1

    # -- 1. Centralized upper bound ----------------------------------------
    print("\n[1] Centralized (union of all facilities)")
    ds = part.centralized_dataset()
    assert len(ds) == part.total_train_samples, (len(ds), part.total_train_samples)
    batch = first_batch(part.centralized_loader(batch_size=8))
    print(f"    len={len(ds)}  {describe(batch)}")
    assert tuple(np.asarray(batch[0][0]).shape) == EXPECTED_SHAPE, "shape != MobileNetV3 contract"
    check_range("centralized", batch[0][0])
    print("    -> PASS")

    # -- 2. Federated clients (feature skew on) ----------------------------
    print("\n[2] Federated clients (per-facility feature skew)")
    loaders = part.all_client_loaders(batch_size=8)
    assert len(loaders) == part.num_clients
    seen_means = []
    for cid in range(min(probe, part.num_clients)):
        ds_c = part.client_dataset(cid)
        assert len(ds_c) == int(part.n_samples[cid]), (cid, len(ds_c), part.n_samples[cid])
        b = first_batch(loaders[cid])
        img0 = b[0][0]
        arr = img0.numpy() if TORCH_AVAILABLE and hasattr(img0, "numpy") else np.asarray(img0)
        assert arr.shape == EXPECTED_SHAPE, f"client {cid}: shape {arr.shape}"
        check_range(f"client_{cid}", img0)
        seen_means.append(float(arr.mean()))
        prof = part.profiles[cid]
        print(f"    client_{cid:02d}  n_k={len(ds_c):<6d} w_k={part.client_weight(cid):.4f}  "
              f"batch={describe(b)}")
        print(f"                domain: {prof.name}  bright={prof.brightness_factor:.3f} "
              f"contrast={prof.contrast_factor:.3f} sat={prof.saturation_factor:.3f} "
              f"blur={prof.blur_radius:.2f} noise={prof.noise_std:.2f}")
    if part.feature_skew != "none" and len(set(round(m, 4) for m in seen_means)) == 1 and len(seen_means) > 1:
        print("    ! all clients produced identical means -- feature skew may not be applied")
    print("    -> PASS")

    # -- 3. Local-only lower bound -----------------------------------------
    print("\n[3] Local-only (same manifest as Federated, trained in isolation)")
    ds_local = part.client_dataset(0)
    b = first_batch(part.client_loader(0, batch_size=8, shuffle=True))
    print(f"    client_00 len={len(ds_local)}  {describe(b)}")
    print("    -> PASS (identical data to regime [2]; difference is aggregation, not data)")

    # -- 4. Global test set ------------------------------------------------
    print("\n[4] Global test (shared evaluation for all three regimes)")
    ds_t = part.global_test_dataset()
    b = first_batch(part.global_test_loader(batch_size=16))
    labels = b[1].numpy() if TORCH_AVAILABLE and hasattr(b[1], "numpy") else np.asarray(b[1])
    assert labels.min() >= 0 and labels.max() < part.num_classes, "test labels out of range"
    print(f"    len={len(ds_t)}  {describe(b)}  labels in [0,{part.num_classes})")
    print("    -> PASS")

    # -- 5. Feature-skew ablation (same manifests, skew off) ---------------
    print("\n[5] Ablation: replay the same partition with feature_skew='none'")
    part_off = FedAvgPartition(partition_dir, feature_skew="none", augment_train=False)
    b_off = first_batch(part_off.client_loader(0, batch_size=8, shuffle=False))
    b_on = first_batch(FedAvgPartition(partition_dir, augment_train=False)
                       .client_loader(0, batch_size=8, shuffle=False))
    a_on = b_on[0][0]
    a_off = b_off[0][0]
    a_on = a_on.numpy() if TORCH_AVAILABLE and hasattr(a_on, "numpy") else np.asarray(a_on)
    a_off = a_off.numpy() if TORCH_AVAILABLE and hasattr(a_off, "numpy") else np.asarray(a_off)
    identical = bool(np.allclose(a_on, a_off, atol=1e-6))
    if part.meta["partition"]["feature_skew"] == "none":
        print("    partition was generated with feature_skew='none' -> identical tensors expected")
        assert identical, "tensors differ although feature skew is disabled"
    else:
        print(f"    skew ON vs OFF differ: {not identical} "
              f"(mean |Δ| = {float(np.abs(a_on - a_off).mean()):.4f})")
        assert not identical, "feature skew had no effect on the pixels"
    print("    -> PASS")

    # -- 6. Determinism ----------------------------------------------------
    print("\n[6] Determinism: reloading the same partition twice")
    p1 = FedAvgPartition(partition_dir, augment_train=False)
    p2 = FedAvgPartition(partition_dir, augment_train=False)
    b1 = first_batch(p1.client_loader(0, batch_size=8, shuffle=True))
    b2 = first_batch(p2.client_loader(0, batch_size=8, shuffle=True))
    a1 = b1[0][0]
    a2 = b2[0][0]
    a1 = a1.numpy() if TORCH_AVAILABLE and hasattr(a1, "numpy") else np.asarray(a1)
    a2 = a2.numpy() if TORCH_AVAILABLE and hasattr(a2, "numpy") else np.asarray(a2)
    assert np.allclose(a1, a2, atol=1e-6), "same partition + same seed produced different batches"
    assert np.array_equal(p1.aggregation_weights, p2.aggregation_weights)
    print("    -> PASS")

    # -- 7. Real torchvision MobileNetV3 forward contract -----------------
    print("\n[7] torchvision MobileNetV3-Small CPU forward")
    if TORCH_AVAILABLE:
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
        except ImportError:
            print("    -> NOT RUN: torchvision is not installed")
        else:
            model = mobilenet_v3_small(weights=None, num_classes=part.num_classes).cpu().eval()
            model_batch = first_batch(part.global_test_loader(batch_size=2))[0].cpu()
            assert model_batch.dtype == torch.float32
            with torch.no_grad():
                logits = model(model_batch)
            assert tuple(logits.shape) == (len(model_batch), part.num_classes)
            assert bool(torch.isfinite(logits).all())
            print(f"    input={tuple(model_batch.shape)} output={tuple(logits.shape)} finite=True")
            print("    -> PASS")
    else:
        print("    -> NOT RUN: torch and torchvision are not installed")

    print("\n" + "=" * 74)
    if TORCH_AVAILABLE:
        print("  DATA LAYER PASSED -- see step [7] for torchvision forward status")
    else:
        print("  DATA LAYER PASSED -- MOBILENETV3 FORWARD NOT VERIFIED (dependencies missing)")
    print("=" * 74)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FedAvg/MobileNetV3 readiness smoke test.")
    ap.add_argument("--partition-dir", type=str, default=None)
    ap.add_argument("--probe", type=int, default=3, help="How many clients to load.")
    args = ap.parse_args()

    if args.partition_dir:
        target = Path(args.partition_dir)
    else:
        candidates = sorted((ROOT / "data" / "partitions_v3").rglob("fedavg_meta.json"))
        if not candidates:
            print("No partitions found. Run scripts/sweep_alpha.py first.")
            return 1
        target = candidates[0].parent
    target = target if target.is_absolute() else ROOT / target
    return run(target.resolve(), args.probe)


if __name__ == "__main__":
    sys.exit(main())
