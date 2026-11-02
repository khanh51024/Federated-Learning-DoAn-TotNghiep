"""Học phân tán có tự gây sai lệch so với học tập trung không, hay sai lệch đến từ học cục bộ trên dữ liệu lệch? Đo trên TOÀN BỘ mô hình MobileNetV3-Small với ảnh thật.

Cơ sở: FedSGD (mỗi client đúng một bước gradient toàn shard rồi gộp theo n_k) về lý thuyết TRÙNG một bước gradient toàn tập của học tập trung, vì gradient toàn tập là trung bình gradient các shard theo n_k.
Vậy nếu truyền hoặc gộp trọng số có lỗi, A sẽ lệch. Nếu A trùng, mọi chênh lệch còn lại phải đến từ các yếu tố được bật dần ở B và C:

  A. FedSGD so với tập trung: BatchNorm đóng băng, không dropout, GD thuần. Kỳ vọng: float64 lệch ở mức làm tròn; float32 lớn hơn một chút.
  B. Nhiều bước cục bộ: K bước cục bộ mỗi vòng, CÙNG tổng số bước, trên shard đồng nhất và shard lệch nhãn.
  C. BatchNorm thật: như A nhưng BatchNorm tính theo batch của từng client, nên thống kê phụ thuộc cách chia dữ liệu.
  D. Đối chứng không có học phân tán: chạy TẬP TRUNG hai lần trên cùng ảnh, chỉ đổi thứ tự ảnh trong batch. Về toán học phải trùng, nên mọi chênh lệch ở đây là sai số làm tròn bị khuếch đại qua các bước học.
     Đo theo số bước để thấy tốc độ khuếch đại, và là thước đo để đọc A ở float32: A float32 chỉ có nghĩa khi so với D float32.

Đây là phép kiểm công thức trên một tập ảnh nhỏ và ít bước, KHÔNG phải kết quả huấn luyện đầy đủ; dùng checkpoint nào cũng không liên quan vì xuất phát từ W0.

    python scripts/verify_fedsgd_equivalence.py --device cuda
Kết quả: artifacts/fedsgd_equivalence.json
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHARD_SIZES = (24, 12, 6, 10, 16)       # cỡ khác nhau có chủ ý: để kiểm cả trọng số n_k của phép gộp
TOTAL_STEPS = 20
LR = 0.01


def batch_gradient_step(model: Any, images: Any, targets: Any, lr: float) -> None:
    """Một bước GD thuần trên cả tập đưa vào: mất mát trung bình, không động lượng, không weight decay."""
    import torch

    for p in model.parameters():
        p.grad = None
    torch.nn.functional.cross_entropy(model(images), targets).backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad


def set_mode(model: Any, batchnorm_live: bool) -> None:
    """Luôn tắt dropout. BatchNorm đóng băng thì thống kê lấy từ W0, không phụ thuộc cách chia dữ liệu."""
    import torch

    model.eval()
    if batchnorm_live:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.train()


def centralized(model: Any, shards: list[tuple[Any, Any]], steps: int, lr: float, batchnorm_live: bool) -> Any:
    import torch

    model = copy.deepcopy(model)
    set_mode(model, batchnorm_live)
    images = torch.cat([s[0] for s in shards])
    targets = torch.cat([s[1] for s in shards])
    for _ in range(steps):
        batch_gradient_step(model, images, targets, lr)
    return model


def federated(model: Any, shards: list[tuple[Any, Any]], rounds: int, local_steps: int, lr: float, batchnorm_live: bool) -> Any:
    """FedAvg gộp theo n_k trên toàn state_dict; bộ đếm của BatchNorm lấy của client đầu."""
    import torch

    model = copy.deepcopy(model)
    total = sum(len(s[1]) for s in shards)
    for _ in range(rounds):
        states = []
        for images, targets in shards:
            local = copy.deepcopy(model)
            set_mode(local, batchnorm_live)
            for _ in range(local_steps):
                batch_gradient_step(local, images, targets, lr)
            states.append((local.state_dict(), len(targets)))
        merged = {}
        for key, value in states[0][0].items():
            if value.is_floating_point():
                merged[key] = sum(state[key] * (n / total) for state, n in states)
            else:
                merged[key] = value
        with torch.no_grad():
            model.load_state_dict(merged)
    return model


def shuffled_order_gap(model: Any, shards: list[tuple[Any, Any]], steps: int, lr: float, seed: int) -> Any:
    """Tập trung trên đúng các ảnh đó nhưng đảo thứ tự: gradient trung bình về toán học không đổi."""
    import torch

    images = torch.cat([s[0] for s in shards])
    targets = torch.cat([s[1] for s in shards])
    order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(seed)).to(targets.device)
    return centralized(model, [(images[order], targets[order])], steps, lr, False)


def distance(a: Any, b: Any, w0: Any) -> dict[str, float]:
    """Lệch giữa hai mô hình trên tham số học được, và so với quãng đường mô hình tập trung đã đi kể từ W0."""
    params_a, params_b, params_0 = dict(a.named_parameters()), dict(b.named_parameters()), dict(w0.named_parameters())
    square, moved, worst = 0.0, 0.0, 0.0
    for name, value in params_b.items():
        diff = (params_a[name] - value).detach().double()
        square += float((diff * diff).sum())
        worst = max(worst, float(diff.abs().max()))
        step = (value - params_0[name]).detach().double()
        moved += float((step * step).sum())
    return {"l2": square ** 0.5, "max_abs": worst, "l2_relative_to_centralized_travel": (square / moved) ** 0.5 if moved else 0.0}


def agreement(a: Any, b: Any, images: Any, batchnorm_live: bool) -> float:
    import torch

    a, b = copy.deepcopy(a).eval(), copy.deepcopy(b).eval()
    with torch.inference_mode():
        return float((a(images).argmax(1) == b(images).argmax(1)).float().mean())


def load_shards(release: dict[str, Any], release_dir: Path, scheme: str, dataset_root: Path, seed: int, device: Any, dtype: Any) -> list[tuple[Any, Any]]:
    import torch
    from PIL import Image

    from src.common.image_io import to_rgb
    from src.data.robust_transforms import get_eval_transform
    from src.training.data import load_shard

    transform = get_eval_transform()
    shards = []
    for k, size in enumerate(SHARD_SIZES):
        rows = load_shard(release_dir, scheme, k, release)
        chosen = random.Random(seed + k).sample(rows, size)
        images = []
        for row in chosen:
            with Image.open(dataset_root / row["relative_path"]) as handle:
                images.append(transform(to_rgb(handle)))
        shards.append((torch.stack(images).to(device=device, dtype=dtype), torch.tensor([int(r["class_id"]) for r in chosen], device=device)))
    return shards


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset_sources")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "fedsgd_equivalence.json")
    args = parser.parse_args()

    import torch

    from src.model.mobilenetv3 import create_model
    from src.training.data import load_release

    started = time.time()
    device = torch.device(args.device)
    release_dir = args.dataset_root / "mixed" / "pv_pd_v4"
    release = load_release(release_dir, None)
    torch.manual_seed(args.seed)
    base = create_model(num_classes=38, pretrained=True)
    report: dict[str, Any] = {"evidence": "phép kiểm công thức trên ảnh thật, số ảnh nhỏ và ít bước; không phải kết quả huấn luyện đầy đủ", "seed": args.seed, "device": args.device, "shard_sizes": list(SHARD_SIZES),
                              "total_steps": TOTAL_STEPS, "lr": LR, "torch": torch.__version__, "release_manifest_sha256": release["sha256"], "checks": []}

    def run(name: str, scheme: str, dtype: Any, rounds: int, local_steps: int, batchnorm_live: bool) -> None:
        shards = load_shards(release, release_dir, scheme, args.dataset_root, args.seed, device, dtype)
        w0 = copy.deepcopy(base).to(device=device, dtype=dtype)
        reference = centralized(w0, shards, rounds * local_steps, LR, batchnorm_live)
        fed = federated(w0, shards, rounds, local_steps, LR, batchnorm_live)
        all_images = torch.cat([s[0] for s in shards])
        entry = {"check": name, "scheme": scheme, "dtype": str(dtype).replace("torch.", ""), "rounds": rounds, "local_steps": local_steps, "batchnorm_live": batchnorm_live,
                 **distance(fed, reference, w0), "prediction_agreement": agreement(fed, reference, all_images, batchnorm_live)}
        report["checks"].append(entry)
        print(f"[fedsgd] {name} {scheme} {entry['dtype']} K={local_steps}: lệch L2 {entry['l2']:.3e}, lớn nhất {entry['max_abs']:.3e}, tương đối {entry['l2_relative_to_centralized_travel']:.3e}", flush=True)

    for scheme in ("label_alpha_0_1", "iid"):
        run("A_fedsgd", scheme, torch.float64, TOTAL_STEPS, 1, False)
    run("A_fedsgd", "label_alpha_0_1", torch.float32, TOTAL_STEPS, 1, False)
    for scheme in ("iid", "label_alpha_0_1"):
        for local_steps in (2, 5, 10):
            run("B_local_steps", scheme, torch.float64, TOTAL_STEPS // local_steps, local_steps, False)
    for scheme in ("iid", "label_alpha_0_1"):
        run("C_batchnorm_live", scheme, torch.float64, TOTAL_STEPS, 1, True)
    report["amplification"] = []
    for dtype in (torch.float64, torch.float32):
        shards = load_shards(release, release_dir, "label_alpha_0_1", args.dataset_root, args.seed, device, dtype)
        w0 = copy.deepcopy(base).to(device=device, dtype=dtype)
        all_images = torch.cat([s[0] for s in shards])
        for steps in (1, 2, 5, 10, TOTAL_STEPS):
            reference = centralized(w0, shards, steps, LR, False)
            control = shuffled_order_gap(w0, shards, steps, LR, args.seed)
            fed = federated(w0, shards, steps, 1, LR, False)
            entry = {"dtype": str(dtype).replace("torch.", ""), "steps": steps, "control_l2": distance(control, reference, w0)["l2"], "fedsgd_l2": distance(fed, reference, w0)["l2"],
                     "control_agreement": agreement(control, reference, all_images, False), "fedsgd_agreement": agreement(fed, reference, all_images, False)}
            report["amplification"].append(entry)
            print(f"[fedsgd] D {entry['dtype']} {steps} bước: đối chứng đổi thứ tự {entry['control_l2']:.3e}, FedSGD {entry['fedsgd_l2']:.3e}, dự đoán trùng {entry['control_agreement']:.3f} / {entry['fedsgd_agreement']:.3f}", flush=True)
    report["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[fedsgd] ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
