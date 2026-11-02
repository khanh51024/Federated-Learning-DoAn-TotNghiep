"""Trước khi tốn GPU cho DP (N4.3): mô hình FedAvg đã học chịu được nhiễu Gaussian lớn đến đâu trên tham số học được, và mức nhiễu đó ứng với ε bao nhiêu.

Chỉ đo một lần cộng nhiễu lên mô hình đã học xong, trên val; không huấn luyện. Đây là ước lượng LẠC QUAN cho DP-FedAvg: khi học thật, nhiễu bị cộng ở mọi vòng. Nếu mô hình đã sụp đổ ở mức nhiễu này
thì học thật cũng không cứu được; nếu chưa sụp đổ thì vẫn phải chạy thử mới biết. Đổi từ nhiễu mỗi tọa độ σ sang hệ số nhiễu z theo src/federated/privacy.py: z = σ / (clip · max(n_k/n)).

    python scripts/eval_dp_noise_sensitivity.py --run runs/v4b/fedavg_s42
Kết quả: artifacts/dp_noise_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402
from src.federated.privacy import add_noise, rdp_epsilon  # noqa: E402
from src.training.data import ManifestDataset, load_release, load_shard  # noqa: E402

SIGMAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
EVERY = 4           # đo trên mỗi ảnh thứ tư của val để đủ nhanh trên CPU
ROUNDS = 40
DELTA = 0.01
NOISE_DRAWS = 2


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "v4b" / "fedavg_s42")
    parser.add_argument("--clip", type=float, default=5.0, help="trần chuẩn L2 giả định để đổi σ sang z; lấy theo delta_l2_without_bn thật khi đã có run có nhật ký")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset_sources")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    import torch

    from src.data.robust_transforms import get_eval_transform
    from src.model.mobilenetv3 import create_model
    from src.training.metrics import macro_f1

    torch.set_num_threads(args.threads)
    config = json.loads((args.run / "final.json").read_text(encoding="utf-8"))["config"]
    release_dir = args.dataset_root / "mixed" / "pv_pd_v4"
    release = load_release(release_dir, config["release_manifest_sha256"])
    rows = release["val"][::EVERY]
    n_k = [len(load_shard(release_dir, config["scheme"], k, release)) for k in range(5)]
    weight_max = max(n_k) / sum(n_k)
    targets = np.array([int(r["class_id"]) for r in rows])
    model = create_model(num_classes=38, pretrained=False)
    state = torch.load(args.run / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
    trainable = {name for name, _ in model.named_parameters()}
    num_params = sum(state[name].numel() for name in trainable)
    weight_norm = float(torch.cat([state[name].flatten().float() for name in trainable]).norm())
    loader = torch.utils.data.DataLoader(ManifestDataset(rows, args.dataset_root, get_eval_transform()), batch_size=64, shuffle=False, num_workers=0)
    batches = [images for images, _, _ in loader]

    results = []
    for sigma in SIGMAS:
        scores = []
        for draw in range(NOISE_DRAWS if sigma else 1):
            model.load_state_dict(add_noise(state, trainable, sigma, 1000 + draw) if sigma else state)
            model.eval()
            with torch.inference_mode():
                preds = torch.cat([model(images).argmax(1) for images in batches]).numpy()
            scores.append({"top1": float(np.mean(preds == targets)), "macro_f1": macro_f1(targets, preds)})
        z = sigma / (args.clip * weight_max)
        entry = {"sigma_per_coordinate": sigma, "noise_l2_norm": sigma * num_params ** 0.5, "z": z, "epsilon_zero_out": rdp_epsilon(z, ROUNDS, DELTA) if z else None,
                 "top1": float(np.mean([s["top1"] for s in scores])), "macro_f1": float(np.mean([s["macro_f1"] for s in scores]))}
        results.append(entry)
        print(f"[dp] σ={sigma:g}: z={z:.2e}, ε={entry['epsilon_zero_out'] if z else float('inf'):.3g}, val top-1 {100 * entry['top1']:.1f}%", flush=True)
    report = {"evidence": "một lần cộng nhiễu lên mô hình đã học, đo trên val; ước lượng lạc quan cho DP-FedAvg vì khi học thật nhiễu được cộng ở mọi vòng; KHÔNG phải kết quả DP-FedAvg", "run": args.run.name,
              "val_images": len(rows), "clip_assumed": args.clip, "max_client_weight": weight_max, "rounds_for_epsilon": ROUNDS, "delta": DELTA, "num_trainable_parameters": num_params,
              "weight_l2_norm": weight_norm, "noise_draws": NOISE_DRAWS, "results": results}
    out = reg.abs_path("artifacts/dp_noise_sensitivity.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[dp] ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
