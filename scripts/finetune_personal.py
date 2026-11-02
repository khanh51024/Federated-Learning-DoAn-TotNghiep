"""Cải tiến đề xuất N3.3: mô hình chung cộng phần tinh chỉnh riêng cho từng cơ sở, không cần huấn luyện liên kết lại.

Lý do (đo được, artifacts/sweep_v4): ở α = 0,1, trên phân phối nhãn của chính mình, Local-only đạt khoảng 97% còn mô hình FedAvg chung khoảng 91%; nhưng Local-only gần như không nhận ra lớp
client chưa có ảnh. Mô hình chung thấy mọi lớp nhưng đầu phân loại bị kéo về trung bình toàn hệ thống. Cho từng client học lại riêng đầu phân loại trên dữ liệu của mình, giữ nguyên phần trích
đặc trưng học chung: không tốn thêm truyền thông, dữ liệu vẫn không rời client.

Cách làm: đặc trưng của mô hình tốt nhất (trước đầu phân loại, biến đổi `canonical_v1`) lưu sẵn cho shard, val và tập thử; mỗi client học lại đầu phân loại từ đầu phân loại chung bằng SGD,
cùng label smoothing và cân bằng hai miền như công thức chung. Số epoch chọn theo độ chính xác trên val đánh trọng số theo cặp (miền, lớp) của shard client, cùng cách đo với scripts/eval_sweep_v4.py
(không dùng tập thử, không đụng tập niêm phong). Mô hình cá nhân hóa không có logit trên tập niêm phong nên scripts/run_sealed_test.py từ chối ô này.

Đầu ra: thư mục run mới ở `--out`, cùng định dạng scripts/train_v4.py, có `logits_personalized.npz` (như FedBN) để scripts/eval_sweep_v4.py đọc; `logits_best.npz` và nhật ký là của mô hình chung.
Việc tốn thời gian nhất là trích đặc trưng; chạy được trên CPU.

    python scripts/finetune_personal.py --run runs/v4b/fedavg_s42 --out runs/personal
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_client_fairness import client_accuracy, domain_class_ids, group_recall  # noqa: E402
from src.training.data import NUM_CLIENTS, ManifestDataset, domain_balanced_weights, load_release, load_shard  # noqa: E402

EPOCH_GRID = (1, 2, 5, 10)
SPLITS = ("val", "legacy_diagnostic_test")
NUM_CLASSES = 38


def extract_features(model: Any, rows: list[dict[str, Any]], dataset_root: Path, batch_size: int, workers: int, device: Any) -> Any:
    """Đặc trưng sau avgpool của MobileNetV3-Small, đúng biến đổi đánh giá trên web."""
    import torch

    from src.data.robust_transforms import get_eval_transform

    loader = torch.utils.data.DataLoader(ManifestDataset(rows, dataset_root, get_eval_transform()), batch_size=batch_size, shuffle=False, num_workers=workers)
    parts = []
    model.eval()
    with torch.inference_mode():
        for images, _, _ in loader:
            parts.append(torch.flatten(model.avgpool(model.features(images.to(device))), 1).cpu())
    return torch.cat(parts)


def train_head(head: Any, features: Any, targets: Any, weights: Any, epochs: int, lr: float, momentum: float, weight_decay: float, label_smoothing: float, batch_size: int, seed: int) -> Any:
    """Học lại đầu phân loại trên đặc trưng của shard; dropout của đầu phân loại vẫn bật như lúc học bình thường. Trả bản sao đã học."""
    import torch

    head = copy.deepcopy(head)
    head.train()
    torch.manual_seed(seed)                     # dropout của đầu phân loại lấy từ bộ sinh chung
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = torch.nn.functional.cross_entropy(head(features[batch]), targets[batch], label_smoothing=label_smoothing, reduction="none")
            loss = (loss * weights[batch]).sum() / weights[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


def choose_epochs(head: Any, shard_features: Any, shard_targets: Any, shard_weights: Any, val_features: Any, val_targets: np.ndarray, val_groups: np.ndarray, shard_group_counts: np.ndarray, recipe: dict[str, Any], seed: int,
                  grid: tuple[int, ...] = EPOCH_GRID) -> dict[str, Any]:
    """Thử các số epoch trên val đánh trọng số theo shard; 0 epoch (đầu phân loại chung) cũng là một ứng viên, nên tinh chỉnh chỉ được chọn khi thật sự tốt hơn trên val."""
    import torch

    best = {"epochs": 0, "val_accuracy": -1.0, "head": head}
    tried = {}
    for epochs in (0, *grid):
        candidate = head.eval() if epochs == 0 else train_head(head, shard_features, shard_targets, shard_weights, epochs, recipe["lr"], recipe["momentum"], recipe["weight_decay"],
                                                                 recipe["label_smoothing"], recipe["batch_size"], seed)
        with torch.inference_mode():
            preds = candidate(val_features).argmax(1).numpy()
        accuracy = client_accuracy(group_recall(val_groups, preds == val_targets, len(shard_group_counts)), shard_group_counts)["accuracy"]
        tried[epochs] = accuracy
        if accuracy is not None and accuracy > best["val_accuracy"]:
            best = {"epochs": epochs, "val_accuracy": accuracy, "head": candidate}
    return {**best, "tried": tried}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="thư mục run FedAvg đã xong")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "personal")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset_sources")
    parser.add_argument("--lr", type=float, default=0.005, help="nhỏ hơn lr học liên kết vì chỉ tinh chỉnh")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    from src.model.mobilenetv3 import create_model

    torch.set_num_threads(args.threads)
    run = args.run
    final = json.loads((run / "final.json").read_text(encoding="utf-8"))
    config = final["config"]
    if config["method"] != "fedavg" or not (run / "DONE").is_file():
        parser.error(f"{run}: cần một run học liên kết đã xong")
    target = args.out / f"{run.name}_ft"
    if (target / "DONE").is_file():
        print(f"[finetune] {target.name}: đã xong, bỏ qua")
        return 0
    release_dir = args.dataset_root / "mixed" / "pv_pd_v4"
    release = load_release(release_dir, config["release_manifest_sha256"])
    device = torch.device(args.device)
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"])
    model.to(device).eval()
    head = model.classifier.cpu()
    scheme = config["scheme"]
    seed = int(config["seed"])
    started = time.time()

    eval_rows = {name: release[name] for name in SPLITS}
    features = {name: extract_features(model, rows, args.dataset_root, args.batch_size, args.workers, device) for name, rows in eval_rows.items()}
    targets = {name: np.array([int(r["class_id"]) for r in rows]) for name, rows in eval_rows.items()}
    print(f"[finetune] {run.name}: xong đặc trưng tập đo sau {time.time() - started:.0f}s", flush=True)

    recipe = {"lr": args.lr, "momentum": config["momentum"], "weight_decay": config["weight_decay"], "label_smoothing": config["label_smoothing"], "batch_size": args.batch_size}
    saved, chosen = {}, {}
    for k in range(NUM_CLIENTS):
        rows = load_shard(release_dir, scheme, k, release)
        shard_features = extract_features(model, rows, args.dataset_root, args.batch_size, args.workers, device)
        shard_targets = torch.tensor([int(r["class_id"]) for r in rows])
        shard_weights = torch.tensor(domain_balanced_weights(rows), dtype=torch.float32)
        counts = np.bincount(domain_class_ids(rows, NUM_CLASSES), minlength=2 * NUM_CLASSES).astype(np.float64)
        result = choose_epochs(head, shard_features, shard_targets, shard_weights, features["val"], targets["val"], domain_class_ids(eval_rows["val"], NUM_CLASSES), counts, recipe, seed * 10 + k)
        chosen[k] = {"epochs": result["epochs"], "val_accuracy": result["val_accuracy"], "tried": result["tried"]}
        with torch.inference_mode():
            for name in SPLITS:
                saved[f"{name}_client_{k}"] = result["head"](features[name]).numpy()
        print(f"[finetune] {run.name} client {k}: chọn {result['epochs']} epoch, val theo shard {result['val_accuracy']:.4f} (thử {result['tried']}) {time.time() - started:.0f}s", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    for name in ("logits_best.npz", "history.jsonl", "checkpoint_best.pt"):
        shutil.copy2(run / name, target / name)
    if (run / "plantwild_logits.npz").is_file():
        shutil.copy2(run / "plantwild_logits.npz", target / "plantwild_logits.npz")
    np.savez_compressed(target / "logits_personalized.npz", **saved)
    new_config = {**config, "personalization": {"kind": "head_finetune", "source_run": run.name, "lr": args.lr, "epoch_grid": list(EPOCH_GRID), "chosen": chosen}, "tag": f"{config.get('tag', '')}_ft"}
    (target / "config.json").write_text(json.dumps(new_config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # final.json của run v4b có tóm tắt tập niêm phong từ trước khi có quy định: không chép sang
    (target / "final.json").write_text(json.dumps({**{k: v for k, v in final.items() if k != "calibration"}, "config": new_config}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (target / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    print(f"[finetune] {target.name}: xong sau {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
