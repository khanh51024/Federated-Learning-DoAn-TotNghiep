"""Huấn luyện MobileNetV3-Small trên release v4 theo một trong ba cách, cùng một công thức (xem src/training/__init__.py).

  centralized  học trên toàn bộ train, `--budget` epoch
  fedavg       mô phỏng 5 client của phân vùng `--scheme`; mỗi vòng: mọi client học `--local-epochs` epoch trên shard của mình từ mô hình chung,
               máy chủ gộp bằng src.federated.strategy.aggregate_fedavg (trung bình theo n_k, tích lũy float64); `--budget` vòng
  local        một client (`--client k`) chỉ học trên shard của mình, `--budget` x `--local-epochs` epoch (bằng số bước client đó làm trong FedAvg)
Cùng số bước gradient: Centralized `budget` epoch toàn tập = FedAvg `budget` vòng x 1 epoch cục bộ trên 5 shard; Local-only bằng phần của một client.

Công thức chung: khởi tạo ImageNet + đầu 38 lớp theo hạt giống (w0 giống nhau giữa ba cách cùng hạt giống); SGD động lượng 0,9; tốc độ học cosine theo
epoch/vòng; tăng cường mạnh (src/data/robust_transforms.py) khi học, `canonical_v1` khi đánh giá; label smoothing 0,1; trọng số mất mát cân bằng ảnh
phòng thí nghiệm với ảnh ngoài đồng; chọn checkpoint theo macro-F1 trung bình giữa các nguồn trên tập val (src/training/metrics.py).

Đầu ra ở `--out/<cách>[_cK]_s<hạt giống>/`: config.json, history.jsonl, checkpoint_last.pt (để chạy tiếp khi bị ngắt), checkpoint_best.pt
(`model_state`, cùng định dạng mô hình trên web), logits_best.npz (val, calibration, legacy_diagnostic_test), final.json, DONE.
Chạy lại cùng lệnh sẽ tiếp tục từ checkpoint_last.pt; thư mục có DONE thì bỏ qua.

Ví dụ:
    python scripts/train_v4.py --method centralized --seed 42 --dataset-root /data/dataset_sources --expected-release-sha <sha>
    python scripts/train_v4.py --method fedavg --seed 42 ...
    python scripts/train_v4.py --method local --client 0 --seed 42 ...
    python scripts/train_v4.py --method fedavg --seed 1 --limit 200 --budget 2 --workers 0   # chạy thử nhanh trên CPU
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.data import (NUM_CLIENTS, ManifestDataset, balanced_sampling_weights, class_prior, domain_balanced_weights,  # noqa: E402
                                load_release, load_shard)
from src.training.loop import cosine_lr, logit_adjustment, make_loader, make_optimizer, predict, train_epoch  # noqa: E402
from src.training.metrics import summarize  # noqa: E402

CODE_FILES = ("scripts/train_v4.py", "src/training/data.py", "src/training/loop.py", "src/training/metrics.py", "src/data/robust_transforms.py",
              "src/training/gpu_augment.py", "src/federated/strategy.py", "src/model/mobilenetv3.py", "src/common/image_io.py")
AUG_DESCRIPTION = {
    "robust_v1": "robust_v1 (src/data/robust_transforms.py)",
    "gpu_v2": "gpu_v2 (get_cpu_train_transform + src/training/gpu_augment.py)",
    "gpu_v2_ta": "gpu_v2_ta (get_cpu_train_transform với TrivialAugmentWide + src/training/gpu_augment.py, bỏ phần màu/xoay trên GPU)",
}


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for rel in CODE_FILES:
        digest.update(rel.encode())
        digest.update((ROOT / rel).read_bytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001 - gói mã không kèm .git
        version = ROOT / "CODE_VERSION"
        return version.read_text(encoding="utf-8").strip() if version.is_file() else None


def subset(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return sorted(rng.sample(rows, limit), key=lambda r: r["sample_id"])


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(seed: int) -> Any:
    """ImageNet-1K V1 + đầu 38 lớp; đầu khởi tạo theo hạt giống nên ba cách cùng hạt giống có cùng w0."""
    from src.model.mobilenetv3 import create_model

    seed_everything(seed)
    return create_model(num_classes=38, pretrained=True)


def run_name(args: argparse.Namespace) -> str:
    return f"{args.method}" + (f"_c{args.client}" if args.method == "local" else "") + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else "")


def save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("centralized", "fedavg", "local"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--client", type=int, default=None, help="chỉ với --method local")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "dataset_sources")
    parser.add_argument("--release-dir", type=Path, default=None, help="mặc định <dataset-root>/mixed/pv_pd_v4")
    parser.add_argument("--expected-release-sha", default=None, help="SHA-256 của release_manifest.json; bắt buộc trừ khi --limit")
    parser.add_argument("--scheme", default="label_alpha_0_1")
    parser.add_argument("--budget", type=int, default=40, help="số epoch (centralized, local: nhân thêm --local-epochs) hoặc số vòng (fedavg)")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1, help="số epoch/vòng khởi động tuyến tính của tốc độ học")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "v4")
    parser.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh mỗi tập (chạy thử); kết quả KHÔNG dùng để báo cáo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aug", choices=tuple(AUG_DESCRIPTION), default="robust_v1",
                        help="robust_v1: tăng cường toàn bộ trên CPU (run mốc); gpu_v2: phần nặng chạy trên GPU; gpu_v2_ta: như gpu_v2 + TrivialAugmentWide")
    parser.add_argument("--sampling", choices=("weights", "balanced"), default="weights",
                        help="weights: trọng số mất mát cân bằng hai miền (run mốc); balanced: lấy mẫu có hoàn lại cân bằng hai miền")
    parser.add_argument("--class-power", type=float, default=1.0,
                        help="với --sampling balanced: lớp c được chọn tỉ lệ n_c^p trong miền (1 giữ tỉ lệ, 0,5 căn bậc hai)")
    parser.add_argument("--logit-adjust", type=float, default=0.0, help="τ của logit adjustment khi học (0 = tắt)")
    parser.add_argument("--train-size", type=int, default=224, help="cạnh ảnh khi học; đánh giá luôn ở 224 như web")
    parser.add_argument("--channels-last", action="store_true", help="bố cục bộ nhớ channels_last (nhanh hơn với fp16 trên GPU)")
    parser.add_argument("--tag", default="", help="hậu tố tên run (ablation)")
    parser.add_argument("--amp", choices=("fp16", "bf16"), default="fp16", help="kiểu số khi học trên GPU; bf16 không cần GradScaler (bớt một lần đồng bộ mỗi bước)")
    args = parser.parse_args()

    import torch

    from src.data.robust_transforms import get_cpu_train_transform, get_eval_transform, get_robust_train_transform
    from src.training.gpu_augment import GpuAugment
    from src.federated.strategy import aggregate_fedavg
    from src.model.mobilenetv3 import get_parameters, set_parameters

    if args.method == "local" and args.client is None:
        parser.error("--method local cần --client")
    if not args.expected_release_sha and not args.limit:
        parser.error("cần --expected-release-sha (SHA-256 của release_manifest.json) khi chạy thật")
    release_dir = args.release_dir or args.dataset_root / "mixed" / "pv_pd_v4"
    run_dir = args.out / run_name(args)
    if (run_dir / "DONE").is_file():
        print(f"[train] {run_dir.name}: đã xong, bỏ qua")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    # Tắt dò thuật toán cuDNN: đo trên RTX 4080 (MobileNetV3-Small, channels_last, fp16), mỗi kích thước batch MỚI tốn khoảng 40 giây dò
    # (batch cuối lẻ của từng client, batch đánh giá...), còn tốc độ ổn định không hơn khi tắt (20,9 so với 19,7 ms/bước).
    torch.backends.cudnn.benchmark = False

    release = load_release(release_dir, args.expected_release_sha)
    if args.method == "centralized":
        train_sets = [subset(release["train"], args.limit, args.seed)]
    elif args.method == "fedavg":
        train_sets = [subset(load_shard(release_dir, args.scheme, k, release), args.limit // NUM_CLIENTS if args.limit else 0, args.seed + k)
                      for k in range(NUM_CLIENTS)]
    else:
        train_sets = [subset(load_shard(release_dir, args.scheme, args.client, release), args.limit, args.seed)]
    eval_rows = {name: subset(release[name], args.limit, args.seed) for name in ("val", "calibration", "legacy_diagnostic_test")}

    config = {
        "method": args.method, "seed": args.seed, "client": args.client, "scheme": args.scheme if args.method != "centralized" else None,
        "budget": args.budget, "local_epochs": args.local_epochs, "batch_size": args.batch_size, "lr": args.lr, "momentum": args.momentum,
        "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing, "warmup": args.warmup, "limit": args.limit,
        "train_transform": AUG_DESCRIPTION[args.aug], "train_size": args.train_size, "eval_transform": "canonical_v1",
        "loss_weighting": "domain_balanced" if args.sampling == "weights" else f"balanced_sampler(class_power={args.class_power})",
        "logit_adjust_tau": args.logit_adjust, "channels_last": args.channels_last, "tag": args.tag, "amp": args.amp,
        "selection": "mean macro-F1 over val sources with >= 30 images", "release_name": release["meta"]["release_name"],
        "release_manifest_sha256": release["sha256"], "train_sizes": [len(rows) for rows in train_sets], "code_fingerprint": code_fingerprint(),
        "git_commit": git_commit(), "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "python": platform.python_version(),
    }
    config_path = run_dir / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        keys = [k for k in config if k not in ("git_commit", "torch", "device", "gpu", "python")]
        changed = {k: (previous.get(k), config[k]) for k in keys if previous.get(k) != config[k]}
        if changed:
            raise SystemExit(f"{run_dir} đã có lần chạy với cấu hình khác, không chạy tiếp lên đó: {changed}")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    eval_tf = get_eval_transform()
    if args.aug == "robust_v1":
        train_tf = get_robust_train_transform(args.train_size)
    else:
        train_tf = get_cpu_train_transform(args.train_size, trivial_augment=args.aug == "gpu_v2_ta")
    if args.sampling == "balanced":
        sampling = [balanced_sampling_weights(rows, args.class_power) for rows in train_sets]
        weights = [torch.ones(len(rows), dtype=torch.float32) for rows in train_sets]   # cân bằng đã nằm ở bộ lấy mẫu
        seen_weights = sampling
    else:
        sampling = [None] * len(train_sets)
        weights = [torch.tensor(domain_balanced_weights(rows), dtype=torch.float32) for rows in train_sets]
        seen_weights = [w.tolist() for w in weights]
    # logit adjustment theo tần suất lớp mà chính tập học (với FedAvg/Local: shard của client) thực sự đưa vào mất mát
    adjusts = [logit_adjustment(class_prior(rows, w), args.logit_adjust) for rows, w in zip(train_sets, seen_weights)]
    loaders = [make_loader(ManifestDataset(rows, args.dataset_root, train_tf), args.batch_size, True, args.workers, args.seed * 100 + k,
                           persistent=args.method == "fedavg", sampling_weights=sampling[k]) for k, rows in enumerate(train_sets)]
    eval_loaders = {name: make_loader(ManifestDataset(rows, args.dataset_root, eval_tf), args.batch_size * 2, False, args.workers, 0)
                    for name, rows in eval_rows.items()}
    aug_generators: list[Any] = []
    augments: list[Any] = [None] * len(train_sets)
    if args.aug != "robust_v1":
        for k in range(len(train_sets)):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed * 1000 + k)
            aug_generators.append(generator)
            augments[k] = GpuAugment(generator, color=args.aug == "gpu_v2", rotate=args.aug == "gpu_v2")

    model = build_model(args.seed).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    steps_total = args.budget * (args.local_epochs if args.method == "local" else 1)
    optimizer = make_optimizer(model, args.lr, args.momentum, args.weight_decay) if args.method != "fedavg" else None
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start, best = 0, {"score": -1.0, "step": None}
    last_path, best_path = run_dir / "checkpoint_last.pt", run_dir / "checkpoint_best.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        start, best = state["step"] + 1, state["best"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        for generator, rng in zip(aug_generators, state.get("aug_rng") or []):
            generator.set_state(rng)
        print(f"[train] {run_dir.name}: chạy tiếp từ bước {start}/{steps_total}")

    history = run_dir / "history.jsonl"
    for step in range(start, steps_total):
        lr = cosine_lr(args.lr, step, steps_total, args.warmup)
        started = time.time()
        if args.method == "fedavg":
            parent = get_parameters(model)
            updates, client_stats = [], []
            for k, (loader, w) in enumerate(zip(loaders, weights)):
                local = copy.deepcopy(model)
                local_opt = make_optimizer(local, lr, args.momentum, args.weight_decay)   # trạng thái bộ tối ưu của client làm mới mỗi vòng
                local_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
                stats = [train_epoch(local, loader, local_opt, local_scaler, device, w, args.label_smoothing, augments[k], adjusts[k], args.channels_last,
                                     args.amp) for _ in range(args.local_epochs)]
                updates.append({"weights": get_parameters(local), "num_examples": len(train_sets[k]), "client_id": k})
                client_stats.append({"client": k, "loss": stats[-1]["loss"], "steps": sum(s["steps"] for s in stats)})
                del local, local_opt
            set_parameters(model, aggregate_fedavg(updates, parent_weights=parent))
            train_info = {"clients": client_stats, "steps": sum(c["steps"] for c in client_stats)}
        else:
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_info = train_epoch(model, loaders[0], optimizer, scaler, device, weights[0], args.label_smoothing, augments[0], adjusts[0],
                                     args.channels_last, args.amp)
        val = summarize(predict(model, eval_loaders["val"], device, args.channels_last, args.amp), eval_rows["val"])
        record = {"step": step, "lr": lr, "train": train_info, "val": val, "seconds": round(time.time() - started, 1)}
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if val["selection_score"] is not None and val["selection_score"] > best["score"]:
            best = {"score": val["selection_score"], "step": step}
            save(best_path, {"model_state": model.state_dict(), "step": step, "selection_score": val["selection_score"], "method": args.method,
                             "seed": args.seed, "client": args.client, "release_manifest_sha256": release["sha256"], "code_fingerprint": config["code_fingerprint"]})
        save(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                         "scaler_state": scaler.state_dict(), "step": step, "best": best, "torch_rng": torch.get_rng_state(),
                         "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "aug_rng": [g.get_state() for g in aug_generators]})
        print(f"[train] {run_dir.name} {step + 1}/{steps_total} lr {lr:.4f} val top1 {val['top1']} chọn {val['selection_score']} "
              f"(tốt nhất {best['score']} ở bước {best['step']}) {record['seconds']}s", flush=True)

    state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    logits = {name: predict(model, loader, device, args.channels_last, args.amp) for name, loader in eval_loaders.items()}
    np.savez_compressed(run_dir / "logits_best.npz", **logits, **{f"{n}_sample_ids": np.array([r["sample_id"] for r in eval_rows[n]]) for n in eval_rows})
    final = {"run": run_dir.name, "best_step": state["step"], "selection_score": state["selection_score"], "config": config,
             **{name: summarize(logits[name], eval_rows[name]) for name in eval_rows}}
    (run_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    print(f"[train] {run_dir.name}: xong; bước tốt nhất {state['step']}, test top1 {final['legacy_diagnostic_test']['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
