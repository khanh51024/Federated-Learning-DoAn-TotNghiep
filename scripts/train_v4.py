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
`calibration` là tập kiểm thử niêm phong (scripts/run_sealed_test.py): logit được lưu để khỏi chạy lại mô hình, nhưng không tóm tắt độ chính xác vào final.json.
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
import dataclasses
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

from src.federated.algorithms import ALGORITHMS  # noqa: E402
from src.federated.compression import MODES  # noqa: E402
from src.training.client_round import AUG_DESCRIPTION, Recipe, compress_upload, prepare_client, train_one_client, train_transform  # noqa: E402
from src.training.data import NUM_CLIENTS, ManifestDataset, extra_manifest_sha256, load_release, load_shard, subset  # noqa: E402
from src.training.loop import cosine_lr, make_loader, make_optimizer, predict, train_epoch  # noqa: E402
from src.training.metrics import summarize  # noqa: E402

CODE_FILES = ("scripts/train_v4.py", "src/training/data.py", "src/training/loop.py", "src/training/metrics.py", "src/training/client_round.py", "src/data/robust_transforms.py",
              "src/training/gpu_augment.py", "src/federated/strategy.py", "src/federated/algorithms.py", "src/federated/flower_app.py", "src/federated/compression.py",
              "src/federated/privacy.py", "src/model/mobilenetv3.py",
              "src/common/image_io.py")
SEALED_SPLIT = "calibration"


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
    parser.add_argument("--algo", choices=ALGORITHMS, default="fedavg",
                        help="thuật toán học liên kết (chỉ với --method fedavg): fedavg, fedprox, fedbn, scaffold, fedavgm, moon, fedbnprox (src/federated/algorithms.py)")
    parser.add_argument("--prox-mu", type=float, default=0.01, help="FedProx, FedBN+FedProx: hệ số μ của (μ/2)·‖w − w_toàn_cục‖²")
    parser.add_argument("--moon-mu", type=float, default=1.0, help="MOON: hệ số của mất mát đối chiếu")
    parser.add_argument("--moon-tau", type=float, default=0.5, help="MOON: nhiệt độ của mất mát đối chiếu")
    parser.add_argument("--server-momentum", type=float, default=0.9, help="FedAvgM: động lượng β phía máy chủ")
    parser.add_argument("--server-lr", type=float, default=0.1, help="FedAvgM: tốc độ học η_s phía máy chủ (hiệu dụng η_s/(1−β))")
    parser.add_argument("--update-compression", choices=MODES, default=None,
                        help="chỉ với --method fedavg: nén bản cập nhật gửi lên ở mỗi vòng (src/federated/compression.py); mặc định gửi nguyên")
    parser.add_argument("--dp-noise", type=float, default=0.0, help="chỉ với --algo fedavg: hệ số nhiễu z của DP phía máy chủ (src/federated/privacy.py); 0 = tắt")
    parser.add_argument("--dp-clip", type=float, default=0.0, help="với --dp-noise: trần chuẩn L2 của độ lệch tham số học được từng client; chọn theo delta_l2_without_bn của một run FedAvg, không theo độ chính xác")
    parser.add_argument("--dp-delta", type=float, default=0.01, help="với --dp-noise: δ của (ε, δ)-DP cấp client; phải nhỏ hơn 1/5 vì có 5 client")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd",
                        help="chỉ với --method centralized: adamw là bản đối chứng N1.1 (thường đi với --lr 0.001); FedAvg và Local-only luôn dùng SGD như công thức chung")
    parser.add_argument("--runtime", choices=("simulation", "flower"), default="simulation",
                        help="chỉ với --method fedavg: simulation = mô phỏng một tiến trình; flower = ServerApp, ClientApp, mô phỏng bằng Ray (src/federated/flower_app.py)")
    parser.add_argument("--flower-client-cpus", type=int, default=2, help="với --runtime flower: số CPU Ray cấp cho một client (có GPU thì mỗi client giữ trọn GPU, chạy lần lượt)")
    parser.add_argument("--flower-round-timeout", type=float, default=3600.0, help="với --runtime flower: số giây tối đa chờ 5 client trả lời trong một vòng; quá thì dừng run")
    args = parser.parse_args()

    import torch

    from src.data.robust_transforms import get_eval_transform
    from src.federated.algorithms import FedBN, make_algorithm
    from src.federated.compression import decode_update
    from src.federated.privacy import add_noise, clip_update, noise_std, rdp_epsilon
    from src.model.mobilenetv3 import get_parameters, set_parameters

    if args.method == "local" and args.client is None:
        parser.error("--method local cần --client")
    if args.algo != "fedavg" and args.method != "fedavg":
        parser.error("--algo chỉ dùng với --method fedavg")
    if args.optimizer != "sgd" and args.method != "centralized":
        parser.error("--optimizer adamw chỉ dùng với --method centralized (bản đối chứng N1.1)")
    if args.runtime != "simulation" and args.method != "fedavg":
        parser.error("--runtime flower chỉ dùng với --method fedavg")
    if args.update_compression and args.method != "fedavg":
        parser.error("--update-compression chỉ dùng với --method fedavg")
    if args.dp_noise and (args.method != "fedavg" or args.algo != "fedavg" or args.dp_clip <= 0):
        parser.error("--dp-noise chỉ dùng với --method fedavg --algo fedavg và cần --dp-clip > 0")
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
    if args.method != "centralized" and args.scheme not in release["meta"]["partition_shard_hashes"]:
        config["partition_extra_manifest_sha256"] = extra_manifest_sha256(release_dir)   # cách chia bổ sung (scripts/build_extra_partitions.py)
    if args.optimizer != "sgd":    # chỉ ghi khi khác mặc định, như các khóa dưới đây: cấu hình run thường giữ nguyên như đợt v4b
        config["optimizer"] = args.optimizer
    if args.runtime != "simulation":
        config["runtime"] = args.runtime
    if args.update_compression:
        config["update_compression"] = args.update_compression
    if args.dp_noise:
        config["server_dp"] = {"noise_multiplier": args.dp_noise, "clip": args.dp_clip, "delta": args.dp_delta, "rounds": args.budget, "level": "client",
                               "epsilon": rdp_epsilon(args.dp_noise, args.budget, args.dp_delta), "accounting": "RDP, cơ chế Gaussian, tham gia đầy đủ"}
    if args.algo != "fedavg":      # chỉ ghi khi khác FedAvg: cấu hình của run FedAvg thường giữ nguyên như đợt v4b
        config["algorithm"] = {"name": args.algo, **{"fedprox": {"mu": args.prox_mu}, "fedbnprox": {"mu": args.prox_mu}, "moon": {"mu": args.moon_mu, "tau": args.moon_tau},
                                                       "fedavgm": {"beta": args.server_momentum, "server_lr": args.server_lr},
                                                       "scaffold": {"momentum": args.momentum}}.get(args.algo, {})}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        keys = [k for k in config if k not in ("git_commit", "torch", "device", "gpu", "python")]
        changed = {k: (previous.get(k), config[k]) for k in keys if previous.get(k) != config[k]}
        if changed:
            raise SystemExit(f"{run_dir} đã có lần chạy với cấu hình khác, không chạy tiếp lên đó: {changed}")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    eval_tf = get_eval_transform()
    recipe = Recipe(dataset_root=str(args.dataset_root), seed=args.seed, batch_size=args.batch_size, momentum=args.momentum, weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing, local_epochs=args.local_epochs, workers=args.workers, aug=args.aug, train_size=args.train_size, sampling=args.sampling,
                    class_power=args.class_power, logit_adjust=args.logit_adjust, channels_last=args.channels_last, amp=args.amp)
    # với Flower, dữ liệu học nằm ở ClientApp: máy chủ chỉ giữ tập đánh giá
    clients = [] if args.runtime == "flower" else [prepare_client(rows, k, recipe, train_transform(recipe), device, persistent=args.method == "fedavg") for k, rows in enumerate(train_sets)]
    eval_loaders = {name: make_loader(ManifestDataset(rows, args.dataset_root, eval_tf), args.batch_size * 2, False, args.workers, 0)
                    for name, rows in eval_rows.items()}
    aug_generators = [c["generator"] for c in clients if c["generator"] is not None]

    model = build_model(args.seed).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    steps_total = args.budget * (args.local_epochs if args.method == "local" else 1)
    optimizer = make_optimizer(model, args.lr, args.momentum, args.weight_decay, args.optimizer) if args.method != "fedavg" else None
    algo = make_algorithm(args.algo, model, NUM_CLIENTS, mu=args.prox_mu, moon_mu=args.moon_mu, moon_tau=args.moon_tau,
                          server_momentum=args.server_momentum, server_lr=args.server_lr, momentum=args.momentum)
    use_scaler = device.type == "cuda" and args.amp == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start, best = 0, {"score": -1.0, "step": None}
    last_path, best_path = run_dir / "checkpoint_last.pt", run_dir / "checkpoint_best.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        algo.load_state_dict(state.get("algo_state") or {})
        scaler.load_state_dict(state["scaler_state"])
        start, best = state["step"] + 1, state["best"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        for generator, rng in zip(aug_generators, state.get("aug_rng") or []):
            generator.set_state(rng)
        print(f"[train] {run_dir.name}: chạy tiếp từ bước {start}/{steps_total}")

    history = run_dir / "history.jsonl"

    def simulate_round(step: int, lr: float, parent: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Một vòng học liên kết trong cùng tiến trình; đường Flower có hàm cùng dạng (src/federated/flower_app.py)."""
        updates, client_stats = [], []
        for k, data in enumerate(clients):
            local = copy.deepcopy(model)
            local_state, stats = train_one_client(local, k, parent, algo, data, lr, recipe, device)
            if args.update_compression:
                # đi qua đúng đường nén và giải nén như khi gửi thật, nên sai số của nén nằm trong quá trình hội tụ
                local_state = decode_update(compress_upload(local_state, parent, args.update_compression, stats, algo.extra_bytes(parent)[0]), parent, args.update_compression)
            updates.append({"weights": local_state, "num_examples": data["num_examples"], "client_id": k})
            client_stats.append(stats)
            del local
        return updates, client_stats

    def training_loop(run_round: Any) -> None:
        """Lặp theo epoch/vòng, đánh giá, ghi nhật ký, lưu checkpoint. `run_round` là cách chạy một vòng học liên kết (không dùng với Centralized, Local-only)."""
        nonlocal best
        for step in range(start, steps_total):
            lr = cosine_lr(args.lr, step, steps_total, args.warmup)
            started = time.time()
            if args.method == "fedavg":
                parent = get_parameters(model)
                updates, client_stats = run_round(step, lr, parent)
                if args.dp_noise:
                    trainable = {name for name, _ in model.named_parameters()}
                    for update, stats in zip(updates, client_stats):
                        update["weights"], stats["norm_before_clip"] = clip_update(update["weights"], parent, trainable, args.dp_clip)
                    std = noise_std(args.dp_noise, args.dp_clip, [u["num_examples"] for u in updates])
                    set_parameters(model, add_noise(algo.aggregate(updates, parent), trainable, std, args.seed * 100000 + step))
                else:
                    set_parameters(model, algo.aggregate(updates, parent))
                train_info = {"clients": client_stats, "steps": sum(c["steps"] for c in client_stats)}
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr
                only = clients[0]
                train_info = train_epoch(model, only["loader"], optimizer, scaler, device, only["weights"], args.label_smoothing, only["augment"], only["adjust"],
                                         args.channels_last, args.amp)
            val = summarize(predict(model, eval_loaders["val"], device, args.channels_last, args.amp), eval_rows["val"])
            record = {"step": step, "lr": lr, "train": train_info, "val": val, "seconds": round(time.time() - started, 1)}
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if val["selection_score"] is not None and val["selection_score"] > best["score"]:
                best = {"score": val["selection_score"], "step": step}
                best_payload = {"model_state": model.state_dict(), "step": step, "selection_score": val["selection_score"], "method": args.method,
                                "seed": args.seed, "client": args.client, "release_manifest_sha256": release["sha256"], "code_fingerprint": config["code_fingerprint"]}
                if isinstance(algo, FedBN):
                    best_payload["client_bn"] = algo.client_bn          # BatchNorm riêng từng client tại bước tốt nhất, để chấm mô hình cá nhân hóa
                save(best_path, best_payload)
            save(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                             "algo_state": algo.state_dict(), "scaler_state": scaler.state_dict(), "step": step, "best": best, "torch_rng": torch.get_rng_state(),
                             "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "aug_rng": [g.get_state() for g in aug_generators]})
            print(f"[train] {run_dir.name} {step + 1}/{steps_total} lr {lr:.4f} val top1 {val['top1']} chọn {val['selection_score']} "
                  f"(tốt nhất {best['score']} ở bước {best['step']}) {record['seconds']}s", flush=True)

    if args.runtime == "flower":
        from src.federated.flower_app import ClientPlan, run_on_flower

        plan = ClientPlan(recipe=dataclasses.asdict(recipe), release_dir=str(release_dir), release_sha=release["sha256"], scheme=args.scheme, limit=args.limit,
                          algorithm=args.algo, algorithm_options={"mu": args.prox_mu, "moon_mu": args.moon_mu, "moon_tau": args.moon_tau, "server_momentum": args.server_momentum,
                                                                  "server_lr": args.server_lr, "momentum": args.momentum},
                          state_dir=str(run_dir / "client_state"), device=str(device), compression=args.update_compression)
        run_on_flower(training_loop, algo, plan, client_cpus=args.flower_client_cpus, round_timeout=args.flower_round_timeout, use_gpu=device.type == "cuda")
    else:
        training_loop(simulate_round)

    state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    logits = {name: predict(model, loader, device, args.channels_last, args.amp) for name, loader in eval_loaders.items()}
    np.savez_compressed(run_dir / "logits_best.npz", **logits, **{f"{n}_sample_ids": np.array([r["sample_id"] for r in eval_rows[n]]) for n in eval_rows})
    final = {"run": run_dir.name, "best_step": state["step"], "selection_score": state["selection_score"], "config": config,
             **{name: summarize(logits[name], eval_rows[name]) for name in eval_rows if name != SEALED_SPLIT}}
    if isinstance(algo, FedBN):
        # Mô hình cá nhân hóa của FedBN: trọng số toàn cục + BatchNorm riêng của từng client; chấm trên val và tập giữ riêng (mô hình toàn cục ở trên dùng BN đã gộp)
        global_state = get_parameters(model)
        personalized, personal_logits = {}, {}
        for k in range(NUM_CLIENTS):
            set_parameters(model, {**global_state, **state["client_bn"][k]})
            personal = {name: predict(model, eval_loaders[name], device, args.channels_last, args.amp) for name in ("val", "legacy_diagnostic_test")}
            personalized[f"client_{k}"] = {name: summarize(personal[name], eval_rows[name]) for name in personal}
            personal_logits.update({f"{name}_client_{k}": value for name, value in personal.items()})
        set_parameters(model, global_state)
        np.savez_compressed(run_dir / "logits_personalized.npz", **personal_logits)
        final["personalized"] = personalized
    (run_dir / "final.json").write_text(json.dumps(final, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    print(f"[train] {run_dir.name}: xong; bước tốt nhất {state['step']}, test top1 {final['legacy_diagnostic_test']['top1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
