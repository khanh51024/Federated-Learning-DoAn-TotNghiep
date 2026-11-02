"""In kết quả kiểm chứng ra màn hình cho dễ xem khi demo (docs/HUONG_DAN_DEMO_THU_CONG.md). Chỉ đọc tệp đã có, không train, không đọc split calibration.

    python scripts/demo_show.py flower --history tmp/demo/history.json     # server_app thủ công: băm, độ trễ, tốc độ từng client từng vòng
    python scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower   # run train_v4: thông số Flower theo vòng
    python scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42   # so trọng số và dự đoán của hai mô hình
    python scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower   # W0 dựng từ hạt giống so với W0 run đã ghi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flower_lines(history: dict[str, Any]) -> list[str]:
    """Bảng từng vòng từ nhật ký của src/federated/server_app.py."""
    lines = [f"Flower: {history['num_clients_required']} client, {history['rounds']} vòng, seed {history['seed']}, máy chủ {history['server_address']}"]
    checks = passed = 0
    for round_ in history["audit_rounds"]:
        lines.append(f"\nVòng {round_['round']}: {round_['accepted_clients']} client được nhận, {round_['rejected_clients']} bị từ chối; cả vòng {round_['round_fit_elapsed_ms'] / 1000:.2f} giây; "
                     f"gộp {round_['server_aggregate_ms']:.1f} ms; truyền ước tính {round_['transfer_estimate_ms'] / 1000:.2f} giây")
        lines.append(f"  mã băm mô hình chung: trước {round_['parent_model_hash'][:12]}  sau {round_['global_model_hash'][:12]}")
        for client in sorted(round_["clients"], key=lambda c: c["client_id"]):
            checks += 1
            same = bool(client["upload_hash_ok"] and client["client_reported_hash"] == client["server_recomputed_hash"])
            passed += same
            rtt = client.get("probe_rtt_ms")
            lines.append(f"  client {client['client_id']}: {client['num_examples']} ảnh, học {client['local_train_time_s']:.2f} giây, RTT {'chưa đo' if rtt is None else f'{rtt:.1f} ms'}, "
                         f"độ lệch cập nhật {client['received_learnable_update_norm']:.4f}; băm client gửi {client['client_reported_hash'][:12]} / máy chủ tính lại {client['server_recomputed_hash'][:12]} -> {'KHỚP' if same else 'SAI'}")
    lines.append(f"\nTổng cộng {passed}/{checks} lượt băm khớp. Mã băm mô hình cuối: {history['final_model_hash'][:16]}")
    return lines


def run_lines(run: Path, every: int) -> list[str]:
    """Bảng từng vòng từ history.jsonl của scripts/train_v4.py; run cũ không có trường nào thì ghi chưa có."""
    config = read_json(run / "config.json")
    rows = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [f"{run.name}: {config['method']}, đường chạy {config.get('runtime', 'một tiến trình')}, cách chia {config.get('scheme')}, seed {config['seed']}, GPU {config.get('gpu')}, {len(rows)} vòng", ""]
    checks = passed = 0
    for row in rows:
        clients = row["train"].get("clients") or []
        for client in clients:
            for key in ("transport_down_exact", "transport_up_exact"):
                if key in client:
                    checks += 1
                    passed += bool(client[key])
        if row["step"] % every and row["step"] != len(rows) - 1:
            continue
        check = (row["train"].get("aggregation_check") or {}).get("max_abs_diff")
        line = f"vòng {row['step'] + 1:>3}: lr {row['lr']:.4f}, val top-1 {100 * row['val']['top1']:.2f}%, {row['seconds']:.1f} giây"
        if clients:
            line += f", sai số gộp {'chưa có' if check is None else f'{check:.2e}'}"
            line += f", lên {sum(c.get('bytes_up', 0) for c in clients) / 1e6:.1f} MB, xuống {sum(c.get('bytes_down', 0) for c in clients) / 1e6:.1f} MB"
        lines.append(line)
        for client in clients:
            if "l2_to_merged_without_bn" in client:
                lines.append(f"    client {client['client']}: loss {client['loss']:.3f}, {client['steps']} bước, học {client.get('train_seconds', 0):.1f} giây, "
                             f"đã đi {client['delta_l2_without_bn']:.3f}, cách mô hình chung {client['l2_to_merged_without_bn']:.3f}")
    if checks:
        lines.append(f"\nBăm hai chiều: {passed}/{checks} lượt khớp.")
    return lines


def compare_states(a: dict[str, Any], b: dict[str, Any], learnable: set[str]) -> dict[str, Any]:
    """Khoảng cách L2 giữa hai bộ trọng số, tách tham số học được và thống kê BatchNorm; mức trùng tuyệt đối để phân biệt với sai lệch thật."""
    groups = {"learnable": 0.0, "other": 0.0}
    identical = float_tensors = 0
    per_tensor = []
    for key in a:
        if not a[key].is_floating_point():
            continue
        float_tensors += 1
        identical += int(a[key].equal(b[key]))
        square = float(((a[key].double() - b[key].double()) ** 2).sum())
        groups["learnable" if key in learnable else "other"] += square
        if key in learnable:
            per_tensor.append((square ** 0.5, key))
    return {"l2_learnable": groups["learnable"] ** 0.5, "l2_other": groups["other"] ** 0.5, "identical_tensors": identical, "float_tensors": float_tensors, "largest": sorted(per_tensor, reverse=True)[:5]}


def compare_lines(run_a: Path, run_b: Path) -> list[str]:
    import torch

    from src.model.mobilenetv3 import create_model

    state_a = torch.load(run_a / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
    state_b = torch.load(run_b / "checkpoint_best.pt", map_location="cpu", weights_only=False)["model_state"]
    learnable = {name for name, _ in create_model(num_classes=38, pretrained=False).named_parameters()}
    result = compare_states(state_a, state_b, learnable)
    lines = [f"So {run_a.name} với {run_b.name}", f"  L2 tham số học được: {result['l2_learnable']:.4f}", f"  L2 thống kê BatchNorm: {result['l2_other']:.4f}", f"  tensor trùng tuyệt đối: {result['identical_tensors']}/{result['float_tensors']}"]
    lines += ["  năm tensor lệch nhiều nhất:"] + [f"    {name}: {value:.4f}" for value, name in result["largest"]]
    logits_a, logits_b = np.load(run_a / "logits_best.npz", allow_pickle=False), np.load(run_b / "logits_best.npz", allow_pickle=False)
    split = "legacy_diagnostic_test"
    if logits_a[f"{split}_sample_ids"].tolist() != logits_b[f"{split}_sample_ids"].tolist():
        raise SystemExit("hai run có danh sách ảnh tập thử khác nhau")
    preds_a, preds_b = logits_a[split].argmax(1), logits_b[split].argmax(1)
    lines.append(f"  dự đoán trùng nhau trên tập thử: {100 * float(np.mean(preds_a == preds_b)):.2f}% của {len(preds_a)} ảnh")
    return lines


def w0_lines(seed: int, run: Path | None) -> list[str]:
    from scripts.train_v4 import build_model
    from src.model.mobilenetv3 import compute_model_hash, get_parameters

    value = compute_model_hash(get_parameters(build_model(seed)))
    lines = [f"W0 dựng lại từ hạt giống {seed}: {value}"]
    if run is not None:
        recorded = read_json(run / "final.json").get("w0_sha256")
        lines.append(f"W0 run {run.name} đã ghi:     {recorded or 'run này không ghi W0'}")
        if recorded:
            lines.append("-> KHỚP" if recorded == value else "-> KHÁC")
    return lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("flower").add_argument("--history", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run", type=Path, required=True)
    run_parser.add_argument("--every", type=int, default=10, help="in mỗi N vòng")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--a", type=Path, required=True)
    compare_parser.add_argument("--b", type=Path, required=True)
    w0_parser = commands.add_parser("w0")
    w0_parser.add_argument("--seed", type=int, required=True)
    w0_parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "flower":
        lines = flower_lines(read_json(args.history))
    elif args.command == "run":
        lines = run_lines(args.run, args.every)
    elif args.command == "compare":
        lines = compare_lines(args.a, args.b)
    else:
        lines = w0_lines(args.seed, args.run)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
