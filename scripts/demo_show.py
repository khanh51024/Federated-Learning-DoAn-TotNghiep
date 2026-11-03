"""In kết quả kiểm chứng ra màn hình cho dễ xem khi demo (docs/HUONG_DAN_DEMO_THU_CONG.md). Chỉ đọc tệp đã có, không train, không đọc split calibration.

    python scripts/demo_show.py flower --history tmp/demo/history.json     # server_app thủ công: băm, độ trễ, tốc độ từng client từng vòng
    python scripts/demo_show.py flower                                     # lần chạy mới nhất trong artifacts/network_live: đúng lần trang #flower đang hiện
    python scripts/demo_show.py run --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower   # run train_v4: thông số Flower theo vòng
    python scripts/demo_show.py compare --a runs/v4b/centralized_s42 --b runs/v4b/fedavg_s42   # so trọng số và dự đoán của hai mô hình
    python scripts/demo_show.py w0 --seed 42 --run runs/v4_sweep/fedavg_s42_label_alpha_0_1_flower   # W0 dựng từ hạt giống so với W0 run đã ghi
    python scripts/demo_show.py sweep | fairness | partitions | compression   # bảng kết quả đọc từ tệp đã lưu
    python scripts/demo_show.py speed --history tmp/demo_lan/history.json --names "0=Laptop 1,1=Laptop 2"   # bảng gọn cho phép đo giữa các máy
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.run_view import list_runs, load_run, run_view  # noqa: E402
from src.federated.verification_view import SWEEP_ROWS, sweep_view  # noqa: E402,F401


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def view_lines(view: dict[str, Any], server_address: Any = None) -> list[str]:
    """In bản tóm tắt của src/federated/run_view.py: cùng một hàm tính với trang #flower của web, nên số hai bên trùng nhau."""
    address = f", máy chủ {server_address}" if server_address else ""
    lines = [f"Flower: {view['num_clients']} client, {view['total_rounds']} vòng, seed {view['seed']}{address}"]
    for round_ in view["rounds"]:
        transfer = round_["transfer_s"]
        transfer_text = "chưa ước tính" if transfer is None else f"{transfer:.2f} giây"
        lines.append(f"\nVòng {round_['round']}: {round_['accepted']} client được nhận, {round_['rejected']} bị từ chối; cả vòng {round_['elapsed_s']:.2f} giây; "
                     f"gộp {round_['aggregate_ms']:.1f} ms; truyền ước tính {transfer_text}")
        # Độ chính xác do MÁY CHỦ tính trên ảnh giữ riêng của máy chủ (client không gửi độ chính xác).
        if round_["accuracy"] is not None:
            lines.append(f"  máy chủ tự kiểm mô hình chung trên ảnh giữ riêng: độ chính xác {round_['accuracy'] * 100:.2f}%, loss {round_['loss']:.3f}")
        lines.append(f"  mã băm mô hình chung: trước {round_['parent_hash']}  sau {round_['global_hash']}")
        for client in round_["clients"]:
            speed = "" if client["up_mb_s"] is None or client["down_mb_s"] is None else f", lên {client['up_mb_s']:.2f} MiB/s, xuống {client['down_mb_s']:.2f} MiB/s"
            train = "chưa rõ" if client["train_s"] is None else f"{client['train_s']:.2f}"
            who = " (máy chủ ước lượng)" if client["train_is_server_estimate"] else ""
            reply = "" if client["response_s"] is None else f", phản hồi {client['response_s']:.2f} giây"
            rtt = "chưa đo" if client["rtt_ms"] is None else f"{client['rtt_ms']:.1f} ms"
            place = "" if client["machine"] == "chưa rõ" else f" ({client['machine']})"
            learnable = "" if client["learnable_update_norm"] is None else f", độ lệch cập nhật {client['learnable_update_norm']:.4f}"
            lines.append(f"  client {client['client_id']}{place}: {client['num_examples']} ảnh, học {train} giây{who}{reply}, RTT {rtt}{speed}{learnable}")
            if client["update_norm_reported"] is not None:
                lines.append(f"            độ biến thiên: client khai {client['update_norm_reported']:.4f} / máy chủ tự tính {client['update_norm_server']:.4f} -> {'KHỚP' if client['update_norm_matches'] is True else 'SAI'}")
            lines.append(f"            băm client gửi {client['client_hash']} / máy chủ tính lại {client['server_hash']} -> {'KHỚP' if client['hash_ok'] else 'SAI'}")
    if view["client_reported_losses"]:
        lines.append("\nMức sai (loss) qua các vòng: " + " -> ".join(f"{value:.2f}" for _, value in view["client_reported_losses"]))
    checks = view["checks"]
    lines.append(f"\nTổng cộng {checks['hash_ok']}/{checks['hash_total']} lượt băm khớp. Mã băm mô hình cuối: {view['final_model_hash']}")
    if checks["host_check_available"]:
        lines.append(f"Client đến từ {checks['machines']} máy; số client chung máy với máy chủ: {checks['clients_on_server_host']}.")
    if view["final_test"]:
        lines.append(f"Máy chủ kiểm lần cuối trên ảnh giữ riêng: độ chính xác {view['final_test']['accuracy'] * 100:.2f}%, macro-F1 {view['final_test']['macro_f1']:.4f}.")
    for text in (view["shared_machine_note"], view["note"], view["caveat"]):
        if text:
            lines.append(text)
    return lines


def flower_lines(history: dict[str, Any]) -> list[str]:
    """Bảng từng vòng từ nhật ký của src/federated/server_app.py."""
    return view_lines(run_view({"history": history}), history.get("server_address"))


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


def sweep_lines(table: Path) -> list[str]:
    """Bảng độ chính xác lấy từ verification_view: cùng hàm với tab "Phân tán và tập trung" của web."""
    view = sweep_view(table)
    lines = [f"Độ chính xác trên {view['images']} ảnh thử, trung bình ± độ lệch chuẩn của {view['seeds']} lần chạy:", ""]
    for row in view["rows"]:
        gap = "" if row["gap"] is None else f"   thấp hơn tập trung {row['gap']:5.2f} điểm"
        lines.append(f"  {row['label']:<40} {row['top1']:6.2f} ± {row['top1_std']:.2f}%   macro-F1 {row['macro_f1']:6.2f}%{gap}")
    lines += ["", "Tách theo nguồn ảnh (độ chính xác, %):", f"  {'':<40} " + " ".join(f"{name:>12}" for name in view["sources"])]
    for row in view["rows"]:
        lines.append(f"  {row['label']:<40} " + " ".join(f"{row['by_source'][name]:12.2f}" for name in view["sources"]))
    return lines


FAIRNESS_NAMES = {"iid": "chia đều", "label_alpha_100_0": "α=100", "label_alpha_10_0": "α=10", "label_alpha_1_0": "α=1", "label_alpha_0_5": "α=0,5", "label_alpha_0_1": "α=0,1"}
FAIRNESS_METHODS = {"fedavg": "Phân tán tự cài", "fedavg+flower": "Phân tán Flower", "local": "Tự học một mình"}


def fairness_lines(table: Path) -> list[str]:
    """Độ chính xác theo từng client từ table_fairness.csv, cùng phép lọc với báo cáo: tập thử, mọi miền."""
    with table.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "legacy_diagnostic_test" and row["domain"] == "all"]
    lines = ["Độ chính xác trên ảnh thử có tỉ lệ loại bệnh giống dữ liệu của từng client, trung bình 3 lần chạy (%):", "",
             f"  {'Cách học':<18}{'Mức lệch':<10}{'Trung bình':>11}{'Lệch chuẩn':>12}{'Mạnh - yếu':>12}{'Client nhỏ nhất':>17}"]
    for row in rows:
        lines.append(f"  {FAIRNESS_METHODS[row['variant']]:<18}{FAIRNESS_NAMES[row['scheme']]:<10}{float(row['mean_%']):>11.2f}{float(row['std_between_clients_%']):>12.2f}{float(row['gap_best_worst_points']):>12.2f}{float(row['smallest_client_%']):>17.2f}")
    return lines


def partition_lines(manifest: Path) -> list[str]:
    """Các cách chia dữ liệu đã dựng, đọc từ extra_manifest.json; chỉ là dữ liệu, không có số của mô hình."""
    schemes = read_json(manifest)["schemes"]
    names = [("label_alpha_100_0", "Lệch nhãn, α=100"), ("label_alpha_10_0", "Lệch nhãn, α=10"), ("label_alpha_1_0", "Lệch nhãn, α=1"), ("label_alpha_0_5", "Lệch nhãn, α=0,5"),
             ("quantity_alpha_0_5", "Lệch số lượng"), ("source_skew", "Lệch nguồn ảnh")]
    lines = ["Số ảnh của 5 client theo từng cách chia:", ""]
    for key, label in names:
        block = schemes[key]
        counts = " ".join(f"{count:>6}" for count in block["n_k"])
        lines.append(f"  {label:<18} {counts}   máy nhiều nhất gấp {block['size_ratio_max_over_min']:.1f} lần máy ít nhất")
    field = schemes["source_skew"]["heterogeneity"]["field_fraction"]
    lines += ["", "Lệch nguồn ảnh, tỉ lệ ảnh ngoài đồng của từng client: " + ", ".join(f"{100 * field[name]:.0f}%" for name in sorted(field))]
    passed = all(schemes[key]["audit"]["disjoint"] and schemes[key]["audit"]["union_equals_train"] for key, _ in names)
    lines.append("Mỗi cách chia đều dùng hết ảnh học, không client nào trùng ảnh với client khác." if passed else "CẢNH BÁO: có cách chia không đạt kiểm tra.")
    return lines


COMPRESSION_NAMES = (("fp32", "Gốc, không nén"), ("fp16", "Số 16-bit"), ("delta_int8_per_tensor", "Số 8-bit, gửi phần thay đổi"))


def compression_lines(path: Path) -> list[str]:
    """Nén trọng số và mô hình int8, đọc từ artifacts/gd4_compression.json; đo trên một mô hình đã học xong của seed 42."""
    report = read_json(path)
    model = report["models"]["fedavg"]
    full_round_trip_mb = 40 * 5 * 2 * model["weights_transport"]["fp32"]["payload_bytes"] / 1e6
    lines = [f"Mô hình FedAvg α=0,1, seed 42, đo trên {report['count']} ảnh thử. Nén một lần sau khi học xong, CHƯA học với nén ở mỗi vòng.", "", "Nén trọng số khi gửi:"]
    for key, label in COMPRESSION_NAMES:
        row = model["weights_transport"][key]
        lines.append(f"  {label:<30} {row['payload_bytes'] / 1e6:5.2f} MB ({100 * row['payload_ratio']:5.1f}%)   độ chính xác {100 * row['top1']:.2f}%   {row['predictions_changed']:>3} ảnh đổi đáp án   "
                     f"40 vòng ước tính {full_round_trip_mb * row['payload_ratio']:6.0f} MB")
    lines += ["  Gửi toàn bộ ảnh về trung tâm một lần: 1524 MB. Số 40 vòng là phép nhân ước tính, chưa chạy thật.", "", "Mô hình nhẹ để chạy trên thiết bị (ONNX):"]
    onnx, latency = model["onnx"], model["latency"]
    lines.append(f"  {'Gốc float32':<30} {onnx['fp32_bytes'] / 1e6:5.2f} MB   độ chính xác {100 * onnx['onnx_fp32']['top1']:.2f}%   {latency['onnx_fp32_ms']:.2f} ms một ảnh")
    variant = onnx["variants"]["int8_se_gemm_pw_late"]
    lines.append(f"  {'int8 đã chọn':<30} {variant['bytes'] / 1e6:5.2f} MB   độ chính xác {100 * onnx['onnx_int8_se_gemm_pw_late']['top1']:.2f}%   {latency['onnx_int8_se_gemm_pw_late_ms']:.2f} ms một ảnh")
    everything = onnx["variants"]["int8_all"]
    lines.append(f"  {'int8 nén toàn bộ (không dùng)':<30} {everything['bytes'] / 1e6:5.2f} MB   độ chính xác {100 * onnx['onnx_int8_all']['top1']:.2f}%   {latency['onnx_int8_all_ms']:.2f} ms một ảnh")
    lines.append("  Tốc độ đo trên laptop i5-12500H, một luồng CPU; chưa đo trên điện thoại hay máy nhúng.")
    return lines


def speed_lines(history: dict[str, Any], names: dict[int, str]) -> list[str]:
    """Bảng gọn cho phép đo giữa các máy: trung vị qua các vòng, cùng trường với summarize_network_speedtest."""
    by_client: dict[int, list[dict[str, Any]]] = {}
    for round_ in history["audit_rounds"]:
        for client in round_["clients"]:
            by_client.setdefault(int(client["client_id"]), []).append(client)
    lines = [f"Đo qua {len(history['audit_rounds'])} vòng, mỗi máy học {by_client[min(by_client)][0]['num_examples']} ảnh mỗi vòng, số dưới đây là trung vị:", ""]
    train_times = {}
    for client_id in sorted(by_client):
        rows = by_client[client_id]
        train_times[client_id] = statistics.median(row["local_train_time_s"] for row in rows)
        rtt = statistics.median(row["probe_rtt_persistent_ms"] for row in rows)
        up = statistics.median(row["probe_up_mb_per_s"] for row in rows)
        down = statistics.median(row["probe_down_mb_per_s"] for row in rows)
        same = sum(bool(row["upload_hash_ok"] and row["client_reported_hash"] == row["server_recomputed_hash"]) for row in rows)
        lines.append(f"  {names.get(client_id, f'client {client_id}'):<24} học {train_times[client_id]:6.2f} giây | độ trễ {rtt:7.1f} ms | lên {up:8.2f} MiB/s | về {down:8.2f} MiB/s | khớp {same}/{len(rows)}")
    lines += ["", f"Máy nhanh nhất học nhanh gấp {max(train_times.values()) / min(train_times.values()):.1f} lần máy chậm nhất."]
    return lines


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    flower_parser = commands.add_parser("flower", help="không truyền gì = lần chạy mới nhất trong artifacts/network_live, đúng lần web đang hiện")
    flower_parser.add_argument("--history", type=Path, default=None)
    flower_parser.add_argument("--run", type=Path, default=None, help="thư mục một lần chạy trong artifacts/network_live")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run", type=Path, required=True)
    run_parser.add_argument("--every", type=int, default=10, help="in mỗi N vòng")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--a", type=Path, required=True)
    compare_parser.add_argument("--b", type=Path, required=True)
    w0_parser = commands.add_parser("w0")
    w0_parser.add_argument("--seed", type=int, required=True)
    w0_parser.add_argument("--run", type=Path, default=None)
    commands.add_parser("sweep").add_argument("--table", type=Path, default=ROOT / "artifacts" / "sweep_v4" / "table_main.csv")
    commands.add_parser("fairness").add_argument("--table", type=Path, default=ROOT / "artifacts" / "sweep_v4" / "table_fairness.csv")
    commands.add_parser("partitions").add_argument("--manifest", type=Path, default=ROOT / "data" / "dataset_sources" / "mixed" / "pv_pd_v4" / "partitions_extra" / "extra_manifest.json")
    commands.add_parser("compression").add_argument("--report", type=Path, default=ROOT / "artifacts" / "gd4_compression.json")
    speed_parser = commands.add_parser("speed")
    speed_parser.add_argument("--history", type=Path, required=True)
    speed_parser.add_argument("--names", default="", help="client_id=tên máy, cách nhau bởi dấu phẩy")
    args = parser.parse_args()
    if args.command == "flower":
        if args.history is not None:
            lines = flower_lines(read_json(args.history))
        else:
            runs = [args.run] if args.run is not None else list_runs(ROOT / "artifacts" / "network_live")[:1]
            if not runs:
                raise SystemExit("Chưa có lần chạy nào trong artifacts/network_live; truyền --history hoặc --run")
            lines = view_lines(run_view(load_run(runs[0])))
    elif args.command == "run":
        lines = run_lines(args.run, args.every)
    elif args.command == "compare":
        lines = compare_lines(args.a, args.b)
    elif args.command == "sweep":
        lines = sweep_lines(args.table)
    elif args.command == "fairness":
        lines = fairness_lines(args.table)
    elif args.command == "partitions":
        lines = partition_lines(args.manifest)
    elif args.command == "compression":
        lines = compression_lines(args.report)
    elif args.command == "speed":
        names = {int(item.split("=", 1)[0]): item.split("=", 1)[1] for item in args.names.split(",") if item}
        lines = speed_lines(read_json(args.history), names)
    else:
        lines = w0_lines(args.seed, args.run)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
