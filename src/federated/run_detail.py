"""Chi tiết của từng mục trên trang #flower: số đó làm ra thế nào, kèm các dòng terminal.

Web (/api/flower/detail) và terminal (scripts/demo_show.py flower --detail ...) cùng gọi detail_view, nên ra đúng cùng các dòng.
Số lấy từ run_view; vài trường run_view không trả (thời gian đo mạng, số nguồn mạng, kết quả từng lớp) đọc thêm từ bản ghi gốc.
Kết quả không chứa IP, flower_cid, đường dẫn tuyệt đối. Chỉ đọc kết quả tập val và test, không đọc split calibration.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.federated.run_view import AGGREGATION_TOLERANCE, _client_view, _machine_labels, list_runs, load_run, run_view

LOG_FILE = "server.log"
LOG_TAIL_LINES = 80
MAX_LOG_LINE_CHARS = 300
PROOF_ITEMS = ("hash", "norm", "rounds", "host", "bytes", "accuracy")
ITEM = re.compile(r"^(?:client:\d{1,3}|machine:(?:[A-Z]|\d{1,3})|server|hash|norm|rounds|host|bytes|accuracy|validator|tamper)$")
SOURCE_LINES = "Máy chủ ghi trong lúc đo (progress.json, history.json). Lệnh trên in đúng các dòng này."
CLIENT_WINDOW_NOTE = ("Cửa sổ client nằm trên laptop và không gửi nhật ký lên máy chủ. Đây là những gì máy chủ nhận và đo được, không phải màn hình của laptop. "
                      "Muốn xem màn hình client, mở cửa sổ client trên laptop và so mã băm với trang này.")
CLIENT_SENDS = ("trọng số", "số ảnh", "độ biến thiên", "mã băm trọng số gửi đi", "mã băm mô hình vừa nhận", "số vòng, số epoch, seed", "mã client, mã băm tên máy", "mã danh sách ảnh, ảnh thật hay giả")
CLIENT_NOT_SENT = ("ảnh", "độ chính xác", "loss", "thời gian học", "độ trễ, tốc độ mạng")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_IPV4 = re.compile(r"\b(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b")
_CID = re.compile(r"\b[0-9a-f]{32}\b")
_ABS_PATH = re.compile(r"(?:/home/|/root/|/tmp/|[A-Za-z]:[\\/])[^\s'\",)]*")


def clean_log_line(line: str) -> str:
    """Bỏ mã màu; che IP, flower_cid và đường dẫn tuyệt đối để nhật ký thật hiện được trên trang công khai."""
    out = _ANSI.sub("", line.rstrip("\r\n"))
    out = _IPV4.sub("[đã che IP]", out)
    out = _CID.sub("[đã che mã]", out)
    out = _ABS_PATH.sub("[đã che đường dẫn]", out)
    return out[:MAX_LOG_LINE_CHARS]


def read_log_tail(run_dir: Path, limit: int = LOG_TAIL_LINES) -> Optional[dict[str, Any]]:
    """Các dòng cuối của nhật ký thật. Không có tệp thì trả None."""
    try:
        raw = (Path(run_dir) / LOG_FILE).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    # Bảng tổng kết cuối của Flower dài hàng trăm dòng và lặp lại số đã có, sẽ che mất các dòng theo vòng: bỏ đi và báo số dòng đã bỏ.
    kept: list[str] = []
    skipped = 0
    in_summary = False
    for line in raw:
        cleaned = clean_log_line(line)
        if "[SUMMARY]" in cleaned:
            in_summary = True
        elif in_summary and cleaned.startswith("[+]"):
            in_summary = False
        if in_summary:
            skipped += 1
        else:
            kept.append(cleaned)
    return {"total_lines": len(raw), "skipped_summary_lines": skipped, "lines": kept[-limit:]}


def _clients(view: dict[str, Any]) -> list[int]:
    ids = {c["client_id"] for r in view["rounds"] for c in r["clients"]}
    ids.update(reply.get("client_id") for reply in view["round_replies"] if isinstance(reply.get("client_id"), int))
    return sorted(ids | set(range(int(view["num_clients"] or 0))))


def _pick_round(view: dict[str, Any], round_no: Optional[int]) -> Optional[dict[str, Any]]:
    rows = view["rounds"]
    if round_no is None:
        return rows[-1] if rows else None
    return next((row for row in rows if row["round"] == round_no), None)


def _raw_audits(record: dict[str, Any]) -> list[dict[str, Any]]:
    source = record.get("history") or record.get("progress") or {}
    return [a for a in (source.get("audit_rounds") or source.get("rounds") or []) if isinstance(a, dict)]


def _raw_client(audits: list[dict[str, Any]], round_no: int, client_id: int) -> dict[str, Any]:
    for audit in audits:
        if int(audit.get("round", 0)) == round_no:
            for client in audit.get("clients") or []:
                if int(client.get("client_id", -1)) == client_id:
                    return client
    return {}


def _text(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "chưa đo"
    return f"{value:.{digits}f}{unit}"


def _yes(ok: Any) -> str:
    return "KHỚP" if ok else "SAI"


def _mib(num_bytes: Any) -> str:
    return f"{float(num_bytes or 0) / (1024 * 1024):.1f} MiB"


def _client_detail(view: dict[str, Any], audits: list[dict[str, Any]], client_id: int, round_no: Optional[int]) -> dict[str, Any]:
    row = _pick_round(view, round_no)
    client = next((c for c in row["clients"] if c["client_id"] == client_id), None) if row else None
    lines: list[str] = []
    if view["state"] == "running" and round_no is None:
        reply = next((r for r in view["round_replies"] if r.get("client_id") == client_id), None)
        current = view["round_in_progress"]
        lines.append(f"Vòng {current} đang chạy: đã gửi về sau {reply['response_ms'] / 1000:.1f} giây." if reply and reply.get("response_ms") is not None
                     else f"Vòng {current} đang chạy: máy chủ chưa nhận trọng số của client này.")
    if client is None:
        lines.append("Client này chưa có số ở vòng đang xem." if row else "Chưa xong vòng nào nên chưa có số.")
    else:
        raw = _raw_client(audits, row["round"], client_id)
        lines += [
            f"Vòng {row['round']}, client {client_id} ({client['machine']}), trạng thái {client['status']}",
            f"  dữ liệu: {client['num_examples']} ảnh {'thật' if client['data_mode'] == 'real' else client['data_mode']}, mã danh sách ảnh {str(raw.get('client_manifest_sha256') or '')[:12]}",
            "CLIENT GỬI LÊN",
            f"  trọng số: {client['payload_bytes_up']} byte, mã băm {client['client_hash']}",
            f"  độ biến thiên khai: {_text(client['update_norm_reported'], 4)}",
            "MÁY CHỦ TỰ TÍNH VÀ TỰ ĐO",
            f"  mã băm tính lại: {client['server_hash']} -> {_yes(client['hash_ok'])}",
            f"  mô hình gửi xuống {row['parent_hash']}, client báo đã nhận {client['reported_parent']} -> {_yes(client['parent_hash_ok'])}" if client["reported_parent"] else "  mô hình client báo đã nhận: lần đo này chưa ghi mã",
            f"  độ biến thiên tính lại: {_text(client['update_norm_server'], 4)} -> {_yes(client['update_norm_matches'])}",
            f"  RTT {_text(client['rtt_ms'], 1, ' ms')}, lên {_text(client['up_mb_s'], 2, ' MiB/s')}, xuống {_text(client['down_mb_s'], 2, ' MiB/s')}",
            f"  phản hồi {_text(client['response_s'], 2, ' giây')}",
            f"  học (ước lượng) = {_text(client['response_s'], 2)} - đo mạng {_text((raw.get('probe_total_ms') or 0) / 1000, 2)} - truyền dự đoán {_text((raw.get('predicted_transfer_ms') or 0) / 1000, 2)} = {_text(client['train_s'], 2, ' giây')}",
        ]
    history = [(r["round"], c["train_s"]) for r in view["rounds"] for c in r["clients"] if c["client_id"] == client_id and c["train_s"] is not None]
    if history:
        lines.append("THỜI GIAN HỌC QUA CÁC VÒNG (giây)")
        lines.append("  " + "  ".join(f"{number}:{seconds:.0f}" for number, seconds in history))
    return {
        "title": f"Client {client_id}",
        "round_bound": True,
        "how": [
            "Máy chủ gửi mô hình chung của vòng xuống. Client tính mã băm bản vừa nhận để máy chủ đối chiếu.",
            f"Client học {view['local_epochs'] or '?'} epoch trên ảnh riêng. Ảnh không rời khỏi máy.",
            "Client gửi lên trọng số, số ảnh, độ biến thiên và các mã để máy chủ đối chiếu. Client không gửi độ chính xác, loss hay thời gian.",
            "Máy chủ tính lại mã băm và độ biến thiên từ chính trọng số nhận được. Lệch thì loại client đó khỏi vòng.",
            "Máy chủ tự bấm giờ và tự đo mạng. Thời gian học là ước lượng: phản hồi trừ đo mạng và truyền dự đoán.",
        ],
        "lines": lines,
        "note": CLIENT_WINDOW_NOTE,
        "sent": list(CLIENT_SENDS),
        "not_sent": list(CLIENT_NOT_SENT),
    }


def _machine_detail(view: dict[str, Any], audits: list[dict[str, Any]], label: str) -> Optional[dict[str, Any]]:
    machine = next((m for m in view["machines"] if m["label"] == label), None)
    if machine is None:
        return None
    ids = set(machine["client_ids"])
    raw = [c for a in audits for c in a.get("clients") or [] if int(c.get("client_id", -1)) in ids]
    codes = sorted({str(c.get("machine_id")) for c in raw if c.get("machine_id")})
    sources = {c.get("peer_hash") for c in raw if c.get("peer_hash")}
    lines = [
        f"{label}: {len(ids)} client ({', '.join(str(i) for i in sorted(ids))}), mã máy {', '.join(codes) or 'chưa có'}",
        f"  chung máy với máy chủ: {'CÓ' if machine['same_host_as_server'] else 'không'}",
        f"  số nguồn mạng máy chủ thấy: {len(sources)}",
        "TRUNG VỊ QUA CÁC VÒNG, MỖI CLIENT",
        f"  RTT {_text(machine['rtt_ms'], 1, ' ms')}, lên {_text(machine['up_mb_s'], 2, ' MiB/s')}, xuống {_text(machine['down_mb_s'], 2, ' MiB/s')}, học {_text(machine['train_s'], 1, ' giây')}",
        "CLIENT CHẬM NHẤT CỦA MÁY THEO VÒNG (giây phản hồi)",
    ]
    for row in view["rounds"]:
        replies = [c["response_s"] for c in row["clients"] if c["client_id"] in ids and c["response_s"] is not None]
        if replies:
            lines.append(f"  vòng {row['round']:>2}: {max(replies):.1f}")
    return {
        "title": label,
        "round_bound": False,
        "how": [
            "Mỗi client gửi mã băm của tên máy, không gửi tên máy. Cùng mã nghĩa là cùng một máy.",
            "Mỗi vòng, máy chủ gửi khối byte cùng cỡ mô hình và tự đo độ trễ, tốc độ lên, xuống trên đồng hồ của nó. Client chỉ làm theo, không gửi số nào.",
            "Máy chủ so IP nối đến với IP của chính nó để biết client có chung máy không, rồi chỉ lưu mã băm có muối của IP.",
            "Các client trên một máy dùng chung CPU và đường truyền, nên tốc độ mỗi client chỉ là phần chia của đường truyền đó.",
        ],
        "lines": lines,
        "note": "Số nguồn mạng lớn hơn 1 nghĩa là IP ra của máy đó đã đổi trong lúc đo, ví dụ khi dùng VPN.",
    }


def _server_detail(view: dict[str, Any], round_no: Optional[int]) -> dict[str, Any]:
    row = _pick_round(view, round_no)
    lines = [f"Máy chủ: {view['num_clients']} client, tối thiểu {view['min_clients']}, {view['total_rounds']} vòng, {view['local_epochs']} epoch mỗi vòng, seed {view['seed']}"]
    if row is None:
        lines.append("Chưa xong vòng nào nên chưa có số.")
    else:
        lines += [
            f"Vòng {row['round']}: nhận {row['accepted']}, loại {row['rejected']}, nghi ngờ {row['suspicious']}",
            f"  cả vòng {_text(row['elapsed_s'], 2, ' giây')} = học và chờ {_text(row['learn_and_wait_s'], 2)} + truyền {_text(row['transfer_s'], 2)} + máy chủ {_text(row['server_s'], 2)}",
            f"  gộp FedAvg mất {_text(row['aggregate_ms'], 1, ' ms')}",
            f"  tính lại phép gộp độc lập: lệch lớn nhất {row['aggregation_max_diff']:.2e} -> {_yes(row['aggregation_max_diff'] <= AGGREGATION_TOLERANCE)}" if row["aggregation_max_diff"] is not None else "  tính lại phép gộp độc lập: lần đo này chưa ghi",
            f"  mô hình trước {row['parent_hash']} -> sau {row['global_hash']}",
        ]
        if row["accuracy"] is not None:
            lines.append(f"  máy chủ tự kiểm: độ chính xác {row['accuracy'] * 100:.2f}%, loss {row['loss']:.4f}")
    if view["final_model_hash"]:
        lines.append(f"Mô hình cuối: {view['final_model_hash']}")
    return {
        "title": "Máy chủ",
        "round_bound": True,
        "how": [
            "Máy chủ chạy riêng trên VM bằng scripts/remote/run_network_server.sh và chờ đủ client mới bắt đầu.",
            "Mỗi vòng, máy chủ gửi mô hình chung xuống rồi chờ đủ các client trả về.",
            "Kiểm từng client: mã băm, độ biến thiên, vòng, seed, danh sách ảnh và ảnh thật. Sai thì loại.",
            "Gộp FedAvg: mô hình mới là trung bình trọng số của các client, theo tỉ lệ số ảnh của từng client. Sau đó tính lại phép gộp bằng một đoạn mã độc lập rồi so.",
            "Tự kiểm mô hình mới trên ảnh giữ riêng của máy chủ. Client không có ảnh này.",
        ],
        "lines": lines,
        "note": None,
    }


def _hash_detail(view: dict[str, Any]) -> dict[str, Any]:
    checks = view["checks"]
    lines = [f"Tổng: {checks['hash_ok']}/{checks['hash_total']} lượt mã băm khớp; chuỗi mô hình {'LIỀN' if checks['chain_ok'] else 'ĐỨT'}",
             f"Client báo nhận đúng mô hình: {checks['parent_ok']}/{checks['parent_total']}" if checks["parent_total"] else "Client báo nhận đúng mô hình: lần đo này chưa ghi mã"]
    previous = None
    for row in view["rounds"]:
        ok = sum(c["hash_ok"] for c in row["clients"])
        link = "đầu chuỗi" if previous is None else ("nối đúng" if previous == row["parent_hash"] else "ĐỨT")
        lines.append(f"  vòng {row['round']:>2}: {row['parent_hash']} -> {row['global_hash']}  {link}, khớp {ok}/{len(row['clients'])}")
        previous = row["global_hash"]
    bad = [(row["round"], c) for row in view["rounds"] for c in row["clients"] if not c["hash_ok"]]
    lines.append("Lượt sai: " + ("không có" if not bad else ""))
    lines += [f"  vòng {number}, client {c['client_id']}: client {c['client_hash']} / máy chủ {c['server_hash']}" for number, c in bad]
    if view["final_model_hash"]:
        lines.append(f"Mô hình cuối: {view['final_model_hash']}")
    return {
        "title": "Trọng số gửi đúng",
        "round_bound": False,
        "how": [
            "Mã băm giống dấu vân tay của bộ trọng số: chỉ cần đổi một con số là mã đổi hẳn.",
            "Chiều xuống: client băm mô hình vừa nhận rồi báo lại, máy chủ so với mô hình đã gửi.",
            "Chiều lên: client băm SHA-256 toàn bộ trọng số sắp gửi (hàm compute_model_hash, src/model/mobilenetv3.py).",
            "Máy chủ chạy đúng hàm đó trên trọng số nhận được rồi so hai mã. Lệch nghĩa là trọng số đã đổi trên đường truyền hoặc client khai sai, và client đó bị loại.",
            "Mô hình cuối vòng này phải là mô hình đầu vòng sau. Nhờ vậy không vòng nào bị thay mô hình giữa chừng.",
        ],
        "lines": lines,
        "note": "Đây là kiểm tra toàn vẹn, không phải xác thực: máy chủ đo không có mật khẩu hay mã hóa.",
    }


def _norm_detail(view: dict[str, Any]) -> dict[str, Any]:
    checks = view["checks"]
    lines = [f"Tổng: {checks['norm_ok']}/{checks['norm_total']} lượt khớp"]
    largest = 0.0
    for row in view["rounds"]:
        pairs = [(c["update_norm_reported"], c["update_norm_server"]) for c in row["clients"] if c["update_norm_reported"] is not None and c["update_norm_server"] is not None]
        if not pairs:
            lines.append(f"  vòng {row['round']:>2}: client chưa khai")
            continue
        gap = max(abs(a - b) for a, b in pairs)
        largest = max(largest, gap)
        ok = sum(c["update_norm_matches"] is True for c in row["clients"])
        low, high = min(b for _, b in pairs), max(b for _, b in pairs)
        lines.append(f"  vòng {row['round']:>2}: khớp {ok}/{len(row['clients'])}, máy chủ tính {low:.4f} đến {high:.4f}, lệch lớn nhất {gap:.4f}")
    lines.append(f"Lệch lớn nhất cả lần đo: {largest:.4f}")
    return {
        "title": "Độ biến thiên khớp",
        "round_bound": False,
        "how": [
            "Độ biến thiên là độ lớn phần trọng số thay đổi so với mô hình nhận được: căn của tổng bình phương các thay đổi.",
            "Client tính số này trước khi gửi và khai kèm trọng số.",
            "Máy chủ tự tính lại từ trọng số nhận được, không tin số client khai.",
            "Lệch quá 0,1% thì client đó bị loại khỏi vòng.",
        ],
        "lines": lines,
        "note": "Client khai làm tròn 4 chữ số nên số lệch rất nhỏ là bình thường.",
    }


def _rounds_detail(view: dict[str, Any]) -> dict[str, Any]:
    checks = view["checks"]
    lines = [f"Tổng: {checks['full_rounds'] or 0}/{checks['rounds_total']} vòng đủ {view['num_clients']} client; tối thiểu để bắt đầu {view['min_clients']}"]
    lines += [f"  vòng {row['round']:>2}: nhận {row['accepted']}, loại {row['rejected']}, nghi ngờ {row['suspicious']}, cả vòng {_text(row['elapsed_s'], 1, ' giây')}" for row in view["rounds"]]
    return {
        "title": "Vòng nhận đủ client",
        "round_bound": False,
        "how": [
            "Máy chủ chỉ bắt đầu vòng khi đã nối đủ số client tối thiểu, nên không client nào vào trễ.",
            "Máy chủ chờ tất cả client trả về rồi mới gộp. Vì vậy thời gian vòng bằng thời gian của client chậm nhất.",
            "Client nào không qua kiểm tra thì bị đếm vào cột loại.",
        ],
        "lines": lines,
        "note": None,
    }


def _host_detail(view: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    checks = view["checks"]
    available = checks["host_check_available"]
    lines = [f"Client chung máy với máy chủ: {checks['clients_on_server_host'] if available else 'chưa kiểm'}; client đến từ {checks['machines']} máy"]
    for machine in view["machines"]:
        ids = set(machine["client_ids"])
        sources = {c.get("peer_hash") for a in audits for c in a.get("clients") or [] if int(c.get("client_id", -1)) in ids and c.get("peer_hash")}
        lines.append(f"  {machine['label']}: client {', '.join(str(i) for i in sorted(ids))}; chung máy chủ: {'CÓ' if machine['same_host_as_server'] else 'không'}; RTT {_text(machine['rtt_ms'], 1, ' ms')}; {len(sources)} nguồn mạng")
    return {
        "title": "Client chung máy chủ",
        "round_bound": False,
        "how": [
            "Mỗi client mở kết nối đo mạng tới máy chủ. Máy chủ thấy IP nguồn của kết nối đó.",
            "Nếu IP nguồn là 127.x, ::1 hoặc trùng IP của máy chủ thì client nằm trên chính máy chủ.",
            "Máy chủ chỉ lưu kết quả và mã băm có muối của IP. Trang này không hiện IP.",
            "Số này phải bằng 0 thì mới gọi là máy chủ tách riêng.",
        ],
        "lines": lines,
        "note": None,
    }


def _bytes_detail(view: dict[str, Any]) -> dict[str, Any]:
    checks = view["checks"]
    lines = [f"Tổng: lên {_mib(checks['bytes_up'])}, xuống {_mib(checks['bytes_down'])}, cộng {_mib(checks['bytes_up'] + checks['bytes_down'])}"]
    for row in view["rounds"]:
        up = sum(int(c["payload_bytes_up"] or 0) for c in row["clients"])
        down = sum(int(c["payload_bytes_down"] or 0) for c in row["clients"])
        lines.append(f"  vòng {row['round']:>2}: lên {_mib(up)}, xuống {_mib(down)}, truyền ước lượng {_text(row['transfer_s'], 1, ' giây')}")
    sample = next((c for r in view["rounds"] for c in r["clients"]), None)
    if sample:
        lines.append(f"Mỗi lượt một client: lên {sample['payload_bytes_up']} byte, xuống {sample['payload_bytes_down']} byte")
    return {
        "title": "Đã truyền",
        "round_bound": False,
        "how": [
            "Máy chủ đếm số byte của trọng số đúng như Flower đóng gói để gửi, ở cả hai chiều, cho từng client từng vòng.",
            "Phép đo mạng gửi thêm một khối byte mỗi chiều. Phần đó không tính vào số này.",
            "Thời gian truyền mỗi vòng là ước lượng: số byte chia cho tốc độ máy chủ đo được, lấy client lâu nhất.",
        ],
        "lines": lines,
        "note": None,
    }


def _accuracy_detail(view: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    lines: list[str] = []
    final = view["final_test"]
    if final:
        source = (record.get("history") or record.get("progress") or {}).get("final_test_metrics")
        per_class = source.get("per_class_metrics") if isinstance(source, dict) else None
        classes = {str(k): v for k, v in per_class.items() if isinstance(v, dict)} if isinstance(per_class, dict) else {}
        images = sum(int(v.get("support") or 0) for v in classes.values())
        lines.append(f"KẾT QUẢ CUỐI TRÊN TẬP TEST{f' ({images} ảnh)' if images else ''}: độ chính xác {final['accuracy'] * 100:.2f}%, macro-F1 {final['macro_f1']:.4f}, loss {final['loss']:.4f}")
        weakest = sorted(((k, v) for k, v in classes.items() if isinstance(v.get("f1"), (int, float))), key=lambda item: (item[1]["f1"], item[0]))[:5]
        if weakest:
            lines.append("5 LỚP YẾU NHẤT TRÊN TẬP TEST (số thứ tự lớp)")
            lines += [f"  lớp {key:>2}: F1 {value['f1']:.3f}, {value.get('support', '?')} ảnh" for key, value in weakest]
    else:
        lines.append("Chưa có kết quả cuối trên tập test.")
    val_images = f" ({view['eval_images']} ảnh)" if view["eval_images"] else ""
    lines.append(f"TRONG LÚC HỌC, MÁY CHỦ KIỂM SAU MỖI VÒNG TRÊN TẬP VAL{val_images}")
    lines += [f"  vòng {row['round']:>2}: {_text(row['accuracy'] * 100 if row['accuracy'] is not None else None, 2, '%')}, loss {_text(row['loss'], 4)}" for row in view["rounds"]]
    return {
        "title": "Độ chính xác máy chủ tính",
        "round_bound": False,
        "how": [
            "Client không gửi độ chính xác. Máy chủ tự tính trên ảnh của nó, client không có các ảnh này.",
            "Vì vậy 2 số khác nhau: biểu đồ từng vòng là tập val, số cuối là tập test.",
            "Sau mỗi vòng, máy chủ kiểm mô hình vừa gộp trên tập val.",
            "Khi học xong, máy chủ kiểm mô hình cuối một lần trên tập test.",
        ],
        "lines": lines,
        "note": view["caveat"],
    }


def _validator_detail(view: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    validation = view["validation"]
    if validation is None:
        lines = ["Lần đo này chưa có kết quả của trình kiểm."]
    else:
        checks = (record.get("validation") or {}).get("checks") or {}
        lines = [f"Kết quả: {'ĐẠT' if validation['ok'] else 'CHƯA ĐẠT'}, {validation['passed']}/{validation['total']} mục"]
        lines += [f"  {'đạt ' if value is True else 'SAI '} {name}" for name, value in sorted(checks.items())]
    return {
        "title": "Trình kiểm lần đo",
        "round_bound": False,
        "how": [
            "Sau khi đo xong, một chương trình riêng (scripts/validate_network_artifact.py) đọc lại tệp kết quả và tệp mô hình.",
            "Nó kiểm đủ vòng, đủ client, ảnh thật, đúng seed, và mã băm tệp mô hình cuối khớp với kết quả.",
            "Chương trình này không chạy trong lúc học, nên không phụ thuộc vào phần đang được kiểm.",
        ],
        "lines": lines,
        "note": None,
    }


def tamper_view(root: Path) -> Optional[dict[str, Any]]:
    """Lần thử cố tình đổi trọng số mới nhất: client nào bị loại, 2 mã băm, lý do. Không có thì trả None."""
    runs = list_runs(root)
    if not runs:
        return None
    record = load_run(runs[0])
    failed = (record.get("progress") or {}).get("failed_round") or {}
    clients = [c for c in failed.get("clients") or [] if isinstance(c, dict)]
    labels = _machine_labels([{"clients": clients}])
    rows = sorted((_client_view(c, labels) for c in clients), key=lambda c: c["client_id"])
    rejected = [c for c in rows if c["status"] != "ok"]
    if not rejected:
        return None
    return {"run_id": record["run_id"], "round": failed.get("round"), "clients": rows, "rejected": rejected[0], "error": (record.get("progress") or {}).get("error")}


def _tamper_detail(root: Path) -> Optional[dict[str, Any]]:
    tamper = tamper_view(root)
    if tamper is None:
        return None
    lines = [f"Lần thử {tamper['run_id']}, vòng {tamper['round']}: máy chủ trên VM, client từ laptop qua Internet"]
    for client in tamper["clients"]:
        verdict = "KHỚP, nhận" if client["hash_ok"] else f"LỆCH, LOẠI ({client['validation_reason']})"
        lines.append(f"  client {client['client_id']}: client gửi {client['client_hash']} / máy chủ tính lại {client['server_hash']} -> {verdict}")
    if tamper["error"]:
        lines.append(f"Máy chủ dừng lần đo: {tamper['error']}")
    return {
        "title": "Thử cố tình sai",
        "item": "tamper",
        "run_id": tamper["run_id"],
        "state": "failed",
        "round": None,
        "round_bound": False,
        "how": [
            "Một client được chạy ở chế độ cố tình sai: sau khi đã băm, nó đổi đúng một con số trong trọng số, giống như dữ liệu bị đổi trên đường truyền.",
            "Máy chủ băm lại trọng số nhận được, thấy khác mã client gửi, và loại client đó.",
            "Khi có client bị loại, máy chủ dừng cả lần đo chứ không gộp tiếp, nên mô hình chung không bị ảnh hưởng.",
        ],
        "lines": lines,
        "note": "Đây là kiểm tra toàn vẹn, không phải xác thực: máy chủ đo không có mật khẩu hay mã hóa, nên người cố ý sửa cả trọng số lẫn mã băm thì phép kiểm này không bắt được.",
        "command": "python scripts/demo_show.py flower --detail tamper",
        "source": SOURCE_LINES,
        "log": None,
    }


def detail_view(record: dict[str, Any], item: str, round_no: Optional[int] = None, now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """Chi tiết một mục. Trả None khi mục hoặc vòng không tồn tại."""
    if not ITEM.fullmatch(item):
        return None
    view = run_view(record, now=now)
    if not view.get("available"):
        return None
    if round_no is not None and not any(row["round"] == round_no for row in view["rounds"]):
        return None
    audits = _raw_audits(record)
    kind, _, ident = item.partition(":")
    if kind == "client":
        if int(ident) not in _clients(view):
            return None
        built: Optional[dict[str, Any]] = _client_detail(view, audits, int(ident), round_no)
    elif kind == "machine":
        built = _machine_detail(view, audits, f"Máy {ident}")
    elif kind == "server":
        built = _server_detail(view, round_no)
    elif kind == "hash":
        built = _hash_detail(view)
    elif kind == "norm":
        built = _norm_detail(view)
    elif kind == "rounds":
        built = _rounds_detail(view)
    elif kind == "host":
        built = _host_detail(view, audits)
    elif kind == "bytes":
        built = _bytes_detail(view)
    elif kind == "validator":
        built = _validator_detail(view, record)
    else:
        built = _accuracy_detail(view, record)
    if built is None:
        return None
    run_id = record.get("run_id")
    shown_round = f" --round {round_no}" if round_no is not None and built["round_bound"] else ""
    return {
        **built,
        "item": item,
        "run_id": run_id,
        "state": view["state"],
        "round": round_no if built["round_bound"] else None,
        "command": f"python scripts/demo_show.py flower --run artifacts/network_live/{run_id} --detail {item}{shown_round}" if run_id else None,
        "source": SOURCE_LINES,
        "log": None,
    }


def detail_for_run(run_dir: Path, item: str, round_no: Optional[int] = None, now: Optional[datetime] = None, tamper_root: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Đọc một lần đo và dựng chi tiết. Riêng mục máy chủ kèm nhật ký thật; mục thử cố tình sai đọc thư mục riêng của nó."""
    if item == "tamper":
        return _tamper_detail(tamper_root) if tamper_root is not None and round_no is None else None
    built = detail_view(load_run(run_dir), item, round_no, now=now)
    if built is not None and item == "server":
        built["log"] = read_log_tail(run_dir)
        if built["log"]:
            built["log"]["title"] = log_title(built["log"])
    return built


def log_title(log: dict[str, Any]) -> str:
    skipped = f", đã bỏ {log['skipped_summary_lines']} dòng bảng tổng kết của Flower" if log["skipped_summary_lines"] else ""
    return f"Nhật ký thật của máy chủ ({LOG_FILE}): {len(log['lines'])} dòng cuối trong {log['total_lines']} dòng{skipped}"


def detail_lines(built: dict[str, Any]) -> list[str]:
    """Bản in terminal của `demo_show flower --detail`, đúng các dòng mà web hiện."""
    lines = [built["title"]]
    lines += [f"  {index}. {step}" for index, step in enumerate(built["how"], 1)]
    if built.get("sent"):
        lines += ["  Client gửi: " + ", ".join(built["sent"]), "  Client không gửi: " + ", ".join(built["not_sent"])]
    lines += [""] + built["lines"]
    if built["note"]:
        lines += ["", built["note"]]
    if built["log"]:
        lines += ["", log_title(built["log"])] + built["log"]["lines"]
    return lines
