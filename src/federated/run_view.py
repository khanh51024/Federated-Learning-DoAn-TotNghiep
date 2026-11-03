"""Một chỗ tính duy nhất cho lần chạy Flower qua mạng: web (/api/flower/live) và terminal (scripts/demo_show.py) cùng gọi, nên số hai bên luôn trùng nhau.

Đọc `progress.json` (server ghi liên tục) và `history.json` (server ghi khi xong), trả một bản tóm tắt để hiển thị. Không nạp flwr hay torch.
Kết quả không chứa IP: server chỉ lưu mã băm có muối của nguồn nối đến, nên trả thẳng ra trang công khai được.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILE = "progress.json"
HISTORY_FILE = "history.json"
# Server ghi nhịp mỗi HEARTBEAT_S giây kể cả khi đang chờ; quá STALE_AFTER_S không có gì mới nghĩa là server đã chết hoặc mất kết nối.
HEARTBEAT_S = 20
STALE_AFTER_S = 90
CAVEAT_PRETRAINED = ("Phép đo này dùng tập ảnh nhỏ và mô hình khởi tạo từ ImageNet để chứng minh hệ thống chạy đúng qua mạng thật. "
                     "Đây không phải độ chính xác của đề tài; kết quả chính xem ở bảng so sánh phân tán và tập trung.")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):       # chưa có tệp, hoặc đang được thay đúng lúc đọc: coi như chưa có, lần hỏi sau sẽ đọc được
        return None
    return data if isinstance(data, dict) else None


def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    return {"run_id": run_dir.name, "progress": _read_json(run_dir / PROGRESS_FILE), "history": _read_json(run_dir / HISTORY_FILE)}


def list_runs(root: Path) -> list[Path]:
    """Các thư mục lần chạy, mới nhất trước (tên thư mục bắt đầu bằng giờ UTC nên xếp theo tên là xếp theo thời gian)."""
    root = Path(root)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and ((p / PROGRESS_FILE).is_file() or (p / HISTORY_FILE).is_file())]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _num(value: Any, digits: Optional[int] = None) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), digits) if digits is not None else float(value)


def _seconds(ms: Any) -> Optional[float]:
    value = _num(ms)
    return round(value / 1000.0, 3) if value is not None else None


def _median(values: list[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 3) if clean else None


def _hash_ok(client: dict[str, Any]) -> bool:
    return bool(client.get("upload_hash_ok") and client.get("client_reported_hash") == client.get("server_recomputed_hash"))


def _client_view(client: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    rtt = client.get("probe_rtt_persistent_ms", client.get("probe_rtt_ms"))
    return {
        "client_id": int(client.get("client_id", -1)),
        "machine": labels.get(_machine_key(client), "chưa rõ"),
        "num_examples": client.get("num_examples"),
        "status": client.get("status"),
        "rtt_ms": _num(rtt, 3),
        "up_mb_s": _num(client.get("probe_up_mb_per_s"), 3),
        "down_mb_s": _num(client.get("probe_down_mb_per_s"), 3),
        "response_s": _seconds(client.get("response_ms")),
        "train_s": _num(client.get("local_train_time_s"), 3),
        "train_is_server_estimate": client.get("local_train_time_source", "client_measured") != "client_measured",
        "update_norm_reported": _num(client.get("update_norm_reported"), 4),
        "update_norm_server": _num(client.get("received_update_norm"), 4),
        "update_norm_matches": client.get("update_norm_matches"),
        "learnable_update_norm": _num(client.get("received_learnable_update_norm"), 4),
        "hash_ok": _hash_ok(client),
        "client_hash": str(client.get("client_reported_hash") or "")[:12],
        "server_hash": str(client.get("server_recomputed_hash") or "")[:12],
        "payload_bytes_up": client.get("payload_bytes_up"),
        "payload_bytes_down": client.get("payload_bytes_down"),
        "same_host_as_server": client.get("same_host_as_server"),
        "data_mode": client.get("data_mode"),
    }


def _machine_key(client: dict[str, Any]) -> str:
    return str(client.get("machine_id") or client.get("peer_hash") or "")


def _machine_labels(audits: list[dict[str, Any]]) -> dict[str, str]:
    """Đặt tên Máy A, Máy B theo client nhỏ nhất của mỗi máy, để tên không đổi giữa các vòng."""
    first_client: dict[str, int] = {}
    for audit in audits:
        for client in audit.get("clients") or []:
            key = _machine_key(client)
            if key:
                first_client[key] = min(first_client.get(key, 10 ** 9), int(client.get("client_id", 10 ** 9)))
    ordered = sorted(first_client, key=lambda key: first_client[key])
    return {key: f"Máy {chr(ord('A') + index)}" if index < 26 else f"Máy {index + 1}" for index, key in enumerate(ordered)}


def _round_view(audit: dict[str, Any], labels: dict[str, str]) -> dict[str, Any]:
    metrics = audit.get("server_validation_metrics") or {}
    elapsed = _seconds(audit.get("round_fit_elapsed_ms"))
    server = sum(_num(audit.get(key)) or 0.0 for key in ("server_validate_ms", "server_aggregate_ms", "server_eval_ms")) / 1000.0
    transfer = _seconds(audit.get("transfer_estimate_ms"))
    learn = max(elapsed - server - (transfer or 0.0), 0.0) if elapsed is not None else None
    return {
        "round": int(audit.get("round", 0)),
        "accepted": audit.get("accepted_clients"),
        "rejected": audit.get("rejected_clients"),
        "suspicious": audit.get("suspicious_clients"),
        "accuracy": _num(metrics.get("accuracy"), 4),
        "loss": _num(metrics.get("loss"), 4),
        "macro_f1": _num(metrics.get("macro_f1"), 4),
        "elapsed_s": elapsed,
        # Ba phần cộng lại bằng cả vòng: học và chờ client chậm nhất / truyền (ước lượng) / máy chủ kiểm, gộp, đánh giá.
        "learn_and_wait_s": round(learn, 3) if learn is not None else None,
        "transfer_s": transfer,
        "server_s": round(server, 3),
        "aggregate_ms": _num(audit.get("server_aggregate_ms"), 1),
        "parent_hash": str(audit.get("parent_model_hash") or "")[:12],
        "global_hash": str(audit.get("global_model_hash") or "")[:12],
        "clients": sorted((_client_view(c, labels) for c in audit.get("clients") or []), key=lambda c: c["client_id"]),
    }


def _machines(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for round_ in rounds:
        for client in round_["clients"]:
            grouped.setdefault(client["machine"], []).append(client)
    out = []
    for label in sorted(grouped):
        rows = grouped[label]
        out.append({
            "label": label,
            "client_ids": sorted({row["client_id"] for row in rows}),
            "rtt_ms": _median([row["rtt_ms"] for row in rows]),
            "up_mb_s": _median([row["up_mb_s"] for row in rows]),
            "down_mb_s": _median([row["down_mb_s"] for row in rows]),
            "train_s": _median([row["train_s"] for row in rows]),
            "same_host_as_server": any(row["same_host_as_server"] is True for row in rows),
        })
    return out


def _checks(audits: list[dict[str, Any]], rounds: list[dict[str, Any]], required: Optional[int], final_hash: Optional[str]) -> dict[str, Any]:
    clients = [client for round_ in rounds for client in round_["clients"]]
    known_norm = [c for c in clients if c["update_norm_matches"] is not None]
    known_host = [c for c in clients if c["same_host_as_server"] is not None]
    # Chuỗi băm: mô hình cuối vòng r phải là mô hình gửi xuống ở vòng r+1, và mô hình cuối cùng phải là mô hình của vòng cuối.
    chain = [str(a.get("global_model_hash") or "") for a in audits], [str(a.get("parent_model_hash") or "") for a in audits]
    linked = all(chain[0][i] and chain[0][i] == chain[1][i + 1] for i in range(len(audits) - 1))
    if final_hash and chain[0]:
        linked = linked and chain[0][-1] == final_hash
    return {
        "hash_ok": sum(c["hash_ok"] for c in clients),
        "hash_total": len(clients),
        "norm_ok": sum(c["update_norm_matches"] is True for c in known_norm),
        "norm_total": len(known_norm),
        "chain_ok": bool(audits) and linked,
        "full_rounds": sum(r["accepted"] == required for r in rounds) if required else None,
        "rounds_total": len(rounds),
        "required_clients": required,
        "machines": len({c["machine"] for c in clients if c["machine"] != "chưa rõ"}),
        "clients_on_server_host": len({c["client_id"] for c in known_host if c["same_host_as_server"] is True}),
        "host_check_available": bool(known_host),
        "all_real_images": bool(clients) and all(c["data_mode"] == "real" for c in clients),
        "bytes_up": sum(int(c["payload_bytes_up"] or 0) for c in clients),
        "bytes_down": sum(int(c["payload_bytes_down"] or 0) for c in clients),
    }


def _state(progress: Optional[dict[str, Any]], history: Optional[dict[str, Any]], now: datetime) -> tuple[str, Optional[int]]:
    """(trạng thái, số giây từ lần ghi cuối). `history.json` chỉ được ghi khi đã xong và đã kiểm đủ vòng, nên có nó là xong."""
    if progress is None:
        return ("finished" if history is not None else "none"), None
    status = str(progress.get("status") or "")
    age: Optional[int] = None
    stamp = progress.get("updated_at_utc")
    if stamp:
        try:
            age = max(0, int((now - datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()))
        except ValueError:
            age = None
    if status == "finished" or history is not None:
        return "finished", age
    if status == "failed":
        return "failed", age
    if age is not None and age > STALE_AFTER_S:
        return "stale", age
    return (status or "running"), age


def run_view(record: dict[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
    progress, history = record.get("progress"), record.get("history")
    if progress is None and history is None:
        return {"available": False, "run_id": record.get("run_id"), "state": "none"}
    now = now or datetime.now(timezone.utc)
    state, age = _state(progress, history, now)
    source = history or progress or {}
    audits = [a for a in (source.get("audit_rounds") or source.get("rounds") or []) if isinstance(a, dict)]
    labels = _machine_labels(audits)
    rounds = [_round_view(audit, labels) for audit in audits]
    meta = (progress or {}).get("meta") or {}
    num_clients = meta.get("num_clients", (history or {}).get("num_clients_required"))
    min_clients = meta.get("min_clients", (history or {}).get("min_clients_required", num_clients))
    total_rounds = meta.get("rounds", (history or {}).get("rounds"))
    pretrained = bool(meta.get("pretrained_init", (history or {}).get("pretrained_init", False)))
    final_hash = (history or {}).get("final_model_hash") or (progress or {}).get("final_model_hash")
    final_test = (history or {}).get("final_test_metrics") or (progress or {}).get("final_test_metrics") or None
    return {
        "available": True,
        "run_id": record.get("run_id"),
        "state": state,
        "seconds_since_update": age,
        "error": (progress or {}).get("error"),
        "failed_round": ((progress or {}).get("failed_round") or {}).get("round"),
        "num_clients": num_clients,
        "min_clients": min_clients,
        "total_rounds": total_rounds,
        "local_epochs": meta.get("local_epochs", (history or {}).get("local_epochs")),
        "seed": meta.get("seed", (history or {}).get("seed")),
        "pretrained_init": pretrained,
        "eval_images": meta.get("eval_images"),
        "started_at_utc": meta.get("started_at_utc", (history or {}).get("started_at_utc")),
        "finished_at_utc": (progress or {}).get("finished_at_utc", (history or {}).get("finished_at_utc")),
        "connected_clients": (progress or {}).get("connected_clients", num_clients if state == "finished" else 0),
        "round_in_progress": (progress or {}).get("round_in_progress", 0),
        "completed_rounds": len(rounds),
        "round_replies": (progress or {}).get("round_replies") or [],
        "round_participants": (progress or {}).get("round_participants"),
        "rounds": rounds,
        "machines": _machines(rounds),
        "checks": _checks(audits, rounds, num_clients, final_hash),
        "final_test": {k: _num(final_test.get(k), 4) for k in ("accuracy", "loss", "macro_f1")} if isinstance(final_test, dict) else None,
        "final_model_hash": str(final_hash or "")[:16],
        "client_reported_losses": (history or {}).get("losses_distributed") or [],     # chỉ có ở nhật ký cũ: trước đây client tự đánh giá và gửi loss
        "caveat": CAVEAT_PRETRAINED if pretrained else None,
    }
