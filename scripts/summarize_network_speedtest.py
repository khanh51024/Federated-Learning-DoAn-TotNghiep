"""Tóm tắt các lần chạy đo tốc độ FedAvg giữa hai máy thành MỘT báo cáo.

Đọc `audit_rounds` mà Flower server ghi (xem src/federated/server_app.py) cho từng kịch bản (loopback, WAN, LAN...). Mọi thời gian
đo trên đồng hồ của CHÍNH máy đó (client đo thời gian tính/đường truyền của nó, server đo thời gian vòng của nó); không so giờ giữa hai
máy nên không cần đồng bộ giờ.

Cột quan trọng:
  local_train_time_s, client_wall_ms : thời gian tính trên máy client (huấn luyện / cả lượt fit gồm giải mã + mã hóa)
  probe_rtt_persistent_ms            : RTT trên kết nối giữ sẵn, không tính bắt tay TCP
  probe_up/down_mb_per_s             : băng thông đo bằng khối byte cùng cỡ mô hình, đã trừ một RTT
  payload_bytes_up/down              : byte Flower tuần tự hóa thật (gồm phần đầu .npy)
  transfer_estimate_ms               : thời gian vòng trừ việc server tự làm và client chậm nhất tự tính = tải xuống + tải lên + xếp hàng
  predicted_transfer_ms              : payload / băng thông đo được (xuống + lên), để đối chiếu với transfer_estimate_ms

Chạy:
    python scripts/summarize_network_speedtest.py --run loopback=tmp/speedtest/out/loopback_history.json \\
        --run wan=tmp/speedtest/out/wan_history.json --machines 0=laptop,1=vm --label wan_vm_oracle
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402


def _median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.median(clean), 3) if clean else None


def _spread(values: list[float | None]) -> dict[str, float | None]:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {"min": round(clean[0], 3), "median": round(statistics.median(clean), 3), "max": round(clean[-1], 3), "n": len(clean)}


def summarize_run(history: dict[str, Any], machines: dict[int, str]) -> dict[str, Any]:
    rounds = history["audit_rounds"]
    per_client: dict[int, dict[str, list[float | None]]] = {}
    out_rounds = []
    for audit in rounds:
        entry: dict[str, Any] = {
            "round": audit["round"], "round_fit_elapsed_ms": round(audit["round_fit_elapsed_ms"], 1),
            "server_validate_ms": audit.get("server_validate_ms"), "server_aggregate_ms": audit.get("server_aggregate_ms"),
            "server_eval_ms": audit.get("server_eval_ms"), "slowest_client_wall_ms": audit.get("slowest_client_wall_ms"),
            "transfer_estimate_ms": audit.get("transfer_estimate_ms"), "clients": [],
        }
        predicted_worst = 0.0
        for client in audit["clients"]:
            cid = int(client["client_id"])
            up_bps, down_bps = client.get("probe_up_mb_per_s"), client.get("probe_down_mb_per_s")
            pred_ms = None
            if up_bps and down_bps and client.get("payload_bytes_up") and client.get("payload_bytes_down"):
                pred_ms = 1000.0 * (client["payload_bytes_down"] / 1048576 / down_bps + client["payload_bytes_up"] / 1048576 / up_bps)
                predicted_worst = max(predicted_worst, pred_ms)
            row = {
                "client_id": cid, "machine": machines.get(cid, f"client_{cid}"), "num_examples": client["num_examples"],
                "status": client["status"], "upload_hash_ok": client.get("upload_hash_ok"),
                "local_train_time_s": client["local_train_time_s"], "client_wall_ms": client.get("client_wall_ms"),
                "client_decode_ms": client.get("client_decode_ms"), "client_encode_ms": client.get("client_encode_ms"),
                "probe_connect_ms": client.get("probe_connect_ms"), "probe_rtt_persistent_ms": client.get("probe_rtt_persistent_ms"),
                "probe_rtt_legacy_one_shot_ms": client.get("probe_rtt_ms"),
                "probe_up_mb_per_s": up_bps, "probe_down_mb_per_s": down_bps,
                "payload_bytes_up": client.get("payload_bytes_up"), "payload_bytes_down": client.get("payload_bytes_down"),
                "predicted_transfer_ms": round(pred_ms, 1) if pred_ms is not None else None,
                "train_images_per_s": round(client["num_examples"] / client["local_train_time_s"], 2) if client["local_train_time_s"] else None,
            }
            entry["clients"].append(row)
            bucket = per_client.setdefault(cid, {})
            for key in ("local_train_time_s", "client_wall_ms", "probe_connect_ms", "probe_rtt_persistent_ms", "probe_up_mb_per_s",
                        "probe_down_mb_per_s", "train_images_per_s", "predicted_transfer_ms"):
                bucket.setdefault(key, []).append(row.get(key))
        entry["predicted_transfer_ms_slowest_client"] = round(predicted_worst, 1) if predicted_worst else None
        out_rounds.append(entry)
    return {
        "rounds": len(rounds), "clients": len(per_client),
        "all_upload_hashes_ok": all(c.get("upload_hash_ok") is True for a in rounds for c in a["clients"]),
        "all_clients_accepted": all(a["accepted_clients"] == len(a["clients"]) for a in rounds),
        "final_model_hash": history.get("final_model_hash"),
        "hash_chain_ok": all(rounds[i]["global_model_hash"] == rounds[i + 1]["parent_model_hash"] for i in range(len(rounds) - 1)),
        "by_machine": {
            machines.get(cid, f"client_{cid}"): {k: _spread(v) for k, v in bucket.items()} for cid, bucket in sorted(per_client.items())
        },
        "round_fit_elapsed_ms": _spread([a["round_fit_elapsed_ms"] for a in rounds]),
        "transfer_estimate_ms": _spread([a.get("transfer_estimate_ms") for a in rounds]),
        "predicted_transfer_ms_slowest_client": _spread([r["predicted_transfer_ms_slowest_client"] for r in out_rounds]),
        "per_round": out_rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="tên=đường dẫn history.json (có thể lặp)")
    parser.add_argument("--machines", default="0=client_0,1=client_1", help="client_id=tên máy, cách nhau bằng dấu phẩy")
    parser.add_argument("--label", default="speedtest")
    parser.add_argument("--notes", default="", help="ghi chú về cách dựng kết nối (ví dụ: ssh -L qua Internet)")
    args = parser.parse_args()

    machines = {int(k): v for k, v in (pair.split("=", 1) for pair in args.machines.split(","))}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = reg.abs_path(reg.NETWORK_SPEEDTEST_DIR) / f"{stamp}_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "label": args.label, "created_utc": stamp, "notes": args.notes,
        "evidence": "đo tốc độ trên dữ liệu PlantVillage thật (shard nhỏ); KHÔNG dùng làm kết quả độ chính xác",
        "machines": {str(k): v for k, v in machines.items()}, "runs": {},
    }
    for item in args.run:
        name, _, path_text = item.partition("=")
        path = Path(path_text)
        raw = json.loads(path.read_text(encoding="utf-8"))
        shutil.copy2(path, out_dir / f"{name}_history.json")
        summary = summarize_run(raw, machines)
        summary["history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary["server_address"] = raw.get("server_address")
        report["runs"][name] = summary
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[speedtest] đã ghi {out_dir.relative_to(ROOT).as_posix()}/summary.json")
    for name, run in report["runs"].items():
        print(name, {"hash_ok": run["all_upload_hashes_ok"], "chain_ok": run["hash_chain_ok"], "round_ms": run["round_fit_elapsed_ms"]["median"],
                     "transfer_ms": run["transfer_estimate_ms"]["median"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
