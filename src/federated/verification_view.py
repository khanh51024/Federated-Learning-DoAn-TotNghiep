"""Số liệu kiểm chứng đã lưu, đọc một chỗ cho cả terminal (scripts/demo_show.py) và web (/api/evidence/verification).

Chỉ đọc tệp đã có. Chỉ lấy các khóa `val` và `legacy_diagnostic_test`: không đọc split `calibration` (tập niêm phong, docs/EXPERIMENT_PROTOCOL.md mục 9).
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Optional

SWEEP_ROWS = (
    ("centralized", "", "Tập trung, học bằng SGD (trần chính)"),
    ("centralized+adamw", "", "Tập trung, học bằng AdamW (đối chứng)"),
    ("fedavg+flower", "iid", "Phân tán trên Flower, chia đều"),
    ("fedavg", "label_alpha_100_0", "Phân tán, gần như đều (α=100)"),
    ("fedavg", "label_alpha_10_0", "Phân tán, lệch ít (α=10)"),
    ("fedavg", "label_alpha_1_0", "Phân tán, lệch vừa (α=1)"),
    ("fedavg", "label_alpha_0_5", "Phân tán, lệch nhiều (α=0,5)"),
    ("fedavg", "label_alpha_0_1", "Phân tán tự cài, rất lệch (α=0,1)"),
    ("fedavg+flower", "label_alpha_0_1", "Phân tán trên Flower, rất lệch (α=0,1)"),
    ("local", "label_alpha_0_1", "Mỗi máy tự học một mình, rất lệch"),
)
SWEEP_SOURCES = (("plantvillage_test", "PlantVillage"), ("field_same_source_test", "Ngoài đồng"), ("plantdoc_holdout_v3", "PlantDoc"))
# Theo thứ tự lệch dần: trục ngang của biểu đồ khoảng cách. Tên hiển thị dùng dấu phẩy thập phân như các trang khác.
ALPHA_ORDER = (("label_alpha_100_0", "α=100"), ("label_alpha_10_0", "α=10"), ("label_alpha_1_0", "α=1"), ("label_alpha_0_5", "α=0,5"), ("label_alpha_0_1", "α=0,1"))
# Các ô yêu cầu đo mà chưa có run: tự biến mất khỏi danh sách khi bảng đã có số.
EXPECTED_CELLS = (
    ("fedavg", "quantity_alpha_0_5", "Phân tán, lệch số lượng ảnh"),
    ("fedavg", "source_skew", "Phân tán, lệch nguồn ảnh"),
    ("local", "label_alpha_100_0", "Tự học một mình, α=100"),
    ("local", "label_alpha_10_0", "Tự học một mình, α=10"),
    ("local", "label_alpha_1_0", "Tự học một mình, α=1"),
    ("local", "label_alpha_0_5", "Tự học một mình, α=0,5"),
    ("local", "quantity_alpha_0_5", "Tự học một mình, lệch số lượng ảnh"),
    ("local", "source_skew", "Tự học một mình, lệch nguồn ảnh"),
)
# Sự thật về các lần đo cũ mà tệp tóm tắt không ghi thành trường riêng (đọc từ `notes` của từng lần đo).
SPEEDTEST_FACTS = {
    "loopback_baseline": {"title": "Cùng một laptop (mốc so sánh)", "server_host_machines": ("laptop", "vm_oracle", "laptop_1_may_chu"), "all_on_server_host": True, "via_ssh_tunnel": False},
    "wan_vm_oracle": {"title": "Laptop và VM Oracle, qua đường hầm SSH", "server_host_machines": ("vm_oracle",), "via_ssh_tunnel": True},
    "wan_may_thue": {"title": "Laptop và máy thuê GPU, qua đường hầm SSH", "server_host_machines": ("may_thue_rtx3090",), "via_ssh_tunnel": True},
    "lan_2_laptop": {"title": "2 laptop cùng mạng", "server_host_machines": ("laptop_1_may_chu",), "via_ssh_tunnel": False},
    "wan_5_client_1_laptop": {"title": "5 client trên 1 laptop, máy chủ riêng trên VM, qua Internet thật", "server_host_machines": (), "via_ssh_tunnel": False},
}
MACHINE_NAMES = {
    "laptop": "Laptop", "vm_oracle": "VM Oracle", "laptop_i5_12500h_cpu": "Laptop i5-12500H", "may_thue_rtx3090": "Máy thuê RTX 3090",
    "laptop_1_may_chu": "Laptop 1", "laptop_2": "Laptop 2",
}


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sweep_view(table: Path) -> dict[str, Any]:
    """Bảng độ chính xác đọc thẳng từ artifacts/sweep_v4/table_main.csv để số trên màn hình luôn khớp báo cáo."""
    with Path(table).open(encoding="utf-8", newline="") as handle:
        cells = {(row["variant"], row["scheme"], row["part"]): row for row in csv.DictReader(handle)}
    first = cells[("centralized", "", "all_test")]
    rows = []
    for variant, scheme, label in SWEEP_ROWS:
        row = cells.get((variant, scheme, "all_test"))
        if row is None:
            continue
        rows.append({
            "variant": variant, "scheme": scheme, "label": label,
            "top1": _float(row["top1_mean_%"]), "top1_std": _float(row["top1_std_%"]), "macro_f1": _float(row["macro_f1_mean_%"]),
            "gap": _float(row.get("gap_top1_points")) if scheme else None,
            "by_source": {name: _float(cells[(variant, scheme, part)]["top1_mean_%"]) if (variant, scheme, part) in cells else None for part, name in SWEEP_SOURCES},
        })

    def points(variant: str) -> list[dict[str, Any]]:
        out = []
        for scheme, name in ALPHA_ORDER:
            row = cells.get((variant, scheme, "all_test"))
            out.append({"alpha": name, "top1": _float(row["top1_mean_%"]) if row else None, "top1_std": _float(row["top1_std_%"]) if row else None,
                        "gap": _float(row.get("gap_top1_points")) if row else None})
        return out

    return {
        "available": True,
        "images": int(first["n"]),
        "seeds": int(first["seeds"]),
        "centralized_top1": _float(first["top1_mean_%"]),
        "sources": [name for _, name in SWEEP_SOURCES],
        "rows": rows,
        "by_alpha": {"labels": [name for _, name in ALPHA_ORDER], "fedavg": points("fedavg"), "flower": points("fedavg+flower"), "local": points("local")},
        "not_measured": [label for variant, scheme, label in EXPECTED_CELLS if (variant, scheme, "all_test") not in cells],
    }


def speed_view(root: Path) -> dict[str, Any]:
    """Các lần đo tốc độ đã lưu, cũ nhất trước, kèm ghi chú lần đo nào có client chung máy với máy chủ, lần nào qua đường hầm."""
    runs = []
    root = Path(root)
    for path in sorted(root.glob("*/summary.json")) if root.is_dir() else []:
        summary = _read_json(path)
        if summary is None:
            continue
        label = str(summary.get("label") or path.parent.name)
        facts = SPEEDTEST_FACTS.get(label, {})
        for name, run in (summary.get("runs") or {}).items():
            machines = []
            for machine, stats in (run.get("by_machine") or {}).items():
                on_server = bool(facts.get("all_on_server_host")) or machine in facts.get("server_host_machines", ())
                machines.append({
                    "name": MACHINE_NAMES.get(machine, machine.replace("_", " ")),
                    "on_server_host": on_server,
                    "rtt_ms": (stats.get("probe_rtt_persistent_ms") or {}).get("median"),
                    "up_mb_s": (stats.get("probe_up_mb_per_s") or {}).get("median"),
                    "down_mb_s": (stats.get("probe_down_mb_per_s") or {}).get("median"),
                    "train_s": (stats.get("local_train_time_s") or {}).get("median"),
                })
            hosted = run.get("clients_on_server_host")
            runs.append({
                "id": f"{path.parent.name}/{name}",
                "label": label,
                "title": facts.get("title") or str(summary.get("title") or label),
                "date": str(summary.get("created_utc") or path.parent.name)[:8],
                "rounds": run.get("rounds"),
                "clients": run.get("clients"),
                "local_epochs": run.get("local_epochs"),
                "hashes_ok": run.get("all_upload_hashes_ok"),
                "via_ssh_tunnel": bool(facts.get("via_ssh_tunnel", False)),
                "client_on_server_host": any(m["on_server_host"] for m in machines) or bool(hosted),
                "measured_by": str(summary.get("measured_by") or "client"),
                "round_seconds": (run.get("round_fit_elapsed_ms") or {}).get("median"),
                "machines": machines,
            })
    return {"available": bool(runs), "runs": runs}


def weight_view(verification_dir: Path, fedsgd_json: Path) -> dict[str, Any]:
    """Bằng chứng về trọng số: gửi đúng từng bit qua Flower, gộp đúng, phân tán so với tập trung khác nhau bao nhiêu."""
    verification_dir = Path(verification_dir)
    comparison = _read_json(verification_dir / "weight_comparison.json")
    audit = _read_json(verification_dir / "batch_audit.json")
    roundoff = _read_json(verification_dir / "aggregation_roundoff_analysis.json")
    fedsgd = _read_json(Path(fedsgd_json))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    w0_recorded = w0_match = 0
    for item in (comparison or {}).get("comparisons") or []:
        groups.setdefault((str(item.get("runtime")), str(item.get("scheme"))), []).append(item)
        w0 = item.get("w0") or {}
        if w0.get("fedavg_recorded"):
            w0_recorded += 1
            w0_match += w0["fedavg_recorded"] == w0.get("reconstructed_from_seed")
    distance = []
    for (runtime, scheme), items in groups.items():
        learnable = [i["weight_distance"]["learnable:all"] for i in items]
        agreement = [i["prediction_agreement"]["legacy_diagnostic_test"] for i in items]
        distance.append({
            "runtime": runtime, "scheme": scheme, "runs": len(items),
            "relative_distance": round(statistics.mean(x["relative"] for x in learnable), 4),
            "cosine": round(statistics.mean(x["cosine"] for x in learnable), 4),
            "prediction_agreement": round(statistics.mean(x["agreement"] for x in agreement), 4),
            "agreement_images": agreement[0]["n"],
        })
    order = {scheme: index for index, (scheme, _) in enumerate(ALPHA_ORDER)}
    distance.sort(key=lambda row: (row["runtime"] != "flower", order.get(row["scheme"], -1)))
    runs = (audit or {}).get("runs") or []
    # Chỉ run chạy trên Flower mới có tin nhắn thật để so mã băm: số vòng và số run của ô này phải tính đúng trên các run đó, không phải toàn bộ.
    transported = [r for r in runs if r.get("transport_comparisons")]
    checks = (fedsgd or {}).get("checks") or []
    # Ngưỡng chốt trước chỉ áp cho float64; float32 lệch cỡ sai số làm tròn (đối chứng tập trung đổi thứ tự ảnh cũng lệch cùng cỡ), không gộp chung một số.
    exact_rows = [c for c in checks if c.get("check") == "A_fedsgd" and c.get("dtype") == "float64"]
    return {
        "available": comparison is not None or audit is not None or fedsgd is not None,
        "transport": {
            "runs_audited": len(runs),
            "runs_audited_passed": sum(bool(r.get("passed")) for r in runs),
            "runs": len(transported),
            "passed": sum(bool(r.get("passed")) for r in transported),
            "comparisons": sum(int(r["transport_comparisons"]) for r in transported),
            "rounds": sum(int(r.get("rounds") or 0) for r in transported),
            "max_aggregation_diff": max((float(r["aggregation_max_abs_diff"]) for r in runs if r.get("aggregation_max_abs_diff") is not None), default=None),
        },
        "w0": {"recorded": w0_recorded, "match": w0_match, "total_runs": sum(len(v) for v in groups.values())},
        "distance": distance,
        "fedsgd": {
            "evidence": (fedsgd or {}).get("evidence"),
            "max_l2_float64": max((float(c["l2"]) for c in exact_rows), default=None),
            "schemes_float64": len(exact_rows),
            "agreement_float64": min((float(c["prediction_agreement"]) for c in exact_rows), default=None),
            "rows": [{"check": c.get("check"), "scheme": c.get("scheme"), "dtype": c.get("dtype"), "local_steps": c.get("local_steps"), "batchnorm_live": c.get("batchnorm_live"),
                      "l2": float(c["l2"]), "prediction_agreement": round(float(c["prediction_agreement"]), 4)} for c in checks],
        },
        "roundoff": {"max_abs_diff": roundoff.get("max_abs_diff"), "max_ulp_diff": roundoff.get("max_ulp_diff"), "float32_epsilon": roundoff.get("float32_epsilon")} if roundoff else None,
    }
