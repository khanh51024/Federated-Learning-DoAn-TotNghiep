"""Tóm tắt bằng chứng cho giao diện: một nơi đọc các báo cáo đã sinh trên đĩa.

Nguyên tắc (giống phần còn lại của web):
- Chỉ ĐỌC; không tính lại gì, không suy diễn số mới ngoài phép tổng hợp đơn giản trên số có sẵn.
- Fail-closed TỪNG PHẦN: báo cáo thiếu, hỏng hoặc không khớp danh tính bundle đã kiểm hash (protocol, release,
  validation, SHA-256 từng checkpoint, số công bố) thì phần đó trả `available: false` kèm lý do — không hiện số nửa vời.
- Phần có số liệu của mô hình chỉ được dựng khi bundle đã qua cổng registry (`evidence["available"]`); khi bị chặn,
  checklist chỉ giữ id/tiêu đề/trạng thái (chuỗi `detail` của checklist có chứa số của mô hình).
- Mọi giá trị trả ra đã được ép kiểu (int/float/str) để giao diện không phải tin dữ liệu thô của báo cáo.
- Mọi đường dẫn lấy từ `src/registry.py`.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional

from src import registry as reg
from web.services import model_service

MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
SOURCES = ("plantdoc", "plantvillage")
MAIN_MODELS = ("centralized", "fedavg")
# Mô hình trong per-class report/domain-gap report được đặt tên hiển thị thay vì model_id
_DISPLAY = {"centralized": "Centralized v6", "fedavg": "FedAvg v7"}

_JSON_CACHE: Dict[str, Any] = {}


def _load_json(relative: str) -> Optional[Dict[str, Any]]:
    """Đọc JSON dưới thư mục repo, cache theo (kích thước, mtime); None nếu thiếu hoặc hỏng."""
    path = reg.abs_path(relative)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (stat.st_size, stat.st_mtime_ns)
    cached = _JSON_CACHE.get(relative)
    if cached is None or cached[0] != key:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        cached = (key, data if isinstance(data, dict) else None)
        _JSON_CACHE[relative] = cached
    return cached[1]


def _withheld(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def _identity_problems(report: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
    problems = []
    for key in ("protocol_id", "release_manifest_sha256", "validation_manifest_sha256"):
        if report.get(key) != evidence.get(key):
            problems.append(f"{key} khác bundle đã kiểm hash")
    return problems


def _round(value: Any, digits: int = 4) -> Optional[float]:
    """float làm tròn; None cho thiếu/không phải số/NaN/inf (JSON không mang được NaN)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def _metrics(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "count": _int(block.get("count")),
        "correct": _int(block.get("correct")),
        "supported_classes": _int(block.get("supported_classes")),
        "top1": _round(block.get("top1_accuracy")),
        "macro_f1": _round(block.get("macro_f1_supported")),
        "crop_accuracy": _round(block.get("crop_accuracy")),
        "high_confidence_wrong": _int(block.get("high_confidence_wrong_count_0_9")),
    }


def _reproduction_status(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not entry or not entry.get("available"):
        return {"status": "unavailable", "correct_diff_images": None}
    diff = _int(entry.get("correct_diff_images"))
    if entry.get("reproduced") is True and diff in (0, None):
        status = "exact"
    elif entry.get("within_tolerance") is True:
        status = "within_tolerance"
    else:
        status = "diff"
    return {"status": status, "correct_diff_images": diff}


def _calibration_row(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n": _int(block.get("n")),
        "accuracy": _round(block.get("accuracy")),
        "mean_confidence": _round(block.get("mean_confidence_raw")),
        "ece_raw": _round(block.get("ece_raw")),
        "ece_temperature_scaled": _round(block.get("ece_temperature_scaled")),
    }


def _bins(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"lo": _round(b.get("lo")), "hi": _round(b.get("hi")), "n": _int(b.get("n")),
         "confidence": _round(b.get("mean_confidence")), "accuracy": _round(b.get("accuracy"))}
        for b in (block.get("reliability_raw") or [])
    ]


def _published_by_id(evidence: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {m["model_id"]: m for m in evidence.get("models", [])}


def _reeval(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = _load_json(reg.LOCAL_REEVAL_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.LOCAL_REEVAL_JSON} — chạy scripts/evaluate_published_models.py (cần dataset)")
    published = _published_by_id(evidence)
    problems = _identity_problems(report, evidence)
    expected = {mid: m["checkpoint_sha256"] for mid, m in published.items()}
    if (report.get("checkpoint_sha256") or {}) != expected:
        problems.append("SHA-256 checkpoint trong báo cáo không khớp bundle")
    if problems:
        return _withheld("báo cáo tái đánh giá không khớp bundle: " + "; ".join(problems))

    models = []
    pv_recall: Dict[str, Dict[str, Optional[float]]] = {}
    pv_support: Dict[str, int] = {}
    reliability: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    statuses: Dict[str, Dict[str, str]] = {}
    for mid in MODEL_ORDER:
        info = published.get(mid)
        rep = (report.get("models") or {}).get(mid)
        if info is None or rep is None:
            return _withheld(f"báo cáo tái đánh giá thiếu mô hình {mid}")
        by_source = rep.get("by_source") or {}
        calibration = rep.get("calibration") or {}
        vs_published = rep.get("vs_published") or {}
        for source in SOURCES:
            pub = info["metrics"][source]
            own = by_source.get(source) or {}
            if _int(own.get("count")) != _int(pub.get("count")):
                return _withheld(f"{mid}/{source}: số ảnh của báo cáo khác số trong bundle")
            vp = vs_published.get(source) or {}
            if vp.get("available"):
                expected_correct = round(float(pub["top1_accuracy"]) * int(pub["count"]))
                if _int(vp.get("correct_published")) != expected_correct:
                    return _withheld(f"{mid}/{source}: số ảnh đúng 'đã công bố' trong báo cáo khác bundle")
        statuses[mid] = {s: _reproduction_status(vs_published.get(s))["status"] for s in SOURCES}
        models.append({
            "model_id": mid,
            "name": str(info["name"]),
            "phase": str(info["phase"]),
            "overall": _metrics(rep.get("overall") or {}),
            "plantdoc": _metrics(by_source.get("plantdoc") or {}),
            "plantvillage": _metrics(by_source.get("plantvillage") or {}),
            "temperature": _round(rep.get("temperature_fitted_on_calibration")),
            "calibration": {
                "overall": _calibration_row(calibration.get("overall") or {}),
                "plantdoc": _calibration_row(calibration.get("plantdoc") or {}),
                "plantvillage": _calibration_row(calibration.get("plantvillage") or {}),
            },
            "vs_published": {s: _reproduction_status(vs_published.get(s)) for s in SOURCES},
        })
        recall = ((by_source.get("plantvillage") or {}).get("per_class_recall")) or {}
        pv_recall[mid] = {str(name): _round(v.get("recall")) for name, v in recall.items()}
        for name, v in recall.items():
            pv_support.setdefault(str(name), _int(v.get("support")) or 0)
        if mid in MAIN_MODELS:
            reliability[mid] = {s: _bins(calibration.get(s) or {}) for s in SOURCES}

    # Tính lại từ trạng thái từng phép so sánh thay vì tin các tổng trong báo cáo
    flat = [status for per_source in statuses.values() for status in per_source.values()]
    detail = report.get("detail") or {}
    confusions = {
        mid: {
            s: [
                {"target": str(c.get("target")), "predicted": str(c.get("predicted")), "count": _int(c.get("count"))}
                for c in (((detail.get(mid) or {}).get("top_confusions") or {}).get(s) or [])[:6]
            ]
            for s in SOURCES
        }
        for mid in MAIN_MODELS
    }
    tolerance = report.get("tolerance_rule") or {}
    return {
        "available": True,
        "validation_count": _int(report.get("validation_count")),
        "calibration_count": _int(report.get("calibration_count")),
        "comparisons_total": len(flat),
        "comparisons_exact": sum(1 for s in flat if s == "exact"),
        "all_within_tolerance": all(s in ("exact", "within_tolerance") for s in flat),
        "main_models_exact": all(statuses[m][s] == "exact" for m in MAIN_MODELS for s in SOURCES),
        "tolerance_rule": {
            "correct_images": _int(tolerance.get("correct_images")),
            "high_confidence_wrong_images": _int(tolerance.get("high_confidence_wrong_images")),
            "metric": _round(tolerance.get("metric"), 6),
        },
        "note": str(report.get("known_discrepancy_note") or ""),
        "high_confidence_threshold": _round(report.get("high_confidence_threshold")),
        "bins": _int(report.get("bins")),
        "models": models,
        "pv_recall": pv_recall,
        "pv_support": pv_support,
        "reliability": reliability,
        "top_confusions": confusions,
    }


def _published_block_problems(published: Dict[str, Any], display: str, source: str, count: Any, top1: Any) -> Optional[str]:
    pub = published[display]["metrics"][source]
    if _int(count) != _int(pub["count"]):
        return f"số ảnh {source} của {display} khác số trong bundle"
    if _round(top1, 6) is None or abs(float(top1) - float(pub["top1_accuracy"])) > 1e-6:
        return f"top-1 {source} của {display} khác số đã kiểm hash"
    return None


def _plantdoc(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = _load_json(reg.PER_CLASS_REPORT_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.PER_CLASS_REPORT_JSON} — chạy scripts/report_per_class.py")
    if report.get("protocol_id") != evidence.get("protocol_id"):
        return _withheld("báo cáo PlantDoc theo lớp khác protocol của bundle")
    published = {m["name"]: m for m in evidence["models"] if m["model_id"] in _DISPLAY}
    models: Dict[str, Any] = {}
    for display in _DISPLAY.values():
        per_model = (report.get("models") or {}).get(display)
        if per_model is None:
            return _withheld(f"báo cáo PlantDoc theo lớp thiếu {display}")
        for source in SOURCES:
            block = per_model.get(source)
            if block is None:
                return _withheld(f"báo cáo PlantDoc theo lớp thiếu {display}/{source}")
            problem = _published_block_problems(published, display, source, block.get("count"), block.get("top1_accuracy"))
            if problem:
                return _withheld("báo cáo PlantDoc theo lớp: " + problem)
        block = per_model["plantdoc"]
        error = block.get("error_breakdown") or {}
        hcw = block.get("high_confidence_wrong") or {}
        models[display] = {
            "count": _int(block.get("count")),
            "classes_with_support": _int(block.get("classes_with_support")),
            "top1": _round(block.get("top1_accuracy")),
            "crop_accuracy": _round(block.get("crop_accuracy")),
            "error_breakdown": {
                "correct": _int(error.get("correct")),
                "wrong_crop": _int(error.get("wrong_crop")),
                "right_crop_wrong_disease": _int(error.get("right_crop_wrong_disease")),
            },
            "high_confidence_wrong": {
                "threshold": _round(hcw.get("threshold")),
                "count": _int(hcw.get("count")),
                "rate_of_all": _round(hcw.get("rate_of_all")),
                "share_of_errors": _round(hcw.get("share_of_errors")),
            },
        }
    summary = report.get("summary") or {}
    compare = report.get("fedavg_vs_centralized_plantdoc") or {}
    return {
        "available": True,
        "classes_covered": str(summary.get("plantdoc_classes_covered") or ""),
        "weak_threshold": _round(report.get("weak_recall_threshold")),
        "min_support": _int(report.get("min_support_for_reliable_class")),
        "weak_classes": [str(c) for c in (summary.get("weak_classes_fedavg") or [])],
        "weak_images": _int(summary.get("weak_images_fedavg")),
        "fedavg_worse_in": _int(compare.get("fedavg_worse_in")),
        "fedavg_better_in": _int(compare.get("fedavg_better_in")),
        "classes_compared": _int(compare.get("classes_compared")),
        "models": models,
        "table": [
            {
                "class": str(row.get("class")),
                "support": _int(row.get("plantdoc_support")),
                "fedavg_recall": _round(row.get("fedavg_plantdoc_recall")),
                "centralized_recall": _round(row.get("centralized_plantdoc_recall")),
                "fedavg_pv_recall": _round(row.get("fedavg_plantvillage_recall")),
                "domain_gap_pp": _round(row.get("fedavg_domain_gap_pp"), 2),
                "low_support": bool(row.get("low_support")),
            }
            for row in (report.get("fedavg_plantdoc_class_table") or [])
        ],
    }


def _domain_gap(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = _load_json(reg.DOMAIN_GAP_REPORT_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.DOMAIN_GAP_REPORT_JSON}")
    published = {m["name"]: m for m in evidence["models"] if m["model_id"] in _DISPLAY}
    out: Dict[str, Any] = {}
    for display in _DISPLAY.values():
        block = (report.get("models") or {}).get(display)
        if block is None:
            return _withheld(f"báo cáo khoảng cách miền thiếu {display}")
        for source in SOURCES:
            part = block.get(source) or {}
            pct = _round(part.get("accuracy_pct"), 6)
            expected = float(published[display]["metrics"][source]["top1_accuracy"]) * 100.0
            # báo cáo làm tròn 2 chữ số thập phân => sai lệch hợp lệ tối đa 0,005 điểm (nhỏ hơn 1 ảnh PlantVillage)
            if pct is None or abs(pct - expected) > 0.0051:
                return _withheld(f"độ chính xác {source} của {display} trong báo cáo khác số đã kiểm hash")
            if _int(part.get("count")) != _int(published[display]["metrics"][source]["count"]):
                return _withheld(f"số ảnh {source} của {display} trong báo cáo khác số trong bundle")
        out[display] = {
            "plantvillage_pct": _round(block["plantvillage"]["accuracy_pct"], 2),
            "plantdoc_pct": _round(block["plantdoc"]["accuracy_pct"], 2),
            "gap_pp": _round(block.get("plantvillage_minus_plantdoc_accuracy_pp"), 2),
        }
    return {"available": True, "models": out}


def _leakage(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = _load_json(reg.RELEASE_LEAKAGE_AUDIT_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.RELEASE_LEAKAGE_AUDIT_JSON} — chạy scripts/audit_release_leakage.py (cần dataset)")
    if report.get("validation_manifest_sha256") != evidence.get("validation_manifest_sha256"):
        return _withheld("audit rò rỉ làm trên validation khác bundle đã kiểm hash")
    counts = report.get("counts") or {}
    expected_val = sum(int(v) for v in (evidence.get("validation_counts") or {}).values())
    if _int(counts.get("val")) != expected_val:
        return _withheld("số ảnh validation trong audit khác số trong bundle")

    overlaps = report.get("pairwise_overlaps") or {}
    overlap_total = sum(_int(v) or 0 for pair in overlaps.values() for v in pair.values())
    partitions = report.get("partitions") or {}
    clean = [
        name for name, p in partitions.items()
        if p.get("disjoint") and p.get("subset_of_train") and not p.get("groups_split_across_clients")
    ]
    image_bytes = report.get("image_bytes") or {}
    verified = image_bytes.get("verified") is True
    mismatch = _int(image_bytes.get("hash_mismatch"))
    missing = _int(image_bytes.get("missing_files"))
    binding = (report.get("release_binding") or {}).get("matches_pinned") is True
    manifests_ok = bool(report.get("manifest_hashes_match_files")) and all(
        v is True for v in (report.get("manifest_hashes_match_files") or {}).values()
    )
    # `passed` của báo cáo chỉ được tin khi các số bên trong khớp với nó
    consistent_pass = (
        overlap_total == 0 and verified and mismatch == 0 and missing == 0 and binding and manifests_ok
        and len(clean) == len(partitions) and len(partitions) > 0
    )
    if report.get("passed") is True and not consistent_pass:
        return _withheld("báo cáo audit ghi 'đạt' nhưng số liệu bên trong không khớp (trùng lặp, hash ảnh hoặc ràng buộc release)")
    return {
        "available": True,
        "passed": bool(report.get("passed") is True and consistent_pass),
        "release_name": str(report.get("release_name") or ""),
        "counts": {k: _int(counts.get(k)) for k in ("train", "val", "calibration")},
        "overlap_total": overlap_total,
        "pairs": sorted(str(k) for k in overlaps),
        "keys": sorted({str(k) for pair in overlaps.values() for k in pair}),
        "images_verified": _int(image_bytes.get("checked")) if verified else None,
        "hash_mismatch": mismatch,
        "missing_files": missing,
        "partitions_total": len(partitions),
        "partitions_clean": len(clean),
        "release_binding_matches": binding,
        "scope": str(report.get("scope") or ""),
    }


def _shards(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = model_service.shard_report()
    if report is None:
        return _withheld(f"chưa có {reg.CLIENT_SHARD_COUNTS_JSON} hợp lệ — chạy scripts/report_client_shards.py (cần dataset)")
    if report.get("protocol_id") != evidence.get("protocol_id"):
        return _withheld("báo cáo shard khác protocol của bundle")
    clients = report.get("clients") or {}
    published = evidence.get("shard_counts") or {}
    for name, n_k in published.items():
        if _int((clients.get(name) or {}).get("n_k")) != _int(n_k):
            return _withheld(f"n_k của {name} khác số trong bundle đã kiểm hash")
    classes = [str(c) for c in (report.get("classes") or [])]
    totals = report.get("totals_all_clients") or {}
    names = sorted(str(n) for n in clients)
    rows = []
    for cls in classes:
        per_client = {n: _int((clients[n].get("per_class") or {}).get(cls)) or 0 for n in names}
        total = _int(totals.get(cls))
        if total is None:
            total = sum(per_client.values())
        top = max(per_client, key=lambda n: per_client[n]) if per_client else None
        rows.append({
            "class": cls,
            "total": total,
            "per_client": per_client,
            "top_client": top,
            "top_share_pct": round(100.0 * per_client[top] / total, 1) if top and total else None,
            "clients_with_images": sum(1 for v in per_client.values() if v > 0),
        })
    return {
        "available": True,
        "scheme": str(report.get("partition_scheme") or ""),
        "total_train": _int(report.get("total_train_samples")),
        "clients": [
            {"client": n, "n_k": _int(clients[n].get("n_k")) or 0,
             "classes_with_images": _int(clients[n].get("classes_with_images")) or 0}
            for n in names
        ],
        "classes": rows,
        "single_client_classes": sum(1 for r in rows if (r["top_share_pct"] or 0) >= 90.0),
        "classes_total": len(classes),
    }


def _checklist(redact: bool = False) -> Dict[str, Any]:
    """Checklist trạng thái dự án. `redact`: chỉ id/tiêu đề/trạng thái (detail chứa số của mô hình)."""
    report = _load_json(reg.STATUS_REPORT_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.STATUS_REPORT_JSON} — chạy scripts/update_status.py")
    items = []
    for i in report.get("checklist") or []:
        item = {"id": str(i.get("id")), "title": str(i.get("title")), "status": str(i.get("status"))}
        if not redact:
            item["detail"] = str(i.get("detail") or "")
            item["unlock"] = [str(c) for c in (i.get("unlock") or [])]
        items.append(item)
    return {
        "available": True,
        "redacted": redact,
        "readiness": "" if redact else str(report.get("readiness") or ""),
        "total": len(items),
        "done": sum(1 for i in items if i["status"] == "XONG"),
        "blocked": sum(1 for i in items if i["status"] == "BLOCKED"),
        "todo": sum(1 for i in items if i["status"] not in ("XONG", "BLOCKED")),
        "items": items,
    }


def _identity(evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocol_id": str(evidence.get("protocol_id")),
        "evidence_level": str(evidence.get("evidence_level")),
        "release_manifest_sha256": str(evidence.get("release_manifest_sha256")),
        "validation_manifest_sha256": str(evidence.get("validation_manifest_sha256")),
        "seed": _int(evidence.get("seed")),
        "transform": str(evidence.get("transform")),
        "validation_counts": {k: _int(v) for k, v in (evidence.get("validation_counts") or {}).items()},
        "models": [
            {k: str(m.get(k)) for k in ("model_id", "name", "phase", "method", "branch", "commit", "checkpoint_sha256")}
            for m in evidence.get("models", [])
        ],
    }


def _guard(name: str, builder: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return builder()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, ZeroDivisionError, IndexError) as exc:
        return _withheld(f"không đọc được báo cáo {name}: {exc}")


def summary(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp các báo cáo thành một payload cho giao diện. `evidence` là kết quả của cổng registry + load_evidence()."""
    if not evidence.get("available"):
        reason = evidence.get("reason") or "bộ đối chứng chưa khả dụng"
        return {
            "available": False,
            "reason": reason,
            "identity": None,
            "checklist": _guard("checklist", lambda: _checklist(redact=True)),
            **{name: _withheld(reason) for name in ("reeval", "plantdoc", "domain_gap", "leakage", "shards")},
        }
    return {
        "available": True,
        "identity": _identity(evidence),
        "reeval": _guard("tái đánh giá", lambda: _reeval(evidence)),
        "plantdoc": _guard("PlantDoc theo lớp", lambda: _plantdoc(evidence)),
        "domain_gap": _guard("khoảng cách miền", lambda: _domain_gap(evidence)),
        "leakage": _guard("audit rò rỉ", lambda: _leakage(evidence)),
        "shards": _guard("shard", lambda: _shards(evidence)),
        "checklist": _guard("checklist", _checklist),
    }
