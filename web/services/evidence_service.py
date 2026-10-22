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
from web.services.training_evidence import training_report

MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
SOURCES = ("plantdoc", "plantvillage")
MAIN_MODELS = ("centralized", "fedavg")
# Mô hình trong per-class report/domain-gap report được đặt tên hiển thị thay vì model_id
_DISPLAY = {k: v for k, v in reg.WEB_DISPLAY_NAMES.items() if k in ("centralized", "fedavg")}

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


def _plantvillage_reference(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Báo cáo đánh giá lại mô hình PlantVillage ngoài (không huấn luyện lại). Tách khỏi bảy mô hình đã kiểm hash: không có trong
    `MODEL_ORDER`, nên thiếu hay lệch báo cáo này không làm các mục tái đánh giá / test giữ riêng bị giữ lại."""
    report = _load_json(reg.PLANTVILLAGE_EVAL_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.PLANTVILLAGE_EVAL_JSON} — chạy scripts/evaluate_plantvillage_reference.py (cần dataset)")
    problems = _identity_problems(report, evidence)
    if (report.get("model") or {}).get("checkpoint_sha256") != reg.PLANTVILLAGE_TRACED_SHA256:
        problems.append("SHA-256 checkpoint trong báo cáo không khớp mô hình PlantVillage đã ghim")
    if problems:
        return _withheld("báo cáo PlantVillage không khớp: " + "; ".join(problems))

    def split(block: Dict[str, Any]) -> Dict[str, Any]:
        by = block.get("by_source") or {}
        calibration = block.get("calibration") or {}
        keep = ("count", "top1_accuracy", "macro_f1_supported", "crop_accuracy", "top1_ci95")
        return {
            "by_source": {src: {k: by[src].get(k) for k in keep} for src in by},
            "ece": {src: (calibration.get(src) or {}).get("ece_raw") for src in ("overall", "plantvillage", "plantdoc")},
            "temperature": block.get("temperature_fitted_on_calibration"),
        }

    return {
        "available": True,
        "model": report.get("model"),
        "leakage": report.get("leakage"),
        "preprocessing_chosen": report.get("preprocessing_chosen"),
        "preprocessing_comparison_on_val": report.get("preprocessing_comparison_on_val"),
        "validation": split(report.get("validation") or {}),
        "holdout": split(report.get("holdout") or {}),
        "vs_fedavg_on_holdout": report.get("vs_fedavg_on_holdout"),
        "evidence": report.get("evidence"),
    }


def _community_simulation(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Kết quả thí nghiệm mô phỏng học phân tán cộng đồng (scripts/simulate_community.py), đã rút gọn cho giao diện."""
    report = _load_json(reg.COMMUNITY_SIMULATION_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.COMMUNITY_SIMULATION_JSON} — chạy scripts/simulate_community.py")
    runs = []
    for key, run in (report.get("runs") or {}).items():
        runs.append({
            "key": key, "scenario": run["scenario"], "k": run["k"], "gate": run["gate"], "seeds": run["seeds"], "rounds": run["rounds"],
            "final_delta_mean_points": run["final_delta_mean_points"], "final_delta_points": run["final_delta_points"],
            "final_std": run["final_std"], "accepted_round_share": run["accepted_round_share"], "clipped_share": run["clipped_share"],
            "mean_update_norm": run["mean_update_norm"],
            "mean_curve": {m: run["mean_curve"][m] for m in ("accuracy", "macro_recall", "plantdoc_accuracy")},
        })
    if not runs:
        return _withheld("báo cáo mô phỏng không có lần chạy nào")
    return {
        "available": True, "evidence": report.get("evidence"), "policy": report.get("policy"), "local_config": report.get("local_config"),
        "seeds": report.get("seeds"), "rounds": report.get("rounds"), "scenarios": report.get("scenarios"),
        "base_test": report.get("base_test"), "pool": report.get("pool"), "test": report.get("test"), "caveat": report.get("caveat"),
        "runs": runs,
    }


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


def _checkpoint_problems(report: Dict[str, Any], evidence: Dict[str, Any]) -> List[str]:
    expected = {m["model_id"]: m["checkpoint_sha256"] for m in evidence.get("models", [])}
    return [] if (report.get("checkpoint_sha256") or {}) == expected else ["SHA-256 checkpoint trong báo cáo không khớp bundle"]


def _holdout_report(evidence: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], str]:
    """Báo cáo test giữ riêng đã kiểm danh tính, hoặc (None, lý do)."""
    report = _load_json(reg.HOLDOUT_EVAL_JSON)
    if report is None:
        return None, f"chưa có {reg.HOLDOUT_EVAL_JSON} — chạy scripts/evaluate_published_models.py --holdout (cần dataset)"
    problems = _identity_problems(report, evidence) + _checkpoint_problems(report, evidence)
    if report.get("split") != reg.HOLDOUT_SPLIT or report.get("holdout_manifest_sha256") != reg.HOLDOUT_MANIFEST_SHA256:
        problems.append("manifest test khác bản ghim theo release")
    if ({k: _int(v) for k, v in (report.get("holdout_counts_by_source") or {}).items()} != reg.HOLDOUT_COUNTS
            or _int(report.get("holdout_count")) != sum(reg.HOLDOUT_COUNTS.values())):
        problems.append("số ảnh test theo nguồn khác release")
    for mid in MODEL_ORDER:
        by_source = ((report.get("models") or {}).get(mid) or {}).get("by_source") or {}
        for source, count in reg.HOLDOUT_COUNTS.items():
            if _int((by_source.get(source) or {}).get("count")) != count:
                problems.append(f"{mid}/{source}: số ảnh khác release")
    if problems:
        return None, "báo cáo test giữ riêng không khớp: " + "; ".join(sorted(set(problems)))
    return report, ""


def _holdout_per_class(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Số ảnh đúng theo lớp PlantDoc của tập giữ riêng (support × recall; recall lưu dạng thập phân)."""
    rows = []
    for name, item in (block.get("per_class_recall") or {}).items():
        support, recall = _int(item.get("support")), item.get("recall")
        if not support or recall is None:
            continue
        rows.append({"class": str(name), "support": support, "correct": int(round(float(recall) * support))})
    return sorted(rows, key=lambda r: r["class"])


def _holdout(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report, reason = _holdout_report(evidence)
    if report is None:
        return _withheld(reason)
    published = _published_by_id(evidence)
    models = []
    for mid in MODEL_ORDER:
        rep = report["models"][mid]
        calibration = rep.get("calibration") or {}
        entry = {
            "model_id": mid,
            "name": str(published[mid]["name"]),
            "phase": str(published[mid]["phase"]),
            "plantdoc": _metrics(rep["by_source"]["plantdoc"]),
            "plantvillage": _metrics(rep["by_source"]["plantvillage"]),
            "ece": {s: _calibration_row(calibration.get(s) or {}) for s in SOURCES},
            "plantdoc_per_class": _holdout_per_class(rep["by_source"]["plantdoc"]),
        }
        if "vs_recorded" in rep:
            entry["vs_recorded"] = {
                s: {"available": bool(v.get("available")), "reproduced": v.get("reproduced") is True,
                    "correct_diff_images": _int(v.get("correct_diff_images")), "accuracy_recorded": _round(v.get("accuracy_recorded"))}
                for s, v in (rep.get("vs_recorded") or {}).items()
            }
        models.append(entry)
    return {
        "available": True,
        "split": reg.HOLDOUT_SPLIT,
        "count": _int(report.get("holdout_count")),
        "counts_by_source": dict(reg.HOLDOUT_COUNTS),
        "used_for_checkpoint_selection": False,
        "caveat": str(report.get("caveat") or ""),
        "models": models,
    }


_FULLFRAME_MODES = ("annotator_crop", "full_frame", "center_square")


def _ratio(correct: Any, total: Optional[int]) -> Optional[float]:
    c = _int(correct)
    if c is None or not total or c < 0 or c > total:
        return None
    return c / total


def _fullframe(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Độ chính xác PlantDoc khi đưa NGUYÊN KHUNG so với ảnh cắt lá (scripts/eval_fullframe_plantdoc.py).

    Chỉ đọc báo cáo; mọi số đã ép kiểu. Giới hạn của phép đo nằm trong `caveat` (do báo cáo ghi, giao diện hiển thị)."""
    report = _load_json(reg.FULLFRAME_EVAL_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.FULLFRAME_EVAL_JSON} — chạy scripts/eval_fullframe_plantdoc.py --download (cần dataset)")
    problems = _identity_problems(report, evidence) + _checkpoint_problems(report, evidence)
    if report.get("crop_policy") != reg.FULLFRAME_CROP_POLICY:
        problems.append("chính sách cắt ảnh khác release")
    splits_in = report.get("splits") or {}
    splits = {}
    for split in ("val", "test"):
        block = ((splits_in.get(split) or {}).get("single_class_scenes_only")) or {}
        n = None
        entry: Dict[str, Any] = {"scenes": _int((splits_in.get(split) or {}).get("single_class_scenes"))}
        for mode in _FULLFRAME_MODES + ("published_crop",):
            m = block.get(mode)
            if not isinstance(m, dict):
                problems.append(f"{split}/{mode}: thiếu")
                continue
            n = _int(m.get("n"))
            entry[mode] = {"n": n, **{mid: _ratio((m.get(mid) or {}).get("correct"), n) for mid in MODEL_ORDER}}  # correct/n: không làm tròn hai lần ở giao diện
        counts = {_int((block.get(mode) or {}).get("n")) for mode in _FULLFRAME_MODES + ("published_crop",)}
        if len(counts) != 1 or None in counts:
            problems.append(f"{split}: số ảnh giữa các cách đưa ảnh vào không khớp")
        splits[split] = entry
    if problems:
        return _withheld("báo cáo ảnh nguyên khung không khớp: " + "; ".join(sorted(set(problems))))
    return {
        "available": True,
        "crop_policy": str(report.get("crop_policy")),
        "scenes_verified": _int(report.get("scenes_verified_sha256")),
        "excluded_scenes": _int((report.get("excluded_unavailable") or {}).get("scenes")),
        "caveat": str(report.get("caveat") or ""),
        "splits": splits,
    }


def _stat_interval(block: Dict[str, Any]) -> Dict[str, Any]:
    return {"estimate": _round(block.get("estimate")), "low": _round(block.get("low")), "high": _round(block.get("high")),
            "clusters": _int(block.get("clusters"))}


def _statistics(evidence: Dict[str, Any]) -> Dict[str, Any]:
    report = _load_json(reg.STATISTICS_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.STATISTICS_JSON} — chạy scripts/report_statistics.py (cần dataset)")
    problems = _identity_problems(report, evidence) + _checkpoint_problems(report, evidence)
    if problems:
        return _withheld("báo cáo thống kê không khớp bundle: " + "; ".join(problems))
    published = _published_by_id(evidence)
    holdout, _ = _holdout_report(evidence)
    splits: Dict[str, Any] = {}
    for split, block in (report.get("splits") or {}).items():
        if not block.get("available"):
            splits[split] = _withheld(str(block.get("reason") or "chưa có"))
            continue
        expected_manifest = evidence["validation_manifest_sha256"] if split == "val" else reg.HOLDOUT_MANIFEST_SHA256
        if block.get("manifest_sha256") != expected_manifest:
            splits[split] = _withheld("manifest của phần thống kê khác bản đã kiểm")
            continue
        out_sources: Dict[str, Any] = {}
        bad = None
        for source in SOURCES:
            data = (block.get("by_source") or {}).get(source) or {}
            models = {}
            for mid in MODEL_ORDER:
                m = (data.get("models") or {}).get(mid) or {}
                correct = _int(m.get("correct"))
                if split == "val":  # khớp số công bố trong dung sai của báo cáo tái đánh giá (≤ 1 ảnh)
                    pub = published[mid]["metrics"][source]
                    if _int(data.get("count")) != _int(pub["count"]) or correct is None or abs(correct - round(pub["top1_accuracy"] * pub["count"])) > 1:
                        bad = f"{mid}/{source} (val) khác số công bố"
                else:
                    if _int(data.get("count")) != reg.HOLDOUT_COUNTS[source]:
                        bad = f"{source} (test): số ảnh khác release"
                    elif holdout is None:  # không có báo cáo test đã kiểm để đối chiếu thì không hiện số test
                        bad = "chưa có báo cáo test giữ riêng khớp bundle để đối chiếu"
                    elif correct is None or correct != _int(holdout["models"][mid]["by_source"][source].get("correct")):
                        bad = f"{mid}/{source} (test) khác báo cáo test giữ riêng"
                models[mid] = {"correct": correct, **_stat_interval(m)}
            pairs = {}
            for name, pair in (data.get("pairs") or {}).items():
                pairs[str(name)] = {
                    "a": str(pair.get("a")), "b": str(pair.get("b")),
                    "a_right_b_wrong": _int(pair.get("a_right_b_wrong")), "a_wrong_b_right": _int(pair.get("a_wrong_b_right")),
                    "p_value": _round(pair.get("p_value"), 8),
                    "difference": _stat_interval(pair.get("difference_ci") or {}),
                }
            entry = {"count": _int(data.get("count")), "clusters": _int(data.get("clusters")), "models": models, "pairs": pairs}
            sensitivity = data.get("same_leaf_sensitivity")
            if sensitivity:
                entry["same_leaf"] = {
                    "ungrouped_images": _int(sensitivity.get("ungrouped_images")),
                    "suspected_reshoots": _int(sensitivity.get("ungrouped_with_adjacent_train_number")),
                    "models": {mid: {k: _round(v) for k, v in (sensitivity.get("models") or {}).get(mid, {}).items()} for mid in MODEL_ORDER},
                }
            out_sources[source] = entry
        splits[split] = _withheld(bad) if bad else {
            "available": True,
            "count": _int(block.get("count")),
            "used_for_checkpoint_selection": bool(block.get("used_for_checkpoint_selection")),
            "by_source": out_sources,
        }
    selection = report.get("selection_effect") or {}
    return {
        "available": True,
        "bootstrap_resamples": _int(report.get("bootstrap_resamples")),
        "cluster_definition": {k: str(v) for k, v in (report.get("cluster_definition") or {}).items()},
        "reshoot_heuristic": str(report.get("reshoot_heuristic") or ""),
        "best_local_chosen_on_val": {k: str(v) for k, v in (report.get("best_local_chosen_on_val") or {}).items()},
        "splits": splits,
        "selection_effect": {
            "available": selection.get("available") is True,
            "chosen_epoch": _int(selection.get("chosen_epoch")),
            "epochs": _int(selection.get("epochs")),
            "chosen_plantdoc_accuracy": _round(selection.get("chosen_plantdoc_accuracy")),
            "last_epoch": _int(selection.get("last_epoch")),
            "last_plantdoc_accuracy": _round(selection.get("last_plantdoc_accuracy")),
            "max_plantdoc_accuracy": _round(selection.get("max_plantdoc_accuracy")),
            "chosen_rank_among_epochs": _int(selection.get("chosen_rank_among_epochs")),
            "selection_metric": str(selection.get("selection_metric") or ""),
            "fedavg_note": str(selection.get("fedavg_note") or ""),
        },
    }


def temperatures(evidence: Dict[str, Any]) -> Dict[str, float]:
    """T của từng mô hình (fit trên calibration) — chỉ khi báo cáo tái đánh giá khớp bundle; ngược lại rỗng."""
    if not evidence.get("available"):
        return {}
    section = _guard("tái đánh giá", lambda: _reeval(evidence))
    if not section.get("available"):
        return {}
    return {m["model_id"]: m["temperature"] for m in section["models"] if m.get("temperature")}


def _matrix(value: Any, size: int, total: int) -> Optional[List[List[int]]]:
    """Ma trận size×size số nguyên không âm, tổng đúng `total`; ngược lại None."""
    if not isinstance(value, list) or len(value) != size:
        return None
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            return None
        ints = [_int(v) for v in row]
        if any(v is None or v < 0 for v in ints):
            return None
        rows.append(ints)
    return rows if sum(map(sum, rows)) == total else None


def details(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Ma trận nhầm lẫn (val + test giữ riêng) và thư viện ảnh "sai mà tự tin" cho chương Kết quả. Báo cáo hỏng => available: false."""
    return _guard("chi tiết (ma trận nhầm lẫn / ảnh lỗi)", lambda: _details(evidence))


def partitions(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Phân bố lớp theo client của mọi cách chia (chỉ dữ liệu). Báo cáo hỏng => available: false."""
    return _guard("các cách chia", lambda: _partitions(evidence))


def _details(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not evidence.get("available"):
        return {"available": False, "reason": evidence.get("reason") or "bộ đối chứng chưa khả dụng"}
    classes = [str(c) for c in evidence.get("classes") or []]
    published = _published_by_id(evidence)
    confusion: Dict[str, Any] = {}
    reeval = _load_json(reg.LOCAL_REEVAL_JSON)
    if reeval is not None and not (_identity_problems(reeval, evidence) + _checkpoint_problems(reeval, evidence)):
        confusion["val"] = {
            mid: {s: _matrix((((reeval.get("detail") or {}).get(mid) or {}).get("confusion_matrix") or {}).get(s), len(classes),
                             int(published[mid]["metrics"][s]["count"])) for s in SOURCES}
            for mid in MAIN_MODELS
        }
    holdout, _ = _holdout_report(evidence)
    if holdout is not None:
        confusion["test"] = {
            mid: {s: _matrix((((holdout.get("detail") or {}).get(mid) or {}).get("confusion_matrix") or {}).get(s), len(classes),
                             reg.HOLDOUT_COUNTS[s]) for s in SOURCES}
            for mid in MAIN_MODELS
        }
    errors = _withheld(f"chưa có {reg.ERROR_GALLERY_CATALOG_JSON} — chạy scripts/build_error_gallery.py (cần dataset)")
    catalog = _load_json(reg.ERROR_GALLERY_CATALOG_JSON)
    if catalog is not None:
        expected = {m: published[m]["checkpoint_sha256"] for m in MAIN_MODELS}
        if catalog.get("validation_manifest_sha256") != evidence.get("validation_manifest_sha256") or catalog.get("checkpoint_sha256") != expected:
            errors = _withheld("thư viện ảnh lỗi không khớp bundle đã kiểm hash")
        else:
            items = []
            for item in catalog.get("items") or []:
                wrong_by = {m: {"predicted": str(v.get("predicted")), "confidence": _round(v.get("confidence"))}
                            for m, v in (item.get("wrong_by") or {}).items() if m in MAIN_MODELS}
                thumb = str(item.get("thumb") or "")
                if (not wrong_by or not thumb or "/" in thumb or "\\" in thumb or ".." in thumb
                        or item.get("source") not in SOURCES):
                    continue
                items.append({"sample_id": str(item.get("sample_id")), "thumb": thumb, "source": str(item.get("source")),
                              "target": str(item.get("target")), "wrong_by": wrong_by})
            errors = {"available": True, "threshold": _round(catalog.get("threshold")), "attribution": str(catalog.get("attribution") or ""),
                      "count": len(items), "items": items}
    return {"available": True, "classes": classes, "confusion": confusion, "errors": errors}


def _partitions(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Cách chia đã huấn luyện phải khớp n_k của bundle."""
    if not evidence.get("available"):
        return {"available": False, "reason": evidence.get("reason") or "bộ đối chứng chưa khả dụng"}
    report = _load_json(reg.PARTITION_SCHEMES_JSON)
    if report is None:
        return _withheld(f"chưa có {reg.PARTITION_SCHEMES_JSON} — chạy scripts/report_partition_schemes.py (cần dataset)")
    classes = [str(c) for c in report.get("classes") or []]
    if classes != [str(c) for c in evidence.get("classes") or []]:
        return _withheld("thứ tự lớp của báo cáo khác taxonomy của bundle")
    total = _int(report.get("total_train_samples"))
    schemes = {}
    for name, scheme in (report.get("schemes") or {}).items():
        clients = {}
        for client, data in (scheme.get("clients") or {}).items():
            per_class = [_int(v) or 0 for v in data.get("per_class") or []]
            if len(per_class) != len(classes) or sum(per_class) != _int(data.get("n_k")):
                return _withheld(f"{name}/{client}: số ảnh theo lớp không cộng đúng n_k")
            clients[str(client)] = {"n_k": _int(data.get("n_k")), "classes_with_images": _int(data.get("classes_with_images")),
                                    "plantdoc_share": _round(data.get("plantdoc_share")), "per_class": per_class}
        if sum(c["n_k"] for c in clients.values()) != total:
            return _withheld(f"{name}: tổng n_k khác số ảnh train")
        schemes[str(name)] = {"family": str(scheme.get("family")), "family_label": str(scheme.get("family_label")),
                              "alpha": _round(scheme.get("alpha")), "has_trained_models": bool(scheme.get("has_trained_models")),
                              "mean_js_divergence": _round(scheme.get("mean_js_divergence")), "clients": clients}
    trained = schemes.get(reg.PARTITION_SCHEME)
    if trained is None or {c: v["n_k"] for c, v in trained["clients"].items()} != {
            c: _int(v) for c, v in (evidence.get("shard_counts") or {}).items()}:
        return _withheld("cách chia đã huấn luyện không khớp n_k của bundle")
    return {"available": True, "classes": classes, "total_train": total, "trained_scheme": reg.PARTITION_SCHEME, "schemes": schemes}


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
            **{name: _withheld(reason) for name in ("reeval", "plantdoc", "domain_gap", "leakage", "shards", "holdout", "statistics", "fullframe", "plantvillage_reference", "community_simulation", "training")},
        }
    return {
        "available": True,
        "identity": _identity(evidence),
        "training": _guard("cấu hình huấn luyện", lambda: training_report(evidence, _load_json)),
        "reeval": _guard("tái đánh giá", lambda: _reeval(evidence)),
        "plantdoc": _guard("PlantDoc theo lớp", lambda: _plantdoc(evidence)),
        "domain_gap": _guard("khoảng cách miền", lambda: _domain_gap(evidence)),
        "leakage": _guard("audit rò rỉ", lambda: _leakage(evidence)),
        "shards": _guard("shard", lambda: _shards(evidence)),
        "holdout": _guard("test giữ riêng", lambda: _holdout(evidence)),
        "statistics": _guard("thống kê", lambda: _statistics(evidence)),
        "fullframe": _guard("ảnh nguyên khung", lambda: _fullframe(evidence)),
        "plantvillage_reference": _guard("PlantVillage tham chiếu", lambda: _plantvillage_reference(evidence)),
        "community_simulation": _guard("mô phỏng cộng đồng", lambda: _community_simulation(evidence)),
        "checklist": _guard("checklist", _checklist),
    }
