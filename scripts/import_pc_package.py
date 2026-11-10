"""Nhập gói kết quả huấn luyện PC v3 (nhánh pc-v3-ket-qua-*, thư mục pc_v3_ket_qua/) thành MỘT gói mô hình cho web (artifacts/model_bundles/<tên>/).

Khác scripts/import_run_bundle.py: nhóm PC đã train xong và chỉ giao checkpoint + `model_info.json` (không giao thư mục run của scripts/train_v4.py),
nên lịch sử từng vòng lấy từ khóa `history` nằm TRONG checkpoint, còn mọi con số của web do đồ án tự đánh giá lại trên CPU.

Gói web có 13 mô hình:
  - fp32 (state_dict): centralized (AdamW), centralized_sgd, fedavg, scaffold, local_00..local_04;
  - int8 động chỉ lớp Linear (TorchScript): centralized_int8, centralized_sgd_int8, fedavg_int8, scaffold_int8. Tệp int8 được buộc vào SHA-256 của
    checkpoint fp32 gốc (xem docs/TICH_HOP_INT8_PC_V3.md) và đo TỪNG ẢNH MỘT, đúng cách web suy luận.

Chỉ dùng tập validation. Không đọc, không chạy trên tập test, calibration hay tập ngoài miền (nhóm PC mở tập test đúng một lần ở bước P6).
Gói ghi `evidence_scope = "validation_only"` để web biết các mục báo cáo cần test/calibration là cố ý bị giữ lại.

    python scripts/import_pc_package.py --package ../pc-v3-ket-qua/pc_v3_ket_qua --name pc_v3_s42 --report-dir ../pc-v3-ket-qua/pc_v3_ket_qua/kiem_lai_web

Chưa đổi con trỏ `artifacts/active_bundle.json`: việc đó do scripts/swap_web_models.py làm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.import_run_bundle import (  # noqa: E402
    CONTRACT_FILES, artifact_manifest, class_coverage, comparison_rows, count_groups, evaluate_on_cpu, read_jsonl, sha256_file,
    validation_metrics, write_json)
from src import bundle, registry as reg  # noqa: E402
from src.common.checkpoint_io import load_checkpoint  # noqa: E402

EVIDENCE_LEVEL = "externally_trained_locally_reevaluated"
EVIDENCE_SCOPE = "validation_only"
METRICS_SOURCE = "validation_comparison.csv do đồ án tự tính trên CPU từ checkpoint đã băm (scripts/import_pc_package.py), cùng đường suy luận với web"
TEST_POLICY = ("Tập kiểm thử cuối chỉ được nhóm train mở một lần (bước P6, số nằm ở nhánh kết quả của họ). Gói này chỉ đánh giá lại trên tập kiểm định; "
               "đồ án không chạy mô hình PC v3 trên tập kiểm thử, tập niêm phong (calibration) hay tập ngoài miền.")
INT8_SENTENCE = "int8 động, chỉ lớp Linear (classifier); các lớp conv vẫn fp32"
PHASE = "PC v3"
CHECKPOINT_FILE = "checkpoint_best.pt"
INT8_FILE = "model_dynamic_linear_torchscript.pt"      # tên tệp nguyên văn trong gói của nhóm PC (giữ để truy được nguồn)
TOLERANCE = 0.002                                      # luật của nhóm PC: số tính lại lệch số công bố không quá 0,002

# (model_id, thư mục role trong gói PC, tên hiển thị, kiểu huấn luyện)
FP32_MODELS = (
    ("centralized", "centralized_adamw", "Centralized AdamW", "centralized"),
    ("centralized_sgd", "centralized_sgd", "Centralized SGD", "centralized"),
    ("fedavg", "fedavg", "FedAvg", "federated"),
    ("scaffold", "scaffold", "SCAFFOLD", "federated"),
)


def model_plan() -> list[dict[str, Any]]:
    """13 mô hình của gói: id giữ `centralized`, `fedavg`, `local_xx` như mọi gói trước để các bộ đọc cũ dùng lại được; thư mục int8 trùng model_id."""
    plan: list[dict[str, Any]] = []
    for model_id, role, name, kind in FP32_MODELS:
        plan.append({"model_id": model_id, "dir": model_id, "file": CHECKPOINT_FILE, "role": role, "source": f"{role}/{CHECKPOINT_FILE}",
                     "display_name": name, "phase": PHASE, "method": name, "kind": kind, "format": bundle.FORMAT_STATE_DICT, "client": None})
    for k in range(5):
        plan.append({"model_id": f"local_{k:02d}", "dir": "local", "file": f"client_{k:02d}/{CHECKPOINT_FILE}", "role": f"local/client_{k:02d}",
                     "source": f"local/client_{k:02d}/{CHECKPOINT_FILE}", "display_name": f"Local {k:02d}", "phase": PHASE, "method": "Local-only",
                     "kind": "local", "format": bundle.FORMAT_STATE_DICT, "client": k})
    for model_id, role, name, kind in FP32_MODELS:
        plan.append({"model_id": f"{model_id}_int8", "dir": f"{model_id}_int8", "file": INT8_FILE, "role": f"int8_dynamic_linear/{role}",
                     "source": f"int8_dynamic_linear/{role}/{INT8_FILE}", "display_name": f"{name} int8", "phase": PHASE,
                     "method": f"{name} int8 (động, chỉ lớp Linear)", "kind": kind, "format": bundle.FORMAT_TORCHSCRIPT_INT8, "client": None,
                     "fp32_of": model_id, "fp32_role": role})
    return plan


def release_sha_matches(release_manifest: Path, claimed: str) -> bool:
    """Nhóm PC băm release_manifest.json có xuống dòng CRLF cuối tệp; bản ở máy này kết thúc bằng LF. Chấp nhận cả hai, như audit_release_leakage.py."""
    raw = release_manifest.read_bytes()
    return claimed in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.rstrip(b"\r\n") + b"\r\n").hexdigest()}


def git_commit(directory: Path) -> str:
    out = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() if out.returncode == 0 and len(out.stdout.strip()) == 40 else "unknown"


def relative_to_root(path: Path) -> str:
    """Đường dẫn tương đối so với thư mục dự án (không lộ tên thư mục máy); ngoài dự án thì giữ nguyên."""
    root = reg.abs_path(".")
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- kiểm gói nguồn
def pinned_bytes(path: Path, digest: str) -> bytes:
    """Byte của tệp văn bản đúng mã băm đã khai. Git trên Windows (core.autocrlf) có thể đổi LF thành CRLF khi checkout: chấp nhận bản LF khớp mã băm,
    không chấp nhận bất kỳ byte nào khác."""
    raw = path.read_bytes()
    for candidate in (raw, raw.replace(b"\r\n", b"\n")):
        if hashlib.sha256(candidate).hexdigest() == digest:
            return candidate
    raise ValueError(f"{path.name}: không khớp mã băm {digest[:8]} (cả bản gốc lẫn bản LF)")


def verify_package(models_dir: Path, plan: list[dict[str, Any]], release: Path, classes: list[str]) -> dict[str, Any]:
    """Kiểm gói của nhóm PC trước khi chép: SHA từng tệp, danh tính checkpoint (release, hạt giống, W0, vòng cuối), hợp đồng, 38 lớp. Sai điểm nào thì dừng."""
    manifest = read_json(models_dir / "import_manifest.json")
    if manifest.get("release_name") != release.name:
        raise ValueError(f"gói PC dùng release {manifest.get('release_name')!r}, khác {release.name!r}")
    claimed = manifest["release_manifest_sha256"]
    if not release_sha_matches(release / "release_manifest.json", claimed):
        raise ValueError("release_manifest_sha256 của gói PC không khớp release ở máy")
    if list(manifest["classes"]) != classes:
        raise ValueError("thứ tự 38 lớp của gói PC khác taxonomy của release")
    contract: dict[str, bytes] = {}
    for relative, digest in manifest["contract_hashes"].items():
        contract[relative] = pinned_bytes(models_dir / "contract" / Path(relative).name, digest)
    if set(manifest["contract_hashes"]) != set(CONTRACT_FILES):
        raise ValueError("gói PC khai báo hợp đồng khác bộ 3 tệp mà web kiểm")
    steps: dict[str, int] = {}
    shas: dict[str, str] = {}
    for spec in plan:
        path = models_dir / spec["source"]
        info = read_json(path.with_name("model_info.json"))
        if sha256_file(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"{spec['model_id']}: tệp khác SHA-256/kích thước trong model_info.json")
        shas[spec["model_id"]] = info["sha256"]
        if spec["format"] == bundle.FORMAT_TORCHSCRIPT_INT8:
            if info.get("parent_fp32_role") != spec["fp32_role"]:
                raise ValueError(f"{spec['model_id']}: model_info nói bản gốc là {info.get('parent_fp32_role')!r}, mong {spec['fp32_role']!r}")
            continue
        checkpoint = load_checkpoint(io.BytesIO(path.read_bytes()))
        state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        if state is None or len(state) != 244:
            raise ValueError(f"{spec['model_id']}: checkpoint không có đủ 244 tensor của MobileNetV3-Small")
        if checkpoint.get("release_manifest_sha256") != claimed or checkpoint.get("seed") != manifest["seed"] \
                or checkpoint.get("w0_fingerprint") != manifest["w0_fingerprint"]:
            raise ValueError(f"{spec['model_id']}: danh tính trong checkpoint (release/seed/W0) khác gói")
        step = int(checkpoint.get("round") or checkpoint.get("epoch") or 0)
        if step != int(info["checkpoint_step"]) or step < 1:
            raise ValueError(f"{spec['model_id']}: vòng/epoch trong checkpoint ({step}) khác model_info ({info['checkpoint_step']})")
        steps[spec["model_id"]] = step
    for spec in plan:
        if spec["format"] == bundle.FORMAT_TORCHSCRIPT_INT8:
            steps[spec["model_id"]] = steps[spec["fp32_of"]]
    return {"manifest": manifest, "steps": steps, "shas": shas, "contract": contract}


def partition_stats(release: Path, scheme: str) -> tuple[list[int], dict[str, dict[str, int]]]:
    """n_k và histogram lớp của 5 client từ phân hoạch của release (cùng phân hoạch đã dùng để train)."""
    sizes: list[int] = []
    histograms: dict[str, dict[str, int]] = {}
    for k in range(5):
        rows = read_jsonl(release / "partitions" / scheme / f"client_{k:02d}.jsonl")
        sizes.append(len(rows))
        histogram: dict[str, int] = {}
        for row in rows:
            histogram[str(row["class_id"])] = histogram.get(str(row["class_id"]), 0) + 1
        histograms[f"client_{k:02d}"] = histogram
    return sizes, histograms


# --------------------------------------------------------------------------- lịch sử từng vòng
def history_rows(history: list[dict[str, Any]], kind: str, shard_sizes: list[int]) -> list[dict[str, Any]]:
    """Một dòng mỗi epoch/vòng từ khóa `history` của checkpoint. Federated thêm `clients` (n_k, trọng số p_k, mất mát, số bước); `delta_l2` không đưa vào."""
    total = sum(shard_sizes)
    out = []
    for item in history:
        step = int(item.get("epoch") or item["round"])
        row: dict[str, Any] = {
            "epoch": step, "lr": item.get("lr", item.get("lr_next_round")), "val_accuracy": item["val_accuracy"], "val_loss": item.get("val_loss"),
            "val_macro_f1": item.get("val_macro_f1_raw"), "seconds": item.get("duration_sec"),
            "plantdoc_macro_f1_supported": item.get("plantdoc_macro_f1_supported"),
            "plantvillage_macro_f1_supported": item.get("plantvillage_macro_f1_supported")}
        if kind == "federated":
            telemetry = item.get("client_telemetry") or []
            steps = sum(c["steps"] for c in telemetry) or 1
            row["train_loss"] = sum(c["loss"] * c["steps"] for c in telemetry) / steps if telemetry else None
            row["clients"] = []
            for c in telemetry:
                k = int(str(c["client_id"]).rsplit("_", 1)[1])
                if int(c["raw_images_n_k"]) != shard_sizes[k]:
                    raise ValueError(f"vòng {step}: n_k của {c['client_id']} khác phân hoạch")
                row["clients"].append({"client_id": c["client_id"], "train_loss": c["loss"], "optimizer_steps_applied": c["optimizer_steps_applied"],
                                       "sample_count_n_k": shard_sizes[k], "weight_p_k": shard_sizes[k] / total})
        else:
            row["train_loss"] = item.get("train_loss")
        out.append(row)
    return out


# --------------------------------------------------------------------------- đối chiếu với số của nhóm PC
def compare_with_pc(models_dir: Path, plan: list[dict[str, Any]], metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in plan:
        claimed = read_json((models_dir / spec["source"]).with_name("validation_metrics.json"))["validation"]
        blocks = {"overall": (claimed["overall"], metrics[spec["model_id"]]["overall"])}
        for source, block in metrics[spec["model_id"]]["by_source"].items():
            blocks[source] = (claimed["by_source"][source], block)
        for source, (pc, ours) in blocks.items():
            for key in ("top1_accuracy", "macro_f1_supported", "crop_accuracy"):
                diff = float(ours[key]) - float(pc[key])
                rows.append({"model_id": spec["model_id"], "model": spec["display_name"], "source": source, "count": ours["count"], "metric": key,
                             "pc_value": float(pc[key]), "web_value": float(ours[key]), "diff": diff, "within_tolerance": abs(diff) <= TOLERANCE})
    return rows


def write_report(report_dir: Path, rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    header = ["model_id", "model", "source", "count", "metric", "pc_value", "web_value", "diff", "within_tolerance"]
    with (report_dir / "KIEM_LAI_VALIDATION.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    worst = max(rows, key=lambda r: abs(r["diff"]))
    out_of_tolerance = [r for r in rows if not r["within_tolerance"]]
    write_json(report_dir / "KIEM_LAI_VALIDATION.json", {
        **context, "tolerance": TOLERANCE, "comparisons": len(rows), "out_of_tolerance": len(out_of_tolerance),
        "worst": {k: worst[k] for k in ("model_id", "source", "metric", "pc_value", "web_value", "diff")},
        "out_of_tolerance_rows": [{k: r[k] for k in ("model_id", "source", "metric", "pc_value", "web_value", "diff")} for r in out_of_tolerance]})


# --------------------------------------------------------------------------- nhập gói
def import_package(package: Path, name: str, target: Path, release: Path, dataset_root: Path, batch_size: int = 64, workers: int = 5,
                   logits: dict[str, np.ndarray] | None = None, report_dir: Path | None = None) -> Path:
    if target.exists():
        raise FileExistsError(f"Gói đã có, không ghi đè: {target}")
    models_dir = package / "models"
    plan = model_plan()
    release_manifest = read_json(release / "release_manifest.json")
    class_to_idx = read_json(release / "taxonomy.json")["class_to_idx"]
    classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    if len(classes) != 38 or [class_to_idx[c] for c in classes] != list(range(38)):
        raise ValueError("taxonomy.json của release không phải 38 lớp đánh số 0..37")
    checked = verify_package(models_dir, plan, release, classes)
    pc = checked["manifest"]
    release_sha = pc["release_manifest_sha256"]
    validation_sha = release_manifest["manifest_hashes"]["val"]
    if sha256_file(release / "manifests" / "val.jsonl") != validation_sha:
        raise ValueError("val.jsonl khác mã băm trong release_manifest.json")
    val_rows = read_jsonl(release / "manifests" / "val.jsonl")
    if count_groups(val_rows) != pc["validation_counts"]:
        raise ValueError("số ảnh validation của release khác gói PC khai báo")
    # Tập kiểm thử chỉ lấy mã băm + số lượng đã biết của gói cũ; KHÔNG mở tệp manifest của tập kiểm thử.
    holdout = {"manifest_sha256": release_manifest["manifest_hashes"][reg.HOLDOUT_SPLIT], "counts": bundle.LEGACY_HOLDOUT["counts"]}
    if holdout["manifest_sha256"] != bundle.LEGACY_HOLDOUT["manifest_sha256"] \
            or release_manifest["counts"]["legacy_diagnostic_test_samples"] != sum(holdout["counts"].values()):
        raise ValueError("mã băm/số lượng tập kiểm thử của release khác bản đã ghim (bundle.LEGACY_HOLDOUT)")
    scheme = pc["partition_scheme"]
    shard_sizes, histograms = partition_stats(release, scheme)
    if sum(shard_sizes) != bundle.LEGACY_SHARD_TOTAL:
        raise ValueError(f"tổng n_k {sum(shard_sizes)} khác {bundle.LEGACY_SHARD_TOTAL}")
    for k in range(5):
        info = read_json(models_dir / "local" / f"client_{k:02d}" / "model_info.json")
        if info["metrics_validation_pv_pd_v3"]["n_k"] != shard_sizes[k]:
            raise ValueError(f"client_{k:02d}: n_k trong model_info khác phân hoạch")

    commit = git_commit(package)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "contract").mkdir()
        for relative in CONTRACT_FILES:
            (staging / "contract" / Path(relative).name).write_bytes(checked["contract"][relative])
        for spec in plan:
            destination = staging / spec["dir"] / spec["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(models_dir / spec["source"], destination)

        started = time.time()
        if logits is None:
            logits = evaluate_on_cpu(plan, staging, val_rows, dataset_root, batch_size, workers,
                                     cache=reg.abs_path(f"data/reeval/bundle_{name}_logits.npz"))
        print(f"[bundle] đánh giá xong sau {time.time() - started:.0f} giây", flush=True)

        sha = {spec["model_id"]: sha256_file(staging / spec["dir"] / spec["file"]) for spec in plan}
        if sha != checked["shas"]:
            raise ValueError("tệp đã chép khác tệp nguồn")
        val_relative = relative_to_root(release / "manifests" / "val.jsonl")
        metrics: dict[str, dict[str, Any]] = {}
        all_rows: list[dict[str, Any]] = []
        for spec in plan:
            model_id = spec["model_id"]
            metrics[model_id] = validation_metrics(logits[model_id], val_rows, classes)
            payload = {"checkpoint": f"pc_v3_ket_qua/models/{spec['source']}@{commit[:9]}", "checkpoint_sha256": sha[model_id],
                       "checkpoint_round_or_epoch": checked["steps"][model_id], "release_manifest_sha256": release_sha,
                       "validation": {"manifest": val_relative, "manifest_sha256": validation_sha, **metrics[model_id]}}
            if spec["format"] == bundle.FORMAT_TORCHSCRIPT_INT8:
                payload.update(format=spec["format"], fp32_of=spec["fp32_of"], evaluated_per_image=True)
            write_json((staging / spec["dir"] / spec["file"]).with_name("validation_metrics.json"), payload)
            all_rows += comparison_rows(spec["display_name"], metrics[model_id], sha[model_id], release_sha, validation_sha)
        header = ["model", "source", "count", "top1_accuracy", "macro_f1_supported", "crop_accuracy", "checkpoint_sha256",
                  "release_manifest_sha256", "validation_manifest_sha256"]
        directories = list(dict.fromkeys(spec["dir"] for spec in plan))
        for directory in directories:
            with (staging / directory / "validation_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerows(all_rows)

        common = {"seed": pc["seed"], "w0_fingerprint": pc["w0_fingerprint"], "release_manifest_sha256": release_sha}
        selection = "checkpoint cuối (epoch/vòng cuối), không chọn theo validation"
        for spec in plan:
            if spec["kind"] == "local" or spec["format"] != bundle.FORMAT_STATE_DICT:
                continue
            checkpoint = load_checkpoint(io.BytesIO((staging / spec["dir"] / spec["file"]).read_bytes()))
            history = history_rows(checkpoint["history"], spec["kind"], shard_sizes)
            step = checked["steps"][spec["model_id"]]
            if len(history) != step:
                raise ValueError(f"{spec['model_id']}: lịch sử {len(history)} dòng khác vòng cuối {step}")
            key = "best_epoch" if spec["kind"] == "centralized" else "best_round"
            write_json(staging / spec["dir"] / "summary.json", {
                **common, "history": history, key: step, "total_runtime_seconds": round(sum(h["seconds"] or 0 for h in history), 1), "selection": selection})
        clients = {}
        for k in range(5):
            model_id = f"local_{k:02d}"
            by = metrics[model_id]["by_source"]
            clients[f"client_{k:02d}"] = {
                "n_k": shard_sizes[k], "class_coverage": class_coverage(histograms[f"client_{k:02d}"], val_rows),
                "best_validation": {"round": checked["steps"][model_id], "accuracy": metrics[model_id]["overall"]["top1_accuracy"],
                                    "plantdoc_macro_f1_supported": by.get("plantdoc", {}).get("macro_f1_supported"),
                                    "plantvillage_macro_f1_supported": by.get("plantvillage", {}).get("macro_f1_supported")}}
            write_json(staging / "local" / f"client_{k:02d}" / "summary.json", {
                **common, "client": k, "n_k": shard_sizes[k], "history": [], "selection": selection})
        write_json(staging / "local" / "combined_summary.json", {
            "backend": "local_only", "partition_scheme": scheme, "rounds_per_client": checked["steps"]["local_00"], **common, "clients": clients})

        branches = {}
        for directory in directories:
            role = "local" if directory == "local" else next(spec["role"] for spec in plan if spec["dir"] == directory)
            branches[directory] = f"pc_v3_ket_qua/models/{role}@{commit[:9]}"
            write_json(staging / directory / "artifact_manifest.json", artifact_manifest(staging / directory, branches[directory], release_sha, validation_sha))

        models = []
        for spec in plan:
            entry = {k: spec[k] for k in ("model_id", "dir", "file", "display_name", "phase", "method")} | {"registry_id": f"web_{spec['model_id']}"}
            if spec["format"] == bundle.FORMAT_TORCHSCRIPT_INT8:
                entry.update(format=spec["format"], fp32_of=spec["fp32_of"], fp32_checkpoint_sha256=sha[spec["fp32_of"]], quantization="dynamic_linear_qint8")
            models.append(entry)
        release_relative = relative_to_root(release)
        manifest = {
            "bundle_id": name, "protocol_id": f"mixed_{release.name}_pc_r4_seed{pc['seed']}_validation",
            "evidence_level": EVIDENCE_LEVEL, "evidence_scope": EVIDENCE_SCOPE, "metrics_source": METRICS_SOURCE, "test_policy": TEST_POLICY,
            "release_manifest_sha256": release_sha, "release_name": release.name, "release_dir": release_relative,
            "validation_manifest_sha256": validation_sha, "validation_counts": count_groups(val_rows),
            "w0_fingerprint": pc["w0_fingerprint"], "w0_in_checkpoints": True, "seed": pc["seed"], "partition_scheme": scheme,
            "num_classes": 38, "transform": "canonical_v1",
            "contract_hashes": {relative: sha256_file(staging / "contract" / Path(relative).name) for relative in CONTRACT_FILES},
            "classes": classes, "shard_total": sum(shard_sizes), "holdout": holdout,
            "sources": {directory: {"branch": branch, "commit": commit} for directory, branch in branches.items()},
            "models": models, "int8_label": INT8_SENTENCE,
            "package": {"source_bundle_id": pc["bundle_id"], "branch_commit": commit, "selected_solution": pc.get("selected_solution"),
                        "import_manifest_sha256": sha256_file(models_dir / "import_manifest.json")},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(staging / "import_manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if report_dir is not None:
        import torch

        rows = compare_with_pc(models_dir, plan, metrics)
        write_report(report_dir, rows, {
            "bundle": name, "source_bundle_id": pc["bundle_id"], "package_commit": commit, "validation_images": len(val_rows),
            "device": "cpu", "torch": torch.__version__, "int8_evaluated_per_image": True, "created_utc": manifest["created_utc"]})
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="thư mục pc_v3_ket_qua (chứa models/) của nhánh kết quả")
    parser.add_argument("--name", required=True, help="tên thư mục gói, ví dụ pc_v3_s42")
    parser.add_argument("--release", type=Path, default=None, help="thư mục release (mặc định data/dataset_sources/mixed/pv_pd_v3)")
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None, help="ghi bảng đối chiếu số tính lại với số của nhóm PC vào thư mục này")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    release = args.release or reg.abs_path(reg.RELEASE_V3_DIR)
    target = args.target or reg.abs_path(f"{bundle.BUNDLES_DIR}/{args.name}")
    path = import_package(args.package.resolve(), args.name, target, release, reg.abs_path(reg.DATASET_SOURCES_DIR), args.batch_size, args.workers,
                          report_dir=args.report_dir)
    print(f"[bundle] đã tạo {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
