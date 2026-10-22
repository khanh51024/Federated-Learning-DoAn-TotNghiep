"""Gói mô hình đang phục vụ web: MỘT nguồn duy nhất cho nhận dạng (release, tập đánh giá, hạt giống, danh sách mô hình, tên hiển thị).

Trước đây các giá trị này nằm rải rác thành hằng số (SHA release, "Centralized v6", 37.238 ảnh, 269/4.344...) ở hơn 40 chỗ, nên thay mô hình
nghĩa là sửa từng chỗ. Nay:
  * `artifacts/active_bundle.json` chỉ ra thư mục gói đang dùng (`{"bundle_dir": "artifacts/model_bundles/v4b_s42"}`); không có tệp thì dùng gói cũ;
  * `<thư mục gói>/import_manifest.json` ghi toàn bộ nhận dạng của gói; mọi nơi đọc từ đó qua mô-đun này;
  * `scripts/swap_web_models.py` dựng gói từ các run, sinh lại báo cáo rồi mới đổi con trỏ (đổi lại con trỏ là lùi bản).

Gói cũ (`artifacts/branch_demo/20261003`, nhập từ ba nhánh train) không có các trường mới trong manifest; giá trị mặc định của nó nằm ở `LEGACY_*` dưới đây
và CHỈ ở đây.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POINTER = "artifacts/active_bundle.json"
LEGACY_DIR = "artifacts/branch_demo/20261003"
BUNDLES_DIR = "artifacts/model_bundles"     # gói dựng từ các run (scripts/import_run_bundle.py)

LEGACY_PROTOCOL = "mixed_pv_pd_v3_pretrained_seed42_validation"
LEGACY_RELEASE_DIR = "data/dataset_sources/mixed/pv_pd_v3"
LEGACY_VALIDATION_COUNTS = {"plantdoc": 269, "plantvillage": 4344}
LEGACY_SHARD_TOTAL = 37238
LEGACY_HOLDOUT = {"manifest_sha256": "61ee9f178b93aa65dcc6e41c63365c0f6098dc6f79d3dfa7b6e3a8e735f769c2",
                  "counts": {"plantdoc": 172, "plantvillage": 10843}}
LEGACY_COMMUNITY_MODEL_ID = "demo_fedavg_v7"
LEGACY_MODELS: list[dict[str, str]] = [
    {"model_id": "centralized", "dir": "centralized", "file": "checkpoint_best.pt", "display_name": "Centralized v6", "phase": "GĐ1",
     "method": "Centralized v6", "registry_id": "demo_centralized_v6"},
    {"model_id": "fedavg", "dir": "fedavg", "file": "checkpoint_best.pt", "display_name": "FedAvg v7", "phase": "GĐ2",
     "method": "FedAvg v7", "registry_id": "demo_fedavg_v7"},
    *[{"model_id": f"local_{i:02d}", "dir": "local", "file": f"client_{i:02d}/checkpoint_best.pt", "display_name": f"Local {i:02d}",
       "phase": "GĐ1", "method": "Local-only", "registry_id": f"demo_local_{i:02d}"} for i in range(5)],
]


def active_bundle_dir() -> str:
    """Thư mục gói đang dùng, tương đối so với gốc repo. Có tệp con trỏ mà sai thì trả đúng giá trị trong đó (web báo lỗi rõ ràng) thay vì lặng lẽ về gói cũ."""
    pointer = ROOT / POINTER
    if not pointer.is_file():
        return LEGACY_DIR
    try:
        value = json.loads(pointer.read_text(encoding="utf-8")).get("bundle_dir")
    except (OSError, ValueError):
        return LEGACY_DIR
    return str(value).replace("\\", "/").strip("/") if value else LEGACY_DIR


def load_manifest(bundle_dir: str | Path | None = None) -> dict[str, Any]:
    """`import_manifest.json` của gói; trả {} nếu chưa có (gói chưa được nhập)."""
    root = Path(bundle_dir) if bundle_dir is not None else ROOT / active_bundle_dir()
    if not root.is_absolute():
        root = ROOT / root
    try:
        return json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def protocol_id(manifest: dict[str, Any]) -> str:
    return str(manifest.get("protocol_id") or LEGACY_PROTOCOL)


def release_dir(manifest: dict[str, Any]) -> str:
    return str(manifest.get("release_dir") or LEGACY_RELEASE_DIR)


def validation_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in (manifest.get("validation_counts") or LEGACY_VALIDATION_COUNTS).items()}


# Nhóm nguồn khi đo (khóa của validation_counts/holdout.counts): thứ tự trình bày, nhãn và loại ảnh. Nhóm lạ (gói sau này) lấy nhãn = tên nhóm.
SOURCE_PRESENTATION: dict[str, dict[str, str]] = {
    "plantvillage": {"label": "PlantVillage", "inline": "trong phòng", "short": "ảnh trong phòng", "kind": "lab"},
    "field_other": {"label": "Ngoài đồng, bộ khác", "inline": "ngoài đồng, bộ khác", "short": "ảnh ngoài đồng từ các bộ công khai khác", "kind": "field"},
    "plantdoc": {"label": "PlantDoc", "inline": "PlantDoc", "short": "ảnh ngoài đồng, nền phức tạp", "kind": "field"},
}


def source_groups(validation: dict[str, int], holdout_counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Các nhóm nguồn của gói theo thứ tự trình bày: id, nhãn, loại (lab/field), số ảnh val và số ảnh giữ riêng. Gói cũ chỉ có hai nhóm."""
    holdout_counts = holdout_counts or {}
    ordered = [s for s in SOURCE_PRESENTATION if s in validation] + [s for s in validation if s not in SOURCE_PRESENTATION]
    out = []
    for source in ordered:
        info = SOURCE_PRESENTATION.get(source, {"label": source, "inline": source, "short": source, "kind": "field"})
        out.append({"id": source, **info, "val_count": int(validation[source]), "holdout_count": int(holdout_counts.get(source, 0))})
    return out


def shard_total(manifest: dict[str, Any]) -> int:
    return int(manifest.get("shard_total") or LEGACY_SHARD_TOTAL)


def holdout(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("holdout") or LEGACY_HOLDOUT


def model_specs(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Danh sách mô hình của gói: model_id, dir (thư mục con), file (đường dẫn trong thư mục con), display_name, phase, method, registry_id."""
    specs = manifest.get("models") or LEGACY_MODELS
    out = []
    for s in specs:
        out.append({"model_id": s["model_id"], "dir": s["dir"], "file": s.get("file", "checkpoint_best.pt"),
                    "display_name": s["display_name"], "phase": s.get("phase", ""), "method": s.get("method", s["display_name"]),
                    "registry_id": s.get("registry_id", f"web_{s['model_id']}")})
    return out


def community_model_id(manifest: dict[str, Any]) -> str:
    """Mô hình FedAvg của gói, làm nền cho tính năng học cộng đồng (id trong registry)."""
    for s in model_specs(manifest):
        if s["model_id"] == "fedavg":
            return s["registry_id"]
    return LEGACY_COMMUNITY_MODEL_ID


def display_names(manifest: dict[str, Any]) -> dict[str, str]:
    """model_id -> tên hiển thị (ví dụ 'fedavg' -> 'FedAvg v4b')."""
    return {s["model_id"]: s["display_name"] for s in model_specs(manifest)}


def verify_comparison(path: Path, artifact_manifests: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Bảng `validation_comparison.csv` khớp gói: đủ mô hình × nguồn, đúng số ảnh, đúng SHA checkpoint/release/tập đánh giá, chỉ số trong [0, 1]."""
    import csv

    specs = {s["display_name"]: s for s in model_specs(manifest)}
    counts = validation_counts(manifest)
    release, validation = manifest["release_manifest_sha256"], manifest["validation_manifest_sha256"]
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(specs) * len(counts):
        raise ValueError(f"Expected {len(specs)} checkpoints × {len(counts)} validation sources")
    if {r["model"] for r in rows} != set(specs):
        raise ValueError("Unexpected model set in validation comparison")
    shas = {name: artifact_manifests[s["dir"]]["files"][s["file"]]["sha256"] for name, s in specs.items()}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["model"], row["source"])
        if key in seen or row["source"] not in counts:
            raise ValueError("Duplicate or unknown comparison row")
        seen.add(key)
        if int(row["count"]) != counts[row["source"]]:
            raise ValueError(f"Unexpected validation count: {key}")
        if (row["checkpoint_sha256"] != shas[row["model"]] or row["release_manifest_sha256"] != release
                or row["validation_manifest_sha256"] != validation):
            raise ValueError(f"Mismatched checkpoint or split: {key}")
        for metric in ("top1_accuracy", "macro_f1_supported", "crop_accuracy"):
            if not 0.0 <= float(row[metric]) <= 1.0:
                raise ValueError(f"Invalid {metric}: {key}")


def pinned_manifest_sha256(root: str | Path) -> str | None:
    """Mã băm `import_manifest.json` mà con trỏ `active_bundle.json` chốt cho thư mục gói `root`; None nếu con trỏ không chốt (gói cũ) hoặc trỏ gói khác.

    Chốt này thay cho các hằng số SHA từng viết cứng trong mã: sửa manifest (đổi release, tên mô hình, nhãn giao thức...) sau khi nhập gói
    thì web từ chối, vì mã băm không còn khớp con trỏ do `scripts/swap_web_models.py` ghi."""
    pointer = ROOT / POINTER
    try:
        data = json.loads(pointer.read_text(encoding="utf-8"))
        target = ROOT / str(data["bundle_dir"])
    except (OSError, ValueError, KeyError):
        return None
    if target.resolve() != Path(root).resolve():
        return None
    value = data.get("manifest_sha256")
    return str(value) if value else None


def registry_ids(manifest: dict[str, Any]) -> dict[str, str]:
    """model_id (centralized, fedavg, local_00...) -> id trong artifacts/registry.json; web, báo cáo và trạng thái đều tra qua bảng này."""
    return {s["model_id"]: s["registry_id"] for s in model_specs(manifest)}
