"""Model Registry — nguồn sự thật DUY NHẤT cho mọi model/checkpoint/metric.

Nguyên tắc:
- File code KHÔNG tự hardcode đường dẫn checkpoint/metrics nữa; phải lấy hằng số tại đây.
- `artifacts/registry.json` do `scripts/build_registry.py` sinh ra; không bao giờ sửa tay.
- Đọc registry qua `load_registry()` / `get_model()` — fail-closed nếu registry chưa được sinh.

Xem hợp đồng đầy đủ trong docs/MODEL_REGISTRY.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Thư mục chính
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
CHECKPOINTS_DIR = "artifacts/checkpoints"
SMOKE_CHECKPOINTS_DIR = "artifacts/checkpoints/smoke"
METRICS_DIR = "artifacts/metrics"
DEMO_BUNDLE_DIR = "artifacts/branch_demo/20261003"
WEB_RUNS_DIR = "artifacts/web_runs"
REPORT_EXPERIMENTS_DIR = "artifacts/report_experiments"

# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------
PARTITIONS_DIR = "data/partitions_stage2_scratch_v3"
SUITE_JSON = "data/partitions_stage2_scratch_v3/suite.json"
DATASET_MANIFEST_JSON = "data/dataset_manifest.json"

# ---------------------------------------------------------------------------
# Trạng thái / gates (tự sinh bởi script tương ứng)
# ---------------------------------------------------------------------------
REGISTRY_JSON = "artifacts/registry.json"
CANONICAL_MANIFEST_JSON = "artifacts/canonical_manifest.json"
PROVENANCE_VALIDATION_JSON = "artifacts/artifact_provenance_validation.json"
NETWORK_ARTIFACT_VALIDATION_JSON = "artifacts/network_artifact_validation.json"
PRE_DEMO_CHECK_JSON = "artifacts/pre_demo_check.json"
FINAL_ACCEPTANCE_JSON = "artifacts/final_acceptance.json"
STATUS_REPORT_JSON = "artifacts/status_report.json"
DOMAIN_GAP_REPORT_JSON = "artifacts/plantdoc_domain_gap_report.json"
TELEMETRY_HISTORY_JSON = "artifacts/metrics/telemetry_history.json"
NETWORK_FLOWER_HISTORY_JSON = "artifacts/metrics/network_flower_history.json"

# ---------------------------------------------------------------------------
# Canonical slots (chỉ được ghi bởi run khoa học thật, không bao giờ bởi smoke)
# ---------------------------------------------------------------------------
CENTRALIZED_BEST_PT = "artifacts/checkpoints/centralized_best.pt"
FEDAVG_BEST_PT = "artifacts/checkpoints/fedavg_best.pt"
FEDAVG_LAST_PT = "artifacts/checkpoints/fedavg_last.pt"
FEDERATED_NETWORK_GLOBAL_PT = "artifacts/checkpoints/federated_network_global.pt"
PLANTVILLAGE_TRACED_PT = "artifacts/checkpoints/plantvillage_traced.pt"

# Metrics chuẩn (họ scientific_canonical)
CENTRALIZED_RESULTS_JSON = "artifacts/metrics/centralized_results.json"
FEDAVG_RESULTS_JSON = "artifacts/metrics/fedavg_results.json"
LOCAL_ONLY_RESULTS_JSON = "artifacts/metrics/local_only_results.json"

# Run canonical Kaggle (metrics thật; checkpoint KHÔNG có trên máy này)
CANONICAL_RUN_ID = "20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62"
CANONICAL_RUN_DIR = f"results/fedavg/{CANONICAL_RUN_ID}"
CANONICAL_TEST_METRICS_JSON = (
    f"results/fedavg/{CANONICAL_RUN_ID}/evaluations/b0cddc167068f46403c13595/test_metrics.json"
)
CANONICAL_RESOLVED_CONFIG = f"results/fedavg/{CANONICAL_RUN_ID}/resolved_config.yaml"

# Published demo bundle (hash-verified, pretrained mixed PV+PlantDoc)
DEMO_IMPORT_MANIFEST_JSON = f"{DEMO_BUNDLE_DIR}/import_manifest.json"
DEMO_PROTOCOL_ID = "mixed_pv_pd_v3_pretrained_seed42_validation"

# ---------------------------------------------------------------------------
# Family & status semantics (bảng tra cho build_registry + UI)
# ---------------------------------------------------------------------------
FAMILY_SCIENTIFIC_CANONICAL = "scientific_canonical"
FAMILY_PUBLISHED_DEMO = "published_demo"
FAMILY_NETWORK_DEMO = "network_demo"
FAMILY_REFERENCE = "reference"
FAMILY_SMOKE = "smoke"

FAMILY_DESCRIPTIONS = {
    FAMILY_SCIENTIFIC_CANONICAL: (
        "Kết quả khoa học canonical (scratch, PlantVillage thuần) từ run Kaggle "
        f"{CANONICAL_RUN_ID}. Chỉ số từ metrics JSON của run; KHÔNG so sánh với họ published_demo."
    ),
    FAMILY_PUBLISHED_DEMO: (
        "Checkpoint pretrained mixed PlantVillage+PlantDoc (protocol "
        f"{DEMO_PROTOCOL_ID}), hash-verified khi import. Dùng cho demo web/inference, "
        "KHÔNG phải kết quả khoa học của protocol GĐ2 scratch."
    ),
    FAMILY_NETWORK_DEMO: (
        "Checkpoint từ các bài test mạng Flower (RTT, attack matrix, smoke network). "
        "Chỉ dùng demo kỹ thuật, cấm dùng làm chỉ số khoa học."
    ),
    FAMILY_REFERENCE: "Model tham chiếu (TorchScript), không phải checkpoint FedAvg.",
    FAMILY_SMOKE: (
        "Checkpoint synthetic smoke (accuracy 1.0 / 0.0263 do dữ liệu dummy). "
        "CẤM dùng trong luận văn; đã cách ly vào artifacts/checkpoints/smoke/."
    ),
}


def abs_path(relative: str) -> Path:
    """Chuyển đường dẫn tương đối (POSIX) trong registry thành đường dẫn tuyệt đối."""
    return ROOT / relative


def load_registry() -> dict[str, Any]:
    """Đọc artifacts/registry.json. Fail-closed nếu chưa sinh."""
    path = abs_path(REGISTRY_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {REGISTRY_JSON}. Chạy: python scripts/build_registry.py "
            "(hoặc python scripts/update_status.py) để sinh registry."
        )
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError(
            f"{REGISTRY_JSON} có schema_version={registry.get('schema_version')!r} "
            "không tương thích — chạy lại scripts/build_registry.py."
        )
    return registry


def get_model(model_id: str) -> dict[str, Any]:
    """Lấy một entry model theo model_id. KeyError nếu không có."""
    registry = load_registry()
    for model in registry.get("models", []):
        if model.get("model_id") == model_id:
            return model
    known = [m.get("model_id") for m in registry.get("models", [])]
    raise KeyError(f"model_id={model_id!r} không có trong registry. Đang có: {known}")


def iter_models(family: str | None = None) -> list[dict[str, Any]]:
    """Danh sách model, lọc theo family nếu có."""
    registry = load_registry()
    models = registry.get("models", [])
    if family is None:
        return models
    return [m for m in models if m.get("family") == family]


def web_visible_models() -> list[dict[str, Any]]:
    """Các model được phép hiển thị trên web (đã qua gate của registry)."""
    return [m for m in iter_models() if m.get("web_visible") is True]
