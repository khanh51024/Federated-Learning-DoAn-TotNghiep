"""Model Registry — nguồn sự thật DUY NHẤT cho mọi model/checkpoint/metric.

Nguyên tắc:
- File code KHÔNG tự hardcode đường dẫn checkpoint/metrics nữa; phải lấy hằng số tại đây.
- `artifacts/registry.json` do `scripts/build_registry.py` sinh ra; không bao giờ sửa tay.
- Đọc registry qua `load_registry()` / `get_model()` — fail-closed nếu registry chưa được sinh.

Xem hợp đồng đầy đủ trong docs/MODEL_REGISTRY.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# schema_version của artifacts/registry.json (v2: pipeline/role/dataset/taxonomy + cổng web + danh mục .pt)
SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Thư mục chính
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
CHECKPOINTS_DIR = "artifacts/checkpoints"
SMOKE_CHECKPOINTS_DIR = "artifacts/checkpoints/smoke"
METRICS_DIR = "artifacts/metrics"
DEMO_BUNDLE_DIR = "artifacts/branch_demo/20261003"
WEB_RUNS_DIR = "artifacts/web_runs"
VERIFIED_DIR = "artifacts/verified"
VERIFIED_NETWORK_DIR = "artifacts/verified_network"
DATA_PREFLIGHT_JSON = "artifacts/data_preflight.json"
CLIENT_PREFLIGHT_DIR = "artifacts/client_preflight"
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
PER_CLASS_REPORT_JSON = "artifacts/plantdoc_per_class_report.json"
TELEMETRY_HISTORY_JSON = "artifacts/metrics/telemetry_history.json"
NETWORK_FLOWER_HISTORY_JSON = "artifacts/metrics/network_flower_history.json"

# ---------------------------------------------------------------------------
# Canonical slots (chỉ được ghi bởi run khoa học thật, không bao giờ bởi smoke)
# ---------------------------------------------------------------------------
CENTRALIZED_BEST_PT = "artifacts/checkpoints/centralized_best.pt"
FEDAVG_BEST_PT = "artifacts/checkpoints/fedavg_best.pt"
FEDAVG_LAST_PT = "artifacts/checkpoints/fedavg_last.pt"
FEDERATED_NETWORK_GLOBAL_PT = "artifacts/checkpoints/federated_network_global.pt"
NETWORK_SMOKE_RUNS_DIR = "artifacts/web_runs/network_smoke"
ATTACK_MATRIX_OBSERVE_PT = "artifacts/checkpoints/attack_matrix_observe.pt"
ATTACK_MATRIX_REJECT_PT = "artifacts/checkpoints/attack_matrix_reject.pt"
NETWORK_RTT_CHECKPOINT_PT = "artifacts/checkpoints/network_rtt_checkpoint_test.pt"
PLANTVILLAGE_TRACED_PT = "artifacts/checkpoints/plantvillage_traced.pt"
# Ghim độc lập: trùng oid LFS ghi trong git (git lfs ls-files -l). Đổi file = phải sửa hằng số này có chủ đích.
PLANTVILLAGE_TRACED_SHA256 = "eab794cd9d9138f1e77d5dfc6166602da8a984b728c4228a23d7b84228cdf161"
DATASET_CLASSES_JSON = DATASET_MANIFEST_JSON  # nguồn thứ tự 38 lớp của model tham chiếu
SAMPLE_IMAGE = "artifacts/sample_test_leaf.jpg"  # ảnh dùng cho bài kiểm tra forward bắt buộc
INFERENCE_AUDIT_JSON = "artifacts/inference_audit.json"
LABELED_EVAL_JSON = "artifacts/labeled_inference_evaluation.json"
DEMO_RUNS_DIR = "artifacts/demo_runs"
DATA_PREFLIGHT_DEMO_GATE_JSON = "artifacts/data_preflight_demo_gate.json"

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

# Thư mục quét .pt để chắc chắn mọi checkpoint trên đĩa đều có tên trong danh mục
# Chỉ loại ở gốc repo (không loại thư mục trùng tên nằm sâu trong artifacts/...)
CHECKPOINT_SCAN_EXCLUDE_ROOT = (".venv", ".git", ".tmp-agent", ".zcode", "tmp")
# Loại ở mọi độ sâu (tên hoặc tiền tố)
CHECKPOINT_SCAN_EXCLUDE_ANY = ("__pycache__", "pytest-of-", "torchinductor_")
REPORT_EXPERIMENTS_GLOB = "artifacts/report_experiments"
WEB_RUNS_GLOB = "artifacts/web_runs"
KAGGLE_RESULTS_DIR = "results/fedavg"

# ---------------------------------------------------------------------------
# Vai trò (role) — trả lời "model này dùng vào việc gì"
# ---------------------------------------------------------------------------
ROLE_CURRENT = "current"              # kết quả hiện hành (bundle đã kiểm hash)
ROLE_HISTORICAL = "historical"        # kết quả lịch sử, chỉ còn metrics
ROLE_TECHNICAL_DEMO = "technical_demo"  # smoke kỹ thuật (Flower/mạng), không khoa học
ROLE_REFERENCE = "reference"
ROLE_QUARANTINED = "quarantined"      # smoke bị cách ly

PIPELINE_EXTERNAL = "external_branch"  # train ngoài repo, import bundle
PIPELINE_LEGACY = "legacy_kaggle"      # pipeline Kaggle cũ (code ở legacy/)
PIPELINE_SRC = "src"                   # engine trong repo (Flower demo, simulation)

# Backend suy luận mà web/services/model_service.py biết cách gọi
BACKEND_PUBLISHED_BUNDLE = "published_bundle"        # web/demo_evidence.predict_published (kiểm hash bundle)
BACKEND_TORCHSCRIPT_REFERENCE = "torchscript_reference"  # web/plantvillage_reference.predict_plantvillage

KNOWN_PREPROCESSING = frozenset({"imagenet_eval_v1", "canonical_v1", "get_default_transform(eval)"})

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
    if registry.get("schema_version") != SCHEMA_VERSION:
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
    """Các model được phép hiển thị trên web (đã qua đủ 5 điều kiện của cổng registry)."""
    return [m for m in iter_models() if m.get("web_visible") is True]


def code_fingerprint() -> str:
    """Dấu vân tay mã nguồn (src/web/scripts/tests) đọc thẳng từ đĩa, chuẩn hóa xuống dòng.

    Dùng để biết một kết quả gate (vd. final_acceptance) có còn đúng với mã hiện tại không,
    mà không phụ thuộc commit (commit đổi mỗi khi chính file trạng thái được commit)."""
    digest = hashlib.sha256()
    for top in ("src", "web", "scripts", "tests"):
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".py", ".js", ".html", ".css"}
                and "__pycache__" not in path.parts
            ):
                digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
                digest.update(b"\x00")
                digest.update(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()
