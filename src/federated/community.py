"""Lõi học liên kết cộng đồng: chỉ huấn luyện lớp cuối (`classifier.3`) trên đặc trưng 1024 chiều của backbone đóng băng.

Mô-đun này thuần số học (numpy), không biết gì về HTTP hay SQLite. Máy chủ web (web/services/community_store.py) gọi vào đây:

- mã hóa/giải mã head trên đường truyền (float32 little-endian: ma trận trọng số 38×1024 theo hàng, rồi 38 bias);
- kiểm một bản cập nhật (khóa, shape, dtype, hữu hạn, hash head cha) bằng `validate_client_update` có sẵn;
- cắt chuẩn delta ở ngưỡng C phía máy chủ (máy chủ không tin chuẩn do client tự báo);
- gộp FedAvg bằng `aggregate_fedavg` có sẵn (trọng số theo số ảnh, có trần);
- chấm head ứng viên trên tập đặc trưng giám sát và quyết định chấp nhận (cổng không-thoái-lui).

Tập giám sát (đặc trưng của ảnh validation) cũng đã được dùng để chọn mô hình gốc, nên con số của nó chỉ là tín hiệu giám sát
của hệ thống, KHÔNG phải số liệu khoa học và không được trộn với số công bố của bundle.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.federated.strategy import aggregate_fedavg
from src.federated.update_validator import validate_client_update
from src.model.mobilenetv3 import compute_model_hash

NUM_CLASSES = 38
FEATURE_DIM = 1024
WEIGHT_KEY = "classifier.3.weight"
BIAS_KEY = "classifier.3.bias"
HEAD_PARAMS = NUM_CLASSES * FEATURE_DIM + NUM_CLASSES  # 38.950
HEAD_BYTES = HEAD_PARAMS * 4  # 155.800 byte float32
SOURCE_PLANTVILLAGE = 0
SOURCE_PLANTDOC = 1

Head = Dict[str, np.ndarray]


class CommunityError(ValueError):
    """Lỗi do dữ liệu từ client: `code` ổn định để UI/test dựa vào, `message` đọc được bằng tiếng Việt."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Chính sách (mọi ngưỡng ở một chỗ; máy chủ cho quản trị đổi trong giới hạn của `POLICY_BOUNDS`)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    k_target: int = 3            # đủ ngần này cập nhật hợp lệ là gộp ngay
    k_min: int = 2               # hết hạn mà có ít nhất ngần này thì vẫn gộp
    round_ttl_seconds: int = 900  # hạn của một vòng (gia hạn thêm nếu chưa đủ k_min)
    clip_norm: float = 1.0       # trần chuẩn L2 của delta (head mới − head cha)
    n_min: int = 5               # ít hơn ngần này ảnh: từ chối
    n_cap: int = 200             # trọng số FedAvg tối đa của một client (chặn một người áp đảo)
    epsilon: float = 0.005       # head mới được nhận nếu độ chính xác giám sát không giảm quá ngần này
    epsilon_macro: float = 0.005  # ... và macro-recall không giảm quá ngần này
    epsilon_plantdoc: float = 0.02  # ... và độ chính xác ảnh PlantDoc (chỉ 269 ảnh, nhiễu hơn) không giảm quá ngần này
    # Ngân sách TÍCH LŨY: ngưỡng epsilon chỉ so với head cha nên nhiều vòng nhỏ có thể trôi dần xuống. Ứng viên còn phải không
    # thấp hơn mức tốt nhất từng đạt (kể cả head gốc) quá ngần này.
    budget: float = 0.01
    budget_macro: float = 0.01
    budget_plantdoc: float = 0.03
    local_epochs: int = 3        # cấu hình huấn luyện cục bộ khuyến nghị cho client (đo ở bước nghiên cứu)
    local_lr: float = 0.01
    local_batch: int = 16
    holdout_fraction: float = 0.2
    dp_enabled: bool = False     # khuyến nghị client cắt chuẩn + thêm nhiễu Gaussian (máy chủ không ép được)
    dp_clip: float = 1.0
    dp_noise_multiplier: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


POLICY_BOUNDS: Dict[str, tuple] = {
    # k >= 2: một người một mình không được trở thành mô hình chung (dù đã cắt chuẩn và qua cổng)
    "k_target": (2, 1000), "k_min": (2, 1000), "round_ttl_seconds": (30, 7 * 86400),
    "clip_norm": (0.01, 100.0), "n_min": (1, 1000), "n_cap": (1, 100000),
    "epsilon": (0.0, 0.2), "epsilon_macro": (0.0, 0.2), "epsilon_plantdoc": (0.0, 0.5),
    "budget": (0.0, 0.5), "budget_macro": (0.0, 0.5), "budget_plantdoc": (0.0, 0.5),
    "local_epochs": (1, 20), "local_lr": (1e-4, 1.0), "local_batch": (1, 256), "holdout_fraction": (0.0, 0.5),
    "dp_clip": (0.01, 100.0), "dp_noise_multiplier": (0.0, 10.0),
}


def policy_from(values: Optional[Mapping[str, Any]]) -> Policy:
    """Policy mặc định, ghi đè bằng `values` (đã kiểm kiểu và giới hạn). Khóa lạ bị bỏ qua."""
    base = Policy()
    if not values:
        return base
    fixed: Dict[str, Any] = {}
    for name, default in base.to_dict().items():
        if name not in values:
            continue
        raw = values[name]
        if isinstance(default, bool):
            if not isinstance(raw, bool):
                raise CommunityError("bad_setting", f"{name}: phải là true/false.")
            fixed[name] = raw
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw):
            raise CommunityError("bad_setting", f"{name}: phải là một số hợp lệ.")
        low, high = POLICY_BOUNDS[name]
        if not low <= raw <= high:
            raise CommunityError("bad_setting", f"{name}: nằm ngoài khoảng cho phép [{low}, {high}].")
        fixed[name] = int(raw) if isinstance(default, int) else float(raw)
    merged = Policy(**{**base.to_dict(), **fixed})
    if merged.k_min > merged.k_target:
        raise CommunityError("bad_setting", "k_min không được lớn hơn k_target.")
    if merged.n_min > merged.n_cap:
        raise CommunityError("bad_setting", "n_min không được lớn hơn n_cap.")
    return merged


# ---------------------------------------------------------------------------
# Mã hóa head
# ---------------------------------------------------------------------------
def as_head(weight: np.ndarray, bias: np.ndarray) -> Head:
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    bias = np.ascontiguousarray(bias, dtype=np.float32)
    if weight.shape != (NUM_CLASSES, FEATURE_DIM) or bias.shape != (NUM_CLASSES,):
        raise CommunityError("bad_shape", f"Head phải có weight {NUM_CLASSES}×{FEATURE_DIM} và bias {NUM_CLASSES}.")
    return {WEIGHT_KEY: weight, BIAS_KEY: bias}


def head_to_bytes(head: Head) -> bytes:
    return head[WEIGHT_KEY].astype("<f4", copy=False).tobytes() + head[BIAS_KEY].astype("<f4", copy=False).tobytes()


def head_from_bytes(raw: bytes) -> Head:
    if len(raw) != HEAD_BYTES:
        raise CommunityError("bad_size", f"Head phải đúng {HEAD_BYTES} byte (nhận {len(raw)}).")
    flat = np.frombuffer(raw, dtype="<f4").astype(np.float32)  # sao chép: mảng không còn trỏ vào bộ đệm yêu cầu
    return as_head(flat[: NUM_CLASSES * FEATURE_DIM].reshape(NUM_CLASSES, FEATURE_DIM), flat[NUM_CLASSES * FEATURE_DIM:])


def encode_head(head: Head) -> str:
    return base64.b64encode(head_to_bytes(head)).decode("ascii")


def decode_head(text: str) -> Head:
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError, AttributeError) as exc:
        raise CommunityError("bad_encoding", "Trọng số không phải base64 hợp lệ.") from exc
    return head_from_bytes(raw)


def head_sha(head: Head) -> str:
    """Hash tất định của head: dùng lại `compute_model_hash` của engine (tên khóa + byte float32)."""
    return compute_model_hash({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in head.items()})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Kiểm, cắt chuẩn, gộp
# ---------------------------------------------------------------------------
@dataclass
class UpdateCheck:
    head: Head                 # head đã cắt chuẩn (sẽ đem gộp)
    norm_raw: float            # ||head_client − head_cha|| trước khi cắt
    norm_used: float           # sau khi cắt
    clipped: bool
    suspicious: bool
    reasons: List[str] = field(default_factory=list)


def check_update(parent: Head, child: Head, reported_parent_sha: str, policy: Policy) -> UpdateCheck:
    """Kiểm một cập nhật so với head cha của vòng; ném CommunityError nếu phải từ chối.

    `validate_client_update` đã kiểm khóa/shape/dtype/hữu hạn/hash cha và tự tính chuẩn delta từ payload nhận được."""
    result = validate_client_update(parent, child, reported_parent_hash=reported_parent_sha, max_update_norm=None)
    if not result.structure_ok:
        raise CommunityError("bad_structure", f"Cấu trúc trọng số không hợp lệ ({result.reason}).")
    if not result.finite_ok:
        raise CommunityError("non_finite", "Trọng số chứa NaN hoặc vô cực.")
    if not result.parent_hash_ok:
        raise CommunityError("stale_parent", "Trọng số được huấn luyện từ một phiên bản mô hình khác với vòng hiện tại.")
    norm = float(result.received_update_norm)
    if not np.isfinite(norm):  # tổng bình phương float32 tràn: giá trị lớn vô lý, từ chối thay vì để NaN lọt xuống CSDL
        raise CommunityError("non_finite", "Độ lớn cập nhật vượt giới hạn số học (quá lớn).")
    if norm <= 1e-9:
        raise CommunityError("no_change", "Cập nhật không thay đổi mô hình so với phiên bản gốc.")
    scale = min(1.0, policy.clip_norm / norm)
    if scale < 1.0:
        child = {
            key: (parent[key].astype(np.float64) + (child[key].astype(np.float64) - parent[key].astype(np.float64)) * scale
                  ).astype(np.float32)
            for key in parent
        }
    return UpdateCheck(
        head={k: np.ascontiguousarray(v, dtype=np.float32) for k, v in child.items()},
        norm_raw=norm, norm_used=norm * scale, clipped=scale < 1.0, suspicious=result.suspicious,
    )


def aggregate_heads(updates: Sequence[Mapping[str, Any]], parent: Head, policy: Policy) -> Head:
    """FedAvg có trọng số theo số ảnh (có trần `n_cap`) trên hai tensor của head. `updates`: {head, num_examples}."""
    if not updates:
        raise CommunityError("no_updates", "Không có cập nhật nào để gộp.")
    packed = [
        {"weights": item["head"], "num_examples": int(min(max(int(item["num_examples"]), 1), policy.n_cap)), "client_id": index}
        for index, item in enumerate(updates)
    ]
    merged = aggregate_fedavg(packed, parent_weights=parent)
    return {key: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32) for key, tensor in merged.items()}


# ---------------------------------------------------------------------------
# Chấm điểm và cổng
# ---------------------------------------------------------------------------
@dataclass
class EvalSet:
    features: np.ndarray  # (N, 1024) float32
    labels: np.ndarray    # (N,) int64
    sources: np.ndarray   # (N,) int8: 0 = PlantVillage, 1 = PlantDoc
    sha256: str = ""

    def __post_init__(self) -> None:
        self.features = np.ascontiguousarray(self.features, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.sources = np.asarray(self.sources, dtype=np.int8)
        if self.features.ndim != 2 or self.features.shape[1] != FEATURE_DIM:
            raise ValueError(f"Đặc trưng giám sát phải có {FEATURE_DIM} chiều.")
        if not (len(self.features) == len(self.labels) == len(self.sources)):
            raise ValueError("Đặc trưng, nhãn và nguồn của tập giám sát không cùng độ dài.")
        if len(self.labels) and (self.labels.min() < 0 or self.labels.max() >= NUM_CLASSES):
            raise ValueError("Nhãn của tập giám sát nằm ngoài 38 lớp.")


def load_eval_set(path: Path, expected_sha256: Optional[str] = None) -> EvalSet:
    sha = file_sha256(path)
    if expected_sha256 and sha != expected_sha256:
        raise ValueError(f"{path.name}: SHA-256 {sha[:12]}… khác giá trị ghim {expected_sha256[:12]}…")
    with np.load(path, allow_pickle=False) as data:
        return EvalSet(data["features"].astype(np.float32), data["labels"], data["sources"], sha)


def _accuracy(pred: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
    return float((pred == labels).mean()) if len(labels) else None


def predict_labels(head: Head, features: np.ndarray) -> np.ndarray:
    return (features @ head[WEIGHT_KEY].T + head[BIAS_KEY]).argmax(axis=1)


def evaluate_head(head: Head, evalset: EvalSet) -> Dict[str, Any]:
    pred = predict_labels(head, evalset.features)
    recalls = [float((pred[evalset.labels == k] == k).mean()) for k in range(NUM_CLASSES) if (evalset.labels == k).any()]
    return {
        "count": int(len(evalset.labels)),
        "accuracy": _accuracy(pred, evalset.labels),
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "plantvillage_accuracy": _accuracy(pred, evalset.labels, evalset.sources == SOURCE_PLANTVILLAGE),
        "plantdoc_accuracy": _accuracy(pred, evalset.labels, evalset.sources == SOURCE_PLANTDOC),
        "plantvillage_count": int((evalset.sources == SOURCE_PLANTVILLAGE).sum()),
        "plantdoc_count": int((evalset.sources == SOURCE_PLANTDOC).sum()),
    }


GATE_MESSAGES = {
    "accuracy_drop": "độ chính xác trên tập giám sát giảm quá ngưỡng cho phép",
    "macro_recall_drop": "macro-recall (trung bình theo nhóm) giảm quá ngưỡng cho phép",
    "plantdoc_drop": "độ chính xác trên ảnh ngoài đồng (PlantDoc) giảm quá ngưỡng cho phép",
    "accuracy_drift": "độ chính xác thấp hơn mức tốt nhất từng đạt quá ngân sách cho phép",
    "macro_recall_drift": "macro-recall thấp hơn mức tốt nhất từng đạt quá ngân sách cho phép",
    "plantdoc_drift": "độ chính xác ảnh ngoài đồng thấp hơn mức tốt nhất từng đạt quá ngân sách cho phép",
}


def best_of(history: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Mức tốt nhất từng đạt của mỗi chỉ số trên các phiên bản đã được nhận (kể cả head gốc)."""
    best: Dict[str, float] = {}
    for metrics in history:
        for name in ("accuracy", "macro_recall", "plantdoc_accuracy"):
            value = metrics.get(name)
            if value is not None and value > best.get(name, -1.0):
                best[name] = float(value)
    return best


def gate_decision(
    candidate: Mapping[str, Any], parent: Mapping[str, Any], policy: Policy, best: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, List[str]]:
    """Cổng không-thoái-lui: ứng viên bị loại nếu một chỉ số tụt quá ngưỡng so với head cha (từng vòng) hoặc so với mức tốt nhất
    từng đạt (tích lũy, chặn trôi dần qua nhiều vòng nhỏ)."""
    reasons: List[str] = []

    def below(name: str, reference: Optional[Mapping[str, Any]], allowance: float) -> bool:
        new = candidate.get(name)
        old = None if reference is None else reference.get(name)
        if old is None:
            return False
        if new is None:  # chỉ số có ở mức tham chiếu mà ứng viên không tính được: coi là thoái lui (thất bại-đóng)
            return True
        return new < old - allowance - 1e-12

    for name, eps, budget, step_reason, drift_reason in (
        ("accuracy", policy.epsilon, policy.budget, "accuracy_drop", "accuracy_drift"),
        ("macro_recall", policy.epsilon_macro, policy.budget_macro, "macro_recall_drop", "macro_recall_drift"),
        ("plantdoc_accuracy", policy.epsilon_plantdoc, policy.budget_plantdoc, "plantdoc_drop", "plantdoc_drift"),
    ):
        if below(name, parent, eps):
            reasons.append(step_reason)
        elif below(name, best, budget):
            reasons.append(drift_reason)
    return (not reasons), reasons


def label_counts_ok(counts: Sequence[int], num_examples: int) -> None:
    if len(counts) != NUM_CLASSES:
        raise CommunityError("bad_labels", f"Phân bố nhãn phải có {NUM_CLASSES} phần tử.")
    if any((not isinstance(c, int)) or c < 0 for c in counts):
        raise CommunityError("bad_labels", "Phân bố nhãn chỉ gồm số nguyên không âm.")
    if sum(counts) != num_examples:
        raise CommunityError("bad_labels", "Tổng phân bố nhãn không bằng số ảnh đã dùng để huấn luyện.")
