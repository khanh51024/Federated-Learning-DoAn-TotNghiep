"""Baseline registry chỉ đọc (read-only) cho kết quả lịch sử GĐ1.

Không huấn luyện lại Centralized hoặc Local-only.
Mặc định reference_only=True và strict_comparison_eligible=False do:
- Thiếu fingerprint ảnh lịch sử và checkpoint client Local-only.
- Quy tắc chọn checkpoint khác nhau: Centralized/FedAvg chọn best validation accuracy, Local-only chọn final epoch.
"""

import hashlib
from stage1_compat.integrity import read_json
from stage1_compat.constants import UPSTREAM_MANIFEST_PATH
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stage1_compat.constants import (
    SEED,
    UPSTREAM_COMMIT,
    UPSTREAM_DIR,
    UPSTREAM_REPO,
)


@dataclass
class BaselineEntry:
    method: str  # centralized, local_only, fedavg_historical
    seed: int
    alpha: float | None
    client_count: int
    split_info: dict[str, int]
    selection_policy: str
    metric_units: str  # float [0.0, 1.0]
    metrics: dict[str, Any]
    source_file: str
    source_file_sha256: str
    upstream_commit: str
    reference_only: bool = True
    strict_comparison_eligible: bool = False
    verification_level: str = "legacy_protocol_reference"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineRegistry:
    """Registry chứa toàn bộ baselines lịch sử đã xác minh từ snapshot upstream."""

    def __init__(self, upstream_results_dir: Path | None = None):
        self.results_dir = upstream_results_dir or (UPSTREAM_DIR / "experiments" / "results")
        self._entries: dict[str, BaselineEntry] = {}
        self._load_baselines()

    def _file_sha256(self, path: Path) -> str:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = read_json(UPSTREAM_MANIFEST_PATH)["files"].get("experiments/results/" + path.name)
        if expected != actual:
            raise ValueError(f"Historical baseline hash mismatch: {path.name}")
        return actual

    def _load_baselines(self) -> None:
        # 1. Centralized seed 42
        cent_path = self.results_dir / "centralized_seed42_metrics.json"
        if not cent_path.exists():
            cent_path = self.results_dir / "centralized_metrics.json"

        if cent_path.exists():
            cent_raw = json.loads(cent_path.read_text(encoding="utf-8"))
            cent_sha = self._file_sha256(cent_path)
            entry = BaselineEntry(
                method="centralized",
                seed=cent_raw.get("seed", 42),
                alpha=None,  # Centralized train trên toàn bộ tập train union, không phụ thuộc alpha
                client_count=1,
                split_info=cent_raw.get("split", {"train": 39084, "validation": 4328, "test": 10893}),
                selection_policy="best_validation_accuracy",
                metric_units="ratio [0.0, 1.0]",
                metrics={
                    "test_accuracy": cent_raw["test_metrics"]["accuracy"],
                    "test_macro_f1": cent_raw["test_metrics"]["macro_f1"],
                    "test_loss": cent_raw["test_metrics"]["loss"],
                    "best_epoch": cent_raw.get("best_epoch", 8),
                    "best_validation_accuracy": cent_raw.get("best_validation_accuracy"),
                },
                source_file=cent_path.name,
                source_file_sha256=cent_sha,
                upstream_commit=UPSTREAM_COMMIT,
                reference_only=True,
                strict_comparison_eligible=False,
                verification_level="legacy_protocol_reference",
                notes="Centralized upper bound tham chiếu. Train union/transforms không phụ thuộc client.",
            )
            self._entries["centralized_seed42"] = entry

        # 2. Local-only alpha=1.0 seed=42
        loc1_path = self.results_dir / "local_only_alpha1_seed42.json"
        if not loc1_path.exists():
            loc1_path = self.results_dir / "local_only_metrics.json"

        if loc1_path.exists():
            loc1_raw = json.loads(loc1_path.read_text(encoding="utf-8"))
            loc1_sha = self._file_sha256(loc1_path)
            entry = BaselineEntry(
                method="local_only",
                seed=loc1_raw.get("seed", 42),
                alpha=1.0,
                client_count=len(loc1_raw.get("clients", [])) or 5,
                split_info={"train": 39084, "validation": 4328, "test": 10893},
                selection_policy="final_epoch",
                metric_units="ratio [0.0, 1.0]",
                metrics={
                    "accuracy_mean": loc1_raw["average_accuracy"],
                    "accuracy_std": loc1_raw["accuracy_std"],
                    "accuracy_min": loc1_raw["accuracy_min"],
                    "accuracy_max": loc1_raw["accuracy_max"],
                    "macro_f1_mean": loc1_raw["average_macro_f1"],
                    "macro_f1_std": loc1_raw["macro_f1_std"],
                    "macro_f1_min": loc1_raw["macro_f1_min"],
                    "macro_f1_max": loc1_raw["macro_f1_max"],
                    "client_results": loc1_raw.get("clients", []),
                },
                source_file=loc1_path.name,
                source_file_sha256=loc1_sha,
                upstream_commit=UPSTREAM_COMMIT,
                reference_only=True,
                strict_comparison_eligible=False,
                verification_level="legacy_protocol_reference",
                notes="Local-only lower bound alpha=1. Đánh giá test tại epoch cuối (epoch 10), không chọn theo validation.",
            )
            self._entries["local_only_alpha1_seed42"] = entry

        # 3. Local-only alpha=100.0 seed=42
        loc100_path = self.results_dir / "local_only_alpha100_seed42.json"
        if loc100_path.exists():
            loc100_raw = json.loads(loc100_path.read_text(encoding="utf-8"))
            loc100_sha = self._file_sha256(loc100_path)
            entry = BaselineEntry(
                method="local_only",
                seed=loc100_raw.get("seed", 42),
                alpha=100.0,
                client_count=len(loc100_raw.get("clients", [])) or 5,
                split_info={"train": 39084, "validation": 4328, "test": 10893},
                selection_policy="final_epoch",
                metric_units="ratio [0.0, 1.0]",
                metrics={
                    "accuracy_mean": loc100_raw["average_accuracy"],
                    "accuracy_std": loc100_raw["accuracy_std"],
                    "accuracy_min": loc100_raw["accuracy_min"],
                    "accuracy_max": loc100_raw["accuracy_max"],
                    "macro_f1_mean": loc100_raw["average_macro_f1"],
                    "macro_f1_std": loc100_raw["macro_f1_std"],
                    "macro_f1_min": loc100_raw["macro_f1_min"],
                    "macro_f1_max": loc100_raw["macro_f1_max"],
                    "client_results": loc100_raw.get("clients", []),
                },
                source_file=loc100_path.name,
                source_file_sha256=loc100_sha,
                upstream_commit=UPSTREAM_COMMIT,
                reference_only=True,
                strict_comparison_eligible=False,
                verification_level="legacy_protocol_reference",
                notes="Local-only lower bound alpha=100. Đánh giá test tại epoch cuối (epoch 10), không chọn theo validation.",
            )
            self._entries["local_only_alpha100_seed42"] = entry

        # 4. Historical FedAvg alpha=1.0 seed=42 (Chỉ để đối chiếu tái lập, KHÔNG thay thế FedAvg mới)
        fed1_path = self.results_dir / "fedavg_alpha1_seed42.json"
        if not fed1_path.exists():
            fed1_path = self.results_dir / "fedavg_metrics.json"

        if fed1_path.exists():
            fed1_raw = json.loads(fed1_path.read_text(encoding="utf-8"))
            fed1_sha = self._file_sha256(fed1_path)
            entry = BaselineEntry(
                method="fedavg_historical",
                seed=fed1_raw.get("seed", 42),
                alpha=1.0,
                client_count=5,
                split_info=fed1_raw.get("split", {"train": 39084, "validation": 4328, "test": 10893}),
                selection_policy="best_validation_accuracy",
                metric_units="ratio [0.0, 1.0]",
                metrics={
                    "test_accuracy": fed1_raw["test_metrics"]["accuracy"],
                    "test_macro_f1": fed1_raw["test_metrics"]["macro_f1"],
                    "test_loss": fed1_raw["test_metrics"]["loss"],
                    "best_round": fed1_raw.get("best_round", 9),
                    "best_validation_accuracy": fed1_raw.get("best_validation_accuracy"),
                },
                source_file=fed1_path.name,
                source_file_sha256=fed1_sha,
                upstream_commit=UPSTREAM_COMMIT,
                reference_only=True,
                strict_comparison_eligible=False,
                verification_level="legacy_protocol_reference",
                notes="Kết quả FedAvg alpha=1 lịch sử GĐ1 để kiểm tra độ tương thích/tái lập.",
            )
            self._entries["fedavg_historical_alpha1_seed42"] = entry

    def get_centralized(self, seed: int = SEED) -> BaselineEntry | None:
        key = f"centralized_seed{seed}"
        return self._entries.get(key)

    def get_local_only(self, alpha: float, seed: int = SEED) -> BaselineEntry | None:
        alpha_int = int(alpha) if float(alpha).is_integer() else str(alpha).replace(".", "_")
        key = f"local_only_alpha{alpha_int}_seed{seed}"
        return self._entries.get(key)

    def get_historical_fedavg(self, alpha: float, seed: int = SEED) -> BaselineEntry | None:
        alpha_int = int(alpha) if float(alpha).is_integer() else str(alpha).replace(".", "_")
        key = f"fedavg_historical_alpha{alpha_int}_seed{seed}"
        return self._entries.get(key)

    def list_all(self) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._entries.items()}
