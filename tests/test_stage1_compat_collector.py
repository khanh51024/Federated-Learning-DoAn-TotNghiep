"""Unit tests cho Gói 3: Collector, so sánh reference gap, xử lý baseline thiếu và export artifacts."""

import json
from pathlib import Path

from stage1_compat.collector import run_stage1_collect
import pytest
import torch
from tests.test_stage1_compat_runner import actual_runner_fixture
from stage1_compat.budget import BudgetLedger
from stage1_compat.integrity import file_hash, atomic_json
from stage1_compat.checkpoint import atomic_torch_save


def test_collector_exports_all_artifacts(tmp_path):
    res = run_stage1_collect(output_dir=tmp_path)
    assert res["status"] == "SUCCESS"

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "comparison.csv").exists()
    assert (reports_dir / "comparison.json").exists()
    assert (reports_dir / "comparison.md").exists()
    assert (reports_dir / "collection_status.json").exists()

    status_data = json.loads((reports_dir / "collection_status.json").read_text(encoding="utf-8"))
    assert status_data["strict_comparison_eligible"] is False
    assert status_data["scientific_stage2_complete"] is False
    assert status_data["total_conditions"] == 2


def test_collector_gap_and_strict_null_preservation(tmp_path, actual_runner_fixture):
    runner, _, train, val, partitions, job = actual_runner_fixture
    job.job_id = "fedavg_alpha1_seed42"
    metrics = runner.run_single_fedavg_job(job, train, val, val, [str(i) for i in range(4)],
        partitions, tmp_path, BudgetLedger(tmp_path, user_quota_hours=30), torch.device("cpu"))
    # Collector fixture: bind chosen synthetic evaluation values to actual committed weights.
    metrics["test_metrics"]["accuracy"] = .99
    metrics["test_metrics"]["macro_f1"] = .985
    best_path = tmp_path / job.job_id / "best_model.pth"
    best = torch.load(best_path, weights_only=False)
    best["test_metrics"] = metrics["test_metrics"]
    atomic_torch_save(best_path, best)
    metrics["checkpoint_file_sha256"] = file_hash(best_path)
    atomic_json(tmp_path / job.job_id / "fedavg_metrics.json", metrics)

    run_stage1_collect(output_dir=tmp_path)
    comparison_data = json.loads((tmp_path / "reports" / "comparison.json").read_text(encoding="utf-8"))

    rows = comparison_data["table"]
    row_alpha1 = [r for r in rows if r["alpha"] == 1.0][0]
    row_alpha100 = [r for r in rows if r["alpha"] == 100.0][0]

    # Kiểm tra FedAvg mới đã được nạp
    assert row_alpha1["fedavg_new_acc"] == 0.9900
    assert row_alpha1["fedavg_new_status"] == "COMPLETED"

    # Kiểm tra gap tính theo điểm phần trăm: (0.98898... - 0.9900) * 100 ~ -0.10 pp (âm được giữ nguyên)
    assert row_alpha1["reference_gap_cent_minus_fed_pp"] < 0.0

    # strict_gap bắt buộc phải là None / null
    assert row_alpha1["strict_gap"] is None
    assert row_alpha100["strict_gap"] is None

    # FedAvg alpha 100 chưa có thì trạng thái phải là PENDING_OR_MISSING
    assert row_alpha100["fedavg_new_status"] == "MISSING"
    assert row_alpha100["fedavg_historical_acc"] is None
