"""Dry-run cho profile stage1_compat.

Xác minh toàn bộ cấu hình, preflight, split, partition và baseline registry
mà KHÔNG import hoặc gọi bất kỳ training entry point nào.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stage1_compat.baselines import BaselineRegistry
from stage1_compat.config import Stage1CompatProfileConfig, get_default_profile_config
from stage1_compat.identity import generate_job_identity
from stage1_compat.preflight import run_preflight


@dataclass
class DryRunResult:
    status: str  # "SUCCESS", "FAILED"
    preflight_passed: bool
    jobs_planned: list[dict[str, Any]]
    baselines_status: dict[str, Any]
    report_path: str
    summary_message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_dry_run(
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    quota_hours: float = 0.0,
) -> DryRunResult:
    profile_cfg = get_default_profile_config(data_dir, output_dir, quota_hours)
    out_dir = Path(profile_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run preflight
    preflight_res = run_preflight(profile_cfg.data_dir, profile_cfg.output_dir)
    if preflight_res.status != "PASSED":
        report_path = out_dir / "stage1_dry_run_report.json"
        res = DryRunResult(
            status="FAILED",
            preflight_passed=False,
            jobs_planned=[],
            baselines_status={"error": "Preflight kiểm tra không thành công."},
            report_path=str(report_path),
            summary_message="Dry-run thất bại do Preflight không đạt. Kiểm tra stage1_compatibility_report.json.",
        )
        report_path.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return res

    # 2. Plan jobs & generate identities
    jobs_planned = []
    for job in profile_cfg.jobs:
        identity = generate_job_identity(job, profile_cfg.profile_name)
        jobs_planned.append({
            "job_id": job.job_id,
            "method": job.method,
            "alpha": job.alpha,
            "seed": job.seed,
            "rounds": job.rounds,
            "clients": job.num_clients,
            "batch_size": job.batch_size,
            "optimizer": job.optimizer,
            "lr": job.lr,
            "weight_decay": job.weight_decay,
            "identity_hash": identity.identity_hash,
            "action": "PLANNED (dry-run: training loop will NOT be invoked)",
        })

    # 3. Check baselines
    registry = BaselineRegistry()
    baselines_status = {
        "centralized_seed42": registry.get_centralized(42) is not None,
        "local_only_alpha1_seed42": registry.get_local_only(1.0, 42) is not None,
        "local_only_alpha100_seed42": registry.get_local_only(100.0, 42) is not None,
        "fedavg_historical_alpha1_seed42": registry.get_historical_fedavg(1.0, 42) is not None,
        "fedavg_historical_alpha100_seed42": "MISSING_BY_DESIGN (không có trong GĐ1, chuẩn)",
        "other_seeds_baselines": "MISSING (không nội suy/nhân bản từ seed 42)",
        "reference_only": True,
        "strict_comparison_eligible": False,
    }

    # 4. Save report
    dry_run_payload = {
        "status": "SUCCESS",
        "profile": profile_cfg.profile_name,
        "backend": profile_cfg.backend,
        "preflight_passed": True,
        "data_dir": str(profile_cfg.data_dir),
        "output_dir": str(profile_cfg.output_dir),
        "user_quota_hours": profile_cfg.user_quota_hours,
        "jobs_count": len(jobs_planned),
        "jobs_planned": jobs_planned,
        "baselines_status": baselines_status,
        "scientific_stage2_complete": False,
        "guarantee": "Dry-run hoàn thành thành công mà không gọi bất kỳ training execution loop nào.",
    }

    report_path = out_dir / "stage1_dry_run_report.json"
    report_path.write_text(json.dumps(dry_run_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return DryRunResult(
        status="SUCCESS",
        preflight_passed=True,
        jobs_planned=jobs_planned,
        baselines_status=baselines_status,
        report_path=str(report_path),
        summary_message=f"Dry-run thành công cho {len(jobs_planned)} FedAvg jobs. Không gọi training loop.",
    )
