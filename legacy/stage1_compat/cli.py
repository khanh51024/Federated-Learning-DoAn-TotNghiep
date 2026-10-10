"""Giao diện CLI chính cho stage1_compat.

Hỗ trợ các lệnh:
  python -m stage1_compat preflight [--data-dir ...] [--output-dir ...]
  python -m stage1_compat dry-run [--data-dir ...] [--output-dir ...] [--quota-hours ...]
  python -m stage1_compat calibrate [--data-dir ...] [--output-dir ...] [--quota-hours ...]
  python -m stage1_compat run [--data-dir ...] [--output-dir ...] [--quota-hours ...] [--alpha ...] [--seed ...]
  python -m stage1_compat collect [--output-dir ...] [--data-dir ...]
"""

import argparse
import json
import sys
from pathlib import Path

from stage1_compat.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    SEED,
)



def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stage1_compat",
        description="stage1_compat: Profile huấn luyện FedAvg tương thích GĐ1 và đối chiếu baseline lịch sử.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Lệnh cần thực thi")

    # 1. preflight
    p_preflight = subparsers.add_parser("preflight", help="Kiểm tra tính toàn vẹn upstream, split, partition, dataset và baselines.")
    p_preflight.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Đường dẫn PlantVillage raw/color")
    p_preflight.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất báo cáo")

    # 2. dry-run
    p_dryrun = subparsers.add_parser("dry-run", help="Chạy kiểm tra kế hoạch, identity và config mà không thực thi training loop.")
    p_dryrun.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Đường dẫn PlantVillage raw/color")
    p_dryrun.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất báo cáo")
    p_dryrun.add_argument("--quota-hours", type=float, default=0.0, help="Tổng số giờ quota người dùng nhập thực tế")

    # 3. calibrate
    p_calib = subparsers.add_parser("calibrate", help="Chạy calibration FedAvg trên dữ liệu thật (5 clients, không test) để đo tốc độ.")
    p_calib.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Đường dẫn PlantVillage raw/color")
    p_calib.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục xuất kết quả")
    p_calib.add_argument("--quota-hours", type=float, default=0.0, help="Tổng số giờ quota khả dụng")
    p_calib.add_argument("--rounds", type=int, default=1, help="Số round đo calibration (mặc định 1 round qua 5 client)")
    p_calib.add_argument("--device", type=str, default=None, help="Device ép buộc (cpu hoặc cuda)")

    # 4. run
    p_run = subparsers.add_parser("run", help="Thực thi hoặc tiếp tục huấn luyện sequential FedAvg.")
    p_run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Đường dẫn PlantVillage raw/color")
    p_run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục lưu checkpoints và ledger")
    p_run.add_argument("--quota-hours", type=float, default=0.0, help="Tổng số giờ quota khả dụng")
    p_run.add_argument("--alpha", type=float, default=None, help="Chỉ định alpha cụ thể (100.0 hoặc 1.0), nếu bỏ trống sẽ chạy cả 2")
    p_run.add_argument("--seed", type=int, default=SEED, help="Training seed (mặc định 42)")
    p_run.add_argument("--rounds", type=int, default=10, help="Số rounds FedAvg (mặc định 10)")
    p_run.add_argument("--device", type=str, default=None, help="Device ép buộc (cpu hoặc cuda)")
    p_run.add_argument("--resume", action="store_true", default=True, help="Tự động khôi phục từ checkpoint atomic trước đó")

    # 5. collect
    p_collect = subparsers.add_parser("collect", help="Tổng hợp kết quả, so sánh với baselines lịch sử, xuất bảng và biểu đồ.")
    p_collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Thư mục chứa kết quả FedAvg mới")
    p_collect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Đường dẫn PlantVillage raw/color nếu cần đánh giá lại")

    for budget_parser in (p_calib, p_run):
        budget_parser.add_argument("--already-used-hours", type=float, default=0.0,
                                   help="Total GPU hours already consumed in this 30h experiment; includes other workflows")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    actual_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(actual_argv)
    import os
    if args.command in ("calibrate", "run") and not os.environ.get("STAGE1_SUPERVISED"):
        from stage1_compat.launcher import launch
        return launch(actual_argv, args)

    if args.command == "preflight":
        from stage1_compat.preflight import run_preflight
        res = run_preflight(args.data_dir, args.output_dir)
        print(f"Preflight status: {res.status}")
        print(f"Upstream manifest matched: {res.upstream_files_matched}/{res.upstream_files_checked}")
        print(f"Class order valid: {res.class_order_ok} ({res.total_classes} classes)")
        print(f"Splits & Partitions: split={res.split_integrity_ok}, partitions={res.partitions_integrity_ok}")
        print(f"Baselines available: {res.baselines_available}")
        print(f"Báo cáo chi tiết: {res.report_path}")
        return 0 if res.status == "PASSED" else 1

    elif args.command == "dry-run":
        from stage1_compat.dry_run import run_dry_run
        res = run_dry_run(args.data_dir, args.output_dir, args.quota_hours)
        print(f"Dry-run status: {res.status}")
        print(res.summary_message)
        print(f"Jobs planned: {len(res.jobs_planned)}")
        for j in res.jobs_planned:
            print(f"  - {j['job_id']} (alpha={j['alpha']}, seed={j['seed']}) -> {j['action']}")
        print(f"Báo cáo chi tiết: {res.report_path}")
        return 0 if res.status == "SUCCESS" else 1

    elif args.command == "calibrate":
        from stage1_compat.calibrate import run_calibration
        res = run_calibration(args.data_dir, args.output_dir, args.quota_hours, args.rounds, args.device)
        print(f"Calibration status: {res.get('status')}")
        print(f"Báo cáo: {res.get('report_path')}")
        return 0 if res.get("status") == "SUCCESS" else 1

    elif args.command == "run":
        from stage1_compat.runner import run_stage1_compat
        res = run_stage1_compat(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            quota_hours=args.quota_hours,
            target_alpha=args.alpha,
            seed=args.seed,
            rounds=args.rounds,
            device_str=args.device,
        )
        print(f"Run completed: status={res.get('status')}")
        return 75 if res.get("status") == "PAUSED_QUOTA" else (0 if res.get("status") == "COMPLETED" else 1)

    elif args.command == "collect":
        from stage1_compat.collector import run_stage1_collect
        res = run_stage1_collect(output_dir=args.output_dir, data_dir=args.data_dir)
        print(f"Collection status: {res.get('status')}")
        print(f"Comparison summary exported to: {res.get('comparison_csv')}")
        return 0 if res.get("status") == "SUCCESS" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
