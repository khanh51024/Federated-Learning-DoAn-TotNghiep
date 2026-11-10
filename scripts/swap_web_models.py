"""Thay mô hình của web bằng các run mới trong MỘT lệnh, hoặc lùi về gói trước.

    python scripts/swap_web_models.py --runs runs/v4b --seed 42 --name v4b_s42 --label v4b      # thay
    python scripts/swap_web_models.py --rollback artifacts/branch_demo/20261003                # lùi về gói cũ

Mọi nhận dạng (release, tập validation, tên mô hình, giao thức train) nằm ở `import_manifest.json` của gói (src/bundle.py); lệnh này chỉ:
  1. nhập các run thành gói artifacts/model_bundles/<tên> (scripts/import_run_bundle.py) nếu gói chưa có;
  2. CHỤP báo cáo và thư viện ảnh sai của gói đang dùng vào artifacts/report_snapshots/<gói>/ (để lùi nguyên vẹn);
  3. đổi con trỏ artifacts/active_bundle.json (kèm mã băm manifest của gói: sửa manifest sau đó thì web từ chối);
  4. sinh lại toàn bộ báo cáo theo đúng thứ tự phụ thuộc, rồi phần học cộng đồng, rồi danh mục mô hình và trạng thái;
  5. một bước lỗi thì tự LÙI (khôi phục con trỏ và báo cáo đã chụp) và thoát mã khác 0.

Gói cũ không bị xóa. `--steps` liệt kê các bước; `--skip-slow` bỏ hai bước chậm (ảnh nguyên khung PlantDoc, bộ ảnh ngoài PlantWild); `--only <bước>` chạy riêng một bước, `--from <bước>` chạy tiếp từ một bước (cả hai chỉ cho gói đã được chọn, không đổi con trỏ).
Lệnh kiểm trước các thư viện cần cho mọi bước (onnx, onnxruntime, scipy, sklearn) để không chạy cả buổi rồi mới lỗi ở bước cuối.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bundle, registry as reg  # noqa: E402

SNAPSHOTS = reg.REPORT_SNAPSHOTS_DIR
COMMUNITY_SNAPSHOTS = "data/report_snapshots"          # tệp cộng đồng nằm ngoài git (.gitignore), nên chụp ngoài git
# Báo cáo (tracked) mà web đọc theo gói; thư viện ảnh sai đi kèm catalog của nó
REPORTS = (
    reg.LOCAL_REEVAL_JSON, reg.HOLDOUT_EVAL_JSON, reg.STATISTICS_JSON, reg.TRAINING_PROTOCOL_REPORT_JSON, reg.PARTITION_SCHEMES_JSON,
    reg.RELEASE_LEAKAGE_AUDIT_JSON, reg.CLIENT_SHARD_COUNTS_JSON, reg.PER_CLASS_REPORT_JSON, reg.DOMAIN_GAP_REPORT_JSON, reg.FULLFRAME_EVAL_JSON,
    reg.EXTERNAL_EVAL_RESULTS_JSON, reg.PLANTVILLAGE_EVAL_JSON, reg.COMMUNITY_SIMULATION_JSON, reg.CLIENT_FAIRNESS_JSON,
)
REPORT_DIRS = (reg.ERROR_GALLERY_DIR,)
COMMUNITY_FILES = (reg.COMMUNITY_FEATURE_MODEL_ONNX, reg.COMMUNITY_BASE_HEAD_NPZ, reg.COMMUNITY_EVAL_FEATURES_NPZ, reg.COMMUNITY_MANIFEST_JSON)
COMMUNITY_DIRS = (reg.COMMUNITY_STATE_DIR,)

PY = [sys.executable, "-X", "utf8"]
# (tên bước, lệnh, chậm?)
STEPS: list[tuple[str, list[str], bool]] = [
    ("registry", PY + ["scripts/update_status.py"], False),
    ("client_shards", PY + ["scripts/report_client_shards.py"], False),
    ("partition_schemes", PY + ["scripts/report_partition_schemes.py"], False),
    ("leakage_audit", PY + ["scripts/audit_release_leakage.py"], False),
    ("per_class", PY + ["scripts/report_per_class.py"], False),
    ("domain_gap", PY + ["scripts/report_domain_gap.py"], False),
    ("training_protocol", PY + ["scripts/report_training_protocol.py"], False),
    ("reeval_val", PY + ["scripts/evaluate_published_models.py", "--reuse"], False),
    ("reeval_holdout", PY + ["scripts/evaluate_published_models.py", "--holdout", "--reuse"], False),
    ("statistics", PY + ["scripts/report_statistics.py"], False),
    ("client_fairness", PY + ["scripts/report_client_fairness.py"], False),
    ("plantvillage_reference", PY + ["scripts/evaluate_plantvillage_reference.py"], False),
    ("error_gallery", PY + ["scripts/build_error_gallery.py"], False),
    ("community_export", PY + ["scripts/export_community_feature_model.py"], False),
    ("community_eval_features", PY + ["scripts/build_community_eval_features.py"], False),
    ("community_sim_features", PY + ["scripts/build_community_sim_features.py"], False),
    ("community_simulation", PY + ["scripts/simulate_community.py"], False),
    ("fullframe", PY + ["scripts/eval_fullframe_plantdoc.py"], True),
    ("external_eval", PY + ["scripts/eval_external_set.py", "--reuse"], True),
    ("status", PY + ["scripts/update_status.py"], False),
]


# Gói nhập từ nhóm train khác (PC v3) chỉ được đánh giá trên validation: tập kiểm thử đã mở đúng một lần ở bước P6 của họ, tập calibration niêm phong,
# dữ liệu ngoài miền không được chạy thêm. Các bước dưới đây mở một trong ba loại đó nên `--validation-only` bỏ chúng.
SEALED_DATA_STEPS = ("leakage_audit", "reeval_holdout", "training_protocol", "statistics", "client_fairness", "plantvillage_reference",
                     "community_sim_features", "community_simulation", "fullframe", "external_eval")
# Báo cáo mà web đọc do từng bước bỏ: sau khi đã chụp gói cũ thì xóa, để web báo "chưa có" thay vì hiện số của gói cũ dưới tên gói mới
SKIPPED_STEP_OUTPUTS = {
    "leakage_audit": reg.RELEASE_LEAKAGE_AUDIT_JSON, "reeval_holdout": reg.HOLDOUT_EVAL_JSON, "training_protocol": reg.TRAINING_PROTOCOL_REPORT_JSON,
    "statistics": reg.STATISTICS_JSON, "client_fairness": reg.CLIENT_FAIRNESS_JSON, "plantvillage_reference": reg.PLANTVILLAGE_EVAL_JSON,
    "community_simulation": reg.COMMUNITY_SIMULATION_JSON, "fullframe": reg.FULLFRAME_EVAL_JSON, "external_eval": reg.EXTERNAL_EVAL_RESULTS_JSON,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_id(directory: str) -> str:
    return Path(directory).name


def copy_in(relative: str, destination: Path) -> bool:
    source = ROOT / relative
    target = destination / relative
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return True
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return True
    return False


def snapshot(current_dir: str) -> Path:
    """Chụp báo cáo, thư viện ảnh sai và tệp cộng đồng của gói đang dùng; ghi danh sách tệp có/không có để khôi phục chính xác."""
    ident = bundle_id(current_dir)
    reports = ROOT / SNAPSHOTS / ident
    community = ROOT / COMMUNITY_SNAPSHOTS / ident
    for directory in (reports, community):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    present = {"reports": [r for r in (*REPORTS, *REPORT_DIRS) if copy_in(r, reports)],
               "community": [c for c in (*COMMUNITY_FILES, *COMMUNITY_DIRS) if copy_in(c, community)]}
    (reports / "snapshot.json").write_text(json.dumps({"bundle_dir": current_dir, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                       **present}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[swap] đã chụp {len(present['reports'])} báo cáo và {len(present['community'])} tệp cộng đồng của {ident}", flush=True)
    return reports


def restore(ident: str) -> bool:
    """Khôi phục bản chụp của gói `ident` (báo cáo, thư viện ảnh sai, tệp cộng đồng). False nếu chưa có bản chụp."""
    reports = ROOT / SNAPSHOTS / ident
    community = ROOT / COMMUNITY_SNAPSHOTS / ident
    meta_path = reports / "snapshot.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for relative in (*REPORTS, *REPORT_DIRS):
        target = ROOT / relative
        if relative in meta["reports"]:
            source = reports / relative
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        elif target.is_file():
            target.unlink()          # gói được khôi phục không có tệp này: không để lại tệp của gói khác
    for relative in (*COMMUNITY_FILES, *COMMUNITY_DIRS):
        target = ROOT / relative
        if relative in meta["community"]:
            source = community / relative
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        elif target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    return True


def read_pointer() -> dict[str, Any]:
    try:
        return json.loads((ROOT / bundle.POINTER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


REQUIRED_MODULES = ("onnx", "onnxruntime", "scipy", "sklearn", "torch", "torchvision", "PIL")


def missing_modules() -> list[str]:
    import importlib.util

    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def write_pointer(directory: str | None, previous: str | None = None) -> None:
    """`previous`: gói trước đó, để một lần thay bị gián đoạn (chạy tiếp bằng --from) vẫn lùi đúng về gói trước chứ không về chính nó."""
    pointer = ROOT / bundle.POINTER
    if directory is None or directory.rstrip("/") == bundle.LEGACY_DIR:
        if pointer.exists():
            pointer.unlink()         # không có con trỏ = gói cũ (mặc định trong src/bundle.py)
        return
    manifest = ROOT / directory / "import_manifest.json"
    pointer.write_text(json.dumps({"bundle_dir": directory, "manifest_sha256": sha256_file(manifest), "previous_bundle_dir": previous,
                                   "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n", encoding="utf-8")


def run_step(name: str, command: list[str], log: Path) -> bool:
    started = time.time()
    print(f"[swap] >>> {name}: {' '.join(command[3:])}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== {name} ({time.strftime('%F %T')}) =====\n")
        process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                 env={**os.environ, "PYTHONUTF8": "1"})
        stream.write(process.stdout)
    tail = [line for line in process.stdout.strip().splitlines() if line.strip()][-2:]
    print(f"[swap]     {'OK' if process.returncode == 0 else 'LỖI'} sau {time.time() - started:.0f} giây: {' | '.join(tail)[:300]}", flush=True)
    return process.returncode == 0


def skipped_steps(args: argparse.Namespace) -> set[str]:
    names = [step[0] for step in STEPS]
    skipped = set(SEALED_DATA_STEPS) if args.validation_only else set()
    for name in filter(None, (args.skip or "").split(",")):
        if name not in names:
            raise SystemExit(f"--skip: không có bước '{name}'; các bước: {names}")
        skipped.add(name)
    return skipped


def select_steps(skip_slow: bool, only: str | None, start: str | None = None, skipped: set[str] = frozenset(),
                 validation_only: bool = False) -> list[tuple[str, list[str], bool]]:
    names = [step[0] for step in STEPS]
    for label, value in (("--only", only), ("--from", start)):
        if value and value not in names:
            raise SystemExit(f"{label}: không có bước '{value}'; các bước: {names}")
    if only and only in skipped:
        raise SystemExit(f"--only {only}: bước này bị bỏ bởi --skip/--validation-only")
    chosen = [step for step in STEPS if step[0] == only] if only else (STEPS[names.index(start):] if start else STEPS)
    chosen = [step for step in chosen if step[0] not in skipped and not (skip_slow and not only and step[2])]
    if validation_only:   # đánh giá lại chỉ trên validation: không mở calibration, không hiệu chỉnh nhiệt độ
        chosen = [(n, c + ["--validation-only"] if n == "reeval_val" else c, slow) for n, c, slow in chosen]
    return chosen


def swap(args: argparse.Namespace) -> int:
    current = bundle.active_bundle_dir()
    target_dir = f"{bundle.BUNDLES_DIR}/{args.name}"
    target = ROOT / target_dir
    absent = missing_modules()
    if absent:
        print(f"[swap] thiếu thư viện: {absent}. Cài: pip install -r requirements.txt (chưa đổi gì)", file=sys.stderr)
        return 1
    if (args.only or args.start) and current != target_dir:
        print(f"[swap] --only/--from chỉ chạy khi gói {target_dir} đã được chọn (hiện là {current}); chạy lại không kèm hai tùy chọn đó", file=sys.stderr)
        return 1
    # gói sẽ lùi về khi lỗi: gói đang dùng; nếu đang tiếp tục một lần thay dở thì là gói trước đó ghi trong con trỏ
    previous = (read_pointer().get("previous_bundle_dir") or bundle.LEGACY_DIR) if current == target_dir else current
    log = ROOT / "tmp" / f"swap_{args.name}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("", encoding="utf-8")
    if not target.is_dir():
        command = PY + ["scripts/import_run_bundle.py", "--runs", str(args.runs), "--seed", str(args.seed), "--name", args.name, "--label", args.label]
        if not run_step("import_bundle", command, log):
            print(f"[swap] nhập gói lỗi, xem {log}", file=sys.stderr)
            return 1
    if current != target_dir:
        snapshot(current)
    write_pointer(target_dir, previous)
    if current != target_dir:
        # tệp cộng đồng của gói cũ không dùng được với gói mới (dấu vân tay tài sản): xóa để bước xuất tạo lại; trạng thái cũ đã nằm trong bản chụp
        for relative in (*COMMUNITY_FILES, *COMMUNITY_DIRS):
            path = ROOT / relative
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        # báo cáo của các bước bị bỏ cũng là của gói cũ: xóa (đã chụp ở trên; --rollback khôi phục lại)
        for step_name in sorted(skipped_steps(args)):
            output = SKIPPED_STEP_OUTPUTS.get(step_name)
            if output and (ROOT / output).is_file():
                (ROOT / output).unlink()
                print(f"[swap] bỏ bước {step_name}: đã gỡ {output} của gói cũ khỏi chỗ web đọc", flush=True)
    for name, command, _slow in select_steps(args.skip_slow, args.only, args.start, skipped_steps(args), args.validation_only):
        if not run_step(name, command, log):
            print(f"[swap] bước {name} lỗi: lùi về {previous}. Chi tiết: {log}", file=sys.stderr)
            write_pointer(previous)
            restored = restore(bundle_id(previous))
            run_step("registry_after_rollback", PY + ["scripts/update_status.py"], log)
            print(f"[swap] đã lùi (khôi phục bản chụp: {restored})", file=sys.stderr)
            return 2
    print(f"[swap] XONG: web dùng gói {target_dir}. Lùi: python scripts/swap_web_models.py --rollback {previous}")
    return 0


def rollback(args: argparse.Namespace) -> int:
    directory = args.rollback.replace("\\", "/").rstrip("/")
    if not (ROOT / directory / "import_manifest.json").is_file():
        print(f"không có gói {directory}", file=sys.stderr)
        return 1
    current = bundle.active_bundle_dir()
    log = ROOT / "tmp" / "swap_rollback.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("", encoding="utf-8")
    if current != directory:
        snapshot(current)             # giữ trạng thái của gói đang dùng, để thay lại được
    write_pointer(directory)
    restored = restore(bundle_id(directory))
    print(f"[swap] khôi phục bản chụp báo cáo của {bundle_id(directory)}: {restored}", flush=True)
    if not restored:
        print("[swap] CHƯA có bản chụp của gói này: báo cáo hiện tại là của gói khác; chạy lại các bước báo cáo cho gói này", file=sys.stderr)
    ok = run_step("registry", PY + ["scripts/update_status.py"], log)
    return 0 if ok and restored else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, help="thư mục các run (ví dụ runs/v4b)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", help="tên gói (ví dụ v4b_s42)")
    parser.add_argument("--label", help="nhãn hiển thị cạnh tên mô hình (ví dụ v4b)")
    parser.add_argument("--rollback", metavar="THƯ_MỤC_GÓI", help="lùi về gói đã có, ví dụ artifacts/branch_demo/20261003")
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument("--skip", metavar="BƯỚC[,BƯỚC]", help="bỏ các bước này (báo cáo của chúng được gỡ khỏi chỗ web đọc, web báo chưa có)")
    parser.add_argument("--validation-only", action="store_true",
                        help="gói chỉ được đánh giá trên validation: bỏ mọi bước mở tập kiểm thử/calibration/ngoài miền và đánh giá lại không hiệu chỉnh nhiệt độ")
    parser.add_argument("--only", help="chỉ chạy một bước của gói đã chọn (xem --steps)")
    parser.add_argument("--from", dest="start", metavar="BƯỚC", help="chạy tiếp từ một bước của gói đã chọn (sau khi sửa nguyên nhân lỗi)")
    parser.add_argument("--steps", action="store_true", help="liệt kê các bước rồi thoát")
    args = parser.parse_args()
    if args.steps:
        for name, command, slow in STEPS:
            print(f"{name:26} {'(chậm) ' if slow else ''}{' '.join(command[3:])}")
        return 0
    if args.rollback:
        return rollback(args)
    exists = bool(args.name) and (ROOT / bundle.BUNDLES_DIR / args.name).is_dir()
    if not args.name or not (exists or (args.runs and args.label)):
        parser.error("cần --name của gói đã có (artifacts/model_bundles/<tên>), hoặc --name --label --runs để nhập từ các run")
    return swap(args)


if __name__ == "__main__":
    raise SystemExit(main())
