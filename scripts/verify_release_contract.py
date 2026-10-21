"""Kiểm release dữ liệu bằng CHÍNH các bộ kiểm của nhánh huấn luyện, để biết bộ chạy có nhận release hay không (chỉ đọc, không sửa gì).

Mã kiểm nằm ở nhánh `origin/train-centralized` (plant_data_contract/*, training-workflows/shared/data_selection.py). Script đọc các tệp đó bằng `git show` và chạy
trong bộ nhớ, nên không cần checkout nhánh và không chép mã vào repo này. Các kiểm tra:
  1. verify_release_integrity       SHA release, taxonomy, mapping, detection, scene_groups, manifest từng split, khóa và SHA từng shard của mọi phân vùng
  2. verify_manifest_records_contract   sample_id/group_id duy nhất, quarantine không supervised_eligible, đường dẫn hợp lệ
  3. verify_partition_directory + verify_partition_audit   shard khớp train, rời nhau, không cắt nhóm; partition_spec_v4.json PASS và khớp SHA release
  4. verify_full_data_gate          quality_audit_v4.json PASS, khớp SHA release (cổng rò rỉ trước khi train đầy đủ)
  5. inspect_dataset (data_selection.py)  cách bộ chạy chọn dữ liệu cho một job; đặt RELEASE_NAME của bộ chạy sang `pv_pd_v4` ngay trong bộ nhớ,
     vì bản gốc gắn cứng `pv_pd_v3` (việc cần sửa ở nhánh train trước khi train thật)
  6. (tùy chọn --images) verify_image_dataset_integrity: ảnh tồn tại, đúng SHA-256, giải mã được

Chạy:
    python scripts/verify_release_contract.py [--release data/dataset_sources/mixed/pv_pd_v4] [--branch origin/train-centralized] [--images]
    python scripts/verify_release_contract.py --release D/dataset_sources/mixed/pv_pd_v4 --dataset-root D/dataset_sources --contract-dir train_contract --images
        (chạy trên gói đã giải nén ngoài repo: không cần git; `train_contract/` do scripts/pack_train_bundle.py đóng kèm)
Mã thoát 0 = mọi kiểm tra qua; 1 = có kiểm tra không qua.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

DEFAULT_RELEASE = "data/dataset_sources/mixed/pv_pd_v4"
DEFAULT_BRANCH = "origin/train-centralized"
MODULE_FILES = {
    "integrity": "plant_data_contract/plant_data_contract/integrity.py",
    "partitions": "plant_data_contract/plant_data_contract/partitions.py",
    "full_gate": "plant_data_contract/plant_data_contract/full_gate.py",
    "data_selection": "training-workflows/shared/data_selection.py",
}
NUM_CLIENTS = 5
SCHEMES = ("label_alpha_0_1", "iid")


def load_branch_module(branch: str, path: str, contract_dir: Path | None = None) -> dict[str, Any]:
    """Chạy một tệp .py của nhánh `branch` (hoặc bản chép trong `contract_dir`) trong không gian tên riêng (các tệp này chỉ import thư viện chuẩn) và trả không gian tên đó."""
    if contract_dir is not None:
        source = (contract_dir / Path(path).name).read_text(encoding="utf-8")
        label = str(contract_dir / Path(path).name)
    else:
        source = subprocess.check_output(["git", "show", f"{branch}:{path}"], cwd=ROOT).decode("utf-8")
        label = f"{branch}:{path}"
    namespace: dict[str, Any] = {"__name__": f"branch_{Path(path).stem}", "__file__": path}
    exec(compile(source, label, "exec"), namespace)  # noqa: S102 - mã của chính repo này, chỉ đọc
    return namespace


def run_checks(release: Path, modules: dict[str, dict[str, Any]], with_images: bool, dataset_root: Path) -> list[tuple[str, bool, str]]:
    integrity, partitions, full_gate, selection = (modules[k] for k in ("integrity", "partitions", "full_gate", "data_selection"))
    release_sha = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], Any]) -> None:
        try:
            detail = fn()
            results.append((name, True, str(detail)[:160] if detail is not None else ""))
        except Exception as error:  # noqa: BLE001 - mọi lỗi của bộ kiểm đều là kết quả cần báo
            results.append((name, False, f"{type(error).__name__}: {str(error)[:400]}"))

    for scheme in SCHEMES:
        check(f"verify_release_integrity[{scheme}]", lambda s=scheme: integrity["verify_release_integrity"](
            release, expected_release_sha=release_sha, partition_scheme=s)["status"])
    check("verify_manifest_records_contract", lambda: integrity["verify_manifest_records_contract"](release)["status"])
    clients = [f"client_{i:02d}" for i in range(NUM_CLIENTS)]
    for scheme in SCHEMES:
        check(f"verify_partition_directory[{scheme}]", lambda s=scheme: partitions["verify_partition_directory"](release, s, clients).get("status", "ok"))
        check(f"verify_partition_audit[{scheme}]", lambda s=scheme: partitions["verify_partition_audit"](release, s, clients, release_sha)["status"])
    audit_path = release / "quality_audit_v4.json"
    check("verify_full_data_gate", lambda: full_gate["verify_full_data_gate"](
        release, release_sha, {"quality_gate": {"audit_filename": audit_path.name, "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}})["status"])

    def selection_view(scheme: str | None) -> str:
        saved = selection["RELEASE_NAME"]
        selection["RELEASE_NAME"] = release.name  # bản gốc gắn cứng pv_pd_v3; hàm đọc tên từ globals nên đặt tạm trong bộ nhớ
        try:
            out = selection["inspect_dataset"](dataset_root, scheme, NUM_CLIENTS, release_sha)
        finally:
            selection["RELEASE_NAME"] = saved
        return f"train={out['train_count']}" + (f" shards={[v['count'] for v in out['client_shards'].values()]}" if scheme else "")

    check("data_selection.inspect_dataset[full]", lambda: selection_view(None))
    for scheme in SCHEMES:
        check(f"data_selection.inspect_dataset[{scheme}]", lambda s=scheme: selection_view(s))
    if with_images:
        check("verify_image_dataset_integrity", lambda: integrity["verify_image_dataset_integrity"](
            dataset_root, release, splits=("train", "val", "calibration", "legacy_diagnostic_test"), verify_hashes=True, check_decode=True,
            sample_decode_count=1500)["status"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--images", action="store_true", help="kiểm cả byte và khả năng giải mã của mọi ảnh (chậm)")
    parser.add_argument("--dataset-root", default=None, help="thư mục chứa mixed/<release> và ảnh (mặc định data/dataset_sources của repo)")
    parser.add_argument("--contract-dir", default=None, help="thư mục chứa bản chép các tệp kiểm của nhánh train (thay cho git show)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    release = Path(args.release) if Path(args.release).is_absolute() else reg.abs_path(args.release)
    contract_dir = Path(args.contract_dir).resolve() if args.contract_dir else None
    modules = {name: load_branch_module(args.branch, path, contract_dir) for name, path in MODULE_FILES.items()}
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else reg.abs_path(reg.DATASET_SOURCES_DIR)
    results = run_checks(release, modules, args.images, dataset_root)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print(f"[contract] {len(results) - len(failed)}/{len(results)} qua" + (f"; không qua: {failed}" if failed else ""))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"release": str(release), "branch": args.branch, "checks": [
            {"name": n, "ok": ok, "detail": d} for n, ok, d in results]}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
