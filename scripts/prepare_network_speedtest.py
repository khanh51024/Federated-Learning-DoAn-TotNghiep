"""Dựng bộ dữ liệu nhỏ và gói mã để đo tốc độ FedAvg giữa hai máy.

Phép đo này đo TỐC ĐỘ (mạng, thời gian tính từng máy, byte trên dây), KHÔNG đo độ chính xác. Mỗi client có một shard nhỏ ảnh
PlantVillage thật (mỗi lớp N ảnh, hai shard rời nhau) để thời gian tính là thời gian huấn luyện thật chứ không phải ảnh nhiễu.

Tạo ra thư mục <out>/:
    partition/                  hợp đồng 2 client (partition_config.json + clients/client_XX.csv) cho Flower server
    images_client_XX/           ảnh của từng client, giữ nguyên đường dẫn tương đối (để chép sang máy khác)
    bundle_client_01.tar.gz     mã src/ + partition/ + ảnh của client 01 + deploy/lan_client/: gói chép sang máy thứ hai
    goi_laptop_2.zip            cùng nội dung, dạng .zip để giải nén trên Windows
    speedtest_manifest.json     mã băm từng tệp, số ảnh, byte

Chế độ nhiều client có ảnh kiểm tra của máy chủ (cho phép đo 10 client, server tách riêng):
    python scripts/prepare_network_speedtest.py --out tmp/net10 --clients 10 --per-class 10 \\
        --val-per-class 3 --test-per-class 5 --bundles 0-4,5-9
  thêm partition/global_val.csv + global_test.csv + images_eval/ (chỉ máy chủ giữ), và mỗi gói .zip chỉ chứa ảnh của các client trong khoảng.
  Ảnh kiểm tra và ảnh học rời nhau; ảnh có mã băm trùng split `calibration` (tập niêm phong của release v4) bị loại khỏi mọi phần.
  Mô hình ở phép đo này khởi tạo từ ImageNet và không thuộc protocol v4, nên số độ chính xác của phép đo chỉ là bằng chứng hệ thống.

Chạy (hai client như cũ):
    python scripts/prepare_network_speedtest.py --out tmp/speedtest --per-class 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SEED = 42
CLIENTS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_hashes(manifest: Path) -> frozenset[str]:
    """Chỉ đọc cột mã băm byte của các ảnh thuộc split `calibration` để loại chúng; không đọc nhãn hay độ chính xác nào."""
    hashes = set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                hashes.add(json.loads(line)["byte_sha256"])
    return frozenset(hashes)


def pick_disjoint_shards(root: Path, classes: list[str], per_class: int, clients: int, seed: int, *, val_per_class: int = 0,
                         test_per_class: int = 0, sealed: frozenset[str] = frozenset()) -> tuple[list[list[tuple[str, int, str]]], list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Mỗi client nhận `per_class` ảnh mỗi lớp, các client không trùng ảnh; ảnh kiểm tra (val, test) rời với mọi shard.
    Trả (shards, val, test), mỗi phần tử là (đường dẫn tương đối, nhãn, lớp). Ảnh có mã băm trong `sealed` bị bỏ qua."""
    rng = random.Random(seed)
    shards: list[list[tuple[str, int, str]]] = [[] for _ in range(clients)]
    val: list[tuple[str, int, str]] = []
    test: list[tuple[str, int, str]] = []
    need = per_class * clients + val_per_class + test_per_class
    for label, name in enumerate(classes):
        files = sorted(p.name for p in (root / name).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(files)
        if sealed:
            usable: list[str] = []
            for file in files:
                if sha256_file(root / name / file) not in sealed:
                    usable.append(file)
                    if len(usable) == need:
                        break
            files = usable
        if len(files) < need:
            raise ValueError(f"lớp {name} chỉ có {len(files)} ảnh dùng được, cần {need}")
        for k in range(clients):
            for file in files[k * per_class:(k + 1) * per_class]:
                shards[k].append((f"{name}/{file}", label, name))
        base = per_class * clients
        val += [(f"{name}/{file}", label, name) for file in files[base:base + val_per_class]]
        test += [(f"{name}/{file}", label, name) for file in files[base + val_per_class:base + val_per_class + test_per_class]]
    return shards, val, test


def write_eval_manifests(out: Path, val: list[tuple[str, int, str]], test: list[tuple[str, int, str]]) -> None:
    """global_val.csv / global_test.csv của máy chủ (cùng cột với csv của client) và bản sao ảnh trong images_eval/."""
    for name, rows in (("global_val", val), ("global_test", test)):
        with (out / "partition" / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["relative_path", "label", "class_name", "client_id", "split"])
            for rel, label, cls in rows:
                writer.writerow([rel, label, cls, -1, name.removeprefix("global_")])


def write_partition(out: Path, shards: list[list[tuple[str, int, str]]], classes: list[str]) -> None:
    clients_dir = out / "partition" / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    config = {"protocol": "network_speedtest_v1", "num_clients": len(shards), "seed": SEED, "class_names": classes,
              "n_k": [len(s) for s in shards], "note": "Shard nhỏ chỉ để đo tốc độ; không dùng làm kết quả độ chính xác."}
    (out / "partition" / "partition_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, shard in enumerate(shards):
        with (clients_dir / f"client_{k:02d}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["relative_path", "label", "class_name", "client_id", "split"])
            for rel, label, name in shard:
                writer.writerow([rel, label, name, k, "train"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="tmp/speedtest")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--clients", type=int, default=CLIENTS, help="số client (shard rời nhau)")
    parser.add_argument("--val-per-class", type=int, default=0, help="ảnh kiểm tra của máy chủ mỗi lớp (val, đo mỗi vòng); 0 = không dựng tập kiểm tra")
    parser.add_argument("--test-per-class", type=int, default=0, help="ảnh test cuối của máy chủ mỗi lớp")
    parser.add_argument("--bundles", default="", help="khoảng client cho từng gói .zip, ví dụ 0-4,5-9; để trống = gói cũ cho client 01")
    parser.add_argument("--release-dir", default=reg.RELEASE_V4_DIR if hasattr(reg, "RELEASE_V4_DIR") else "data/dataset_sources/mixed/pv_pd_v4")
    args = parser.parse_args()
    if args.clients < 2:
        print("[speedtest] cần ít nhất 2 client", file=sys.stderr)
        return 1
    with_eval = args.val_per_class > 0 or args.test_per_class > 0
    if with_eval and (args.val_per_class <= 0 or args.test_per_class <= 0):
        print("[speedtest] --val-per-class và --test-per-class phải cùng lớn hơn 0", file=sys.stderr)
        return 1
    ranges: list[tuple[int, int]] = []
    for part in filter(None, args.bundles.split(",")):
        first, _, last = part.partition("-")
        low, high = int(first), int(last or first)
        if not 0 <= low <= high < args.clients:
            print(f"[speedtest] khoảng gói {part!r} nằm ngoài 0..{args.clients - 1}", file=sys.stderr)
            return 1
        ranges.append((low, high))

    from web.demo_evidence import load_evidence

    classes = load_evidence()["classes"]
    pv_root = reg.abs_path(reg.DATASET_SOURCES_DIR) / "PlantVillage-Dataset" / "raw" / "color"
    if not pv_root.is_dir():
        print(f"[speedtest] thiếu {pv_root}", file=sys.stderr)
        return 1
    out = (ROOT / args.out).resolve()
    allowed = (ROOT / "tmp").resolve()
    if out == allowed or allowed not in out.parents:
        print(f"[speedtest] --out phải nằm trong {allowed} (script xóa thư mục này trước khi dựng lại)", file=sys.stderr)
        return 1
    if out.exists():
        shutil.rmtree(out)
    sealed: frozenset[str] = frozenset()
    if with_eval:
        calibration = reg.abs_path(args.release_dir) / "manifests" / "calibration.jsonl"
        if not calibration.is_file():
            print(f"[speedtest] thiếu {calibration}: không loại được ảnh của split niêm phong", file=sys.stderr)
            return 1
        sealed = sealed_hashes(calibration)
    shards, val, test = pick_disjoint_shards(pv_root, classes, args.per_class, args.clients, SEED,
                                             val_per_class=args.val_per_class, test_per_class=args.test_per_class, sealed=sealed)
    write_partition(out, shards, classes)
    if with_eval:
        write_eval_manifests(out, val, test)
        for rel, _, _ in [*val, *test]:
            target = out / "images_eval" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pv_root / rel, target)
    for k, shard in enumerate(shards):
        for rel, _, _ in shard:
            target = out / f"images_client_{k:02d}" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pv_root / rel, target)

    bundles: dict[str, Any] = {}
    if ranges:
        # Mỗi gói cho một máy: mã src/, hợp đồng partition (chỉ csv của client; KHÔNG có ảnh/csv kiểm tra của máy chủ) và ảnh của các client trong khoảng.
        for low, high in ranges:
            zip_path = out / f"goi_client_{low:02d}_{high:02d}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted((ROOT / "src").rglob("*.py")):
                    archive.write(path, path.relative_to(ROOT).as_posix())
                archive.write(out / "partition" / "partition_config.json", "partition/partition_config.json")
                for k in range(low, high + 1):
                    archive.write(out / "partition" / "clients" / f"client_{k:02d}.csv", f"partition/clients/client_{k:02d}.csv")
                    for image in sorted((out / f"images_client_{k:02d}").rglob("*")):
                        if image.is_file():
                            archive.write(image, image.relative_to(out).as_posix())
                for path in sorted((ROOT / "deploy" / "lan_client").iterdir()):
                    archive.write(path, path.name)
            bundles[zip_path.name] = {"clients": f"{low}-{high}", "bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path)}
        bundle_info: dict[str, Any] = {"files": bundles}
    else:
        # Gói cho máy thứ hai (client 01): mã nguồn Python của src/, hợp đồng partition và ảnh của client 01
        bundle = out / "bundle_client_01.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            for path in sorted((ROOT / "src").rglob("*.py")):
                tar.add(path, arcname=path.relative_to(ROOT).as_posix())
            tar.add(out / "partition", arcname="partition")
            tar.add(out / "images_client_01", arcname="images_client_01")
            for path in sorted((ROOT / "deploy" / "lan_client").iterdir()):
                tar.add(path, arcname=path.name)
        # Bản .zip cùng nội dung: Windows mở được ngay, kể cả bản không tự giải nén .tar.gz
        bundle_zip = out / "goi_laptop_2.zip"
        with tarfile.open(bundle, "r:gz") as tar, zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in tar.getmembers():
                if member.isfile():
                    archive.writestr(member.name, tar.extractfile(member).read())
        bundle_info = {"file": bundle.name, "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)}
    manifest: dict[str, Any] = {
        "per_class": args.per_class, "n_k": [len(s) for s in shards], "seed": SEED,
        "bundle": bundle_info,
        "partition_config_sha256": sha256_file(out / "partition" / "partition_config.json"),
        "client_manifest_sha256": {f"client_{k:02d}": sha256_file(out / "partition" / "clients" / f"client_{k:02d}.csv") for k in range(args.clients)},
        "images_bytes": {f"client_{k:02d}": sum(p.stat().st_size for p in (out / f"images_client_{k:02d}").rglob("*") if p.is_file()) for k in range(args.clients)},
    }
    if with_eval:
        manifest["server_eval"] = {
            "val_per_class": args.val_per_class, "test_per_class": args.test_per_class, "val": len(val), "test": len(test),
            "global_val_sha256": sha256_file(out / "partition" / "global_val.csv"), "global_test_sha256": sha256_file(out / "partition" / "global_test.csv"),
            "excluded_sealed_calibration_hashes": len(sealed),
            "note": "Ảnh PlantVillage thật, rời mọi shard; mô hình khởi tạo ImageNet nên độ chính xác chỉ là bằng chứng hệ thống, không phải kết quả protocol v4.",
        }
    (out / "speedtest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
