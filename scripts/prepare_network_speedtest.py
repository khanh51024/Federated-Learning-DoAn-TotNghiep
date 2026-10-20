"""Dựng bộ dữ liệu nhỏ và gói mã để đo tốc độ FedAvg giữa hai máy.

Phép đo này đo TỐC ĐỘ (mạng, thời gian tính từng máy, byte trên dây), KHÔNG đo độ chính xác. Mỗi client có một shard nhỏ ảnh
PlantVillage thật (mỗi lớp N ảnh, hai shard rời nhau) để thời gian tính là thời gian huấn luyện thật chứ không phải ảnh nhiễu.

Tạo ra thư mục <out>/:
    partition/                  hợp đồng 2 client (partition_config.json + clients/client_XX.csv) cho Flower server
    images_client_XX/           ảnh của từng client, giữ nguyên đường dẫn tương đối (để chép sang máy khác)
    bundle_client_01.tar.gz     mã src/ + partition/ + ảnh của client 01: gói chép sang máy thứ hai
    speedtest_manifest.json     mã băm từng tệp, số ảnh, byte

Chạy:
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


def pick_disjoint_shards(root: Path, classes: list[str], per_class: int, clients: int, seed: int) -> list[list[tuple[str, int, str]]]:
    """Mỗi client nhận `per_class` ảnh mỗi lớp, các client không trùng ảnh. Trả [(đường dẫn tương đối, nhãn, lớp)]."""
    rng = random.Random(seed)
    shards: list[list[tuple[str, int, str]]] = [[] for _ in range(clients)]
    for label, name in enumerate(classes):
        files = sorted(p.name for p in (root / name).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(files) < per_class * clients:
            raise ValueError(f"lớp {name} chỉ có {len(files)} ảnh, cần {per_class * clients}")
        rng.shuffle(files)
        for k in range(clients):
            for file in files[k * per_class:(k + 1) * per_class]:
                shards[k].append((f"{name}/{file}", label, name))
    return shards


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
    args = parser.parse_args()

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
    shards = pick_disjoint_shards(pv_root, classes, args.per_class, CLIENTS, SEED)
    write_partition(out, shards, classes)
    for k, shard in enumerate(shards):
        for rel, _, _ in shard:
            target = out / f"images_client_{k:02d}" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pv_root / rel, target)

    # Gói cho máy thứ hai (client 01): mã nguồn Python của src/, hợp đồng partition và ảnh của client 01
    bundle = out / "bundle_client_01.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for path in sorted((ROOT / "src").rglob("*.py")):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        tar.add(out / "partition", arcname="partition")
        tar.add(out / "images_client_01", arcname="images_client_01")
    manifest: dict[str, Any] = {
        "per_class": args.per_class, "n_k": [len(s) for s in shards], "seed": SEED,
        "bundle": {"file": bundle.name, "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)},
        "partition_config_sha256": sha256_file(out / "partition" / "partition_config.json"),
        "client_manifest_sha256": {f"client_{k:02d}": sha256_file(out / "partition" / "clients" / f"client_{k:02d}.csv") for k in range(CLIENTS)},
        "images_bytes": {f"client_{k:02d}": sum(p.stat().st_size for p in (out / f"images_client_{k:02d}").rglob("*") if p.is_file()) for k in range(CLIENTS)},
    }
    (out / "speedtest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
