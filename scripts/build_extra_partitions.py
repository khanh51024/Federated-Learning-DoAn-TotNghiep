"""Thêm cách chia 5 client cho release v4 (các mức α khác, hạt giống chia khác) mà KHÔNG đổi release_manifest.json.

Release `mixed_pv_pd_v4` chỉ có `iid` và `label_alpha_0_1`; mã băm của release được ghim khắp nơi (gói web, kiểm rò rỉ, đợt huấn luyện v4b)
nên không dựng lại. Script này chia lại CÙNG tập train, bằng cùng bộ chia theo nhóm (src/data/dirichlet_split.py) và cùng cách gom nhóm theo lớp
(scripts/build_release_v4.py::groups_by_primary_class), rồi ghi:

    <release>/partitions_extra/<cách chia>/client_00..04.jsonl
    <release>/partitions_extra/extra_manifest.json   mã băm release + mã băm từng shard + số ảnh/lớp mỗi client + kết quả kiểm toán

Tên cách chia: `label_alpha_<α>` với hạt giống chia 42 (như `label_alpha_0_1` có sẵn), `label_alpha_<α>_p<hạt giống>` với hạt giống khác.
Chạy huấn luyện bằng `scripts/train_v4.py --scheme <tên>`; `src/training/data.py::load_shard` kiểm shard theo manifest này.

    python scripts/build_extra_partitions.py                                    # 0.5, 1.0, 10.0 ở hạt giống 42 + α=0.1 ở hạt giống 123 và 2026
    python scripts/build_extra_partitions.py --specs 0.5:42 1.0:42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_release_leakage import audit_partition  # noqa: E402
from scripts.build_release_v4 import groups_by_primary_class, write_jsonl  # noqa: E402
from src import registry as reg  # noqa: E402
from src.data.dirichlet_split import partition  # noqa: E402
from src.training.data import EXTRA_DIR, EXTRA_MANIFEST, NUM_CLIENTS, load_release, sha256_file  # noqa: E402

DEFAULT_SPECS = "0.5:42,1.0:42,10.0:42,0.1:123,0.1:2026"


def scheme_name(alpha: float, seed: int) -> str:
    base = "label_alpha_" + str(alpha).replace(".", "_")
    return base if seed == 42 else f"{base}_p{seed}"


def build_scheme(train_rows: list[dict[str, Any]], release_dir: Path, alpha: float, seed: int) -> dict[str, Any]:
    scheme = scheme_name(alpha, seed)
    shards, diagnostics = partition(groups_by_primary_class(train_rows), "label_skew", NUM_CLIENTS, alpha=alpha, seed=seed)
    out_dir = release_dir / EXTRA_DIR / scheme
    shard_rows, shard_sha = {}, {}
    for client in range(NUM_CLIENTS):
        rows = sorted(shards[client], key=lambda r: r["sample_id"])
        name = f"client_{client:02d}"
        shard_sha[f"{name}.jsonl"] = write_jsonl(out_dir / f"{name}.jsonl", rows)
        shard_rows[name] = rows
    audit = audit_partition(shard_rows, {r["sample_id"] for r in train_rows})
    classes = {name: len({r["class_id"] for r in rows}) for name, rows in shard_rows.items()}
    return {
        "shard_sha256": shard_sha, "n_k": [len(shard_rows[f"client_{c:02d}"]) for c in range(NUM_CLIENTS)], "classes_with_images": classes,
        "metadata": {"skew_type": "label_dirichlet", "alpha_label": alpha, "partition_seed": seed}, "diagnostics": diagnostics, "audit": audit,
        "ok": bool(audit["disjoint"] and audit["union_equals_train"] and audit["groups_split_across_clients"] == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, default=reg.abs_path(reg.RELEASE_DIR))
    parser.add_argument("--specs", default=DEFAULT_SPECS, help="danh sách α:hạt_giống_chia, cách nhau dấu phẩy")
    args = parser.parse_args()

    release = load_release(args.release_dir, None)
    train = release["train"]
    specs = [(float(a), int(s)) for a, s in (item.split(":") for item in args.specs.split(","))]
    existing = args.release_dir / EXTRA_DIR / EXTRA_MANIFEST
    schemes: dict[str, Any] = json.loads(existing.read_text(encoding="utf-8"))["schemes"] if existing.is_file() else {}
    for alpha, seed in specs:
        built = build_scheme(train, args.release_dir, alpha, seed)
        if not built["ok"]:
            print(f"[extra] {scheme_name(alpha, seed)} không đạt kiểm toán: {built['audit']}", file=sys.stderr)
            return 2
        schemes[scheme_name(alpha, seed)] = built
        print(f"[extra] {scheme_name(alpha, seed)}: n_k {built['n_k']}, lớp mỗi client {list(built['classes_with_images'].values())}, "
              f"min_size_satisfied {built['diagnostics'].get('min_size_satisfied')}")
    manifest = {"release_name": release["meta"]["release_name"], "release_manifest_sha256": release["sha256"], "schemes": dict(sorted(schemes.items())),
                "note": "Cách chia bổ sung, ngoài release_manifest.json. Dựng bằng scripts/build_extra_partitions.py trên cùng tập train."}
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[extra] ghi {existing.relative_to(ROOT).as_posix()} (SHA-256 {sha256_file(existing)[:16]}…) cho release {release['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
