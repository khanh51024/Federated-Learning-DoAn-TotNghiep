"""Thêm cách chia 5 client cho release v4 (các mức α khác, lệch số lượng, lệch đặc trưng) mà KHÔNG đổi release_manifest.json.

Release `mixed_pv_pd_v4` chỉ có `iid` và `label_alpha_0_1`; mã băm của release được ghim khắp nơi (gói web, kiểm rò rỉ, đợt huấn luyện v4b)
nên không dựng lại. Script này chia lại CÙNG tập train, bằng cùng cách gom nhóm theo lớp (scripts/build_release_v4.py::groups_by_primary_class), rồi ghi:

    <release>/partitions_extra/<cách chia>/client_00..04.jsonl
    <release>/partitions_extra/extra_manifest.json   mã băm release + mã băm từng shard + số ảnh/lớp mỗi client + kết quả kiểm toán

Ba dạng lệch của đề cương, mỗi dạng tách riêng để đo được ảnh hưởng của đúng dạng đó:
  label_alpha_<α>[_p<hạt giống>]   lệch nhãn: tỉ lệ lớp theo Dirichlet(α) (src/data/dirichlet_split.py). α lớn = gần đồng nhất; hạt giống chia 42 như `label_alpha_0_1`.
  quantity_alpha_<α>[_p<hạt giống>] lệch số lượng: một vectơ cỡ client q ~ Dirichlet(α) dùng chung cho mọi lớp, nên trộn lớp trong client vẫn gần toàn cục.
  source_skew[_p<hạt giống>]        lệch đặc trưng: tỉ lệ ảnh ngoài đồng khác nhau giữa các client (`FIELD_SHARE`), mô phỏng khác thiết bị và điều kiện chụp;
                                    cỡ client gần bằng nhau và trong từng miền, từng lớp ảnh chia theo cùng tỉ lệ, nên lệch nhãn và lệch số lượng ở mức thấp.

Chạy huấn luyện bằng `scripts/train_v4.py --scheme <tên>`; `src/training/data.py::load_shard` kiểm shard theo manifest này.

    python scripts/build_extra_partitions.py                                    # toàn bộ cách chia dùng trong đề tài
    python scripts/build_extra_partitions.py --specs 0.5:42,1.0:42 --quantity-specs "" --source-specs ""   # chỉ dựng hai mức α
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_release_leakage import audit_partition  # noqa: E402
from scripts.build_release_v4 import groups_by_primary_class, write_jsonl  # noqa: E402
from src import registry as reg  # noqa: E402
from src.data.dirichlet_split import partition  # noqa: E402
from src.training.data import EXTRA_DIR, EXTRA_MANIFEST, LAB_DOMAIN, NUM_CLIENTS, load_release, sha256_file  # noqa: E402

DEFAULT_SPECS = "0.5:42,1.0:42,10.0:42,100.0:42,0.1:123,0.1:2026"
# quantity_alpha = 0,5 với hạt giống 42: client lớn nhất gấp 7,7 lần client nhỏ nhất và mọi client đủ 38 lớp, nên chỉ riêng số lượng bị lệch. Chọn theo tính chất dữ liệu, trước khi huấn luyện.
DEFAULT_QUANTITY_SPECS = "0.5:42"
DEFAULT_SOURCE_SPECS = "42"
# Phần ảnh ngoài đồng của toàn hệ thống mà mỗi client giữ. Với release v4 (17.341 ảnh ngoài đồng trên 52.464 ảnh) tỉ lệ ngoài đồng trong client khoảng 0%, 8%, 25%, 50%, 83%.
FIELD_SHARE = (0.0, 0.05, 0.15, 0.30, 0.50)


def scheme_name(alpha: float, seed: int, prefix: str = "label_alpha") -> str:
    base = f"{prefix}_" + str(alpha).replace(".", "_")
    return base if seed == 42 else f"{base}_p{seed}"


def source_scheme_name(seed: int) -> str:
    return "source_skew" if seed == 42 else f"source_skew_p{seed}"


def _is_lab(group: dict[str, Any]) -> bool:
    """Nhóm thuộc miền nào theo số đông ảnh (nhóm là một lá hoặc một cảnh nên thực tế chỉ có một miền)."""
    lab = sum(item["source_domain"] == LAB_DOMAIN for item in group["items"])
    return lab * 2 >= len(group["items"])


def domain_shares(num_lab: int, num_field: int, field_share: tuple[float, ...]) -> list[float]:
    """Phần ảnh phòng thí nghiệm mỗi client giữ để cỡ client bằng nhau khi đã biết phần ảnh ngoài đồng; lỗi nếu không thể."""
    if abs(sum(field_share) - 1.0) > 1e-9:
        raise ValueError(f"field_share phải có tổng bằng 1: {field_share}")
    per_client = (num_lab + num_field) / len(field_share)
    shares = [(per_client - f * num_field) / num_lab for f in field_share]
    if min(shares) < 0:
        raise ValueError(f"client giữ quá nhiều ảnh ngoài đồng, không đủ chỗ để cỡ client bằng nhau: {shares}")
    return shares


def source_skew_shards(groups_by_class: dict[int, list[dict[str, Any]]], field_share: tuple[float, ...], seed: int) -> dict[int, list[dict[str, Any]]]:
    """Chia trọn từng nhóm ảnh theo phần đã định của từng miền. Duyệt theo lớp, nhóm đi về client đang thiếu nhiều nhất so với phần của nó, nên tỉ lệ lớp trong
    từng miền của mọi client gần như nhau: khác biệt chính giữa các client là nguồn ảnh, không phải nhãn. Phần thiếu được mang qua các lớp (`seen`, `assigned`
    tính trên cả miền): nếu tính riêng từng lớp, lớp ít nhóm luôn làm tròn về cùng một client và sai số cộng dồn, client có phần nhỏ gần như không nhận được gì."""
    groups = [g for items in groups_by_class.values() for g in items]
    num_lab = sum(len(g["items"]) for g in groups if _is_lab(g))
    num_field = sum(len(g["items"]) for g in groups) - num_lab
    lab_share = domain_shares(num_lab, num_field, field_share)
    rng = random.Random(seed)
    shards: dict[int, list[dict[str, Any]]] = {k: [] for k in range(len(field_share))}
    plans = ((True, lab_share), (False, field_share))
    assigned = {lab: [0] * len(field_share) for lab, _ in plans}
    seen = {lab: 0 for lab, _ in plans}
    for class_id in sorted(groups_by_class):
        for lab, share in plans:
            members = sorted((g for g in groups_by_class[class_id] if _is_lab(g) == lab), key=lambda g: g["group_id"])
            rng.shuffle(members)
            seen[lab] += sum(len(g["items"]) for g in members)
            candidates = [k for k, s in enumerate(share) if s > 0]   # client có phần bằng 0 không bao giờ nhận
            for group in members:
                k = max(candidates, key=lambda c: (share[c] * seen[lab] - assigned[lab][c], -c))
                shards[k].extend(group["items"])
                assigned[lab][k] += len(group["items"])
    return shards


def write_scheme(train_rows: list[dict[str, Any]], release_dir: Path, scheme: str, shards: dict[int, list[dict[str, Any]]], diagnostics: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    out_dir = release_dir / EXTRA_DIR / scheme
    shard_rows, shard_sha = {}, {}
    for client in range(NUM_CLIENTS):
        rows = sorted(shards[client], key=lambda r: r["sample_id"])
        name = f"client_{client:02d}"
        shard_sha[f"{name}.jsonl"] = write_jsonl(out_dir / f"{name}.jsonl", rows)
        shard_rows[name] = rows
    audit = audit_partition(shard_rows, {r["sample_id"] for r in train_rows})
    classes = {name: len({r["class_id"] for r in rows}) for name, rows in shard_rows.items()}
    n_k = [len(shard_rows[f"client_{c:02d}"]) for c in range(NUM_CLIENTS)]
    built = {
        "shard_sha256": shard_sha, "n_k": n_k, "classes_with_images": classes, "metadata": metadata, "diagnostics": diagnostics, "audit": audit,
        "size_ratio_max_over_min": round(max(n_k) / min(n_k), 2) if min(n_k) else None,
        "ok": bool(audit["disjoint"] and audit["union_equals_train"] and audit["groups_split_across_clients"] == 0),
    }
    if all("source_domain" in r for r in train_rows):       # đo mức lệch thật của cả ba dạng để bảng báo cáo đọc được, không suy từ tên
        built["heterogeneity"] = heterogeneity(shard_rows)
    return built


def heterogeneity(shard_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Tỉ lệ ảnh ngoài đồng từng client và khoảng cách biến thiên toàn phần (TVD) giữa phân phối lớp của client với toàn cục."""
    overall: Counter[int] = Counter()
    for rows in shard_rows.values():
        overall.update(int(r["class_id"]) for r in rows)
    total = sum(overall.values())
    field_fraction, label_tvd = {}, {}
    for name, rows in shard_rows.items():
        counts: Counter[int] = Counter(int(r["class_id"]) for r in rows)
        n = len(rows)
        field_fraction[name] = round(sum(r["source_domain"] != LAB_DOMAIN for r in rows) / n, 4) if n else None
        label_tvd[name] = round(0.5 * sum(abs(counts[c] / n - overall[c] / total) for c in overall), 4) if n else None
    return {"field_fraction": field_fraction, "label_tvd": label_tvd}


def build_scheme(train_rows: list[dict[str, Any]], release_dir: Path, alpha: float, seed: int) -> dict[str, Any]:
    shards, diagnostics = partition(groups_by_primary_class(train_rows), "label_skew", NUM_CLIENTS, alpha=alpha, seed=seed)
    return write_scheme(train_rows, release_dir, scheme_name(alpha, seed), shards, diagnostics, {"skew_type": "label_dirichlet", "alpha_label": alpha, "partition_seed": seed})


def build_quantity_scheme(train_rows: list[dict[str, Any]], release_dir: Path, quantity_alpha: float, seed: int) -> dict[str, Any]:
    shards, diagnostics = partition(groups_by_primary_class(train_rows), "quantity_skew", NUM_CLIENTS, quantity_alpha=quantity_alpha, seed=seed)
    return write_scheme(train_rows, release_dir, scheme_name(quantity_alpha, seed, "quantity_alpha"), shards, diagnostics,
                        {"skew_type": "quantity_dirichlet", "alpha_quantity": quantity_alpha, "partition_seed": seed})


def build_source_scheme(train_rows: list[dict[str, Any]], release_dir: Path, seed: int) -> dict[str, Any]:
    shards = source_skew_shards(groups_by_primary_class(train_rows), FIELD_SHARE, seed)
    return write_scheme(train_rows, release_dir, source_scheme_name(seed), shards, {"scenario": "source_skew", "field_share": list(FIELD_SHARE), "seed": seed},
                        {"skew_type": "feature_source", "field_share": list(FIELD_SHARE), "partition_seed": seed})


def parse_pairs(text: str) -> list[tuple[float, int]]:
    return [(float(a), int(s)) for a, s in (item.split(":") for item in text.split(",") if item)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, default=reg.abs_path(reg.RELEASE_DIR))
    parser.add_argument("--specs", default=DEFAULT_SPECS, help="lệch nhãn: danh sách α:hạt_giống_chia, cách nhau dấu phẩy; rỗng = bỏ qua")
    parser.add_argument("--quantity-specs", default=DEFAULT_QUANTITY_SPECS, help="lệch số lượng: danh sách α:hạt_giống_chia; rỗng = bỏ qua")
    parser.add_argument("--source-specs", default=DEFAULT_SOURCE_SPECS, help="lệch đặc trưng theo nguồn ảnh: danh sách hạt_giống_chia; rỗng = bỏ qua")
    args = parser.parse_args()

    release = load_release(args.release_dir, None)
    train = release["train"]
    existing = args.release_dir / EXTRA_DIR / EXTRA_MANIFEST
    schemes: dict[str, Any] = json.loads(existing.read_text(encoding="utf-8"))["schemes"] if existing.is_file() else {}
    jobs: list[tuple[str, Any]] = []
    jobs += [(scheme_name(a, s), lambda a=a, s=s: build_scheme(train, args.release_dir, a, s)) for a, s in parse_pairs(args.specs)]
    jobs += [(scheme_name(a, s, "quantity_alpha"), lambda a=a, s=s: build_quantity_scheme(train, args.release_dir, a, s)) for a, s in parse_pairs(args.quantity_specs)]
    jobs += [(source_scheme_name(int(s)), lambda s=s: build_source_scheme(train, args.release_dir, int(s))) for s in args.source_specs.split(",") if s]
    for name, job in jobs:
        built = job()
        if not built["ok"]:
            print(f"[extra] {name} không đạt kiểm toán: {built['audit']}", file=sys.stderr)
            return 2
        previous = schemes.get(name)
        if previous and previous["shard_sha256"] != built["shard_sha256"]:
            print(f"[extra] {name} dựng lại ra shard khác bản đã ghim trong manifest: dừng để không làm sai các run đã dùng bản cũ", file=sys.stderr)
            return 2
        schemes[name] = built
        print(f"[extra] {name}: n_k {built['n_k']} (lớn/nhỏ {built['size_ratio_max_over_min']}), lớp mỗi client {list(built['classes_with_images'].values())}, "
              f"ngoài đồng {list(built.get('heterogeneity', {}).get('field_fraction', {}).values())}, TVD nhãn {list(built.get('heterogeneity', {}).get('label_tvd', {}).values())}")
    manifest = {"release_name": release["meta"]["release_name"], "release_manifest_sha256": release["sha256"], "schemes": dict(sorted(schemes.items())),
                "note": "Cách chia bổ sung, ngoài release_manifest.json. Dựng bằng scripts/build_extra_partitions.py trên cùng tập train."}
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[extra] ghi {existing.as_posix()} (SHA-256 {sha256_file(existing)[:16]}…) cho release {release['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
