"""Đo 7 checkpoint đã công bố trên các lá PlantDoc đang bị CÁCH LY (nhãn chưa chắc chắn) của release mixed_pv_pd_v3.

Câu hỏi: 15 nhóm chưa có ảnh ngoài đồng nào trong tập học (gồm 12 nhóm lá khỏe) thì mô hình hiện tại nhận ngoài đồng
đến đâu? Đây là con số đầu tiên cho các nhóm đó. Chỉ ĐỌC: không đổi release, không đổi checkpoint.

Nhãn của các lá này CHƯA được kiểm bằng mắt (mapping_status = assumed_healthy / ambiguous_alias): độ chính xác ở đây là
"khớp với nhãn ngầm định", KHÔNG phải độ chính xác thật. Mô hình đoán "sai" có thể do nhãn sai. Vì vậy báo cáo tách theo
mapping_status và không được đưa lên web cho tới khi nhãn được kiểm (bước 2 của kế hoạch).

Chạy:
    python scripts/eval_quarantined_plantdoc.py              # đo và ghi artifacts/quarantined_plantdoc_eval.json
    python scripts/eval_quarantined_plantdoc.py --limit 64   # thử nhanh (KHÔNG ghi artifacts/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

MODEL_ORDER = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
OUT_JSON = reg.QUARANTINED_EVAL_JSON


# --------------------------------------------------------------------------- phần thuần (có test)
def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lá PlantDoc bị cách ly nhưng ảnh sạch: quarantine_status == "clean". Loại hẳn nhóm xung đột nhãn và các lá
    không có nhãn trong taxonomy (class_id = -1, mapping_status = unmapped_in_classification_yaml)."""
    return [r for r in rows if r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean"
            and r.get("target_class") and int(r.get("class_id", -1)) >= 0]


def summarize(rows: list[dict[str, Any]], preds: dict[str, list[int]], classes: list[str]) -> dict[str, Any]:
    """Gom độ chính xác theo nhóm và theo mapping_status. `preds[model][i]` là lớp đoán cho rows[i]."""
    by_class: dict[str, dict[str, Any]] = {}
    by_mapping: dict[str, dict[str, Any]] = {}
    index: dict[str, list[int]] = defaultdict(list)
    mapping_index: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        index[classes[int(row["class_id"])]].append(i)
        mapping_index[str(row.get("mapping_status"))].append(i)

    def block(idx: list[int]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(idx)}
        for model in MODEL_ORDER:
            correct = sum(int(preds[model][i] == int(rows[i]["class_id"])) for i in idx)
            out[model] = {"correct": correct, "top1": round(correct / len(idx), 4) if idx else None}
        return out

    for name, idx in sorted(index.items()):
        entry = block(idx)
        for model in ("centralized", "fedavg"):
            top, count = Counter(classes[preds[model][i]] for i in idx).most_common(1)[0]
            entry[model]["most_common_prediction"] = {"class": top, "share": round(count / len(idx), 4)}
        by_class[name] = entry
    for name, idx in sorted(mapping_index.items()):
        by_mapping[name] = block(idx)
    return {"overall": block(list(range(len(rows)))), "by_class": by_class, "by_mapping_status": by_mapping}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- chạy
def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="chỉ xét N lá đầu (thử nhanh)")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    import io

    import numpy as np  # noqa: F401
    import torch
    from PIL import Image, ImageOps

    from scripts.evaluate_published_models import _load_contract_transforms, _load_model
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[quarantined] bundle chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    release = reg.abs_path(reg.RELEASE_V3_DIR)
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    manifest = release / "manifests" / "quarantine.jsonl"
    if not manifest.is_file():
        print("[quarantined] thiếu data/dataset_sources/mixed/pv_pd_v3 (nhánh dataset-sources)", file=sys.stderr)
        return 1
    classes = evidence["classes"]
    all_rows = _jsonl(manifest)
    rows = select_rows(all_rows)
    excluded_unmapped = sum(1 for r in all_rows if r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean"
                            and r not in rows)
    if args.limit:
        rows = rows[: args.limit]
    for row in rows:
        if classes[int(row["class_id"])] != row["target_class"]:
            print(f"[quarantined] nhãn lệch thứ tự taxonomy ở {row['sample_id']}", file=sys.stderr)
            return 1

    torch.set_num_threads(max(1, args.threads))
    transform = _load_contract_transforms(bundle)
    files = {
        "centralized": bundle / "centralized" / "checkpoint_best.pt",
        "fedavg": bundle / "fedavg" / "checkpoint_best.pt",
        **{f"local_{i:02d}": bundle / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)},
    }
    expected = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    for name, path in files.items():
        if sha256_bytes(path.read_bytes()) != expected[name]:
            print(f"[quarantined] checkpoint {name} sai SHA-256", file=sys.stderr)
            return 1
    models = {name: _load_model(path) for name, path in files.items()}

    def open_rgb(data: bytes):
        image = Image.open(io.BytesIO(data))
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:  # noqa: BLE001
            pass
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (0, 0, 0))
            background.paste(rgba, mask=rgba.split()[3])
            return background
        return image.convert("RGB")

    preds: dict[str, list[int]] = {name: [] for name in models}
    hash_bad = 0
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            tensors = []
            for row in chunk:
                data = (dataset_root / row["relative_path"]).read_bytes()
                if sha256_bytes(data) != row["byte_sha256"]:
                    hash_bad += 1
                tensors.append(transform(open_rgb(data)))
            batch = torch.stack(tensors)
            for name, model in models.items():
                preds[name].extend(model(batch).argmax(1).tolist())
            if (start // args.batch_size) % 10 == 0:
                print(f"[quarantined] {start + len(chunk)}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    if hash_bad:
        print(f"[quarantined] {hash_bad} ảnh sai SHA-256: dừng, không ghi báo cáo", file=sys.stderr)
        return 1

    report = {
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "checkpoint_sha256": expected,
        "source": "quarantine.jsonl của release mixed_pv_pd_v3, chỉ các lá có quarantine_status == clean",
        "images": len(rows),
        "images_verified_sha256": len(rows),
        "excluded_without_taxonomy_label": excluded_unmapped,
        "caveat": "Nhãn các lá này CHƯA được kiểm bằng mắt (assumed_healthy / ambiguous_alias): độ chính xác là 'khớp nhãn ngầm "
                  "định', không phải độ chính xác thật; mô hình 'sai' có thể do nhãn sai. Ảnh không thuộc tập học của bất kỳ "
                  "checkpoint nào đã công bố (đang bị cách ly). Chưa dùng làm bằng chứng.",
        **summarize(rows, preds, classes),
        "seconds": round(time.time() - started, 1),
    }
    if args.limit:
        print(json.dumps(report["overall"], ensure_ascii=False))
        print("[quarantined] --limit: không ghi artifacts/")
        return 0
    out = reg.abs_path(OUT_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[quarantined] đã ghi {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
