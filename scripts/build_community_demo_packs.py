"""Build six small, disjoint browser-client packs from the pinned TRAIN split.

The image bytes come from the local ``origin/dataset-sources`` Git ref. Every
blob is checked against the release manifest before it enters a pack. No image
is copied to the web server by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/dataset_sources/mixed/pv_pd_v3/manifests"
DEFAULT_OUT = ROOT / "artifacts/community_demo_packs"
CLIENT_CLASSES = (
    "Tomato___Bacterial_spot",
    "Potato___Early_blight",
    "Corn_(maize)___Common_rust_",
)
SOURCES_PER_ROUND = {"plantvillage": 10, "plantdoc": 5}
ROUNDS = (1, 2)
SEED = "community-demo-v1-seed42"
GIT_REF = "origin/dataset-sources"


def rows(name: str) -> list[dict]:
    with (MANIFESTS / name).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def selection_key(row: dict) -> tuple[str, str]:
    sample_id = row["sample_id"]
    return hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest(), sample_id


def pick_rows() -> dict[tuple[int, int], list[dict]]:
    excluded = [*rows("val.jsonl"), *rows("calibration.jsonl"), *rows("legacy_diagnostic_test.jsonl")]
    excluded_ids = {r["sample_id"] for r in excluded}
    excluded_bytes = {r["byte_sha256"] for r in excluded}
    excluded_groups = {r.get("group_id") for r in excluded if r.get("group_id")}
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows("train.jsonl"):
        if row.get("split") != "train" or row.get("supervised_eligible") is not True:
            continue
        if row["target_class"] not in CLIENT_CLASSES or row["source_domain"] not in SOURCES_PER_ROUND:
            continue
        if (row["sample_id"] in excluded_ids or row["byte_sha256"] in excluded_bytes
                or row.get("group_id") in excluded_groups):
            continue
        candidates[(row["target_class"], row["source_domain"])].append(row)

    chosen: dict[tuple[int, int], list[dict]] = {}
    used_ids: set[str] = set()
    used_bytes: set[str] = set()
    used_groups: set[str] = set()
    for client, label in enumerate(CLIENT_CLASSES):
        for source, per_round in SOURCES_PER_ROUND.items():
            available = sorted(candidates[(label, source)], key=selection_key)
            selected = []
            for row in available:
                group = row.get("group_id") or row["sample_id"]
                if row["sample_id"] in used_ids or row["byte_sha256"] in used_bytes or group in used_groups:
                    continue
                selected.append(row)
                used_ids.add(row["sample_id"])
                used_bytes.add(row["byte_sha256"])
                used_groups.add(group)
                if len(selected) == per_round * len(ROUNDS):
                    break
            if len(selected) != per_round * len(ROUNDS):
                raise ValueError(f"Not enough distinct TRAIN images for {label} / {source}")
            for index, round_number in enumerate(ROUNDS):
                chosen.setdefault((client, round_number), []).extend(
                    selected[index * per_round:(index + 1) * per_round]
                )
    assert len(used_ids) == len(ROUNDS) * len(CLIENT_CLASSES) * sum(SOURCES_PER_ROUND.values())
    return chosen


def git_blob(ref: str, relative_path: str) -> bytes:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not relative_path.startswith("mixed/"):
        raise ValueError(f"Unsafe manifest path: {relative_path}")
    return subprocess.check_output(["git", "show", f"{ref}:{relative_path}"], cwd=ROOT)


def add_zip(zipped: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zipped.writestr(info, data)


def build(output: Path, ref: str = GIT_REF) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = pick_rows()
    ref_sha = subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    summary = {"purpose": "technical_demo_only", "source_ref": ref, "source_commit": ref_sha,
               "seed": SEED, "split": "train", "rounds": {}, "total_images": 0}
    with tempfile.TemporaryDirectory(prefix="community-packs-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        for (client, round_number), batch in sorted(selected.items()):
            pack_name = f"client_{client:02d}_round_{round_number}.zip"
            label = CLIENT_CLASSES[client]
            entries = []
            with zipfile.ZipFile(temp / pack_name, "w") as zipped:
                for index, row in enumerate(sorted(batch, key=lambda r: (r["source_domain"], selection_key(r))), 1):
                    data = git_blob(ref, row["relative_path"])
                    actual_sha = hashlib.sha256(data).hexdigest()
                    if actual_sha != row["byte_sha256"]:
                        raise ValueError(f"SHA-256 mismatch: {row['sample_id']}")
                    suffix = PurePosixPath(row["relative_path"]).suffix.lower()
                    filename = f"images/{index:02d}_{row['source_domain']}_{row['sample_id']}{suffix}"
                    add_zip(zipped, filename, data)
                    entries.append({"file": filename, "sample_id": row["sample_id"], "sha256": actual_sha,
                                    "source_domain": row["source_domain"], "class": label, "group_id": row.get("group_id")})
                add_zip(zipped, "manifest.json", json.dumps({"client": client, "round": round_number,
                        "label": label, "split": "train", "images": entries}, ensure_ascii=False, indent=2).encode("utf-8"))
                add_zip(zipped, "README.txt", (f"Client {client:02d}, vòng {round_number}. Giải nén, thêm 15 ảnh trong images/ "
                        f"vào web và gán nhãn {label}. Chỉ dùng để chứng minh luồng kỹ thuật; "
                        "không dùng để báo cáo độ chính xác.\n").encode("utf-8"))
            summary["rounds"][pack_name] = {"client": client, "round": round_number, "label": label,
                "count": len(entries), "plantvillage": sum(e["source_domain"] == "plantvillage" for e in entries),
                "plantdoc": sum(e["source_domain"] == "plantdoc" for e in entries),
                "zip_sha256": hashlib.sha256((temp / pack_name).read_bytes()).hexdigest()}
            summary["total_images"] += len(entries)
        (temp / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ref", default=GIT_REF)
    args = parser.parse_args()
    result = build(args.output, args.ref)
    print(f"Built {len(result['rounds'])} verified packs / {result['total_images']} TRAIN images: {args.output}")


if __name__ == "__main__":
    main()
