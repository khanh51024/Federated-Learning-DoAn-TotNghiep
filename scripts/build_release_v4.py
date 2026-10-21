"""Dựng release dữ liệu `mixed_pv_pd_v4`: v3 cộng ảnh ngoài đồng đã qua kiểm nhãn, lá PlantDoc cách ly đã được nhận và ảnh tự chụp.

Mã dựng v3 không còn trong repo, nên đây là bộ dựng MỚI, chạy trên các manifest v3 đã ghim (đọc và bổ sung, không sửa v3). Đầu ra ở
data/dataset_sources/mixed/pv_pd_v4/ có đúng cấu trúc mà runner huấn luyện đọc được (xem docs/DATASET_V4.md):
    release_manifest.json  taxonomy.json  label_mapping.csv  annotations/detection.json  audits/
    manifests/{train,val,calibration,legacy_diagnostic_test,quarantine,scene_groups}.jsonl
    partitions/{label_alpha_0_1,iid}/client_00..04.jsonl   (5 client, chia theo nhóm; bộ chia v3: src/data/dirichlet_split.py)
    partition_spec_v4.json (kiểm toán phân vùng, bộ chạy đọc), quality_audit_v4.json (cổng rò rỉ trước khi train đầy đủ, ghim SHA trong spec huấn luyện)
Ảnh ngoài đồng mới được nối cứng (hardlink) vào data/dataset_sources/external_v4/ để dùng chung một --dataset-root.

Quy tắc (đặt trước, ghi vào release_manifest.json):
  * Ảnh ngoài đồng chỉ vào release khi: nhãn `trusted`, hoặc nhóm (nguồn, lớp) được nhận ở data/label_review_ext/status.json.
  * Loại ảnh gần trùng (dHash 64 bit, Hamming <= 6, cả bản lật ngang) với: PlantVillage, cảnh và lá PlantDoc của v3, PlantWild (tập thử khóa) và phần thử ảnh tự chụp.
  * Ảnh gần trùng giữa các nguồn (Hamming <= 4) được gộp CÙNG MỘT nhóm để chia train/test cùng phía.
  * Trong CÙNG MỘT nguồn và CÙNG MỘT lớp, ảnh Hamming <= `--same-source-hamming` (mặc định 10, tính cả bản lật ngang) và ảnh chụp liên tiếp trong `--session-seconds` giây
    (mốc giờ trong tên tệp, ví dụ 20250404_154210) cũng gộp cùng nhóm: ảnh chụp liên tiếp của một lá không được rơi vào cả train lẫn test.
    Ảnh đã xóa nền (capture_note "nền đã xóa") có dHash gần như chỉ phản ánh nền đen nên KHÔNG gộp theo dHash, chỉ gộp theo phiên chụp.
    Nguồn lá đặt trên nền đồng màu (giấy trắng, giấy xanh) có dHash phản ánh nền nhiều hơn lá, nên ngưỡng 10 nối chuỗi nhiều lá khác nhau thành một nhóm
    khổng lồ (đo được: nhóm 916 ảnh ở cà chua ruộng, 294 ở cà chua Bangladesh): nguồn đó khai `same_source_hamming: 4` trong configs/external_sources.yaml.
  * Cùng MỘT lá chụp nhiều lần (xoay lá trên nền giấy, đổi góc) không nhận ra bằng dHash: ảnh cùng nguồn, cùng lớp có độ tương tự cosine của đặc trưng DINOv2
    >= `--feature-threshold` (data/external/dino_features.npz, scripts/extract_external_features.py) cũng gộp cùng nhóm. Đo được ở cà chua ruộng: một lá
    chụp xoay 5-10 lần rơi cả vào train lẫn test.
  * Cổng cân bằng split: mỗi cặp (nguồn, lớp) có >= 100 ảnh phải có test <= 35% và train >= 40%; nhóm lớn nhất > 25% cặp đó chỉ cảnh báo.
  * Hai ảnh gần trùng (Hamming <= 4, mọi nguồn) mang hai nhãn khác nhau: ít nhất một nhãn sai, bỏ CẢ HAI ảnh đó (không bỏ cả nhóm). Cổng rò rỉ: tỉ lệ ảnh có ảnh cùng nguồn, cùng lớp, Hamming <= `--same-source-hamming`
    ở split khác phải <= 2% (nhóm hóa ở trên nên thực tế bằng 0; cổng bắt lỗi dựng).
  * Chia ảnh ngoài đồng theo nhóm bằng mã băm cố định: 70% train, 10% val, 10% calibration, 10% legacy_diagnostic_test.
  * PlantVillage train giữ nguyên (mặc định); `--pv-cap N` > 0 giới hạn N ảnh mỗi lớp (chọn theo mã băm). Cân bằng miền nên làm lúc train bằng trọng số lấy mẫu.
  * Lá PlantDoc cách ly chỉ vào khi nhóm lớp được nhận ở data/label_review/status.json; giữ nguyên split dự kiến của v3.
  * Ảnh tự chụp: split theo cây (scripts/check_own_photos.py); phần thử vào legacy_diagnostic_test.
  * source_domain của mọi ảnh ngoài đồng = "plantdoc" (runner chỉ biết hai miền); nguồn thật ở `source_dataset`.

Chạy:
    python scripts/build_release_v4.py                 # dựng và kiểm rò rỉ; ghi artifacts/release_v4_report.json
    python scripts/build_release_v4.py --seed 20261022
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

RELEASE_NAME = "mixed_pv_pd_v4"
V4_DIR = "data/dataset_sources/mixed/pv_pd_v4"
EXTERNAL_LINK_DIR = "external_v4"
EXT_SOURCES = "data/external/sources"
EXT_STATUS = "data/label_review_ext/status.json"
CANDIDATE_STATUS = "data/label_review_cand/status.json"   # ảnh ứng viên nạp bởi scripts/import_field_candidates.py (nhận từng ảnh)
QUARANTINE_STATUS = "data/label_review/status.json"
QUARANTINE_IMAGE_STATUS = "data/label_review_q2/status.json"   # lá cách ly được nhận TỪNG LÁ (scripts/build_quarantine_image_review.py)
OWN_MANIFEST = "data/own_photos/manifest.jsonl"
DHASH_CACHE = "data/external/dhash_cache.npz"
DUP_HAMMING = 6
GROUP_MERGE_HAMMING = 4
FEATURE_CACHE = "data/external/dino_features.npz"
FEATURE_THRESHOLD = 0.95        # cosine DINOv2 tối thiểu để coi hai ảnh cùng nguồn, cùng lớp là cùng một lá: xem tay 6 cặp mỗi dải, >= 0,95 đều là một lá; 0,92 trở xuống bắt đầu nối lá khác nhau và dồn cả lớp vào một nhóm
SAME_SOURCE_HAMMING = 10        # ảnh cùng nguồn gần nhau tới mức này (dHash, cả bản lật) phải cùng phía train/test
INFORMATIONAL_HAMMING = 14      # chỉ báo cáo: mức gần trùng lỏng hơn
SESSION_SECONDS = 60            # ảnh cùng nguồn chụp cách nhau tối đa từng này giây tính là một phiên chụp
SESSION_MAX_SECONDS = 600       # một phiên không kéo dài quá mức này (tránh nhóm khổng lồ khi chụp liên tục cả giờ)
LEAK_GATE = 0.02                # tỉ lệ ảnh thử có ảnh học cùng nguồn trong SAME_SOURCE_HAMMING
BALANCE_MIN_IMAGES = 100        # cổng cân bằng split chỉ xét cặp (nguồn, lớp) có từng này ảnh trở lên
BALANCE_MAX_TEST = 0.35
BALANCE_MIN_TRAIN = 0.40
BALANCE_WARN_GROUP = 0.25
PARTITION_SEED = 42
JS_LABEL_SKEW_MIN = 0.2         # phân vùng lệch nhãn phải thật sự lệch (JS bit so với phân bố toàn cục)
JS_IID_MAX = 0.05
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.80), ("calibration", 0.90), ("legacy_diagnostic_test", 1.00))
SPLIT_FILES = ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine")
NUM_CLIENTS = 5


# --------------------------------------------------------------------------- phần thuần (có test)
def split_for_group(group_id: str, seed: int) -> str:
    """70/10/10/10 theo mã băm cố định của nhóm; cùng nhóm luôn cùng split."""
    h = int(hashlib.sha256(f"{seed}|{group_id}".encode()).hexdigest()[:8], 16) / 2**32
    for name, upper in SPLIT_FRACTIONS:
        if h < upper:
            return name
    return "legacy_diagnostic_test"


def min_hamming(queries: np.ndarray, refs: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Khoảng cách Hamming nhỏ nhất từ mỗi phần tử `queries` (uint64) tới tập `refs`; 64 nếu refs rỗng."""
    if len(refs) == 0:
        return np.full(len(queries), 64, dtype=np.int64)
    out = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        block = queries[start:start + chunk]
        out[start:start + chunk] = np.bitwise_count(np.bitwise_xor(block[:, None], refs[None, :])).min(axis=1)
    return out


def merge_near_duplicate_groups(hashes: np.ndarray, groups: list[str], max_hamming: int, *, sources: list[str] | None = None,
                                same_source_max_hamming: int | None = None, flipped: np.ndarray | None = None,
                                links: list[tuple[int, int]] | tuple[()] = (), classes: list[int] | None = None,
                                conflicts: set[int] | None = None, same_source_limits: np.ndarray | None = None) -> list[str]:
    """Union-find: ảnh có Hamming <= max_hamming (bất kể nguồn) gộp nhóm. Trả group_id mới (đại diện = nhóm nhỏ nhất theo thứ tự chữ).

    Tùy chọn: hai ảnh có CÙNG khóa trong `sources` (ở build là "nguồn|lớp") gộp nhóm từ Hamming <= same_source_max_hamming; nếu có `flipped` (dHash bản lật ngang)
    thì lấy khoảng cách nhỏ hơn giữa bản gốc và bản lật. `links` là các cặp chỉ số ảnh buộc phải cùng nhóm (ví dụ cùng phiên chụp).
    Nếu có `classes`: hai ảnh khác lớp KHÔNG bao giờ gộp nhóm; cặp khác lớp mà Hamming <= max_hamming là nhãn mâu thuẫn, chỉ số của cả hai được thêm vào `conflicts`.
    `same_source_limits` (mỗi ảnh một ngưỡng) thay `same_source_max_hamming`: cặp cùng khóa dùng ngưỡng nhỏ hơn của hai ảnh."""
    parent = {g: g for g in set(groups)}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            keep, drop = sorted((a, b))
            parent[drop] = keep

    src_ids = np.unique(np.array(sources), return_inverse=True)[1] if sources is not None else None
    for start in range(0, len(hashes), 256):
        block = hashes[start:start + 256]
        dist = np.bitwise_count(np.bitwise_xor(block[:, None], hashes[None, :]))
        base = dist <= max_hamming
        mask = base
        if src_ids is not None and (same_source_max_hamming is not None or same_source_limits is not None):
            same = src_ids[start:start + 256, None] == src_ids[None, :]
            if flipped is not None:
                dist = np.minimum(dist, np.bitwise_count(np.bitwise_xor(flipped[start:start + 256][:, None], hashes[None, :])))
            limit = (np.minimum(same_source_limits[start:start + 256, None], same_source_limits[None, :]) if same_source_limits is not None
                     else same_source_max_hamming)
            mask = mask | (same & (dist <= limit))
        rows, cols = np.nonzero(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i == c:
                continue
            if classes is not None and classes[i] != classes[c]:
                if conflicts is not None and base[r, c]:
                    conflicts.update((i, c))
                continue
            union(groups[i], groups[c])
    for i, j in links:
        union(groups[i], groups[j])
    return [find(g) for g in groups]


_CAPTURE_TIME = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)")


def capture_time(name: str) -> float | None:
    """Mốc chụp (giây, UTC giả định) lấy từ tên tệp dạng 20250404_154210 hoặc 20250404-154210; None nếu không có hoặc không hợp lệ."""
    m = _CAPTURE_TIME.search(name)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()), tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def session_links(sources: list[str], names: list[str], seconds: float, max_seconds: float) -> list[tuple[int, int]]:
    """Cặp chỉ số ảnh liên tiếp cùng nguồn cách nhau <= `seconds` giây, mỗi phiên dài tối đa `max_seconds` giây kể từ ảnh đầu."""
    by_source: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, (source, name) in enumerate(zip(sources, names)):
        t = capture_time(name)
        if t is not None:
            by_source[source].append((t, i))
    links: list[tuple[int, int]] = []
    for items in by_source.values():
        items.sort()
        start, previous = items[0][0], items[0]
        for t, i in items[1:]:
            if t - previous[0] <= seconds and t - start <= max_seconds:
                links.append((previous[1], i))
            else:
                start = t
            previous = (t, i)
    return links


def conflicting_groups(classes: list[int], groups: list[str]) -> set[str]:
    """Nhóm gần trùng có ảnh mang hơn một nhãn: hai ảnh gần như giống hệt mà khác nhãn thì ít nhất một nhãn sai."""
    seen: dict[str, set[int]] = defaultdict(set)
    for class_id, group in zip(classes, groups):
        seen[group].add(int(class_id))
    return {g for g, c in seen.items() if len(c) > 1}


def groups_by_primary_class(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Gom ảnh theo nhóm rồi xếp trọn mỗi nhóm vào lớp chiếm đa số (hòa thì lớp nhỏ hơn): bộ chia theo lớp không cắt một nhóm cho nhiều client."""
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        members[r["group_id"]].append(r)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for gid, items in sorted(members.items()):
        counts = Counter(int(i["class_id"]) for i in items)
        primary = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        by_class[primary].append({"group_id": gid, "items": items})
    return dict(by_class)


SCENE_SPLIT_NAME = {"train": "train", "val": "validation", "calibration": "calibration", "legacy_diagnostic_test": "test", "quarantine": "quarantine"}
SCENE_ROLE = {"train": "supervised_train", "validation": "supervised_val", "calibration": "post_selection_calibration",
              "test": "legacy_diagnostic", "quarantine": "quarantine"}


def build_scene_groups(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Một dòng mỗi group_id theo định dạng v3 (split dùng tên 'validation'/'test'). Nhóm vừa ở quarantine vừa ở split khác (một cảnh có lá được nhận,
    lá khác còn cách ly) tính theo split không phải quarantine để group_id không lặp (bộ kiểm toàn vẹn của runner yêu cầu duy nhất)."""
    chosen: dict[str, str] = {}
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("quarantine", "train", "val", "calibration", "legacy_diagnostic_test"):  # quarantine trước để bị ghi đè
        for r in rows_by_split[split]:
            chosen[r["group_id"]] = SCENE_SPLIT_NAME[split]
            members[r["group_id"]].append(r)
    out = []
    for gid in sorted(chosen):
        sources = Counter(r["source_domain"] for r in members[gid])
        out.append({"evaluation_role": SCENE_ROLE[chosen[gid]], "group_id": gid, "num_samples": len(members[gid]), "sources": dict(sources), "split": chosen[gid]})
    return out


def cap_per_class(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """Giữ tối đa `cap` dòng mỗi lớp, chọn theo mã băm cố định (không phụ thuộc thứ tự). cap <= 0: giữ hết."""
    if cap <= 0:
        return sorted(rows, key=lambda r: r["sample_id"])
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    kept: list[dict[str, Any]] = []
    for cid, items in by_class.items():
        items = sorted(items, key=lambda r: hashlib.sha256(f"{seed}|{r['sample_id']}".encode()).hexdigest())
        kept.extend(items[:cap])
    return sorted(kept, key=lambda r: r["sample_id"])


def feature_links(keys: list[str], feats: np.ndarray, valid: np.ndarray, threshold: float, chunk: int = 512) -> list[tuple[int, int]]:
    """Cặp chỉ số (i < j) có CÙNG khóa trong `keys` ("nguồn|lớp") và cosine của đặc trưng >= threshold; ảnh có valid[i] = False bị bỏ qua.

    Dùng làm `links` của merge_near_duplicate_groups để gộp các ảnh chụp lại cùng một lá. Chuẩn hóa L2 ở đây nên đặc trưng đầu vào không cần chuẩn hóa trước."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        if valid[i]:
            by_key[key].append(i)
    links: list[tuple[int, int]] = []
    for idx in by_key.values():
        if len(idx) < 2:
            continue
        sub = feats[idx].astype(np.float32)
        sub /= np.linalg.norm(sub, axis=1, keepdims=True).clip(1e-12)
        for start in range(0, len(idx), chunk):
            sim = sub[start:start + chunk] @ sub.T
            rows, cols = np.nonzero(sim >= threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if start + r < c:
                    links.append((idx[start + r], idx[c]))
    return links


def cross_split_feature_pairs(keys: list[str], feats: np.ndarray, valid: np.ndarray, splits: list[str], threshold: float) -> int:
    """Số cặp cùng khóa, cosine >= threshold nhưng ở hai split khác nhau (phải bằng 0 sau khi đã gộp nhóm)."""
    return sum(1 for i, j in feature_links(keys, feats, valid, threshold) if splits[i] != splits[j])


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {str(k): v for k, v in zip(data["keys"], data["feats"])}


def promote_quarantine(rows: list[dict[str, Any]], class_ok: set[str], sample_ok: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tách lá PlantDoc cách ly của v3 thành (được nhận, còn cách ly). Một lá được nhận khi lớp của nó được nhận cả nhóm HOẶC chính lá đó được duyệt riêng
    (`accepted_samples` của scripts/build_quarantine_image_review.py); lá giữ nguyên split và nhóm cảnh của v3, chỉ đổi cờ supervised_eligible và label_trust."""
    promoted, kept = [], []
    for r in rows:
        usable = (r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean" and r.get("target_class")
                  and int(r.get("class_id", -1)) >= 0 and (r["target_class"] in class_ok or r["sample_id"] in sample_ok)
                  and r.get("split") in ("train", "val", "calibration", "legacy_diagnostic_test"))
        if usable:
            promoted.append({**r, "supervised_eligible": True, "label_trust": "reviewed_quarantine"})
        else:
            kept.append(r)
    return promoted, kept


def split_balance(rows: list[dict[str, Any]], min_images: int = BALANCE_MIN_IMAGES, max_test: float = BALANCE_MAX_TEST,
                  min_train: float = BALANCE_MIN_TRAIN, warn_group: float = BALANCE_WARN_GROUP) -> dict[str, Any]:
    """Tỉ lệ split của từng cặp (nguồn, lớp) ngoài đồng có >= min_images ảnh. `violations`: test > max_test hoặc train < min_train (dừng dựng);
    `warnings`: một nhóm chiếm > warn_group số ảnh của cặp (có thể là phiên chụp thật). Ảnh một nhóm luôn cùng split, nên nhóm khổng lồ làm tỉ lệ lệch."""
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pairs[(r["source_dataset"], r["target_class"])].append(r)
    stats, violations, warnings = {}, [], []
    for (source, target), items in sorted(pairs.items()):
        n = len(items)
        if n < min_images:
            continue
        shares = {name: sum(1 for r in items if r["split"] == name) / n for name in ("train", "val", "calibration", "legacy_diagnostic_test")}
        biggest = Counter(r["group_id"] for r in items).most_common(1)[0][1] / n
        key = f"{source}|{target}"
        stats[key] = {"images": n, **{k: round(v, 3) for k, v in shares.items()}, "largest_group_share": round(biggest, 3)}
        if shares["legacy_diagnostic_test"] > max_test or shares["train"] < min_train:
            violations.append(key)
        elif biggest > warn_group:
            warnings.append(key)
    return {"pairs": stats, "violations": violations, "warnings": warnings}


def class_counts(rows: list[dict[str, Any]], num_classes: int = 38) -> list[int]:
    counts = [0] * num_classes
    for r in rows:
        counts[int(r["class_id"])] += 1
    return counts


def shannon_entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    return -sum(c / total * math.log2(c / total) for c in counts if c) if total else 0.0


def js_divergence_bits(p_counts: list[int], q_counts: list[int]) -> float:
    """Jensen-Shannon (log2, tối đa 1) giữa hai phân bố lớp; cùng quy ước với partition_spec_v4.json của v3."""
    p_total, q_total = sum(p_counts), sum(q_counts)
    if not p_total or not q_total:
        return 0.0
    out = 0.0
    for a, b in zip(p_counts, q_counts):
        p, q = a / p_total, b / q_total
        m = (p + q) / 2
        if p:
            out += 0.5 * p * math.log2(p / m)
        if q:
            out += 0.5 * q * math.log2(q / m)
    return out


def build_partition_spec(release_name: str, release_sha: str, train_rows: list[dict[str, Any]], schemes: dict[str, dict[str, Any]],
                         num_scene_groups: int, num_classes: int = 38) -> dict[str, Any]:
    """partition_spec_v4.json theo định dạng của v3 (bộ chạy đọc: plant_data_contract.partitions.verify_partition_audit, training-workflows/shared/data_selection.py).

    `schemes[tên] = {"shards": {client_00: rows}, "shard_sha256": {client_00: sha}, "metadata": {...}, "structural_ok": bool}`.
    status PASS khi cấu trúc đạt (rời nhau, hợp bằng train, không cắt nhóm) và độ lệch đúng loại: lệch nhãn JS >= 0,2; IID JS <= 0,05."""
    global_counts = class_counts(train_rows, num_classes)
    scenarios: dict[str, Any] = {}
    for scheme, info in schemes.items():
        stats: dict[str, Any] = {}
        for client, rows in info["shards"].items():
            counts = class_counts(rows, num_classes)
            domains = Counter(r["source_domain"] for r in rows)
            stats[client] = {
                "n_k": len(rows), "num_groups": len({r["group_id"] for r in rows}), "shard_sha256": info["shard_sha256"][client],
                "supported_classes": sum(1 for c in counts if c), "shannon_entropy": round(shannon_entropy_bits(counts), 4),
                "js_divergence_to_global": round(js_divergence_bits(counts, global_counts), 4), "source_histogram": dict(domains),
                "source_dataset_histogram": dict(Counter(r["source_dataset"] for r in rows)),
                "class_histogram": {str(i): c for i, c in enumerate(counts)},
                "plantdoc_ratio": round(domains.get("plantdoc", 0) / len(rows), 4) if rows else 0.0,
            }
        sizes = [s["n_k"] for s in stats.values()]
        mean_js = sum(s["js_divergence_to_global"] for s in stats.values()) / len(stats)
        skewed = info["metadata"]["skew_type"] == "label_dirichlet"
        ok = bool(info["structural_ok"]) and (mean_js >= JS_LABEL_SKEW_MIN if skewed else mean_js <= JS_IID_MAX)
        scenarios[scheme] = {
            "scenario": scheme, "status": "PASS" if ok else "FAIL", "seed": PARTITION_SEED, "num_clients": len(stats), "total_samples": len(train_rows),
            "min_client_samples": min(sizes), "max_client_samples": max(sizes), "mean_client_samples": round(sum(sizes) / len(sizes), 1),
            "mean_entropy": round(sum(s["shannon_entropy"] for s in stats.values()) / len(stats), 4), "mean_js_divergence": round(mean_js, 4),
            "shard_hashes": {f"{client}.jsonl": s["shard_sha256"] for client, s in stats.items()}, "client_stats": stats, "metadata": info["metadata"],
        }
    num_clients = {s["num_clients"] for s in scenarios.values()}
    return {"spec_version": "4.0.0", "release_name": release_name, "release_manifest_sha256": release_sha, "seed": PARTITION_SEED,
            "num_clients": num_clients.pop() if len(num_clients) == 1 else None, "n_train_samples": len(train_rows),
            "num_scene_groups": num_scene_groups, "scenarios": scenarios}


def min_hamming_to_other_split(direct: np.ndarray, flipped: np.ndarray, sources: list[str], splits: list[str], chunk: int = 256) -> np.ndarray:
    """Với mỗi ảnh: Hamming nhỏ nhất (bản gốc hoặc bản lật) tới ảnh CÙNG khóa `sources` ("nguồn|lớp" ở build) nằm ở split KHÁC; 64 nếu không có ảnh nào như vậy."""
    out = np.full(len(direct), 64, dtype=np.int64)
    src, spl = np.array(sources), np.array(splits)
    for source in np.unique(src):
        idx = np.nonzero(src == source)[0]
        h, f, s = direct[idx], flipped[idx], spl[idx]
        for start in range(0, len(idx), chunk):
            block = slice(start, start + chunk)
            d = np.minimum(np.bitwise_count(np.bitwise_xor(h[block][:, None], h[None, :])),
                           np.bitwise_count(np.bitwise_xor(f[block][:, None], h[None, :]))).astype(np.int64)
            out[idx[block]] = np.where(s[block][:, None] != s[None, :], d, 64).min(axis=1)
    return out


def verify_image_bytes(rows: list[dict[str, Any]], root: Path, workers: int) -> tuple[int, list[str]]:
    """Băm lại từng tệp ảnh đang nằm trong release và so với byte_sha256 của manifest. Trả (số tệp khác nhau đã kiểm, danh sách tệp lệch hoặc thiếu)."""
    unique = list({r["relative_path"]: r for r in rows}.values())

    def check(r: dict[str, Any]) -> str | None:
        try:
            return None if sha256_file(root / r["relative_path"]) == r["byte_sha256"] else r["relative_path"]
        except OSError:
            return r["relative_path"]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        bad = [name for name in pool.map(check, unique) if name]
    return len(unique), bad


def build_quality_audit(release_name: str, release_sha: str, rows_by_split: dict[str, list[dict[str, Any]]], overlaps: dict[str, dict[str, int]],
                        files_checked: int, bad_files: list[str], near_duplicate: dict[str, Any], inherited: dict[str, Any]) -> dict[str, Any]:
    """quality_audit_v4.json: cổng rò rỉ mà plant_data_contract.full_gate.verify_full_data_gate đòi trước khi train đầy đủ (ghim SHA tệp này trong spec huấn luyện)."""
    within = sum(len(rows) - len({r["byte_sha256"] for r in rows}) for rows in rows_by_split.values())
    flat = {f"{pair.replace('__', '/')}/{key}": n for pair, counts in overlaps.items() for key, n in counts.items()}
    scene_leaks = sum(n for k, n in flat.items() if k.endswith(("/group_id", "/parent_scene_id")))
    unresolved = int(near_duplicate["unresolved_duplicate_candidates"])
    ok = not bad_files and unresolved == 0 and within == 0 and not any(flat.values())
    return {
        "release_name": release_name, "release_manifest_sha256": release_sha, "status": "PASS" if ok else "FAIL",
        "current_bytes_verified": not bad_files, "active_records_verified": files_checked, "bytes_mismatches": bad_files[:10],
        "unresolved_duplicate_candidates": unresolved, "within_split_duplicate_rows": within, "confirmed_or_critical_scene_leakages": scene_leaks,
        "exact_cross_split_overlaps": flat, "near_duplicate_audit": near_duplicate, "inherited_from_v3": inherited,
        "generated_by": "scripts/build_release_v4.py",
    }


# --------------------------------------------------------------------------- I/O
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def accepted_set(path: str, key: str = "accepted") -> set[str]:
    p = reg.abs_path(path)
    return set(json.loads(p.read_text(encoding="utf-8")).get(key, [])) if p.is_file() else set()


def mirrored_hash(path: Path) -> int:
    from PIL import Image, ImageOps

    from scripts.build_external_eval_set import dhash64

    with Image.open(path) as image:
        return dhash64(ImageOps.mirror(image.convert("RGB")))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_reference_hashes(v3_rows: list[dict[str, Any]], own_test: list[dict[str, Any]], workers: int) -> tuple[np.ndarray, dict[str, int]]:
    """dHash của mọi ảnh phải được tách biệt: PlantVillage, cảnh PlantDoc, lá PlantDoc v3, PlantWild và ảnh tự chụp phần thử."""
    from PIL import Image

    from scripts.build_external_eval_set import dhash64, hash_many

    cache_path = reg.abs_path(DHASH_CACHE)
    if not cache_path.is_file():
        raise SystemExit("thiếu data/external/dhash_cache.npz: chạy scripts/build_external_eval_set.py trước")
    data = np.load(cache_path, allow_pickle=False)
    cache = {str(k): int(v) for k, v in zip(data["paths"], data["hashes"])}
    base = reg.abs_path(reg.DATASET_SOURCES_DIR)
    crop_paths = [base / r["relative_path"] for r in v3_rows if r["source_domain"] == "plantdoc"]
    hash_many(crop_paths, cache, workers)
    np.savez_compressed(cache_path, paths=np.array(list(cache)), hashes=np.array(list(cache.values()), dtype=np.uint64))
    hashes = np.array(list(cache.values()), dtype=np.uint64)
    own = []
    for r in own_test:
        with Image.open(reg.abs_path("data") / r["relative_path"]) as image:
            own.append(dhash64(image))
    if own:
        hashes = np.concatenate([hashes, np.array(own, dtype=np.uint64)])
    return hashes, {"reference_images": int(len(hashes)), "own_test_images": len(own)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pv-cap", type=int, default=0, help="giới hạn ảnh PlantVillage train mỗi lớp; 0 = giữ đủ (cân bằng miền làm lúc train)")
    parser.add_argument("--same-source-hamming", type=int, default=SAME_SOURCE_HAMMING)
    parser.add_argument("--session-seconds", type=float, default=SESSION_SECONDS)
    parser.add_argument("--feature-cache", default=FEATURE_CACHE)
    parser.add_argument("--feature-threshold", type=float, default=FEATURE_THRESHOLD)
    parser.add_argument("--no-features", action="store_true", help="không gộp theo đặc trưng DINOv2 (chỉ để thử; release thật phải gộp)")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-link", action="store_true", help="không tạo liên kết ảnh (chỉ thử dựng manifest)")
    args = parser.parse_args()

    from scripts.audit_release_leakage import audit_overlaps, audit_partition
    from src.data.dirichlet_split import partition

    v3 = reg.abs_path(reg.RELEASE_V3_DIR)
    v4 = reg.abs_path(V4_DIR)
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    classes = json.loads(reg.abs_path(reg.CLIENT_SHARD_COUNTS_JSON).read_text(encoding="utf-8"))["classes"]

    # ---- v3
    v3_rows = {name: read_jsonl(v3 / "manifests" / f"{name}.jsonl") for name in SPLIT_FILES}
    # quarantine.jsonl của v3 chứa 23 dòng lặp y hệt nhau (cùng sample_id): giữ một bản để sample_id duy nhất trong cả release
    seen_ids: set[str] = set()
    unique_quarantine = []
    for r in v3_rows["quarantine"]:
        if r["sample_id"] not in seen_ids:
            seen_ids.add(r["sample_id"])
            unique_quarantine.append(r)
    v3_rows["quarantine"] = unique_quarantine
    own_all = read_jsonl(reg.abs_path(OWN_MANIFEST))
    own_test = [r for r in own_all if r["split"] == "test"]
    own_train = [r for r in own_all if r["split"] == "train"]

    # ---- ảnh ngoài đồng sẵn sàng
    ext_ok = accepted_set(EXT_STATUS)
    ext_samples_ok = accepted_set(EXT_STATUS, "accepted_samples") | accepted_set(CANDIDATE_STATUS, "accepted_samples")
    ext_rows: list[dict[str, Any]] = []
    not_ready: Counter[str] = Counter()
    for manifest in sorted(reg.abs_path(EXT_SOURCES).glob("*/manifest.jsonl")):
        for r in read_jsonl(manifest):
            if r["label_review_status"] == "trusted" or f"{r['source_dataset']}|{r['target_class']}" in ext_ok or r["sample_id"] in ext_samples_ok:
                ext_rows.append(r)
            else:
                not_ready[f"{r['source_dataset']}|{r['target_class']}"] += 1
    print(f"[v4] ảnh ngoài đồng sẵn sàng: {len(ext_rows)}; chờ kiểm nhãn: {sum(not_ready.values())}", flush=True)

    # ---- loại trùng với các tập đã khóa
    ref_hashes, ref_info = load_reference_hashes(
        [r for name in ("train", "val", "calibration", "legacy_diagnostic_test", "quarantine") for r in v3_rows[name]], own_test, args.workers)
    ext_base = reg.abs_path(EXT_SOURCES)
    paths = [ext_base / r["relative_path"] for r in ext_rows]
    direct = np.array([int(r["dhash64"], 16) for r in ext_rows], dtype=np.uint64)
    flipped = np.array([int(r["dhash64_flip"], 16) if r.get("dhash64_flip") else mirrored_hash(p) for r, p in zip(ext_rows, paths)], dtype=np.uint64)
    d_direct, d_flip = min_hamming(direct, ref_hashes), min_hamming(flipped, ref_hashes)
    is_dup = (np.minimum(d_direct, d_flip) <= DUP_HAMMING)
    dup_by_source = Counter(r["source_dataset"] for r, d in zip(ext_rows, is_dup) if d)
    kept_rows = [r for r, d in zip(ext_rows, is_dup) if not d]
    kept_hashes, kept_flip = direct[~is_dup], flipped[~is_dup]
    print(f"[v4] loại {int(is_dup.sum())} ảnh gần trùng với tập khóa/PlantVillage: {dict(dup_by_source)}", flush=True)

    # ---- gộp nhóm gần trùng giữa các nguồn, chia split theo nhóm
    session_keys = [f"{r['source_dataset']}|{r['class_id']}" for r in kept_rows]
    no_background = [r.get("capture_note") == "nền đã xóa" for r in kept_rows]   # dHash của ảnh xóa nền chỉ phản ánh nền đen: không dùng để gộp
    dhash_keys = [f"{k}|nobg|{i}" if bg else k for i, (k, bg) in enumerate(zip(session_keys, no_background))]
    links = session_links(session_keys, [Path(r.get("original_file", "")).name for r in kept_rows], args.session_seconds, SESSION_MAX_SECONDS)
    session_link_count = len(links)
    feature_cos: dict[str, Any] = {"enabled": False}
    kept_feats = kept_valid = None
    if not args.no_features:
        cache_path = reg.abs_path(args.feature_cache)
        if not cache_path.is_file():
            raise SystemExit(f"thiếu {cache_path}: chạy tmp/venv-fm/Scripts/python.exe scripts/extract_external_features.py (hoặc --no-features để thử)")
        cache = load_feature_cache(cache_path)
        kept_valid = np.array([r["sample_id"] in cache for r in kept_rows])
        kept_feats = np.stack([cache.get(r["sample_id"], np.zeros(384, np.float16)) for r in kept_rows])
        links = links + feature_links(session_keys, kept_feats, kept_valid, args.feature_threshold)
        feature_cos = {"enabled": True, "threshold": args.feature_threshold, "images_without_features": int((~kept_valid).sum()),
                       "links": len(links) - session_link_count}
        print(f"[v4] gộp theo đặc trưng DINOv2 (cosine >= {args.feature_threshold}): {feature_cos['links']} cặp; ảnh thiếu đặc trưng {feature_cos['images_without_features']}", flush=True)
    import yaml

    source_cfg = yaml.safe_load(reg.abs_path("configs/external_sources.yaml").read_text(encoding="utf-8"))["sources"]
    limits = np.array([int(source_cfg.get(r["source_dataset"], {}).get("same_source_hamming", args.same_source_hamming)) for r in kept_rows], dtype=np.int64)
    lowered = Counter(r["source_dataset"] for r, lim in zip(kept_rows, limits) if lim != args.same_source_hamming)
    if lowered:
        print(f"[v4] ngưỡng gộp cùng nguồn hạ xuống (nền đồng màu): { {s: int(source_cfg[s]['same_source_hamming']) for s in lowered} }", flush=True)
    conflict_pairs: set[int] = set()
    merged = merge_near_duplicate_groups(kept_hashes, [r["group_id"] for r in kept_rows], GROUP_MERGE_HAMMING, sources=dhash_keys,
                                         flipped=kept_flip, links=links, same_source_limits=limits,
                                         classes=[int(r["class_id"]) for r in kept_rows], conflicts=conflict_pairs)
    # Hai ảnh gần như giống hệt nhau mà mang hai nhãn khác nhau: ít nhất một nhãn sai -> bỏ cả hai ảnh đó, không đoán nhãn nào đúng.
    # `conflicting_groups` còn lại là lưới an toàn: nhóm có hai lớp (không xảy ra khi gộp theo lớp) cũng bị bỏ hẳn.
    residual_groups = conflicting_groups([int(r["class_id"]) for r in kept_rows], merged)
    dropped = conflict_pairs | {i for i, g in enumerate(merged) if g in residual_groups}
    conflict_by_source = Counter(kept_rows[i]["source_dataset"] for i in dropped)
    print(f"[v4] bỏ {len(dropped)} ảnh có ảnh gần trùng mang nhãn khác ({len(conflict_pairs)} ảnh theo cặp, {len(residual_groups)} nhóm còn lẫn lớp): {dict(conflict_by_source)}", flush=True)
    ext_out: list[dict[str, Any]] = []
    keep_idx = [i for i in range(len(merged)) if i not in dropped]
    for i in keep_idx:
        r, group = kept_rows[i], merged[i]
        split = split_for_group(group, args.seed)
        link_rel = f"{EXTERNAL_LINK_DIR}/{r['source_dataset']}/{r['target_class']}/{r['byte_sha256'][:16]}.jpg"
        if not args.no_link:
            link_or_copy(ext_base / r["relative_path"], dataset_root / link_rel)
        ext_out.append({
            "sample_id": r["sample_id"], "relative_path": link_rel, "original_path": r.get("original_file", ""), "byte_sha256": r["byte_sha256"],
            "class_id": int(r["class_id"]), "target_class": r["target_class"], "source_dataset": r["source_dataset"], "source_domain": "plantdoc",
            "group_id": group, "parent_scene_id": group, "image_id": r["sample_id"], "annotation_id": "", "crop_policy_version": "whole_leaf_canonical",
            "mapping_status": r["mapping_status"], "quarantine_status": "clean", "supervised_eligible": True, "split": split,
            "license": r.get("license"), "label_trust": r["label_trust"], "capture_note": r.get("capture_note", ""),
        })

    # ---- lá PlantDoc cách ly đã được nhận
    q_ok = accepted_set(QUARANTINE_STATUS)
    q_samples_ok = accepted_set(QUARANTINE_IMAGE_STATUS, "accepted_samples")
    promoted, still_quarantined = promote_quarantine(v3_rows["quarantine"], q_ok, q_samples_ok)

    # ---- ảnh tự chụp
    own_rows = [{"sample_id": r["sample_id"], "relative_path": r["relative_path"], "original_path": r.get("original_file", ""), "byte_sha256": r["byte_sha256"],
                 "class_id": r["class_id"], "target_class": r["target_class"], "source_dataset": "own_photos", "source_domain": "plantdoc",
                 "group_id": r["group_id"], "parent_scene_id": r["group_id"], "image_id": r["sample_id"], "annotation_id": "",
                 "crop_policy_version": "whole_leaf_canonical", "mapping_status": "exact_match", "quarantine_status": "clean", "supervised_eligible": True,
                 "split": "train" if r["split"] == "train" else "legacy_diagnostic_test", "license": "own", "label_trust": r.get("label_trust", "self")}
                for r in own_all]
    for r in own_rows:  # ảnh tự chụp nằm ngoài dataset-root: nối cứng vào external_v4 như các nguồn khác
        link_rel = f"{EXTERNAL_LINK_DIR}/own_photos/{r['target_class']}/{r['byte_sha256'][:16]}.jpg"
        if not args.no_link:
            link_or_copy(reg.abs_path("data") / r["relative_path"], dataset_root / link_rel)
        r["relative_path"] = link_rel

    # ---- gộp theo split
    out: dict[str, list[dict[str, Any]]] = {name: list(v3_rows[name]) for name in ("train", "val", "calibration", "legacy_diagnostic_test")}
    pv_before = sum(1 for r in out["train"] if r["source_domain"] == "plantvillage")
    pv_train = [r for r in out["train"] if r["source_domain"] == "plantvillage"]
    out["train"] = [r for r in out["train"] if r["source_domain"] != "plantvillage"] + cap_per_class(pv_train, args.pv_cap, args.seed)
    for r in [*ext_out, *promoted, *own_rows]:
        out[r["split"]].append(r)
    out["quarantine"] = still_quarantined
    for name in out:
        out[name] = sorted(out[name], key=lambda r: r["sample_id"])

    # ---- kiểm rò rỉ (cùng khóa với release v3) + bốn tập thử giữ nguyên
    overlaps = audit_overlaps({k: out[k] for k in ("train", "val", "calibration", "legacy_diagnostic_test")})
    total_overlap = sum(v for pair in overlaps.values() for v in pair.values())
    if total_overlap:
        print(f"[v4] RÒ RỈ: {overlaps}", file=sys.stderr)
        return 2
    ids = Counter(r["sample_id"] for name in out for r in out[name])
    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        print(f"[v4] sample_id trùng: {dup_ids[:5]}", file=sys.stderr)
        return 2

    # ---- cổng rò rỉ cùng nguồn: ảnh gần trùng (cả bản lật) nằm ở split khác
    ext_split_of = {r["sample_id"]: r["split"] for r in ext_out}
    min_dist = min_hamming_to_other_split(kept_hashes[keep_idx], kept_flip[keep_idx], [dhash_keys[i] for i in keep_idx],
                                          [ext_split_of[kept_rows[i]["sample_id"]] for i in keep_idx])
    leak = {}
    for label, bound in ((str(args.same_source_hamming), limits[keep_idx]), (str(INFORMATIONAL_HAMMING), INFORMATIONAL_HAMMING)):
        hit = min_dist <= bound   # ngưỡng của từng ảnh (nguồn nền đồng màu dùng ngưỡng thấp hơn); mức báo cáo 14 bit dùng chung
        by_src = Counter(kept_rows[i]["source_dataset"] for i, h in zip(keep_idx, hit) if h)
        leak[f"hamming_le_{label}"] = {"images": int(hit.sum()), "fraction": round(float(hit.mean()), 4) if len(hit) else 0.0, "by_source": dict(by_src)}
    if kept_feats is not None:
        split_of_kept = [ext_split_of.get(r["sample_id"], "dropped") for r in kept_rows]
        feature_cos["cross_split_pairs"] = cross_split_feature_pairs(
            [k if s != "dropped" else f"dropped|{i}" for i, (k, s) in enumerate(zip(session_keys, split_of_kept))], kept_feats, kept_valid, split_of_kept, args.feature_threshold)
        leak["feature_cosine"] = feature_cos
        print(f"[v4] cặp ảnh cùng lá (cosine >= {args.feature_threshold}) nằm ở hai split khác nhau: {feature_cos['cross_split_pairs']}", flush=True)
        if feature_cos["cross_split_pairs"]:
            print("[v4] RÒ RỈ THEO ĐẶC TRƯNG: còn ảnh cùng lá ở hai split", file=sys.stderr)
            return 2
    unresolved = leak[f"hamming_le_{args.same_source_hamming}"]["images"]
    group_sizes = Counter(r["group_id"] for r in ext_out)
    group_stats = {"groups": len(group_sizes), "images": len(ext_out), "largest": [{"group_id": g, "images": n} for g, n in group_sizes.most_common(5)],
                   "session_links": session_link_count, "background_removed_images_without_dhash_grouping": sum(no_background[i] for i in keep_idx)}
    print(f"[v4] nhóm ngoài đồng: {group_stats['groups']} cho {group_stats['images']} ảnh; lớn nhất {group_stats['largest'][0]['images'] if group_sizes else 0}; "
          f"ảnh còn gần trùng cùng nguồn ở split khác: Hamming<={args.same_source_hamming}: {unresolved}, <={INFORMATIONAL_HAMMING}: {leak[f'hamming_le_{INFORMATIONAL_HAMMING}']['images']}", flush=True)
    if leak[f"hamming_le_{args.same_source_hamming}"]["fraction"] > LEAK_GATE:
        print(f"[v4] RÒ RỈ CÙNG NGUỒN vượt {LEAK_GATE:.0%}: {leak}", file=sys.stderr)
        return 2

    balance = split_balance(ext_out)
    print(f"[v4] cân bằng split: {len(balance['pairs'])} cặp (nguồn, lớp) >= {BALANCE_MIN_IMAGES} ảnh; vi phạm {balance['violations']}; cảnh báo nhóm lớn {balance['warnings']}", flush=True)
    if balance["violations"]:
        print(f"[v4] TỈ LỆ SPLIT LỆCH (test > {BALANCE_MAX_TEST:.0%} hoặc train < {BALANCE_MIN_TRAIN:.0%}): "
              f"{ {k: balance['pairs'][k] for k in balance['violations']} }", file=sys.stderr)
        return 2

    # ---- ghi release
    (v4 / "manifests").mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {name: write_jsonl(v4 / "manifests" / f"{name}.jsonl", out[name]) for name in SPLIT_FILES}
    scene_groups = build_scene_groups(out)
    groups = [(g["group_id"], g["split"]) for g in scene_groups]
    hashes["scene_groups"] = write_jsonl(v4 / "manifests" / "scene_groups.jsonl", scene_groups)
    for name in ("taxonomy.json", "label_mapping.csv"):
        shutil.copy2(v3 / name, v4 / name)
    (v4 / "annotations").mkdir(exist_ok=True)
    shutil.copy2(v3 / "annotations" / "detection.json", v4 / "annotations" / "detection.json")
    (v4 / "audits").mkdir(exist_ok=True)
    shutil.copy2(v3 / "audits" / "release_repair_ledger.json", v4 / "audits" / "release_repair_ledger.json")

    # ---- phân vùng 5 client trên train (theo nhóm)
    shard_hashes: dict[str, dict[str, str]] = {}
    audits: dict[str, Any] = {}
    spec_inputs: dict[str, dict[str, Any]] = {}
    train_ids = {r["sample_id"] for r in out["train"]}
    for scheme, scenario, alpha in (("label_alpha_0_1", "label_skew", 0.1), ("iid", "iid", 1.0)):
        # Bộ chia theo lớp: một nhóm có ảnh thuộc nhiều lớp (vài cảnh PlantDoc của v3) được xếp trọn vào lớp chiếm đa số để không bị chia cho nhiều client.
        groups_by_class = groups_by_primary_class(out["train"])
        shards, diag = partition(groups_by_class, scenario, NUM_CLIENTS, alpha=alpha, seed=42)
        shard_hashes[scheme] = {}
        shard_rows: dict[str, list[dict[str, Any]]] = {}
        shard_sha: dict[str, str] = {}
        for client in range(NUM_CLIENTS):
            rows = sorted(shards[client], key=lambda r: r["sample_id"])
            name = f"client_{client:02d}"
            shard_sha[name] = write_jsonl(v4 / "partitions" / scheme / f"{name}.jsonl", rows)
            shard_hashes[scheme][f"{name}.jsonl"] = shard_sha[name]  # bộ chạy so khóa với tên tệp, có đuôi .jsonl
            shard_rows[name] = rows
        audits[scheme] = {**audit_partition(shard_rows, train_ids), "diagnostics": diag,
                          "n_k": [len(shard_rows[f"client_{c:02d}"]) for c in range(NUM_CLIENTS)]}
        a = audits[scheme]
        spec_inputs[scheme] = {"shards": shard_rows, "shard_sha256": shard_sha,
                               "structural_ok": bool(a["disjoint"] and a["union_equals_train"] and a["groups_split_across_clients"] == 0),
                               "metadata": {"skew_type": "label_dirichlet", "alpha_label": alpha} if scenario == "label_skew" else {"skew_type": "iid"}}
    for scheme, a in audits.items():
        if not spec_inputs[scheme]["structural_ok"]:
            print(f"[v4] phân vùng {scheme} không đạt: {a}", file=sys.stderr)
            return 2

    v3_manifest = json.loads((v3 / "release_manifest.json").read_text(encoding="utf-8"))
    counts = {"train_samples": len(out["train"]), "validation_samples": len(out["val"]), "calibration_samples": len(out["calibration"]),
              "legacy_diagnostic_test_samples": len(out["legacy_diagnostic_test"]), "num_scene_groups": len(scene_groups),
              "detection_images": v3_manifest["counts"]["detection_images"], "detection_annotations": v3_manifest["counts"]["detection_annotations"],
              "total_classification_samples": sum(len(out[n]) for n in out)}
    by_source = {n: dict(Counter(r["source_dataset"] if r["source_domain"] == "plantdoc" and r["source_dataset"] not in ("PlantDoc", "plantdoc") else r["source_domain"]
                                 for r in out[n])) for n in ("train", "val", "calibration", "legacy_diagnostic_test")}
    manifest = {
        "release_name": RELEASE_NAME, "schema_version": "3.0.0", "num_classes": 38, "crop_policy": v3_manifest["crop_policy"],
        "preprocessing_version": v3_manifest["preprocessing_version"], "derived_from_release": v3_manifest["release_name"],
        "derived_from_release_manifest_sha256": sha256_file(v3 / "release_manifest.json"),
        "counts": {**counts, "quarantined_samples": len(out["quarantine"]), "external_images_added": len(ext_out), "own_photos": len(own_rows),
                   "plantdoc_quarantine_promoted": len(promoted), "plantvillage_train_before_cap": pv_before, "pv_cap_per_class": args.pv_cap},
        "manifest_hashes": hashes, "partition_shard_hashes": shard_hashes, "partitions_verified": sorted(audits),
        "taxonomy_sha256": sha256_file(v4 / "taxonomy.json"), "mapping_sha256": sha256_file(v4 / "label_mapping.csv"),
        "detection_json_sha256": sha256_file(v4 / "annotations" / "detection.json"), "group_graph_sha256": hashes["scene_groups"],
        "repair_ledger_sha256": sha256_file(v4 / "audits" / "release_repair_ledger.json"),
        "split_sha256": sha256_bytes(json.dumps(groups, sort_keys=True).encode()),
        "rules": {"feature_threshold": None if args.no_features else args.feature_threshold, "dedup_hamming_max": DUP_HAMMING, "group_merge_hamming_max": GROUP_MERGE_HAMMING, "same_source_hamming_max": args.same_source_hamming,
                  "session_seconds": args.session_seconds, "external_split": dict(SPLIT_FRACTIONS), "seed": args.seed,
                  "locked_tests": "PlantVillage/PlantDoc v3 (val, legacy_diagnostic_test), PlantWild, ảnh tự chụp phần thử"},
    }
    (v4 / "release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    release_sha = sha256_file(v4 / "release_manifest.json")
    spec = build_partition_spec(RELEASE_NAME, release_sha, out["train"], spec_inputs, len(scene_groups))
    (v4 / "partition_spec_v4.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    failed_scenarios = [name for name, s in spec["scenarios"].items() if s["status"] != "PASS"]
    if failed_scenarios:
        print(f"[v4] phân vùng không đạt tiêu chí độ lệch: { {n: spec['scenarios'][n]['mean_js_divergence'] for n in failed_scenarios} }", file=sys.stderr)
    all_rows = [r for name in ("train", "val", "calibration", "legacy_diagnostic_test") for r in out[name]]
    print(f"[v4] băm lại {len({r['relative_path'] for r in all_rows})} tệp ảnh để kiểm byte_sha256 ...", flush=True)
    files_checked, bad_files = verify_image_bytes(all_rows, dataset_root, args.workers)
    audit = build_quality_audit(
        RELEASE_NAME, release_sha, {name: out[name] for name in ("train", "val", "calibration", "legacy_diagnostic_test")}, overlaps, files_checked, bad_files,
        {"same_source_hamming": args.same_source_hamming, "session_seconds": args.session_seconds, "unresolved_duplicate_candidates": unresolved,
         "cross_split_same_source": leak, "groups": group_stats, "dropped_label_conflict_images": len(dropped)},
        {"release_manifest_sha256": manifest["derived_from_release_manifest_sha256"],
         "quality_audit_sha256": sha256_file(v3 / "quality_audit_v3.json") if (v3 / "quality_audit_v3.json").is_file() else None})
    (v4 / "quality_audit_v4.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (v4 / "dataset_card.md").write_text(
        "# Mixed PlantVillage–PlantDoc v4\n\nv3 cộng ảnh ngoài đồng đã kiểm nhãn, lá PlantDoc cách ly đã được nhận và ảnh tự chụp. "
        "Xem docs/DATASET_V4.md. Ảnh ngoài đồng nối cứng vào external_v4/; source_domain=plantdoc nghĩa là \"ngoài đồng\".\n", encoding="utf-8")
    report = {
        "release": RELEASE_NAME, "release_manifest_sha256": release_sha, "counts": manifest["counts"],
        "partition_spec_sha256": sha256_file(v4 / "partition_spec_v4.json"), "quality_audit_sha256": sha256_file(v4 / "quality_audit_v4.json"),
        "quality_audit_status": audit["status"], "partition_status": {n: s["status"] for n, s in spec["scenarios"].items()},
        "same_source_leakage": leak, "ext_groups": group_stats, "split_balance": balance,
        "reference": ref_info, "removed_near_duplicates": {"total": int(is_dup.sum()), "by_source": dict(dup_by_source)},
        "dropped_label_conflict": {"images": len(dropped), "images_in_conflicting_pairs": len(conflict_pairs), "residual_mixed_class_groups": len(residual_groups),
                                   "by_source": dict(conflict_by_source)},
        "waiting_label_review": dict(not_ready), "overlaps": overlaps, "partitions": audits, "by_source": by_source,
        "by_class": {c: {n: sum(1 for r in out[n] if r["target_class"] == c) for n in ("train", "val", "calibration", "legacy_diagnostic_test")} for c in classes},
    }
    reg.abs_path(reg.RELEASE_V4_REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[v4] xong: {counts}; ảnh ngoài đồng thêm {len(ext_out)}, lá PlantDoc cách ly nhận lại {len(promoted)}, ảnh tự chụp {len(own_rows)}")
    print(f"[v4] quality_audit: {audit['status']} (byte lệch {len(bad_files)}, within_split_dup {audit['within_split_duplicate_rows']}, unresolved {unresolved}); "
          f"partition: { {n: s['status'] for n, s in spec['scenarios'].items()} }; SHA release {release_sha}")
    return 0 if audit["status"] == "PASS" and not failed_scenarios else 2


if __name__ == "__main__":
    raise SystemExit(main())
