"""Thống kê cho số liệu đã công bố: khoảng tin cậy theo cụm, McNemar theo cặp, độ nhạy với "cùng lá", hiệu ứng chọn checkpoint.

Không chạy lại mô hình: dùng logits đã lưu bởi scripts/evaluate_published_models.py (data/reeval/logits.npz cho val,
data/reeval/holdout_logits.npz cho test giữ riêng). Danh tính được kiểm: fingerprint của cache phải khớp checkpoint sha +
manifest hiện tại, nếu không thì phần đó bị bỏ (ghi lý do), không đoán.

Nội dung:
- Khoảng tin cậy 95% của top-1 bằng bootstrap theo CỤM (PlantDoc: parent_scene_id — nhiều ảnh cắt từ một cảnh;
  PlantVillage: group_id — nhóm lá của release), cho 7 mô hình × {val, test} × {PlantDoc, PlantVillage}.
- So sánh theo cặp trên CÙNG ảnh: Centralized vs FedAvg, FedAvg vs Local-only tốt nhất (chọn theo val, không theo test):
  McNemar chính xác (giả định ảnh độc lập) + khoảng tin cậy bootstrap cụm cho hiệu số (không cần giả định đó).
- Độ nhạy "cùng lá" (PlantVillage): accuracy chỉ trên ảnh thuộc nhóm lá có nhiều ảnh, và sau khi bỏ ảnh không được gom
  nhóm có ảnh train mang số thứ tự liền kề cùng lớp + cùng tiền tố (dấu hiệu ảnh chụp lại cùng một lá; là ƯỚC LƯỢNG).
- Hiệu ứng chọn checkpoint: Centralized được chọn theo macro-F1 PlantDoc trên chính val -> so epoch được chọn với epoch cuối.

Chạy: python scripts/report_statistics.py   ->  artifacts/statistics_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SOURCES = ("plantdoc", "plantvillage")
MODELS = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
LOCALS = MODELS[2:]
NPZ_DIR = "data/reeval"
BOOTSTRAP = 2000
SEED = 20261011
_NAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)(?P<suffix>(?: \d+)?(?: copy(?: \d+)?)?)$")


# --------------------------------------------------------------------------- thống kê thuần numpy (có test)
def cluster_sums(clusters: list[str], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gộp theo cụm: (số ảnh mỗi cụm, tổng giá trị mỗi cụm)."""
    index: dict[str, int] = {}
    ids = np.array([index.setdefault(c, len(index)) for c in clusters], dtype=np.int64)
    n = np.bincount(ids, minlength=len(index)).astype(np.float64)
    s = np.bincount(ids, weights=values.astype(np.float64), minlength=len(index))
    return n, s


def cluster_bootstrap_ci(clusters: list[str], values: np.ndarray, rng: np.random.Generator,
                         resamples: int = BOOTSTRAP, alpha: float = 0.05) -> dict[str, Any]:
    """Khoảng tin cậy percentile của trung bình `values` khi lấy mẫu lại theo cụm (có hoàn lại)."""
    n, s = cluster_sums(clusters, values)
    k = len(n)
    estimate = float(s.sum() / n.sum()) if n.sum() else float("nan")
    if k < 2:
        return {"estimate": estimate, "low": None, "high": None, "clusters": int(k)}
    stats = np.empty(resamples)
    for start in range(0, resamples, 250):  # theo lô để không cấp phát ma trận quá lớn
        stop = min(resamples, start + 250)
        idx = rng.integers(0, k, size=(stop - start, k))
        stats[start:stop] = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": estimate, "low": float(low), "high": float(high), "clusters": int(k)}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """McNemar chính xác hai phía trên các cặp lệch: b = A đúng/B sai, c = A sai/B đúng."""
    from scipy.stats import binomtest

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def parse_pv_name(original_path: str) -> tuple[str, str, int] | None:
    """'.../<Lớp>/<uuid>___RS_HL 5751 1.JPG' -> (lớp, 'RS_HL ', 5751). None nếu không theo mẫu."""
    parts = original_path.replace("\\", "/").split("/")
    if len(parts) < 2 or "___" not in parts[-1]:
        return None
    stem = parts[-1].rsplit(".", 1)[0].split("___", 1)[1].strip()
    match = _NAME_RE.match(stem)
    if not match:
        return None
    return parts[-2], match.group("prefix"), int(match.group("number"))


def adjacent_reshoot_flags(query_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]],
                           singleton_groups: set[str]) -> np.ndarray:
    """Ảnh PlantVillage KHÔNG được gom nhóm lá mà có ảnh train cùng lớp + cùng tiền tố mang số ±1 (ước lượng chụp lại)."""
    train_keys = set()
    for row in train_rows:
        parsed = parse_pv_name(row["original_path"])
        if parsed:
            train_keys.add(parsed)
    flags = np.zeros(len(query_rows), dtype=bool)
    for i, row in enumerate(query_rows):
        if row["group_id"] not in singleton_groups:
            continue
        parsed = parse_pv_name(row["original_path"])
        if not parsed:
            continue
        cls, prefix, number = parsed
        flags[i] = (cls, prefix, number - 1) in train_keys or (cls, prefix, number + 1) in train_keys
    return flags


# --------------------------------------------------------------------------- dữ liệu
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _softmax_argmax(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=1)  # softmax không đổi argmax


def _load_split_logits(evidence: dict[str, Any], split: str, rows: list, cal_rows: list, val_rows: list) -> tuple[dict | None, str]:
    store = reg.abs_path(NPZ_DIR)
    checkpoints = {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]}
    if split == "val":
        path = store / "logits.npz"
        fingerprint = hashlib.sha256(json.dumps(
            [evidence["validation_manifest_sha256"], len(val_rows), len(cal_rows), checkpoints], sort_keys=True).encode()).hexdigest()
        prefix = "val_"
    else:
        path = store / "holdout_logits.npz"
        manifest_sha = _sha256(reg.abs_path(reg.RELEASE_DIR) / "manifests" / f"{reg.HOLDOUT_SPLIT}.jsonl")
        fingerprint = hashlib.sha256(json.dumps([manifest_sha, len(rows), checkpoints], sort_keys=True).encode()).hexdigest()
        prefix = "test_"
    if not path.is_file():
        return None, f"chưa có {NPZ_DIR}/{path.name} — chạy scripts/evaluate_published_models.py{' --holdout' if split == 'test' else ''}"
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint:
            return None, f"{path.name} không khớp checkpoint/manifest hiện tại — chạy lại scripts/evaluate_published_models.py"
        logits = {m: data[f"{prefix}{m}"] for m in MODELS}
    if any(v.shape[0] != len(rows) for v in logits.values()):
        return None, f"{path.name}: số dòng khác manifest"
    return logits, ""


def _cluster_of(row: dict[str, Any]) -> str:
    return row["parent_scene_id"] if row["source_dataset"] == "PlantDoc" else row["group_id"]


def analyze_split(rows: list[dict[str, Any]], logits: dict[str, np.ndarray], best_local: dict[str, str],
                  rng: np.random.Generator, reshoot: np.ndarray, singleton_groups: set[str]) -> dict[str, Any]:
    targets = np.array([r["class_id"] for r in rows], dtype=np.int64)
    sources = np.array([r["source_dataset"].lower() for r in rows])
    correct = {m: _softmax_argmax(logits[m]) == targets for m in MODELS}
    out: dict[str, Any] = {}
    for source in SOURCES:
        mask = sources == source
        sub_rows = [r for r, keep in zip(rows, mask) if keep]
        clusters = [_cluster_of(r) for r in sub_rows]
        block: dict[str, Any] = {"count": int(mask.sum()), "clusters": len(set(clusters)), "models": {}, "pairs": {}}
        for m in MODELS:
            ci = cluster_bootstrap_ci(clusters, correct[m][mask].astype(np.float64), rng)
            block["models"][m] = {"correct": int(correct[m][mask].sum()), **ci}
        pairs = {"centralized_vs_fedavg": ("centralized", "fedavg"), "fedavg_vs_best_local": ("fedavg", best_local[source])}
        for name, (a, b) in pairs.items():
            diff = correct[a][mask].astype(np.float64) - correct[b][mask].astype(np.float64)
            block["pairs"][name] = {
                "a": a, "b": b,
                **mcnemar_exact(correct[a][mask], correct[b][mask]),
                "difference_ci": cluster_bootstrap_ci(clusters, diff, rng),
            }
        if source == "plantvillage":
            grouped = np.array([r["group_id"] not in singleton_groups for r in sub_rows])
            flagged = reshoot[mask]
            block["same_leaf_sensitivity"] = {
                "ungrouped_images": int((~grouped).sum()),
                "ungrouped_with_adjacent_train_number": int(flagged.sum()),
                "models": {
                    m: {
                        "all": float(correct[m][mask].mean()),
                        "grouped_only": float(correct[m][mask][grouped].mean()) if grouped.any() else None,
                        "without_suspected_reshoots": float(correct[m][mask][~flagged].mean()) if (~flagged).any() else None,
                        "suspected_reshoots_only": float(correct[m][mask][flagged].mean()) if flagged.any() else None,
                    }
                    for m in MODELS
                },
            }
        out[source] = block
    return out


def _selection_block(model_id: str) -> dict[str, Any]:
    """Checkpoint của một mô hình được chọn trên chính validation: so điểm PlantDoc ở lần được chọn với lần cuối và cực đại.

    Gói cũ ghi accuracy PlantDoc theo epoch (quality.by_source); gói train_v4 chỉ ghi macro-F1 PlantDoc theo epoch/vòng nên dùng số đó (`value_kind`)."""
    from src import bundle

    directory = model_id
    summary = json.loads(reg.abs_path(f"{reg.DEMO_BUNDLE_DIR}/{directory}/summary.json").read_text(encoding="utf-8"))
    history = summary.get("history") or []
    selected = summary.get("best_epoch", summary.get("best_round"))
    points = []
    kind = None
    for entry in history:
        pd = ((entry.get("quality") or {}).get("by_source") or {}).get("plantdoc") or {}
        if pd.get("count"):
            count = pd["count"]
            value, kind_now = (count - int(pd.get("wrong_crop_count", 0)) - int(pd.get("right_crop_wrong_disease_count", 0))) / count, "accuracy"
        elif entry.get("plantdoc_macro_f1_supported") is not None:
            value, kind_now = float(entry["plantdoc_macro_f1_supported"]), "macro_f1"
        else:
            continue
        kind = kind or kind_now
        points.append({"epoch": int(entry.get("epoch", entry.get("round"))), "value": value, "is_best": bool(entry.get("is_best"))})
    if not points or selected is None:
        return {"available": False, "reason": "summary.json không có lịch sử PlantDoc theo epoch/vòng"}
    chosen = next((p for p in points if p["epoch"] == int(selected)), None)
    if chosen is None:
        return {"available": False, "reason": "epoch/vòng được chọn không nằm trong lịch sử"}
    training = (bundle.load_manifest(reg.abs_path(reg.DEMO_BUNDLE_DIR)).get("training") or {}).get(model_id) or {}
    return {
        "available": True,
        "model": model_id,
        "value_kind": kind,
        "selection_metric": str(training.get("selection") or "macro-F1 PlantDoc trên validation (trong các epoch đạt ngưỡng F1 PlantVillage)"),
        "epochs": len(points),
        "chosen_epoch": chosen["epoch"],
        "chosen_plantdoc_value": chosen["value"],
        "last_epoch": points[-1]["epoch"],
        "last_plantdoc_value": points[-1]["value"],
        "max_plantdoc_value": max(p["value"] for p in points),
        "chosen_rank_among_epochs": 1 + sum(1 for p in points if p["value"] > chosen["value"]),
    }


def selection_effect() -> dict[str, Any]:
    """Centralized (và FedAvg khi gói ghi lịch sử từng vòng): checkpoint chọn trên chính val; so với epoch cuối và cực đại."""
    central = _selection_block("centralized")
    if not central.get("available"):
        return central
    fedavg = _selection_block("fedavg")
    note = "" if fedavg.get("available") else "Bundle FedAvg không công bố lịch sử từng vòng nên không đo được hiệu ứng chọn của nó."
    return {**central, "fedavg": fedavg, "fedavg_note": note}


def build_report(rng_seed: int = SEED) -> dict[str, Any]:
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        raise RuntimeError(f"bundle không hợp lệ: {evidence.get('reason')}")
    release = reg.abs_path(reg.RELEASE_DIR)
    manifests = release / "manifests"
    if not (manifests / "val.jsonl").is_file():
        raise FileNotFoundError(f"thiếu dataset ở {reg.DATASET_SOURCES_DIR}")
    # Băm chính các manifest được đọc và đối chiếu release_manifest.json; ghi hash TÍNH ĐƯỢC vào báo cáo (web kiểm lại).
    expected_hashes = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))["manifest_hashes"]
    manifest_sha = {}
    for split in ("train", "val", "calibration", reg.HOLDOUT_SPLIT):
        manifest_sha[split] = _sha256(manifests / f"{split}.jsonl")
        if manifest_sha[split] != expected_hashes.get(split):
            raise ValueError(f"{split}.jsonl: hash khác release_manifest.json")
    if manifest_sha["val"] != evidence["validation_manifest_sha256"]:
        raise ValueError("val.jsonl không phải bộ validation mà bundle đã công bố")
    train_rows = _jsonl(manifests / "train.jsonl")
    val_rows = _jsonl(manifests / "val.jsonl")
    cal_rows = _jsonl(manifests / "calibration.jsonl")
    test_rows = _jsonl(manifests / f"{reg.HOLDOUT_SPLIT}.jsonl")

    pv_sizes = Counter(r["group_id"] for rows in (train_rows, val_rows, cal_rows, test_rows)
                       for r in rows if r["source_dataset"] == "PlantVillage")
    singleton_groups = {g for g, n in pv_sizes.items() if n == 1}
    train_pv = [r for r in train_rows if r["source_dataset"] == "PlantVillage"]

    rng = np.random.default_rng(rng_seed)
    report: dict[str, Any] = {
        "evidence": "thống kê trên logits đã lưu của lần chấm cục bộ (không chạy lại mô hình); bootstrap theo cụm, McNemar chính xác",
        "protocol_id": evidence["protocol_id"],
        "release_manifest_sha256": evidence["release_manifest_sha256"],
        "validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "checkpoint_sha256": {m["model_id"]: m["checkpoint_sha256"] for m in evidence["models"]},
        "bootstrap_resamples": BOOTSTRAP,
        "bootstrap_seed": rng_seed,
        "cluster_definition": {"plantdoc": "parent_scene_id", "plantvillage": "group_id (nhóm lá của release)"},
        "reshoot_heuristic": ("ảnh PlantVillage thuộc nhóm một-ảnh có ảnh train cùng lớp, cùng tiền tố tên, số thứ tự ±1 — "
                              "ƯỚC LƯỢNG ảnh chụp lại cùng lá, không phải phát hiện chắc chắn"),
        "splits": {},
        "selection_effect": selection_effect(),
    }
    val_logits, reason_val = _load_split_logits(evidence, "val", val_rows, cal_rows, val_rows)
    best_local: dict[str, str] = {}
    if val_logits is not None:
        val_targets = np.array([r["class_id"] for r in val_rows])
        val_sources = np.array([r["source_dataset"].lower() for r in val_rows])
        for source in SOURCES:
            mask = val_sources == source
            best_local[source] = max(LOCALS, key=lambda m: float((_softmax_argmax(val_logits[m][mask]) == val_targets[mask]).mean()))
        report["best_local_chosen_on_val"] = best_local
    for split, rows in (("val", val_rows), ("test", test_rows)):
        logits, reason = (val_logits, reason_val) if split == "val" else _load_split_logits(evidence, "test", test_rows, cal_rows, val_rows)
        if logits is None or not best_local:
            report["splits"][split] = {"available": False, "reason": reason or reason_val}
            continue
        reshoot = adjacent_reshoot_flags(rows, train_pv, singleton_groups)
        report["splits"][split] = {
            "available": True,
            "name": "validation" if split == "val" else reg.HOLDOUT_SPLIT,
            "manifest_sha256": manifest_sha["val" if split == "val" else reg.HOLDOUT_SPLIT],
            "count": len(rows),
            "used_for_checkpoint_selection": split == "val",
            "by_source": analyze_split(rows, logits, best_local, rng, reshoot, singleton_groups),
        }
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = reg.abs_path(reg.STATISTICS_JSON)
    try:
        report = build_report()
    except FileNotFoundError as exc:
        if out.is_file():
            print(f"[statistics] {exc} — giữ nguyên {reg.STATISTICS_JSON} đã sinh")
            return 0
        print(f"[statistics] {exc}", file=sys.stderr)
        return 1
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for split, block in report["splits"].items():
        if not block.get("available"):
            print(f"[statistics] {split}: bỏ qua — {block.get('reason')}")
            continue
        for source, data in block["by_source"].items():
            c, f = data["models"]["centralized"], data["models"]["fedavg"]
            pair = data["pairs"]["centralized_vs_fedavg"]
            print(f"[statistics] {split}/{source:12} n={data['count']:5} cụm={data['clusters']:5} | "
                  f"Centralized {c['estimate']:.4f} [{c['low']:.4f}, {c['high']:.4f}] | "
                  f"FedAvg {f['estimate']:.4f} [{f['low']:.4f}, {f['high']:.4f}] | McNemar p={pair['p_value']:.3g}")
    sel = report["selection_effect"]
    if sel.get("available"):
        print(f"[statistics] chọn checkpoint Centralized: epoch {sel['chosen_epoch']} PD ({sel['value_kind']}) {sel['chosen_plantdoc_value']:.4f} "
              f"(hạng {sel['chosen_rank_among_epochs']}/{sel['epochs']}), epoch cuối {sel['last_plantdoc_value']:.4f}")
    print(f"[statistics] wrote {reg.STATISTICS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
