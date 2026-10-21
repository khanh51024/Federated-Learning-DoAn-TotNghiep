"""Đo cổng từ chối "ảnh không phải lá cây" cho yêu cầu nhận ảnh bất kỳ (chưa đưa vào web; không huấn luyện lớp nào).

Mô hình 38 lớp luôn trả một tên bệnh, kể cả khi ảnh là xe hay người. Cổng từ chối cho điểm "ảnh này là lá của 38 lớp" và loại ảnh dưới ngưỡng. Bốn cổng được so:
  msp_centralized / msp_fedavg   xác suất lớn nhất của chính mô hình đã công bố (hiện web chỉ gắn cờ "không chắc" khi < 50%)
  maxlogit_centralized           logit lớn nhất (không chuẩn hóa softmax)
  dino_knn_sim                   độ tương tự cosine trung bình 5 láng giềng gần nhất trong thư viện v3 (PlantVillage 40 ảnh mỗi lớp + lá PlantDoc), đặc trưng DINOv2-small
  bioclip_leaf                   xác suất BioCLIP zero-shot cho nhóm câu "ảnh lá cây" so với câu loại (quả, hoa, cây cả, người...)
Tập ĐÚNG (phải nhận): PlantWild bệnh + lá khỏe (5.045 ảnh, nguồn lạ) và lá PlantDoc kiểm tra + giữ riêng (441).
Tập SAI (phải từ chối): ảnh Wikimedia không phải lá (scripts/fetch_ood_images.py: nhóm `general`, và các ảnh Wikimedia đã bị loại ở bước kiểm nhãn) và lá của loài
ngoài 38 lớp (`other_leaf`, nhóm khó, chỉ báo cáo).
Chỉ tiêu đặt trước khi đo: ở ngưỡng chọn trên nửa "hiệu chỉnh" sao cho nhận đúng >= 95% ảnh lá, nửa "báo cáo" phải từ chối >= 90% ảnh `general`.
Ngưỡng luôn chọn trên nửa hiệu chỉnh và báo trên nửa còn lại (chia theo mã băm của ảnh); không chọn cổng hay ngưỡng theo nửa báo cáo.

Ba bước chạy trong ba môi trường (nặng nhẹ khác nhau; chạy lần lượt, không song song):
    python scripts/eval_ood_gate.py prepare                                 # .venv: lập manifest, khóa SHA
    python scripts/eval_ood_gate.py logits                                  # .venv: logit của Centralized và FedAvg trên ảnh SAI
    tmp/venv-fm/Scripts/python.exe scripts/eval_ood_gate.py features        # tmp/venv-fm: đặc trưng BioCLIP và DINOv2 của ảnh SAI
    python scripts/eval_ood_gate.py evaluate                                # .venv: ghi artifacts/ood_gate_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

OOD_DIR = "data/external/ood"
OOD_MANIFEST = f"{OOD_DIR}/manifest.json"
OOD_LOGITS = "data/reeval/ood_logits.npz"
OOD_FEATURES = "data/reeval/ood_features.npz"
SOURCES = "data/external/sources"
REVIEW_DECISIONS = "data/label_review_ext/decisions_ai.json"
REJECTED_WM_SOURCES = ("inat_wm_orange___haunglongbing__citrus_greening", "inat_wm_pepper__bell___bacterial_spot", "inat_wm_tomato___bacterial_spot")
ID_ACCEPT_TARGET = 0.95
OOD_REJECT_TARGET = 0.90
KNN_K = 5
SEED_SALT = "ood-gate-v1"


# --------------------------------------------------------------------------- phần thuần (có test)
def half_of(key: str) -> str:
    """Chia đôi ổn định theo mã băm của ảnh: 'calibration' (chọn ngưỡng) hay 'report' (báo kết quả)."""
    return "calibration" if int(hashlib.sha256(f"{SEED_SALT}|{key}".encode()).hexdigest()[:8], 16) % 2 == 0 else "report"


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(điểm của ảnh ĐÚNG > điểm của ảnh SAI), hòa tính 1/2 (thống kê Mann-Whitney)."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    scores = np.concatenate([positive, negative])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def threshold_for_accept(id_scores: np.ndarray, accept: float) -> float:
    """Ngưỡng lớn nhất sao cho tỉ lệ ảnh ĐÚNG có điểm >= ngưỡng vẫn đạt `accept` (nhận ảnh khi điểm >= ngưỡng)."""
    ordered = np.sort(id_scores)
    index = int(math.floor((1.0 - accept) * len(ordered)))
    return float(ordered[min(index, len(ordered) - 1)])


def threshold_for_reject(ood_scores: np.ndarray, reject: float) -> float:
    """Ngưỡng nhỏ nhất sao cho tỉ lệ ảnh SAI có điểm < ngưỡng đạt `reject`."""
    ordered = np.sort(ood_scores)
    index = int(math.ceil(reject * len(ordered)))
    return float(np.nextafter(ordered[min(index, len(ordered)) - 1], np.inf))


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def rate(mask: np.ndarray) -> dict[str, Any]:
    n, k = int(len(mask)), int(np.sum(mask))
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def knn_similarity(query: np.ndarray, gallery: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Cosine trung bình của k láng giềng gần nhất trong thư viện (đặc trưng đã chuẩn hóa L2)."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
    g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-12)
    out = np.empty(len(q))
    for start in range(0, len(q), 512):
        sims = q[start:start + 512] @ g.T
        out[start:start + 512] = -np.sort(-sims, axis=1)[:, :k].mean(axis=1)
    return out


def leaf_probability(image_feats: np.ndarray, text_feats: np.ndarray, n_positive: int, temperature: float = 100.0) -> np.ndarray:
    """Tổng xác suất của n_positive câu đầu (câu 'ảnh lá') trong softmax BioCLIP zero-shot trên toàn bộ câu."""
    x = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True).clip(1e-12)
    t = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True).clip(1e-12)
    logits = temperature * x @ t.T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True))[:, :n_positive].sum(axis=1)


def evaluate_gate(id_scores: np.ndarray, ood_scores: np.ndarray, id_half: np.ndarray, ood_half: np.ndarray, ood_group: np.ndarray) -> dict[str, Any]:
    """Chọn ngưỡng trên nửa hiệu chỉnh, báo trên nửa báo cáo; hai điểm làm việc: nhận 95% ảnh lá, và từ chối 90% ảnh không phải lá."""
    cal_id, cal_ood = id_scores[id_half == "calibration"], ood_scores[ood_half == "calibration"]
    rep_id, rep_ood = id_scores[id_half == "report"], ood_scores[ood_half == "report"]
    rep_group = ood_group[ood_half == "report"]
    general_cal = ood_scores[(ood_half == "calibration") & (ood_group == "general")]
    out: dict[str, Any] = {"auroc_general": round(auroc(id_scores, ood_scores[ood_group == "general"]), 4),
                           "auroc_other_leaf": round(auroc(id_scores, ood_scores[ood_group == "other_leaf"]), 4)}
    t = threshold_for_accept(cal_id, ID_ACCEPT_TARGET)
    out["at_id_accept_95"] = {
        "threshold": float(t), "id_accept": rate(rep_id >= t),
        "general_reject": rate(rep_ood[rep_group == "general"] < t), "other_leaf_reject": rate(rep_ood[rep_group == "other_leaf"] < t),
        "meets_target": bool(np.mean(rep_ood[rep_group == "general"] < t) >= OOD_REJECT_TARGET) if (rep_group == "general").any() else False,
    }
    t2 = threshold_for_reject(general_cal, OOD_REJECT_TARGET) if len(general_cal) else float("nan")
    out["at_general_reject_90"] = {
        "threshold": float(t2), "id_accept": rate(rep_id >= t2), "general_reject": rate(rep_ood[rep_group == "general"] < t2),
        "other_leaf_reject": rate(rep_ood[rep_group == "other_leaf"] < t2),
    }
    return out


# --------------------------------------------------------------------------- bước chạy
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare() -> int:
    base = reg.abs_path(OOD_DIR)
    meta = base / "metadata.jsonl"
    if not meta.is_file():
        print("[ood] thiếu metadata: chạy scripts/fetch_ood_images.py trước", file=sys.stderr)
        return 1
    items: dict[str, dict[str, Any]] = {}
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = base / r["file"]
        if not path.is_file():
            continue
        key = sha256_file(path)
        items.setdefault(key, {"key": key, "relative_path": f"{OOD_DIR}/{r['file']}".replace("data/", "", 1), "group": r["group"], "topic": r["topic"],
                               "license": r["license"], "page_url": r["page_url"], "attribution": r["attribution"]})
    decisions = json.loads(reg.abs_path(REVIEW_DECISIONS).read_text(encoding="utf-8"))["decisions"]
    sources = reg.abs_path(SOURCES)
    rejected = 0
    for source in REJECTED_WM_SOURCES:
        manifest = sources / source / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if decisions.get(r["sample_id"]) != "sai":
                continue
            path = sources / r["relative_path"]
            key = r["byte_sha256"]
            if key not in items and path.is_file():
                items[key] = {"key": key, "relative_path": f"external/sources/{r['relative_path']}", "group": "general", "topic": "wikimedia_rejected_in_label_review",
                              "license": r.get("license"), "page_url": r.get("source_url") or r.get("observation_url"), "attribution": r.get("attribution")}
                rejected += 1
    rows = sorted(items.values(), key=lambda r: r["key"])
    for r in rows:
        r["half"] = half_of(r["key"])
    out = reg.abs_path(OOD_MANIFEST)
    payload = {"evidence": "ảnh KHÔNG phải lá cây dùng để đo cổng từ chối; không bao giờ dùng để học", "samples": rows}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sha = sha256_file(out)
    counts = {g: sum(1 for r in rows if r["group"] == g) for g in ("general", "other_leaf")}
    print(f"[ood] {len(rows)} ảnh ({counts}), trong đó {rejected} ảnh Wikimedia đã loại ở bước kiểm nhãn; manifest SHA-256 {sha}")
    reg.abs_path(reg.OOD_EVAL_SET_JSON).write_text(json.dumps({"manifest_sha256": sha, "counts": counts, "total": len(rows),
                                                                      "licenses": sorted({str(r["license"]) for r in rows})}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


def load_ood() -> list[dict[str, Any]]:
    path = reg.abs_path(OOD_MANIFEST)
    locked = json.loads(reg.abs_path(reg.OOD_EVAL_SET_JSON).read_text(encoding="utf-8"))["manifest_sha256"]
    if sha256_file(path) != locked:
        raise SystemExit("manifest ảnh SAI khác mã băm đã khóa: chạy lại bước prepare")
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def run_logits(threads: int) -> int:
    import torch
    from PIL import Image

    from scripts.eval_inference_variants import view_images
    from scripts.evaluate_published_models import _load_contract_transforms, _load_model
    from src.common.image_io import to_rgb

    items = load_ood()
    torch.set_num_threads(max(1, threads))
    bundle = reg.abs_path(reg.DEMO_BUNDLE_DIR)
    transform = _load_contract_transforms(bundle)
    models = {name: _load_model(bundle / name / "checkpoint_best.pt") for name in ("centralized", "fedavg")}
    out = {name: np.zeros((len(items), 38), np.float32) for name in models}
    unreadable = []
    root = reg.abs_path("data")
    started = time.time()
    for i, r in enumerate(items):
        try:
            with Image.open(root / r["relative_path"]) as handle:
                x = transform(view_images(to_rgb(handle))["squash"]).unsqueeze(0)
        except Exception:  # noqa: BLE001
            unreadable.append(r["key"])
            continue
        with torch.inference_mode():
            for name, model in models.items():
                out[name][i] = model(x).numpy()[0]
        if (i + 1) % 100 == 0:
            print(f"[ood-logits] {i + 1}/{len(items)} ({(i + 1) / (time.time() - started):.1f} ảnh/s)", flush=True)
    np.savez_compressed(reg.abs_path(OOD_LOGITS), keys=np.array([r["key"] for r in items]), unreadable=np.array(unreadable, dtype=str), **out)
    print(f"[ood-logits] xong {len(items)} ảnh, không đọc được {len(unreadable)}")
    return 0


def run_features(threads: int) -> int:
    import open_clip
    import torch

    from scripts.eval_foundation_models import BIOCLIP, extract_features
    from scripts.filter_leaf_photos import prompts_for

    items = load_ood()
    root = reg.abs_path("data")
    feats = extract_features([root / r["relative_path"] for r in items], threads)
    pos, neg = prompts_for("plant leaf")
    model, _, _ = open_clip.create_model_and_transforms(BIOCLIP)
    model.eval()
    tokenizer = open_clip.get_tokenizer(BIOCLIP)
    with torch.inference_mode():
        text = model.encode_text(tokenizer(pos + neg)).numpy()
    np.savez_compressed(reg.abs_path(OOD_FEATURES), keys=np.array([r["key"] for r in items]), text_leaf=text, n_positive=np.array(len(pos)), **feats)
    print(f"[ood-features] {len(items)} ảnh")
    return 0


def evaluate() -> int:
    from scripts.eval_foundation_models import build_items, read_gallery_features_v3
    from scripts.eval_inference_variants import softmax

    ood = load_ood()
    classes_n = 38
    sets, gallery = build_items(0)
    order = [(name, item) for name, items in sets.items() for item in items]
    eval_x, gal_x = read_gallery_features_v3(len(order), len(gallery))
    foundation = np.load(reg.abs_path("data/reeval/foundation_features.npz"), allow_pickle=False)
    bio_eval = foundation["bioclip"][:len(order)]
    names = np.array([n for n, _ in order])
    keys = [str(i["key"]) for _, i in order]
    is_id = np.isin(names, ["ext_strict", "ext_healthy", "pd_val", "pd_test"])

    external = np.load(reg.abs_path("data/reeval/external_logits.npz"), allow_pickle=False)
    from scripts.build_external_eval_set import MANIFEST   # logit PlantWild đã lưu theo thứ tự manifest; tra bằng byte_sha256
    wild_rows = json.loads(reg.abs_path(MANIFEST).read_text(encoding="utf-8"))["samples"]
    wild_index = {r["byte_sha256"]: i for i, r in enumerate(wild_rows)}
    variant = np.load(reg.abs_path("data/reeval/variant_logits.npz"), allow_pickle=False)
    pd_keys = {"pd_val": [i["key"] for i in sets["pd_val"]], "pd_test": [i["key"] for i in sets["pd_test"]]}

    # logit của mô hình công bố trên ảnh ĐÚNG, góc nhìn 'squash' (chỉ số 0); PlantDoc: variant_logits[mô hình 0 = Centralized, 1 = FedAvg]
    id_logits = {"centralized": np.zeros((len(order), classes_n), np.float32), "fedavg": np.zeros((len(order), classes_n), np.float32)}
    pd_pos = {"pd_val": 0, "pd_test": 0}
    for row, (name, item) in enumerate(order):
        if name in ("ext_strict", "ext_healthy"):
            j = wild_index[item["key"]]
            id_logits["centralized"][row], id_logits["fedavg"][row] = external["centralized"][j, 0], external["fedavg"][j, 0]
        else:
            j = pd_pos[name]
            pd_pos[name] += 1
            id_logits["centralized"][row], id_logits["fedavg"][row] = variant[name][0, j, 0], variant[name][1, j, 0]
    assert pd_pos["pd_val"] == len(pd_keys["pd_val"]) and pd_pos["pd_test"] == len(pd_keys["pd_test"])
    # kiểm tra thứ tự: logit lưu sẵn phải khớp nhãn thật của từng ảnh (sai thứ tự thì độ chính xác rơi về mức ngẫu nhiên, ~3%)
    targets_all = np.array([i["target"] for _, i in order])
    sanity = {n: round(float((id_logits["centralized"][names == n].argmax(1) == targets_all[names == n]).mean()), 3)
              for n in ("ext_strict", "pd_val", "pd_test")}
    if min(sanity.values()) < 0.2:
        raise SystemExit(f"logit đã lưu không khớp thứ tự ảnh (độ chính xác Centralized {sanity}): chạy lại eval_external_set.py và eval_inference_variants.py")

    feats = np.load(reg.abs_path(OOD_FEATURES), allow_pickle=False)
    logits = np.load(reg.abs_path(OOD_LOGITS), allow_pickle=False)
    if list(feats["keys"]) != [r["key"] for r in ood] or list(logits["keys"]) != [r["key"] for r in ood]:
        raise SystemExit("đặc trưng hoặc logit ảnh SAI không khớp manifest: chạy lại các bước logits và features")
    n_positive = int(feats["n_positive"])
    bad = set(map(str, logits["unreadable"]))
    ok_ood = np.array([r["key"] not in bad for r in ood])
    id_half = np.array([half_of(k) for k, keep in zip(keys, is_id) if keep])
    ood_half = np.array([r["half"] for r in ood])[ok_ood]
    ood_group = np.array([r["group"] for r in ood])[ok_ood]

    def scores(kind: str) -> tuple[np.ndarray, np.ndarray]:
        if kind.startswith("msp_"):
            m = kind.split("_", 1)[1]
            return softmax(id_logits[m][is_id]).max(axis=1), softmax(logits[m][ok_ood]).max(axis=1)
        if kind == "maxlogit_centralized":
            return id_logits["centralized"][is_id].max(axis=1), logits["centralized"][ok_ood].max(axis=1)
        if kind == "dino_knn_sim":
            return knn_similarity(eval_x[is_id], gal_x), knn_similarity(feats["dinov2_small"][ok_ood], gal_x)
        if kind == "bioclip_leaf":
            text = feats["text_leaf"]
            return leaf_probability(bio_eval[is_id], text, n_positive), leaf_probability(feats["bioclip"][ok_ood], text, n_positive)
        raise KeyError(kind)

    gates: dict[str, Any] = {}
    topics = sorted({r["topic"] for r in ood})
    id_set = names[is_id]
    topic_of = np.array([r["topic"] for r in ood])[ok_ood]
    for kind in ("msp_centralized", "msp_fedavg", "maxlogit_centralized", "dino_knn_sim", "bioclip_leaf"):
        s_id, s_ood = scores(kind)
        gates[kind] = evaluate_gate(s_id, s_ood, id_half, ood_half, ood_group)
        t = gates[kind]["at_id_accept_95"]["threshold"]      # chi tiết ở điểm làm việc "nhận 95% ảnh lá", nửa báo cáo
        rep_id, rep_ood = id_half == "report", ood_half == "report"
        gates[kind]["at_id_accept_95"]["id_accept_by_set"] = {n: rate(s_id[(id_set == n) & rep_id] >= t) for n in ("ext_strict", "ext_healthy", "pd_val", "pd_test")}
        gates[kind]["at_id_accept_95"]["reject_by_topic"] = {topic: rate(s_ood[(topic_of == topic) & rep_ood] < t) for topic in topics}
    report = {
        "evidence": "ngưỡng chọn trên nửa hiệu chỉnh, báo trên nửa báo cáo (chia theo mã băm ảnh); không huấn luyện lớp nào",
        "targets": {"id_accept_min": ID_ACCEPT_TARGET, "general_reject_min": OOD_REJECT_TARGET},
        "ood_manifest_sha256": json.loads(reg.abs_path(reg.OOD_EVAL_SET_JSON).read_text(encoding="utf-8"))["manifest_sha256"],
        "id": {"images": int(is_id.sum()), "by_set": {n: int((names == n).sum()) for n in ("ext_strict", "ext_healthy", "pd_val", "pd_test")},
               "centralized_top1_sanity_check": sanity},
        "ood": {"images": int(ok_ood.sum()), "unreadable": int((~ok_ood).sum()), "general": int((ood_group == "general").sum()), "other_leaf": int((ood_group == "other_leaf").sum()),
                "topics": {t: int(sum(1 for r in ood if r["topic"] == t)) for t in topics}},
        "gates": gates,
        "note": "other_leaf (lá của loài ngoài 38 lớp) chỉ báo cáo, không có chỉ tiêu: cổng 'có phải lá không' không thể loại các ảnh này.",
    }
    out = reg.abs_path(reg.OOD_GATE_EVAL_JSON)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{'cổng':22s} {'AUROC':>6s} | nhận 95% lá: nhận / loại general / loại other_leaf | loại 90% general: nhận lá")
    for kind, g in gates.items():
        a, b = g["at_id_accept_95"], g["at_general_reject_90"]
        print(f"{kind:22s} {g['auroc_general']:6.3f} | {a['id_accept']['rate']:.3f} / {a['general_reject']['rate']:.3f} / {a['other_leaf_reject']['rate']:.3f}"
              f" | {b['id_accept']['rate']:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "logits", "features", "evaluate"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    return {"prepare": prepare, "logits": lambda: run_logits(args.threads), "features": lambda: run_features(args.threads), "evaluate": evaluate}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
