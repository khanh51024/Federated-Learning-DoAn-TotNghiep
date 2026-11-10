"""Web package r4 builder (TU VAN PC request 2026-11-10). Create-once into DST (default <PC_PATH>/pc_transfer_r2/web_pc_v3_a01_s42_r4).
r4 = r3 copied byte-equal (r3 SHA256SUMS verified) + validation_metrics.json in every model dir (web schema, scripts/evaluate_published_models.py
definitions) + crop_accuracy filled in the int8 model_info.json + README section "Thay doi r4" + new SHA256SUMS + zip.
NO model is run: metrics are recomputed from the stored P6 dry2 VAL logits (p6_dryrun/logits/<name>__VAL.npz, int8 evaluated at batch 1) joined
with p6_dryrun/sets/VAL.json (targets, sources of val.jsonl, 4613 rows = PlantDoc 269 + PlantVillage 4344). The sets file is READ only, never copied.
Definitions (web scripts/evaluate_published_models.py): top1 = mean(pred == target); macro_f1_supported = sklearn macro F1 over the classes
present in the subset targets; crop_accuracy = mean(crop(target) == crop(pred)), crop = class name prefix before "___"; high-confidence wrong =
(pred != target) and max softmax(logits, T = 1) >= 0.9. No absolute C: path is written into any new file.
Usage: python -I -B build_web_pkg_r4.py [DST]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
E = Path("experiments/train-improve-20261015")
R3 = Path("<PC_PATH>/pc_transfer_r2/web_pc_v3_a01_s42_r3")
DST = Path(sys.argv[1] if len(sys.argv) > 1 else "<PC_PATH>/pc_transfer_r2/web_pc_v3_a01_s42_r4")
REL = Path("dataset/mixed/pv_pd_v3")
HC = 0.9
assert not DST.exists() and not DST.with_suffix(".zip").exists(), "target exists"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# 1) r3 intact -> copy
for line in (R3 / "SHA256SUMS.txt").read_text(encoding="utf-8").replace("\r", "").splitlines():
    h, rel = line.split("  ", 1)
    assert sha(R3 / rel) == h, ("r3 changed", rel)
SKIP = {"SHA256SUMS.txt", "README_TICH_HOP.md"}
for p in sorted(R3.rglob("*")):
    rel = p.relative_to(R3).as_posix()
    if p.is_file() and rel not in SKIP:
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, DST / rel)
        assert sha(DST / rel) == sha(p)

# 2) inputs
PL = J(E / "p6_inputs/P6_CHECKPOINTS_PC.json")
ENT = {c["name"]: c for c in PL["checkpoints"]}
VS = J(E / "p6_dryrun/sets/VAL.json")
tg = np.array(VS["targets"])
src = np.array(VS["sources"])
assert len(tg) == 4613 and int((src == "plantdoc").sum()) == 269 and int((src == "plantvillage").sum()) == 4344, "val composition"
c2i = J(REL / "taxonomy.json")["class_to_idx"]
classes = [None] * len(c2i)
for k, v in c2i.items():
    classes[v] = k
assert all(classes) and len(classes) == 38
crops = [c.split("___", 1)[0] for c in classes]
VAL_MANIFEST_SHA = sha(REL / "manifests/val.jsonl")
REL_MANIFEST_SHA = sha(REL / "manifests/release_manifest.json") if (REL / "manifests/release_manifest.json").exists() else None
DRY = J(E / "p6_dryrun/DRYRUN_RESULT.json")["dry_checks"]["val_vs_known"]

MAP = {"centralized_adamw": "pc_central_adamw_s42", "centralized_sgd": "pc_central_sgd_s42", "fedavg": "pc_fedavg_a01_s42", "fedprox": "pc_fedprox_a01_s42",
       "moon": "pc_moon_a01_s42", "scaffold": "pc_scaffold_a01_s42", "scaffold_bn": "pc_scaffold_bn_a01_s42"}
MAP.update({f"local/client_0{k}": f"pc_local_a01_s42_client_0{k}" for k in range(5)})
MAP.update({f"int8_dynamic_linear/{r}": f"pc_int8_dynamic_{r}" for r in ("centralized_adamw", "centralized_sgd", "fedavg", "scaffold")})


def f1_score(t, p, labels, average=None, zero_division=0):
    """macro F1 over `labels` = sklearn f1_score(average='macro', zero_division=0) (no sklearn in the venv): F1_c = 2tp / (2tp + fp + fn)."""
    out = []
    for c in labels:
        tp = int(((p == c) & (t == c)).sum()); fp = int(((p == c) & (t != c)).sum()); fn = int(((p != c) & (t == c)).sum())
        out.append(zero_division if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def softmax(x):
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def group(t, p, conf):
    sup = sorted(set(t.tolist()))
    return {"count": int(len(t)), "supported_classes": len(sup), "top1_accuracy": float((t == p).mean()),
            "macro_f1_supported": float(f1_score(t, p, labels=sup, average="macro", zero_division=0)),
            "crop_accuracy": float(np.mean([crops[a] == crops[b] for a, b in zip(t, p)])),
            "high_confidence_wrong_count_0_9": int(((t != p) & (conf >= HC)).sum())}


def quality(t, p, conf):
    wrong = t != p
    wc = int(sum(crops[a] != crops[b] for a, b in zip(t, p)))
    rc = int(sum(crops[a] == crops[b] for a, b in zip(t[wrong], p[wrong])))
    hcw = int((wrong & (conf >= HC)).sum())
    pcr = {}
    for i, n in enumerate(classes):
        m = t == i
        if m.any():
            pcr[n] = {"support": int(m.sum()), "recall": float((p[m] == i).mean())}
    return {"count": int(len(t)), "crop_accuracy": float(np.mean([crops[a] == crops[b] for a, b in zip(t, p)])), "wrong_crop_count": wc,
            "right_crop_wrong_disease_count": rc, "high_confidence_wrong_rate": hcw / len(t), "high_confidence_wrong_count": hcw, "per_class_recall": pcr}


def top_conf(t, p, n=10):
    cnt = {}
    for a, b in zip(t, p):
        if a != b:
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    return [{"target": classes[a], "predicted": classes[b], "count": c} for (a, b), c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


TABLE = []
for d, name in MAP.items():
    ent = ENT[name]
    ckf = DST / d / ("model_dynamic_linear_torchscript.pt" if d.startswith("int8") else "checkpoint_best.pt")
    assert sha(ckf) == ent["sha256"], ("checkpoint sha != P6 list entry", d, name)           # (a)
    lg = np.load(E / "p6_dryrun/logits" / f"{name}__VAL.npz")["logits"].astype(np.float64)
    assert lg.shape == (4613, 38)
    pred = lg.argmax(1)
    conf = softmax(lg).max(1)
    ov = group(tg, pred, conf)
    assert ov["count"] == 4613
    if name in DRY:                                                                           # (b)
        assert abs(ov["macro_f1_supported"] - DRY[name]["dry"]) <= 2e-3, (name, ov["macro_f1_supported"], DRY[name]["dry"])
    by = {s: group(tg[src == s], pred[src == s], conf[src == s]) for s in ("plantdoc", "plantvillage")}
    assert by["plantdoc"]["count"] + by["plantvillage"]["count"] == ov["count"] == 4613     # (c)
    q = {s: quality(tg[src == s], pred[src == s], conf[src == s]) for s in ("plantdoc", "plantvillage")}
    doc = {"checkpoint_sha256": ent["sha256"], "checkpoint_round_or_epoch": ent["selection"],
           "source_of_numbers": "recomputed from stored P6 dry2 VAL logits (no model run); int8 models at batch 1 (one image at a time, as the web infers)",
           "validation": {"manifest": "pv_pd_v3/manifests/val.jsonl", "manifest_sha256": VAL_MANIFEST_SHA, "overall": ov, "by_source": by,
                          "quality": {"high_confidence_threshold": HC, "by_source": q}, "top_confusions": top_conf(tg, pred)}}
    (DST / d / "validation_metrics.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if d.startswith("int8"):
        mi = J(DST / d / "model_info.json")
        mi["crop_accuracy"] = {"overall": ov["crop_accuracy"], "plantvillage": by["plantvillage"]["crop_accuracy"], "plantdoc": by["plantdoc"]["crop_accuracy"]}
        mi["crop_accuracy_note"] = "computed from P6 dry2 VAL logits (batch 1) with the web definition: crop(target) == crop(prediction), crop = class prefix before '___'"
        (DST / d / "model_info.json").write_text(json.dumps(mi, ensure_ascii=False, indent=1), encoding="utf-8")
    TABLE.append((d, ov["top1_accuracy"], by["plantvillage"]["top1_accuracy"], by["plantdoc"]["top1_accuracy"], ov["crop_accuracy"],
                  by["plantvillage"]["crop_accuracy"], by["plantdoc"]["crop_accuracy"]))
    print(f"{d:40s} top1 {ov['top1_accuracy']:.4f} PV {by['plantvillage']['top1_accuracy']:.4f} PD {by['plantdoc']['top1_accuracy']:.4f} crop {ov['crop_accuracy']:.4f}")

# 3) manifest / README / sums
man = J(R3 / "import_manifest.json")
man.update({"bundle_id": "pc_v3_a01_s42_r4", "created": "2026-11-10",
            "source": "PC campaign PLAN_PC_R9 (release v3), package r4 (= r3 + validation_metrics.json per model in the web schema + int8 crop_accuracy)"})
man["validation_metrics_note"] = "per-source numbers and crop_accuracy recomputed from stored P6 dry2 VAL logits; no model run"
(DST / "import_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
rd = (R3 / "README_TICH_HOP.md").read_text(encoding="utf-8")
add = ("# Gói mô hình PC cho web app — bản r4 (10/11/2026)\n\n## Thay đổi r4\n"
       "- Mỗi thư mục mô hình (kể cả `local/client_*` và `int8_dynamic_linear/*`) có thêm `validation_metrics.json` đúng schema của web "
       "(`validation.overall`, `by_source.plantvillage / plantdoc`, `quality.by_source`, `top_confusions`).\n"
       "- Định nghĩa (theo `scripts/evaluate_published_models.py`): `top1_accuracy` = tỉ lệ đúng; `macro_f1_supported` = macro-F1 trên các lớp có mặt trong tập con; "
       "`crop_accuracy` = tỉ lệ ảnh có **loại cây** (tiền tố tên lớp trước `___`) dự đoán đúng; `high_confidence_wrong` = sai và softmax tối đa ≥ 0,9.\n"
       "- Nguồn số: tính lại từ **logit validation đã lưu** của lần chạy thử P6 (dry2), tập `val.jsonl` 4.613 ảnh (PlantVillage 4.344 + PlantDoc 269). "
       "Không chạy lại mô hình nào. Mô hình int8 được đo ở batch 1 (từng ảnh, như web suy luận).\n"
       "- `int8_dynamic_linear/*/model_info.json`: `crop_accuracy` đã có số (trước đó `null`).\n"
       "- Đây là số **validation**, không phải test cuối. Phần dưới là README r3 (giữ nguyên).\n\n---\n\n")
(DST / "README_TICH_HOP.md").write_text(add + rd, encoding="utf-8")
files = sorted(p for p in DST.rglob("*") if p.is_file())
(DST / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(DST).as_posix()}\n" for p in files), encoding="utf-8")
bad = [p for p in DST.rglob("*") if p.is_file() and ((p.suffix.lower() in (".jpg", ".jpeg", ".npz")) or "logits" in p.name.lower() or "sets" in p.parts)]
assert not bad, bad
for p in DST.rglob("validation_metrics.json"):
    assert "<PC_PATH>/" not in p.read_text(encoding="utf-8") and "<PC_PATH>/" not in p.read_text(encoding="utf-8")
print("files", len(files) + 1)
