"""PC build (pc_r9/build_r4_sel_eval_pc.py from r8_draft/r4_sel_eval.py 24d44567): detector laptop_runs/detl, out r4_sel_pc_v1, 4 threads.
R4 per-leaf rule on the EXT-SEL FieldPlant images (DEPLOY_RUNS_SPEC r2 section 4), measured ONCE for description and for the
registered web-app default-mode rule. Reads EXT_SEL_MANIFEST.jsonl (never EXT_FINAL), the FP0 manifest (raw GT bbox per crop id),
the detector checkpoint_last (DET_RESULT sha, min / max size from its config) and the EXT-SEL winner classifier (EXT_SEL_RESULT.json,
sha passed in). CPU only, 2 threads, deterministic, python -I.
Pipeline per image: open -> EXIF transpose -> RGB; detector (score >= 0.40, at most 100 boxes) -> each box expanded by 8 %
(plant_data_contract crop_bounding_box, bbox_margin_8pct_v1) -> classifier (canonical_v1 eval) -> class + max prob. Image label = class of
the detection with the highest classifier probability; no detection >= 0.40 -> whole-image prediction ("fallback").
Measures (no precision, human_GT = 0): (a) detector recall on the EXT-SEL GT boxes (IoU >= 0.5; greedy by score, each prediction / GT
matched once, as det_train.recall_matches); (b) matched GT boxes: classifier correct on the detector crop vs on the GT crop
(crop_xyxy of the manifest), paired, exact McNemar; (c) image level: rule vs whole-image classification, paired, exact McNemar.
Web-app default (registered product rule): per-leaf if rule accuracy >= whole-image accuracy - 0.02, else whole-image.
Usage: python -I -B r4_sel_eval.py <EXT_SEL_RESULT.json> <its sha256>      (writes ext_test/r4_sel_v1, create-once)
Selftest: env R4_SELFTEST_MANIFEST=<jsonl outside ext_test> + R4_SELFTEST_OUT=<dir outside ext_test> (rows in the EXT schema with
crop_xyxy and an "fp0_bbox_xyxy" field; the classifier is taken from the given result file) + R4_SELFTEST_DET_DIR=<detector run dir>.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

WS = Path(r"<WORKSPACE_ROOT>")
E = WS / "experiments/train-improve-20261015"
SPLIT = E / "ext_test/split_v1"
LOCK_SHA = "15601109da68b39961386a85ee1661503bc2c9947278c0003f8dde258ad9a56f"
SEL_SHA = "9a3d0e3ff95f66fe4cc7562cd22bb647bc1462481d53e7b57606a53aa990f6f2"
FP0 = (E / "fp0/FP0_MANIFEST.json", "9d05270a68ddb8fcc417193abb88e547b3e4e1e2e2a829ddb596770f2e4023d0")
DET_DIR = E / "laptop_runs/detl/det_out/run"   # detector trained on the laptop, not retrained
REAL_OUT = E / "ext_test/r4_sel_pc_v1"
SCORE = 0.40
MARGIN = 0.08
DEFAULT_TOL = 0.02


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(pred, scores, gt):
    """det_train.recall_matches order and rule, returning the pairs {gt_index: pred_index}."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:100]
    used, pairs = set(), {}
    for p in order:
        if scores[p] < SCORE:
            continue
        ch = [(iou(pred[p], g), i) for i, g in enumerate(gt) if i not in used]
        if ch:
            ov, gi = max(ch, key=lambda v: v[0])
            if ov >= 0.5:
                used.add(gi)
                pairs[gi] = p
    return pairs


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(argv):
    st_m = os.environ.get("R4_SELFTEST_MANIFEST")
    assert len(argv) == 2, "usage: <EXT_SEL_RESULT.json> <sha256>"
    res_p, res_sha = Path(argv[0]).resolve(), argv[1]
    assert sha(res_p) == res_sha, "EXT_SEL_RESULT sha"
    sel = json.loads(res_p.read_text(encoding="utf-8"))
    if st_m:
        manifest, out = Path(st_m).resolve(), Path(os.environ["R4_SELFTEST_OUT"]).resolve()
        for p in (manifest, out):
            assert not p.is_relative_to((E / "ext_test").resolve()), ("selftest must stay outside ext_test", p)
    else:
        assert sha(SPLIT / "EXT_SPLIT_LOCK.json") == LOCK_SHA and sha(SPLIT / "EXT_SEL_MANIFEST.jsonl") == SEL_SHA
        assert sel["selftest"] is False and sel["manifest_sha256"] == SEL_SHA, "result must be the real EXT-SEL selection"
        manifest, out = SPLIT / "EXT_SEL_MANIFEST.jsonl", REAL_OUT
    assert not out.exists(), ("create-once: R4 on EXT-SEL is measured once", out)
    tmp = out.with_name(out.name + ".tmp")
    assert not tmp.exists(), ("stale tmp dir", tmp)
    winner = sel["selection"]["winner"]
    wc = sel["candidates"][winner]
    ck = (WS / wc["checkpoint"]).resolve()
    assert sha(ck) == wc["checkpoint_sha256"], "winner checkpoint sha"
    det_dir = Path(os.environ["R4_SELFTEST_DET_DIR"]) if st_m else DET_DIR
    det_res = json.loads((det_dir / "DET_RESULT.json").read_text(encoding="utf-8"))
    det_ck = det_dir / "checkpoint_last.pt"
    assert det_res["status"] == "DONE" and sha(det_ck) == det_res["checkpoint_sha256"], "detector checkpoint"

    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    fp = [r for r in rows if r["source"] == "fieldplant"]
    if not st_m:
        assert sha(FP0[0]) == FP0[1], "FP0 manifest sha"
        crops = {c["crop_id"]: c for c in json.loads(FP0[0].read_text(encoding="utf-8"))["crops"]}
        for r in fp:
            x, y, w, h = crops[r["ext_id"]]["bbox_xywh"]
            r["fp0_bbox_xyxy"] = [x, y, x + w, y + h]
            assert crops[r["ext_id"]]["crop_xyxy"] == r["crop_xyxy"]
    for r in fp:
        assert sha(WS / r["path"]) == r["byte_sha256"], ("image sha", r["ext_id"])
    images = defaultdict(list)
    for r in fp:
        images[r["path"]].append(r)

    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision.models import mobilenet_v3_small
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.transforms.functional import to_tensor
    sys.path.insert(0, str(WS / "plant_data_contract"))
    from plant_data_contract.crop_geometry import crop_bounding_box, normalize_exif_orientation
    from plant_data_contract.image_rgb import to_rgb_on_black
    from plant_data_contract.transforms import get_transforms
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tf = get_transforms("canonical_v1", is_train=False)
    dck = torch.load(det_ck, map_location="cpu", weights_only=False)
    cfg = dck["config"]
    det = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=cfg["min_size"],
                                            max_size=cfg["max_size"])
    det.load_state_dict(dck["model_state_dict"], strict=True)
    det.roi_heads.detections_per_img = 100
    det.eval()
    state = torch.load(ck, map_location="cpu", weights_only=False)
    clf = mobilenet_v3_small(weights=None)
    clf.classifier[-1] = nn.Linear(clf.classifier[-1].in_features, 38)
    clf.load_state_dict(state["model_state"], strict=True)
    clf.eval()

    def classify(img):
        with torch.no_grad():
            p = torch.softmax(clf(tf(to_rgb_on_black(img)).unsqueeze(0)), dim=1)[0]
        k = int(p.argmax())
        return k, float(p[k])

    img_rows, box_rows = [], []
    gt_total = gt_matched = 0
    for path in sorted(images):
        rs = images[path]
        classes = {int(r["class_id"]) for r in rs}
        with Image.open(WS / path) as raw:
            im = normalize_exif_orientation(raw).convert("RGB")
        w, h = im.size
        with torch.no_grad():
            o = det([to_tensor(im)])[0]
        boxes = [[float(v) for v in b] for b in o["boxes"][:100]]
        scores = [float(s) for s in o["scores"][:100]]
        keep = [i for i, s in enumerate(scores) if s >= SCORE]
        gt = [r["fp0_bbox_xyxy"] for r in rs]
        pairs = match(boxes, scores, gt)
        gt_total += len(gt)
        gt_matched += len(pairs)
        det_cls = {}
        for i in keep:
            x0, y0, x1, y1 = crop_bounding_box(w, h, boxes[i], MARGIN)
            det_cls[i] = classify(im.crop((x0, y0, x1, y1)))
        for gi, r in enumerate(rs):
            gcls = classify(im.crop(tuple(r["crop_xyxy"])))
            row = {"ext_id": r["ext_id"], "path": path, "target_id": int(r["class_id"]), "gt_crop_pred": gcls[0],
                   "matched": gi in pairs, "det_score": scores[pairs[gi]] if gi in pairs else None,
                   "det_crop_pred": det_cls[pairs[gi]][0] if gi in pairs else None}
            box_rows.append(row)
        whole = classify(im)
        if det_cls:
            best = max(det_cls, key=lambda i: (det_cls[i][1], -i))
            rule, fallback = det_cls[best][0], False
        else:
            rule, fallback = whole[0], True
        img_rows.append({"path": path, "target_ids": sorted(classes), "single_class": len(classes) == 1, "n_gt": len(gt),
                         "n_det": len(keep), "matched": len(pairs), "rule_pred": rule, "fallback": fallback, "whole_pred": whole[0]})

    m = [b for b in box_rows if b["matched"]]
    det_ok = [b["det_crop_pred"] == b["target_id"] for b in m]
    gt_ok = [b["gt_crop_pred"] == b["target_id"] for b in m]
    bb = sum(1 for d, g in zip(det_ok, gt_ok) if d and not g)
    cc = sum(1 for d, g in zip(det_ok, gt_ok) if g and not d)
    single = [r for r in img_rows if r["single_class"]]
    r_ok = [r["rule_pred"] == r["target_ids"][0] for r in single]
    w_ok = [r["whole_pred"] == r["target_ids"][0] for r in single]
    b2 = sum(1 for a, b in zip(r_ok, w_ok) if a and not b)
    c2 = sum(1 for a, b in zip(r_ok, w_ok) if b and not a)
    acc_rule = sum(r_ok) / len(single)
    acc_whole = sum(w_ok) / len(single)
    result = {"stage": "R4 per-leaf rule on EXT-SEL FieldPlant (DEPLOY_RUNS_SPEC r2 section 4)", "selftest": bool(st_m),
              "manifest_sha256": sha(manifest), "ext_sel_result_sha256": res_sha, "winner": winner,
              "classifier_checkpoint_sha256": wc["checkpoint_sha256"], "detector_checkpoint_sha256": det_res["checkpoint_sha256"],
              "detector_min_max_size": [cfg["min_size"], cfg["max_size"]], "score_threshold": SCORE, "margin": MARGIN,
              "n_images": len(img_rows), "n_single_class_images": len(single), "n_gt_boxes": gt_total,
              "a_detector_recall_iou05": gt_matched / gt_total if gt_total else None, "a_matched": gt_matched,
              "b_matched_boxes": len(m), "b_acc_detector_crop": sum(det_ok) / len(m) if m else None,
              "b_acc_gt_crop": sum(gt_ok) / len(m) if m else None, "b_mcnemar": {"det_only": bb, "gt_only": cc, "p": mcnemar(bb, cc)},
              "c_acc_rule": acc_rule, "c_acc_whole_image": acc_whole, "c_fallback_images": sum(r["fallback"] for r in single),
              "c_mcnemar": {"rule_only": b2, "whole_only": c2, "p": mcnemar(b2, c2)},
              "web_app_default": "per_leaf" if acc_rule >= acc_whole - DEFAULT_TOL else "whole_image",
              "labels": ["selection, biased (EXT-SEL)", "no precision reported (human_GT = 0, annotations not exhaustive)",
                         "model_visual_check_claude (not human QA)"]}
    tmp.mkdir(parents=True)
    for name, data in (("R4_IMAGES.csv", img_rows), ("R4_BOXES.csv", box_rows)):
        with open(tmp / name, "w", encoding="utf-8", newline="\n") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0].keys()), lineterminator="\n")
            wr.writeheader()
            wr.writerows(data)
    result["csv_sha256"] = {n: sha(tmp / n) for n in ("R4_IMAGES.csv", "R4_BOXES.csv")}
    (tmp / "R4_SEL_RESULT.json").write_bytes(json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8"))
    os.replace(tmp, out)
    print(json.dumps({k: result[k] for k in ("winner", "a_detector_recall_iou05", "b_acc_detector_crop", "b_acc_gt_crop", "c_acc_rule",
                                             "c_acc_whole_image", "web_app_default")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
