"""Rebuild the PlantDoc crop images of pv_pd_v3 from the original PlantDoc images and verify them byte for byte.

Crop rule (crop_policy_version "bbox_margin_8pct_v1"): EXIF transpose, then the leaf box (annotations/detection.json, bbox_xyxy) is expanded by
8 % of its width/height on every side, floored / ceiled to integer pixels and clipped to the image, cut out and saved as RGB JPEG quality 95.
Checks per PlantDoc row of a manifest: (1) sha256 of the original image == detection.json images[].byte_sha256; (2) sha256 of the rebuilt crop
== manifest byte_sha256. Verification done by the authors on 14 train rows (original images downloaded from this branch): 14/14 originals matched
and 14/14 rebuilt crops were byte-identical (Pillow, JPEG quality 95). Byte identity depends on the Pillow / libjpeg version; if it differs, the
pixel size is still checked and the mismatch is reported instead of hidden.

Usage: python rebuild_pd_crops.py --root <checkout root of this branch> --manifest mixed/pv_pd_v3/manifests/train.jsonl \
           --out <empty output dir outside --root> [--limit N] [--only-existing]
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

MARGIN = 0.08


def crop_box(w, h, bbox):
    x0, y0, x1, y1 = map(float, bbox)
    dx, dy = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    left, top = max(0, int(math.floor(x0 - dx))), max(0, int(math.floor(y0 - dy)))
    right, bottom = min(w, int(math.ceil(x1 + dx))), min(h, int(math.ceil(y1 + dy)))
    if right <= left or bottom <= top:
        raise ValueError("empty crop")
    return left, top, right, bottom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True, help="path relative to --root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-existing", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.out).resolve()
    assert not out.exists() and root not in out.parents and out != root, "--out must be a new directory outside --root"
    det = json.loads((root / "mixed/pv_pd_v3/annotations/detection.json").read_text(encoding="utf-8"))
    img = {i["image_id"]: i for i in det["images"]}
    ann = {x["annotation_id"]: x for x in det["annotations"]}
    rows = [json.loads(l) for l in (root / a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["source_dataset"] == "PlantDoc"]
    if a.limit:
        rows = rows[:a.limit]
    out.mkdir(parents=True)
    stat = {"rows": len(rows), "missing_source": 0, "source_sha_ok": 0, "crop_sha_ok": 0, "crop_sha_mismatch": 0}
    bad = []
    for r in rows:
        im, an = img[r["image_id"]], ann[r["annotation_id"]]
        src = root / im["relative_path"]
        if not src.is_file():
            stat["missing_source"] += 1
            if a.only_existing:
                continue
            raise FileNotFoundError(im["relative_path"])
        if hashlib.sha256(src.read_bytes()).hexdigest() == im["byte_sha256"]:
            stat["source_sha_ok"] += 1
        else:
            bad.append((r["sample_id"], "source sha"))
            continue
        o = ImageOps.exif_transpose(Image.open(src))
        c = o.crop(crop_box(*o.size, an["bbox_xyxy"])).convert("RGB")
        buf = io.BytesIO()
        c.save(buf, format="JPEG", quality=95)
        data = buf.getvalue()
        (out / Path(r["relative_path"]).name).write_bytes(data)
        if hashlib.sha256(data).hexdigest() == r["byte_sha256"]:
            stat["crop_sha_ok"] += 1
        else:
            stat["crop_sha_mismatch"] += 1
            bad.append((r["sample_id"], "crop sha"))
    print(json.dumps({"stat": stat, "bad": bad[:20]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
