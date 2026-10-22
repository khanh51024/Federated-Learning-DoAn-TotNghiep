"""Kiểm mô hình của gói đang dùng NGAY TRÊN WEB: gửi ảnh thật qua POST /api/inference/all rồi đối chiếu với tính toán offline.

Không ghi trạng thái nào trên máy chủ (suy luận chỉ tính rồi trả lời); gửi tuần tự từng ảnh để không làm nặng máy chủ.

Kiểm:
  1. Trùng khớp: dự đoán của web = argmax logits tính offline cùng đường tiền xử lý (ảnh val lấy từ logits đã lưu lúc nhập gói,
     ảnh PlantWild và ảnh mẫu tính lại bằng scripts/evaluate_published_models.collect_logits). Kỳ vọng ~100%.
  2. Độ chính xác qua web theo nhóm nguồn (PlantVillage, PlantDoc, ngoài đồng khác, PlantWild nguồn lạ) và theo mô hình.
  3. Lọc loại cây (`crop=`): mọi dự đoán phải thuộc đúng loại cây; so độ chính xác khi có và không lọc.
  4. Grad-CAM (`explain=true`): đủ 7 mô hình, không có mục lỗi.
  5. Ảnh lớn chạy được; tệp > 10 MB, ảnh quá nhiều điểm ảnh, tệp không phải ảnh bị từ chối 4xx có lý do, không phải 500.
  6. Thời gian: trung vị và p95 của `timing_ms.predict` và của cả vòng HTTP.

Chạy:
    python scripts/web_model_check.py --base-url http://127.0.0.1:8088          # máy chủ cục bộ (làm trước)
    python scripts/web_model_check.py                                           # production
Kết quả: artifacts/web_model_check.json (kèm bảng in ra). Thoát mã 1 nếu lệch quá ngưỡng hoặc có lỗi cứng.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRODUCTION = "https://datn.tranngochung.id.vn"
OUT_JSON = "artifacts/web_model_check.json"
MODEL_KEYS = ("centralized", "fedavg", "local_00", "local_01", "local_02", "local_03", "local_04")
MISMATCH_LIMIT = 0.005          # tối đa 0,5% ảnh x mô hình lệch dự đoán so với offline
SAMPLES_DIR = "web/static/samples/demo"


# --------------------------------------------------------------------------- HTTP
def post_image(base: str, data: bytes, filename: str, query: str = "", timeout: int = 180) -> tuple[int, dict[str, Any] | str, float]:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{base}/api/inference/all{query}", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
                return response.status, payload, (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            if exc.code in (429, 502, 503) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            try:
                return exc.code, json.loads(raw), (time.perf_counter() - started) * 1000
            except ValueError:
                return exc.code, raw[:300], (time.perf_counter() - started) * 1000
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return 0, f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000
    return 0, "hết lượt thử", 0.0


# --------------------------------------------------------------------------- mẫu ảnh
def build_samples(per_group: int, external: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.evaluate_published_models import source_group
    from src import bundle, registry as reg

    rng = np.random.default_rng(seed)
    manifest = bundle.load_manifest()
    release = reg.abs_path(bundle.release_dir(manifest))
    dataset_root = reg.abs_path("data/dataset_sources")
    rows = [json.loads(line) for line in (release / "manifests" / "val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[source_group(row["source_dataset"])].append(index)
    samples: list[dict[str, Any]] = []
    for group in ("plantvillage", "plantdoc", "field_other"):
        pool = by_group.get(group, [])
        for index in sorted(rng.choice(len(pool), size=min(per_group, len(pool)), replace=False)):
            row = rows[pool[index]]
            samples.append({"group": group, "path": dataset_root / row["relative_path"], "root": dataset_root, "relative_path": row["relative_path"],
                            "target": row["target_class"], "val_index": pool[index]})
    if external:
        from scripts.build_external_eval_set import EXTERNAL_DIR, MANIFEST

        ext_root = reg.abs_path(EXTERNAL_DIR)
        ext_rows = [r for r in json.loads(reg.abs_path(MANIFEST).read_text(encoding="utf-8"))["samples"] if r["tier"] == "strict"]
        for i in sorted(rng.choice(len(ext_rows), size=min(external, len(ext_rows)), replace=False)):
            row = ext_rows[i]
            samples.append({"group": "plantwild", "path": ext_root / row["relative_path"], "root": ext_root, "relative_path": row["relative_path"],
                            "target": row["target_class"], "val_index": None})
    demo_root = reg.abs_path(SAMPLES_DIR)
    for path in sorted(demo_root.glob("*")):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            samples.append({"group": "demo", "path": path, "root": demo_root, "relative_path": path.name, "target": None, "val_index": None})
    samples = [s for s in samples if s["path"].is_file()]
    return samples, {"rows": rows, "manifest": manifest, "release": release}


def offline_logits(samples: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, np.ndarray]:
    """logits[model] = [N, 38]: ảnh val lấy từ logits đã lưu lúc nhập gói, ảnh còn lại tính lại cùng đường tiền xử lý của lúc nhập."""
    from scripts.evaluate_published_models import _load_contract_transforms, _load_model, collect_logits
    from src import bundle, registry as reg

    out = {key: np.zeros((len(samples), 38), dtype=np.float32) for key in MODEL_KEYS}
    manifest = context["manifest"]
    name = Path(bundle.active_bundle_dir()).name
    cache = reg.abs_path(f"data/reeval/bundle_{name}_logits.npz")
    todo = list(range(len(samples)))
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if all(key in data.files and data[key].shape[0] == len(context["rows"]) for key in MODEL_KEYS):
            todo = []
            for i, sample in enumerate(samples):
                if sample["val_index"] is None:
                    todo.append(i)
                    continue
                for key in MODEL_KEYS:
                    out[key][i] = data[key][sample["val_index"]]
    if todo:
        bundle_dir = reg.abs_path(bundle.active_bundle_dir())
        transform = _load_contract_transforms(bundle_dir)
        paths = {"centralized": bundle_dir / "centralized" / "checkpoint_best.pt", "fedavg": bundle_dir / "fedavg" / "checkpoint_best.pt",
                 **{f"local_{i:02d}": bundle_dir / "local" / f"client_{i:02d}" / "checkpoint_best.pt" for i in range(5)}}
        models = {key: _load_model(path) for key, path in paths.items()}
        by_root: dict[Path, list[int]] = defaultdict(list)
        for i in todo:
            by_root[samples[i]["root"]].append(i)
        for root, indices in by_root.items():
            rows = [{"relative_path": samples[i]["relative_path"]} for i in indices]
            logits = collect_logits(rows, root, transform, models, 16, 0)
            for key in MODEL_KEYS:
                out[key][indices] = logits[key]
    del manifest
    return out


# --------------------------------------------------------------------------- kiểm
def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=PRODUCTION)
    parser.add_argument("--per-group", type=int, default=100, help="số ảnh val mỗi nhóm nguồn")
    parser.add_argument("--external", type=int, default=50, help="số ảnh PlantWild (nguồn lạ, tầng strict)")
    parser.add_argument("--crop-check", type=int, default=30)
    parser.add_argument("--explain-check", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--out", default=OUT_JSON)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    from PIL import Image

    from src import registry as reg
    from web.demo_evidence import load_evidence

    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[web_check] gói cục bộ chưa qua cổng kiểm hash: {evidence.get('reason')}", file=sys.stderr)
        return 1
    classes: list[str] = evidence["classes"]
    samples, context = build_samples(args.per_group, args.external, args.seed)
    print(f"[web_check] {len(samples)} ảnh → {base}", flush=True)
    offline = offline_logits(samples, context)
    results: dict[str, Any] = {"base_url": base, "protocol_id": evidence["protocol_id"], "images": len(samples), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # 1) + 2) từng ảnh qua web
    web_class: dict[tuple[int, str], str] = {}
    predict_ms: list[float] = []
    wall_ms: list[float] = []
    failures: list[str] = []
    for i, sample in enumerate(samples):
        status, payload, elapsed = post_image(base, sample["path"].read_bytes(), sample["path"].name)
        if status != 200 or not isinstance(payload, dict):
            failures.append(f"{sample['relative_path']}: HTTP {status} {str(payload)[:120]}")
            continue
        wall_ms.append(elapsed)
        predict_ms.append(float(payload["timing_ms"]["predict"]))
        for key in MODEL_KEYS:
            web_class[(i, key)] = payload["models"][key]["predicted_class"]
        if (i + 1) % 50 == 0:
            print(f"[web_check]   {i + 1}/{len(samples)}", flush=True)
    results["request_failures"] = failures

    mismatches: list[dict[str, Any]] = []
    compared = 0
    for i in range(len(samples)):
        for key in MODEL_KEYS:
            if (i, key) not in web_class:
                continue
            compared += 1
            logits = offline[key][i]
            top2 = np.sort(logits)[-2:]
            expected = classes[int(np.argmax(logits))]
            if web_class[(i, key)] != expected:
                mismatches.append({"image": samples[i]["relative_path"], "model": key, "web": web_class[(i, key)], "offline": expected,
                                   "offline_margin": round(float(top2[1] - top2[0]), 5)})
    results["agreement"] = {"compared": compared, "mismatches": len(mismatches), "rate": round(len(mismatches) / max(1, compared), 5),
                            "near_ties_margin_lt_0_01": sum(1 for m in mismatches if m["offline_margin"] < 0.01), "examples": mismatches[:12]}

    accuracy: dict[str, dict[str, Any]] = {}
    for group in ("plantvillage", "plantdoc", "field_other", "plantwild"):
        idx = [i for i, s in enumerate(samples) if s["group"] == group]
        accuracy[group] = {"images": len(idx)}
        for key in MODEL_KEYS:
            seen = [i for i in idx if (i, key) in web_class]
            accuracy[group][key] = round(sum(web_class[(i, key)] == samples[i]["target"] for i in seen) / max(1, len(seen)), 4)
    results["accuracy_via_web"] = accuracy

    # 3) lọc loại cây trên một phần ảnh val đã có kết quả không lọc
    crop_rows: list[dict[str, Any]] = []
    pool = [i for i, s in enumerate(samples) if s["group"] in ("plantvillage", "plantdoc", "field_other")]
    rng = np.random.default_rng(args.seed + 1)
    chosen = [pool[j] for j in rng.choice(len(pool), size=min(args.crop_check, len(pool)), replace=False)] if pool else []
    crop_violations, crop_before, crop_after = 0, Counter(), Counter()
    for i in chosen:
        crop = samples[i]["target"].split("___", 1)[0]
        status, payload, _ = post_image(base, samples[i]["path"].read_bytes(), samples[i]["path"].name, f"?crop={urllib.request.quote(crop)}")
        if status != 200 or not isinstance(payload, dict):
            failures.append(f"crop={crop}: HTTP {status} {str(payload)[:100]}")
            continue
        for key in MODEL_KEYS:
            predicted = payload["models"][key]["predicted_class"]
            crop_violations += not predicted.startswith(crop + "___")
            crop_after[key] += predicted == samples[i]["target"]
            crop_before[key] += web_class.get((i, key)) == samples[i]["target"]
    results["crop_filter"] = {"images": len(chosen), "violations": crop_violations,
                              "accuracy_without": {k: round(crop_before[k] / max(1, len(chosen)), 4) for k in MODEL_KEYS},
                              "accuracy_with": {k: round(crop_after[k] / max(1, len(chosen)), 4) for k in MODEL_KEYS}}

    # 4) Grad-CAM
    explain_bad: list[str] = []
    for i in range(min(args.explain_check, len(samples))):
        status, payload, _ = post_image(base, samples[i]["path"].read_bytes(), samples[i]["path"].name, "?explain=true", timeout=300)
        if status != 200 or not isinstance(payload, dict):
            explain_bad.append(f"{samples[i]['relative_path']}: HTTP {status}")
            continue
        for key in MODEL_KEYS:
            entry = (payload.get("explain") or {}).get(key)
            if not entry or "error" in entry:
                explain_bad.append(f"{samples[i]['relative_path']}/{key}: {(entry or {}).get('error', 'thiếu')}")
    results["gradcam"] = {"images": min(args.explain_check, len(samples)), "problems": explain_bad}

    # 5) ảnh lớn và ảnh xấu
    edge: dict[str, Any] = {}
    sample_path = samples[0]["path"]
    with Image.open(sample_path) as source:
        rgb = source.convert("RGB")
    big = rgb.resize((6000, 4000), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    big.save(buffer, "JPEG", quality=88)
    status, _, elapsed = post_image(base, buffer.getvalue(), "big.jpg", timeout=300)
    edge["large_24mp_jpeg"] = {"bytes": buffer.tell(), "status": status, "ms": round(elapsed)}
    huge = Image.new("RGB", (8000, 6000), (30, 120, 40))
    buffer = io.BytesIO()
    huge.save(buffer, "PNG", optimize=True)
    status, payload, _ = post_image(base, buffer.getvalue(), "huge.png")
    edge["over_pixel_limit_48mp"] = {"bytes": buffer.tell(), "status": status, "detail": (payload.get("detail") if isinstance(payload, dict) else str(payload))[:120]}
    status, payload, _ = post_image(base, b"\xff\xd8\xff" + bytes(11 * 1024 * 1024), "toolarge.jpg")
    edge["over_10mb"] = {"status": status, "detail": (payload.get("detail") if isinstance(payload, dict) else str(payload))[:120]}
    status, payload, _ = post_image(base, "đây không phải ảnh".encode("utf-8"), "notimage.jpg")
    edge["not_an_image"] = {"status": status, "detail": (payload.get("detail") if isinstance(payload, dict) else str(payload))[:120]}
    results["edge_cases"] = edge

    # 6) thời gian
    results["timing_ms"] = {"predict_median": round(percentile(predict_ms, 50), 1), "predict_p95": round(percentile(predict_ms, 95), 1),
                            "round_trip_median": round(percentile(wall_ms, 50)), "round_trip_p95": round(percentile(wall_ms, 95))}

    # kết luận
    problems: list[str] = []
    if failures:
        problems.append(f"{len(failures)} yêu cầu lỗi")
    if results["agreement"]["rate"] > MISMATCH_LIMIT:
        problems.append(f"lệch dự đoán so với offline {results['agreement']['rate']:.2%} > {MISMATCH_LIMIT:.1%}")
    if crop_violations:
        problems.append(f"{crop_violations} dự đoán vượt ra ngoài loại cây đã chọn")
    if explain_bad:
        problems.append(f"{len(explain_bad)} lỗi Grad-CAM")
    if edge["large_24mp_jpeg"]["status"] != 200:
        problems.append("ảnh 24 MP không chạy được")
    for name in ("over_pixel_limit_48mp", "over_10mb", "not_an_image"):
        if not 400 <= edge[name]["status"] < 500:
            problems.append(f"{name}: mong đợi 4xx, nhận {edge[name]['status']}")
    results["problems"] = problems
    out = reg.abs_path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    agree = results["agreement"]
    print(f"\n[web_check] trùng offline: {agree['compared'] - agree['mismatches']}/{agree['compared']} ({1 - agree['rate']:.2%}), "
          f"lệch sát hòa (chênh logit < 0,01): {agree['near_ties_margin_lt_0_01']}")
    print("[web_check] độ chính xác qua web (top-1):")
    print(f"  {'nhóm':14}{'ảnh':>5} " + " ".join(f"{k:>11}" for k in MODEL_KEYS))
    for group, row in accuracy.items():
        print(f"  {group:14}{row['images']:>5} " + " ".join(f"{row[k]:>11.1%}" for k in MODEL_KEYS))
    cf = results["crop_filter"]
    print(f"[web_check] lọc loại cây ({cf['images']} ảnh): vi phạm {cf['violations']}; fedavg {cf['accuracy_without']['fedavg']:.1%} → {cf['accuracy_with']['fedavg']:.1%}; "
          f"centralized {cf['accuracy_without']['centralized']:.1%} → {cf['accuracy_with']['centralized']:.1%}")
    print(f"[web_check] Grad-CAM: {'đạt' if not explain_bad else explain_bad[:3]}")
    print(f"[web_check] ảnh lớn/xấu: {json.dumps(edge, ensure_ascii=False)}")
    print(f"[web_check] thời gian: {results['timing_ms']}")
    print(f"[web_check] {'ĐẠT' if not problems else 'VẤN ĐỀ: ' + '; '.join(problems)} → {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
