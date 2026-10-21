"""Dựng trang kiểm nhãn bằng mắt cho các lá PlantDoc đang bị cách ly (quarantine_status == clean).

Vì sao: nhãn của các lá này chỉ là "ngầm định" (assumed_healthy: nhãn kiểu "Apple leaf" được hiểu là lá khỏe;
ambiguous_alias: tên nhãn PlantDoc khớp gần đúng với nhãn PlantVillage). Học nhãn sai còn tệ hơn không học, nên nhóm nào
được nhận vào tập học phải do NGƯỜI kiểm. Script lấy ngẫu nhiên (seed cố định, tái lập được) tối đa N lá mỗi nhóm.

Quy tắc nhận nhóm (đặt TRƯỚC khi xem kết quả): nhóm được nhận khi trong các lá đã kiểm, "đúng" ≥ 90% và số lá "không rõ"
≤ 20%; dưới mức đó nhóm vẫn ở trong cách ly. Quy tắc nằm trong trang và trong samples.json.

Chạy:
    python scripts/build_label_review.py                 # ghi data/label_review/{index.html,samples.json}
    python scripts/build_label_review.py --per-class 30 --seed 20261012
    python scripts/build_label_review.py --external      # lấy mẫu từ mọi nguồn ngoài đồng chưa kiểm nhãn (data/external/sources/*/manifest.jsonl)
                                                         # -> data/label_review_ext/{index.html,samples.json}; nhóm = (nguồn, lớp)
Mở data/label_review/index.html bằng trình duyệt (không cần máy chủ), bấm kiểm, rồi "Tải kết quả" → decisions.json
đặt cạnh index.html. Ảnh được nạp trực tiếp từ data/dataset_sources (đường dẫn tương đối); không có gì được gửi đi đâu.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

OUT_DIR = "data/label_review"  # gitignored (data/*)
ACCEPT_CORRECT = 0.90
ACCEPT_UNSURE_MAX = 0.20


# --------------------------------------------------------------------------- phần thuần (có test)
def select_clean_plantdoc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chỉ lá sạch có nhãn trong taxonomy (loại class_id = -1: nhãn PlantDoc không có trong 38 nhóm)."""
    return [r for r in rows if r.get("source_domain") == "plantdoc" and r.get("quarantine_status") == "clean"
            and r.get("target_class") and int(r.get("class_id", -1)) >= 0]


def group_key(row: dict[str, Any], by_source: bool) -> str:
    """Khóa nhóm kiểm nhãn: lớp (PlantDoc cách ly) hoặc 'nguồn|lớp' (nguồn ngoài đồng; mỗi nguồn được nhận/loại riêng)."""
    return f"{row['source_dataset']}|{row['target_class']}" if by_source else row["target_class"]


def sample_per_class(rows: list[dict[str, Any]], per_class: int, seed: int, by_source: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Tối đa `per_class` lá mỗi nhóm; sắp theo sample_id trước khi lấy ngẫu nhiên để kết quả không phụ thuộc thứ tự tệp."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        by_class[group_key(row, by_source)].append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(by_class):
        items = by_class[name]
        rng = random.Random(f"{seed}:{name}")
        out[name] = sorted(rng.sample(items, min(per_class, len(items))), key=lambda r: r["sample_id"])
    return out


def classify(decisions: dict[str, str], sampled_ids: list[str]) -> dict[str, Any]:
    """Áp quy tắc nhận nhóm. `decisions[sample_id]` ∈ {"dung", "sai", "khong_ro"}; lá chưa kiểm tính là chưa xong."""
    total = len(sampled_ids)
    counts = {k: sum(1 for s in sampled_ids if decisions.get(s) == k) for k in ("dung", "sai", "khong_ro")}
    reviewed = sum(counts.values())
    complete = reviewed == total and total > 0
    ok = complete and counts["dung"] / total >= ACCEPT_CORRECT and counts["khong_ro"] / total <= ACCEPT_UNSURE_MAX
    return {"total": total, "reviewed": reviewed, **counts, "complete": complete, "accepted": bool(ok)}


# --------------------------------------------------------------------------- trang HTML
_PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kiểm nhãn lá PlantDoc</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f6f4ec;color:#1c261d}
 header{position:sticky;top:0;background:#fffef9;border-bottom:1px solid #d8d3c0;padding:10px 16px;z-index:2}
 h1{font-size:18px;margin:0 0 4px} .rule{color:#465045;font-size:13px}
 main{padding:12px 16px 60px;max-width:1500px;margin:auto}
 section{background:#fffef9;border:1px solid #d8d3c0;border-radius:10px;margin:14px 0;padding:12px}
 section h2{font-size:16px;margin:0 0 4px} .meta{color:#5e6759;font-size:13px;margin-bottom:8px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
 .card{border:2px solid #d8d3c0;border-radius:8px;padding:6px;background:#fff}
 .card img{width:100%;height:170px;object-fit:contain;background:#eee;cursor:zoom-in;display:block}
 .card .lab{font-size:12px;color:#5e6759;margin:4px 0;word-break:break-word}
 .btns{display:flex;gap:4px} .btns button{flex:1;padding:6px 2px;border:1px solid #b9b39c;border-radius:6px;background:#f4f1e4;cursor:pointer}
 .card.dung{border-color:#1f6b45}.card.sai{border-color:#b4233f}.card.khong_ro{border-color:#9a5408}
 .dung .btns .b-dung{background:#1f6b45;color:#fff}.sai .btns .b-sai{background:#b4233f;color:#fff}.khong_ro .btns .b-khong_ro{background:#9a5408;color:#fff}
 .status{font-weight:600}.ok{color:#1f6b45}.no{color:#b4233f}.wait{color:#9a5408}
 .actions button{padding:6px 12px;margin-right:8px}
</style></head><body>
<header>
 <h1>Kiểm nhãn lá PlantDoc bị cách ly</h1>
 <div class="rule">Với mỗi ảnh: nhãn ghi bên dưới có <b>đúng</b> với chiếc lá không? Chỉ bấm "đúng" khi chắc chắn; không chắc thì bấm "không rõ".
 Nhóm được nhận vào tập học khi <b>≥ __ACCEPT__% đúng</b> và <b>≤ __UNSURE__% không rõ</b> (đặt trước khi xem kết quả). Ảnh không đổi tên bệnh.</div>
 <div class="actions"><span id="progress"></span>
  <button id="dl">Tải kết quả (decisions.json)</button><button id="reset">Xóa mọi lựa chọn</button></div>
</header>
<main id="root"></main>
<script>
const DATA = __DATA__;
const KEY = "label_review_v1:" + DATA.seed;
let dec = {};
try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { dec = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {} };
const root = document.getElementById("root");
const label = c => c.replace("___", " — ").replace(/_/g, " ");
function stats(cls) {
  const ids = cls.items.map(i => i.id); const n = ids.length;
  const c = {dung: 0, sai: 0, khong_ro: 0}; ids.forEach(id => { if (dec[id]) c[dec[id]]++; });
  const done = c.dung + c.sai + c.khong_ro;
  const ok = done === n && c.dung / n >= DATA.accept_correct && c.khong_ro / n <= DATA.accept_unsure_max;
  return {n, done, ...c, ok};
}
function render() {
  let totalAll = 0, doneAll = 0;
  root.innerHTML = "";
  DATA.classes.forEach(cls => {
    const s = stats(cls); totalAll += s.n; doneAll += s.done;
    const sec = document.createElement("section");
    const st = s.done < s.n ? `<span class="status wait">chưa xong (${s.done}/${s.n})</span>`
      : s.ok ? `<span class="status ok">ĐƯỢC NHẬN (${s.dung}/${s.n} đúng)</span>`
      : `<span class="status no">KHÔNG nhận (${s.dung}/${s.n} đúng, ${s.khong_ro} không rõ)</span>`;
    sec.innerHTML = `<h2>${cls.title || label(cls.name)} — ${st}</h2><div class="meta">${cls.total} lá bị cách ly · kiểm ${s.n} · trạng thái nhãn: ${cls.mappings.join(", ")}</div><div class="grid"></div>`;
    const grid = sec.querySelector(".grid");
    cls.items.forEach(it => {
      const d = dec[it.id] || "";
      const card = document.createElement("div"); card.className = "card " + d;
      card.innerHTML = `<a href="${it.src}" target="_blank" rel="noopener"><img loading="lazy" src="${it.src}" alt="lá ${label(cls.name)}"></a>
        <div class="lab">Nhãn PlantDoc gốc: <b>${it.source_label || "—"}</b><br>${it.mapping}</div>
        <div class="btns"><button class="b-dung">Đúng</button><button class="b-sai">Sai</button><button class="b-khong_ro">Không rõ</button></div>`;
      ["dung", "sai", "khong_ro"].forEach(k => card.querySelector(".b-" + k).addEventListener("click", () => {
        dec[it.id] = (dec[it.id] === k) ? undefined : k; if (!dec[it.id]) delete dec[it.id]; save(); render();
      }));
      grid.appendChild(card);
    });
    root.appendChild(sec);
  });
  document.getElementById("progress").textContent = `Đã kiểm ${doneAll}/${totalAll} lá · `;
}
document.getElementById("dl").addEventListener("click", () => {
  const out = {seed: DATA.seed, per_class: DATA.per_class, accept_correct: DATA.accept_correct, accept_unsure_max: DATA.accept_unsure_max,
               release_manifest_sha256: DATA.release_manifest_sha256, decisions: dec};
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "decisions.json"; a.click();
});
document.getElementById("reset").addEventListener("click", () => { if (confirm("Xóa mọi lựa chọn?")) { dec = {}; save(); render(); } });
render();
</script></body></html>
"""


def build_page(sampled: dict[str, list[dict[str, Any]]], totals: dict[str, int], source_labels: dict[str, str],
               seed: int, per_class: int, release_sha: str, images_rel: str) -> str:
    classes = []
    for name, items in sampled.items():
        title = None
        if "|" in name:  # nhóm "nguồn|lớp"
            source, _, cls = name.partition("|")
            title = f"{source} → {cls.replace('___', ' — ').replace('_', ' ')}"
        classes.append({
            "name": name,
            "title": title,
            "total": totals[name],
            "mappings": sorted({str(r.get("mapping_status")) for r in items}),
            "items": [{
                "id": r["sample_id"],
                "src": f"{images_rel}/{r['relative_path']}",
                "mapping": str(r.get("mapping_status")),
                "source_label": html.escape(source_labels.get(r.get("annotation_id", ""), "")),
            } for r in items],
        })
    data = {"seed": seed, "per_class": per_class, "accept_correct": ACCEPT_CORRECT, "accept_unsure_max": ACCEPT_UNSURE_MAX,
            "release_manifest_sha256": release_sha, "classes": classes}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # không để </script> thoát khỏi thẻ
    return (_PAGE.replace("__DATA__", payload).replace("__ACCEPT__", str(int(ACCEPT_CORRECT * 100)))
            .replace("__UNSURE__", str(int(ACCEPT_UNSURE_MAX * 100))))


EXTERNAL_OUT_DIR = "data/label_review_ext"  # gitignored (data/*)
PER_IMAGE_PREFIXES = ("inat_",)  # iNaturalist và Wikimedia (inat_wm_): lẫn nhiều ảnh không phải lá/không đúng bệnh, nên kiểm TỪNG ảnh thay vì lấy mẫu 30
PER_IMAGE_MAX = 150
EXTERNAL_SOURCES = "data/external/sources"


def load_external_rows() -> list[dict[str, Any]]:
    """Mọi dòng manifest của nguồn ngoài đồng còn `label_review_status == unreviewed` (nhãn cộng đồng hoặc ánh xạ gần đúng)."""
    rows: list[dict[str, Any]] = []
    for manifest in sorted(reg.abs_path(EXTERNAL_SOURCES).glob("*/manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("label_review_status") == "unreviewed":
                    rows.append(row)
    return rows


def build_external(per_class: int, seed: int) -> int:
    rows = load_external_rows()
    if not rows:
        print("[label-review] không có ảnh ngoài đồng nào cần kiểm nhãn (chạy scripts/import_external_source.py trước)", file=sys.stderr)
        return 1
    totals: dict[str, int] = defaultdict(int)
    source_labels: dict[str, str] = {}
    for row in rows:
        totals[group_key(row, True)] += 1
        row["annotation_id"] = row["sample_id"]
        source_labels[row["sample_id"]] = str(row.get("original_file", ""))
    # Kiểm từng ảnh chỉ cho nhóm iNaturalist/Wikimedia nhỏ (<= PER_IMAGE_MAX ảnh); nhóm lớn hơn lấy mẫu 30 ảnh theo quy tắc nhận nhóm như nguồn khoa học
    per_image_rows = [r for r in rows if r["source_dataset"].startswith(PER_IMAGE_PREFIXES) and totals[group_key(r, True)] <= PER_IMAGE_MAX]
    per_image_ids = {r["sample_id"] for r in per_image_rows}
    sampled = sample_per_class([r for r in rows if r["sample_id"] not in per_image_ids], per_class, seed, by_source=True)
    sampled.update(sample_per_class(per_image_rows, PER_IMAGE_MAX, seed, by_source=True))
    per_image_groups = sorted({group_key(r, True) for r in per_image_rows})
    out = reg.abs_path(EXTERNAL_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    images_rel = "../external/sources"  # data/label_review_ext/../external/sources
    (out / "index.html").write_text(build_page(sampled, dict(totals), source_labels, seed, per_class, "", images_rel), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps({
        "seed": seed, "per_class": per_class, "accept_correct": ACCEPT_CORRECT, "accept_unsure_max": ACCEPT_UNSURE_MAX,
        "classes": {name: [r["sample_id"] for r in items] for name, items in sampled.items()}, "totals": dict(totals),
        "per_image_groups": per_image_groups,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n = sum(len(v) for v in sampled.values())
    print(f"[label-review] {len(sampled)} nhóm (nguồn, lớp), {n} ảnh cần kiểm -> {EXTERNAL_OUT_DIR}/index.html")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--external", action="store_true", help="lấy mẫu từ các nguồn ngoài đồng chưa kiểm nhãn thay cho PlantDoc cách ly")
    args = parser.parse_args()

    if args.external:
        return build_external(args.per_class, args.seed)

    release = reg.abs_path(reg.RELEASE_V3_DIR)
    manifest = release / "manifests" / "quarantine.jsonl"
    detection = release / "annotations" / "detection.json"
    if not manifest.is_file() or not detection.is_file():
        print("[label-review] thiếu data/dataset_sources/mixed/pv_pd_v3", file=sys.stderr)
        return 1
    with manifest.open(encoding="utf-8") as stream:
        rows = select_clean_plantdoc([json.loads(line) for line in stream if line.strip()])
    annotations = json.loads(detection.read_text(encoding="utf-8"))["annotations"]
    source_labels = {a["annotation_id"]: str(a.get("source_label") or "") for a in annotations}
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        totals[row["target_class"]] += 1
    sampled = sample_per_class(rows, args.per_class, args.seed)

    out = reg.abs_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    release_sha = json.loads((release / "release_manifest.json").read_text(encoding="utf-8")).get("source_release_manifest_sha256", "")
    images_rel = "../dataset_sources"  # data/label_review/../dataset_sources
    (out / "index.html").write_text(build_page(sampled, dict(totals), source_labels, args.seed, args.per_class, release_sha, images_rel), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps({
        "seed": args.seed, "per_class": args.per_class, "accept_correct": ACCEPT_CORRECT, "accept_unsure_max": ACCEPT_UNSURE_MAX,
        "classes": {name: [r["sample_id"] for r in items] for name, items in sampled.items()},
        "totals": dict(totals),
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n = sum(len(v) for v in sampled.values())
    print(f"[label-review] {len(sampled)} nhóm, {n} lá cần kiểm → {OUT_DIR}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
