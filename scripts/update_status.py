"""Đồng bộ một lệnh: registry → gates → PROJECT_STATUS.md + status_report.json.

Sau mỗi phiên làm việc chạy:
    python scripts/update_status.py            # registry + provenance + pre-demo + sinh file
    python scripts/update_status.py --skip-gates   # chỉ sinh lại file từ JSON hiện có
    python scripts/update_status.py --full-gates   # + final_acceptance (chạy pytest, chậm)

PROJECT_STATUS.md là file thống kê DUY NHẤT. Toàn bộ phần sinh tự được script
ghi lại; chỉ khối giữa <!-- ROADMAP-BEGIN --> và <!-- ROADMAP-END --> là người
tự tay sửa và được bảo toàn qua mỗi lần chạy.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

STATUS_MD = ROOT / "PROJECT_STATUS.md"
STATUS_JSON = ROOT / reg.STATUS_REPORT_JSON

ROADMAP_BEGIN = "<!-- ROADMAP-BEGIN (khối này TỰ TAY SỬA - script bảo toàn nội dung) -->"
ROADMAP_END = "<!-- ROADMAP-END -->"

FAMILY_TITLES = {
    reg.FAMILY_SCIENTIFIC_CANONICAL: "Họ A — Khoa học canonical (scratch, PlantVillage thuần)",
    reg.FAMILY_PUBLISHED_DEMO: "Họ B — Published demo (pretrained, mixed PV+PlantDoc)",
    reg.FAMILY_NETWORK_DEMO: "Họ C — Network demo (smoke kỹ thuật)",
    reg.FAMILY_REFERENCE: "Model tham chiếu",
    reg.FAMILY_SMOKE: "Họ D — Smoke (CẤM dùng trong luận văn)",
}

DEFAULT_ROADMAP = """### Checklist nghiệm thu

- [x] Lập model registry thống nhất (artifacts/registry.json) — 2026-10-09
- [x] Cách ly checkpoint smoke khỏi slot canonical + chặn ghi đè từ code — 2026-10-09
- [x] Tái sinh canonical_manifest.json trung thực từ nguồn thật — 2026-10-09
- [x] File thống kê duy nhất PROJECT_STATUS.md (tự sinh) — 2026-10-09
- [ ] Tách web inference/training/monitoring đọc qua registry
- [ ] Dọn tài liệu mâu thuẫn (xóa archive, tạo RUNBOOK/MODEL_REGISTRY)
- [ ] Kiểm chứng provenance & checkpoint (cần tái chạy run canonical hoặc khôi phục weights từ Kaggle)
- [ ] Chốt một protocol khoa học duy nhất cho mỗi thí nghiệm
- [ ] Audit leakage theo từng run (cần dataset gốc)
- [ ] Đánh giá PlantDoc theo từng lớp
- [ ] Phân tích lỗi confidence cao (100 ảnh sai với confidence > 0.9 ở FedAvg v7)
- [ ] Chạy multi-seed non-IID (seed 42/123/2026)
- [ ] Đồng bộ báo cáo & artifacts sau khi có run mới
- [ ] Full tests + release sạch

### Nhật ký

- 2026-10-09: Đồng bộ hóa — lập registry (18 models/5 family), phát hiện & cách ly smoke
  ghi đè slot canonical, xác nhận checkpoint canonical KHÔNG còn trên máy (metrics-only).
"""


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def run_script(rel: str, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel), *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def refresh_gates(full: bool) -> None:
    from scripts.build_registry import main as build_registry_main  # noqa: E402

    build_registry_main()

    code, out = run_script("scripts/validate_artifact_provenance.py")
    if code not in (0, 2):  # 2 = unverified (trung thực), khác = lỗi thật
        print(f"[update_status] CẢNH BÁO: validate_artifact_provenance.py exit={code}\n{out[-500:]}")

    code, out = run_script("scripts/pre_demo_check.py")
    if code not in (0, 1):
        print(f"[update_status] CẢNH BÁO: pre_demo_check.py exit={code}\n{out[-500:]}")

    if full:
        run_script("scripts/final_acceptance.py")


# ---------------------------------------------------------------------------
# Load dữ liệu đã sinh
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def results_table(models: list[dict[str, Any]], family: str) -> str:
    rows = [m for m in models if m["family"] == family]
    if not rows:
        return "_Không có model nào trong họ này._\n"
    lines = [
        "| Model | Trạng thái | Checkpoint | Số liệu chính | Nguồn số liệu |",
        "|---|---|---|---|---|",
    ]
    for m in rows:
        ck = "✅ có, hash ok" if m.get("checkpoint_status") in ("hash_verified", "matches_manifest") else (
            "❌ không có trên máy" if not m.get("checkpoint_exists") else str(m.get("checkpoint_status"))
        )
        metrics = m.get("metrics") or {}
        parts = []
        if family == reg.FAMILY_SCIENTIFIC_CANONICAL:
            if m["model_id"] == "local_only_canonical":
                parts.append(f"mean acc {fmt_pct(metrics.get('mean_accuracy'))} ± {fmt_pct(metrics.get('std_accuracy'))}")
            else:
                parts.append(f"test acc {fmt_pct(metrics.get('test_accuracy'), 4) if m['model_id']=='fedavg_canonical' else fmt_pct(metrics.get('test_accuracy'))}")
                parts.append(f"macro-F1 {fmt_pct(metrics.get('macro_f1'))}")
                parts.append(f"test {metrics.get('total_samples', 'n/a')} ảnh")
        elif family == reg.FAMILY_PUBLISHED_DEMO:
            pd, pv = metrics.get("plantdoc") or {}, metrics.get("plantvillage") or {}
            if pd:
                parts.append(f"PlantDoc {fmt_pct(pd.get('top1_accuracy'))} (n={pd.get('count')})")
            if pv:
                parts.append(f"PlantVillage {fmt_pct(pv.get('top1_accuracy'))} (n={pv.get('count')})")
            ov = metrics.get("overall") or {}
            if ov and not pd and not pv:
                parts.append(f"overall {fmt_pct(ov.get('top1_accuracy'))} (n={ov.get('count')})")
        else:
            for k in ("test_accuracy", "last_accuracy", "val_accuracy"):
                if metrics.get(k) is not None:
                    parts.append(f"{k}={metrics[k]}")
        if not parts:
            parts.append("(không có số liệu)")
        lines.append(
            f"| {m['display_name']} | {m.get('display_label') or m['status']} | {ck} | {'; '.join(parts)} | {m.get('metrics_source') or '—'} |"
        )
    return "\n".join(lines) + "\n"


def gates_section(gates: dict[str, Any]) -> str:
    lines = ["| Gate | Kết quả | Chi tiết |", "|---|---|---|"]
    p = gates.get("provenance") or {}
    lines.append(f"| Provenance checkpoints | **{p.get('status', '—')}** | critical: {', '.join(p.get('critical_issues') or []) or 'không'} |")
    pd = gates.get("pre_demo_check") or {}
    lines.append(f"| Pre-demo check | **{pd.get('overall', '—')}** | {pd.get('source', '')} |")
    fa = gates.get("final_acceptance") or {}
    if fa:
        lines.append(f"| Final acceptance | technical **{fa.get('technical', '—')}** / scientific **{fa.get('scientific', '—')}** | blockers: {len(fa.get('scientific_blockers') or [])} |")
    na = gates.get("network_artifact") or {}
    if na:
        lines.append(f"| Network artifact | **{'PASS' if na.get('ok') else 'FAIL'}** | {na.get('source', '')} |")
    return "\n".join(lines) + "\n"


def readiness_verdict(registry: dict[str, Any]) -> str:
    gates = registry.get("gates") or {}
    prov = (gates.get("provenance") or {}).get("status")
    fa = gates.get("final_acceptance") or {}
    demo_ok = any(
        m["family"] == reg.FAMILY_PUBLISHED_DEMO and m.get("checkpoint_status") == "hash_verified"
        for m in registry["models"]
    )
    if prov == "verified" and fa.get("scientific") == "pass":
        return "SẴN SÀNG NGHIỆM THU KHOA HỌC"
    if demo_ok:
        return "DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học (checkpoint canonical mất, chưa multi-seed)"
    return "CHƯA SẴN SÀNG"


# ---------------------------------------------------------------------------
# ROADMAP block
# ---------------------------------------------------------------------------
def read_roadmap_block() -> str:
    if STATUS_MD.exists():
        text = STATUS_MD.read_text(encoding="utf-8")
        match = re.search(
            re.escape(ROADMAP_BEGIN) + r"\n(.*?)" + re.escape(ROADMAP_END),
            text, re.DOTALL,
        )
        if match:
            return match.group(1).rstrip() + "\n"
    return DEFAULT_ROADMAP


def extract_open_items(roadmap: str) -> list[str]:
    return [ln.strip()[6:].strip() for ln in roadmap.splitlines() if ln.strip().startswith("- [ ]")]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(registry: dict[str, Any], roadmap: str, now: datetime) -> str:
    models = registry["models"]
    gates = registry.get("gates") or {}
    dataset = registry.get("dataset_truth") or {}
    split = dataset.get("split") or {}
    demo = registry.get("canonical_run_id")

    open_items = extract_open_items(roadmap)

    parts: list[str] = []
    parts.append(f"# PROJECT STATUS — Federated Learning PlantVillage")
    parts.append("")
    parts.append(f"> **Cập nhật lúc:** {now.strftime('%Y-%m-%d %H:%M')} · sinh tự bởi `scripts/update_status.py`")
    parts.append(f"> **Đánh giá hiện tại:** {readiness_verdict(registry)}")
    parts.append(">")
    parts.append("> File này là nguồn thống kê DUY NHẤT về tiến độ. Con số nào trong repo")
    parts.append("> phải truy được về `artifacts/registry.json`. Không sửa phần sinh tự —")
    parts.append("> chỉ sửa khối ROADMAP giữa hai marker.")
    parts.append("")

    parts.append("## 1. Trạng thái hệ thống (sự thật từ registry)")
    parts.append("")
    fam_count: dict[str, int] = {}
    for m in models:
        fam_count[m["family"]] = fam_count.get(m["family"], 0) + 1
    parts.append(f"- Tổng số model trong registry: **{len(models)}** (" + ", ".join(f"{k}: {v}" for k, v in fam_count.items()) + ")")
    parts.append(f"- Dataset reproduction cục bộ: `{reg.PARTITIONS_DIR}` — split **{split.get('train')}/{split.get('val')}/{split.get('test')}** ({dataset.get('num_clients')} clients, seed {dataset.get('split_seed')}, protocol `{dataset.get('protocol')}`)")
    parts.append(f"- Run canonical Kaggle: `{demo}` — metrics CÓ, **checkpoint KHÔNG còn trên máy**")
    parts.append("- Demo web dùng họ published_demo (hash-verified) — hoạt động bình thường")
    parts.append("")

    parts.append("## 2. Kết quả chuẩn theo họ (không trộn lẫn)")
    parts.append("")
    for family in (
        reg.FAMILY_SCIENTIFIC_CANONICAL,
        reg.FAMILY_PUBLISHED_DEMO,
        reg.FAMILY_NETWORK_DEMO,
        reg.FAMILY_REFERENCE,
        reg.FAMILY_SMOKE,
    ):
        parts.append(f"### {FAMILY_TITLES[family]}")
        parts.append("")
        desc = (registry.get("families") or {}).get(family, {}).get("description")
        if desc:
            parts.append(f"_{desc}_")
            parts.append("")
        parts.append(results_table(models, family))

    parts.append("## 3. Gates kỹ thuật")
    parts.append("")
    parts.append(gates_section(gates))

    parts.append("## 4. Lộ trình & checklist (khối dưới đây TỰ TAY SỬA)")
    parts.append("")
    parts.append(ROADMAP_BEGIN)
    parts.append(roadmap.rstrip())
    parts.append(ROADMAP_END)
    parts.append("")

    parts.append("## 5. Việc cần làm tiếp theo (sinh từ checklist chưa tick)")
    parts.append("")
    if open_items:
        for i, item in enumerate(open_items, 1):
            parts.append(f"{i}. {item}")
    else:
        parts.append("_(không còn mục nào mở trong checklist)_")
    parts.append("")

    parts.append("## 6. Cách đồng bộ")
    parts.append("")
    parts.append("```powershell")
    parts.append(".\\.venv\\Scripts\\python.exe scripts\\update_status.py          # chuẩn")
    parts.append(".\\.venv\\Scripts\\python.exe scripts\\update_status.py --skip-gates  # chỉ sinh lại file")
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def build_status_report(registry: dict[str, Any], roadmap: str, now: datetime) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(timespec="seconds"),
        "readiness": readiness_verdict(registry),
        "canonical_run_id": registry.get("canonical_run_id"),
        "dataset_truth": registry.get("dataset_truth"),
        "model_counts_by_family": {
            fam: sum(1 for m in registry["models"] if m["family"] == fam)
            for fam in reg.FAMILY_DESCRIPTIONS
        },
        "canonical_checkpoint_status": {
            m["model_id"]: {
                "checkpoint_exists": m.get("checkpoint_exists"),
                "checkpoint_status": m.get("checkpoint_status"),
                "status": m.get("status"),
            }
            for m in registry["models"]
            if m["family"] == reg.FAMILY_SCIENTIFIC_CANONICAL
        },
        "gates": registry.get("gates"),
        "open_checklist_items": extract_open_items(roadmap),
        "source": "scripts/update_status.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gates", action="store_true", help="Chỉ đọc JSON hiện có, không chạy lại registry/gates")
    parser.add_argument("--full-gates", action="store_true", help="Chạy thêm final_acceptance (pytest, chậm)")
    args = parser.parse_args()

    now = datetime.now()

    if not args.skip_gates:
        print("[update_status] refresh registry + gates ...")
        refresh_gates(full=args.full_gates)

    registry = load_json(ROOT / reg.REGISTRY_JSON)
    if registry is None:
        print(f"[update_status] LỖI: chưa có {reg.REGISTRY_JSON} — chạy không --skip-gates để sinh.", file=sys.stderr)
        return 1

    roadmap = read_roadmap_block()
    STATUS_MD.write_text(render(registry, roadmap, now), encoding="utf-8")
    print(f"[update_status] wrote PROJECT_STATUS.md")

    STATUS_JSON.write_text(
        json.dumps(build_status_report(registry, roadmap, now), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[update_status] wrote {reg.STATUS_REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
