"""Đồng bộ một lệnh: registry -> gates -> PROJECT_STATUS.md + status_report.json.

Sau mỗi phiên làm việc chạy:
    python scripts/update_status.py                # registry + provenance + pre-demo + sinh file
    python scripts/update_status.py --skip-gates   # chỉ sinh lại file từ JSON hiện có
    python scripts/update_status.py --full-gates   # + final_acceptance (chạy pytest, chậm)
    python scripts/update_status.py --check        # không ghi; thoát mã 1 nếu PROJECT_STATUS.md đang lệch

PROJECT_STATUS.md là file thống kê DUY NHẤT. Toàn bộ nội dung do script sinh từ sự thật trên đĩa;
CHECKLIST DO MÁY CHẤM (mỗi mục có hàm kiểm tra), không tick tay. Chỉ khối "Nhật ký" giữa hai marker
là người viết tay và được bảo toàn.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

STATUS_MD = ROOT / "PROJECT_STATUS.md"
STATUS_JSON = ROOT / reg.STATUS_REPORT_JSON

JOURNAL_BEGIN = "<!-- ROADMAP-BEGIN (khối này TỰ TAY SỬA - script bảo toàn nội dung) -->"
JOURNAL_END = "<!-- ROADMAP-END -->"
TIMESTAMP_RE = re.compile(r"^> \*\*Cập nhật lúc:\*\* .*$", re.M)

DONE, TODO, BLOCKED = "XONG", "CHƯA", "BLOCKED"
PATH_LITERAL_RE = re.compile(r"^(?:artifacts[/\\][^\s]*|artifacts)$")  # cả chuỗi là một đường dẫn dưới artifacts/

FAMILY_TITLES = {
    reg.FAMILY_PUBLISHED_DEMO: "Hiện hành — bundle đã kiểm hash (pretrained, mixed PV+PlantDoc)",
    reg.FAMILY_SCIENTIFIC_CANONICAL: "Lịch sử — Kaggle scratch, PlantVillage thuần (chỉ còn metrics)",
    reg.FAMILY_NETWORK_DEMO: "Network demo (smoke kỹ thuật Flower)",
    reg.FAMILY_REFERENCE: "Model tham chiếu",
    reg.FAMILY_SMOKE: "Smoke — CẤM dùng trong luận văn",
}

ALLOWED_DOCS = {
    "README.md", "MODEL_REGISTRY.md", "RUNBOOK.md", "EXPERIMENT_PROTOCOL.md", "ARCHITECTURE.md",
    "TONG_HOP_FEDERATED_LEARNING.md", "WEB_DEMO_GD1_GD2.md", "AWS_DEPLOYMENT_GUIDE.md",
    "SERVER_DEPLOY_20261005.md", "REFERENCES_IEEE.md",
}
# Số liệu từng xuất hiện trong tài liệu cũ và mâu thuẫn với artifact: không được quay lại docs hiện hành.
FORBIDDEN_NUMBERS = ("97,12", "97.12", "90,44", "90.44")
LEGACY_ROOT_ENTRIES = (
    "train", "fl_training", "stage1_compat", "stage2_matched", "stage2_scratch", "experiments", "team_runner.py",
)
# File sinh tự: không tính vào "release sạch"
DEFAULT_JOURNAL = """### Nhật ký

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


def refresh_gates(full: bool) -> list[str]:
    """Chạy lại registry + các gate. Trả về danh sách script gate bị crash (mã thoát lạ)."""
    from scripts import build_registry as br  # noqa: E402

    crashed: list[str] = []
    for rel in ("scripts/report_domain_gap.py", "scripts/report_per_class.py"):  # suy ra từ bundle đã kiểm hash
        code, out = run_script(rel)
        if code != 0:
            crashed.append(rel)
            print(f"[update_status] CẢNH BÁO: {rel} exit={code}\n{out[-500:]}")

    br.main_build()

    for rel, ok_codes in (
        ("scripts/validate_artifact_provenance.py", (0, 2)),  # 2 = unverified (trung thực)
        ("scripts/pre_demo_check.py", (0, 2)),
        ("scripts/validate_network_artifact.py", (0, 2)),
    ):
        code, out = run_script(rel)
        if code not in ok_codes:
            crashed.append(rel)
            print(f"[update_status] CẢNH BÁO: {rel} exit={code} — gate này KHÔNG được làm mới\n{out[-500:]}")

    if full:
        code, out = run_script("scripts/final_acceptance.py")
        if code not in (0, 2):
            crashed.append("scripts/final_acceptance.py")
            print(f"[update_status] CẢNH BÁO: final_acceptance exit={code}\n{out[-500:]}")

    br.refresh_registry_gates()
    return crashed


# ---------------------------------------------------------------------------
# Đọc dữ liệu đã sinh
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


@dataclass
class Result:
    status: str
    detail: str
    needs: str = ""
    unlock: list[str] = field(default_factory=list)


@dataclass
class Item:
    key: str
    title: str
    check: Callable[["Ctx"], Result]


class Ctx:
    def __init__(self, registry: dict[str, Any]):
        self.registry = registry
        self.models = {m["model_id"]: m for m in registry["models"]}
        self.gates = registry.get("gates") or {}
        self.final = load_json(reg.abs_path(reg.FINAL_ACCEPTANCE_JSON)) or {}
        self.fingerprint = reg.code_fingerprint()

    def models_of(self, role: str) -> list[dict[str, Any]]:
        return [m for m in self.registry["models"] if m.get("role") == role]


# ---------------------------------------------------------------------------
# Các hàm kiểm tra — mỗi hàm trả sự thật từ đĩa/registry, không đọc ý kiến
# ---------------------------------------------------------------------------
def check_registry(ctx: Ctx) -> Result:
    cat = ctx.gates.get("checkpoint_catalog") or {}
    ok = ctx.registry.get("schema_version") == reg.SCHEMA_VERSION and cat.get("status") == "complete"
    if ok:
        return Result(DONE, f"schema v{reg.SCHEMA_VERSION}; {len(ctx.models)} model; "
                            f"{cat.get('total_pt_files')} file .pt đều có tên "
                            f"({cat.get('in_models')} trong model, {cat.get('in_groups')} trong nhóm)")
    return Result(TODO, f"danh mục .pt chưa đủ: {cat.get('unclassified') or 'registry chưa đúng schema'}")


def check_smoke_isolated(ctx: Ctx) -> Result:
    bad = [
        m["model_id"] for m in ctx.models_of(reg.ROLE_HISTORICAL)
        if m.get("checkpoint_status") in ("smoke_overwrite_detected", "unknown_mismatch")
    ]
    if bad:
        return Result(TODO, f"slot canonical đang chứa checkpoint không khớp: {bad}")
    smoke = ctx.models_of(reg.ROLE_QUARANTINED)
    return Result(DONE, f"slot canonical không chứa smoke; {len(smoke)} checkpoint smoke nằm ở {reg.SMOKE_CHECKPOINTS_DIR}/")


def check_single_engine(ctx: Ctx) -> Result:
    leftovers = [n for n in LEGACY_ROOT_ENTRIES if (ROOT / n).exists()]
    leftovers += [p.name for p in ROOT.glob("kaggle_*.ipynb")] + [p.name for p in ROOT.glob("release_manifest_v*.json")]
    if leftovers:
        return Result(TODO, f"còn ở gốc repo: {sorted(leftovers)}")
    if not (ROOT / "legacy").is_dir():
        return Result(TODO, "thiếu thư mục legacy/")
    return Result(DONE, "engine duy nhất là src/; pipeline Kaggle/stage* nằm trong legacy/ (chạy riêng: pytest legacy/tests)")


def check_web_split(ctx: Ctx) -> Result:
    app = (ROOT / "web/app.py").read_text(encoding="utf-8")
    inline_routes = re.findall(r'@app\.(?:get|post|put|delete)\("/api', app)
    router_text = (ROOT / "web/routers/inference.py").read_text(encoding="utf-8")
    direct = [n for n in ("predict_published", "predict_plantvillage") if re.search(rf"import .*\b{n}\b", router_text)]
    missing = [p for p in ("web/routers/monitoring.py", "web/routers/inference.py", "web/routers/training.py",
                           "web/services/model_service.py", "web/state.py") if not (ROOT / p).exists()]
    if inline_routes or direct or missing:
        return Result(TODO, f"route còn trong app.py: {len(inline_routes)}; router import loader trực tiếp: {direct}; thiếu: {missing}")
    return Result(DONE, "app.py chỉ mount router; inference chỉ đi qua web/services/model_service.py (cổng web_visible)")


def path_literals(path: Path, *, only_prefixes: tuple[str, ...] = ()) -> list[str]:
    """Hằng chuỗi trong file Python trông như đường dẫn dưới artifacts/ (bỏ docstring, help=, câu thông báo).

    Quét bằng AST nên bắt được cả `ROOT / "artifacts" / "x"` và `Path("artifacts")`, không chỉ chuỗi có dấu nháy sát tên thư mục."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[int, ast.AST] = {}
    skip: set[int] = set()
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                skip.add(id(body[0].value))
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in ("help", "description") and isinstance(keyword.value, ast.Constant):
                    skip.add(id(keyword.value))

    hits: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or id(node) in skip:
            continue
        value = node.value
        if not PATH_LITERAL_RE.match(value):
            continue
        if value in ("artifacts",):  # tên trần chỉ tính khi dùng làm thành phần đường dẫn
            parent = parents.get(id(node))
            is_path_part = isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
            is_path_call = isinstance(parent, ast.Call) and getattr(parent.func, "id", getattr(parent.func, "attr", "")) == "Path"
            if not (is_path_part or is_path_call):
                continue
        if only_prefixes and not value.replace("\\", "/").startswith(only_prefixes):
            continue
        hits.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    return hits


def check_no_hardcode(ctx: Ctx) -> Result:
    hits: list[str] = []
    for top in ("src", "web", "scripts"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts or path == ROOT / "src/registry.py":
                continue
            hits += path_literals(path)
    if hits:
        return Result(TODO, f"{len(hits)} chỗ hardcode đường dẫn artifacts/ (quét AST): {hits[:5]}")
    return Result(DONE, "mọi đường dẫn artifacts/ trong src/web/scripts lấy từ src/registry.py (quét AST)")


def check_docs_clean(ctx: Ctx) -> Result:
    root_md = {p.name for p in ROOT.glob("*.md")}
    extra_root = sorted(root_md - {"README.md", "PROJECT_STATUS.md"})
    extra_docs = sorted({p.name for p in (ROOT / "docs").glob("*.md")} - ALLOWED_DOCS)
    if extra_root or extra_docs:
        return Result(TODO, f"tài liệu ngoài danh mục: gốc={extra_root} docs={extra_docs}")
    return Result(DONE, "gốc repo chỉ còn README.md + PROJECT_STATUS.md; docs/ chỉ gồm tài liệu hiện hành")


def check_docs_numbers(ctx: Ctx) -> Result:
    found: list[str] = []
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for number in FORBIDDEN_NUMBERS:
            if number in text:
                found.append(f"{path.name}:{number}")
    if found:
        return Result(TODO, f"còn số liệu mâu thuẫn đã biết: {found}")
    return Result(DONE, f"không còn số liệu mâu thuẫn đã biết ({', '.join(FORBIDDEN_NUMBERS[:2])}...) trong README/docs")


def check_tests_isolated(ctx: Ctx) -> Result:
    hits: list[str] = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        hits += path_literals(path, only_prefixes=(reg.CHECKPOINTS_DIR, reg.METRICS_DIR))
    if hits:
        return Result(TODO, f"test ghi thẳng vào artifacts/checkpoints|metrics (ghi đè artifact đang track mỗi lần chạy): {sorted(set(h.split(':')[0] for h in hits))}")
    return Result(DONE, "test không tham chiếu artifacts/checkpoints|metrics (dùng tmp_path)")


def check_full_tests(ctx: Ctx) -> Result:
    final = ctx.final
    if not final:
        return Result(TODO, f"chưa có {reg.FINAL_ACCEPTANCE_JSON} — chạy update_status.py --full-gates")
    if final.get("code_fingerprint") != ctx.fingerprint:
        return Result(TODO, f"kết quả cũ ({final.get('generated_at', '?')}) không còn khớp mã nguồn hiện tại — chạy lại --full-gates")
    if final.get("skip_network") is not False or not any(c.get("name") == "network_tests" for c in final.get("checks", [])):
        return Result(TODO, "lần chạy gần nhất bỏ qua test mạng Flower (--skip-network) nên không tính là full tests")
    if final.get("technical_acceptance") != "pass":
        failed = [c["name"] for c in final.get("checks", []) if c.get("status") != "pass"]
        return Result(TODO, f"technical_acceptance={final.get('technical_acceptance')}; nhóm lỗi: {failed}")
    return Result(DONE, f"technical_acceptance=pass lúc {final.get('generated_at')} (gồm test mạng, {len(final.get('test_files', []))} file test) trên đúng mã nguồn hiện tại")


def check_release(ctx: Ctx) -> Result:
    pending = [k for k in TECH_KEYS if RESULTS_CACHE.get(k, Result(TODO, "")).status != DONE]
    if pending:
        return Result(TODO, f"còn mục kỹ thuật chưa XONG: {pending}")
    return Result(DONE, "mọi mục kỹ thuật XONG; trước khi nộp vẫn phải `git status` sạch và gắn tag")


def check_provenance(ctx: Ctx) -> Result:
    prov = ctx.gates.get("provenance") or {}
    if prov.get("status") == "verified":
        return Result(DONE, "checkpoint canonical khớp manifest (hash + epoch + accuracy)")
    oids: dict[str, str] = {}
    for group in ctx.registry.get("artifact_groups", []):
        if group["group_id"] == "kaggle_run_lfs_pointer":
            oids = group.get("lfs_oids") or {}
    best = next((o for p, o in oids.items() if p.endswith("/best.pt")), "?")
    return Result(
        BLOCKED,
        f"provenance={prov.get('status')}; lỗi: {prov.get('critical_issues')}",
        needs=(
            f"HAI file checkpoint thật: FedAvg best.pt của run canonical (oid sha256:{best[:16]}…) và Centralized best — "
            "hiện chỉ là con trỏ Git-LFS / không có, repo không có remote nên không thể `git lfs pull`"
        ),
        unlock=[
            "# 1) tải hai file từ Kaggle, 2) đặt đúng vào slot canonical (đường mà provenance validator và registry đọc):",
            f"Copy-Item <fedavg_best.pt tải về> {reg.FEDAVG_BEST_PT}",
            f"Copy-Item <centralized_best.pt tải về> {reg.CENTRALIZED_BEST_PT}",
            ".\\.venv\\Scripts\\python.exe scripts\\update_status.py   # so state-hash với canonical_manifest",
        ],
    )


def check_protocol(ctx: Ctx) -> Result:
    text = (ROOT / "docs/EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    needed = [
        (ctx.registry.get("dataset_truth") or {}).get("protocol"),
        (ctx.registry.get("official_direction") or {}).get("protocol_id"),
    ]
    missing = [p for p in needed if p and p not in text]
    if missing:
        return Result(TODO, f"docs/EXPERIMENT_PROTOCOL.md chưa nêu protocol: {missing}")
    return Result(DONE, "mỗi protocol (scratch GĐ2 cục bộ / bundle mixed pretrained) có mục riêng trong EXPERIMENT_PROTOCOL.md")


def check_leakage(ctx: Ctx) -> Result:
    cur = ctx.models_of(reg.ROLE_CURRENT)
    unknown = [m["model_id"] for m in cur if m.get("leakage_audit") != "passed"]
    if cur and not unknown:
        return Result(DONE, "mọi model hiện hành có leakage_audit=passed")
    return Result(
        BLOCKED,
        f"{len(unknown)}/{len(cur)} model hiện hành có leakage_audit=unknown (bundle chỉ công bố validation_manifest_sha256, không kèm manifest train). [mục kiểm tay: chưa có script đọc kết quả audit để tự chuyển trạng thái]",
        needs="manifest train/val của bundle (hoặc dataset gốc để dựng lại split) để kiểm leaf_group không giao nhau",
        unlock=[
            ".\\.venv\\Scripts\\python.exe scripts\\verify_integrity.py --all   # chỉ audit split cục bộ partitions_*",
            "# audit split của bundle: cần manifest train từ nhánh train-* (chưa có trên máy)",
        ],
    )


def _per_class_report() -> dict[str, Any] | None:
    return load_json(reg.abs_path(reg.PER_CLASS_REPORT_JSON))


def _per_class_consistent(report: dict[str, Any]) -> bool:
    return all(all(src["consistency"].values()) for model in report["models"].values() for src in model.values())


def check_per_class(ctx: Ctx) -> Result:
    report = _per_class_report()
    if report is None:
        return Result(TODO, f"chưa có {reg.PER_CLASS_REPORT_JSON} — chạy scripts/report_per_class.py")
    if not _per_class_consistent(report):
        return Result(TODO, "báo cáo theo lớp không khớp tổng/top-1 công bố — kiểm lại bundle")
    s = report["summary"]
    cmp = report["fedavg_vs_centralized_plantdoc"]
    return Result(
        DONE,
        f"PlantDoc phủ {s['plantdoc_classes_covered']} lớp; {len(s['weak_classes_fedavg'])} lớp recall<50% "
        f"({s['weak_images_fedavg']}/{report['models']['FedAvg v7']['plantdoc']['count']} ảnh); FedAvg tệ hơn Centralized ở "
        f"{cmp['fedavg_worse_in']}/{cmp['classes_compared']} lớp (số công bố, đã kiểm nhất quán; xem mục 5)",
    )


def check_reeval(ctx: Ctx) -> Result:
    return Result(
        BLOCKED,
        "số theo lớp ở trên là số CÔNG BỐ; chưa tái đánh giá cục bộ nên chưa có confusion matrix đầy đủ. [mục kiểm tay]",
        needs="ảnh validation (269 PlantDoc + 4.344 PlantVillage) khớp validation_manifest của bundle + checkpoint đã kiểm hash",
        unlock=[
            "# chưa có script tái đánh giá — viết scripts/evaluate_published_per_class.py khi có ảnh validation",
            "# (cùng điều kiện dữ liệu với mục audit leakage; checkpoint bundle đã có và đã kiểm hash)",
            ".\\.venv\\Scripts\\python.exe scripts\\audit_realworld_inference.py --images <thư_mục_ảnh_ngoài>",
            ".\\.venv\\Scripts\\python.exe scripts\\evaluate_inference_audit.py --audit artifacts\\inference_audit.json --labels <nhãn.csv>",
        ],
    )


def check_confident_errors(ctx: Ctx) -> Result:
    report = _per_class_report()
    if report is None:
        return Result(TODO, f"chưa có {reg.PER_CLASS_REPORT_JSON} — chạy scripts/report_per_class.py")
    rows = []
    for name in ("Centralized v6", "FedAvg v7"):
        hc = report["models"][name]["plantdoc"]["high_confidence_wrong"]
        rows.append(f"{name}: {hc['count']}/{report['models'][name]['plantdoc']['count']} ảnh PlantDoc sai với confidence>{hc['threshold']} "
                    f"({hc['share_of_errors']:.0%} số lỗi)")
    return Result(DONE, "mức tổng hợp từ metrics công bố — " + "; ".join(rows) + " (xem mục 5)")


def check_calibration(ctx: Ctx) -> Result:
    return Result(
        BLOCKED,
        "chưa phân tích được TỪNG ảnh sai tự tin và chưa đo calibration (softmax chưa hiệu chuẩn, UI cảnh báo is_uncertain chỉ là heuristic). [mục kiểm tay]",
        needs="ảnh validation + checkpoint để liệt kê ảnh sai confidence cao, đo ECE/reliability và thử temperature scaling trên tập validation độc lập",
        unlock=["# cùng điều kiện dữ liệu với mục 'tái đánh giá cục bộ'; viết scripts/calibration_report.py khi có ảnh validation"],
    )


def check_multiseed(ctx: Ctx) -> Result:
    summary = load_json(reg.abs_path(reg.REPORT_EXPERIMENTS_DIR) / "multiseed_real_summary.json") or {}
    if summary.get("scientific_valid"):
        return Result(DONE, "multiseed_real_summary.json hợp lệ (mean ± std, 3 seed)")
    return Result(
        BLOCKED,
        "chưa có multi-seed THẬT (chỉ có bản smoke accuracy 0.0263 trên ảnh tổng hợp)",
        needs="dataset PlantVillage gốc + máy train (máy này CPU, không train)",
        unlock=[
            ".\\scripts\\run_real_multiseed_report.ps1 -DatasetRoot D:\\PlantVillage -CentralizedEpochs 10 -FederatedRounds 10 -Seeds 42,123,2026",
            "# hoặc train ngoài rồi import bundle: python scripts/import_branch_demo.py (xem docs/MODEL_REGISTRY.md)",
        ],
    )


def check_reports_synced(ctx: Ctx) -> Result:
    if RESULTS_CACHE.get("multiseed", Result(TODO, "")).status != DONE:
        return Result(BLOCKED, "phụ thuộc multi-seed thật", needs="chạy xong mục multi-seed rồi đồng bộ báo cáo",
                      unlock=[".\\.venv\\Scripts\\python.exe scripts\\update_status.py"])
    return Result(DONE, "báo cáo sinh lại từ registry sau multi-seed")


ITEMS = [
    Item("registry", "Registry v2 + danh mục mọi checkpoint trên đĩa", check_registry),
    Item("smoke_isolated", "Smoke cách ly khỏi slot canonical", check_smoke_isolated),
    Item("single_engine", "Một engine train (pipeline cũ gom vào legacy/)", check_single_engine),
    Item("web_split", "Web tách router/service, inference đi một cửa", check_web_split),
    Item("no_hardcode", "Không hardcode đường dẫn artifact", check_no_hardcode),
    Item("docs_clean", "Dọn tài liệu mâu thuẫn", check_docs_clean),
    Item("docs_numbers", "Số liệu trong docs không mâu thuẫn artifact", check_docs_numbers),
    Item("protocol", "Chốt một protocol khoa học cho mỗi thí nghiệm", check_protocol),
    Item("tests_isolated", "Test không ghi đè artifact đang track", check_tests_isolated),
    Item("full_tests", "Full tests pass trên mã nguồn hiện tại", check_full_tests),
    Item("provenance", "Kiểm chứng provenance & checkpoint canonical", check_provenance),
    Item("leakage", "Audit leakage theo từng run", check_leakage),
    Item("per_class", "Đánh giá PlantDoc theo từng lớp (từ số công bố)", check_per_class),
    Item("reeval", "Tái đánh giá cục bộ trên ảnh validation (xác nhận số công bố)", check_reeval),
    Item("confident_errors", "Phân tích lỗi confidence cao (mức tổng hợp, từ số công bố)", check_confident_errors),
    Item("calibration", "Phân tích từng ảnh sai tự tin + đo calibration", check_calibration),
    Item("multiseed", "Chạy multi-seed non-IID (seed 42/123/2026)", check_multiseed),
    Item("reports_synced", "Đồng bộ báo cáo & artifacts sau khi có run mới", check_reports_synced),
    Item("release", "Release sạch (mọi gate kỹ thuật XONG)", check_release),
]
# Mục kỹ thuật (không cần dữ liệu ngoài): "Release sạch" chỉ xét các mục này
TECH_KEYS = [
    "registry", "smoke_isolated", "single_engine", "web_split", "no_hardcode",
    "docs_clean", "docs_numbers", "protocol", "tests_isolated", "full_tests",
]
RESULTS_CACHE: dict[str, Result] = {}


def evaluate(ctx: Ctx) -> list[tuple[Item, Result]]:
    RESULTS_CACHE.clear()
    results: list[tuple[Item, Result]] = []
    # release phụ thuộc kết quả các mục khác -> tính sau cùng nhưng giữ thứ tự hiển thị
    order = [i for i in ITEMS if i.key != "release"] + [i for i in ITEMS if i.key == "release"]
    for item in order:
        RESULTS_CACHE[item.key] = item.check(ctx)
    for item in ITEMS:
        results.append((item, RESULTS_CACHE[item.key]))
    return results


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def results_table(models: list[dict[str, Any]], family: str) -> str:
    rows = [m for m in models if m["family"] == family]
    if not rows:
        return "_Không có model nào trong họ này._\n"
    lines = [
        "| Model | Nhãn | Checkpoint | Số liệu chính | Nguồn số liệu |",
        "|---|---|---|---|---|",
    ]
    for m in rows:
        if m.get("checkpoint_status") in ("hash_verified", "matches_manifest"):
            ck = "✅ có, hash ok"
        elif not m.get("checkpoint_exists") and m.get("checkpoint_path"):
            ck = "❌ không có trên máy"
        else:
            ck = str(m.get("checkpoint_status"))
        metrics = m.get("metrics") or {}
        parts: list[str] = []
        if family == reg.FAMILY_SCIENTIFIC_CANONICAL:
            if m["model_id"] == "local_only_canonical":
                parts.append(f"mean acc {fmt_pct(metrics.get('mean_accuracy'))} ± {fmt_pct(metrics.get('std_accuracy'))}")
            else:
                digits = 4 if m["model_id"] == "fedavg_canonical" else 2
                parts.append(f"test acc {fmt_pct(metrics.get('test_accuracy'), digits)}")
                parts.append(f"macro-F1 {fmt_pct(metrics.get('macro_f1'))}")
                parts.append(f"test {metrics.get('total_samples', 'n/a')} ảnh")
        elif family == reg.FAMILY_PUBLISHED_DEMO:
            pd, pv = metrics.get("plantdoc") or {}, metrics.get("plantvillage") or {}
            if pd:
                parts.append(f"PlantDoc {fmt_pct(pd.get('top1_accuracy'))} (n={pd.get('count')})")
            if pv:
                parts.append(f"PlantVillage {fmt_pct(pv.get('top1_accuracy'))} (n={pv.get('count')})")
        else:
            for k in ("test_accuracy", "last_accuracy", "val_accuracy"):
                if metrics.get(k) is not None:
                    parts.append(f"{k}={metrics[k]}")
        if not parts:
            parts.append("(không có số liệu)")
        web = "web: ✅" if m.get("web_visible") else "web: —"
        lines.append(
            f"| {m['display_name']} | {m.get('display_label') or m['status']} ({web}) | {ck} | "
            f"{'; '.join(parts)} | {m.get('metrics_source') or '—'} |"
        )
    return "\n".join(lines) + "\n"


def quality_section(registry: dict[str, Any]) -> list[str]:
    models = {m["model_id"]: m for m in registry["models"]}
    lines = ["| Model | PlantVillage | PlantDoc | Chênh | Đúng cây (PlantDoc) | Sai mà confidence > 0.9 |", "|---|---|---|---|---|---|"]
    gap = (registry.get("evidence_reports") or {}).get("domain_gap") or {}
    for name, mid in (("Centralized v6", "demo_centralized_v6"), ("FedAvg v7", "demo_fedavg_v7")):
        info = (gap.get("models") or {}).get(name) or {}
        pv, pd = info.get("plantvillage") or {}, info.get("plantdoc") or {}
        high = (((models.get(mid) or {}).get("metrics") or {}).get("overall") or {}).get("high_confidence_wrong_count_0_9")
        if pv and pd:
            lines.append(
                f"| {name} | {pv.get('accuracy_pct')}% | {pd.get('accuracy_pct')}% | "
                f"{info.get('plantvillage_minus_plantdoc_accuracy_pp')} điểm | {pd.get('crop_accuracy_pct')}% | {high if high is not None else '—'} |"
            )
    lines.append("")
    audit = (registry.get("evidence_reports") or {}).get("inference_audit")
    if audit:
        lines.append(
            f"- Audit ảnh ngoài (không có nhãn, chỉ dự đoán): {audit['image_count']} ảnh; Centralized và FedAvg "
            f"**bất đồng trên {audit['models_disagree_count']}/{audit['image_count']}** ảnh. Đây không phải accuracy."
        )
    fed = models.get("demo_fedavg_v7") or {}
    conf = (((fed.get("metrics") or {}).get("quality") or {}).get("top_confusions")) or []
    if conf:
        lines.append("- Nhầm lẫn lớn nhất của FedAvg v7 (metrics công bố): " + "; ".join(
            f"{c['target']} → {c['predicted']} ({c['count']} ảnh)" for c in conf[:3]))
    lines.append("- Nhận định: khoảng cách PlantVillage↔PlantDoc ~45 điểm ở cả hai phương pháp ⇒ vấn đề là tổng quát hóa giữa miền "
                 "dữ liệu, không phải riêng frontend. Chưa đủ bằng chứng để kết luận nguyên nhân (thiếu dữ liệu, nhãn, hay chiến lược train).")
    lines.append("- Số liệu trên là kết quả công bố của bundle, **chưa tái đánh giá cục bộ** (evidence: published_artifact_hash_verified_not_locally_reevaluated).")

    report = _per_class_report()
    if report and _per_class_consistent(report):
        fed_pd = report["models"]["FedAvg v7"]["plantdoc"]
        eb, hc = fed_pd["error_breakdown"], fed_pd["high_confidence_wrong"]
        lines += [
            "",
            "**Theo lớp trên PlantDoc (FedAvg v7, số công bố; sắp theo recall tăng dần):**",
            "",
            f"- PlantDoc chỉ phủ **{report['summary']['plantdoc_classes_covered']}** lớp của taxonomy; "
            f"{len(report['summary']['weak_classes_fedavg'])} lớp có recall < 50% chiếm {report['summary']['weak_images_fedavg']}/{fed_pd['count']} ảnh.",
            f"- Phân rã {fed_pd['count']} ảnh: đúng {eb['correct']} · sai loại cây {eb['wrong_crop']} · đúng cây nhưng sai bệnh {eb['right_crop_wrong_disease']}; "
            f"sai với confidence > {hc['threshold']}: {hc['count']} ảnh ({hc['share_of_errors']:.0%} số lỗi).",
            "",
            "| Lớp | Ảnh PlantDoc | FedAvg PlantDoc | Centralized PlantDoc | FedAvg PlantVillage | Chênh (điểm) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in report["fedavg_plantdoc_class_table"]:
            note = " ⚠ ít mẫu" if row["low_support"] else ""
            cen = row["centralized_plantdoc_recall"]
            lines.append(
                f"| {row['class']}{note} | {row['plantdoc_support']} | {row['fedavg_plantdoc_recall']:.0%} | "
                f"{'—' if cen is None else format(cen, '.0%')} | {row['fedavg_plantvillage_recall']:.0%} | {row['fedavg_domain_gap_pp']} |"
            )
        lines += ["", "- Không có trong số công bố (cần ảnh validation): " + "; ".join(report["not_available_without_images"]) + "."]
    return lines


def gates_section(registry: dict[str, Any], ctx: Ctx) -> str:
    gates = registry.get("gates") or {}
    lines = ["| Gate | Kết quả | Chi tiết |", "|---|---|---|"]
    p = gates.get("provenance") or {}
    lines.append(f"| Provenance checkpoint canonical | **{p.get('status', '—')}** | lỗi: {', '.join(p.get('critical_issues') or []) or 'không'} |")
    pd = gates.get("pre_demo_check") or {}
    lines.append(f"| Pre-demo check | **{pd.get('overall', '—')}** | {pd.get('source', '')} |")
    fa = gates.get("final_acceptance") or {}
    if fa:
        stale = ctx.final.get("code_fingerprint") != ctx.fingerprint
        note = " — **CŨ**: mã nguồn đã đổi kể từ lần chạy" if stale else ""
        lines.append(
            f"| Final acceptance | technical **{fa.get('technical', '—')}** / scientific **{fa.get('scientific', '—')}** | "
            f"{len(fa.get('scientific_blockers') or [])} blocker khoa học{note} |"
        )
    na = gates.get("network_artifact") or {}
    if na:
        scientific = na.get("require_real_clients") is True
        label = "PASS" if na.get("ok") else "CHƯA CÓ run mạng khoa học (artifact hiện tại là smoke kỹ thuật)" if not scientific else "FAIL"
        lines.append(f"| Network artifact | **{label}** | {na.get('source', '')} |")
    cat = gates.get("checkpoint_catalog") or {}
    if cat:
        lines.append(f"| Danh mục checkpoint | **{cat.get('status')}** | {cat.get('total_pt_files')} file .pt, vô danh: {len(cat.get('unclassified') or [])} |")
    return "\n".join(lines) + "\n"


def readiness_verdict(registry: dict[str, Any], results: list[tuple[Item, Result]]) -> str:
    gates = registry.get("gates") or {}
    prov = (gates.get("provenance") or {}).get("status")
    fa = gates.get("final_acceptance") or {}
    blocked = sum(1 for _, r in results if r.status == BLOCKED)
    pending = sum(1 for _, r in results if r.status == TODO)
    demo_ok = any(m["role"] == reg.ROLE_CURRENT and m.get("web_visible") for m in registry["models"])
    # Chỉ "sẵn sàng" khi MỌI mục XONG (kể cả full_tests đúng mã nguồn hiện tại) — không để một gate cũ đủ điều kiện
    if prov == "verified" and fa.get("scientific") == "pass" and not blocked and not pending:
        return "SẴN SÀNG NGHIỆM THU KHOA HỌC"
    if demo_ok:
        return (
            f"DEMO KỸ THUẬT DÙNG ĐƯỢC — CHƯA nghiệm thu khoa học "
            f"({blocked} mục BLOCKED cần dữ liệu/máy train, {pending} mục kỹ thuật chưa XONG)"
        )
    return "CHƯA SẴN SÀNG"


def read_journal() -> str:
    """Giữ NGUYÊN khối viết tay. Chỉ khi nhận ra khối kiểu cũ (có checklist tick tay) mới cắt lấy phần nhật ký."""
    if STATUS_MD.exists():
        text = STATUS_MD.read_text(encoding="utf-8")
        match = re.search(re.escape(JOURNAL_BEGIN) + r"\n(.*?)" + re.escape(JOURNAL_END), text, re.DOTALL)
        if match:
            block = match.group(1)
            if "### Checklist nghiệm thu" in block:  # định dạng cũ: bỏ checklist tick tay, giữ nhật ký
                marker = block.find("### Nhật ký")
                return (block[marker:] if marker != -1 else DEFAULT_JOURNAL).rstrip() + "\n"
            if block.strip():
                return block.rstrip() + "\n"
    return DEFAULT_JOURNAL


def render(registry: dict[str, Any], journal: str, ctx: Ctx, results: list[tuple[Item, Result]]) -> str:
    models = registry["models"]
    dataset = registry.get("dataset_truth") or {}
    split = dataset.get("split") or {}
    direction = registry.get("official_direction") or {}
    counts = {s: sum(1 for _, r in results if r.status == s) for s in (DONE, TODO, BLOCKED)}

    out: list[str] = []
    out += ["# PROJECT STATUS — Federated Learning PlantVillage", ""]
    out += [f"> **Cập nhật lúc:** {{TIMESTAMP}} · sinh tự bởi `scripts/update_status.py`"]
    out += [f"> **Đánh giá hiện tại:** {readiness_verdict(registry, results)}", ">"]
    out += ["> File này là nguồn thống kê DUY NHẤT về tiến độ. Checklist do máy chấm từ sự thật trên đĩa;",
            "> con số nào trong repo phải truy được về `artifacts/registry.json`. Không sửa tay — chỉ khối Nhật ký cuối file.", ""]

    out += ["## 1. Hướng chính thức", ""]
    out += [f"- **Luồng một chiều:** {direction.get('flow')}",
            f"- **Kết quả hiện hành:** họ `{direction.get('current_family')}`, protocol `{direction.get('protocol_id')}`.",
            "- **Engine trong repo:** `src/` (Flower demo, evaluator, web). Pipeline Kaggle cũ nằm ở `legacy/` — chỉ để tra cứu/tái lập lịch sử.",
            "- **Máy này:** CPU, không train; chỉ nhận kết quả đã train rồi kiểm hash + test.", ""]

    fam_count: dict[str, int] = {}
    for m in models:
        fam_count[m["family"]] = fam_count.get(m["family"], 0) + 1
    visible = [m for m in models if m.get("web_visible")]
    out += ["## 2. Trạng thái hệ thống (sự thật từ registry)", ""]
    out += [f"- Registry: **{len(models)}** model (" + ", ".join(f"{k}: {v}" for k, v in fam_count.items()) + ")",
            f"- Web được phép dùng: **{len(visible)}** model (đã qua đủ 5 điều kiện: hash, taxonomy, preprocessing, metrics↔protocol, forward test)",
            f"- Dataset cục bộ để tái lập: `{reg.PARTITIONS_DIR}` — split **{split.get('train')}/{split.get('val')}/{split.get('test')}** "
            f"({dataset.get('num_clients')} clients, seed {dataset.get('split_seed')}, protocol `{dataset.get('protocol')}`)",
            f"- Run Kaggle lịch sử: `{registry.get('canonical_run_id')}` — metrics CÓ, **checkpoint KHÔNG còn trên máy**", ""]
    rejected = [m for m in models if m["role"] in (reg.ROLE_CURRENT, reg.ROLE_REFERENCE) and not m.get("web_visible")]
    for m in rejected:
        out.append(f"- ⚠ `{m['model_id']}` bị cổng web chặn: {'; '.join(m.get('web_gate_failures') or [])}")
    if rejected:
        out.append("")

    out += [f"## 3. Checklist nghiệm thu — XONG {counts[DONE]}/{len(results)} · CHƯA {counts[TODO]} · BLOCKED {counts[BLOCKED]}", ""]
    out += ["| # | Hạng mục | Trạng thái | Bằng chứng (máy kiểm) |", "|---|---|---|---|"]
    mark = {DONE: "✅ XONG", TODO: "⬜ CHƯA", BLOCKED: "⛔ BLOCKED"}
    for number, (item, result) in enumerate(results, 1):
        out.append(f"| {number} | {item.title} | {mark[result.status]} | {result.detail.replace('|', '/')} |")
    out.append("")

    blocked = [(i, r) for i, r in results if r.status == BLOCKED]
    if blocked:
        out += ["### Việc đang BLOCKED — cần gì và mở khóa bằng lệnh nào", ""]
        for item, result in blocked:
            out += [f"**{item.title}**", f"- Cần: {result.needs}", "- Mở khóa:", "", "```powershell", *result.unlock, "```", ""]

    todo = [(i, r) for i, r in results if r.status == TODO]
    out += ["### Việc làm được ngay (chưa BLOCKED)", ""]
    if todo:
        out += [f"{n}. {i.title} — {r.detail}" for n, (i, r) in enumerate(todo, 1)]
    else:
        out.append("_(không còn mục kỹ thuật nào mở)_")
    out.append("")

    out += ["## 4. Kết quả chuẩn theo họ (không trộn lẫn)", ""]
    for family in (reg.FAMILY_PUBLISHED_DEMO, reg.FAMILY_SCIENTIFIC_CANONICAL, reg.FAMILY_NETWORK_DEMO,
                   reg.FAMILY_REFERENCE, reg.FAMILY_SMOKE):
        out += [f"### {FAMILY_TITLES[family]}", ""]
        desc = (registry.get("families") or {}).get(family, {}).get("description")
        if desc:
            out += [f"_{desc}_", ""]
        out.append(results_table(models, family))

    groups = registry.get("artifact_groups") or []
    if groups:
        out += ["### Checkpoint nằm ngoài danh sách model (nhóm có khai báo)", "", "| Nhóm | Số file | Vai trò | Mô tả |", "|---|---|---|---|"]
        out += [f"| `{g['group_id']}` | {g['count']} | {g['role']} | {g['description']} |" for g in groups]
        out.append("")

    out += ["## 5. Chất lượng nhận diện (số liệu công bố, chưa tái đánh giá)", ""]
    out += quality_section(registry)
    out.append("")

    out += ["## 6. Gates kỹ thuật", "", gates_section(registry, ctx)]

    out += ["## 7. Nhật ký (khối dưới đây TỰ TAY SỬA)", "", JOURNAL_BEGIN, journal.rstrip(), JOURNAL_END, ""]

    out += ["## 8. Cách đồng bộ", "", "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\update_status.py              # chuẩn: registry + gates + file này",
            ".\\.venv\\Scripts\\python.exe scripts\\update_status.py --skip-gates # chỉ sinh lại file từ JSON hiện có",
            ".\\.venv\\Scripts\\python.exe scripts\\update_status.py --full-gates # + pytest toàn bộ (chậm)",
            ".\\.venv\\Scripts\\python.exe scripts\\update_status.py --check      # kiểm file có đang lệch không (dùng cho pre-commit)",
            "```", ""]
    return "\n".join(out)


def build_status_report(registry: dict[str, Any], results: list[tuple[Item, Result]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "readiness": readiness_verdict(registry, results),
        "canonical_run_id": registry.get("canonical_run_id"),
        "dataset_truth": registry.get("dataset_truth"),
        "model_counts_by_family": {
            fam: sum(1 for m in registry["models"] if m["family"] == fam) for fam in reg.FAMILY_DESCRIPTIONS
        },
        "web_visible_models": [m["model_id"] for m in registry["models"] if m.get("web_visible")],
        "gates": registry.get("gates"),
        "checklist": [
            {"id": i.key, "title": i.title, "status": r.status, "detail": r.detail, "unlock": r.unlock}
            for i, r in results
        ],
        "source": "scripts/update_status.py",
    }


def stamp(body: str, previous: str | None, now: datetime) -> str:
    """Giữ nguyên dấu thời gian cũ nếu nội dung không đổi (để chạy lặp không tạo diff)."""
    if previous:
        old = TIMESTAMP_RE.search(previous)
        if old:
            old_value = old.group(0).split("** ", 1)[1].split(" · ")[0]
            if body.replace("{TIMESTAMP}", old_value) == previous:
                return previous
    return body.replace("{TIMESTAMP}", now.strftime("%Y-%m-%d %H:%M"))


def registry_staleness(registry: dict[str, Any]) -> list[str]:
    """registry.json có còn khớp đĩa không (hash checkpoint, file .pt mới/mất). Nhanh: không chạy forward."""
    from scripts import build_registry as br  # noqa: E402

    problems: list[str] = []
    known: set[str] = set()
    for model in registry["models"]:
        rel = model.get("checkpoint_path")
        if not rel:
            continue
        known.add(rel)
        path = ROOT / rel
        if bool(model.get("checkpoint_exists")) != path.exists():
            problems.append(f"{rel}: tồn tại trên đĩa khác registry")
        elif path.exists() and model.get("file_sha256") and br.file_sha256(path) != model["file_sha256"]:
            problems.append(f"{rel}: hash đổi so với registry")
    for group in registry.get("artifact_groups", []):
        known.update(group.get("paths", []))
    problems += [f"{rel}: file .pt chưa có trong registry" for rel in br._walk_pt_files() if rel not in known]
    return problems


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # console Windows (cp1258) không in được tiếng Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-gates", action="store_true", help="Chỉ đọc JSON hiện có, không chạy lại registry/gates")
    parser.add_argument("--full-gates", action="store_true", help="Chạy thêm final_acceptance (pytest, chậm)")
    parser.add_argument("--check", action="store_true", help="Không ghi gì; mã thoát 1 nếu PROJECT_STATUS.md hoặc registry.json đang lệch")
    args = parser.parse_args()

    crashed: list[str] = []
    if not args.skip_gates and not args.check:
        print("[update_status] refresh registry + gates ...")
        crashed = refresh_gates(full=args.full_gates)

    registry = load_json(ROOT / reg.REGISTRY_JSON)
    if registry is None:
        print(f"[update_status] LỖI: chưa có {reg.REGISTRY_JSON} — chạy không --skip-gates để sinh.", file=sys.stderr)
        return 1

    ctx = Ctx(registry)
    results = evaluate(ctx)
    previous = STATUS_MD.read_text(encoding="utf-8") if STATUS_MD.exists() else None
    body = render(registry, read_journal(), ctx, results)
    text = stamp(body, previous, datetime.now())

    if args.check:
        stale = registry_staleness(registry)
        for problem in stale:
            print(f"[update_status] registry lệch đĩa: {problem}", file=sys.stderr)
        if stale or previous != text:
            print("[update_status] LỆCH so với trạng thái thật — chạy: python scripts/update_status.py", file=sys.stderr)
            return 1
        print("[update_status] PROJECT_STATUS.md và registry.json đang đúng.")
        return 0

    if previous != text:
        STATUS_MD.write_text(text, encoding="utf-8", newline="\n")
        print("[update_status] wrote PROJECT_STATUS.md")
    else:
        print("[update_status] PROJECT_STATUS.md không đổi")
    STATUS_JSON.write_text(
        json.dumps(build_status_report(registry, results), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[update_status] wrote {reg.STATUS_REPORT_JSON}")
    if crashed:
        print(f"[update_status] CÓ GATE BỊ CRASH (kết quả có thể cũ): {crashed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
