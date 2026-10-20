"""Bằng chứng Flower có chạy thật không: cài đặt, phiên gần nhất, 5 client, các vòng, chuỗi mã băm, checkpoint.

Không import `flwr` (nặng): chỉ hỏi metadata. Không chạy gì: chỉ đọc các tệp mà một phiên Flower đã ghi
(`session.json`, `history.json`, `progress.json`, `global.pt`) trong `artifacts/web_runs/network_smoke/`.
Máy chủ chỉ báo "đang hoạt động" khi đã có MỘT phiên hoàn tất với đủ 5 client, mọi vòng nhận 5/5 cập nhật hợp lệ và
chuỗi mã băm khớp từ vòng đầu đến checkpoint cuối; "Server Online" hay cổng đang mở không đủ.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src import registry as reg

EXPECTED_CLIENTS = 5
FLOWER_PORT = 8089
PROBE_PORT = 8090
SMOKE_KIND = "flower_localhost_synthetic_smoke"

WORKING, NOT_VERIFIED, BROKEN = "working", "not_verified", "broken"


def runs_root() -> Path:
    return reg.abs_path(reg.NETWORK_SMOKE_RUNS_DIR)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item(item_id: str, ok: Optional[bool], label: str, detail: str) -> Dict[str, Any]:
    return {"id": item_id, "ok": ok, "label": label, "detail": detail}


# ---------------------------------------------------------------------------
# 1. flwr có cài cho đúng Python của web không
# ---------------------------------------------------------------------------
def flwr_status() -> Dict[str, Any]:
    """`sys.executable` là Python mà `NetworkDemoManager` dùng để sinh 6 tiến trình, nên phải kiểm đúng môi trường này."""
    python = sys.version.split()[0]
    if importlib.util.find_spec("flwr") is None:
        return {"installed": False, "version": None, "python": python,
                "error": "Python của máy chủ web không có gói flwr nên server và client Flower không thể khởi động."}
    try:
        version = importlib.metadata.version("flwr")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "python": python, "error": "Không đọc được phiên bản flwr."}
    return {"installed": True, "version": version, "python": python, "error": None}


# ---------------------------------------------------------------------------
# 2. phiên gần nhất (kể cả thất bại) và phiên hoàn tất gần nhất
# ---------------------------------------------------------------------------
def _run_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name, reverse=True)


def _log_tail(path: Path) -> Optional[str]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return None
    return lines[-1][:240] if lines else None


def attempt_summary(run_dir: Path) -> Dict[str, Any]:
    session = _read_json(run_dir / "session.json") or {}
    summary = {
        "name": run_dir.name, "status": session.get("status"), "started_at_utc": session.get("started_at_utc"),
        "finished_at_utc": session.get("finished_at_utc"), "error": session.get("error"), "rounds": session.get("rounds"),
        "client_processes": session.get("client_processes"),
    }
    if session.get("status") in ("failed",) and not summary["error"]:
        summary["error"] = _log_tail(run_dir / "server.log")
    if session.get("status") == "failed":
        summary["log_tail"] = _log_tail(run_dir / "server.log")
    return summary


def _is_valid_completed(run_dir: Path, session: Dict[str, Any], history: Dict[str, Any]) -> bool:
    audits = history.get("audit_rounds")
    return bool(
        session.get("kind") == SMOKE_KIND and session.get("status") == "completed"
        and session.get("client_processes") == EXPECTED_CLIENTS
        and isinstance(audits, list) and len(audits) >= 1 and len(audits) == session.get("rounds")
        and [a.get("round") for a in audits] == list(range(1, len(audits) + 1))
        and (run_dir / "global.pt").is_file()
    )


def latest_completed_run(root: Optional[Path] = None) -> Optional[Path]:
    for run_dir in _run_dirs(root or runs_root()):
        session, history = _read_json(run_dir / "session.json"), _read_json(run_dir / "history.json")
        if session and history and _is_valid_completed(run_dir, session, history):
            return run_dir
    return None


def newest_run_dir(root: Optional[Path] = None) -> Optional[Path]:
    """Thư mục run mới nhất kể cả thất bại: tên mã hoá UTC nên tên lớn nhất là phiên gần nhất."""
    dirs = _run_dirs(root or runs_root())
    return dirs[0] if dirs else None


# ---------------------------------------------------------------------------
# 3. bằng chứng của một phiên hoàn tất
# ---------------------------------------------------------------------------
def checkpoint_model_hash(path: Path) -> Optional[str]:
    """`model_hash` ghi trong checkpoint, đọc bằng bộ nạp an toàn (weights_only); None nếu không đọc được."""
    try:
        from src.common.checkpoint_io import load_checkpoint

        payload = load_checkpoint(path)
    except Exception:  # noqa: BLE001 — tệp hỏng không được làm sập trang kiểm tra
        return None
    value = payload.get("model_hash") if isinstance(payload, dict) else None
    return str(value) if value else None


def run_evidence(run_dir: Path, read_checkpoint_hash: Callable[[Path], Optional[str]] = checkpoint_model_hash) -> Dict[str, Any]:
    session = _read_json(run_dir / "session.json") or {}
    history = _read_json(run_dir / "history.json") or {}
    progress = _read_json(run_dir / "progress.json") or {}
    audits = [a for a in history.get("audit_rounds", []) if isinstance(a, dict)]
    rounds = [{
        "round": a.get("round"), "accepted": a.get("accepted_clients"), "suspicious": a.get("suspicious_clients"),
        "rejected": a.get("rejected_clients"), "fit_ms": a.get("round_fit_elapsed_ms"), "mean_rtt_ms": a.get("mean_probe_rtt_ms"),
        "max_rtt_ms": a.get("max_probe_rtt_ms"), "parent_hash": a.get("parent_model_hash"), "global_hash": a.get("global_model_hash"),
        "clients": len(a.get("clients") or []),
    } for a in audits]

    connected = progress.get("connected_clients")
    if connected is None and rounds:
        connected = max(int(r["clients"] or 0) for r in rounds)

    chain: Optional[bool] = None
    if rounds and all(r["global_hash"] for r in rounds):
        links = [rounds[i]["parent_hash"] == rounds[i - 1]["global_hash"] for i in range(1, len(rounds))]
        chain = all(links) if links else True
        if not all(r["parent_hash"] for r in rounds[1:]):
            chain = None

    checkpoint_path = run_dir / "global.pt"
    final_hash = history.get("final_model_hash")
    ckpt_hash = read_checkpoint_hash(checkpoint_path) if checkpoint_path.is_file() else None
    final_matches: Optional[bool] = None
    if final_hash and rounds and rounds[-1]["global_hash"]:
        final_matches = (final_hash == rounds[-1]["global_hash"]) and (ckpt_hash is None or ckpt_hash == final_hash)
    return {
        "name": run_dir.name, "status": session.get("status"), "started_at_utc": session.get("started_at_utc"),
        "finished_at_utc": session.get("finished_at_utc"), "configured_rounds": session.get("rounds"),
        "connected_clients": connected, "rounds": rounds, "hash_chain_ok": chain, "final_hash": final_hash,
        "final_matches_checkpoint": final_matches,
        "checkpoint": {
            "present": checkpoint_path.is_file(),
            "bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else None,
            "sha256": _sha256(checkpoint_path) if checkpoint_path.is_file() else None,
            "model_hash": ckpt_hash,
        },
        "log_check": log_crosscheck(run_dir, len(rounds)) if rounds else None,
        "scientific_valid": False,
        "note": "Phiên dùng ảnh nhiễu ngẫu nhiên: chỉ chứng minh hệ thống chạy thật, không đo độ chính xác.",
    }


# ---------------------------------------------------------------------------
# 3b. đối chiếu độc lập từ log gốc
# ---------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_RESULTS_RE = re.compile(r"(aggregate_fit|aggregate_evaluate): received (\d+) results and (\d+) failures")
_FINISHED_RE = re.compile(r"Run finished (\d+) round\(s\)")


def _clean_lines(path: Path) -> Optional[List[str]]:
    try:
        return _ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).splitlines()
    except OSError:
        return None


def log_crosscheck(run_dir: Path, rounds: int) -> Dict[str, Any]:
    """Đếm lại từ log do chính Flower/tiến trình client in ra, không dùng history.json.

    history.json là bản tóm tắt do mã của ta ghi; server.log và client_XX.log là output của tiến trình thật. Hai nguồn
    này phải khớp nhau thì mới kết luận Flower đã chạy: server gộp đủ 5 kết quả mỗi vòng cả fit lẫn evaluate, và mỗi
    client thật sự nhận `rounds` lệnh train + `rounds` lệnh evaluate.
    """
    problems: List[str] = []
    server = _clean_lines(run_dir / "server.log")
    fit: List[tuple] = []
    evaluate: List[tuple] = []
    finished: Optional[int] = None
    if server is None:
        problems.append("thiếu server.log")
    else:
        for line in server:
            match = _RESULTS_RE.search(line)
            if match:
                (fit if match.group(1) == "aggregate_fit" else evaluate).append((int(match.group(2)), int(match.group(3))))
            done = _FINISHED_RE.search(line)
            if done:
                finished = int(done.group(1))
        if len(fit) != rounds or any(ok != EXPECTED_CLIENTS or bad for ok, bad in fit):
            problems.append(f"server.log: aggregate_fit {fit} (cần {rounds} vòng × {EXPECTED_CLIENTS} kết quả, 0 lỗi)")
        if len(evaluate) != rounds or any(ok != EXPECTED_CLIENTS or bad for ok, bad in evaluate):
            problems.append(f"server.log: aggregate_evaluate {evaluate} (cần {rounds} vòng × {EXPECTED_CLIENTS} kết quả, 0 lỗi)")
        if finished != rounds:
            problems.append(f"server.log: 'Run finished' ghi {finished} vòng, cần {rounds}")
    clients: List[Dict[str, Any]] = []
    for index in range(EXPECTED_CLIENTS):
        lines = _clean_lines(run_dir / f"client_{index:02d}.log")
        if lines is None:
            problems.append(f"thiếu client_{index:02d}.log")
            clients.append({"name": f"client_{index:02d}", "train": None, "evaluate": None})
            continue
        train = sum("Received: train message" in line for line in lines)
        evaluated = sum("Received: evaluate message" in line for line in lines)
        clients.append({"name": f"client_{index:02d}", "train": train, "evaluate": evaluated})
        if train != rounds or evaluated != rounds:
            problems.append(f"client_{index:02d}.log: nhận {train} lệnh train / {evaluated} lệnh evaluate, cần {rounds}/{rounds}")
    return {"ok": not problems, "problems": problems, "fit": fit, "evaluate": evaluate, "finished_rounds": finished,
            "clients": clients, "rounds": rounds}


# ---------------------------------------------------------------------------
# 4. tổng hợp
# ---------------------------------------------------------------------------
def build_health(snapshot: Optional[Dict[str, Any]] = None, port_open: Optional[Callable[[int], bool]] = None,
                 read_checkpoint_hash: Callable[[Path], Optional[str]] = checkpoint_model_hash) -> Dict[str, Any]:
    root = runs_root()
    flwr = flwr_status()
    dirs = _run_dirs(root)
    newest = attempt_summary(dirs[0]) if dirs else None
    completed_dir = latest_completed_run(root)
    evidence = run_evidence(completed_dir, read_checkpoint_hash) if completed_dir else None
    live = snapshot or {}
    active = bool(live.get("active"))

    items: List[Dict[str, Any]] = []
    items.append(_item("flwr", flwr["installed"], "Gói flwr cài cho Python của máy chủ web",
                       f"flwr {flwr['version']} · Python {flwr['python']}" if flwr["installed"] else flwr["error"]))

    if port_open is not None:
        busy = [p for p in (FLOWER_PORT, PROBE_PORT) if port_open(p)]
        ports_ok = (not busy) or active
        items.append(_item("ports", ports_ok, f"Cổng Flower {FLOWER_PORT} và RTT {PROBE_PORT}",
                           "đang do phiên demo dùng" if busy and active else "trống, sẵn sàng khởi động" if not busy
                           else f"cổng {', '.join(map(str, busy))} đang bị tiến trình khác chiếm; phiên mới sẽ không khởi động được"))

    if newest is None:
        items.append(_item("attempt", None, "Phiên Flower gần nhất", "chưa có phiên nào được chạy trên máy chủ này"))
    else:
        detail = f"{newest['name']} · {newest['status'] or 'không rõ'}"
        if newest.get("error"):
            detail += f" · {newest['error']}"
        items.append(_item("attempt", {"completed": True, "failed": False}.get(newest["status"]), "Phiên Flower gần nhất", detail))

    if evidence is None:
        for item_id, label in (("clients", f"{EXPECTED_CLIENTS}/{EXPECTED_CLIENTS} client kết nối"), ("rounds", "Mọi vòng nhận đủ cập nhật hợp lệ"),
                               ("logs", "Log gốc của server và 5 client khớp với bản tóm tắt"),
                               ("hashes", "Chuỗi mã băm khớp đến checkpoint")):
            items.append(_item(item_id, None, label, "chưa có phiên hoàn tất nào để kiểm"))
    else:
        rounds = evidence["rounds"]
        clients_ok = evidence["connected_clients"] == EXPECTED_CLIENTS
        items.append(_item("clients", clients_ok, f"{EXPECTED_CLIENTS}/{EXPECTED_CLIENTS} client kết nối",
                           f"{evidence['connected_clients']}/{EXPECTED_CLIENTS} client trong phiên {evidence['name']}"))
        rounds_ok = bool(rounds) and all(r["accepted"] == EXPECTED_CLIENTS and not r["rejected"] for r in rounds)
        items.append(_item("rounds", rounds_ok, "Mọi vòng nhận đủ cập nhật hợp lệ",
                           f"{len(rounds)}/{evidence['configured_rounds']} vòng xong; nhận " + ", ".join(f"{r['accepted']}/{EXPECTED_CLIENTS}" for r in rounds)))
        check = evidence["log_check"]
        if check is None:
            items.append(_item("logs", None, "Log gốc của server và 5 client khớp với bản tóm tắt", "phiên chưa có vòng nào để đối chiếu"))
        elif check["ok"]:
            items.append(_item("logs", True, "Log gốc của server và 5 client khớp với bản tóm tắt",
                               f"server gộp {EXPECTED_CLIENTS}/{EXPECTED_CLIENTS} kết quả mỗi vòng ({check['rounds']} vòng, fit và evaluate); "
                               f"mỗi client nhận {check['rounds']} lệnh train + {check['rounds']} lệnh evaluate"))
        else:
            items.append(_item("logs", False, "Log gốc của server và 5 client khớp với bản tóm tắt", "; ".join(check["problems"])[:400]))
        chain, final = evidence["hash_chain_ok"], evidence["final_matches_checkpoint"]
        if chain is None and final is None:
            items.append(_item("hashes", None, "Chuỗi mã băm khớp đến checkpoint", "phiên cũ chưa ghi mã băm; chạy lại một phiên để có bằng chứng này"))
        else:
            hashes_ok = chain is not False and final is not False
            ckpt = evidence["checkpoint"]
            items.append(_item("hashes", hashes_ok, "Chuỗi mã băm khớp đến checkpoint",
                               f"vòng {len(rounds)} → mô hình cuối → global.pt (SHA-256 {str(ckpt['sha256'])[:16]}…, {ckpt['bytes']} byte)" if hashes_ok
                               else "mã băm giữa các vòng hoặc checkpoint không khớp"))

    if active:
        items.append(_item("live", True, "Đang có phiên chạy",
                           f"{live.get('client_processes_running', 0)} client đang chạy · {live.get('connected_clients', 0)}/{EXPECTED_CLIENTS} đã kết nối"
                           + (f" · vòng {live['round_in_progress']}" if live.get("round_in_progress") else "")))

    verdict, summary = _verdict(flwr, newest, evidence, items)
    return {"verdict": verdict, "summary": summary, "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "flwr": flwr, "items": items, "latest_attempt": newest, "evidence": evidence, "live": _live_view(live)}


def _live_view(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {key: snapshot.get(key) for key in ("active", "client_processes_running", "connected_clients", "round_in_progress",
                                                "round_completed", "server_running", "mean_probe_rtt_ms")}


def _verdict(flwr: Dict[str, Any], newest: Optional[Dict[str, Any]], evidence: Optional[Dict[str, Any]],
             items: List[Dict[str, Any]]) -> tuple[str, str]:
    if not flwr["installed"]:
        return BROKEN, "Flower chưa chạy được: " + str(flwr["error"])
    if newest and newest["status"] == "failed":
        return BROKEN, f"Phiên Flower gần nhất ({newest['name']}) bị lỗi: {newest.get('error') or 'không rõ nguyên nhân'}."
    core = {i["id"]: i["ok"] for i in items}
    if evidence and core.get("clients") and core.get("rounds") and core.get("logs") and core.get("hashes") is not False:
        return WORKING, (f"Flower đã chạy thật: {evidence['connected_clients']} client, {len(evidence['rounds'])} vòng, "
                         "log gốc của server và từng client khớp với kết quả, mọi vòng nhận đủ cập nhật hợp lệ" + (", chuỗi mã băm khớp đến checkpoint." if core.get("hashes") else "."))
    if evidence:
        return BROKEN, "Phiên Flower gần nhất hoàn tất nhưng bằng chứng không đạt (thiếu client, thiếu cập nhật, log gốc không khớp hoặc mã băm lệch)."
    return NOT_VERIFIED, "Flower đã cài nhưng chưa có phiên nào chạy xong trên máy chủ này; hãy chạy một phiên kiểm tra để có bằng chứng."
