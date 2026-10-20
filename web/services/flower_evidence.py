"""Bằng chứng thô cho từng mục trên trang Flower: dòng lệnh đã chạy thật + output của nó, đọc lúc người xem bấm mở.

Mục đích: một dòng "đạt ✓" không thuyết phục được hội đồng. Mỗi mục vì thế kèm cách kiểm (assertion), lệnh đã chạy và
kết quả nguyên bản (log của server/client, nội dung session/history/progress, mã băm tính lại từ tệp).

An toàn: `item_id` chỉ nhận đúng danh sách ITEM_IDS, mọi đường dẫn do máy chủ tự tính từ thư mục run nên không có đường
dẫn nào đến từ client. Lệnh chạy bằng argv tĩnh (không qua shell), có timeout, chỉ đọc, output bị che đường dẫn gốc.
Endpoint chỉ được gọi khi bấm mở (không gọi theo poll) và có cache để không lặp lại subprocess.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src import registry as reg
from web.services import flower_health as fh

ITEM_IDS = ("flwr", "ports", "attempt", "clients", "rounds", "hashes", "live")

MAX_BLOCK_CHARS = 40_000   # một block không được kéo dài bất tận khi có nhiều người mở liên tục
COMMAND_TIMEOUT_S = 6.0
LOG_TAIL_LINES = 500
CACHE_TTL_S = 15.0
LIVE_TTL_S = 2.0

ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()


# ---------------------------------------------------------------------------
# khối hiển thị
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prefixes() -> Tuple[Tuple[str, str], ...]:
    """Bảng thay thế đường dẫn: người xem cần thấy nơi chạy lệnh để tự kiểm, nhưng không cần thấy cây thư mục của máy chủ.

    Có cả dạng backslash đã nhân đôi vì vài công cụ (sha256sum của Git for Windows) in tên tệp với hai dấu gạch chéo,
    nên phép thay thế đơn sẽ bỏ lọt đường dẫn tuyệt đối.
    """
    runs = str(fh.runs_root())
    root, root_fwd = str(ROOT), str(ROOT).replace("\\", "/")
    return ((root + os.sep, ""), (root.replace("\\", "\\\\") + "\\\\", ""), (root_fwd + "/", ""), (root, "."),
            (runs + os.sep, reg.NETWORK_SMOKE_RUNS_DIR + "/"), (runs, reg.NETWORK_SMOKE_RUNS_DIR),
            (str(HOME) + os.sep, "~/"), (str(HOME), "~"))


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Log của Flower tô màu bằng mã ANSI (`ESC[92mINFO ESC[0m`); trên trang web chúng hiện thành `[92mINFO [0m` khó đọc."""
    return _ANSI.sub("", text)


def _redact(text: str) -> str:
    out = str(text)
    for needle, replacement in _prefixes():
        if needle and needle in out:
            out = out.replace(needle, replacement)
    return out


def _runnable_path(path: Path) -> str:
    """Đường dẫn đưa vào lệnh: tương đối từ thư mục ứng dụng, để output in ra đúng thứ người xem chạy lại được."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _clip(text: str) -> Tuple[str, bool]:
    if len(text) <= MAX_BLOCK_CHARS:
        return text, False
    return text[:MAX_BLOCK_CHARS], True


def _note(text: str) -> Dict[str, Any]:
    return {"kind": "note", "text": _redact(text)}


def _shown_path(path: Path) -> str:
    return _redact(str(path))


def _file_block(title: str, path: Path, *, tail_lines: int = LOG_TAIL_LINES, prettify_json: bool = False) -> Dict[str, Any]:
    block: Dict[str, Any] = {"kind": "file", "title": title, "path": _shown_path(path), "content": "",
                             "truncated": False, "line_count": 0, "source": None}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
    except OSError as error:
        # Python nhét nguyên đường dẫn tuyệt đối vào thông báo lỗi: phải che trước khi cho người xem thấy.
        block["content"] = f"(không đọc được tệp: {_redact(str(error))})"
        return block
    if prettify_json:
        try:
            raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except ValueError:
            pass  # tệp hỏng thì in nguyên bản, để người xem thấy đúng thứ nằm trên đĩa
    colored = bool(_ANSI.search(raw))
    raw = _strip_ansi(raw)
    lines = raw.splitlines()
    cut = len(lines) > tail_lines
    content, clipped = _clip("\n".join(lines[-tail_lines:]) if cut else raw)
    block.update({"content": _redact(content), "truncated": clipped or cut, "line_count": len(lines),
                  "source": f"{path.name} ({size} byte)"})
    if prettify_json:
        block["note"] = "In lại thụt gọn từ tệp gốc, không sửa giá trị."
    elif colored:
        block["note"] = "Đã bỏ mã màu ANSI của terminal; chữ giữ nguyên."
    return block


def _records_block(title: str, columns: List[Tuple[str, str]], rows: List[Dict[str, Any]], *, caption: str = "") -> Dict[str, Any]:
    """giá trị chuỗi trong bảng cũng phải che đường dẫn: dòng lệnh spawn chứa `sys.executable` tuyệt đối."""
    def clean(value: Any) -> Any:
        return _redact(value) if isinstance(value, str) else value

    return {"kind": "records", "title": title, "caption": _redact(caption),
            "columns": [{"key": key, "label": label} for key, label in columns],
            "rows": [{key: clean(row.get(key)) for key, _ in columns} for row in rows]}


# ---------------------------------------------------------------------------
# chạy một lệnh read-only
# ---------------------------------------------------------------------------
def run_command(title: str, argv: List[str], *, timeout: float = COMMAND_TIMEOUT_S,
                keep: Optional[Tuple[str, ...]] = None,
                runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> Dict[str, Any]:
    """Chạy argv tĩnh (không shell hoá) rồi chụp exit code + stdout/stderr nguyên bản để người xem tự đối chiếu."""
    block: Dict[str, Any] = {"kind": "command", "title": title, "command": _redact(shlex.join(argv)), "cwd": ".",
                             "platform": sys.platform, "exit_code": None, "output": "", "truncated": False}
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"  # máy chủ Windows/Linux đều có thể in tiếng Việt từ tiến trình con
    try:
        done = runner(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                      timeout=timeout, cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        block["output"] = f"(lệnh không trả lời sau {timeout:g} giây — đã dừng)"
        return block
    except OSError as error:
        block["output"] = f"(không chạy được lệnh trên máy chủ này: {error})"
        return block
    block["exit_code"] = done.returncode
    out = _strip_ansi((done.stdout or "") + (f"\n[stderr]\n{done.stderr}" if (done.stderr or "").strip() else ""))
    if keep:
        lines = out.splitlines()
        header = lines[:1]  # ss/netstat luôn in tiêu đề cột; giữ nguyên dòng đầu để người xem đọc đúng cột
        matched = [line for line in lines[1:] if any(token in line for token in keep)]
        block["output"] = _redact("\n".join(header + matched) or f"(không có dòng nào chứa {' hoặc '.join(keep)})")
        block["note"] = _redact(f"Lọc output theo {' / '.join(keep)}; tiêu đề cột giữ nguyên dòng đầu tiên của lệnh.")
        return block
    content, clipped = _clip(out.strip())
    block["output"] = _redact(content)
    block["truncated"] = clipped
    return block


def _python_snippet(title: str, source: str) -> Dict[str, Any]:
    return run_command(title, [sys.executable, "-c", source])


# ---------------------------------------------------------------------------
# 1. flwr
# ---------------------------------------------------------------------------
# Nháy kép bên trong đoạn mã: shlex.quote phải lồng nháy đơn cho lệnh hiển thị, toàn là nháy kép thì dòng lệnh in ra đọc được.
_FLWR_SOURCE = (
    'import sys, flwr, importlib.metadata as md\n'
    'print("sys.executable        =", sys.executable)\n'
    'print("Python                =", sys.version.split()[0])\n'
    'print("flwr.__version__      =", getattr(flwr, "__version__", "(không có attribute)"))\n'
    'print("metadata version      =", md.version("flwr"))\n'
    'print("flwr package location =", flwr.__file__)\n'
)


def _flwr_blocks() -> List[Dict[str, Any]]:
    status = fh.flwr_status()
    return [
        _python_snippet("Hỏi đúng Python của máy chủ web xem nó thấy flwr ở đâu", _FLWR_SOURCE),
        run_command("Sổ đăng ký gói của pip (Name / Version / Location)", [sys.executable, "-m", "pip", "show", "flwr"]),
        _note(f"Máy chủ kiểm bằng `importlib.util.find_spec('flwr')` rồi `importlib.metadata.version('flwr')` ngay trong "
              f"`sys.executable` ({status['python']}) — tức môi trường mà `NetworkDemoManager` dùng để sinh 6 tiến trình "
              f"server + 5 client, không phải một môi trường khác có flwr. Kết quả: flwr {status['version']}."),
    ]


# ---------------------------------------------------------------------------
# 2. cổng
# ---------------------------------------------------------------------------
# Placeholder @...@ thay vì format(): đoạn sinh ra chứa f-string nên dấu ngoặc nhọn sẽ bị .format() ăn mất.
_PORTS_SOURCE = (
    'import socket, time\n'
    'for port in (@FLOWER@, @PROBE@):\n'
    '    t = time.perf_counter()\n'
    '    try:\n'
    '        with socket.create_connection(("127.0.0.1", port), timeout=0.3):\n'
    '            print(f"{port}: CÓ tiến trình lắng nghe (kết nối được, " + format((time.perf_counter() - t) * 1000, ".1f") + " ms)")\n'
    '    except OSError as e:\n'
    '        print(f"{port}: TRỐNG ({type(e).__name__}, " + format((time.perf_counter() - t) * 1000, ".1f") + " ms)")\n'
).replace("@FLOWER@", str(fh.FLOWER_PORT)).replace("@PROBE@", str(fh.PROBE_PORT))


def _ports_blocks() -> List[Dict[str, Any]]:
    blocks = [_python_snippet(f"Thử kết nối TCP vào 127.0.0.1:{fh.FLOWER_PORT} và :{fh.PROBE_PORT}", _PORTS_SOURCE)]
    keep = (f":{fh.FLOWER_PORT}", f":{fh.PROBE_PORT}")
    if shutil.which("ss"):
        blocks.append(run_command("Tiến trình đang lắng nghe ở hai cổng (ss -ltnp)", ["ss", "-ltnp"], keep=keep))
    elif shutil.which("netstat"):
        blocks.append(run_command("Tiến trình đang lắng nghe ở hai cổng (netstat -ano -p tcp)", ["netstat", "-ano", "-p", "tcp"], keep=keep))
    blocks += [
        _note("Máy chủ dùng đúng phép thử `socket.create_connection(('127.0.0.1', cổng), timeout=0.3)` ở "
              "`web/state.py:check_port_open`. Ngoài ra `NetworkDemoManager.run()` gọi `socket.socket().connect_ex()` với "
              f"cả {fh.FLOWER_PORT} và {fh.PROBE_PORT} trước khi khởi động, và báo lỗi ngay nếu cổng đã bị chiếm."),
        _note("Hai cổng này bind trên 127.0.0.1 nên không mở ra ngoài mạng. 'Đạt' nghĩa là cổng trống để phiên mới bind, "
              "hoặc đang được chính phiên demo giữ khi có phiên chạy — chứ không phải 'có dịch vụ bên ngoài truy cập được'."),
    ]
    return blocks


# ---------------------------------------------------------------------------
# 3. phiên gần nhất
# ---------------------------------------------------------------------------
def _attempt_blocks(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_dir is None:
        return [_note("Thư mục `artifacts/web_runs/network_smoke/` chưa có phiên nào. Mỗi lần bấm Bắt đầu ở Phụ lục mục D, "
                      "máy chủ tạo một thư mục run mới chứa session.json, server.log, client_00..04.log, history.json, "
                      "progress.json và global.pt.")]
    session = fh._read_json(run_dir / "session.json") or {}
    blocks = [_file_block("session.json — cấu hình và trạng thái phiên", run_dir / "session.json", prettify_json=True)]
    commands = session.get("commands")
    if isinstance(commands, dict) and commands:
        rows = [{"name": name, "pid": (spec or {}).get("pid"), "command": shlex.join((spec or {}).get("argv") or [])}
                for name, spec in commands.items() if isinstance(spec, dict)]
        blocks.append(_records_block("Dòng lệnh máy chủ đã thực sự chạy", [("name", "Tiến trình"), ("pid", "PID"),
                                       ("command", "Lệnh")], rows,
                                     caption="Được ghi đúng lúc spawn, không phải lệnh viết tay để minh hoạ."))
    else:
        blocks.append(_note("Phiên này chạy trước khi máy chủ ghi dòng lệnh spawn vào session.json, nên bằng chứng lệnh là "
                            "các tệp log bên dưới: dòng đầu của server.log và client_XX.log in ra địa chỉ + cổng mà tiến "
                            "trình đã dùng để kết nối."))
    blocks.append(_file_block("server.log — log gốc của tiến trình Flower server", run_dir / "server.log"))
    return blocks


# ---------------------------------------------------------------------------
# 4. client
# ---------------------------------------------------------------------------
def _clients_blocks(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_dir is None:
        return [_note("Chưa có phiên hoàn tất nào để đọc bản ghi client.")]
    history = fh._read_json(run_dir / "history.json") or {}
    rows: List[Dict[str, Any]] = []
    for audit in history.get("audit_rounds", []):
        if not isinstance(audit, dict):
            continue
        for client in audit.get("clients") or []:
            if isinstance(client, dict):
                rows.append({**client, "round": audit.get("round")})
    columns = [("round", "Vòng"), ("client_id", "Client"), ("flower_cid", "Flower CID"), ("num_examples", "Ảnh"),
               ("local_train_time_s", "Giờ học (s)"), ("probe_rtt_ms", "RTT (ms)"), ("data_mode", "Dữ liệu"),
               ("partition_id", "Phân vùng"), ("received_update_norm", "Chuẩn cập nhật"),
               ("learnable_update_relative", "Chuẩn học được"), ("validation_reason", "Lý do nghiệm thu"), ("status", "Trạng thái")]
    blocks = [_records_block("Từng cập nhật server nhận được (history.json → audit_rounds[].clients)", columns, rows,
                             caption="flower_cid do Flower gán; probe_rtt_ms là thời gian đo round-trip thực sự sang client đó.")]
    logs = sorted(run_dir.glob("client_*.log"))
    for log in logs:
        blocks.append(_file_block(f"{log.name} — log gốc của tiến trình client", log, tail_lines=120))
    if not logs:
        blocks.append(_note("Phiên này không để lại client_XX.log."))
    blocks.append(_note("Mỗi client là một tiến trình Python riêng do `NetworkDemoManager` sinh bằng Popen, log của nó in "
                        "đúng số lượt 'Received: train message <UUID>' theo số vòng — không có được nếu chỉ mô phỏng trong "
                        "một tiến trình."))
    blocks.append(_note("Dòng 'DEPRECATED FEATURE: start_server()/start_client()' trong log là cảnh báo của Flower 1.36 về "
                        "cách khởi động kiểu cũ (nên chuyển sang flower-superlink/flower-supernode). Không phải lỗi: các vòng "
                        "vẫn chạy và dòng cần đọc là 'Received: train message'."))
    return blocks


# ---------------------------------------------------------------------------
# 5. các vòng
# ---------------------------------------------------------------------------
_ROUND_LOG_TOKENS = ("[ROUND", "[INIT]", "[SUMMARY]", "configure_fit", "aggregate_fit", "configure_evaluate",
                     "aggregate_evaluate", "Run finished", "num_rounds")


def _rounds_blocks(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_dir is None:
        return [_note("Chưa có phiên hoàn tất nào để đọc lịch sử vòng.")]
    history = fh._read_json(run_dir / "history.json") or {}
    rows = [a for a in history.get("audit_rounds", []) if isinstance(a, dict)]
    columns = [("round", "Vòng"), ("accepted_clients", "Nhận hợp lệ"), ("suspicious_clients", "Đáng ngờ"),
               ("rejected_clients", "Bị loại"), ("round_fit_elapsed_ms", "Thời gian vòng (ms)"),
               ("estimated_comm_overhead_ms", "Giao tiếp ước tính (ms)"), ("mean_probe_rtt_ms", "RTT trung bình (ms)"),
               ("max_probe_rtt_ms", "RTT lớn nhất (ms)"), ("mean_received_learnable_update_norm", "Chuẩn học được TB"),
               ("parent_model_hash", "Mã băm vào"), ("global_model_hash", "Mã băm ra")]
    blocks = [
        _records_block("audit_rounds nguyên bản trong history.json", columns, rows,
                       caption="estimated_comm_overhead_ms = thời gian vòng (từ lúc gửi cấu hình đến lúc gộp xong) trừ giờ học "
                                "lâu nhất của một client. Đây là ước tính thô, không phải số đo riêng gRPC: nó gồm cả truyền "
                                "weights, đóng/mở gói tensor và chờ client sẵn sàng, nên vòng 1 thường lớn hơn vòng sau "
                                "(client còn nạp mô hình lần đầu)."),
        _file_block("progress.json — tiến trình server ghi theo từng vòng", run_dir / "progress.json", prettify_json=True),
    ]
    log_block = _file_block("server.log — các dòng Flower in ra khi chạy từng vòng", run_dir / "server.log")
    kept = [line for line in (log_block["content"] or "").splitlines() if any(token in line for token in _ROUND_LOG_TOKENS)]
    if kept:
        log_block["content"] = _redact("\n".join(kept))
        log_block["truncated"] = False
        log_block["note"] = f"Lọc {len(kept)} dòng cấu hình vòng / số kết quả nhận được / tóm tắt từ server.log gốc."
    blocks.append(log_block)
    return blocks


# ---------------------------------------------------------------------------
# 6. chuỗi mã băm
# ---------------------------------------------------------------------------
def _sha_command(path: Path) -> Optional[Dict[str, Any]]:
    """Tính lại SHA-256 của checkpoint bằng một tiến trình riêng, độc lập với mã đang phục vụ trang này."""
    target = _runnable_path(path)
    if shutil.which("sha256sum"):
        return run_command("Tính lại SHA-256 của global.pt", ["sha256sum", target], timeout=20.0)
    if sys.platform == "win32" and shutil.which("certutil"):
        return run_command("Tính lại SHA-256 của global.pt", ["certutil", "-hashfile", target, "SHA256"], timeout=20.0)
    return _python_snippet("Tính lại SHA-256 của global.pt",
                           'import hashlib, pathlib\n'
                           f'print(hashlib.sha256(pathlib.Path({json.dumps(target)}).read_bytes()).hexdigest())\n')


_SHA_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")


def _first_sha256(output: str) -> Optional[str]:
    """Lấy mã SHA-256 đầu tiên trong output của sha256sum / certutil (certutil có thể tách hex bằng dấu cách)."""
    match = _SHA_RE.search(output or "")
    if match:
        return match.group(0).lower()
    for line in (output or "").splitlines():
        compact = line.replace(" ", "")
        if _SHA_RE.fullmatch(compact):
            return compact.lower()
    return None


def _hashes_blocks(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if run_dir is None:
        return [_note("Chưa có phiên hoàn tất nào có mã băm.")]
    history = fh._read_json(run_dir / "history.json") or {}
    audits = [a for a in history.get("audit_rounds", []) if isinstance(a, dict)]
    rows, previous = [], None
    for index, audit in enumerate(audits):
        parent, current = audit.get("parent_model_hash"), audit.get("global_model_hash")
        rows.append({"round": audit.get("round"), "parent_model_hash": parent, "global_model_hash": current,
                     "matches_previous": None if previous is None else parent == previous})
        previous = current
    checkpoint = run_dir / "global.pt"
    blocks = [_records_block("Chuỗi liên kết giữa các vòng (mã băm đầy đủ, không rút gọn)",
                             [("round", "Vòng"), ("parent_model_hash", "Mô hình đầu vào"),
                              ("global_model_hash", "Mô hình sau FedAvg"), ("matches_previous", "Khớp vòng trước?")], rows,
                             caption="Vòng r chỉ nối tiếp vòng r-1 nếu parent_model_hash của r đúng bằng global_model_hash của r-1.")]
    last_hash = audits[-1].get("global_model_hash") if audits else None
    final_hash = history.get("final_model_hash")
    ckpt_model_hash = fh.checkpoint_model_hash(checkpoint) if checkpoint.is_file() else None
    file_sha = fh._sha256(checkpoint) if checkpoint.is_file() else None
    sha_block = _sha_command(checkpoint) if checkpoint.is_file() else None
    # Băm lại bằng công cụ bên ngoài (sha256sum/certutil/python con) phải ra đúng giá trị mà máy chủ tự tính.
    tool_sha = _first_sha256(sha_block.get("output", "")) if sha_block else None

    def same(value: Optional[str], other: Optional[str]) -> Optional[bool]:
        return None if not value or not other else value == other

    blocks.append(_records_block("Mô hình cuối: ba nơi ghi mã băm phải trùng nhau, cộng SHA-256 của tệp", [
        ("node", "Đầu mối"), ("hash", "SHA-256"), ("matches", "Khớp?")], [
        {"node": f"global_model_hash của vòng cuối ({len(audits)})", "hash": last_hash, "matches": same(last_hash, final_hash)},
        {"node": "final_model_hash trong history.json", "hash": final_hash, "matches": same(final_hash, last_hash)},
        {"node": "model_hash ghi bên trong global.pt (nạp bằng weights_only)", "hash": ckpt_model_hash, "matches": same(ckpt_model_hash, final_hash)},
        {"node": "SHA-256 tính lại từ từng byte của global.pt", "hash": file_sha, "matches": same(file_sha, tool_sha)},
    ], caption="Ba dòng đầu là cùng một phép băm nội dung tensor, phải bằng nhau (cột Khớp? so với final_model_hash). Dòng cuối là "
               "băm cả container .pt nên khác ba dòng trên by design; cột Khớp? của nó so với kết quả lệnh tính lại bên dưới, "
               "để chứng minh tệp trên đĩa đúng là tệp đã được kiểm."))
    if checkpoint.is_file():
        if sha_block:
            blocks.append(sha_block)
        blocks.append(_note(f"Người xem tự kiểm trên máy chủ: `sha256sum <thư mục run>/{checkpoint.name}` rồi so với dòng "
                            "cuối của bảng. Thư mục run nằm ở `artifacts/web_runs/network_smoke/`."))
    else:
        blocks.append(_note("Phiên này không có global.pt."))
    blocks.append(_file_block("history.json — tệp cội của mọi con số ở trên", run_dir / "history.json", prettify_json=True))
    return blocks


# ---------------------------------------------------------------------------
# 7. phiên đang chạy
# ---------------------------------------------------------------------------
def _process_list_command(pids: List[int]) -> Optional[Dict[str, Any]]:
    pids = [p for p in pids if isinstance(p, int) and p > 0][:12]
    if not pids:
        return None
    if sys.platform == "win32" and shutil.which("tasklist"):
        # `ps` mà Git for Windows đặt trong PATH không nhận tham số chọn cột (`-o`), nên Windows dùng tasklist.
        block = run_command("tasklist trên đúng các PID mà máy chủ đã spawn", ["tasklist", "/FO", "CSV", "/NH"])
        kept = [line for line in (block.get("output") or "").splitlines() if any(f'"{p}",' in line for p in pids)]
        block["output"] = _redact("\n".join(kept) or f"(tasklist không trả về dòng nào cho PID {' / '.join(map(str, pids))})")
        block["note"] = _redact(f"Lọc tasklist theo PID {' / '.join(map(str, pids))}.")
        return block
    if shutil.which("ps"):
        return run_command("ps trên đúng các PID mà máy chủ đã spawn",
                           ["ps", "-o", "pid,ppid,etime,%cpu,%mem,args", "-p", ",".join(map(str, pids))])
    return None


def _live_blocks(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [p for p in (snapshot.get("processes") or []) if isinstance(p, dict)]
    blocks = [_records_block("Các tiến trình do NetworkDemoManager spawn (PID thật của hệ điều hành)",
                             [("name", "Tiến trình"), ("pid", "PID"), ("running", "Đang sống?"),
                              ("exit_code", "Mã thoát")], rows,
                             caption="server + client_00..04. Máy chủ gọi Popen() với sys.executable rồi poll() để biết "
                                     "tiến trình còn sống hay đã thoát với mã nào.")]
    listing = _process_list_command([int(r["pid"]) for r in rows if r.get("pid")])
    if listing:
        blocks.append(listing)
    run_dir_raw = snapshot.get("run_dir")
    if run_dir_raw:
        run_dir = Path(run_dir_raw)
        blocks.append(_file_block("progress.json đang được server ghi dồn", run_dir / "progress.json", tail_lines=120, prettify_json=True))
        blocks.append(_file_block("server.log đang được ghi trực tiếp", run_dir / "server.log", tail_lines=120))
    else:
        blocks.append(_note("Phiên đang chạy chưa có thư mục run."))
    return blocks


# ---------------------------------------------------------------------------
# tổng hợp
# ---------------------------------------------------------------------------
ASSERTIONS: Dict[str, str] = {
    "flwr": "đạt khi `importlib.util.find_spec('flwr')` thấy gói trong chính `sys.executable` của máy chủ web và "
            "`importlib.metadata.version('flwr')` đọc được phiên bản.",
    "ports": f"đạt khi {fh.FLOWER_PORT} và {fh.PROBE_PORT} trống, hoặc đang bị chiếm nhưng bởi đúng phiên demo này. "
             "Bị tiến trình khác chiếm mà không có phiên nào chạy → không đạt.",
    "attempt": "đạt khi session.json của phiên mới nhất có status = completed; failed → không đạt kèm dòng cuối server.log.",
    "clients": f"đạt khi connected_clients của phiên hoàn tất bằng đúng {fh.EXPECTED_CLIENTS}.",
    "rounds": f"đạt khi mọi vòng trong audit_rounds có accepted_clients = {fh.EXPECTED_CLIENTS} và rejected_clients = 0.",
    "hashes": "đạt khi parent_model_hash của vòng r bằng global_model_hash của vòng r-1 trên suốt chuỗi, và mã băm vòng "
              "cuối khớp giá trị ghi trong global.pt.",
    "live": "chỉ hiện khi có phiên đang chạy; đạt khi manager báo active = true.",
}


def build(item_id: str, snapshot: Optional[Dict[str, Any]] = None, *, completed_dir: Optional[Path] = None,
          newest_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Trả về bằng chứng thô của một mục; None nếu id không nằm trong danh sách được mở."""
    if item_id not in ITEM_IDS:
        return None
    completed_dir = completed_dir if completed_dir is not None else fh.latest_completed_run()
    newest_dir = newest_dir if newest_dir is not None else fh.newest_run_dir()
    builders: Dict[str, Callable[[], List[Dict[str, Any]]]] = {
        "flwr": _flwr_blocks, "ports": _ports_blocks, "attempt": lambda: _attempt_blocks(newest_dir),
        "clients": lambda: _clients_blocks(completed_dir), "rounds": lambda: _rounds_blocks(completed_dir),
        "hashes": lambda: _hashes_blocks(completed_dir), "live": lambda: _live_blocks(snapshot or {}),
    }
    run = newest_dir if item_id == "attempt" else completed_dir
    return {"id": item_id, "generated_at": _utc_now(), "assertion": ASSERTIONS[item_id],
            "run": run.name if run else None, "blocks": builders[item_id]()}


# ---------------------------------------------------------------------------
# cache: bấm mở lại hay nhiều người xem cùng lúc không được spawn thêm tiến trình kiểm tra
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[Tuple[Any, ...], float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def cached_build(item_id: str, snapshot: Optional[Dict[str, Any]] = None, ttl: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Bao ngoài build(): gộp kết quả cho nhiều người xem cùng lúc. `live` chỉ sống 2 giây vì phải phản ánh tiến trình đang chạy.

    Khóa cache gồm cả tên phiên mới nhất / phiên hoàn tất mới nhất: có phiên mới xong thì không được trả bằng chứng của
    phiên cũ cho đến khi hết hạn (trang đã chuyển sang phiên mới, bằng chứng phải đi theo).
    """
    if item_id not in ITEM_IDS:
        return None
    ttl = (LIVE_TTL_S if item_id == "live" else CACHE_TTL_S) if ttl is None else ttl
    completed_dir, newest_dir = fh.latest_completed_run(), fh.newest_run_dir()
    key = (newest_dir.name if newest_dir else None, completed_dir.name if completed_dir else None,
           (snapshot or {}).get("run_dir") if item_id == "live" else None)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(item_id)
        if hit and hit[0] == key and now - hit[1] < ttl:
            return hit[2]
    built = build(item_id, snapshot, completed_dir=completed_dir, newest_dir=newest_dir)
    if built is not None:
        with _cache_lock:
            _cache[item_id] = (key, time.monotonic(), built)
    return built
