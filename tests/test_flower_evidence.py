"""Bảng bằng chứng thô của trang Flower: đúng danh sách mục được mở, lệnh chạy argv tĩnh, không lộ đường dẫn máy chủ.

Test không spawn tiến trình thật: `run_command` được thay bằng bộ giả lập để kiểm phần ghép block, che đường dẫn và
giới hạn kích thước — những thứ có nguy cơ hỏng khi đọc tệp của phiên chạy.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest
from fastapi.testclient import TestClient

from tests.test_flower_health import H0, H1, H2, _write_run
from web.app import app
from web.services import flower_evidence as fe
from web.services import flower_health as fh

_REAL_RUN_COMMAND = fe.run_command  # fixture `runs` thay run_command bằng bộ giả lập; vài test cần bản thật


def _fake_run_command(title: str, argv, *, keep=None, **_kwargs):
    """Thay `run_command` trong các test ghép block: không spawn tiến trình, output trả theo từ khóa của argv."""
    answers = {"pip": "Name: flwr\nVersion: 1.39.0\nLocation: .venv/lib/python3.11/site-packages",
               "-c": "flwr package location = .venv/lib/python3.11/site-packages/flwr/__init__.py",
               "sha256": f"{H2}  artifacts/global.pt",
               "netstat": "Active Connections\n  Proto  Local Address  State  PID\n  TCP  0.0.0.0:22  LISTENING  812",
               "ss": "State  Local Address:Port\nLISTEN 0.0.0.0:22"}
    text = " ".join(argv)
    out = next((value for key, value in answers.items() if key in text), f"GIẢ: {text}")
    if keep:
        lines = out.splitlines()
        out = "\n".join(lines[:1] + [line for line in lines[1:] if any(token in line for token in keep)])
    return {"kind": "command", "title": title, "command": text, "cwd": ".", "platform": "linux",
            "exit_code": 0, "output": out, "truncated": False}


@pytest.fixture()
def runs(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "artifacts" / "web_runs" / "network_smoke"
    root.mkdir(parents=True)
    monkeypatch.setattr(fh, "runs_root", lambda: root)
    monkeypatch.setattr(fh, "flwr_status", lambda: {"installed": True, "version": "1.39.0", "python": "3.11.17", "error": None})
    monkeypatch.setattr(fe, "run_command", _fake_run_command)
    monkeypatch.setattr(fe, "ROOT", tmp_path)  # để đường dẫn tmp bị che đúng như đường dẫn cài đặt trên máy chủ
    monkeypatch.setattr(fe, "_cache", {})
    return root


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_every_listed_item_has_an_assertion_and_blocks(runs):
    _write_run(runs, "20261020T010000Z_aaaa1111", with_checkpoint=False)
    for item_id in fe.ITEM_IDS:
        built = fe.build(item_id, {"active": False, "processes": [], "run_dir": None})
        assert built["id"] == item_id and built["assertion"], item_id
        assert built["blocks"], f"{item_id} không có bằng chứng nào"
        assert {b["kind"] for b in built["blocks"]} <= {"command", "file", "records", "note"}


def test_an_id_outside_the_whitelist_has_no_evidence(runs):
    assert fe.build("session.json", {}) is None
    assert fe.build("../../etc/passwd", {}) is None
    assert fe.build("history", {}) is None


def test_run_command_uses_static_argv_and_shows_the_real_exit_code():
    def runner(argv, **_kwargs):
        assert "shell" not in _kwargs, "bằng chứng không được chạy qua shell"
        return type("Done", (), {"returncode": 1, "stdout": "ra stdout\n", "stderr": "ra stderr\n"})()

    block = fe.run_command("Kiểm", ["python", "-c", "print(1)"], runner=runner)
    assert block["exit_code"] == 1 and block["output"].startswith("ra stdout") and "ra stderr" in block["output"]

    def timeout_runner(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 6)

    assert "không trả lời" in fe.run_command("Kiểm", ["python"], runner=timeout_runner)["output"]
    assert "không chạy được lệnh" in fe.run_command("Kiểm", ["python"], runner=lambda *a, **k: (_ for _ in ()).throw(PermissionError()))["output"]


def test_endpoint_serves_only_listed_items_and_is_read_only(runs, monkeypatch):
    _write_run(runs, "20261020T010000Z_bbbb2222", with_checkpoint=False)
    client = TestClient(app)
    ok = client.get("/api/flower/evidence/flwr")
    assert ok.status_code == 200
    body = ok.json()
    assert body["id"] == "flwr" and any(b["kind"] == "command" for b in body["blocks"])
    assert client.get("/api/flower/evidence/whatever").status_code == 404
    assert client.post("/api/flower/evidence/flwr").status_code == 405


def test_the_open_live_panel_cannot_outrun_the_rate_limit(runs):
    """Bảng "đang chạy" tự đọc lại khi người xem đang mở nó. Nhịp phải chậm hơn giới hạn của endpoint:
    trang poll 3 giây một lần, và một phiên Flower sống động đúng lúc hội đồng đang xem mà chuyển sang 429 thì mất khách."""
    from web.routers import flower as flower_router
    js = (Path(__file__).resolve().parents[1] / "web" / "static" / "js" / "pages" / "flower.js").read_text(encoding="utf-8")
    auto = re.search(r"function refreshLive\(data\) \{(.*?)\n\}", js, re.S).group(1)
    assert 'OPEN.has("live")' in auto, "phải kiểm tra người xem đang mở bảng đã, không đọc theo đồng hồ"
    assert "LIVE_REFRESH_MS" in auto and js.count('loadEvidence("live"') == 1, "chỉ refreshLive được gọi lại theo chu kỳ"
    cadence_ms = int(re.search(r"LIVE_REFRESH_MS\s*=\s*(\d+)", js).group(1))
    assert 60_000 // cadence_ms < flower_router.EVIDENCE_PER_MINUTE, "một bảng mở liên tục phải còn dư hạn mức cho các mục khác"


def test_no_server_path_leaks_into_evidence(runs):
    run = _write_run(runs, "20261020T010000Z_cccc3333")
    session = {"partition_dir": str(run / "data"), "bind_address": "127.0.0.1:18089",
               "commands": {"server": {"argv": [str(run / "python.exe"), "-u", "-m", "src.federated.server_app",
                                                "--metrics-file", str(run / "history.json")], "pid": 4242}}}
    (run / "session.json").write_text(json.dumps(session), encoding="utf-8")
    (run / "server.log").write_text(f'Traceback\n  File "{run / "server.py"}", line 1\n', encoding="utf-8")
    snapshot = {"active": True, "run_dir": str(run), "processes": [{"name": "server", "pid": 4242, "running": True, "exit_code": None}]}
    for item_id in fe.ITEM_IDS:
        built = fe.build(item_id, snapshot)
        texts = list(_strings(built))
        assert any("python.exe" in t for t in texts) or item_id != "attempt", "dòng lệnh spawn phải còn hiện ra, chỉ bỏ tiền tố thư mục"
        for text in texts:
            assert str(runs.parent.parent) not in text and str(Path.home()) not in text, f"{item_id} để lộ đường dẫn: {text[:120]}"


def test_recorded_spawn_commands_are_shown_with_their_pid(runs):
    run = _write_run(runs, "20261020T020000Z_dddd4444", with_checkpoint=False)
    session = json.loads((run / "session.json").read_text(encoding="utf-8"))
    session["commands"] = {"server": {"argv": ["python", "-m", "src.federated.server_app", "--port", "18089"], "pid": 4242},
                           "client_00": {"argv": ["python", "-m", "src.federated.client_app", "--client-id", "0"], "pid": 4243}}
    (run / "session.json").write_text(json.dumps(session), encoding="utf-8")
    blocks = fe.build("attempt", {})["blocks"]
    table = next(b for b in blocks if b["kind"] == "records")
    assert [r["pid"] for r in table["rows"]] == [4242, 4243]
    assert "--port" in table["rows"][0]["command"]


def test_a_run_without_recorded_commands_says_so_instead_of_inventing_one(runs):
    _write_run(runs, "20261020T020000Z_eeee5555", with_checkpoint=False)
    blocks = fe.build("attempt", {})["blocks"]
    assert not [b for b in blocks if b["kind"] == "records"]
    assert any("trước khi máy chủ ghi dòng lệnh" in b.get("text", "") for b in blocks)


def test_newest_attempt_is_the_failed_one_while_clients_use_the_completed_run(runs):
    _write_run(runs, "20261020T010000Z_ffff6666")
    _write_run(runs, "20261020T030000Z_gggg7777", status="failed", log="Boom\nValueError: thiếu ảnh\n")
    assert fe.build("attempt", {})["run"] == "20261020T030000Z_gggg7777"
    assert fe.build("clients", {})["run"] == "20261020T010000Z_ffff6666"
    log_block = next(b for b in fe.build("attempt", {})["blocks"] if b.get("title", "").startswith("server.log"))
    assert "ValueError: thiếu ảnh" in log_block["content"]


def test_client_evidence_carries_the_per_client_records(runs):
    run = _write_run(runs, "20261020T010000Z_hhhh8888")
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    for audit in history["audit_rounds"]:
        audit["clients"] = [{"client_id": c, "flower_cid": f"cid{c}", "num_examples": 25, "probe_rtt_ms": 12.5,
                             "local_train_time_s": 3.2, "data_mode": "synthetic", "validation_reason": "ok", "status": "ok"}
                            for c in range(5)]
    (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (run / "client_00.log").write_text("[*] Starting Flower Client 0 -> Server: 127.0.0.1:18089\nINFO : Received: train message abc\n", encoding="utf-8")
    blocks = fe.build("clients", {})["blocks"]
    table = next(b for b in blocks if b["kind"] == "records")
    assert len(table["rows"]) == 10  # 5 client × 2 vòng
    assert table["rows"][0]["flower_cid"] == "cid0" and table["rows"][0]["round"] == 1
    logs = [b for b in blocks if b["kind"] == "file"]
    assert any("Starting Flower Client 0" in b["content"] for b in logs)


def test_a_huge_log_is_cut_and_says_it_was_cut(runs):
    run = _write_run(runs, "20261020T010000Z_iiii9999", with_checkpoint=False)
    (run / "server.log").write_text("\n".join(f"dòng {i}" for i in range(200_000)), encoding="utf-8")
    block = [b for b in fe.build("attempt", {})["blocks"] if b["kind"] == "file"][-1]
    assert block["truncated"] is True and len(block["content"]) <= fe.MAX_BLOCK_CHARS


def test_missing_files_degrade_to_a_message_not_a_crash(runs):
    run = _write_run(runs, "20261020T010000Z_jjjj0000", with_checkpoint=False)
    for name in ("history.json", "progress.json", "server.log"):
        (run / name).unlink(missing_ok=True)
    for item_id in ("attempt", "clients", "rounds", "hashes"):
        built = fe.build(item_id, {})
        assert built["blocks"], item_id


def test_live_evidence_lists_the_processes_and_their_pids(runs):
    run = _write_run(runs, "20261020T010000Z_kkkk1111", with_checkpoint=False)
    snapshot = {"active": True, "run_dir": str(run),
                "processes": [{"name": "server", "pid": 111, "running": True, "exit_code": None},
                              {"name": "client_00", "pid": 112, "running": True, "exit_code": None}]}
    blocks = fe.build("live", snapshot)["blocks"]
    table = next(b for b in blocks if b["kind"] == "records")
    assert [r["pid"] for r in table["rows"]] == [111, 112] and all(r["running"] for r in table["rows"])


def test_cache_merges_repeated_opens_but_the_live_item_stays_fresher(runs, monkeypatch):
    _write_run(runs, "20261020T010000Z_llll2222", with_checkpoint=False)
    calls: List[str] = []
    monkeypatch.setattr(fe, "build", lambda item_id, snapshot=None, **kw: (calls.append(item_id), {"id": item_id, "blocks": [], "assertion": ""})[1])
    for _ in range(4):
        fe.cached_build("flwr", {})
    assert calls == ["flwr"], "bấm mở lại trong hạn cache không được spawn thêm tiến trình kiểm tra"
    for _ in range(3):
        fe.cached_build("live", {}, ttl=0.0)
    assert calls.count("live") == 3
    assert fe.LIVE_TTL_S < fe.CACHE_TTL_S and fe.LIVE_TTL_S <= 3.0, "bảng 'đang chạy' phải mới hơn chu kỳ poll 3 giây của trang"


def test_inline_probes_use_double_quotes_so_the_displayed_command_stays_readable(runs, monkeypatch):
    """Trang này in nguyên văn dòng lệnh cho hội đồng đọc: nháy đơn trong đoạn mã sẽ bị shlex biến thành `'"'"'`.

    Thụt vào bằng nháy kép nên `shlex.join` chỉ cần một cặp nháy đơn bọc ngoài là xong."""
    for name in ("_FLWR_SOURCE", "_PORTS_SOURCE"):
        joined = shlex.join(["python", "-c", getattr(fe, name)])
        assert joined.count("'") == 2, f"{name}: vẫn còn nháy đơn lồng trong lệnh hiển thị"
    monkeypatch.setattr(fe.shutil, "which", lambda _exe: None)  # không có sha256sum/certutil → về đường Python
    fallback = fe._sha_command(runs / "global.pt")
    assert json.dumps(fe._runnable_path(runs / "global.pt")) in fallback["command"]
    assert "'" not in fallback["command"], "đoạn tính SHA dự phòng phải dùng nháy kép"


def test_a_missing_file_is_reported_without_the_absolute_path(runs, tmp_path):
    """Python nhét đường dẫn tuyệt đối vào OSError; dòng '(không đọc được tệp: …)' cũng phải qua _redact."""
    block = fe._file_block("thử đọc tệp không có thật", runs / "khong-co.json")
    assert "khong-co.json" in block["content"]
    assert str(tmp_path) not in block["content"], "thông báo lỗi để lộ thư mục cài đặt của máy chủ"


def test_the_process_list_falls_back_to_tasklist_on_windows(runs, monkeypatch):
    """`ps` mà Git for Windows đặt trong PATH không nhận `-o` nên bảng PID trên máy Windows sẽ toàn lỗi; dùng tasklist."""
    argvs: List[List[str]] = []

    def capture(title, argv, **_kw):
        argvs.append(list(argv))
        return {"kind": "command", "title": title, "command": " ".join(argv), "cwd": ".", "platform": "x",
                "exit_code": 0, "output": "", "truncated": False}

    monkeypatch.setattr(fe, "run_command", capture)
    monkeypatch.setattr(fe.shutil, "which", lambda exe: f"/usr/bin/{exe}")  # máy có cả ps lẫn tasklist
    for platform, expected in (("win32", "tasklist"), ("linux", "ps")):
        monkeypatch.setattr(fe, "sys", types.SimpleNamespace(platform=platform, executable=sys.executable))
        block = fe._process_list_command([111, 112])
        assert argvs[-1][0] == expected, platform
        assert str(block["exit_code"]) == "0"


def test_terminal_colour_codes_never_reach_the_page(runs):
    """Log của Flower tô màu bằng ESC[92m…ESC[0m; trên web chúng thành `[92mINFO [0m` khó đọc."""
    run = _write_run(runs, "20261020T010000Z_mmmm3333", with_checkpoint=False)
    (run / "server.log").write_text("\x1b[92mINFO\x1b[0m:      [ROUND 1]\n\x1b[93mWARNING\x1b[0m: DEPRECATED\n", encoding="utf-8")
    block = next(b for b in fe.build("attempt", {})["blocks"] if b.get("title", "").startswith("server.log"))
    assert "\x1b" not in block["content"] and "INFO:      [ROUND 1]" in block["content"]
    assert "ANSI" in block["note"]
    out = _REAL_RUN_COMMAND("màu", [sys.executable, "-c", "print('\x1b[92mxanh\x1b[0m')"])
    assert "\x1b" not in out["output"] and "xanh" in out["output"]


def test_cache_does_not_serve_a_previous_run_after_a_new_one_finishes(runs):
    _write_run(runs, "20261020T010000Z_nnnn4444", with_checkpoint=False)
    first = fe.cached_build("attempt", {})
    assert first["run"] == "20261020T010000Z_nnnn4444"
    _write_run(runs, "20261020T050000Z_oooo5555", with_checkpoint=False)
    assert fe.cached_build("attempt", {})["run"] == "20261020T050000Z_oooo5555", "đổi phiên thì phải dựng lại dù chưa hết TTL"


def test_hash_table_marks_each_row_as_matching(runs):
    run = _write_run(runs, "20261020T010000Z_pppp6666")
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    sha = fh._sha256(run / "global.pt")
    final = history["audit_rounds"][-1]["global_model_hash"]
    history["final_model_hash"] = final
    (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
    monkey_sha = f"{sha}  {run.name}/global.pt"
    original = fe._sha_command
    fe._sha_command = lambda path: {"kind": "command", "title": "sha", "command": "sha256sum", "cwd": ".", "platform": "x",
                                    "exit_code": 0, "output": monkey_sha, "truncated": False}
    try:
        blocks = fe.build("hashes", {})["blocks"]
    finally:
        fe._sha_command = original
    table = next(b for b in blocks if b["kind"] == "records" and any(c["key"] == "matches" for c in b["columns"]))
    assert table["rows"][-1]["matches"] is True, "SHA tính lại từ tệp phải khớp output của lệnh sha256sum"
    assert table["rows"][1]["matches"] is True  # final_model_hash so với mã băm vòng cuối
    assert fe._first_sha256("ABC " + sha.upper()) == sha
