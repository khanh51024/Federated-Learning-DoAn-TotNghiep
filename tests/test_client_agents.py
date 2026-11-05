"""Máy đã cài bộ client tự báo danh với web và tự mở client khi được giao: web chỉ giao được số client, quản trị viên mới chọn được máy."""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.services import client_agents as ca

ROOT = Path(__file__).resolve().parents[1]
AGENT_FILE = ROOT / "deploy" / "lan_client" / "client_agent.py"
KEY_A, KEY_B, KEY_C = "a" * 32, "b" * 32, "c" * 32
ADMIN = ("admin", "mat-khau-thu-du-dai")


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("client_agent_under_test", AGENT_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _empty_registry():
    ca._agents.clear()
    yield
    ca._agents.clear()


def _beat(key, *, running=None, result=None, result_job=None, run_active=True, now=None, source="1.1.1.1"):
    return ca.heartbeat(key, source=source, cores=8, ram_gb=16.0, ram_free_gb=9.5, running=running, result=result, result_job=result_job, run_active=run_active, now=now)


def _id(key):
    return ca.agent_id_of(key)


# ---- phần web

def test_split_gives_consecutive_clients_and_covers_all_of_them():
    assert ca.split(10, 3) == [(0, 4), (4, 3), (7, 3)]
    assert ca.split(10, 1) == [(0, 10)]
    assert ca.split(3, 3) == [(0, 1), (1, 1), (2, 1)]


def test_machine_appears_after_its_first_heartbeat_and_goes_offline_when_silent():
    assert ca.view(run_active=False, now=100.0) == []
    _beat(KEY_A, run_active=False, now=100.0)
    row = ca.view(run_active=False, now=101.0)[0]
    assert row["label"] == "Máy A" and row["id"] == _id(KEY_A) and row["code"] == _id(KEY_A)[:6] and row["state"] == "ready" and row["ram_free_gb"] == 9.5
    assert ca.view(run_active=False, now=100.0 + ca.ONLINE_S + 1)[0]["state"] == "offline"
    assert ca.view(run_active=False, now=100.0 + ca.FORGET_S + 1) == []


def test_machines_that_went_offline_do_not_take_the_first_letters():
    _beat(KEY_A, run_active=False, now=100.0)
    _beat(KEY_B, run_active=False, now=100.0 + ca.ONLINE_S)
    rows = ca.view(run_active=False, now=100.0 + ca.ONLINE_S + 1)
    assert [(row["label"], row["id"], row["state"]) for row in rows] == [("Máy A", _id(KEY_B), "ready"), ("Máy B", _id(KEY_A), "offline")]


def test_job_is_sent_until_accepted_and_not_again_after_that():
    for key in (KEY_A, KEY_B):
        _beat(key, now=100.0)
    ca.assign([_id(KEY_A), _id(KEY_B)], 10)
    first = _beat(KEY_A, now=101.0)["job"]
    assert (first["first"], first["count"]) == (0, 5)
    assert _beat(KEY_A, now=102.0)["job"] == first, "câu trả lời có thể lạc trên đường nên phải gửi lại tới khi máy nhận"
    reply = _beat(KEY_A, running=first["id"], now=103.0)
    assert reply["job"] is None and reply["stop"] is False
    assert _beat(KEY_A, now=104.0)["job"] is None, "máy khởi động lại không được mở trùng client"
    second = _beat(KEY_B, now=105.0)["job"]
    assert (second["first"], second["count"]) == (5, 5)


def test_labels_during_a_run_match_run_view_which_names_machines_by_smallest_client():
    for offset, key in enumerate((KEY_A, KEY_B, KEY_C)):
        _beat(key, now=100.0 + offset)
    ca.assign([_id(KEY_C), _id(KEY_B)], 10)
    rows = ca.view(run_active=True, now=103.0)
    assert [(row["label"], row["id"], row["clients"][:1]) for row in rows] == [
        ("Máy A", _id(KEY_C), [0]), ("Máy B", _id(KEY_B), [5]), ("Máy C", _id(KEY_A), [])]
    assert rows[0]["state"] == "starting" and rows[2]["state"] == "ready"


def test_when_the_run_ends_the_machine_is_told_to_stop_and_job_disappears():
    _beat(KEY_A, now=100.0)
    ca.assign([_id(KEY_A)], 10)
    job = _beat(KEY_A, now=101.0)["job"]
    _beat(KEY_A, running=job["id"], now=102.0)
    assert ca.view(run_active=True, now=103.0)[0]["state"] == "running"
    reply = _beat(KEY_A, running=job["id"], run_active=False, now=104.0)
    assert reply["stop"] is True and reply["job"] is None
    assert ca.view(run_active=False, now=105.0)[0]["clients"] == []


def test_page_sees_at_once_when_a_machine_fails_or_finishes_during_the_run():
    for key in (KEY_A, KEY_B):
        _beat(key, now=100.0)
    ca.assign([_id(KEY_A), _id(KEY_B)], 10)
    job = _beat(KEY_A, now=101.0)["job"]
    _beat(KEY_A, running=job["id"], now=102.0)
    _beat(KEY_B, running=job["id"], now=102.0)
    assert [row["state"] for row in ca.view(run_active=True, now=103.0)] == ["running", "running"]
    assert _beat(KEY_A, result="failed", result_job=job["id"], now=104.0)["job"] is None, "máy đã nhận rồi thì không giao lại dù nó đã dừng"
    _beat(KEY_B, result="ok", result_job=job["id"], now=104.0)
    rows = ca.view(run_active=True, now=105.0)
    assert [row["state"] for row in rows] == ["failed", "done"] and rows[0]["clients"] == [0, 1, 2, 3, 4]


def test_machine_that_fails_before_ever_reporting_running_is_not_stuck_on_starting():
    _beat(KEY_A, now=100.0)
    ca.assign([_id(KEY_A)], 5)
    job = _beat(KEY_A, now=101.0)["job"]
    assert ca.view(run_active=True, now=102.0)[0]["state"] == "starting"
    assert _beat(KEY_A, result="failed", result_job=job["id"], now=103.0)["job"] is None
    assert ca.view(run_active=True, now=104.0)[0]["state"] == "failed"


def test_failed_machine_is_shown_until_it_gets_new_work_and_old_results_do_not_come_back():
    _beat(KEY_A, now=100.0)
    ca.assign([_id(KEY_A)], 5)
    job = _beat(KEY_A, now=101.0)["job"]
    _beat(KEY_A, result="failed", result_job=job["id"], now=102.0)
    assert _beat(KEY_A, result="failed", result_job=job["id"], run_active=False, now=103.0)["job"] is None
    assert ca.view(run_active=False, now=104.0)[0]["state"] == "failed"
    ca.assign([_id(KEY_A)], 5)
    _beat(KEY_A, result="failed", result_job=job["id"], now=105.0)
    assert ca.view(run_active=True, now=106.0)[0]["state"] == "starting", "máy vẫn gửi lại kết quả cũ cho tới khi nhận việc mới"


def test_key_with_the_same_short_code_cannot_take_over_a_machine(monkeypatch):
    """Tìm khóa trùng mã 6 ký tự chỉ mất vài chục giây, nên giả lập hai máy trùng mã: phần việc vẫn theo định danh đầy đủ."""
    monkeypatch.setattr(ca, "code_of", lambda _agent_id: "abcdef")
    victim, forged = "1" * 32, "2" * 32
    _beat(victim, now=100.0)
    ca.assign([_id(victim)], 5)
    assert _beat(forged, now=101.0, source="9.9.9.9")["job"] is None, "khóa khác không nhận được việc"
    job = _beat(victim, now=102.0)["job"]
    assert job is not None, "máy thật vẫn nhận được việc"
    _beat(forged, running=job["id"], now=103.0, source="9.9.9.9")
    assert _beat(victim, now=104.0)["job"] == job, "khóa khác không báo nhận thay được"
    rows = {row["id"]: row for row in ca.view(run_active=True, now=105.0)}
    assert rows[_id(victim)]["state"] == "starting" and rows[_id(forged)]["state"] == "ready" and rows[_id(forged)]["clients"] == []


def test_one_source_cannot_fill_the_list_and_machines_that_left_make_room():
    keys = [f"{index:032x}" for index in range(ca.MAX_ONLINE_PER_SOURCE)]
    for key in keys:
        _beat(key, run_active=False, now=100.0, source="6.6.6.6")
    with pytest.raises(ca.AgentsFull):
        _beat("f" * 32, run_active=False, now=100.0, source="6.6.6.6")
    assert _beat("e" * 32, run_active=False, now=100.0, source="7.7.7.7")["job"] is None, "máy thật ở IP khác vẫn nối được"
    assert _beat("f" * 32, run_active=False, now=100.0 + ca.ONLINE_S + 1, source="6.6.6.6")["job"] is None, "máy cũ đã im thì nguồn đó lại nối được"


def test_full_list_refuses_only_while_everyone_is_online_and_known_machines_keep_working():
    keys = [f"{index:032x}" for index in range(ca.MAX_AGENTS)]
    for index, key in enumerate(keys):
        _beat(key, run_active=False, now=100.0, source=f"s{index}")
    with pytest.raises(ca.AgentsFull):
        _beat("f" * 32, run_active=False, now=100.0, source="moi")
    assert _beat(keys[0], run_active=False, now=101.0, source="s0")["job"] is None
    assert _beat("f" * 32, run_active=False, now=100.0 + ca.ONLINE_S + 2, source="moi")["job"] is None, "máy đã tắt nhường chỗ"


def test_assignment_needs_online_machines_and_enough_clients():
    _beat(KEY_A, run_active=False, now=100.0)
    agent_id = _id(KEY_A)
    ca.check_assignable([agent_id], 5, now=101.0)
    with pytest.raises(ca.CannotAssign):
        ca.check_assignable([agent_id], 5, now=100.0 + ca.ONLINE_S + 1)
    with pytest.raises(ca.CannotAssign):
        ca.check_assignable(["a" * 64], 5, now=101.0)
    with pytest.raises(ca.CannotAssign):
        ca.check_assignable([agent_id, agent_id], 5, now=101.0)
    _beat(KEY_B, run_active=False, now=101.0)
    with pytest.raises(ca.CannotAssign):
        ca.check_assignable([agent_id, _id(KEY_B)], 1, now=101.0)


def test_no_machine_gets_more_than_five_clients():
    _beat(KEY_A, run_active=False, now=100.0)
    _beat(KEY_B, run_active=False, now=100.0)
    with pytest.raises(ca.CannotAssign, match="tối đa 5"):
        ca.check_assignable([_id(KEY_A)], 6, now=101.0)
    ca.check_assignable([_id(KEY_A), _id(KEY_B)], 10, now=101.0)
    for machines in range(2, 11):
        clients = min(10, ca.MAX_PER_MACHINE * machines)
        assert max(count for _first, count in ca.split(clients, machines)) <= ca.MAX_PER_MACHINE


# ---- API

class _FakeProcess:
    pid = 4321

    def poll(self):
        return None


@pytest.fixture
def api(monkeypatch, tmp_path):
    from web.app import app
    from web.services import admin_auth, flower_demo as fd
    monkeypatch.setenv("COMMUNITY_ADMIN_TOKEN", ADMIN[1])
    monkeypatch.setattr(admin_auth.LIMITER, "allow", lambda *_a, **_k: (True, 0))
    monkeypatch.setattr(fd, "_process", None)
    monkeypatch.setattr(fd, "_output", None)
    monkeypatch.setattr(fd, "_blocked_reason", lambda: None)
    monkeypatch.setattr(fd, "_light_data_dir", lambda: tmp_path)
    (tmp_path / "partition").mkdir()
    monkeypatch.setattr(fd, "EARLY_EXIT_WAIT_S", 0.0)
    monkeypatch.setattr(fd.subprocess, "Popen", lambda *_a, **_k: _FakeProcess())
    return TestClient(app)


def _post_beat(api, key, **extra):
    return api.post("/api/flower/agents/heartbeat", json={"key": key, "cores": 8, "ram_gb": 16, "ram_free_gb": 9.5, **extra})


def test_page_sees_machines_without_key_or_address(api):
    assert api.get("/api/flower/agents").json() == {"agents": []}
    reply = _post_beat(api, KEY_A)
    assert reply.status_code == 200 and reply.json()["job"] is None and reply.json()["code"] == _id(KEY_A)[:6]
    rows = api.get("/api/flower/agents").json()["agents"]
    assert [row["id"] for row in rows] == [_id(KEY_A)]
    assert KEY_A not in str(rows) and "testclient" not in str(rows)


@pytest.mark.parametrize("bad", [{"key": "ngan"}, {"key": "Z" * 32}, {"key": KEY_A, "cores": -1}, {"key": KEY_A, "ram_gb": 1e9}, {"key": KEY_A, "running": "../../etc"}, {"key": KEY_A, "result": "<b>x</b>"}])
def test_heartbeat_refuses_anything_outside_the_expected_shape(api, bad):
    assert api.post("/api/flower/agents/heartbeat", json=bad).status_code == 422
    assert ca._agents == {}


def test_only_admin_can_put_a_machine_to_work(api):
    _post_beat(api, KEY_A)
    agent_id = _id(KEY_A)
    body = {"clients": 5, "rounds": 10, "light": True, "machines": [agent_id]}
    assert api.post("/api/flower/demo/start", json=body).status_code == 401
    assert _post_beat(api, KEY_A).json()["job"] is None
    reply = api.post("/api/flower/demo/start", json=body, auth=ADMIN)
    assert reply.status_code == 200 and reply.json()["assigned"] == [{"id": agent_id, "first": 0, "count": 5}]
    job = _post_beat(api, KEY_A).json()["job"]
    assert (job["first"], job["count"]) == (0, 5)
    assert set(job) == {"id", "first", "count"}, "web không được gửi địa chỉ hay lệnh xuống máy"


def test_machines_are_rejected_for_the_full_image_set_and_when_offline(api):
    _post_beat(api, KEY_A)
    assert api.post("/api/flower/demo/start", json={"clients": 7, "rounds": 3, "machines": [_id(KEY_A)]}, auth=ADMIN).status_code == 400
    assert api.post("/api/flower/demo/start", json={"clients": 10, "rounds": 3, "light": True, "machines": ["a" * 64]}, auth=ADMIN).status_code == 409
    assert api.post("/api/flower/demo/start", json={"clients": 10, "rounds": 3, "light": True, "machines": [_id(KEY_A)[:6]]}, auth=ADMIN).status_code == 422, "mã ngắn đoán được nên không dùng để chọn máy"
    assert api.post("/api/flower/demo/start", json={"clients": 10, "rounds": 3, "light": True, "machines": ["; rm -rf /"]}, auth=ADMIN).status_code == 422


def test_more_than_five_clients_on_one_machine_is_refused_before_the_server_opens(api, monkeypatch):
    from web.services import flower_demo as fd
    opened = []
    monkeypatch.setattr(fd, "start", lambda *a, **k: opened.append(a))
    _post_beat(api, KEY_A)
    reply = api.post("/api/flower/demo/start", json={"clients": 10, "rounds": 3, "light": True, "machines": [_id(KEY_A)]}, auth=ADMIN)
    assert reply.status_code == 409 and "tối đa 5" in reply.json()["detail"]
    by_hand = api.post("/api/flower/demo/start", json={"clients": 6, "rounds": 3, "light": True}, auth=ADMIN)
    assert by_hand.status_code == 400 and "5 client" in by_hand.json()["detail"]
    assert opened == []


def test_two_laptops_by_hand_on_the_full_image_set_still_work(api):
    assert api.post("/api/flower/demo/start", json={"clients": 7, "rounds": 3}, auth=ADMIN).status_code == 200


def test_start_that_fails_gives_no_machine_any_work(api, monkeypatch):
    from web.services import flower_demo as fd
    _post_beat(api, KEY_A)

    def refused(*_a, **_k):
        raise fd.DemoError(409, "Cổng đang bận")

    monkeypatch.setattr(fd, "start", refused)
    assert api.post("/api/flower/demo/start", json={"clients": 5, "rounds": 3, "light": True, "machines": [_id(KEY_A)]}, auth=ADMIN).status_code == 409
    assert _post_beat(api, KEY_A).json()["job"] is None


def test_start_without_machines_answers_as_before(api):
    reply = api.post("/api/flower/demo/start", json={"clients": 7, "rounds": 3}, auth=ADMIN)
    assert reply.status_code == 200 and "assigned" not in reply.json()


# ---- phía máy

def test_agent_only_needs_the_standard_library_so_it_stays_small_while_idle():
    tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)


def test_agent_and_web_compute_the_same_machine_code():
    assert _load_agent_module().code_of(KEY_A) == ca.code_of(ca.agent_id_of(KEY_A))


@pytest.mark.parametrize("raw", [
    None, "x", {"id": "abc", "first": 0, "count": 1},
    {"id": "a" * 16, "first": -1, "count": 1},
    {"id": "a" * 16, "first": 0, "count": 0},
    {"id": "a" * 16, "first": 8, "count": 3},
    {"id": "a" * 16, "first": 0, "count": 6},
    {"id": "a" * 16, "first": "0 & calc", "count": 1},
    {"id": "a" * 16, "first": True, "count": 1},
    {"id": "a" * 16 + " & calc", "first": 0, "count": 1},
])
def test_agent_ignores_a_job_that_is_not_two_small_numbers(raw):
    assert _load_agent_module().parse_job(raw) is None


class _Web:
    """Web giả: trả lần lượt các câu đã định, ghi lại mọi lần báo danh."""

    def __init__(self):
        self.seen, self.replies = [], []

    def __call__(self, body):
        self.seen.append(body)
        return self.replies.pop(0) if self.replies else {"poll_s": 5, "job": None, "stop": False}


def _agent(tmp_path, **extra):
    module = _load_agent_module()
    root = tmp_path / "app"
    for client in range(10):
        (root / "mini10" / f"images_client_{client:02d}").mkdir(parents=True)
    (root / "mini10" / "partition").mkdir()
    web, launched, killed = _Web(), [], []
    process = extra.pop("process", None) or _RunningProcess()

    def launch(*args):
        launched.append(args)
        return process

    state = tmp_path / "state"
    state.mkdir()
    agent = module.Agent("https://web.example", "1.2.3.4", "mini10", state, root, post=web, launch=launch, kill=lambda p: killed.append(p),
                         server_ready=extra.pop("server_ready", lambda _host: True), clock=extra.pop("clock", lambda: 0.0))
    return agent, web, launched, killed, process


class _RunningProcess:
    pid = 77
    code = None

    def poll(self):
        return self.code


JOB = {"id": "a" * 16, "first": 4, "count": 3}


def test_agent_opens_the_assigned_clients_against_its_own_server_and_reports_the_result(tmp_path):
    agent, web, launched, _killed, process = _agent(tmp_path)
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    assert web.seen[-1]["running"] is None and launched == []
    agent.tick()
    assert web.seen[-1]["running"] == JOB["id"]
    assert len(launched) == 1 and launched[0][2:] == (4, 3, "1.2.3.4", "mini10")
    agent.tick()
    assert len(launched) == 1, "không mở trùng khi đang chạy"
    process.code = 0
    agent.tick()
    assert web.seen[-1]["running"] is None and web.seen[-1]["result"] == "ok" and web.seen[-1]["result_job"] == JOB["id"]


def test_agent_reports_failure_when_a_client_fails(tmp_path):
    agent, web, _launched, _killed, process = _agent(tmp_path)
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    process.code = 1
    agent.tick()
    assert web.seen[-1]["result"] == "failed" and web.seen[-1]["running"] is None


def test_agent_does_not_run_the_same_job_twice_if_web_repeats_it(tmp_path):
    agent, web, launched, _killed, process = _agent(tmp_path)
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    process.code = 0
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    assert len(launched) == 1


def test_agent_kills_clients_when_the_run_is_stopped_on_the_web(tmp_path):
    agent, web, _launched, killed, process = _agent(tmp_path)
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    web.replies.append({"job": None, "stop": True})
    agent.tick()
    assert killed == [process] and agent.job is None
    assert agent.tick()["stop"] is False and web.seen[-1]["running"] is None


def test_agent_waits_for_the_server_port_and_gives_up_after_the_deadline(tmp_path):
    now = {"t": 0.0}
    ready = {"open": False}
    agent, web, launched, _killed, _process = _agent(tmp_path, server_ready=lambda _host: ready["open"], clock=lambda: now["t"])
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    assert launched == [] and web.seen[-1]["running"] == JOB["id"], "phải báo đã nhận ngay cả khi cổng chưa mở"
    now["t"] = agent.deadline + 1
    agent.tick()
    assert launched == [] and web.seen[-1]["result"] == "failed" and web.seen[-1]["running"] is None


def test_agent_fails_the_job_when_images_are_missing_without_opening_anything(tmp_path):
    agent, web, launched, _killed, _process = _agent(tmp_path)
    (agent.root / "mini10" / "images_client_05").rmdir()
    web.replies.append({"job": JOB, "stop": False})
    agent.tick()
    agent.tick()
    assert launched == [] and web.seen[-1]["result"] == "failed" and web.seen[-1]["result_job"] == JOB["id"]


def test_agent_survives_a_broken_reply_instead_of_dying_silently(tmp_path):
    import http.client
    agent, _web, _launched, _killed, _process = _agent(tmp_path, clock=time.monotonic)
    calls = []

    def broken(_body):
        calls.append(1)
        if len(calls) == 2:
            (agent.state_dir / "agent.stop").touch()
        raise http.client.IncompleteRead(b"")

    agent.post = broken
    module = sys.modules["client_agent_under_test"]
    original = module.POLL_S
    module.POLL_S = 0
    try:
        agent.run()
    finally:
        module.POLL_S = original
    assert len(calls) == 2 and "chưa nối được" in (agent.state_dir / "agent.log").read_text(encoding="utf-8")


def test_agent_notices_the_stop_file_quickly_even_while_retrying_after_a_network_failure(tmp_path):
    agent, _web, _launched, _killed, _process = _agent(tmp_path, clock=time.monotonic)
    calls = []

    def offline(_body):
        calls.append(1)
        (agent.state_dir / "agent.stop").touch()    # người dùng gỡ cài đúng lúc đang mất mạng
        raise OSError("mất mạng")

    agent.post = offline
    started = time.monotonic()
    agent.run()
    assert len(calls) == 1 and time.monotonic() - started < 3, "gỡ cài không phải chờ hết thời gian thử lại"
    assert not (agent.state_dir / "agent.stop").exists()


def test_key_survives_a_restart(tmp_path):
    module = _load_agent_module()
    first = module.load_key(tmp_path)
    assert module.KEY.fullmatch(first) and module.load_key(tmp_path) == first


@pytest.mark.skipif(sys.platform != "win32", reason="khóa một tiến trình chỉ có trên Windows")
def test_second_agent_on_the_same_machine_exits_instead_of_opening_clients_twice(tmp_path):
    module = _load_agent_module()
    holder = module.single_instance(tmp_path)
    assert holder is not None
    other = subprocess.run([sys.executable, str(AGENT_FILE), "--web", "https://web.example", "--server", "1.2.3.4", "--data", "mini10", "--state-dir", str(tmp_path)],
                           capture_output=True, text=True, timeout=30, check=False)
    holder.close()
    assert other.returncode == 0 and "đã có agent khác" in (tmp_path / "agent.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra", [["--web", "https://x.vn\" & calc"], ["--server", "1.2.3.4 & calc"], ["--data", "mini10 & calc"]])
def test_agent_refuses_arguments_that_could_break_out_of_the_command_line(tmp_path, extra):
    base = {"--web": "https://web.example", "--server": "1.2.3.4", "--data": "mini10"}
    base[extra[0]] = extra[1]
    argv = [item for pair in base.items() for item in pair] + ["--state-dir", str(tmp_path)]
    assert _load_agent_module().main(argv) == 2


def test_stop_running_waits_for_the_lock_and_cleans_up(tmp_path):
    module = _load_agent_module()
    assert module.stop_running(tmp_path, wait_s=0) is True, "không có agent nào thì xong ngay"
    holder = module.single_instance(tmp_path)
    if module.msvcrt is not None:
        assert module.stop_running(tmp_path, wait_s=0) is False, "agent chưa thoát thì phải báo, không được xóa thư mục cài"
        assert (tmp_path / "agent.stop").exists()
    holder.close()
    assert module.stop_running(tmp_path, wait_s=0) is True
    assert not (tmp_path / "agent.stop").exists(), "tệp dừng còn lại sẽ làm agent mới thoát ngay"


@pytest.mark.skipif(sys.platform != "win32", reason="file .bat chỉ chạy trên Windows")
def test_client_launcher_refuses_more_than_five_clients_on_one_machine():
    done = subprocess.run(["cmd", "/c", str(ROOT / "deploy" / "lan_client" / "CHAY_CLIENT.bat"), "0", "6", "127.0.0.1"], env={**os.environ, "DEMO_NOPAUSE": "1"},
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=False)
    assert done.returncode == 1 and "tối đa 5 client" in done.stdout.decode("utf-8", "replace")


@pytest.mark.skipif(sys.platform != "win32", reason="registry chỉ có trên Windows")
def test_autostart_value_keeps_paths_with_special_characters_as_one_argument():
    """Ghi vào một khóa thử rồi xóa, không đụng khóa tự chạy thật. Đọc lại đúng cách Windows tách dòng lệnh."""
    import winreg
    module = _load_agent_module()
    subkey = r"Software\CropFedClientTest"
    command = [r"C:\Users\Nguyen & Tran (Nha)\app\python\pythonw.exe", r"C:\Users\Nguyen & Tran (Nha)\app\client_agent.py", "--web", "https://x.vn", "--server", "1.2.3.4", "--data", "mini10"]
    try:
        module.register_autostart(command, subkey=subkey)
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey) as key:
            value, kind = winreg.QueryValueEx(key, module.RUN_VALUE)
        assert kind == winreg.REG_SZ
        assert ctypes_argv(value) == command
    finally:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)


def ctypes_argv(line):
    import ctypes
    argc = ctypes.c_int()
    shell32 = ctypes.windll.shell32
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = shell32.CommandLineToArgvW(ctypes.c_wchar_p(line), ctypes.byref(argc))
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv)


def test_background_agent_uses_the_same_state_folder_as_the_install_that_printed_the_code(tmp_path):
    module = _load_agent_module()
    command = module.background_command("https://x.vn", "1.2.3.4", "mini10", tmp_path)
    assert command[command.index("--state-dir") + 1] == str(tmp_path)


def test_the_light_button_on_the_page_opens_five_clients():
    install = (ROOT / "deploy" / "lan_client" / "CAI_NUT_WEB.bat").read_text(encoding="utf-8")
    assert 'CHAY_CLIENT.bat\\" 0 5 %VM% %LIGHT%' in install and " 0 10 " not in install


def test_real_autostart_key_is_what_install_writes_and_what_remove_deletes():
    module = _load_agent_module()
    remove = (ROOT / "deploy" / "lan_client" / "GO_NUT_WEB.bat").read_text(encoding="utf-8")
    assert module.RUN_KEY in remove and module.RUN_VALUE in remove and "agent.stop" in remove
    assert "client_agent.py\" --install" in (ROOT / "deploy" / "lan_client" / "CAI_NUT_WEB.bat").read_text(encoding="utf-8")
