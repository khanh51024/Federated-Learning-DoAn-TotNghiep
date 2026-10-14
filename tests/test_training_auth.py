"""Bảo vệ các API huấn luyện trên máy chủ công khai: đăng nhập admin kiểm ở backend, một job một lúc (409), trần tham số,
hẹn giờ dừng, hạn mức và nhật ký thao tác. Không chạy huấn luyện thật (hàm huấn luyện được thay bằng bản giả)."""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import registry as reg
from web.app import app
from web.routers import training
from web.services import admin_auth, training_guard
from web.state import CURRENT_TASK

PASSWORD = "mat-khau-quan-tri-cho-test"


def _basic(user: str, password: str) -> dict[str, str]:
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


@pytest.fixture()
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("COMMUNITY_ADMIN_TOKEN", PASSWORD)
    monkeypatch.delenv("COMMUNITY_ADMIN_USER", raising=False)
    monkeypatch.setattr(reg, "TRAINING_AUDIT_JSONL", str(tmp_path / "audit.jsonl"))  # đường dẫn tuyệt đối thắng ROOT / ...
    monkeypatch.setattr(training, "run_centralized_training", lambda **kwargs: {"test_metrics": {"accuracy": 0.5}})
    admin_auth.LIMITER.reset()
    CURRENT_TASK.update(status="idle", cancel_requested=False, live_progress=[], job_id=None)
    yield TestClient(app)
    admin_auth.LIMITER.reset()
    CURRENT_TASK.update(status="idle", cancel_requested=False, live_progress=[], job_id=None)


def _audit_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


START = {"epochs": 2, "smoke": True, "seed": 1}


def test_without_a_configured_password_training_stays_open_for_local_use(monkeypatch: pytest.MonkeyPatch, api: TestClient):
    monkeypatch.delenv("COMMUNITY_ADMIN_TOKEN")
    assert api.post("/api/centralized/train", json=START).status_code == 200
    assert api.get("/api/status").json()["training"]["requires_admin"] is False


@pytest.mark.parametrize("path,payload", [
    ("/api/centralized/train", START),
    ("/api/federated/train", {"engine": "simulation", "rounds": 1, "local_epochs": 1, "alpha": 0.1}),
    ("/api/train/stop", {}),
])
def test_all_three_endpoints_refuse_unauthenticated_and_wrong_credentials(api: TestClient, path: str, payload: dict):
    assert api.post(path, json=payload).status_code == 401
    assert api.post(path, json=payload, headers=_basic("admin", "sai-mat-khau-sai-mat")).status_code == 401
    assert api.post(path, json=payload, headers=_basic("root", PASSWORD)).status_code == 401
    assert CURRENT_TASK["status"] == "idle", "yêu cầu bị từ chối không được làm đổi trạng thái tác vụ"


def test_authenticated_admin_can_start_via_basic_and_bearer(api: TestClient):
    first = api.post("/api/centralized/train", json=START, headers=_basic("admin", PASSWORD))
    assert first.status_code == 200 and first.json()["status"] == "started"
    second = api.post("/api/centralized/train", json=START, headers={"Authorization": f"Bearer {PASSWORD}"})
    assert second.status_code == 200  # job trước đã xong (hàm giả chạy ngay), nên được phép chạy tiếp


def test_failures_lock_the_ip_out_even_for_the_right_password(api: TestClient):
    codes = [api.post("/api/train/stop", json={}, headers=_basic("admin", "sai-mat-khau-so-" + str(i))).status_code for i in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    assert api.post("/api/train/stop", json={}, headers=_basic("admin", PASSWORD)).status_code == 429


def test_short_configured_password_disables_training_endpoints(monkeypatch: pytest.MonkeyPatch, api: TestClient):
    monkeypatch.setenv("COMMUNITY_ADMIN_TOKEN", "ngan")
    response = api.post("/api/centralized/train", json=START, headers=_basic("admin", "ngan"))
    assert response.status_code == 503 and response.json()["detail"]["code"] == "admin_disabled"


def test_only_one_job_runs_at_a_time_and_the_second_gets_409(monkeypatch: pytest.MonkeyPatch, api: TestClient):
    release = threading.Event()
    monkeypatch.setattr(training, "run_centralized_training", lambda **kwargs: release.wait(20) and {"test_metrics": {"accuracy": 0.5}})
    holder: dict[str, int] = {}
    worker = threading.Thread(
        target=lambda: holder.update(code=api.post("/api/centralized/train", json=START, headers=_basic("admin", PASSWORD)).status_code))
    worker.start()
    for _ in range(100):  # đợi tới khi job đầu đã được đặt trạng thái running
        if CURRENT_TASK["status"] == "running":
            break
        time.sleep(0.05)
    assert CURRENT_TASK["status"] == "running"
    clash = api.post("/api/federated/train", json={"engine": "simulation", "rounds": 1, "local_epochs": 1, "alpha": 0.1},
                     headers=_basic("admin", PASSWORD))
    clash_centralized = api.post("/api/centralized/train", json=START, headers=_basic("admin", PASSWORD))
    release.set()
    worker.join(timeout=30)
    assert clash.status_code == 409 and clash_centralized.status_code == 409
    assert holder["code"] == 200


def test_parameter_caps_reject_oversized_jobs_with_422(api: TestClient):
    caps = training_guard.limits()
    auth = _basic("admin", PASSWORD)
    too_many_epochs = api.post("/api/centralized/train", json={**START, "epochs": caps["max_centralized_epochs"] + 1}, headers=auth)
    assert too_many_epochs.status_code == 422 and str(caps["max_centralized_epochs"]) in too_many_epochs.json()["detail"]
    too_many_rounds = api.post("/api/federated/train", headers=auth, json={
        "engine": "simulation", "rounds": caps["max_rounds"] + 1, "local_epochs": 1, "alpha": 0.1})
    assert too_many_rounds.status_code == 422
    too_many_local = api.post("/api/federated/train", headers=auth, json={
        "engine": "simulation", "rounds": 1, "local_epochs": caps["max_local_epochs"] + 1, "alpha": 0.1})
    assert too_many_local.status_code == 422
    assert CURRENT_TASK["status"] == "idle"


def test_caps_can_be_changed_by_environment_within_hard_bounds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAIN_MAX_ROUNDS", "4")
    monkeypatch.setenv("TRAIN_MAX_SECONDS", "999999")  # vượt trần cứng 7200
    monkeypatch.setenv("TRAIN_MAX_LOCAL_EPOCHS", "không phải số")
    limits = training_guard.limits()
    assert limits["max_rounds"] == 4 and limits["max_seconds"] == 7200 and limits["max_local_epochs"] == 3


def test_start_rate_limit_per_ip(monkeypatch: pytest.MonkeyPatch, api: TestClient):
    monkeypatch.setenv("TRAIN_STARTS_PER_10_MIN", "3")
    codes = [api.post("/api/centralized/train", json=START, headers=_basic("admin", PASSWORD)).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_actions_are_written_to_the_audit_log(api: TestClient, tmp_path: Path):
    auth = _basic("admin", PASSWORD)
    assert api.post("/api/centralized/train", json=START, headers=auth).status_code == 200
    events = _audit_events(tmp_path)
    kinds = [event["event"] for event in events]
    assert kinds[:2] == ["train_start", "train_end"], kinds
    start = events[0]
    assert start["user"] == "admin" and start["config"]["epochs"] == 2 and start["kind"] == "centralized"
    assert events[1]["job_id"] == start["job_id"] and events[1]["status"] == "completed"
    assert "mat-khau" not in json.dumps(events), "nhật ký không được chứa mật khẩu"


def test_expire_stops_only_the_matching_running_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(reg, "TRAINING_AUDIT_JSONL", str(tmp_path / "audit.jsonl"))
    CURRENT_TASK.update(status="running", type="federated", job_id="job-a", cancel_requested=False, message="")
    training._expire_job("job-b")
    assert CURRENT_TASK["cancel_requested"] is False
    training._expire_job("job-a")
    assert CURRENT_TASK["cancel_requested"] is True and "tự dừng" in CURRENT_TASK["message"]
    CURRENT_TASK.update(status="idle", cancel_requested=False, job_id=None)


def test_status_exposes_limits_and_admin_requirement(api: TestClient):
    block = api.get("/api/status").json()["training"]
    assert block["requires_admin"] is True
    assert set(block["limits"]) >= {"max_centralized_epochs", "max_rounds", "max_local_epochs", "max_seconds"}
    assert "mat-khau" not in json.dumps(block)


def test_community_admin_and_training_share_one_failure_counter(api: TestClient):
    """Đoán mật khẩu bằng cách xen kẽ hai nhóm API không được có nhiều lượt hơn."""
    from web.routers import community_fl
    from web.services.community_store import RateLimiter

    for index in range(3):
        api.post("/api/train/stop", json={}, headers=_basic("admin", f"sai-so-{index}-sai-so-{index}"))
    for _ in range(2):
        admin_auth.record_failure(admin_auth.LIMITER, "testclient")
    assert admin_auth.locked_out(admin_auth.LIMITER, "testclient")
    assert not admin_auth.locked_out(RateLimiter(), "testclient")  # bộ đếm riêng của store test không bị ảnh hưởng
    assert community_fl.admin_auth is admin_auth


# ───────────────────────── giao diện bước 5 ─────────────────────────
def test_training_page_checks_the_gate_before_posting_and_sends_the_admin_header():
    root = Path(__file__).resolve().parents[1] / "web" / "static"
    source = (root / "js" / "pages" / "training.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")
    start = source[source.index("async function start()"):source.index("async function stopTraining()")]
    assert start.index("ensureAdmin()") < start.index("validateTrainForm()") < start.index("postJson(")
    assert start.count("storedAuthHeader()") == 2, "cả Flower và Centralized đều phải gửi tiêu đề đăng nhập"
    stop = source[source.index("async function stopTraining()"):source.index("export async function mount")]
    assert stop.index("ensureAdmin()") < stop.index("postJson(") and "storedAuthHeader()" in stop
    assert "handleAuthFailure(error)" in start and "handleAuthFailure(error)" in stop
    assert 'id="train-gate"' in html and 'href="/quan-tri"' in html
    assert "/api/status" in (root / "js" / "core" / "status.js").read_text(encoding="utf-8")


def test_admin_page_and_training_page_share_one_credential_module():
    root = Path(__file__).resolve().parents[1] / "web" / "static"
    shared = (root / "js" / "core" / "admin-auth.js").read_text(encoding="utf-8")
    assert 'const KEY = "cropfed.admin.token"' in shared
    admin = (root / "admin" / "admin.js").read_text(encoding="utf-8")
    training_js = (root / "js" / "pages" / "training.js").read_text(encoding="utf-8")
    assert "/static/js/core/admin-auth.js" in admin and "../core/admin-auth.js" in training_js
    assert "cropfed.admin.token" not in admin and "cropfed.admin.token" not in training_js, "khóa lưu chỉ khai báo một nơi"


def test_requests_without_credentials_do_not_lock_out_the_real_admin(api: TestClient):
    """Bot gọi không kèm đăng nhập chỉ nhận 401, không được tính là lần đoán sai (nếu không sẽ khóa được admin thật)."""
    for _ in range(12):
        assert api.post("/api/train/stop", json={}).status_code == 401
    assert api.post("/api/train/stop", json={}, headers=_basic("admin", PASSWORD)).status_code == 200
