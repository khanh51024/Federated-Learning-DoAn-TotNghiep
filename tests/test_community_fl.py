"""Học phân tán cộng đồng: lõi số học, trạng thái vòng (SQLite) và giao thức HTTP.

Không cần dataset, ONNX hay trình duyệt: tập đặc trưng giám sát và head khởi điểm được dựng tổng hợp (38 cụm Gaussian trong
không gian 1024 chiều) nên cổng chấp nhận có nghĩa thật. Test tích hợp với artifact thật nằm ở tests/test_community_assets.py.
"""
from __future__ import annotations

import base64
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.federated import community as core
from src.federated.community import CommunityError, EvalSet, Policy
from web.app import app
from web.routers import community_fl
from web.routers.community_fl import StoreState, store_state
from web.services.community_store import CommunityAssets, CommunityStore

ADMIN = "admin-token-for-tests-0123456789"
SECRET = "secret-for-tests-only"


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _synthetic_assets(tmp_path: Path, with_model: bool = True) -> CommunityAssets:
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(core.NUM_CLASSES, core.FEATURE_DIM)).astype(np.float32)
    per_class = 12
    labels = np.repeat(np.arange(core.NUM_CLASSES), per_class)
    features = centers[labels] + 0.35 * rng.normal(size=(len(labels), core.FEATURE_DIM)).astype(np.float32)
    sources = (np.arange(len(labels)) % 5 == 0).astype(np.int8)  # một phần "PlantDoc"
    head = core.as_head(centers * 0.5, np.zeros(core.NUM_CLASSES, dtype=np.float32))
    model_path = tmp_path / "feature_model.onnx"
    model_path.write_bytes(b"onnx-bytes-for-test" * 50)
    return CommunityAssets(
        classes=[f"Class_{i:02d}___name" for i in range(core.NUM_CLASSES)], base_head=head,
        evalset=EvalSet(features, labels, sources, "test"), manifest={},
        feature_model_path=model_path if with_model else None,
        feature_model_sha=core.file_sha256(model_path) if with_model else None,
        feature_model_bytes=model_path.stat().st_size if with_model else 0,
    )


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
def store(tmp_path: Path, clock: Clock):
    return CommunityStore(tmp_path / "state", _synthetic_assets(tmp_path), SECRET, ADMIN, clock=clock)


@pytest.fixture()
def api(store: CommunityStore):
    app.dependency_overrides[store_state] = lambda: StoreState(store, None)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(store_state, None)


def _admin(client: TestClient, path: str, payload=None, method: str = "post", token: str = ADMIN):
    headers = {"Authorization": f"Bearer {token}"}
    call = getattr(client, method)
    return call(f"/api/community/admin/{path}", headers=headers, **({"json": payload} if method == "post" else {}))


def _open(client: TestClient, **changes) -> None:
    body = {"open": True, "changes": changes}
    assert _admin(client, "settings", body).status_code == 200


def _codes(client: TestClient, count: int = 1, max_uses: int = 50) -> list[str]:
    response = _admin(client, "codes", {"count": count, "max_uses": max_uses, "ttl_hours": 2})
    assert response.status_code == 200, response.text
    return response.json()["codes"]


def _join(client: TestClient, code: str) -> dict[str, str]:
    response = client.post("/api/community/join", json={"code": code})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _parent_head(client: TestClient, headers) -> tuple[dict, core.Head]:
    data = client.get("/api/community/round", headers=headers).json()
    return data, core.decode_head(data["head"])


def _client_head(parent: core.Head, scale: float, seed: int) -> core.Head:
    """Mô phỏng một client: dịch nhẹ head cha theo hướng ngẫu nhiên (đủ để kiểm gộp/cắt chuẩn)."""
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=parent[core.WEIGHT_KEY].shape).astype(np.float32)
    delta *= scale / float(np.linalg.norm(delta))
    return core.as_head(parent[core.WEIGHT_KEY] + delta, parent[core.BIAS_KEY].copy())


def _payload(round_data: dict, client_head: core.Head, n: int = 30, **overrides) -> dict:
    counts = [0] * core.NUM_CLASSES
    counts[0], counts[1] = n - n // 2, n // 2
    body = {
        "round_id": round_data["round"]["round_id"], "parent_sha": round_data["round"]["parent_sha"],
        "head": core.encode_head(client_head), "num_examples": n, "label_counts": counts,
        "local": {"epochs": 3, "lr": 0.01, "acc_before": 0.5, "acc_after": 0.8, "note": "bị bỏ"}, "dp_applied": False,
    }
    body.update(overrides)
    return body


# ───────────────────────── lõi số học ─────────────────────────
def test_head_wire_format_round_trips_and_rejects_malformed_payloads():
    rng = np.random.default_rng(0)
    head = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    assert core.HEAD_BYTES == 155_800 and core.HEAD_PARAMS == 38_950
    again = core.decode_head(core.encode_head(head))
    assert np.array_equal(again[core.WEIGHT_KEY], head[core.WEIGHT_KEY]) and core.head_sha(again) == core.head_sha(head)
    for bad in ("###", base64.b64encode(b"short").decode(), base64.b64encode(b"\0" * (core.HEAD_BYTES + 4)).decode()):
        with pytest.raises(CommunityError):
            core.decode_head(bad)


def test_check_update_rejects_bad_structure_nan_parent_and_noop_and_clips_the_norm():
    rng = np.random.default_rng(1)
    parent = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    sha = core.head_sha(parent)
    policy = Policy(clip_norm=0.5)
    child = _client_head(parent, 2.0, 3)
    result = core.check_update(parent, child, sha, policy)
    assert result.clipped and result.norm_raw == pytest.approx(2.0, rel=1e-4) and result.norm_used == pytest.approx(0.5, rel=1e-3)
    delta = np.concatenate([(result.head[k] - parent[k]).ravel() for k in parent])
    assert np.linalg.norm(delta) == pytest.approx(0.5, rel=1e-3)

    unclipped = core.check_update(parent, _client_head(parent, 0.2, 4), sha, policy)
    assert not unclipped.clipped
    with pytest.raises(CommunityError) as stale:
        core.check_update(parent, child, "0" * 64, policy)
    assert stale.value.code == "stale_parent"
    nan_head = {k: v.copy() for k, v in child.items()}
    nan_head[core.WEIGHT_KEY][0, 0] = np.nan
    with pytest.raises(CommunityError) as non_finite:
        core.check_update(parent, nan_head, sha, policy)
    assert non_finite.value.code == "non_finite"
    with pytest.raises(CommunityError) as noop:
        core.check_update(parent, {k: v.copy() for k, v in parent.items()}, sha, policy)
    assert noop.value.code == "no_change"
    with pytest.raises(CommunityError) as shape:
        core.check_update(parent, {core.WEIGHT_KEY: parent[core.WEIGHT_KEY][:10], core.BIAS_KEY: parent[core.BIAS_KEY]}, sha, policy)
    assert shape.value.code == "bad_structure"


def test_aggregate_heads_equals_numpy_weighted_average_with_cap():
    rng = np.random.default_rng(2)
    parent = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    heads = [_client_head(parent, 0.3, seed) for seed in (1, 2, 3)]
    sizes = [10, 50, 900]  # 900 bị chặn ở trần 200
    merged = core.aggregate_heads([{"head": h, "num_examples": n} for h, n in zip(heads, sizes)], parent, Policy(n_cap=200))
    weights = np.array([10, 50, 200], dtype=np.float64)
    expected = sum(w * h[core.WEIGHT_KEY].astype(np.float64) for w, h in zip(weights, heads)) / weights.sum()
    assert np.allclose(merged[core.WEIGHT_KEY], expected, atol=1e-6)
    assert merged[core.WEIGHT_KEY].dtype == np.float32


def test_gate_rejects_regressions_and_keeps_small_noise():
    parent = {"accuracy": 0.9, "macro_recall": 0.85, "plantdoc_accuracy": 0.5}
    policy = Policy()
    assert core.gate_decision({"accuracy": 0.899, "macro_recall": 0.85, "plantdoc_accuracy": 0.49}, parent, policy)[0]
    ok, reasons = core.gate_decision({"accuracy": 0.80, "macro_recall": 0.7, "plantdoc_accuracy": 0.3}, parent, policy)
    assert not ok and set(reasons) == {"accuracy_drop", "macro_recall_drop", "plantdoc_drop"}


def test_policy_bounds_are_enforced():
    assert core.policy_from({"k_target": 5, "k_min": 2}).k_target == 5
    for bad in ({"k_target": 0}, {"clip_norm": -1}, {"k_target": 2, "k_min": 3}, {"dp_enabled": "yes"}, {"epsilon": float("nan")}):
        with pytest.raises(CommunityError):
            core.policy_from(bad)


# ───────────────────────── giao thức HTTP ─────────────────────────
def test_status_is_public_and_session_starts_closed(api: TestClient):
    data = api.get("/api/community/status").json()
    assert data["enabled"] and data["open"] is False and data["version"] == 0 and data["round"] is None
    assert len(data["classes"]) == 38 and data["feature_model"]["bytes"] > 0
    assert data["config"]["epochs"] == 3 and data["config"]["lr"] == 0.01
    refused = api.post("/api/community/join", json={"code": "ABCDE-FGHJK"})
    assert refused.status_code == 409


def test_admin_requires_the_token_and_is_disabled_without_one(api: TestClient, store: CommunityStore):
    assert api.get("/api/community/admin/state").status_code == 401
    assert _admin(api, "state", method="get", token="wrong").status_code == 401
    assert _admin(api, "state", method="get").status_code == 200
    store._admin_token = None
    assert _admin(api, "state", method="get").status_code == 503


def test_join_with_bad_wrong_expired_and_exhausted_codes(api: TestClient, clock: Clock):
    _open(api)
    assert api.post("/api/community/join", json={"code": "AAAAA-AAAAA"}).status_code == 403
    assert api.post("/api/community/join", json={"code": "x"}).status_code == 422
    [one_use] = _codes(api, max_uses=1)
    assert api.post("/api/community/join", json={"code": one_use.lower().replace("-", " ")}).status_code == 200
    assert api.post("/api/community/join", json={"code": one_use}).status_code == 403  # hết lượt
    [expiring] = _codes(api)
    clock.advance(3 * 3600)
    assert api.post("/api/community/join", json={"code": expiring}).status_code == 403  # hết hạn (ttl 2 giờ)


def test_codes_are_stored_hashed_not_in_clear(api: TestClient, store: CommunityStore):
    _open(api)
    [code] = _codes(api)
    raw = code.replace("-", "")
    with sqlite3.connect(store.db_path) as conn:
        digests = [row[0] for row in conn.execute("SELECT digest FROM codes")]
    assert digests and all(raw not in d and raw.lower() not in d for d in digests)


def test_token_must_be_valid_unexpired_and_not_revoked(api: TestClient, clock: Clock):
    _open(api)
    headers = _join(api, _codes(api)[0])
    assert api.get("/api/community/round", headers=headers).status_code == 200
    assert api.get("/api/community/round").status_code == 401
    assert api.get("/api/community/round", headers={"Authorization": "Bearer v1.abc.def"}).status_code == 401
    token = headers["Authorization"].split(" ", 1)[1]
    version, body, signature = token.split(".")
    forged = f"{version}.{body}.{signature[:-2]}AA"
    assert api.get("/api/community/round", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    clock.advance(7 * 3600)
    assert api.get("/api/community/round", headers=headers).status_code == 401


def test_participant_revocation_blocks_the_token(api: TestClient):
    _open(api)
    token = _join(api, _codes(api)[0])
    suffix = token["Authorization"].split(".")[1]
    state = _admin(api, "state", method="get").json()
    assert state["participants"]["total"] == 1
    # thu hồi theo mã: mọi người đã vào bằng mã đó đều mất phiên
    code_id = state["codes"][0]["id"]
    assert _admin(api, f"codes/{code_id}/revoke", {}).status_code == 200
    assert api.get("/api/community/round", headers=token).status_code == 401
    assert suffix


def test_full_round_aggregates_exactly_fedavg_and_publishes_a_new_version(api: TestClient, store: CommunityStore):
    _open(api, k_target=3, k_min=2, clip_norm=50.0)
    codes = _codes(api, count=3)
    participants = [_join(api, c) for c in codes]
    round_data, parent = _parent_head(api, participants[0])
    assert round_data["round"]["k_target"] == 3 and round_data["round"]["parent_version"] == 0
    sizes = [20, 40, 60]
    heads = [_client_head(parent, 0.4, seed) for seed in (11, 12, 13)]
    results = []
    for headers, head, n in zip(participants, heads, sizes):
        response = api.post("/api/community/update", headers=headers, json=_payload(round_data, head, n))
        assert response.status_code == 200, response.text
        results.append(response.json())
    assert [r["aggregated"] for r in results] == [False, False, True]
    final = results[-1]["result"]
    assert final["status"] == "aggregated" and final["version"] == 1 and final["clients"] == 3 and final["examples"] == 120

    version, head, _sha, _metrics = store.current_head()
    weights = np.array(sizes, dtype=np.float64)
    expected = sum(w * h[core.WEIGHT_KEY].astype(np.float64) for w, h in zip(weights, heads)) / weights.sum()
    assert version == 1 and np.allclose(head[core.WEIGHT_KEY], expected, atol=1e-5)
    # vòng kế tiếp mở với head cha mới; client cũ có thể gửi lại
    next_round, _ = _parent_head(api, participants[0])
    assert next_round["round"]["round_id"] == round_data["round"]["round_id"] + 1
    assert next_round["round"]["parent_version"] == 1 and next_round["already_submitted"] is False
    history = api.get("/api/community/history").json()
    assert history["current_version"] == 1 and history["totals"]["accepted"] == 1
    assert [v["version"] for v in history["versions"]] == [0, 1]
    assert "participant" not in json.dumps(history)


def test_round_detail_records_weights_norms_latency_and_serves_the_head_file(api: TestClient, store: CommunityStore):
    _open(api, k_target=3, k_min=2, clip_norm=50.0, n_cap=50)
    participants = [_join(api, c) for c in _codes(api, count=3)]
    round_data, parent = _parent_head(api, participants[0])
    sizes = [20, 40, 80]
    for i, (headers, n) in enumerate(zip(participants, sizes)):
        local = {"train_ms": 1000 + i, "extract_ms": 500, "rtt_ms": 40, "acc_before": 0.5, "acc_after": 0.6}
        response = api.post("/api/community/update", headers=headers, json=_payload(round_data, _client_head(parent, 0.4, 20 + i), n, local=local))
        assert response.status_code == 200, response.text
    history = api.get("/api/community/history").json()
    finished = history["rounds"][0]
    people = finished["contributors"]
    assert [c["n"] for c in people] == sizes and [c["n_used"] for c in people] == [20, 40, 50], "trần n_cap áp dụng cho trọng số"
    weights = [c["weight"] for c in people]
    assert sum(weights) == pytest.approx(1.0, abs=1e-4) and weights[1] == pytest.approx(40 / 110, abs=1e-5)
    assert all(c["norm"] == pytest.approx(0.4, rel=1e-3) and not c["clipped"] for c in people)
    assert [c["train_ms"] for c in people] == [1000.0, 1001.0, 1002.0] and all(c["rtt_ms"] == 40.0 and c["extract_ms"] == 500.0 for c in people)
    assert all(c["wait_s"] >= 0 for c in people)
    assert finished["delta_norm"] > 0 and finished["duration_s"] >= 0 and finished["agg_ms"] is not None and finished["n_cap"] == 50
    assert "participant" not in json.dumps(history), "chi tiết đóng góp phải ẩn danh"

    newest = history["versions"][-1]
    assert newest["version"] == 1 and newest["downloadable"] is True
    download = api.get("/api/community/heads/1")
    assert download.status_code == 200 and download.headers["x-head-sha"] == newest["sha"]
    with np.load(io.BytesIO(download.content), allow_pickle=False) as data:
        assert data["weight"].shape == (core.NUM_CLASSES, core.FEATURE_DIM) and data["bias"].shape == (core.NUM_CLASSES,)
        assert core.head_sha(core.as_head(data["weight"], data["bias"])) == newest["sha"]
    assert api.get("/api/community/heads/99").status_code == 404


def test_store_upgrades_a_database_created_before_the_detail_column(tmp_path: Path, clock: Clock):
    first = CommunityStore(tmp_path / "state", _synthetic_assets(tmp_path), SECRET, ADMIN, clock=clock)
    with sqlite3.connect(first.db_path) as conn:
        conn.execute("ALTER TABLE rounds DROP COLUMN detail")
    again = CommunityStore(tmp_path / "state", _synthetic_assets(tmp_path), SECRET, ADMIN, clock=clock)
    with sqlite3.connect(again.db_path) as conn:
        assert "detail" in {row[1] for row in conn.execute("PRAGMA table_info(rounds)")}
    assert again.history()["rounds"] == []


def test_have_sha_skips_resending_the_head(api: TestClient):
    _open(api)
    headers = _join(api, _codes(api)[0])
    first = api.get("/api/community/round", headers=headers).json()
    again = api.get("/api/community/round", params={"have": first["round"]["parent_sha"]}, headers=headers).json()
    assert first["head"] and again["head"] is None


def test_update_validation_rejections_leave_the_round_untouched(api: TestClient, store: CommunityStore):
    _open(api, clip_norm=50.0)
    headers = _join(api, _codes(api)[0])
    round_data, parent = _parent_head(api, headers)
    good = _client_head(parent, 0.3, 5)

    def post(**overrides):
        return api.post("/api/community/update", headers=headers, json=_payload(round_data, good, **overrides))

    assert post(num_examples=3, label_counts=[3] + [0] * 37).status_code == 422          # ít hơn 5 ảnh
    assert post(label_counts=[1] * 38).status_code == 422                                   # tổng nhãn ≠ số ảnh
    assert post(parent_sha="a" * 64).status_code == 409                                     # head cha sai
    assert post(round_id=999).status_code == 409                                            # vòng khác
    assert post(head="@@@" * 100).status_code == 422                                        # base64 hỏng
    bad = {k: v.copy() for k, v in good.items()}
    bad[core.BIAS_KEY][3] = np.inf
    assert api.post("/api/community/update", headers=headers, json=_payload(round_data, bad)).status_code == 422
    still = api.get("/api/community/round", headers=headers).json()
    assert still["round"]["received"] == 0 and still["already_submitted"] is False
    ok = post()
    assert ok.status_code == 200 and ok.json()["accepted"]
    assert post().status_code == 409                                                        # mỗi người một cập nhật mỗi vòng


def test_oversized_update_body_is_refused_before_parsing(api: TestClient):
    _open(api)
    headers = _join(api, _codes(api)[0])
    big = json.dumps({"head": "A" * 400_000})
    assert api.post("/api/community/update", headers={**headers, "Content-Type": "application/json"}, content=big).status_code == 413


def test_server_clips_large_updates_so_one_person_cannot_dominate(api: TestClient, store: CommunityStore):
    _open(api, k_target=2, k_min=2, clip_norm=0.5)
    a, b = (_join(api, c) for c in _codes(api, count=2))
    round_data, parent = _parent_head(api, a)
    huge = _client_head(parent, 40.0, 21)
    response = api.post("/api/community/update", headers=a, json=_payload(round_data, huge)).json()
    assert response["clipped"] and response["norm"] == pytest.approx(40.0, rel=1e-3) and response["norm_used"] == pytest.approx(0.5, rel=1e-2)
    api.post("/api/community/update", headers=b, json=_payload(round_data, _client_head(parent, 0.1, 22)))
    _version, head, _sha, _m = store.current_head()
    drift = np.linalg.norm(head[core.WEIGHT_KEY] - parent[core.WEIGHT_KEY])
    assert drift <= 0.5 + 1e-4  # trung bình có trọng số của hai delta đều ≤ 0,5


def test_gate_rejects_a_poisoned_round_and_keeps_the_old_head(api: TestClient, store: CommunityStore):
    _open(api, k_target=3, k_min=3, clip_norm=100.0)
    people = [_join(api, c) for c in _codes(api, count=3)]
    round_data, parent = _parent_head(api, people[0])
    poison_w = -parent[core.WEIGHT_KEY] * 3.0  # đảo dấu: phá nặng độ chính xác
    poison = core.as_head(poison_w, parent[core.BIAS_KEY])
    outcomes = [api.post("/api/community/update", headers=h, json=_payload(round_data, poison, 25)).json() for h in people]
    result = outcomes[-1]["result"]
    assert result["status"] == "rejected" and "accuracy_drop" in result["reasons"]
    assert result["version"] is None and result["messages"]
    version, head, sha, _ = store.current_head()
    assert version == 0 and sha == core.head_sha(parent)
    assert api.get("/api/community/history").json()["totals"]["rejected"] == 1
    # round sau vẫn mở với đúng head cũ
    again, _ = _parent_head(api, people[0])
    assert again["round"]["parent_version"] == 0 and again["round"]["round_id"] == round_data["round"]["round_id"] + 1


def test_deadline_aggregates_with_k_min_and_extends_without_it(api: TestClient, store: CommunityStore, clock: Clock):
    _open(api, k_target=5, k_min=2, round_ttl_seconds=60, clip_norm=50.0)
    people = [_join(api, c) for c in _codes(api, count=3)]
    round_data, parent = _parent_head(api, people[0])
    api.post("/api/community/update", headers=people[0], json=_payload(round_data, _client_head(parent, 0.2, 31)))
    clock.advance(61)
    extended = api.get("/api/community/round", headers=people[0]).json()  # 1 < k_min: gia hạn, vòng vẫn là vòng cũ
    assert extended["round"]["round_id"] == round_data["round"]["round_id"] and extended["round"]["extensions"] == 1
    api.post("/api/community/update", headers=people[1], json=_payload(round_data, _client_head(parent, 0.2, 32)))
    clock.advance(61)
    api.get("/api/community/status")  # chỉ cần có yêu cầu đến sau hạn
    assert store.current_head()[0] == 1
    assert api.get("/api/community/history").json()["rounds"][0]["clients"] == 2


def test_state_survives_a_restart(tmp_path: Path, clock: Clock):
    state_dir = tmp_path / "state"
    assets = _synthetic_assets(tmp_path)
    first = CommunityStore(state_dir, assets, SECRET, ADMIN, clock=clock)
    first.admin_configure(True, {"k_target": 2, "k_min": 2, "clip_norm": 50.0})
    [code] = first.admin_create_codes(1, 10, 2)
    token = first.join(code)["token"]
    participant = first.authenticate(token)
    view = first.round_view(participant)
    parent = core.decode_head(view["head"])
    body = _payload(view, _client_head(parent, 0.3, 41))
    first.submit_update(participant, body)

    second = CommunityStore(state_dir, assets, SECRET, ADMIN, clock=clock)
    assert second.authenticate(token) == participant  # token ký bằng cùng secret vẫn hợp lệ
    again = second.round_view(participant)
    assert again["round"]["round_id"] == view["round"]["round_id"] and again["round"]["received"] == 1
    assert again["already_submitted"] is True
    [code2] = second.admin_create_codes(1, 10, 2)
    other = second.authenticate(second.join(code2)["token"])
    done = second.submit_update(other, _payload(again, _client_head(parent, 0.3, 42)))
    assert done["aggregated"] and second.current_head()[0] == 1
    third = CommunityStore(state_dir, assets, SECRET, ADMIN, clock=clock)
    assert third.current_head()[0] == 1 and third.history()["totals"]["accepted"] == 1


def test_rollback_restores_old_content_as_a_new_version(api: TestClient, store: CommunityStore):
    _open(api, k_target=2, k_min=2, clip_norm=50.0)
    a, b = (_join(api, c) for c in _codes(api, count=2))
    round_data, parent = _parent_head(api, a)
    api.post("/api/community/update", headers=a, json=_payload(round_data, _client_head(parent, 0.3, 51)))
    api.post("/api/community/update", headers=b, json=_payload(round_data, _client_head(parent, 0.3, 52)))
    assert store.current_head()[0] == 1
    result = _admin(api, "rollback", {"version": 0}).json()
    assert result["version"] == 2 and result["restored_from"] == 0
    version, head, sha, _ = store.current_head()
    assert version == 2 and sha == core.head_sha(parent)
    assert _admin(api, "rollback", {"version": 99}).status_code == 404
    kinds = [v["kind"] for v in api.get("/api/community/history").json()["versions"]]
    assert kinds == ["base", "aggregate", "rollback"]


def test_closing_the_session_cancels_the_open_round_and_blocks_updates(api: TestClient):
    _open(api, clip_norm=50.0)
    headers = _join(api, _codes(api)[0])
    round_data, parent = _parent_head(api, headers)
    assert _admin(api, "settings", {"open": False, "changes": {}}).status_code == 200
    response = api.post("/api/community/update", headers=headers, json=_payload(round_data, _client_head(parent, 0.3, 61)))
    assert response.status_code == 409
    assert api.get("/api/community/round", headers=headers).json()["open"] is False


def test_join_is_rate_limited_after_repeated_failures(api: TestClient):
    _open(api)
    attempts = community_fl.JOIN_FAILS_PER_15_MIN + 5
    statuses = [api.post("/api/community/join", json={"code": f"BADCD-{i:05d}"[:11]}).status_code for i in range(attempts)]
    assert statuses[0] == 403 and statuses[-1] == 429 and statuses.count(403) == community_fl.JOIN_FAILS_PER_15_MIN


def test_join_is_disabled_without_a_secret(tmp_path: Path, clock: Clock):
    local = CommunityStore(tmp_path / "state2", _synthetic_assets(tmp_path), None, ADMIN, clock=clock)
    local.admin_configure(True, {})
    with pytest.raises(CommunityError) as error:
        local.join("ABCDE-FGHJK")
    assert error.value.code == "disabled"


def test_feature_model_is_served_immutable_with_its_sha(api: TestClient, store: CommunityStore):
    response = api.get("/api/community/feature-model")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]
    assert response.headers["x-content-sha256"] == store.assets.feature_model_sha
    assert hashlib_sha(response.content) == store.assets.feature_model_sha


def hashlib_sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_missing_artifacts_disable_the_feature_without_breaking_status(tmp_path: Path):
    app.dependency_overrides[store_state] = lambda: StoreState(None, "Học phân tán cộng đồng chưa được bật trên máy chủ này.",
                                                              "Chưa có community_manifest.json")
    try:
        client = TestClient(app)
        data = client.get("/api/community/status").json()
        assert data["enabled"] is False and "manifest" not in data["reason"]  # lý do công khai không lộ chi tiết nội bộ
        assert client.post("/api/community/join", json={"code": "ABCDE-FGHJK"}).status_code == 503
        assert client.get("/api/community/history").status_code == 503
    finally:
        app.dependency_overrides.pop(store_state, None)


def test_assets_loader_fails_closed_on_sha_mismatch(tmp_path: Path):
    from web.services.community_store import AssetsUnavailable

    rng = np.random.default_rng(3)
    root = tmp_path / "assets"
    root.mkdir()
    head = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    np.savez(root / "base_head.npz", weight=head[core.WEIGHT_KEY], bias=head[core.BIAS_KEY])
    (root / "feature_model.onnx").write_bytes(b"x" * 10)
    np.savez(root / "eval_features.npz", features=np.zeros((4, 1024), np.float16), labels=np.zeros(4, np.int64),
             sources=np.array([0, 1, 0, 1], np.int8))
    feature_sha = core.file_sha256(root / "feature_model.onnx")
    manifest = {
        "classes": [f"c{i}" for i in range(38)],
        "feature_model": {"file": "feature_model.onnx", "sha256": feature_sha, "bytes": 10},
        "base_head": {"file": "base_head.npz", "sha256": core.file_sha256(root / "base_head.npz"), "head_sha": core.head_sha(head)},
        "eval_features": {"file": "eval_features.npz", "sha256": core.file_sha256(root / "eval_features.npz"), "count": 4,
                          "feature_model_sha256": feature_sha},
    }
    path = root / "community_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert CommunityAssets.from_directory(root).feature_model_sha == feature_sha
    (root / "feature_model.onnx").write_bytes(b"tampered")
    with pytest.raises(AssetsUnavailable):
        CommunityAssets.from_directory(root)


def test_reset_store_state_forces_a_rebuild():
    community_fl.reset_store_state()
    assert community_fl._STATE is None


# ───────────────────────── gia cố sau rà soát bảo mật / review ─────────────────────────
def test_one_code_cannot_fill_the_quorum_with_several_tokens(api: TestClient, store: CommunityStore):
    _open(api, k_target=3, k_min=2, clip_norm=50.0)
    [code] = _codes(api, max_uses=5)
    tokens = [_join(api, code) for _ in range(3)]  # một người vào nhiều lần bằng cùng một mã
    round_data, parent = _parent_head(api, tokens[0])
    results = [api.post("/api/community/update", headers=t, json=_payload(round_data, _client_head(parent, 0.3, 70 + i)))
               for i, t in enumerate(tokens)]
    assert [r.status_code for r in results] == [200, 409, 409]
    assert "mã tham gia" in results[1].json()["detail"]["message"].lower()
    assert api.get("/api/community/round", headers=tokens[2]).json()["already_submitted"] is True
    assert api.get("/api/community/round", headers=tokens[2]).json()["round"]["received"] == 1


def test_cumulative_drift_is_bounded_even_when_each_round_is_within_epsilon():
    policy = Policy(epsilon=0.005, budget=0.01)
    best = core.best_of([{"accuracy": 0.90, "macro_recall": 0.85, "plantdoc_accuracy": 0.5}])
    parent = {"accuracy": 0.894, "macro_recall": 0.846, "plantdoc_accuracy": 0.49}      # đã tụt 0,4 điểm so với tốt nhất
    candidate = {"accuracy": 0.8895, "macro_recall": 0.8455, "plantdoc_accuracy": 0.49}  # chỉ tụt 0,45 so với cha (< epsilon)
    ok, reasons = core.gate_decision(candidate, parent, policy, best)
    assert not ok and reasons == ["accuracy_drift"]
    assert core.gate_decision(candidate, parent, policy)[0]  # không có mức tốt nhất thì chỉ xét theo vòng (hành vi cũ)


def test_a_metric_the_candidate_cannot_compute_counts_as_a_regression():
    parent = {"accuracy": 0.9, "macro_recall": 0.8, "plantdoc_accuracy": 0.5}
    ok, reasons = core.gate_decision({"accuracy": 0.9, "macro_recall": 0.8, "plantdoc_accuracy": None}, parent, Policy())
    assert not ok and "plantdoc_drop" in reasons


def test_k_must_be_at_least_two():
    for bad in ({"k_target": 1}, {"k_min": 1}):
        with pytest.raises(CommunityError):
            core.policy_from(bad)


def test_revoked_participants_pending_update_is_not_aggregated(api: TestClient, store: CommunityStore):
    _open(api, k_target=3, k_min=2, clip_norm=50.0)
    codes = _codes(api, count=2)
    a, b = (_join(api, code) for code in codes)
    round_data, parent = _parent_head(api, a)
    for headers, seed in ((a, 81), (b, 82)):
        assert api.post("/api/community/update", headers=headers, json=_payload(round_data, _client_head(parent, 0.3, seed))).status_code == 200
    by_id = sorted(_admin(api, "state", method="get").json()["codes"], key=lambda c: c["id"])
    assert _admin(api, f"codes/{by_id[1]['id']}/revoke", {}).status_code == 200  # thu hồi người thứ hai
    assert api.get("/api/community/round", headers=b).status_code == 401
    closed = _admin(api, "round/close", {}).json()
    assert closed["status"] == "aggregated" and closed["clients"] == 1  # chỉ còn cập nhật của người còn hiệu lực


def test_admin_body_is_not_parsed_before_authentication_and_is_size_limited(api: TestClient):
    huge = json.dumps({"open": True, "changes": {"note": "x" * 5_000_000}})
    unauthenticated = api.post("/api/community/admin/settings", content=huge,
                               headers={"Authorization": "Bearer wrong", "Content-Type": "application/json"})
    assert unauthenticated.status_code == 401
    authenticated = api.post("/api/community/admin/settings", content=huge,
                             headers={"Authorization": f"Bearer {ADMIN}", "Content-Type": "application/json"})
    assert authenticated.status_code == 413


def test_non_ascii_and_deeply_nested_input_never_become_a_500(api: TestClient):
    _open(api)
    for raw in (b"Bearer v1.\xe9.x", b"Bearer v1.abc.\xe9"):  # ti\u00eau \u0111\u1ec1 latin-1 (httpx cho ph\u00e9p g\u1eedi bytes)
        assert api.get("/api/community/round", headers={"Authorization": raw}).status_code == 401
    nested = "[" * 7000
    assert api.post("/api/community/join", content=nested, headers={"Content-Type": "application/json"}).status_code == 400


def test_overflowing_update_norm_is_rejected_cleanly(api: TestClient):
    _open(api, clip_norm=50.0)
    headers = _join(api, _codes(api)[0])
    round_data, parent = _parent_head(api, headers)
    huge = core.as_head(np.full((38, 1024), 1e20, dtype=np.float32), parent[core.BIAS_KEY])
    response = api.post("/api/community/update", headers=headers, json=_payload(round_data, huge))
    assert response.status_code == 422 and response.json()["detail"]["code"] == "non_finite"


def test_closing_the_session_aggregates_when_k_min_is_reached_and_cancels_with_a_count_otherwise(api: TestClient, store: CommunityStore):
    _open(api, k_target=5, k_min=2, clip_norm=50.0)
    a, b = (_join(api, c) for c in _codes(api, count=2))
    round_data, parent = _parent_head(api, a)
    api.post("/api/community/update", headers=a, json=_payload(round_data, _client_head(parent, 0.2, 91)))
    api.post("/api/community/update", headers=b, json=_payload(round_data, _client_head(parent, 0.2, 92)))
    assert _admin(api, "settings", {"open": False, "changes": {}}).status_code == 200
    assert store.current_head()[0] == 1  # đủ k_min: vòng được gộp thay vì bỏ công của người tham gia

    _open(api, k_target=5, k_min=2, clip_norm=50.0)
    [c] = _codes(api)
    lone = _join(api, c)
    again, parent2 = _parent_head(api, lone)
    api.post("/api/community/update", headers=lone, json=_payload(again, _client_head(parent2, 0.2, 93)))
    assert _admin(api, "settings", {"open": False, "changes": {}}).status_code == 200
    latest = api.get("/api/community/history").json()["rounds"][0]
    assert latest["status"] == "cancelled" and latest["clients"] == 1  # ghi số cập nhật đã bị bỏ


def test_policy_changes_apply_to_the_open_round(api: TestClient):
    _open(api, k_target=5, k_min=2)
    headers = _join(api, _codes(api)[0])
    assert api.get("/api/community/round", headers=headers).json()["round"]["k_target"] == 5
    assert _admin(api, "settings", {"changes": {"k_target": 3}}).status_code == 200
    assert api.get("/api/community/round", headers=headers).json()["round"]["k_target"] == 3


def test_failed_joins_are_persisted_in_the_event_trail(api: TestClient):
    _open(api)
    api.post("/api/community/join", json={"code": "AAAAA-AAAAA"})
    events = _admin(api, "state", method="get").json()["events"]
    assert any(e["kind"] == "join_failed" for e in events)


def test_state_is_bound_to_the_artifacts_that_produced_it(tmp_path: Path, clock: Clock):
    from web.services.community_store import AssetsUnavailable

    state_dir = tmp_path / "state"
    first = _synthetic_assets(tmp_path)
    CommunityStore(state_dir, first, SECRET, ADMIN, clock=clock)
    other = _synthetic_assets(tmp_path)
    other.evalset = EvalSet(other.evalset.features + 0.5, other.evalset.labels, other.evalset.sources, "different-eval")
    with pytest.raises(AssetsUnavailable):
        CommunityStore(state_dir, other, SECRET, ADMIN, clock=clock)


def test_eval_set_must_contain_both_sources(tmp_path: Path):
    from web.services.community_store import AssetsUnavailable

    rng = np.random.default_rng(5)
    root = tmp_path / "assets"
    root.mkdir()
    head = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    np.savez(root / "base_head.npz", weight=head[core.WEIGHT_KEY], bias=head[core.BIAS_KEY])
    (root / "feature_model.onnx").write_bytes(b"x" * 10)
    np.savez(root / "eval_features.npz", features=np.zeros((6, 1024), np.float16), labels=np.arange(6, dtype=np.int64),
             sources=np.zeros(6, np.int8))  # chỉ PlantVillage
    feature_sha = core.file_sha256(root / "feature_model.onnx")
    (root / "community_manifest.json").write_text(json.dumps({
        "classes": [f"c{i}" for i in range(38)],
        "feature_model": {"file": "feature_model.onnx", "sha256": feature_sha, "bytes": 10},
        "base_head": {"file": "base_head.npz", "sha256": core.file_sha256(root / "base_head.npz"), "head_sha": core.head_sha(head)},
        "eval_features": {"file": "eval_features.npz", "sha256": core.file_sha256(root / "eval_features.npz"), "count": 6,
                          "feature_model_sha256": feature_sha},
    }), encoding="utf-8")
    with pytest.raises(AssetsUnavailable):
        CommunityAssets.from_directory(root)


def test_short_secret_or_admin_token_disables_the_feature(tmp_path: Path, clock: Clock):
    weak = CommunityStore(tmp_path / "weak", _synthetic_assets(tmp_path), "short", "short-admin", clock=clock)
    weak.admin_configure(True, {})
    with pytest.raises(CommunityError) as joining:
        weak.join("ABCDE-FGHJK")
    assert joining.value.code == "disabled"
    with pytest.raises(CommunityError) as admin:
        weak.check_admin("short-admin")
    assert admin.value.code == "admin_disabled"


def test_rate_limiter_keeps_per_key_windows_when_pruning():
    ticks = [1000.0]
    limiter = community_limiter(lambda: ticks[0])
    for _ in range(5):
        limiter.allow("joinfail:1.2.3.4", 10**6, 900)
    ticks[0] += 100  # nhanh hơn cửa sổ 900 giây của khóa này
    limiter.allow("other", 10, 60)
    ticks[0] += 40
    for index in range(50):
        limiter.allow(f"noise:{index}", 10, 60)
    assert limiter.count("joinfail:1.2.3.4", 900) == 5  # chưa được dọn nhầm theo cửa sổ 60 giây của khóa khác


def community_limiter(clock):
    from web.services.community_store import RateLimiter

    return RateLimiter(clock)


# ───────────────────────── đăng nhập quản trị bằng tài khoản + mật khẩu ─────────────────────────
def _basic(user: str, password: str) -> dict[str, str]:
    import base64 as b64

    return {"Authorization": "Basic " + b64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")}


def test_admin_accepts_username_and_password_via_basic_and_still_bearer(api: TestClient):
    url = "/api/community/admin/state"
    assert api.get(url, headers=_basic("admin", ADMIN)).status_code == 200
    assert api.get(url, headers={"Authorization": f"Bearer {ADMIN}"}).status_code == 200
    # bốn lần thất bại (dưới ngưỡng khóa 5): mỗi lần là 401, không bao giờ 500
    wrong_user = api.get(url, headers=_basic("root", ADMIN))              # sai tài khoản, đúng mật khẩu
    wrong_password = api.get(url, headers=_basic("admin", "sai-mat-khau-sai-mat-khau"))
    broken = api.get(url, headers={"Authorization": "Basic %%%%"})        # base64 hỏng
    empty = api.get(url, headers=_basic("admin", ""))
    assert [r.status_code for r in (wrong_user, wrong_password, broken, empty)] == [401, 401, 401, 401]
    assert wrong_user.json()["detail"]["message"] == wrong_password.json()["detail"]["message"],         "sai tài khoản và sai mật khẩu phải cho cùng một thông báo (không lộ tài khoản có tồn tại)"


def test_admin_username_can_be_changed_by_environment(api: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMMUNITY_ADMIN_USER", "quantri")
    url = "/api/community/admin/state"
    assert api.get(url, headers=_basic("quantri", ADMIN)).status_code == 200
    assert api.get(url, headers=_basic("admin", ADMIN)).status_code == 401


def test_password_of_twelve_characters_is_accepted_and_eleven_is_not(tmp_path: Path, clock: Clock):
    ok = CommunityStore(tmp_path / "s12", _synthetic_assets(tmp_path), SECRET, "a" * 12, clock=clock)
    ok.check_admin("a" * 12)
    short = CommunityStore(tmp_path / "s11", _synthetic_assets(tmp_path), SECRET, "a" * 11, clock=clock)
    with pytest.raises(CommunityError) as error:
        short.check_admin("a" * 11)
    assert error.value.code == "admin_disabled"


def test_failed_basic_logins_are_counted_and_lock_out(api: TestClient):
    url = "/api/community/admin/state"
    codes = [api.get(url, headers=_basic("root", "x" * 20)).status_code for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    assert api.get(url, headers=_basic("admin", ADMIN)).status_code == 429  # đang bị khóa tạm theo IP
