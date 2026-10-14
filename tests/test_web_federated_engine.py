from __future__ import annotations

from fastapi import BackgroundTasks, HTTPException

from web.app import (
    CURRENT_TASK, CentralizedTrainRequest, FederatedTrainRequest,
    trigger_centralized_train, trigger_federated_train,
)


def _reset_task():
    CURRENT_TASK["status"] = "idle"
    CURRENT_TASK["cancel_requested"] = False


def test_web_rejects_unknown_federated_engine():
    _reset_task()
    req = FederatedTrainRequest(engine="not-an-engine")
    try:
        trigger_federated_train(BackgroundTasks(), req)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "simulation" in str(exc.detail)
    else:
        raise AssertionError("Unknown engine must be rejected")


def test_web_network_engine_schedules_real_server_task_only():
    _reset_task()
    bg = BackgroundTasks()
    req = FederatedTrainRequest(
        engine="network",
        rounds=2,
        num_clients=5,
        server_address="127.0.0.1:18089",
    )
    response = trigger_federated_train(bg, req)

    assert response["engine"] == "network"
    assert len(bg.tasks) == 1
    assert bg.tasks[0].func.__name__ == "_run_federated_network_bg"
    assert "5 client" in response["message"]


def test_web_network_engine_validates_host_port_format():
    _reset_task()
    req = FederatedTrainRequest(engine="network", server_address="localhost")
    try:
        trigger_federated_train(BackgroundTasks(), req)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "HOST:PORT" in str(exc.detail)
    else:
        raise AssertionError("Invalid server_address must be rejected")


def test_network_demo_rejects_real_mode_and_wrong_client_count():
    for req in (
        FederatedTrainRequest(engine="network", smoke=False),
        FederatedTrainRequest(engine="network", num_clients=4),
        FederatedTrainRequest(engine="network", alpha=1.0),
        FederatedTrainRequest(engine="network", rounds=3),
        FederatedTrainRequest(engine="network", local_epochs=2),
    ):
        _reset_task()
        try:
            trigger_federated_train(BackgroundTasks(), req)
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Network demo must fail closed on mismatched protocol")


def test_simulation_does_not_substitute_partition_or_real_mode():
    for req in (
        FederatedTrainRequest(engine="simulation", alpha=0.5),
        FederatedTrainRequest(engine="simulation", dataset_root="D:/PlantVillage"),
        FederatedTrainRequest(engine="simulation", smoke=False),
    ):
        _reset_task()
        try:
            trigger_federated_train(BackgroundTasks(), req)
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Simulation must fail closed on an unverified protocol")


def test_centralized_web_real_mode_requires_mixed_release():
    _reset_task()
    try:
        trigger_centralized_train(BackgroundTasks(), CentralizedTrainRequest(smoke=False))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "mixed pv_pd_v3" in str(exc.detail)
    else:
        raise AssertionError("Real-data mode must stay unavailable until release verification")


def test_stop_sent_right_after_start_is_accepted_and_survives_the_background_task(monkeypatch):
    """Lỗi nghiệm thu: Dừng bấm sát sau Bắt đầu bị báo "không có tác vụ" rồi tác vụ vẫn chạy. Endpoint phải đặt
    status=running đồng bộ và BackgroundTask không được xóa cancel_requested."""
    from web.routers import training

    _reset_task()
    monkeypatch.setattr(training, "run_centralized_training", lambda **kwargs: {"test_metrics": {"accuracy": 0.5}})
    background = BackgroundTasks()
    trigger_centralized_train(background, CentralizedTrainRequest(epochs=2, smoke=True))
    assert CURRENT_TASK["status"] == "running" and CURRENT_TASK["type"] == "centralized"
    assert CURRENT_TASK["cancel_requested"] is False

    reply = training.stop_training()
    assert reply["stopped"] is True and CURRENT_TASK["cancel_requested"] is True

    for task in background.tasks:  # Starlette chạy các tác vụ này SAU khi đã trả lời
        task.func(*task.args, **task.kwargs)
    assert CURRENT_TASK["status"] == "cancelled"


def test_simulation_marks_running_before_the_background_task_and_stop_is_idle_safe(monkeypatch):
    import pytest

    from src import registry as reg
    from web.routers import training

    if not (reg.abs_path(reg.PARTITIONS_DIR) / "label01").exists():
        pytest.skip("thiếu thư mục phân hoạch label01")
    _reset_task()
    assert training.stop_training()["stopped"] is False  # không có tác vụ: báo thật, không giả vờ đã dừng
    monkeypatch.setattr(training, "run_federated_simulation", lambda **kwargs: {"best_accuracy": 0.5})
    background = BackgroundTasks()
    trigger_federated_train(background, FederatedTrainRequest(engine="simulation", rounds=1, local_epochs=1, alpha=0.1))
    assert CURRENT_TASK["status"] == "running" and CURRENT_TASK["type"] == "federated"
    assert training.stop_training()["stopped"] is True
    for task in background.tasks:
        task.func(*task.args, **task.kwargs)
    assert CURRENT_TASK["status"] == "cancelled"
