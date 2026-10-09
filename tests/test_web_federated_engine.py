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
