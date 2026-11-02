import pytest

from scripts.summarize_network_speedtest import summarize_run


def _history():
    return {"audit_rounds": [{"round": 1, "round_fit_elapsed_ms": 2000.0,
            "accepted_clients": 1, "parent_model_hash": "a", "global_model_hash": "b",
            "clients": [{"client_id": 0, "num_examples": 10, "status": "ok",
                         "upload_hash_ok": True, "local_train_time_s": 2.0}]}]}


def test_throughput_distinguishes_shard_size_from_processed_images():
    summary = summarize_run(_history(), {0: "cpu"}, local_epochs=2)
    machine = summary["by_machine"]["cpu"]
    assert machine["train_images_per_s"]["median"] == 5.0
    assert machine["processed_train_images_per_s"]["median"] == 10.0


def test_unknown_epochs_do_not_invent_processed_throughput():
    summary = summarize_run(_history(), {0: "cpu"})
    assert summary["by_machine"]["cpu"]["processed_train_images_per_s"]["median"] is None


def test_recorded_epochs_cannot_be_silently_overridden():
    history = _history() | {"local_epochs": 2}
    with pytest.raises(ValueError):
        summarize_run(history, {0: "cpu"}, local_epochs=1)
    assert summarize_run(history, {0: "cpu"})["local_epochs"] == 2
