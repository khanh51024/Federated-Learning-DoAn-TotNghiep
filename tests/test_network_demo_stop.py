"""A Stop request terminates all six processes owned by a localhost demo."""

from __future__ import annotations

import threading
import time

from web.network_demo import NetworkDemoManager


def test_stop_cleans_server_and_five_clients_before_checkpoint():
    manager = NetworkDemoManager()
    manager.prepare()
    failures: list[BaseException] = []

    def run() -> None:
        try:
            manager.run(port=18189, rounds=2, timeout_seconds=120)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 65
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if snapshot["client_processes_running"] == 5:
            break
        if not thread.is_alive():
            break
        time.sleep(0.1)
    else:
        manager.stop()
        thread.join(timeout=10)
        raise AssertionError("Five Flower client processes did not start in time")

    assert manager.snapshot()["client_processes_running"] == 5
    manager.stop()
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert failures and isinstance(failures[0], InterruptedError)
    assert manager.snapshot()["client_processes_running"] == 0
    assert manager.run_dir is not None
    assert not (manager.run_dir / "global.pt").exists()
