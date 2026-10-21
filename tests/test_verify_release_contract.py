"""scripts/verify_release_contract.py: gọi đúng các bộ kiểm của bộ chạy và báo từng kết quả (dùng module giả, không cần nhánh git)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.verify_release_contract import NUM_CLIENTS, SCHEMES, run_checks


def _fake_modules(calls: list[str], fail: str | None = None) -> dict[str, dict[str, Any]]:
    def make(name: str, result: Any):
        def fn(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            if name == fail:
                raise ValueError(f"{name} bị từ chối")
            return result
        return fn

    selection: dict[str, Any] = {"RELEASE_NAME": "pv_pd_v3"}

    def inspect_dataset(root: Path, scheme: str | None, clients: int, sha: str) -> dict[str, Any]:
        calls.append(f"inspect:{selection['RELEASE_NAME']}:{scheme}:{clients}")
        return {"train_count": 3, "client_shards": {"client_00": {"count": 3}}}

    selection["inspect_dataset"] = inspect_dataset
    return {
        "integrity": {"verify_release_integrity": make("release", {"status": "PASS"}), "verify_manifest_records_contract": make("records", {"status": "PASS"}),
                      "verify_image_dataset_integrity": make("images", {"status": "PASS"})},
        "partitions": {"verify_partition_directory": make("directory", {"status": "PASS"}), "verify_partition_audit": make("audit", {"status": "PASS"})},
        "full_gate": {"verify_full_data_gate": make("gate", {"status": "PASS"})},
        "data_selection": selection,
    }


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "mixed" / "pv_pd_v4"
    release.mkdir(parents=True)
    (release / "release_manifest.json").write_text("{}", encoding="utf-8")
    (release / "quality_audit_v4.json").write_text("{}", encoding="utf-8")
    return release


def test_every_check_runs_and_data_selection_sees_the_release_name_of_the_release(tmp_path):
    calls: list[str] = []
    modules = _fake_modules(calls)
    results = run_checks(_release(tmp_path), modules, with_images=False, dataset_root=tmp_path)
    assert all(ok for _, ok, _ in results) and len(results) == 2 + 1 + 2 * len(SCHEMES) + 1 + 1 + len(SCHEMES)
    assert f"inspect:pv_pd_v4:None:{NUM_CLIENTS}" in calls and f"inspect:pv_pd_v4:iid:{NUM_CLIENTS}" in calls
    assert modules["data_selection"]["RELEASE_NAME"] == "pv_pd_v3"        # tên gốc được trả lại sau khi kiểm
    assert "images" not in calls


def test_a_failing_check_is_reported_with_its_message_and_the_others_still_run(tmp_path):
    calls: list[str] = []
    results = run_checks(_release(tmp_path), _fake_modules(calls, fail="gate"), with_images=True, dataset_root=tmp_path)
    failed = [(name, detail) for name, ok, detail in results if not ok]
    assert [name for name, _ in failed] == ["verify_full_data_gate"] and "bị từ chối" in failed[0][1]
    assert "images" in calls and "audit" in calls
