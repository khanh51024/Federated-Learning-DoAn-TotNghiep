"""Các phần evidence mới (test giữ riêng, thống kê, chi tiết, các cách chia, nhiệt độ): mỗi phần fail-closed riêng.

Báo cáo thật được đọc một lần rồi sửa trên BẢN SAO (cache của `_load_json` không bị đụng tới).
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Callable

import pytest

from src import registry as reg
from web.routers.inference import _gated_evidence
from web.services import evidence_service as es


@pytest.fixture(scope="module")
def evidence() -> dict:
    ev = _gated_evidence()
    if not ev.get("available"):
        pytest.skip(f"bundle đối chứng chưa khả dụng: {ev.get('reason')}")
    return ev


def _tamper(monkeypatch, relative: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    real = es._load_json

    def fake(rel: str):
        data = real(rel)
        if rel == relative and data is not None:
            data = copy.deepcopy(data)
            mutate(data)
        return data

    monkeypatch.setattr(es, "_load_json", fake)


def test_new_sections_are_available_with_the_real_reports(evidence):
    result = es.summary(evidence)
    assert result["holdout"]["available"] is True
    assert result["holdout"]["count"] == sum(reg.HOLDOUT_COUNTS.values())
    stats = result["statistics"]
    assert stats["available"] is True
    assert stats["splits"]["val"]["available"] is True and stats["splits"]["test"]["available"] is True
    assert set(es.temperatures(evidence)) == set(es.MODEL_ORDER)
    assert es.details(evidence)["errors"]["available"] is True
    assert es.partitions(evidence)["available"] is True


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("holdout_manifest_sha256", "0" * 64),
    lambda r: r.__setitem__("holdout_count", 11014),
    lambda r: r["holdout_counts_by_source"].__setitem__("plantdoc", 171),
    lambda r: r["checkpoint_sha256"].__setitem__("fedavg", "0" * 64),
    lambda r: r["models"]["local_03"]["by_source"]["plantdoc"].__setitem__("count", 170),
], ids=["manifest", "total", "counts", "checkpoint", "model-count"])
def test_holdout_is_withheld_on_any_identity_mismatch(evidence, monkeypatch, mutate):
    _tamper(monkeypatch, reg.HOLDOUT_EVAL_JSON, mutate)
    result = es.summary(evidence)
    assert result["holdout"]["available"] is False and result["holdout"]["reason"]
    # không có báo cáo test đã kiểm thì số test của phần thống kê cũng bị giữ lại, và ma trận test không xuất hiện
    assert result["statistics"]["splits"]["test"]["available"] is False
    assert "test" not in es.details(evidence)["confusion"]


def test_statistics_val_split_is_withheld_when_correct_differs_from_published(evidence, monkeypatch):
    def mutate(report):
        report["splits"]["val"]["by_source"]["plantdoc"]["models"]["fedavg"]["correct"] += 2  # vượt dung sai 1 ảnh

    _tamper(monkeypatch, reg.STATISTICS_JSON, mutate)
    splits = es.summary(evidence)["statistics"]["splits"]
    assert splits["val"]["available"] is False
    assert splits["test"]["available"] is True


def test_statistics_test_split_is_withheld_when_it_disagrees_with_the_holdout_report(evidence, monkeypatch):
    def mutate(report):
        report["splits"]["test"]["by_source"]["plantvillage"]["models"]["centralized"]["correct"] -= 1

    _tamper(monkeypatch, reg.STATISTICS_JSON, mutate)
    assert es.summary(evidence)["statistics"]["splits"]["test"]["available"] is False


def test_statistics_is_withheld_on_checkpoint_mismatch(evidence, monkeypatch):
    _tamper(monkeypatch, reg.STATISTICS_JSON, lambda r: r["checkpoint_sha256"].__setitem__("local_00", "0" * 64))
    assert es.summary(evidence)["statistics"]["available"] is False


def test_fullframe_section_reports_real_numbers_and_matches_the_artifact(evidence):
    section = es.summary(evidence)["fullframe"]
    assert section["available"] is True and section["crop_policy"] == reg.FULLFRAME_CROP_POLICY
    raw = es._load_json(reg.FULLFRAME_EVAL_JSON)
    for split in ("val", "test"):
        for mode in ("annotator_crop", "full_frame", "center_square", "published_crop"):
            raw_block = raw["splits"][split]["single_class_scenes_only"][mode]
            for model_id in es.MODEL_ORDER:
                assert section["splits"][split][mode][model_id] == raw_block[model_id]["correct"] / raw_block["n"]
    # phát hiện chính: cả ảnh làm FedAvg kém đi so với ảnh cắt, ở cả hai tập
    for split in ("val", "test"):
        assert section["splits"][split]["full_frame"]["fedavg"] < section["splits"][split]["annotator_crop"]["fedavg"]


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("validation_manifest_sha256", "0" * 64),
    lambda r: r["checkpoint_sha256"].__setitem__("fedavg", "0" * 64),
    lambda r: r.__setitem__("crop_policy", "whole_leaf_canonical"),
    lambda r: r["splits"]["test"]["single_class_scenes_only"].pop("full_frame"),
    lambda r: r["splits"]["val"]["single_class_scenes_only"]["full_frame"].__setitem__("n", 1),
], ids=["validation", "checkpoint", "crop-policy", "missing-mode", "count-mismatch"])
def test_fullframe_is_withheld_on_any_mismatch(evidence, monkeypatch, mutate):
    _tamper(monkeypatch, reg.FULLFRAME_EVAL_JSON, mutate)
    result = es.summary(evidence)["fullframe"]
    assert result["available"] is False and result["reason"]
    # các phần khác không bị kéo theo
    assert es.summary(evidence)["holdout"]["available"] is True


def test_error_gallery_is_withheld_on_identity_mismatch(evidence, monkeypatch):
    _tamper(monkeypatch, reg.ERROR_GALLERY_CATALOG_JSON, lambda c: c.__setitem__("validation_manifest_sha256", "f" * 64))
    assert es.details(evidence)["errors"]["available"] is False
    monkeypatch.undo()
    _tamper(monkeypatch, reg.ERROR_GALLERY_CATALOG_JSON, lambda c: c["checkpoint_sha256"].__setitem__("centralized", "0" * 64))
    assert es.details(evidence)["errors"]["available"] is False


def test_error_gallery_drops_unsafe_thumb_names_and_unknown_sources(evidence, monkeypatch):
    bad = ["../../app.py", "a/b.jpg", "a\\b.jpg", ""]

    def mutate(catalog):
        items = catalog["items"]
        for item, thumb in zip(items, bad):
            item["thumb"] = thumb
        items[len(bad)]["source"] = "unknown"

    _tamper(monkeypatch, reg.ERROR_GALLERY_CATALOG_JSON, mutate)
    errors = es.details(evidence)["errors"]
    total = len(es._load_json(reg.ERROR_GALLERY_CATALOG_JSON)["items"])
    assert errors["available"] is True and errors["count"] == total - len(bad) - 1
    for item in errors["items"]:
        assert "/" not in item["thumb"] and "\\" not in item["thumb"] and ".." not in item["thumb"]
        assert item["source"] in es.SOURCES


def test_malformed_reports_become_withheld_not_a_500(evidence, monkeypatch):
    _tamper(monkeypatch, reg.ERROR_GALLERY_CATALOG_JSON, lambda c: c["items"].__setitem__(0, "not a dict"))
    assert es.details(evidence)["available"] is False
    monkeypatch.undo()
    _tamper(monkeypatch, reg.PARTITION_SCHEMES_JSON, lambda r: r["schemes"]["iid"].__setitem__("clients", ["x"]))
    assert es.partitions(evidence)["available"] is False


def test_partitions_are_withheld_when_trained_shards_differ(evidence, monkeypatch):
    def mutate(report):
        client = report["schemes"][reg.PARTITION_SCHEME]["clients"]["client_00"]
        client["per_class"][0] += 1
        client["n_k"] += 1

    _tamper(monkeypatch, reg.PARTITION_SCHEMES_JSON, mutate)
    assert es.partitions(evidence)["available"] is False


def test_temperatures_are_empty_when_the_reeval_report_mismatches(evidence, monkeypatch):
    _tamper(monkeypatch, reg.LOCAL_REEVAL_JSON, lambda r: r["checkpoint_sha256"].__setitem__("fedavg", "0" * 64))
    assert es.temperatures(evidence) == {}


def test_gradcam_and_predict_can_run_concurrently_on_the_cached_model(evidence):
    """Grad-CAM không được sửa trạng thái chung của mô hình đã cache (trước đây hook làm predict song song lỗi 500)."""
    import numpy as np
    from PIL import Image

    from web import demo_evidence
    from web.services import explain_service

    image = Image.new("RGB", (256, 256), (60, 140, 60))
    _, model = demo_evidence._published_model(evidence, "fedavg", demo_evidence.STORE)
    tensor = demo_evidence._input_tensor(image)
    reference = explain_service.gradcam(model, tensor, 5)
    errors: list[BaseException] = []
    cams: list[np.ndarray] = []
    stop = threading.Event()

    def predict_loop():
        try:
            while not stop.is_set():
                demo_evidence.predict_published(image, "fedavg")
        except BaseException as exc:  # noqa: BLE001 — ghi lại để assert ở luồng chính
            errors.append(exc)

    worker = threading.Thread(target=predict_loop)
    worker.start()
    try:
        for _ in range(8):
            cams.append(explain_service.gradcam(model, tensor, 5))
    finally:
        stop.set()
        worker.join(timeout=60)
    assert not errors, errors
    for cam in cams:
        assert np.allclose(cam, reference, atol=1e-5)
    assert all(not p.requires_grad for p in model.parameters())


def test_gradcam_forward_matches_the_model_forward(evidence):
    """Grad-CAM tự dựng forward (features → avgpool → classifier): phải ra đúng logits của model(x)."""
    import torch
    from PIL import Image

    from web import demo_evidence

    _, model = demo_evidence._published_model(evidence, "centralized", demo_evidence.STORE)
    x = demo_evidence._input_tensor(Image.new("RGB", (240, 200), (90, 110, 30)))
    with torch.inference_mode():
        direct = model(x)
        manual = model.classifier(torch.flatten(model.avgpool(model.features(x)), 1))
    assert torch.allclose(direct, manual, atol=1e-6)
