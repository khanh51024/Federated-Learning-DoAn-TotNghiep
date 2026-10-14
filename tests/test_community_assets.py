"""Artifact thật của học liên kết cộng đồng (artifacts/community_fl/): SHA khớp manifest, head khởi điểm tái hiện đúng số đã công bố
trên tập giám sát, ONNX khớp PyTorch. Bỏ qua nếu máy chưa sinh artifact (chạy hai script trong scripts/ trước)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src import registry as reg
from src.federated import community as core
from web.services.community_store import AssetsUnavailable, CommunityAssets

MANIFEST = reg.abs_path(reg.COMMUNITY_MANIFEST_JSON)

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file() or "eval_features" not in json.loads(MANIFEST.read_text(encoding="utf-8")),
    reason="chưa sinh artifacts/community_fl (scripts/export_community_feature_model.py + build_community_eval_features.py)",
)


@pytest.fixture(scope="module")
def assets() -> CommunityAssets:
    try:
        return CommunityAssets.from_directory()
    except AssetsUnavailable as exc:  # pragma: no cover - báo rõ lý do thay vì lỗi khó đọc
        pytest.fail(str(exc))


def test_manifest_hashes_and_shapes_are_consistent(assets: CommunityAssets):
    manifest = assets.manifest
    assert len(assets.classes) == 38 and len(set(assets.classes)) == 38
    assert manifest["feature_model"]["dim"] == core.FEATURE_DIM and manifest["feature_model"]["opset"] == 17
    assert manifest["base_model"]["model_id"] == reg.COMMUNITY_MODEL_ID
    assert manifest["eval_features"]["feature_model_sha256"] == manifest["feature_model"]["sha256"]
    assert assets.base_head[core.WEIGHT_KEY].shape == (38, 1024) and core.HEAD_PARAMS == 38_950
    assert manifest["parity"]["checked"] and manifest["parity"]["min_cosine"] >= 0.999
    assert manifest["parity"]["argmax_agree"] == manifest["parity"]["images"]


def test_base_head_reproduces_the_published_validation_numbers(assets: CommunityAssets):
    """Tập đặc trưng giám sát (float16) + head khởi điểm phải cho ra đúng số của bundle (sai tối đa 1 ảnh mỗi nguồn)."""
    measured = core.evaluate_head(assets.base_head, assets.evalset)
    reference = assets.manifest["base_eval"]["published_reference"]
    for source in ("plantvillage", "plantdoc"):
        count = measured[f"{source}_count"]
        assert count == reference[source]["count"]
        assert abs(measured[f"{source}_accuracy"] - reference[source]["top1_accuracy"]) * count <= 1.0 + 1e-6
    stored = assets.manifest["base_eval"]
    assert measured["accuracy"] == pytest.approx(stored["accuracy"])


def test_onnx_features_match_the_pytorch_bundle_model_on_gallery_images(assets: CommunityAssets):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    from PIL import Image, ImageOps

    from web.demo_evidence import _input_tensor
    from web.services.community_inference import features_of

    session = ort.InferenceSession(str(assets.feature_model_path), providers=["CPUExecutionProvider"])
    images = sorted(reg.abs_path(reg.DEMO_GALLERY_DIR).glob("*.jpg"))[:6]
    assert images, "thiếu ảnh gallery"
    for path in images:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        reference, classes, _info = features_of(image)
        produced = session.run(None, {"input": _input_tensor(image).numpy()})[0][0]
        cosine = float(reference @ produced / (np.linalg.norm(reference) * np.linalg.norm(produced)))
        assert cosine > 0.9999, (path.name, cosine)
        assert classes == assets.classes
