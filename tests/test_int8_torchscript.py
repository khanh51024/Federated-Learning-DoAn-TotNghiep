"""Mô hình int8 (TorchScript, quantize_dynamic chỉ lớp Linear) trong gói web: nạp, kiểm nguồn gốc, từ chối khi sai.

Xem docs/TICH_HOP_INT8_PC_V3.md. Test dựng mô hình MobileNetV3-Small trọng số ngẫu nhiên nên không cần tệp của gói.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from torch import nn

from src import bundle
from web import demo_evidence as de

SHA_A = "a" * 64
SHA_B = "b" * 64


def _int8_torchscript_bytes(num_classes: int = 38) -> bytes:
    from torchvision.models import mobilenet_v3_small

    de.select_quantized_engine()
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    q = torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8).eval()
    with torch.inference_mode():
        ts = torch.jit.freeze(torch.jit.trace(q, (torch.zeros(1, 3, 224, 224),)).eval())
    buf = io.BytesIO()
    torch.jit.save(ts, buf)
    return buf.getvalue()


def _specs(**int8_overrides):
    manifest = {"models": [
        {"model_id": "fedavg", "dir": "fedavg", "display_name": "FedAvg"},
        {"model_id": "fedavg_int8", "dir": "fedavg_int8", "file": "model_int8_dynamic_torchscript.pt", "display_name": "FedAvg int8",
         "format": "torchscript_int8", "fp32_of": "fedavg", "fp32_checkpoint_sha256": SHA_A, **int8_overrides},
    ]}
    return bundle.model_specs(manifest)


def test_old_manifests_default_to_state_dict_and_keep_their_keys():
    specs = bundle.model_specs({})
    assert specs and all(s["format"] == bundle.FORMAT_STATE_DICT for s in specs)
    assert all("fp32_of" not in s for s in specs)


def test_int8_spec_needs_an_fp32_parent_in_the_same_bundle():
    de._check_int8_specs(_specs())
    with pytest.raises(ValueError, match="fp32_of"):
        de._check_int8_specs(_specs(fp32_of="missing"))
    with pytest.raises(ValueError, match="fp32_of"):
        de._check_int8_specs(_specs(fp32_of="fedavg_int8"))
    with pytest.raises(ValueError, match="fp32_checkpoint_sha256"):
        de._check_int8_specs(_specs(fp32_checkpoint_sha256="not-a-sha"))
    with pytest.raises(ValueError, match="unknown model format"):
        de._check_int8_specs(_specs(format="onnx_int8"))


def test_int8_is_bound_to_the_hash_of_its_fp32_checkpoint():
    specs = _specs()
    manifests = {"fedavg": {"files": {"checkpoint_best.pt": {"sha256": SHA_A}}},
                 "fedavg_int8": {"files": {"model_int8_dynamic_torchscript.pt": {"sha256": SHA_B}}}}
    de._check_int8_parents(specs, manifests)
    manifests["fedavg"]["files"]["checkpoint_best.pt"]["sha256"] = SHA_B
    with pytest.raises(ValueError, match="differs from the fp32 checkpoint"):
        de._check_int8_parents(specs, manifests)


def test_int8_torchscript_loads_from_hashed_bytes_and_gives_38_logits():
    model = de._build_torchscript_int8(io.BytesIO(_int8_torchscript_bytes()), 38)
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 38)


def test_int8_with_the_wrong_number_of_classes_is_refused():
    with pytest.raises(ValueError, match="1 x 38"):
        de._build_torchscript_int8(io.BytesIO(_int8_torchscript_bytes(num_classes=10)), 38)


def test_build_model_dispatches_on_format_and_grad_cam_is_refused_for_int8():
    evidence = {"classes": [f"c{i}" for i in range(38)],
                "models": [{"model_id": "fedavg_int8", "format": "torchscript_int8"}, {"model_id": "fedavg"}]}
    assert de.is_int8(evidence, "fedavg_int8") and not de.is_int8(evidence, "fedavg")
    model = de._build_model(io.BytesIO(_int8_torchscript_bytes()), "fedavg_int8", evidence)
    assert isinstance(model, torch.jit.ScriptModule)


def _add_int8_to_bundle_copy(tmp_path: Path) -> Path:
    """Bản sao gói v4b + mục int8 của FedAvg, dựng đúng các bước của docs/TICH_HOP_INT8_PC_V3.md mục 3."""
    import csv
    import hashlib
    import json
    import shutil

    from torchvision.models import mobilenet_v3_small

    src = Path(__file__).resolve().parents[1] / "artifacts" / "model_bundles" / "v4b_s42"
    if not (src / "import_manifest.json").is_file():
        pytest.skip("v4b bundle not present")
    root = tmp_path / "bundle"
    shutil.copytree(src, root)
    meta = json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
    parent = next(m for m in meta["models"] if m["model_id"] == "fedavg")
    parent_sha = hashlib.sha256((root / parent["dir"] / parent["file"]).read_bytes()).hexdigest()

    # 1. tệp int8 dựng từ CHÍNH checkpoint fp32 gốc
    de.select_quantized_engine()
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 38)
    state = torch.load(root / parent["dir"] / parent["file"], map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state, strict=True)
    q = torch.ao.quantization.quantize_dynamic(model.eval(), {nn.Linear}, dtype=torch.qint8).eval()
    with torch.inference_mode():
        ts = torch.jit.freeze(torch.jit.trace(q, (torch.zeros(1, 3, 224, 224),)).eval())
    (root / "fedavg_int8").mkdir()
    int8_file = root / "fedavg_int8" / "model_int8_dynamic_torchscript.pt"
    torch.jit.save(ts, str(int8_file))
    int8_sha = hashlib.sha256(int8_file.read_bytes()).hexdigest()

    # 2. thêm hàng int8 vào validation_comparison.csv (số ở đây chép từ FedAvg chỉ để test; gói thật dùng số đo của tệp int8)
    csv_path = root / parent["dir"] / "validation_comparison.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = list(rows[0].keys())
    rows += [{**r, "model": "FedAvg v4b int8", "checkpoint_sha256": int8_sha} for r in rows if r["model"] == parent["display_name"]]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_bytes = csv_path.read_bytes()
    csv_entry = {"bytes": len(csv_bytes), "sha256": hashlib.sha256(csv_bytes).hexdigest()}

    # 3. mọi thư mục mang cùng một validation_comparison.csv: chép lại và cập nhật artifact_manifest của từng thư mục
    dirs = list(dict.fromkeys(m["dir"] for m in meta["models"]))
    for d in dirs:
        (root / d / "validation_comparison.csv").write_bytes(csv_bytes)
        am_path = root / d / "artifact_manifest.json"
        am = json.loads(am_path.read_text(encoding="utf-8"))
        am["files"]["validation_comparison.csv"] = csv_entry
        am_path.write_text(json.dumps(am, indent=1), encoding="utf-8")
    (root / "fedavg_int8" / "validation_comparison.csv").write_bytes(csv_bytes)
    branch = "runs/v4b/fedavg_s42_int8_dynamic"
    am = json.loads((root / parent["dir"] / "artifact_manifest.json").read_text(encoding="utf-8"))
    am.update(branch=branch, files={
        "model_int8_dynamic_torchscript.pt": {"bytes": int8_file.stat().st_size, "sha256": int8_sha},
        "validation_comparison.csv": csv_entry})
    (root / "fedavg_int8" / "artifact_manifest.json").write_text(json.dumps(am, indent=1), encoding="utf-8")

    # 4. mục int8 trong import_manifest.json
    meta["models"].append({"model_id": "fedavg_int8", "dir": "fedavg_int8", "file": "model_int8_dynamic_torchscript.pt",
                           "display_name": "FedAvg v4b int8", "phase": parent.get("phase", ""), "method": "FedAvg int8 (dynamic, Linear)",
                           "registry_id": "web_fedavg_int8", "format": "torchscript_int8", "fp32_of": "fedavg",
                           "fp32_checkpoint_sha256": parent_sha, "quantization": "dynamic_linear_qint8"})
    meta["sources"]["fedavg_int8"] = {"branch": branch, "commit": meta["sources"][parent["dir"]]["commit"]}
    (root / "import_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return root


def test_int8_end_to_end_in_a_bundle_copy(tmp_path: Path):
    from PIL import Image

    root = _add_int8_to_bundle_copy(tmp_path)
    evidence = de.load_evidence(root)
    assert evidence["available"], evidence.get("reason")
    int8 = next(m for m in evidence["models"] if m["model_id"] == "fedavg_int8")
    assert int8["format"] == "torchscript_int8" and int8["fp32_of"] == "fedavg"
    assert "format" not in next(m for m in evidence["models"] if m["model_id"] == "fedavg")
    image = Image.new("RGB", (60, 50), (30, 120, 40))
    out = de.predict_published(image, "fedavg_int8", root)
    ref = de.predict_published(image, "fedavg", root)
    assert out["model_format"] == "torchscript_int8" and "model_format" not in ref
    assert len(out["top3"]) == 3 and out["predicted_class"] in evidence["classes"]
    with pytest.raises(ValueError, match="Grad-CAM"):
        de.explain_published(image, "fedavg_int8", evidence["classes"][0], root)

    # buộc nguồn gốc: sửa fp32_checkpoint_sha256 là gói bị từ chối (fail-closed)
    import json
    meta = json.loads((root / "import_manifest.json").read_text(encoding="utf-8"))
    meta["models"][-1]["fp32_checkpoint_sha256"] = "0" * 64
    (root / "import_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = de.load_evidence(root)
    assert bad["available"] is False and "fp32_checkpoint_sha256" in bad["reason"]


def test_quantized_engine_is_x86_or_fbgemm_or_a_clear_error():
    assert de.select_quantized_engine() in de.INT8_ENGINES
    with pytest.raises(ValueError, match="x86_64"):
        de.select_quantized_engine(supported=["qnnpack", "none"])
