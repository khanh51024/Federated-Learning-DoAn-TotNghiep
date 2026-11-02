"""Đóng gói mô hình int8 (scripts/package_quantized_model.py): số đo chép từ kết quả đã đo, tệp lạ bị từ chối, máy chủ đo chỉ phục vụ tệp đã khai báo."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts import package_quantized_model as pq


@pytest.fixture()
def gd4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    variant = {"n": 10, "top1": 0.93, "macro_f1": 0.92, "plantvillage_top1": 0.96, "field_top1": 0.8, "predictions_changed_vs_pytorch_fp32": 2}
    report = {"bundle_id": "b1", "release_manifest_sha256": "rel", "split": "test", "count": 10, "environment": {"cpu_count": 1},
              "models": {name: {"checkpoint_sha256": f"ck-{name}", "latency": {"onnx_fp32": 6.0}, "onnx": {"fp32_bytes": 1000, "onnx_fp32": {**variant, "top1": 0.94}, f"onnx_{pq.VARIANT}": variant,
                                                                          "variants": {pq.VARIANT: {"bytes": 340}}}} for name in ("centralized", "fedavg")}}
    path = tmp_path / "gd4.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(pq, "GD4_JSON", path)
    work = tmp_path / "work"
    work.mkdir()
    for name in ("centralized", "fedavg"):
        (work / f"{name}_{pq.VARIANT}.onnx").write_bytes(b"x" * 340)
    return work


def test_package_copies_file_and_takes_numbers_from_the_measurement(tmp_path: Path, gd4: Path):
    out = tmp_path / "out"
    manifest = pq.package(gd4, out, pq.VARIANT, ("centralized", "fedavg"))
    block = manifest["models"]["fedavg"]
    assert (out / block["file"]).read_bytes() == b"x" * 340
    assert block["sha256"] == pq.sha256(out / block["file"])
    assert block["size_ratio"] == pytest.approx(0.34)
    assert block["accuracy"]["top1"] == 0.93 and block["accuracy_fp32"]["top1"] == 0.94
    assert block["source_checkpoint_sha256"] == "ck-fedavg"
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_package_refuses_a_file_that_is_not_the_measured_one(tmp_path: Path, gd4: Path):
    (gd4 / f"fedavg_{pq.VARIANT}.onnx").write_bytes(b"x" * 999)
    with pytest.raises(SystemExit, match="không phải bản đã đo"):
        pq.package(gd4, tmp_path / "out", pq.VARIANT, ("centralized", "fedavg"))
    (gd4 / f"centralized_{pq.VARIANT}.onnx").unlink()
    with pytest.raises(SystemExit, match="thiếu"):
        pq.package(gd4, tmp_path / "out", pq.VARIANT, ("centralized",))


def test_server_serves_only_declared_files_and_collects_results(tmp_path: Path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"abc")
    secret = tmp_path / "secret.txt"
    secret.write_text("x", encoding="utf-8")
    server = pq.build_server({"m": model}, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert "/model/m.onnx" in page and "MODELS" not in page and "REPEATS_VALUE" not in page
        assert urllib.request.urlopen(base + "/model/m.onnx").read() == b"abc"
        assert urllib.request.urlopen(base + "/ort/ort.wasm.min.mjs").headers["Content-Type"] == "text/javascript"
        for path in ("/secret.txt", "/../secret.txt", "/model/../secret.txt", "/ort/../../scripts/train_v4.py"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + path)
            assert caught.value.code == 404
        request = urllib.request.Request(base + "/result", data=json.dumps({"rows": [{"name": "m", "median_ms": 7.5}]}).encode(), method="POST")
        urllib.request.urlopen(request).read()
        assert server.results == [{"rows": [{"name": "m", "median_ms": 7.5}]}]
    finally:
        server.shutdown()
    with pytest.raises(SystemExit, match="thiếu tệp mô hình"):
        pq.build_server({"m": tmp_path / "khong_co.onnx"}, "127.0.0.1", 0)
