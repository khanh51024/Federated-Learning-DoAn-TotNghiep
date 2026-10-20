"""Phần số học JS của học phân tán cộng đồng phải khớp tham chiếu Python: resize kiểu PIL (từng byte), chuẩn hóa, softmax regression,
mã hóa head. Chạy bằng Node (không cần trình duyệt); bỏ qua nếu không có Node."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.federated import community as core

JS_DIR = Path(__file__).resolve().parents[1] / "web" / "static" / "js"
MATH_URL = (JS_DIR / "core" / "community-math.js").resolve().as_uri()

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="cần Node.js")

DRIVER = f"""
import {{ readFileSync, writeFileSync }} from "node:fs";
import * as M from {json.dumps(MATH_URL)};

const input = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = {{}};
const fromB64 = text => Uint8Array.from(Buffer.from(text, "base64"));

if (input.images) {{
  out.images = input.images.map(item => {{
    const rgba = fromB64(item.rgba);
    M.flattenOnBlack(rgba);
    const rgb = M.resizeRgbaToRgb(rgba, item.width, item.height, 224, 224);
    const tensor = M.rgbToTensor(rgb);
    return {{ rgb: Buffer.from(rgb).toString("base64"), tensor: Array.from(tensor.subarray(0, 600)), tensorTail: Array.from(tensor.subarray(tensor.length - 300)) }};
  }});
}}
if (input.train) {{
  const t = input.train;
  const features = new Float32Array(t.features);
  const result = M.trainSoftmax({{
    features, labels: Int32Array.from(t.labels), weight: new Float32Array(t.weight), bias: new Float32Array(t.bias),
    epochs: t.epochs, lr: t.lr, batch: t.batch, seed: t.seed,
  }});
  out.train = {{ weight: Array.from(result.weight), bias: Array.from(result.bias), loss: result.lossHistory }};
  const ev = M.evaluateHead({{ features, labels: Int32Array.from(t.labels), weight: result.weight, bias: result.bias }});
  out.eval = ev;
  out.order = (() => {{ const rng = M.makeRng(t.seed); return Array.from({{ length: 10 }}, () => rng()); }})();
}}
if (input.codec) {{
  const w = new Float32Array(input.codec.weight), b = new Float32Array(input.codec.bias);
  out.encoded = M.encodeHead(w, b);
  const back = M.decodeHead(input.codec.fromPython);
  out.decodedWeight = Array.from(back.weight.subarray(0, 50));
  out.decodedBias = Array.from(back.bias);
}}
if (input.split) {{
  const s = M.splitHoldout(Int32Array.from(input.split.labels), input.split.fraction, 7, 15);
  out.split = s;
}}
if (input.privatize) {{
  const p = input.privatize;
  const r = M.privatize({{ weight0: new Float32Array(p.w0), bias0: new Float32Array(p.b0), weight: new Float32Array(p.w), bias: new Float32Array(p.b),
    clip: p.clip, noiseMultiplier: p.noise, uniform: M.makeRng(5) }});
  out.privatize = {{ clipped: r.clipped, normBefore: r.normBefore, normAfter: M.deltaNorm(new Float32Array(p.w0), new Float32Array(p.b0), r.weight, r.bias) }};
}}
writeFileSync(process.argv[3], JSON.stringify(out));
"""


def run_node(tmp_path: Path, payload: dict) -> dict:
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER, encoding="utf-8")
    source, target = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    done = subprocess.run(["node", str(driver), str(source), str(target)], capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert done.returncode == 0, done.stderr
    return json.loads(target.read_text(encoding="utf-8"))


def _smooth_image(rng: np.random.Generator, width: int, height: int, alpha: bool) -> Image.Image:
    xs = np.linspace(0, 1, width)[None, :]
    ys = np.linspace(0, 1, height)[:, None]
    base = np.stack([
        255 * (0.5 + 0.5 * np.sin(6 * xs + 2 * ys)),
        255 * (0.5 + 0.5 * np.cos(9 * ys - xs)),
        255 * xs * ys,
    ], axis=-1)
    noise = rng.integers(-40, 40, size=(height, width, 3))
    rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
    if not alpha:
        return Image.fromarray(rgb, "RGB").convert("RGBA")
    a = rng.integers(0, 256, size=(height, width, 1)).astype(np.uint8)
    return Image.fromarray(np.concatenate([rgb, a], axis=-1), "RGBA")


def _pil_reference(rgba_image: Image.Image) -> np.ndarray:
    """Đúng đường của bộ tải ảnh nhóm train: RGBA -> nền đen (paste với mask) rồi Resize bilinear antialias."""
    background = Image.new("RGB", rgba_image.size, (0, 0, 0))
    background.paste(rgba_image, mask=rgba_image.split()[3])
    return np.asarray(background.resize((224, 224), Image.BILINEAR), dtype=np.uint8)


def test_resize_and_normalize_match_pillow_and_torchvision_byte_for_byte(tmp_path):
    rng = np.random.default_rng(11)
    cases = [(640, 480, False), (37, 91, False), (1000, 300, False), (224, 224, False), (300, 200, True), (1200, 900, False)]
    images = [_smooth_image(rng, w, h, a) for w, h, a in cases]
    result = run_node(tmp_path, {"images": [
        {"width": im.width, "height": im.height, "rgba": base64.b64encode(im.tobytes()).decode()} for im in images
    ]})
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for image, produced in zip(images, result["images"]):
        expected = _pil_reference(image)
        got = np.frombuffer(base64.b64decode(produced["rgb"]), dtype=np.uint8).reshape(224, 224, 3)
        assert np.array_equal(got, expected), f"{image.size}: lệch tối đa {int(np.abs(got.astype(int) - expected.astype(int)).max())}"
        torch_like = ((expected.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1).reshape(-1)
        assert np.allclose(produced["tensor"], torch_like[:600], atol=1e-6)
        assert np.allclose(produced["tensorTail"], torch_like[-300:], atol=1e-6)


def _mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def step() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return step


def test_softmax_training_matches_numpy_float64_reference(tmp_path):
    rng = np.random.default_rng(3)
    n, dim, classes = 37, core.FEATURE_DIM, core.NUM_CLASSES
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = rng.integers(0, 6, size=n)
    weight = (rng.normal(size=(classes, dim)) * 0.05).astype(np.float32)
    bias = (rng.normal(size=classes) * 0.01).astype(np.float32)
    cfg = {"epochs": 3, "lr": 0.01, "batch": 16, "seed": 99}
    result = run_node(tmp_path, {"train": {**cfg, "features": features.ravel().tolist(), "labels": labels.tolist(),
                                           "weight": weight.ravel().tolist(), "bias": bias.tolist()}})

    uniform = _mulberry32(cfg["seed"])
    order = list(range(n))
    w, b = weight.astype(np.float64).copy(), bias.astype(np.float64).copy()
    f = features.astype(np.float64)
    for _ in range(cfg["epochs"]):
        for i in range(n - 1, 0, -1):  # Fisher–Yates giống shuffleInPlace
            j = int(uniform() * (i + 1))
            order[i], order[j] = order[j], order[i]
        for start in range(0, n, cfg["batch"]):
            idx = order[start:start + cfg["batch"]]
            z = f[idx] @ w.T + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            p[np.arange(len(idx)), labels[idx]] -= 1.0
            p /= len(idx)
            w -= cfg["lr"] * (p.T @ f[idx])
            b -= cfg["lr"] * p.sum(0)
    got_w = np.array(result["train"]["weight"]).reshape(classes, dim)
    assert np.abs(got_w - w).max() < 5e-6
    assert np.abs(np.array(result["train"]["bias"]) - b).max() < 5e-6
    assert result["train"]["loss"][-1] < result["train"]["loss"][0]
    pred = (f @ got_w.T + np.array(result["train"]["bias"])).argmax(1)
    assert result["eval"]["accuracy"] == pytest.approx(float((pred == labels).mean()))
    reference = [_mulberry32(cfg["seed"])() for _ in range(1)]
    assert result["order"][0] == pytest.approx(reference[0])


def test_head_codec_is_interoperable_with_the_python_wire_format(tmp_path):
    rng = np.random.default_rng(4)
    head = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    other = core.as_head(rng.normal(size=(38, 1024)), rng.normal(size=38))
    result = run_node(tmp_path, {"codec": {
        "weight": head[core.WEIGHT_KEY].ravel().tolist(), "bias": head[core.BIAS_KEY].tolist(),
        "fromPython": core.encode_head(other),
    }})
    assert result["encoded"] == core.encode_head(head)  # JS mã hóa ra đúng từng byte như Python
    back = core.decode_head(result["encoded"])
    assert np.array_equal(back[core.WEIGHT_KEY], head[core.WEIGHT_KEY])
    assert np.array_equal(np.array(result["decodedWeight"], dtype=np.float32), other[core.WEIGHT_KEY].ravel()[:50])
    assert np.array_equal(np.array(result["decodedBias"], dtype=np.float32), other[core.BIAS_KEY])


def test_holdout_split_is_stratified_and_skipped_for_small_sets(tmp_path):
    labels = [0] * 10 + [1] * 10 + [2] * 5 + [3]
    result = run_node(tmp_path, {"split": {"labels": labels, "fraction": 0.2}})["split"]
    held = set(result["holdout"])
    assert held and set(result["train"]).isdisjoint(held) and len(result["train"]) + len(held) == len(labels)
    assert {labels[i] for i in held} >= {0, 1, 2}   # mỗi lớp có ≥ 2 ảnh góp ít nhất 1 ảnh
    assert labels.index(3) in result["train"]        # lớp chỉ có một ảnh luôn ở phần học
    small = run_node(tmp_path, {"split": {"labels": [0, 1, 2, 3, 4], "fraction": 0.2}})["split"]
    assert small["holdout"] == [] and len(small["train"]) == 5


def test_privacy_option_clips_the_delta_norm(tmp_path):
    rng = np.random.default_rng(8)
    w0 = rng.normal(size=38 * 1024).astype(np.float32)
    b0 = np.zeros(38, dtype=np.float32)
    w = w0 + rng.normal(size=w0.shape).astype(np.float32) * 0.05
    result = run_node(tmp_path, {"privatize": {"w0": w0.tolist(), "b0": b0.tolist(), "w": w.tolist(), "b": b0.tolist(),
                                               "clip": 1.0, "noise": 0.0}})["privatize"]
    assert result["clipped"] is True and result["normBefore"] > 1.0
    assert result["normAfter"] == pytest.approx(1.0, rel=1e-3)


def test_every_static_js_file_parses_as_an_es_module():
    """`node --check` coi tệp là CommonJS và bỏ sót lỗi chỉ module mới có (khai báo import trùng): kiểm bằng chế độ module."""
    failures = []
    for path in sorted(JS_DIR.rglob("*.js")):
        done = subprocess.run(["node", "--input-type=module", "--check"], input=path.read_text(encoding="utf-8"),
                              capture_output=True, text=True, encoding="utf-8", timeout=60)
        if done.returncode != 0:
            failures.append(f"{path.relative_to(JS_DIR).as_posix()}: {done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    assert not failures, failures


def test_simulation_port_matches_the_browser_trainer_split_and_rng(tmp_path):
    """`src.federated.community_sim` phải học đúng như người tham gia thật: cùng bộ sinh số, cùng cách tách, cùng SGD."""
    from src.federated import community_sim as sim

    rng = np.random.default_rng(11)
    n, dim, classes = 53, core.FEATURE_DIM, core.NUM_CLASSES
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = rng.integers(0, 8, size=n)
    weight = (rng.normal(size=(classes, dim)) * 0.05).astype(np.float32)
    bias = (rng.normal(size=classes) * 0.01).astype(np.float32)
    cfg = {"epochs": 3, "lr": 0.01, "batch": 16, "seed": 5}
    js = run_node(tmp_path, {
        "train": {**cfg, "features": features.ravel().tolist(), "labels": labels.tolist(),
                  "weight": weight.ravel().tolist(), "bias": bias.tolist()},
        "split": {"labels": labels.tolist(), "fraction": 0.2},
    })
    w, b, loss = sim.train_softmax(features, labels, weight, bias, epochs=cfg["epochs"], lr=cfg["lr"], batch=cfg["batch"], seed=cfg["seed"])
    assert np.abs(w.ravel() - np.array(js["train"]["weight"])).max() < 2e-5
    assert np.abs(b - np.array(js["train"]["bias"])).max() < 2e-5
    assert loss[-1] == pytest.approx(js["train"]["loss"][-1], rel=1e-4)
    stream = sim.make_rng(cfg["seed"])
    assert [stream() for _ in range(3)][0] == pytest.approx(js["order"][0])
    train, holdout = sim.split_holdout(labels.tolist(), 0.2)
    assert train == js["split"]["train"] and holdout == js["split"]["holdout"]
