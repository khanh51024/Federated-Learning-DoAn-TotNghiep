"""GĐ4 (chạy được trên CPU): đo đánh đổi dung lượng – độ chính xác của (A) nén trọng số khi truyền và (B) lượng tử hóa int8 mô hình suy luận.

Dùng hai mô hình web của gói đang dùng (Centralized, FedAvg), qua đường nạp đã kiểm SHA-256 của `web.demo_evidence`; đánh giá trên một split của
release (mặc định `val`; dùng `--split legacy_diagnostic_test` để chấm tập giữ riêng).

A. Nén trọng số khi truyền. Chỉ tensor số thực từ 2 chiều trở lên (trọng số conv, depthwise, tuyến tính) bị nén; bias, tham số và thống kê BatchNorm,
   bộ đếm giữ nguyên float32/int64 (chiếm phần rất nhỏ dung lượng). Các cách:
     fp32               nguyên bản
     fp16               ép trọng số về float16 rồi về float32
     int8_per_tensor    int8 đối xứng, một hệ số cho cả tensor (trên TRỌNG SỐ TUYỆT ĐỐI)
     int8_per_channel   int8 đối xứng, một hệ số cho mỗi kênh đầu ra (trên trọng số tuyệt đối)
     delta_int8_per_tensor / delta_int8_per_channel
                        như trên nhưng nén ĐỘ LỆCH w − w0 so với trọng số khởi tạo w0 (ImageNet + đầu theo hạt giống, hai bên đều biết w0 nên không
                        phải gửi lại), phía nhận dựng w0 + dequant(độ lệch)
   Đây là phép dựng lại MỘT LẦN từ mô hình đã huấn luyện xong, KHÔNG phải huấn luyện liên kết có nén ở mỗi vòng: nó cho biết sai số do nén, chưa
   cho biết nén ở mỗi vòng ảnh hưởng thế nào lên quá trình hội tụ (cần chạy huấn luyện trên GPU).
B. Lượng tử hóa suy luận: xuất ONNX fp32, lượng tử hóa tĩnh int8 (QDQ, trọng số theo kênh, hiệu chuẩn bằng ảnh split calibration, không dùng ảnh đánh giá),
   đo dung lượng tệp, độ trễ một ảnh một luồng CPU và độ chính xác.

Chạy:
    python scripts/eval_gd4_compression.py
    python scripts/eval_gd4_compression.py --limit 300 --no-onnx      # thử nhanh
Kết quả: artifacts/gd4_compression.json
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

SCHEMES = ("fp32", "fp16", "int8_per_tensor", "int8_per_channel", "delta_int8_per_tensor", "delta_int8_per_channel")
CALIBRATION_IMAGES = 256
LATENCY_IMAGES = 100


def is_quantizable(tensor: Any) -> bool:
    """Trọng số conv/depthwise/tuyến tính: số thực từ 2 chiều trở lên."""
    return tensor.is_floating_point() and tensor.ndim >= 2


def fake_quant_int8(tensor: Any, per_channel: bool) -> tuple[Any, int]:
    """Lượng tử hóa đối xứng int8 rồi giải lượng tử. Trả (tensor sau khi dựng lại, số hệ số tỉ lệ phải gửi kèm)."""
    import torch

    flat = tensor.reshape(tensor.shape[0], -1) if per_channel else tensor.reshape(1, -1)
    scale = (flat.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    quantized = torch.round(flat / scale[:, None]).clamp(-127, 127)
    return (quantized * scale[:, None]).reshape(tensor.shape), int(scale.numel())


def transform_state(state: dict[str, Any], scheme: str, w0: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
    """(state sau khi gửi–nhận, số byte phải gửi). `w0` bắt buộc với cách `delta_*`."""
    if scheme not in SCHEMES:
        raise ValueError(f"cách nén không có: {scheme}")
    delta = scheme.startswith("delta_")
    if delta and w0 is None:
        raise ValueError("cách delta_* cần w0")
    out: dict[str, Any] = {}
    size = 0
    for key, tensor in state.items():
        if not is_quantizable(tensor) or scheme == "fp32":
            out[key] = tensor.clone()
            size += tensor.numel() * tensor.element_size()
            continue
        base = tensor - w0[key] if delta else tensor
        if scheme == "fp16":
            restored, size_add = base.half().float(), 2 * tensor.numel()
        else:
            restored, scales = fake_quant_int8(base, per_channel=scheme.endswith("per_channel"))
            size_add = tensor.numel() + 4 * scales
        out[key] = restored + w0[key] if delta else restored
        size += size_add
    return out, size


def eval_loader(rows: list[dict[str, Any]], dataset_root: Path, batch: int, workers: int) -> Any:
    import torch

    from src.data.robust_transforms import get_eval_transform
    from src.training.data import ManifestDataset

    return torch.utils.data.DataLoader(ManifestDataset(rows, dataset_root, get_eval_transform()), batch_size=batch, shuffle=False, num_workers=workers)


def export_onnx(model: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    example = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(model.eval(), example, str(path), dynamo=False, **kwargs)
    except TypeError:  # torch cũ chưa có tham số `dynamo`
        torch.onnx.export(model.eval(), example, str(path), **kwargs)


class CalibrationReader:
    """Nguồn ảnh hiệu chuẩn cho quantize_static: các lô nhỏ lấy từ split calibration."""

    def __init__(self, batches: list[np.ndarray]):
        self.batches, self.index = batches, 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.batches):
            return None
        self.index += 1
        return {"input": self.batches[self.index - 1]}

    def rewind(self) -> None:
        self.index = 0


ONNX_VARIANTS = ("int8_all", "int8_se_gemm", "int8_se_gemm_pw_late")


def classify_nodes(onnx_path: Path) -> dict[str, list[str]]:
    """Nhóm các nút có trọng số của MobileNetV3-Small đã xuất ONNX: lớp conv đầu, depthwise, pointwise (1x1 không thuộc khối SE), khối SE (hai conv 1x1 sau
    GlobalAveragePool) và Gemm (lớp phân loại)."""
    import onnx

    model = onnx.load(str(onnx_path))
    producer = {out: node for node in model.graph.node for out in node.output}

    def op_of(name: str) -> str | None:
        node = producer.get(name)
        return node.op_type if node is not None else None

    groups: dict[str, list[str]] = {"first": [], "depthwise": [], "pointwise": [], "se": [], "gemm": []}
    for index, node in enumerate([n for n in model.graph.node if n.op_type == "Conv"]):
        attrs = {a.name: a for a in node.attribute}
        source = producer.get(node.input[0])
        after_pool = op_of(node.input[0]) == "GlobalAveragePool"
        after_se_relu = bool(source is not None and source.op_type == "Relu" and op_of(source.input[0]) == "Conv"
                             and op_of(producer[source.input[0]].input[0]) == "GlobalAveragePool")
        if index == 0:
            groups["first"].append(node.name)
        elif after_pool or after_se_relu:
            groups["se"].append(node.name)
        elif (attrs["group"].i if "group" in attrs else 1) > 1:
            groups["depthwise"].append(node.name)
        else:
            groups["pointwise"].append(node.name)
    groups["gemm"] = [n.name for n in model.graph.node if n.op_type == "Gemm"]
    return groups


def variant_nodes(groups: dict[str, list[str]], variant: str) -> list[str] | None:
    """Nút được lượng tử hóa của từng biến thể; None = tất cả (mặc định của ONNX Runtime).

    Cấu hình `int8_se_gemm*` được chọn sau khi đo độ nhạy từng nhóm lớp trên 200 ảnh val của mô hình FedAvg v4b: lượng tử hóa lớp conv đầu hoặc các lớp depthwise
    làm mô hình sụp đổ (còn 5% và 4,5%), pointwise nửa đầu mạng làm tụt thêm, còn khối SE, Gemm và pointwise nửa sau chịu được."""
    if variant == "int8_all":
        return None
    nodes = groups["se"] + groups["gemm"]
    if variant == "int8_se_gemm_pw_late":
        nodes += groups["pointwise"][len(groups["pointwise"]) // 2:]
    return nodes


def quantize_onnx(fp32_path: Path, int8_path: Path, batches: list[np.ndarray], nodes: list[str] | None = None) -> None:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quantize_static(str(fp32_path), str(int8_path), CalibrationReader(batches), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.MinMax, nodes_to_quantize=nodes)


def latency_ms(run: Any, images: list[np.ndarray], repeat: int = 1) -> float:
    for image in images[:5]:
        run(image)
    times = []
    for _ in range(repeat):
        for image in images:
            started = time.perf_counter()
            run(image)
            times.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(times), 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=("val", "legacy_diagnostic_test"))
    parser.add_argument("--limit", type=int, default=0, help="chỉ dùng N ảnh đầu của split (chạy thử; kết quả KHÔNG ghi vào artifacts/)")
    parser.add_argument("--no-onnx", action="store_true", help="bỏ phần B (lượng tử hóa suy luận)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=["centralized", "fedavg"])
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp" / "gd4")
    args = parser.parse_args()

    import torch

    from scripts.train_v4 import build_model
    from src.model.mobilenetv3 import get_parameters
    from src.training.data import load_release
    from src.training.metrics import summarize
    from web.demo_evidence import STORE, _published_model, load_evidence

    torch.set_num_threads(args.threads)
    evidence = load_evidence()
    if not evidence.get("available"):
        print(f"[gd4] gói web không hợp lệ: {evidence.get('reason')}", file=sys.stderr)
        return 1
    release_dir = reg.abs_path(reg.RELEASE_DIR)
    dataset_root = reg.abs_path(reg.DATASET_SOURCES_DIR)
    release = load_release(release_dir, None)
    manifest = json.loads((STORE / "import_manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["seed"])
    rows = release[args.split][: args.limit] if args.limit else release[args.split]
    calibration = release["calibration"]
    rng = np.random.default_rng(0)
    calibration_rows = [calibration[i] for i in sorted(rng.choice(len(calibration), size=min(CALIBRATION_IMAGES, len(calibration)), replace=False))]
    w0 = get_parameters(build_model(seed))
    print(f"[gd4] {len(rows)} ảnh {args.split}; w0 từ hạt giống {seed}; mô hình {args.models}", flush=True)

    infos, models = {}, {}
    for model_id in args.models:
        info, model = _published_model(evidence, model_id, STORE)
        infos[model_id], models[model_id] = info, model.eval()

    # --- dựng các biến thể trọng số (PyTorch) -------------------------------------------------------------
    variants: dict[str, dict[str, Any]] = {}
    for model_id, model in models.items():
        base_state = get_parameters(model)
        for scheme in SCHEMES:
            state, size = transform_state(base_state, scheme, w0)
            clone = copy.deepcopy(model).eval()
            clone.load_state_dict(state)
            variants[f"{model_id}/{scheme}"] = {"model": clone, "bytes": size}
    ratio_base = {m: variants[f"{m}/fp32"]["bytes"] for m in models}

    # --- ONNX fp32 + int8 ------------------------------------------------------------------------------------
    sessions: dict[str, Any] = {}
    onnx_info: dict[str, Any] = {}
    if not args.no_onnx:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        calib_loader = eval_loader(calibration_rows, dataset_root, 8, args.workers)
        calib_batches = [images.numpy() for images, _, _ in calib_loader]
        for model_id, model in models.items():
            fp32_path = args.work_dir / f"{model_id}_fp32.onnx"
            export_onnx(model, fp32_path)
            groups = classify_nodes(fp32_path)
            sessions[f"{model_id}/onnx_fp32"] = ort.InferenceSession(str(fp32_path), options, providers=["CPUExecutionProvider"])
            onnx_info[model_id] = {"fp32_bytes": fp32_path.stat().st_size, "calibration_images": len(calibration_rows), "calibration_batches": len(calib_batches),
                                   "node_groups": {k: len(v) for k, v in groups.items()}, "variants": {}}
            for variant in ONNX_VARIANTS:
                path = args.work_dir / f"{model_id}_{variant}.onnx"
                nodes = variant_nodes(groups, variant)
                quantize_onnx(fp32_path, path, calib_batches, nodes)
                sessions[f"{model_id}/onnx_{variant}"] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                onnx_info[model_id]["variants"][variant] = {"bytes": path.stat().st_size, "quantized_nodes": len(nodes) if nodes is not None else "tất cả"}
            print(f"[gd4] {model_id}: ONNX fp32 {fp32_path.stat().st_size / 1e6:.2f} MB; " + ", ".join(
                f"{v} {onnx_info[model_id]['variants'][v]['bytes'] / 1e6:.2f} MB" for v in ONNX_VARIANTS), flush=True)

    # --- một lượt qua ảnh đánh giá, mọi biến thể -------------------------------------------------------------
    logits = {name: np.zeros((len(rows), 38), dtype=np.float32) for name in [*variants, *sessions]}
    loader = eval_loader(rows, dataset_root, 64, args.workers)
    started = time.time()
    first_batch: np.ndarray | None = None
    with torch.inference_mode():
        for images, _, index in loader:
            idx = index.numpy()
            if first_batch is None:
                first_batch = images.numpy()
            for name, variant in variants.items():
                logits[name][idx] = variant["model"](images).float().numpy()
            for name, session in sessions.items():
                logits[name][idx] = session.run(None, {"input": images.numpy()})[0]
            print(f"[gd4] {idx[-1] + 1}/{len(rows)} ảnh, {time.time() - started:.0f}s", flush=True)

    # --- độ trễ một ảnh, một luồng -------------------------------------------------------------------------------
    latency: dict[str, Any] = {}
    if not args.no_onnx and first_batch is not None:
        import onnxruntime as ort

        single = ort.SessionOptions()
        single.intra_op_num_threads = 1
        singles = [first_batch[i:i + 1] for i in range(min(LATENCY_IMAGES, len(first_batch)))]
        torch.set_num_threads(1)
        for model_id in models:
            torch_model = models[model_id]
            latency[model_id] = {"pytorch_fp32_ms": latency_ms(lambda x: torch_model(torch.from_numpy(x)), singles)}
            for kind, file_name in [("onnx_fp32", f"{model_id}_fp32.onnx"), *[(f"onnx_{v}", f"{model_id}_{v}.onnx") for v in ONNX_VARIANTS]]:
                session = ort.InferenceSession(str(args.work_dir / file_name), single, providers=["CPUExecutionProvider"])
                latency[model_id][f"{kind}_ms"] = latency_ms(lambda x, s=session: s.run(None, {"input": x}), singles)
        torch.set_num_threads(args.threads)

    # --- tổng hợp --------------------------------------------------------------------------------------------------
    results: dict[str, Any] = {}
    for model_id in models:
        reference = logits[f"{model_id}/fp32"].argmax(1)
        entry: dict[str, Any] = {"checkpoint_sha256": infos[model_id]["checkpoint_sha256"], "weights_transport": {}, "onnx": onnx_info.get(model_id, {})}
        for scheme in SCHEMES:
            name = f"{model_id}/{scheme}"
            entry["weights_transport"][scheme] = {
                "payload_bytes": variants[name]["bytes"], "payload_ratio": round(variants[name]["bytes"] / ratio_base[model_id], 4),
                "predictions_changed": int((logits[name].argmax(1) != reference).sum()), **summarize(logits[name], rows)}
        for kind in ("onnx_fp32", *[f"onnx_{v}" for v in ONNX_VARIANTS]):
            if f"{model_id}/{kind}" in logits:
                name = f"{model_id}/{kind}"
                entry["onnx"][kind] = {"predictions_changed_vs_pytorch_fp32": int((logits[name].argmax(1) != reference).sum()), **summarize(logits[name], rows)}
        entry["latency"] = latency.get(model_id, {})
        results[model_id] = entry
        print(f"[gd4] {model_id}: " + ", ".join(f"{s} {entry['weights_transport'][s]['top1']}" for s in SCHEMES))
    report = {
        "evidence": "dựng lại một lần từ mô hình đã huấn luyện; không phải huấn luyện liên kết có nén ở mỗi vòng. Hiệu chuẩn ONNX int8 dùng ảnh split calibration, "
                    "đánh giá trên split riêng.",
        "protocol_id": manifest.get("protocol_id"), "release_manifest_sha256": release["sha256"], "bundle_id": manifest.get("bundle_id"),
        "split": args.split, "count": len(rows), "seed_w0": seed, "schemes": list(SCHEMES),
        "environment": {"cpu_count": __import__("os").cpu_count(), "torch": torch.__version__, "platform": platform.platform(), "threads_accuracy_pass": args.threads,
                        "latency": "một ảnh 224x224, một luồng, trung vị của " + str(LATENCY_IMAGES) + " lần chạy"},
        "models": results,
    }
    out = reg.abs_path(reg.GD4_COMPRESSION_JSON) if not args.limit else args.work_dir / "gd4_limited.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[gd4] ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
