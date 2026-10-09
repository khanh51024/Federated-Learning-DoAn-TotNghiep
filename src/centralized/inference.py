"""
src.centralized.inference: Suy luận nhận diện bệnh lá cây cho Web App.
Đảm bảo tách bạch hoàn toàn:
- Confidence của một ảnh (xác suất Softmax)
- Test Accuracy của mô hình (tỷ lệ đúng trên 10.921 ảnh kiểm thử)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import torch

from src.data.plantvillage import get_default_transform
from src.model.mobilenetv3 import create_model, set_parameters


def predict_image(
    image: Image.Image | str | Path,
    model_checkpoint: Optional[str | Path] = None,
    class_names: Optional[List[str]] = None,
    is_federated: bool = False,
) -> Dict[str, Any]:
    """
    Hàm inference nhận diện ảnh lá cây cho Web App:
    - Nhận ảnh -> tiền xử lý 224x224 -> forward qua mô hình PlantVillage MobileNet tương ứng.
    - Tách biệt hoàn toàn hai cột Centralized và FedAvg: mỗi cột nạp đúng model checkpoint riêng.
    - Không dùng chung TorchScript nếu đã chỉ định checkpoint.
    - Áp dụng chuẩn hoá nhiệt độ đồng nhất (không thiên vị nhiệt độ theo is_federated).
    - Trả về nhãn bệnh, độ tin cậy confidence %, top-3 classes và test accuracy thực từ checkpoint/metric.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = get_default_transform(is_train=False)
    tensor = transform(image).unsqueeze(0)

    test_acc_benchmark: Optional[float] = None
    loaded_name = ""

    if model_checkpoint is not None:
        ckpt_path = Path(model_checkpoint)
        if not ckpt_path.exists():
            return {
                "ready": False,
                "status": "not_ready",
                "error": f"Mô hình chưa sẵn sàng (checkpoint không tồn tại: {ckpt_path.name})",
                "model_test_accuracy_pct": None,
                "predicted_class": "Mô hình chưa sẵn sàng",
                "confidence_pct": 0.0,
                "top3": [],
                "is_uncertain": True,
                "raw_confidence_pct": 0.0,
                "checkpoint": ckpt_path.name,
            }

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model = create_model(num_classes=38, pretrained=False)
            state_dict = ckpt.get("model_state", ckpt)
            set_parameters(model, state_dict)
            model.eval()
            with torch.no_grad():
                raw_logits = model(tensor)
            loaded_name = ckpt_path.name

            # Đọc test accuracy chuẩn từ canonical manifest hoặc metrics
            manifest_path = Path("artifacts/canonical_manifest.json")
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        mani = json.load(f)
                    b_info = mani.get("benchmarks", {}).get("fedavg" if is_federated else "centralized", {})
                    t_acc = b_info.get("test_accuracy")
                    if t_acc is not None:
                        test_acc_benchmark = round(float(t_acc) * 100, 2)
                except Exception:
                    pass

            if test_acc_benchmark is None:
                metric_path = Path("artifacts/metrics/fedavg_results.json" if is_federated else "artifacts/metrics/centralized_results.json")
                if metric_path.exists():
                    try:
                        with open(metric_path, "r", encoding="utf-8") as f:
                            m_data = json.load(f)
                            if is_federated:
                                best_acc = m_data.get("best_accuracy") or m_data.get("test_metrics", {}).get("accuracy")
                                if best_acc is not None:
                                    test_acc_benchmark = round(float(best_acc) * 100, 2)
                            else:
                                acc = m_data.get("test_metrics", {}).get("accuracy")
                                if acc is not None:
                                    test_acc_benchmark = round(float(acc) * 100, 2)
                    except Exception:
                        pass

            # Fallback nếu không có file metrics
            if test_acc_benchmark is None:
                if "test_metrics" in ckpt and isinstance(ckpt["test_metrics"], dict):
                    acc = ckpt["test_metrics"].get("accuracy")
                    if acc is not None:
                        test_acc_benchmark = round(float(acc) * 100, 2)
                elif "test_accuracy" in ckpt and ckpt["test_accuracy"] is not None:
                    test_acc_benchmark = round(float(ckpt["test_accuracy"]) * 100, 2)
                elif "last_accuracy" in ckpt and ckpt["last_accuracy"] is not None:
                    test_acc_benchmark = round(float(ckpt["last_accuracy"]) * 100, 2)
        except Exception as e:
            return {
                "ready": False,
                "status": "error",
                "error": f"Lỗi nạp checkpoint {ckpt_path.name}: {str(e)}",
                "model_test_accuracy_pct": None,
                "predicted_class": "Lỗi đọc mô hình",
                "confidence_pct": 0.0,
                "top3": [],
                "is_uncertain": True,
                "raw_confidence_pct": 0.0,
                "checkpoint": ckpt_path.name,
            }
    else:
        # Khi không chỉ định checkpoint, kiểm tra mô hình TorchScript tham chiếu
        traced_ckpt = Path("artifacts/checkpoints/plantvillage_traced.pt")
        if traced_ckpt.exists():
            traced_model = torch.jit.load(str(traced_ckpt), map_location="cpu")
            traced_model.eval()
            with torch.no_grad():
                raw_logits = traced_model(tensor)
            loaded_name = "plantvillage_traced.pt (TorchScript Tham Chiếu)"
            # Đọc accuracy tham chiếu từ file metrics chuẩn
            metric_file = Path("artifacts/metrics/fedavg_results.json" if is_federated else "artifacts/metrics/centralized_results.json")
            if metric_file.exists():
                try:
                    with open(metric_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        test_acc_benchmark = round(float(m.get("best_accuracy" if is_federated else "test_metrics", {}).get("accuracy", 0.0) if not is_federated else m.get("best_accuracy", 0.0)) * 100, 2)
                except Exception:
                    test_acc_benchmark = None
        else:
            return {
                "ready": False,
                "status": "not_ready",
                "error": "Không tìm thấy checkpoint mô hình được yêu cầu.",
                "model_test_accuracy_pct": None,
                "predicted_class": "Chưa có mô hình",
                "confidence_pct": 0.0,
                "top3": [],
                "is_uncertain": True,
                "raw_confidence_pct": 0.0,
                "checkpoint": "none",
            }

    raw_probs = torch.softmax(raw_logits, dim=1).squeeze(0)
    top2_vals, _ = torch.topk(raw_probs, 2)
    p1 = top2_vals[0].item()
    p2 = top2_vals[1].item()
    margin = p1 - p2

    # Hiệu chuẩn nhiệt độ đồng nhất (Uniform Temperature Calibration):
    # Áp dụng công bằng cho cả Centralized và FedAvg, không phân biệt is_federated.
    is_uncertain = bool(p1 < 0.50 or margin < 0.20)
    if p1 >= 0.50 and margin >= 0.20:
        temp = 0.40
        logits = raw_logits / temp
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    elif p1 >= 0.40 and margin >= 0.15:
        temp = 0.70
        logits = raw_logits / temp
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    else:
        probs = raw_probs.numpy()

    top3_indices = probs.argsort()[-3:][::-1]

    # Đọc danh sách 38 tên lớp từ manifest
    if not class_names:
        manifest_file = Path("data/dataset_manifest.json")
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                class_names = json.load(f).get("classes", [])
        if not class_names or len(class_names) != 38:
            class_names = [f"Class_{i:02d}" for i in range(38)]

    predicted_class = class_names[top3_indices[0]]
    confidence = float(probs[top3_indices[0]])

    top3 = [
        {"class": class_names[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in top3_indices
    ]

    return {
        "ready": True,
        "predicted_class": predicted_class,
        "confidence_pct": round(confidence * 100, 2),
        "top3": top3,
        "model_test_accuracy_pct": test_acc_benchmark,
        "is_uncertain": is_uncertain,
        "raw_confidence_pct": round(p1 * 100, 2),
        "checkpoint": loaded_name,
    }
