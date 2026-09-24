"""Model builder adapter cho MobileNetV3-Small tuân thủ protocol GĐ1."""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from stage1_compat.constants import NUM_CLASSES


def create_mobilenetv3_stage1(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Tạo MobileNetV3-Small chuẩn GĐ1: nạp pretrained ImageNet rồi thay lớp classifier cuối."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
