"""Kiểm tra môi trường và một forward pass không cần quét toàn bộ dataset."""

import flwr
import torch
import torchvision

from models import create_model
from utils.config import DATA_DIR


def main() -> None:
    print(f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__} | Flower: {flwr.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Dataset path exists: {DATA_DIR.is_dir()} ({DATA_DIR})")
    model = create_model(pretrained=False)
    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 38)
    print("MobileNetV3-Small forward pass: OK")


if __name__ == "__main__":
    main()

