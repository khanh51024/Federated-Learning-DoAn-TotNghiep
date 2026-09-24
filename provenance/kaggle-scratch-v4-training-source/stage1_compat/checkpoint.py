"""Atomic checkpointing cho profile stage1_compat.

Đảm bảo ghi nguyên tử qua file tạm (.tmp) và kiểm tra identity nghiêm ngặt khi resume.
Lưu đầy đủ global weights, best weights, history, RNG state và checkpoint SHA256.
"""

import hashlib
import io
from stage1_compat.integrity import atomic_bytes, file_hash, canonical
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stage1_compat.identity import JobIdentity, verify_job_identity


class CheckpointError(Exception):
    """Lỗi liên quan đến checkpoint hoặc vi phạm tính toàn vẹn / identity."""
    pass


def get_rng_states() -> dict[str, Any]:
    """Chụp trạng thái RNG của Python, NumPy, PyTorch CPU và CUDA (nếu có)."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Khôi phục trạng thái RNG từ checkpoint."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in states:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def compute_state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Tính SHA256 hash của state_dict mô hình."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(canonical([key, str(tensor.dtype), list(tensor.shape)]))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def payload_hash(value):
    """Hash full tensor/RNG/history payload, including shapes and dtypes."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(b"tensor"); h.update(compute_state_dict_sha256({"value": x}).encode())
        elif isinstance(x, np.ndarray):
            h.update(canonical(["numpy", str(x.dtype), list(x.shape)])); h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x):
                visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(canonical([type(x).__name__, len(x)]))
            for item in x:
                visit(item)
        else:
            h.update(canonical(x))
    visit(value)
    return h.hexdigest()


def atomic_torch_save(path, payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    atomic_bytes(path, stream.getvalue())


def save_atomic_checkpoint(
    checkpoint_path: Path | str,
    server_round: int,
    global_model_state: dict[str, torch.Tensor],
    best_model_state: dict[str, torch.Tensor],
    best_round: int,
    best_validation_accuracy: float,
    history: list[dict[str, Any]],
    identity: JobIdentity,
    round_timings: dict | None = None,
) -> Path:
    """Ghi checkpoint nguyên tử (ghi vào file .tmp rồi atomic rename/replace)."""
    target = Path(checkpoint_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)


    state_sha256 = compute_state_dict_sha256(global_model_state)
    rng_states = get_rng_states()

    payload = {
        "server_round": server_round,
        "round_timings": round_timings or {},
        "global_model_state": global_model_state,
        "best_model_state": best_model_state,
        "best_round": best_round,
        "best_validation_accuracy": best_validation_accuracy,
        "history": history,
        "rng_states": rng_states,
        "identity": identity.to_dict(),
        "state_sha256": state_sha256,
    }

    payload["payload_sha256"] = payload_hash(payload)
    atomic_torch_save(target, payload)

    return target


def load_checkpoint(
    checkpoint_path: Path | str,
    expected_identity: JobIdentity | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Nạp checkpoint và xác minh identity cũng như tính toàn vẹn SHA256."""
    target = Path(checkpoint_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {target}")

    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"File checkpoint bị hỏng hoặc không thể nạp: {e}") from e

    # Kiểm tra các trường bắt buộc
    required_fields = [
        "server_round", "global_model_state", "best_model_state",
        "best_round", "best_validation_accuracy", "history",
        "rng_states", "identity", "state_sha256", "payload_sha256",
    ]
    for field in required_fields:
        if field not in payload:
            raise CheckpointError(f"Checkpoint thiếu trường bắt buộc: {field}")

    expected_hash = payload["payload_sha256"]
    body = {k: v for k, v in payload.items() if k != "payload_sha256"}
    if payload_hash(body) != expected_hash:
        raise CheckpointError("Checkpoint full payload hash mismatch")

    # Xác minh identity nếu có yêu cầu
    if expected_identity is not None:
        cand_identity = payload["identity"]
        if not verify_job_identity(expected_identity, cand_identity):
            raise CheckpointError(
                f"Identity mismatch! Checkpoint identity không khớp cấu hình mong đợi.\n"
                f"Kỳ vọng hash: {expected_identity.identity_hash}\n"
                f"Thực tế hash: {cand_identity.get('identity_hash')}"
            )

    # Xác minh SHA256 của weights
    actual_sha = compute_state_dict_sha256(payload["global_model_state"])
    if actual_sha != payload["state_sha256"]:
        raise CheckpointError(
            f"Checkpoint SHA256 mismatch! Weights có thể đã bị thay đổi.\n"
            f"Lưu trong header: {payload['state_sha256']}\n"
            f"Thực tế tính lại: {actual_sha}"
        )

    return payload
