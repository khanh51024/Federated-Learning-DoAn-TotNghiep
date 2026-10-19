"""Nạp checkpoint (.pt/.pth) mà không mở pickle tùy ý.

`torch.load(weights_only=False)` giải pickle bất kỳ class nào có trong file, nên một checkpoint
bị thay trên đĩa là đường dẫn tới thực thi mã. `weights_only=True` chặn hết, nhưng checkpoint của
repo này chứa số đo là **numpy scalar** (`np.float64(...)`) và numpy array — bộ unpickler an toàn
không cho chúng qua mặc định, nên phải allowlist tường minh từng cái thay vì quay về `False`.

Allowlist gồm 4 global numpy thực sự xuất hiện trong 55 file `.pt` của `artifacts/` (đo ngày
2026-10-19 bằng cách quét opcode GLOBAL/STACK_GLOBAL: `ndarray`, `dtype`, `_reconstruct`, `scalar`)
và 7 lớp `numpy.dtypes.*DType` mà `web/demo_evidence.py` đã chấp nhận từ trước cho checkpoint huấn
luyện bằng bản numpy khác. `tests/test_checkpoint_io.py` đối chiếu lại từng file mỗi lần chạy;
thêm class lạ vào đây là mở lại đúng lỗ mà hàm này sinh ra để đóng.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import IO, Any, Union

import numpy as np
import torch

_NUMPY_CORE = getattr(np, "_core", None) or np.core  # numpy 2.x / 1.24
_SAFE_GLOBALS: list[Any] = [
    np.ndarray,
    np.dtype,
    _NUMPY_CORE.multiarray.scalar,
    _NUMPY_CORE.multiarray._reconstruct,
]
# type(np.dtype(kind)) cho đúng 7 kiểu số mà checkpoint của repo dùng, không allowlist cả 33 DType của numpy.
_DTYPE_KINDS = ("float64", "int64", "float32", "uint32", "uint64", "int32", "uint8")
_SAFE_GLOBALS += sorted({type(np.dtype(kind)) for kind in _DTYPE_KINDS}, key=lambda cls: cls.__name__)


def _is_torchscript_archive(source: Union[str, Path]) -> bool:
    """Archive do `torch.jit.save` ghi ra: thành viên `constants.pkl` thay vì `data.pkl` của `torch.save`."""
    try:
        with zipfile.ZipFile(source) as archive:
            return any(Path(name).name == "constants.pkl" for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False  # không phải zip: để torch.load tự báo lỗi đúng bản chất


def load_checkpoint(source: Union[str, Path, IO[bytes]], map_location: str = "cpu") -> Any:
    """Trả về payload của checkpoint; ném lỗi nếu file chứa class ngoài allowlist.

    `source` là đường dẫn hoặc file-like đã được kiểm hash trước khi gọi.
    Archive TorchScript (`plantvillage_traced.pt`) bị `weights_only=True` từ chối thẳng dù nó không
    đi qua pickle, nên được nhận diện trước và đưa cho `torch.jit.load` — đúng việc
    `torch.load(weights_only=False)` vẫn làm ngầm trước đây. Không bắt lỗi rồi mới đoán: torch vừa
    phát UserWarning "dispatching to 'torch.jit.load'" vừa ném RuntimeError cho cùng một file, nên
    nương vào câu chữ lỗi của nó là vừa dễ nhầm vừa vỡ khi torch đổi bản.
    """
    # Nạp lại allowlist MỖI lần: `torch.serialization.safe_globals()` gỡ global theo hiệu hai tập
    # khi thoát with-block, nên một nơi khác dùng nó sẽ âm thầm xóa cả phần mình đã đăng ký.
    # `add_safe_globals` là hợp hai tập ⇒ gọi lại bao nhiêu lần cũng vậy, và không bao giờ gỡ ra
    # (quan trọng vì web phục vụ request trong threadpool, global này không phải của riêng ai).
    torch.serialization.add_safe_globals(_SAFE_GLOBALS)
    if isinstance(source, (str, Path)) and _is_torchscript_archive(source):
        return torch.jit.load(source, map_location=map_location)
    return torch.load(source, map_location=map_location, weights_only=True)
