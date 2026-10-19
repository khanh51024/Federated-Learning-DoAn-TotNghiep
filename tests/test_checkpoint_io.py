"""Checkpoint chỉ được nạp qua cổng an toàn: `weights_only=True` + allowlist numpy tối thiểu.

Lý do có file test này: `torch.serialization.safe_globals()` gỡ global theo HIỆU HAI TẬP khi thoát
with-block, nên một nơi khác trong cùng tiến trình dùng nó sẽ âm thầm xóa allowlist mà
`checkpoint_io` đã đăng ký — và lỗi chỉ hiện ra ở request KẾ TIẾP, không phải chỗ gây ra nó.
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.checkpoint_io import _SAFE_GLOBALS, load_checkpoint


def test_a_numpy_scalar_checkpoint_still_loads_after_another_module_scoped_its_allowlist(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {"w": torch.tensor([1.0])}, "accuracy": np.float64(0.5), "count": np.uint32(7)}, path)

    with torch.serialization.safe_globals(list(_SAFE_GLOBALS)):
        torch.load(path, map_location="cpu", weights_only=True)  # thoát khối này là allowlist bị gỡ

    payload = load_checkpoint(path)
    assert float(payload["accuracy"]) == 0.5 and int(payload["count"]) == 7
    assert torch.equal(payload["model_state"]["w"], torch.tensor([1.0]))


def test_a_payload_holding_an_unlisted_class_is_refused_not_executed(tmp_path: Path):
    """`weights_only=True` phải thật sự có hiệu lực: class lạ trong file là từ chối, không phải thực thi."""
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": {"w": torch.tensor([1.0])}, "args": argparse.Namespace(lr=0.1)}, path)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_checkpoint(path)


def test_a_torchscript_archive_is_loaded_by_jit_not_by_pickle(tmp_path: Path):
    """`weights_only=True` từ chối thẳng archive TorchScript; helper phải nhận diện và đưa cho `torch.jit.load`."""
    class _Double(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2

    path = tmp_path / "traced.pt"
    torch.jit.trace(_Double(), torch.tensor([1.0])).save(str(path))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = load_checkpoint(path)
    assert isinstance(module, torch.jit.ScriptModule)
    assert float(module(torch.tensor([3.0]))[0]) == 6.0
    # Nhận diện TRƯỚC khi gọi torch.load: không được để torch kịp phát warning
    # "dispatching to 'torch.jit.load'" rồi mới từ chối — warning đó nói ngược với việc xảy ra.
    assert not [w for w in caught if "dispatching to 'torch.jit.load'" in str(w.message)]


def test_the_allowlist_covers_every_checkpoint_the_repo_actually_ships():
    """Không để allowlist trôi khỏi thực tế: mọi .pt trong artifacts/ phải nạp được."""
    from src import registry as reg

    paths = sorted((reg.ROOT / "artifacts").rglob("*.pt"))
    if not paths:
        pytest.skip("máy này không có artifacts/ — không có gì để đối chiếu")
    for path in paths:
        assert load_checkpoint(path) is not None, path
