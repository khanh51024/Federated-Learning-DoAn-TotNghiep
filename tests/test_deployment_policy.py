"""Khóa hành vi: chạy Flower ở chế độ deployment phải chọn TƯỜNG MINH một chính sách validator.

Không còn trạng thái "để trống cả hai cờ" — trạng thái đó âm thầm trung bình cả cập nhật Byzantine.
Chế độ observe-only vẫn được giữ vì là một ablation có chủ đích (tests/test_attack_matrix_network.py),
nhưng phải bật công khai và chỉ trên loopback."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from src.federated.server_app import LOOPBACK_HOSTS, check_deployment_policy

ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    base = dict(reject_suspicious_updates=False, observe_only=False, host="127.0.0.1", max_update_norm=10.0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_missing_policy_is_refused_and_names_both_options():
    with pytest.raises(ValueError) as exc:
        check_deployment_policy(_args())
    assert "--reject-suspicious-updates" in str(exc.value) and "--observe-only" in str(exc.value)


def test_the_two_policies_are_mutually_exclusive():
    with pytest.raises(ValueError, match="loại trừ nhau"):
        check_deployment_policy(_args(reject_suspicious_updates=True, observe_only=True))


def test_observe_only_is_refused_off_loopback():
    for host in ("0.0.0.0", "10.0.0.5", "datn.tranngochung.id.vn"):
        with pytest.raises(ValueError, match="loopback"):
            check_deployment_policy(_args(observe_only=True, host=host))


def test_observe_only_is_allowed_on_every_loopback_spelling():
    for host in LOOPBACK_HOSTS:
        assert check_deployment_policy(_args(observe_only=True, host=host)) == []


def test_reject_suspicious_passes_without_warning_when_a_threshold_is_given():
    assert check_deployment_policy(_args(reject_suspicious_updates=True)) == []


def test_missing_update_norm_threshold_warns_but_does_not_block():
    """Web demo spawn server mà không truyền ngưỡng nên không được chặn; nhưng phải kêu to."""
    warnings = check_deployment_policy(_args(reject_suspicious_updates=True, max_update_norm=None))
    assert len(warnings) == 1 and "--max-update-norm" in warnings[0]


def test_the_cli_main_block_actually_wires_the_guard():
    """Unit test ở trên không thấy khối __main__; lệnh này kiểm tra guard thật sự được nối vào CLI
    và argparse thoát mã 2 trước khi kịp khởi động server."""
    result = subprocess.run(
        [sys.executable, "-m", "src.federated.server_app", "--mode", "deployment", "--num-clients", "1", "--rounds", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 2, result.stderr
    assert "--observe-only" in result.stderr