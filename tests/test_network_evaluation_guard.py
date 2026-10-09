from __future__ import annotations

from pathlib import Path

import pytest

from src.federated.server_app import start_flower_server


ROOT = Path(__file__).resolve().parents[1]


def test_network_server_refuses_fake_heldout_evaluator_when_real_required():
    with pytest.raises(RuntimeError, match="Real held-out evaluation is required"):
        start_flower_server(
            server_address="127.0.0.1:18131",
            num_clients=1,
            rounds=1,
            evaluation_dataset_root=ROOT,
            require_real_evaluation=True,
            metrics_file=None,
            checkpoint_file=None,
        )
