from __future__ import annotations

import pytest

from src.federated.client_app import PlantVillageClient


def test_client_can_identify_synthetic_fallback():
    client = PlantVillageClient(client_id=0, smoke=True, dataset_root=None)
    assert client.data_mode == "synthetic"


def test_require_real_data_refuses_synthetic_fallback():
    with pytest.raises(RuntimeError, match="requires real PlantVillage images"):
        PlantVillageClient(
            client_id=0,
            smoke=True,
            dataset_root=None,
            require_real_data=True,
        )
