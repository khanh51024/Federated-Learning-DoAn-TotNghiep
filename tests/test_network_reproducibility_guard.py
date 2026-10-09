from __future__ import annotations

import pytest

from src.federated.client_app import FlowerClientAdapter, PlantVillageClient


def test_flower_client_refuses_server_seed_mismatch_before_training():
    core = PlantVillageClient(client_id=0, smoke=True, seed=42)
    adapter = FlowerClientAdapter(core)
    parameters = adapter.get_parameters({})

    with pytest.raises(ValueError, match="training seed mismatch"):
        adapter.fit(
            parameters,
            {
                "round": 1,
                "local_epochs": 1,
                "seed": 123,
            },
        )


def test_client_exposes_partition_identity_for_network_audit():
    core = PlantVillageClient(
        client_id=0,
        partition_dir="data/partitions_stage2_scratch_v3/label1",
        smoke=True,
        seed=42,
    )

    assert core.partition_id == "label1"
    assert len(core.partition_config_sha256) == 64
    assert core.partition_config_sha256 != "missing"
