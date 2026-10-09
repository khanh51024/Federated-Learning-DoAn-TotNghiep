from web.app import _artifact_scientific_status


def test_current_canonical_artifacts_are_marked_unverified():
    status = _artifact_scientific_status("canonical")
    assert status["overall_scientific_valid"] is False
    assert status["centralized"]["scientific_valid"] is False
    assert status["federated"]["scientific_valid"] is False


def test_current_network_smoke_artifact_is_not_scientifically_valid():
    status = _artifact_scientific_status("network")
    assert status["overall_scientific_valid"] is False
    assert status["federated"]["scientific_valid"] is False
