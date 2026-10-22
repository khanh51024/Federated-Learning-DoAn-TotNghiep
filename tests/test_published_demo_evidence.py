"""Fail-closed checks for the published branch bundle used by the web demo."""

from __future__ import annotations

import json
import shutil

from src import bundle
from web.demo_evidence import STORE, load_evidence


def test_published_bundle_has_seven_independent_models_and_shared_split():
    evidence = load_evidence()
    assert evidence["available"], evidence.get("reason")
    assert len(evidence["models"]) == 7
    assert {m["model_id"] for m in evidence["models"]} == {
        "centralized", "fedavg", *(f"local_{i:02d}" for i in range(5))
    }
    manifest = bundle.load_manifest(STORE)
    assert evidence["validation_counts"] == bundle.validation_counts(manifest)
    assert evidence["scientific_stage2_complete"] is False
    assert all(
        model["commit"] == manifest["sources"][
            "local" if model["model_id"].startswith("local_") else model["model_id"]
        ]["commit"]
        for model in evidence["models"]
    )


def test_missing_or_tampered_bytes_and_wrong_contract_hide_all_results(tmp_path):
    bundle = tmp_path / "published"
    shutil.copytree(STORE, bundle)
    assert load_evidence(bundle)["available"] is True

    checkpoint = bundle / "fedavg/checkpoint_best.pt"
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint[:-1] + bytes([original_checkpoint[-1] ^ 1]))
    assert load_evidence(bundle)["models"] == []
    checkpoint.write_bytes(original_checkpoint)

    checkpoint.unlink()
    assert load_evidence(bundle)["models"] == []
    checkpoint.write_bytes(original_checkpoint)

    manifest_path = bundle / "import_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    for change in (
        {"validation_manifest_sha256": "0" * 64},
        {"release_manifest_sha256": "1" * 64},      # lệch artifact_manifest và checkpoint
        {"seed": 7},                                 # lệch summary
        {"shard_total": 1},                          # lệch tổng n_k
        {"classes": ["wrong_class", *original["classes"][1:]]},
    ):
        manifest_path.write_text(json.dumps({**original, **change}), encoding="utf-8")
        result = load_evidence(bundle)
        assert result["available"] is False
        assert result["models"] == []
