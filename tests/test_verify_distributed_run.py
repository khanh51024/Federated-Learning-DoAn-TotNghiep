"""The acceptance gate must reject incomplete or corrupted evidence."""
import json

import pytest

from scripts.verify_distributed_run import RELEASE_SHA, audit_equivalence, audit_run


@pytest.fixture
def completed_preflight(tmp_path):
    cfg = {"method": "fedavg", "budget": 3, "local_epochs": 1, "batch_size": 128,
           "lr": 0.02, "channels_last": True, "amp": "bf16", "limit": 3000,
           "release_manifest_sha256": RELEASE_SHA, "runtime": "flower", "seed": 42,
           "scheme": "label_alpha_0_1", "device": "cuda", "train_transform": "gpu_v2 (test)",
           "train_sizes": [600] * 5}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "final.json").write_text(json.dumps({"w0_sha256": "a" * 64}))
    for name in ("DONE", "checkpoint_best.pt", "logits_best.npz"):
        (tmp_path / name).write_bytes(b"present")
    rows = []
    for step in range(3):
        clients = [{"client": k, "delta_l2": 1.0, "l2_to_merged": 0.5,
                    "l2_to_merged_without_bn": 0.1, "loss": 1.0,
                    "transport_up_exact": True, "transport_down_exact": True,
                    "bytes_up": 100, "bytes_down": 100} for k in range(5)]
        rows.append({"step": step, "seconds": 1.0, "train": {"clients": clients,
                     "aggregation_check": {"max_abs_diff": 0.0, "tensors_compared": 100}}})
    (tmp_path / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return tmp_path, rows


def test_complete_preflight_counts_both_transport_directions(completed_preflight):
    path, _ = completed_preflight
    result = audit_run(path, preflight=True)
    assert result["passed"] and result["transport_comparisons"] == 30


@pytest.mark.parametrize("fault", ["transport", "missing_client", "aggregation_nan", "aggregation_wrong", "missing_round", "wrong_device"])
def test_gate_rejects_bad_evidence_even_with_done(completed_preflight, fault):
    path, rows = completed_preflight
    if fault == "transport":
        rows[1]["train"]["clients"][2]["transport_up_exact"] = False
    elif fault == "missing_client":
        rows[1]["train"]["clients"].pop()
    elif fault == "aggregation_nan":
        rows[1]["train"]["aggregation_check"]["max_abs_diff"] = float("nan")
    elif fault == "aggregation_wrong":
        rows[1]["train"]["aggregation_check"]["max_abs_diff"] = 1e-4
    elif fault == "missing_round":
        rows.pop()
    else:
        config = json.loads((path / "config.json").read_text())
        config["device"] = "cpu"
        (path / "config.json").write_text(json.dumps(config))
    (path / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    assert not audit_run(path, preflight=True)["passed"]


def test_float32_rounding_at_unity_is_not_rejected(completed_preflight):
    path, rows = completed_preflight
    rows[1]["train"]["aggregation_check"]["max_abs_diff"] = 2 ** -23
    (path / "history.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    assert audit_run(path, preflight=True)["passed"]


def test_equivalence_requires_both_schemes_and_fixed_finite_tolerances(tmp_path):
    checks = [{"check": "A_fedsgd", "dtype": "float64", "scheme": scheme,
               "max_abs": 1e-14, "l2_relative_to_centralized_travel": 1e-12}
              for scheme in ("iid", "label_alpha_0_1")]
    path = tmp_path / "equivalence.json"
    path.write_text(json.dumps({"checks": checks}))
    assert audit_equivalence(path)["passed"]
    checks[0]["max_abs"] = 1e-5
    path.write_text(json.dumps({"checks": checks}))
    assert not audit_equivalence(path)["passed"]
    path.write_text(json.dumps({"checks": []}))
    assert not audit_equivalence(path)["passed"]
