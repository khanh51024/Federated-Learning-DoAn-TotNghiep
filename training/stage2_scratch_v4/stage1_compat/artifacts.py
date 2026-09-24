"""Validate completed evaluations before resume/skip/collection. CPU only."""
from pathlib import Path
from stage1_compat.integrity import read_json, finite, file_hash, digest
from stage1_compat.identity import verify_stored_identity


def validate_completed(job_dir, expected_identity=None):
    import torch
    from stage1_compat.checkpoint import compute_state_dict_sha256, load_checkpoint
    root = Path(job_dir)
    metrics = read_json(root / "fedavg_metrics.json")
    if metrics.get("status") != "COMPLETED" or metrics.get("experiment") != "fedavg":
        raise ValueError("Evaluation is not completed FedAvg")
    identity = verify_stored_identity(metrics["identity"])
    if expected_identity is not None and identity != expected_identity.to_dict():
        raise ValueError("Evaluation identity mismatch")
    context = identity.get("context", {})
    if context.get("calibration"):
        raise ValueError("Calibration output is not a completed evaluation")
    if "dataset_sha256" in context:
        manifest = read_json(root.parent / "frozen_dataset_manifest.v2.json")
        manifest_body = {k: v for k, v in manifest.items() if k != "identity_hash"}
        if digest(manifest_body) != manifest.get("identity_hash") or manifest["identity_hash"] != context["dataset_sha256"]:
            raise ValueError("Evaluation dataset identity mismatch")
        report = read_json(root.parent / "calibration/calibration_report.json")
        body = {k: v for k, v in report.items() if k != "budget_sha256"}
        if digest(body) != report.get("budget_sha256") or report["budget_sha256"] != context.get("budget_sha256"):
            raise ValueError("Evaluation frozen budget mismatch")
    # Resolve only the canonical relative paths, not stale absolute paths in restored JSON.
    best_path = root / "best_model.pth"
    if file_hash(best_path) != metrics["checkpoint_file_sha256"]:
        raise ValueError("Best checkpoint file hash mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["identity"] != identity or best["test_metrics"] != metrics["test_metrics"]:
        raise ValueError("Checkpoint/evaluation identity mismatch")
    if compute_state_dict_sha256(best["model_state"]) != metrics["checkpoint_sha256"]:
        raise ValueError("Best weights hash mismatch")
    checkpoint = load_checkpoint(root / "checkpoints" / f"{metrics['job_id']}_checkpoint.pth")
    rounds = metrics["rounds"]
    if (rounds != identity["rounds"] or metrics["seed"] != identity["seed"]
            or metrics["alpha"] != identity["alpha"] or metrics["clients"] != identity["num_clients"]):
        raise ValueError("Metrics condition/budget differs from identity")
    if (checkpoint["identity"] != identity or checkpoint["server_round"] != rounds
            or metrics["history"] != checkpoint["history"]
            or [row["epoch"] for row in metrics["history"]] != list(range(1, rounds + 1))
            or metrics["best_round"] != checkpoint["best_round"]
            or compute_state_dict_sha256(checkpoint["best_model_state"]) != metrics["checkpoint_sha256"]):
        raise ValueError("Committed round/history/best state mismatch")
    if not 1 <= metrics["best_round"] <= rounds:
        raise ValueError("Invalid best round")
    for name in ("accuracy", "macro_f1"):
        finite(metrics["test_metrics"][name], name, maximum=1)
    finite(metrics["test_metrics"]["loss"], "test loss")
    if metrics["test_metrics"]["samples"] != metrics["split"]["test"]:
        raise ValueError("Test sample count mismatch")
    return metrics
