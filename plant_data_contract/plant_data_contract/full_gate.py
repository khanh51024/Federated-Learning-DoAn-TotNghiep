"""Require an immutable, passing image-leakage audit before any full training."""

import hashlib
import json
from pathlib import Path


def verify_full_data_gate(release_dir, release_sha, spec):
    config = spec.get("quality_gate", {}) if isinstance(spec, dict) else {}
    expected_sha = config.get("audit_sha256")
    audit_name = config.get("audit_filename")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or not isinstance(audit_name, str):
        raise ValueError("Full training requires a pinned quality_gate audit in the versioned spec")
    if Path(audit_name).name != audit_name or audit_name in ("", ".", ".."):
        raise ValueError("quality_gate.audit_filename must be a filename inside the release")
    audit_path = Path(release_dir) / audit_name
    raw = audit_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Quality audit SHA mismatch: expected {expected_sha}, got {actual_sha}")
    audit = json.loads(raw)
    if audit.get("release_manifest_sha256") != release_sha:
        raise ValueError("Quality audit references a different release")
    if (audit.get("status") != "PASS" or audit.get("current_bytes_verified") is not True
            or audit.get("unresolved_duplicate_candidates") != 0
            or audit.get("within_split_duplicate_rows") != 0
            or audit.get("confirmed_or_critical_scene_leakages") != 0):
        raise ValueError("Full training blocked: image-leakage audit has unresolved or confirmed duplicate data")
    return audit
