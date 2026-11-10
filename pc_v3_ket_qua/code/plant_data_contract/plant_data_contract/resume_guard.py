import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("resume_guard")

_SENTINEL = object()


def _is_safe_relative_path(rel_path: Any, base_dir: Path) -> bool:
    """Validate that rel_path is a non-empty relative path strictly contained inside base_dir."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        return False
    s = rel_path.replace("\\", "/")
    if s.startswith("/") or ":" in s or ".." in s.split("/"):
        return False
    try:
        resolved = (base_dir / s).resolve()
        return resolved.is_relative_to(base_dir)
    except Exception:
        return False


def _is_valid_sha256(h: Any) -> bool:
    """Validate that h is a valid 64-character lowercase hexadecimal SHA-256 string."""
    return isinstance(h, str) and bool(re.match(r"^[0-9a-f]{64}$", h.strip().lower()))


def compute_bytes_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file in 64KB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_transaction_manifest(
    manifest: Dict[str, Any],
    base_dir: Union[str, Path],
    verify_hashes: bool = True,
    expected_config: Optional[Dict[str, Any]] = None,
    is_fedavg: bool = False,
    allow_legacy: bool = False,
) -> Dict[str, Any]:
    """Validate a commit manifest strictly according to fail-closed transaction rules.
    Verifies schema, status, relative paths, containment within base_dir, and SHA-256 hashes.
    Raises ValueError or RuntimeError if invalid.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Commit manifest must be a dictionary.")

    if manifest.get("status") != "COMMITTED":
        raise ValueError(f"Manifest status must be 'COMMITTED', got {manifest.get('status')!r}")

    gen = manifest.get("generation")
    if not isinstance(gen, int) or gen < 0:
        raise ValueError(f"Invalid generation in manifest: {gen!r}")

    best_gen = manifest.get("best_generation", gen)
    if not isinstance(best_gen, int) or best_gen > gen:
        raise ValueError(f"Manifest best_generation ({best_gen}) exceeds generation ({gen})")

    last_rel = manifest.get("last_checkpoint_path")
    best_rel = manifest.get("best_checkpoint_path")
    last_sha = manifest.get("last_checkpoint_sha256")
    best_sha = manifest.get("best_checkpoint_sha256")

    if not _is_valid_sha256(last_sha):
        raise ValueError(f"Invalid or missing last_checkpoint_sha256: {last_sha!r}")
    if not _is_valid_sha256(best_sha):
        raise ValueError(f"Invalid or missing best_checkpoint_sha256: {best_sha!r}")

    base_dir_resolved = Path(base_dir).resolve()
    if not _is_safe_relative_path(last_rel, base_dir_resolved):
        raise ValueError(f"last_checkpoint_path is unsafe or escapes base_dir: {last_rel!r}")
    if not _is_safe_relative_path(best_rel, base_dir_resolved):
        raise ValueError(f"best_checkpoint_path is unsafe or escapes base_dir: {best_rel!r}")

    last_file = (base_dir_resolved / last_rel).resolve()
    best_file = (base_dir_resolved / best_rel).resolve()

    files = manifest.get("files")
    if files is not None:
        if not isinstance(files, list):
            raise ValueError(f"Manifest 'files' must be a list, got {type(files)}")
        for f_entry in files:
            if not isinstance(f_entry, dict):
                raise ValueError("Manifest file entry must be a dict")
            f_rel = f_entry.get("relative_path")
            if not f_rel or not _is_safe_relative_path(f_rel, base_dir_resolved):
                raise ValueError(f"Unsafe or missing relative_path in files entry: {f_rel!r}")
            f_sha = f_entry.get("sha256")
            if not _is_valid_sha256(f_sha):
                raise ValueError(f"Invalid sha256 in file entry: {f_sha!r}")
            if verify_hashes:
                target_f = (base_dir_resolved / f_rel).resolve()
                if not target_f.is_file():
                    raise FileNotFoundError(f"Referenced file does not exist: {target_f}")
                actual_sha = compute_bytes_sha256(target_f)
                if actual_sha != f_sha:
                    raise ValueError(f"File SHA mismatch for {f_rel}! Expected {f_sha}, got {actual_sha}")

    if verify_hashes:
        if not last_file.is_file():
            raise FileNotFoundError(f"Referenced last checkpoint does not exist: {last_file}")
        actual_last_sha = compute_bytes_sha256(last_file)
        if actual_last_sha != last_sha:
            raise ValueError(f"last checkpoint SHA mismatch! Expected {last_sha}, got {actual_last_sha}")

        if not best_file.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: referenced best checkpoint does not exist: {best_file}")
        actual_best_sha = compute_bytes_sha256(best_file)
        if actual_best_sha != best_sha:
            raise ValueError(f"best checkpoint SHA mismatch! Expected {best_sha}, got {actual_best_sha}")

        if expected_config is not None:
            last_ckpt = torch.load(last_file, map_location="cpu", weights_only=False)
            best_ckpt = torch.load(best_file, map_location="cpu", weights_only=False)
            validate_config_identity(last_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
            validate_config_identity(best_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)

    return manifest



class ProtocolConfigDict(dict):
    """Configuration dictionary that recomputes canonical protocol_content_sha256 upon modification."""
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key != "protocol_content_sha256" and key in IMMUTABLE_PROTOCOL_KEYS:
            super().__setitem__("protocol_content_sha256", compute_job_protocol_sha256(self))


def save_atomic_checkpoint(state: Dict[str, Any], filepath: Union[str, Path]) -> str:
    """Save PyTorch checkpoint state dictionary crash-safely without in-place corruption.

    Guarantees:
    - Writes to unique temporary file with flush and fsync
    - Computes and verifies SHA-256
    - Converts resolved_config to plain dict to ensure cross-process unpickling portability
    - Uses atomic os.replace where possible
    - Cross-filesystem fallback never overwrites target directly; stages to target temp file first
    """
    p = Path(filepath).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.parent / f".tmp_{p.name}_{uuid.uuid4().hex[:8]}"

    if isinstance(state, dict):
        state = copy.copy(state)
        if "resolved_config" in state and isinstance(state["resolved_config"], dict):
            state["resolved_config"] = dict(state["resolved_config"])

    with open(temp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())

    sha256 = compute_bytes_sha256(temp_path)

    try:
        os.replace(temp_path, p)
    except OSError:
        target_temp = p.parent / f".target_tmp_{p.name}_{uuid.uuid4().hex[:8]}"
        shutil.copy2(temp_path, target_temp)
        target_sha = compute_bytes_sha256(target_temp)
        if target_sha != sha256:
            if target_temp.exists():
                try:
                    target_temp.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Cross-device copy integrity check failed for {p}")
        try:
            os.replace(target_temp, p)
        except OSError:
            backup_path = p.parent / f".bak_{p.name}_{uuid.uuid4().hex[:8]}"
            if p.is_file():
                try:
                    os.replace(p, backup_path)
                except OSError:
                    pass
            shutil.copy2(target_temp, p)
            if backup_path.is_file():
                try:
                    backup_path.unlink()
                except OSError:
                    pass
        finally:
            if target_temp.exists():
                try:
                    target_temp.unlink()
                except OSError:
                    pass
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return sha256


def save_crash_safe_checkpoint(
    output_dir: Union[str, Path],
    state: Dict[str, Any],
    step: int,
    is_best: bool,
    is_fedavg: bool = False,
) -> Dict[str, Any]:
    """Crash-safe generation-committed checkpoint manager.

    1. Writes immutable generation: checkpoints/gen_{step:06d}/checkpoint_last.pt
    2. If is_best: writes checkpoints/gen_{step:06d}/checkpoint_best.pt
    3. Writes checkpoints/gen_{step:06d}/meta.json
    4. Updates top-level checkpoint_last.pt (and checkpoint_best.pt if is_best) crash-safely
    5. Atomically writes and commits commit_manifest.json linking generation, last, best, scores, and hashes
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(state, dict):
        state = copy.copy(state)
        if "resolved_config" in state and isinstance(state["resolved_config"], dict):
            state["resolved_config"] = dict(state["resolved_config"])

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    gen_base = f"gen_{step:06d}"
    gen_name = gen_base
    gen_dir = checkpoints_dir / gen_name
    if gen_dir.exists():
        gen_name = f"{gen_base}_{uuid.uuid4().hex[:6]}"
        gen_dir = checkpoints_dir / gen_name

    temp_gen_dir = checkpoints_dir / f"temp_{gen_name}_{uuid.uuid4().hex[:8]}"
    temp_gen_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write last checkpoint into temp generation dir
    temp_last = temp_gen_dir / "checkpoint_last.pt"
    with open(temp_last, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    last_sha = compute_bytes_sha256(temp_last)
    # Verify deserialization
    torch.load(temp_last, map_location="cpu", weights_only=False)

    # 2. Write best checkpoint if is_best
    best_sha = None
    if is_best:
        temp_best = temp_gen_dir / "checkpoint_best.pt"
        with open(temp_best, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        best_sha = compute_bytes_sha256(temp_best)
        torch.load(temp_best, map_location="cpu", weights_only=False)

    # 3. Read prior commit manifest to retain prior best if current is not best
    commit_manifest_path = out_dir / "commit_manifest.json"
    prior_manifest = {}
    if commit_manifest_path.is_file():
        try:
            prior_manifest = json.loads(commit_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            prior_manifest = {}

    best_val_f1_raw = float(state.get("best_val_f1_raw", state.get("best_val_f1", -1.0)))

    if is_best:
        committed_best_gen = step
        committed_best_path = f"checkpoints/{gen_name}/checkpoint_best.pt"
        committed_best_sha = best_sha
        committed_best_score = best_val_f1_raw
    else:
        committed_best_gen = prior_manifest.get("best_generation", step)
        committed_best_path = prior_manifest.get("best_checkpoint_path")
        committed_best_sha = prior_manifest.get("best_checkpoint_sha256")
        committed_best_score = prior_manifest.get("best_score", best_val_f1_raw)

        # Fallback to state or top-level checkpoint_best.pt if prior_manifest didn't have best_checkpoint_sha256
        if committed_best_sha is None:
            if state.get("best_checkpoint_sha256"):
                committed_best_sha = state.get("best_checkpoint_sha256")
            elif (out_dir / "checkpoint_best.pt").is_file():
                try:
                    torch.load(out_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
                    committed_best_sha = compute_bytes_sha256(out_dir / "checkpoint_best.pt")
                    committed_best_path = "checkpoint_best.pt"
                except Exception:
                    pass
            if committed_best_sha is None and checkpoints_dir.is_dir():
                for g_dir in sorted(checkpoints_dir.glob("gen_*"), reverse=True):
                    cand_best = g_dir / "checkpoint_best.pt"
                    cand_marker = g_dir / "commit_marker.json"
                    if cand_best.is_file() and cand_marker.is_file():
                        try:
                            torch.load(cand_best, map_location="cpu", weights_only=False)
                            committed_best_sha = compute_bytes_sha256(cand_best)
                            committed_best_path = f"checkpoints/{g_dir.name}/checkpoint_best.pt"
                            break
                        except Exception:
                            continue

        # If best checkpoint path points to top-level alias or outside generation tree,
        # copy it into the current generation directory so it becomes permanently immutable
        if committed_best_path == "checkpoint_best.pt" or (committed_best_path and not committed_best_path.startswith("checkpoints/")):
            source_best = out_dir / (committed_best_path or "checkpoint_best.pt")
            if source_best.is_file():
                dest_best = temp_gen_dir / "checkpoint_best.pt"
                shutil.copy2(source_best, dest_best)
                committed_best_path = f"checkpoints/{gen_name}/checkpoint_best.pt"
                committed_best_sha = compute_bytes_sha256(dest_best)
            elif (out_dir / "checkpoint_best.pt").is_file():
                dest_best = temp_gen_dir / "checkpoint_best.pt"
                shutil.copy2(out_dir / "checkpoint_best.pt", dest_best)
                committed_best_path = f"checkpoints/{gen_name}/checkpoint_best.pt"
                committed_best_sha = compute_bytes_sha256(dest_best)

    gen_meta = {
        "generation": step,
        "gen_name": gen_name,
        "is_best": is_best,
        "is_fedavg": is_fedavg,
        "last_sha256": last_sha,
        "best_sha256": committed_best_sha,
        "best_val_f1_raw": best_val_f1_raw,
        "best_score": committed_best_score,
        "best_generation": committed_best_gen,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (temp_gen_dir / "meta.json").write_text(json.dumps(gen_meta, indent=2), encoding="utf-8")

    # 4. Promote temp gen dir to immutable gen dir (NEVER delete existing generations)
    try:
        os.replace(temp_gen_dir, gen_dir)
    except OSError:
        shutil.copytree(temp_gen_dir, gen_dir)
        shutil.rmtree(temp_gen_dir, ignore_errors=True)

    # 5. Atomically commit transaction manifest FIRST
    new_manifest = {
        "status": "COMMITTED",
        "generation": step,
        "gen_name": gen_name,
        "is_best": is_best,
        "is_fedavg": is_fedavg,
        "last_checkpoint_path": f"checkpoints/{gen_name}/checkpoint_last.pt",
        "last_checkpoint_sha256": last_sha,
        "best_checkpoint_path": committed_best_path,
        "best_checkpoint_sha256": committed_best_sha,
        "best_score": committed_best_score,
        "best_generation": committed_best_gen,
        "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # Save copy of commit marker inside the generation directory FIRST
    (gen_dir / "commit_marker.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    # Write root commit_manifest.json atomically
    temp_manifest = out_dir / f".commit_manifest_{uuid.uuid4().hex[:8]}.tmp"
    temp_manifest.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    try:
        os.replace(temp_manifest, commit_manifest_path)
    except OSError:
        shutil.copy2(temp_manifest, commit_manifest_path)
        try:
            temp_manifest.unlink()
        except OSError:
            pass

    # 6. ONLY AFTER manifest is committed: update top-level checkpoint_last.pt and checkpoint_best.pt
    save_atomic_checkpoint(state, out_dir / "checkpoint_last.pt")
    if is_best:
        save_atomic_checkpoint(state, out_dir / "checkpoint_best.pt")

    return new_manifest


def load_verified_checkpoint(
    output_dir: Union[str, Path],
    expected_config: Optional[Dict[str, Any]] = None,
    is_fedavg: bool = False,
    allow_legacy: bool = False,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Load verified checkpoint with atomic generation recovery and paired check.
    Strictly verifies both last and best checkpoints against commit manifest hashes.
    Guarantees READ-ONLY access on output_dir: never mutates or writes to input.
    Selects a single consistent committed transaction.
    """
    out_dir = Path(output_dir).resolve()
    commit_manifest_path = out_dir / "commit_manifest.json"
    top_last = out_dir / "checkpoint_last.pt"
    top_best = out_dir / "checkpoint_best.pt"

    # Step 1: Collect all candidate COMMITTED manifests
    candidate_manifests: List[Dict[str, Any]] = []

    # Check root commit_manifest.json
    if commit_manifest_path.is_file():
        try:
            root_m = json.loads(commit_manifest_path.read_text(encoding="utf-8"))
            if isinstance(root_m, dict) and root_m.get("status") == "COMMITTED":
                candidate_manifests.append(root_m)
        except Exception as e:
            logger.warning("Root commit_manifest.json is torn or unreadable (%s), falling back to generation markers", e)

    # Scan generation commit_marker.json files
    checkpoints_dir = out_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        for marker_path in checkpoints_dir.glob("*/commit_marker.json"):
            try:
                marker_m = json.loads(marker_path.read_text(encoding="utf-8"))
                if isinstance(marker_m, dict) and marker_m.get("status") == "COMMITTED":
                    # Avoid duplicate if matches root
                    if not any(c.get("gen_name") == marker_m.get("gen_name") and c.get("generation") == marker_m.get("generation") for c in candidate_manifests):
                        candidate_manifests.append(marker_m)
            except Exception:
                continue

    # Sort candidates by generation step descending (latest first)
    candidate_manifests.sort(key=lambda m: m.get("generation", 0), reverse=True)

    # Step 2: Try each candidate manifest in order of latest generation
    for manifest in candidate_manifests:
        gen_step = manifest.get("generation", 0)
        best_gen = manifest.get("best_generation", gen_step)
        # Invariant: best_step <= last_step
        if best_gen > gen_step:
            logger.warning("Rejecting candidate commit gen %s: best_generation %s > generation %s", gen_step, best_gen, gen_step)
            continue

        gen_last_rel = manifest.get("last_checkpoint_path")
        exp_last_sha = manifest.get("last_checkpoint_sha256")
        gen_best_rel = manifest.get("best_checkpoint_path")
        exp_best_sha = manifest.get("best_checkpoint_sha256")

        # In modern commits, both SHA hashes must be valid 64-character hex strings
        if not _is_valid_sha256(exp_last_sha) or not _is_valid_sha256(exp_best_sha):
            logger.warning("Rejecting candidate commit gen %s: invalid or missing SHA-256 (last=%s, best=%s)", gen_step, exp_last_sha, exp_best_sha)
            continue

        # Both relative paths must be safe and contained inside out_dir
        if not _is_safe_relative_path(gen_last_rel, out_dir) or not _is_safe_relative_path(gen_best_rel, out_dir):
            logger.warning("Rejecting candidate commit gen %s: path traversal or unsafe relative path (last=%s, best=%s)", gen_step, gen_last_rel, gen_best_rel)
            continue

        gen_last_path = (out_dir / gen_last_rel).resolve()
        gen_best_path = (out_dir / gen_best_rel).resolve()

        # Verify last checkpoint exists and matches expected SHA
        if not gen_last_path.is_file() or compute_bytes_sha256(gen_last_path) != exp_last_sha:
            logger.warning("Candidate commit generation %s last checkpoint failed verification, rolling back...", gen_step)
            continue

        # Verify best checkpoint exists and matches expected SHA
        if not gen_best_path.is_file() or compute_bytes_sha256(gen_best_path) != exp_best_sha:
            logger.warning("Candidate commit generation %s best checkpoint failed verification, rolling back...", gen_step)
            continue

        # Load both checkpoints and validate config identity
        try:
            last_ckpt = torch.load(gen_last_path, map_location="cpu", weights_only=False)
            best_ckpt = torch.load(gen_best_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning("Candidate commit generation %s failed deserialization: %s", gen_step, e)
            continue

        if expected_config is not None:
            try:
                validate_config_identity(last_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
                validate_config_identity(best_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
            except Exception as e:
                logger.warning("Candidate commit generation %s failed config validation: %s", gen_step, e)
                continue

        # Check step invariant from loaded contents
        last_step_val = last_ckpt.get("epoch" if not is_fedavg else "round", gen_step)
        best_step_val = best_ckpt.get("epoch" if not is_fedavg else "round", best_gen)
        if best_step_val > last_step_val:
            logger.warning("Rejecting candidate commit gen %s: best_step %s > last_step %s", gen_step, best_step_val, last_step_val)
            continue

        # Found consistent valid commit!
        return last_ckpt, gen_last_path, gen_best_path

    # Fail closed: If candidate manifests were present, all of them failed verification!
    # Under no circumstances should we fall back to unverified top-level aliases.
    if candidate_manifests:
        raise RuntimeError(
            f"Cannot recover checkpoint in {out_dir}: {len(candidate_manifests)} candidate commit(s) "
            f"were found, but all failed integrity, hash, path containment, or config validation. "
            f"Cannot recover valid checkpoint_last.pt. Cannot recover valid checkpoint_best.pt."
        )

    # Step 3: Legacy fallback when NO candidate commit manifests exist
    if not allow_legacy:
        raise RuntimeError(
            f"Cannot recover checkpoint in {out_dir}: no committed transaction manifests found, "
            f"and allow_legacy=False. Refusing to fall back to unverified top-level checkpoints."
        )

    # Pure legacy mode: paired top_last and top_best required
    if top_last.is_file() and top_best.is_file():
        try:
            last_ckpt = torch.load(top_last, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Cannot recover valid checkpoint_last.pt in {out_dir}: {e}")
        try:
            best_ckpt = torch.load(top_best, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Cannot recover valid checkpoint_best.pt in {out_dir}: {e}")
        if expected_config is not None:
            validate_config_identity(last_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
            validate_config_identity(best_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
        return last_ckpt, top_last, top_best

    if top_last.is_file() and not top_best.is_file():
        raise RuntimeError(f"Cannot recover valid checkpoint_best.pt in {out_dir}: paired checkpoint_best.pt is missing.")

    raise RuntimeError(
        f"Cannot recover valid checkpoints in {out_dir}: all candidate commits and top-level checkpoints are corrupt or missing."
    )


def load_verified_best_checkpoint(
    output_dir: Union[str, Path],
    expected_config: Optional[Dict[str, Any]] = None,
    is_fedavg: bool = False,
    allow_legacy: bool = False,
) -> Tuple[Dict[str, Any], Path]:
    """Load and return verified BEST checkpoint state dictionary and its file path.

    Guarantees that the returned checkpoint is the verified BEST model weights (not LAST).
    """
    _, _, best_path = load_verified_checkpoint(
        output_dir=output_dir,
        expected_config=expected_config,
        is_fedavg=is_fedavg,
        allow_legacy=allow_legacy,
    )
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    if expected_config is not None:
        validate_config_identity(best_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)
    return best_ckpt, best_path


LOCATION_PROVENANCE_KEYS = {
    "spec_file_path",
    "spec_path",
    "campaign_spec_path",
    "output_dir",
    "dataset_root",
    "release_dir",
    "resume_path",
    "checkpoint_path",
    "model_path",
}

PROVENANCE_AUDIT_KEYS = {
    "campaign_spec_sha256",
    "campaign_spec_raw_sha256",
    "spec_raw_sha256",
    "raw_spec_sha256",
}

IMMUTABLE_PROTOCOL_KEYS = (
    "job_id",
    "condition",
    "partition_scheme",
    "seed",
    "rounds",
    "local_epochs",
    "batch_size",
    "client_lr",
    "client_optimizer",
    "client_weight_decay",
    "client_momentum",
    "expected_clients",
    "w0_fingerprint",
    "release_manifest_sha256",
    "aggregation",
    # Centralized parameters
    "mode",
    "optimizer",
    "weight_decay",
    "lr",
    "patience",
    "max_train_batches",
    "max_val_batches",
    "amp",
    "epochs",
    # Semantic protocol identities (F6)
    "architecture_version",
    "preprocessing_version",
    "scheduler_policy",
    "selected_partition_digest",
    # Quality track parameters (Gate C)
    "init",
    "plantdoc_sampling_ratio",
    "pretrained_file_sha256",
    "quality_floor",
    "quality_audit_sha256",
    "convergence_policy",
    "sampler_repeat_policy",
)


def compute_job_protocol_sha256(effective_params: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash from canonical JSON serialization of
    the selected job's effective immutable training, data, and model parameters.
    Excludes formatting, locations, administrative metadata (owner, quota, status, notes),
    metrics, and unrelated jobs.
    """
    canonical_dict = {}
    for k in sorted(IMMUTABLE_PROTOCOL_KEYS):
        if k in effective_params and effective_params[k] is not None:
            # Backward compatibility for historical checkpoints & baseline default:
            # Historical checkpoints omitted sampler_repeat_policy from canonical_dict,
            # and the historical default policy has always been "with_replacement".
            # To preserve identical hashes for all historical checkpoints and default runs,
            # we omit "with_replacement" (treating it as the baseline default),
            # while ANY alternative policy (e.g., "cycle_without_replacement") is included,
            # producing a distinct, verifiable protocol content hash.
            if k == "sampler_repeat_policy" and effective_params[k] == "with_replacement":
                continue
            canonical_dict[k] = effective_params[k]
    canonical_bytes = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def is_location_provenance_key(key: str) -> bool:
    """Return True if key represents file system location/path provenance rather than semantic protocol."""
    if not isinstance(key, str):
        return False
    if key in LOCATION_PROVENANCE_KEYS:
        return True
    lower = key.lower()
    return lower.endswith("_path") or lower.endswith("_dir") or lower.endswith("_file") or lower.endswith("_filepath")


def validate_config_identity(
    ckpt: Dict[str, Any],
    expected_config: Dict[str, Any],
    is_fedavg: bool = False,
    allow_legacy: bool = False,
) -> None:
    """Validate that checkpoint configuration matches expected resolved config identically.
    Fails closed on any protocol, hyperparameter, cap (including explicit None), dataset hash, or seed mismatch.
    Rejects missing resolved_config, missing required fields, and conflicting top-level/nested identities.
    Location metadata and raw whole-spec file hashes are separated from semantic protocol identity.
    """
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    # 1. Require complete typed resolved_config schema for non-legacy checkpoints
    has_resolved_cfg = "resolved_config" in ckpt and isinstance(ckpt["resolved_config"], dict)
    if not has_resolved_cfg:
        if not allow_legacy:
            raise ValueError(
                "Checkpoint missing required 'resolved_config' dictionary schema. Deterministic resume rejected."
            )
        logger.warning("LEGACY CHECKPOINT: Missing resolved_config dictionary. Resuming under allow_legacy=True.")
        ckpt_cfg = {}
    else:
        ckpt_cfg = ckpt["resolved_config"]

    # 2. Reject conflicting top-level and nested identity attributes
    conflict_check_keys = [
        "seed",
        "w0_fingerprint",
        "condition",
        "release_manifest_sha256",
        "mode",
        "job_id",
        "partition_scheme",
    ]
    for ck in conflict_check_keys:
        top_val = ckpt.get(ck)
        nested_val = ckpt_cfg.get(ck)
        if top_val is not None and nested_val is not None and top_val != nested_val:
            raise ValueError(
                f"Conflicting top-level ({top_val!r}) and resolved_config ({nested_val!r}) identity for '{ck}'."
            )

    # 3. Release manifest SHA-256 validation
    ckpt_release_sha = ckpt_cfg.get("release_manifest_sha256", ckpt.get("release_manifest_sha256"))
    exp_release_sha = expected_config.get("release_manifest_sha256")
    if not ckpt_release_sha:
        if not allow_legacy:
            raise ValueError(
                "Checkpoint missing required 'release_manifest_sha256'. Cannot claim verified deterministic resume."
            )
        logger.warning("LEGACY CHECKPOINT: Missing release_manifest_sha256. Resuming without verified dataset hash.")
    elif exp_release_sha and ckpt_release_sha != exp_release_sha:
        raise ValueError(
            f"Release manifest mismatch: checkpoint was trained on '{ckpt_release_sha}', but current runtime expects '{exp_release_sha}'."
        )

    # 4. Strictly validate all required protocol fields from expected_config
    for key, exp_val in expected_config.items():
        if is_location_provenance_key(key):
            # Location provenance metadata (e.g. spec_file_path, output_dir) is preserved for provenance/audit
            # but must not reject resume across different mount paths / environments.
            continue

        if key in PROVENANCE_AUDIT_KEYS:
            # Raw whole-spec file hash is retained for audit provenance only;
            # formatting, comments, or unrelated job status changes must not reject deterministic resume.
            continue

        if key in ckpt_cfg:
            val = ckpt_cfg[key]
        elif key in ckpt:
            val = ckpt[key]
        else:
            val = _SENTINEL

        if key == "protocol_content_sha256":
            # Recompute protocol digest from checkpoint config and compare with expected config
            if ckpt_cfg:
                ckpt_recomputed = compute_job_protocol_sha256(ckpt_cfg)
                # Check for stale stored digest in checkpoint
                if val is not _SENTINEL and val is not None and val != ckpt_recomputed:
                    raise ValueError(
                        f"Stale protocol_content_sha256 digest detected in checkpoint: stored {val!r} does not match recomputed {ckpt_recomputed!r}."
                    )
                # Compare recomputed checkpoint protocol with expected
                exp_recomputed = compute_job_protocol_sha256(expected_config)
                if ckpt_recomputed != exp_recomputed:
                    raise ValueError(
                        f"Protocol identity digest mismatch: checkpoint recomputed hash {ckpt_recomputed!r} "
                        f"does not match expected protocol hash {exp_recomputed!r}."
                    )
            elif val is not _SENTINEL and exp_val is not None and val != exp_val:
                raise ValueError(
                    f"Configuration mismatch for 'protocol_content_sha256': checkpoint has {val!r} vs expected {exp_val!r}."
                )
            continue

        if key == "sampler_repeat_policy":
            # Backward compatibility for historical checkpoints:
            # Missing or None sampler_repeat_policy in checkpoint is semantically equivalent to default "with_replacement"
            ckpt_policy = val if (val is not _SENTINEL and val is not None) else "with_replacement"
            exp_policy = exp_val if exp_val is not None else "with_replacement"
            if ckpt_policy != exp_policy:
                raise ValueError(
                    f"Configuration mismatch for 'sampler_repeat_policy': checkpoint has {val!r} (effective {ckpt_policy!r}) vs expected {exp_val!r}."
                )
            continue

        if val is _SENTINEL:
            if not allow_legacy:
                raise ValueError(
                    f"Checkpoint missing required protocol field '{key}'. Deterministic resume rejected."
                )
            logger.warning("LEGACY CHECKPOINT: Missing protocol field '%s'.", key)
            continue

        # Strict comparison: distinguishes explicit None (uncapped) from integers (capped)
        if val != exp_val:
            raise ValueError(
                f"Configuration mismatch for '{key}': checkpoint has {val!r} vs expected {exp_val!r}."
            )

    # 5. Early stopping state check for Centralized
    if not is_fedavg:
        if "patience_counter" not in ckpt:
            if not allow_legacy:
                raise ValueError(
                    "Checkpoint missing required 'patience_counter'; cannot claim deterministic resume without early stopping state."
                )
            logger.warning("LEGACY CHECKPOINT: Missing patience_counter. Defaulting to 0 with legacy flag.")


def validate_and_restore_rng(ckpt: Dict[str, Any], allow_legacy: bool = False) -> None:
    """Strictly restore all RNG states, failing closed if any engine state is missing, null, or corrupted."""
    if "rng_state" not in ckpt or not isinstance(ckpt["rng_state"], dict):
        if not allow_legacy:
            raise ValueError("Checkpoint missing required 'rng_state' dictionary. Deterministic resume rejected.")
        logger.warning("LEGACY CHECKPOINT: Missing rng_state dictionary.")
        return

    rng_dict = ckpt["rng_state"]
    required_keys = ["python", "numpy", "torch"]
    for rk in required_keys:
        if rk not in rng_dict or rng_dict[rk] is None:
            raise ValueError(f"Incomplete rng_state in checkpoint: missing required engine '{rk}'.")

    # CUDA check: If CUDA is available, checkpoint MUST contain a valid, non-null 'cuda' state
    if torch.cuda.is_available():
        cuda_state = rng_dict.get("cuda")
        if cuda_state is None:
            if not allow_legacy:
                raise ValueError("CUDA is available but checkpoint has null or missing 'cuda' RNG state.")
            logger.warning("LEGACY CHECKPOINT: CUDA is available but checkpoint has missing/null 'cuda' RNG state.")
        else:
            dev_count = torch.cuda.device_count()
            if dev_count > 0:
                if not isinstance(cuda_state, (list, tuple)):
                    raise ValueError(
                        f"CUDA RNG state must be a list/tuple of tensors for {dev_count} devices, got {type(cuda_state).__name__}."
                    )
                if len(cuda_state) != dev_count:
                    raise ValueError(
                        f"CUDA RNG state device count mismatch: checkpoint has {len(cuda_state)} vs current system {dev_count} devices."
                    )
                for idx, s in enumerate(cuda_state):
                    if not isinstance(s, torch.Tensor):
                        raise ValueError(
                            f"CUDA RNG state for device {idx} must be a torch.Tensor, got {type(s).__name__}."
                        )

    try:
        random.setstate(rng_dict["python"])
    except Exception as e:
        raise RuntimeError(f"Failed to restore Python random state: {e}")

    try:
        np.random.set_state(rng_dict["numpy"])
    except Exception as e:
        raise RuntimeError(f"Failed to restore NumPy random state: {e}")

    try:
        t_rng = rng_dict["torch"]
        if not isinstance(t_rng, torch.ByteTensor):
            t_rng = t_rng.to(dtype=torch.uint8, device="cpu")
        torch.set_rng_state(t_rng)
    except Exception as e:
        raise RuntimeError(f"Failed to restore PyTorch CPU RNG state: {e}")

    if torch.cuda.is_available() and rng_dict.get("cuda") is not None:
        if torch.cuda.device_count() > 0:
            try:
                torch.cuda.set_rng_state_all(rng_dict["cuda"])
            except Exception as e:
                raise RuntimeError(f"Failed to restore PyTorch CUDA RNG state: {e}") from e


def import_and_validate_best_checkpoint(
    resume_last_path: Union[str, Path],
    output_dir: Union[str, Path],
    expected_config: Dict[str, Any],
    is_fedavg: bool = False,
    allow_legacy: bool = False,
    source_best_path_override: Optional[Union[str, Path]] = None,
) -> Tuple[Path, float, int]:
    """Ensure prior best checkpoint is validated in-memory BEFORE creating or copying to output.
    Verifies provenance, run identity, and score/epoch consistency between last and best checkpoints.
    Fails closed without writing or modifying target output if validation fails.
    Returns (path_to_best_checkpoint, best_val_f1_raw, best_round_or_epoch).
    """
    resume_last = Path(resume_last_path).resolve()
    out_dir = Path(output_dir).resolve()
    target_best_path = out_dir / "checkpoint_best.pt"
    if source_best_path_override:
        source_best_path = Path(source_best_path_override).resolve()
    else:
        source_best_path = resume_last.parent / "checkpoint_best.pt"

    source_best_exists = source_best_path.is_file()
    target_best_exists = target_best_path.is_file()

    if not source_best_exists and not target_best_exists:
        raise FileNotFoundError(
            f"Incomplete resume package: checkpoint_best.pt must accompany checkpoint_last.pt at {resume_last.parent}. "
            "Prior best model cannot be retained across sessions."
        )

    # 1. Load candidate best checkpoint IN-MEMORY BEFORE WRITING/COPYING ANYTHING
    if source_best_exists:
        best_ckpt = torch.load(source_best_path, map_location="cpu", weights_only=False)
        needs_copy = (target_best_path != source_best_path)
    else:
        best_ckpt = torch.load(target_best_path, map_location="cpu", weights_only=False)
        needs_copy = False

    # 2. Validate candidate best checkpoint config identity
    validate_config_identity(best_ckpt, expected_config, is_fedavg=is_fedavg, allow_legacy=allow_legacy)

    best_val_f1_raw = float(best_ckpt.get("best_val_f1_raw", best_ckpt.get("best_val_f1", -1.0)))
    best_epoch_or_round = int(best_ckpt.get("round" if is_fedavg else "epoch", best_ckpt.get("epoch", -1)))

    # 3. Provenance and Run Trajectory Validation between checkpoint_last and checkpoint_best
    if resume_last.is_file():
        last_ckpt = torch.load(resume_last, map_location="cpu", weights_only=False)

        # Provenance keys must match exactly between last and best
        provenance_keys = ["seed", "w0_fingerprint", "release_manifest_sha256", "condition"]
        if is_fedavg:
            provenance_keys.extend(["job_id", "partition_scheme"])
        else:
            provenance_keys.append("mode")

        for pk in provenance_keys:
            best_k = best_ckpt.get("resolved_config", {}).get(pk, best_ckpt.get(pk))
            last_k = last_ckpt.get("resolved_config", {}).get(pk, last_ckpt.get(pk))
            if best_k is not None and last_k is not None and best_k != last_k:
                raise ValueError(
                    f"Provenance mismatch between checkpoint_last and checkpoint_best for '{pk}': "
                    f"last has {last_k!r} vs best has {best_k!r}."
                )

        # Epoch/round consistency: best round/epoch cannot exceed last round/epoch
        last_epoch_or_round = int(last_ckpt.get("round" if is_fedavg else "epoch", last_ckpt.get("epoch", -1)))
        if best_epoch_or_round > last_epoch_or_round and last_epoch_or_round >= 0:
            raise ValueError(
                f"Provenance mismatch: checkpoint_best epoch/round ({best_epoch_or_round}) "
                f"cannot exceed checkpoint_last epoch/round ({last_epoch_or_round})."
            )

        # Score consistency: checkpoint_best score must match checkpoint_last recorded best score
        last_recorded_best = float(last_ckpt.get("best_val_f1_raw", last_ckpt.get("best_val_f1", -1.0)))
        if abs(best_val_f1_raw - last_recorded_best) > 1e-4 and last_recorded_best >= 0:
            raise ValueError(
                f"Score consistency mismatch: checkpoint_best score ({best_val_f1_raw:.4f}) differs from "
                f"checkpoint_last recorded best score ({last_recorded_best:.4f})."
            )

    # 4. Check if target_best_path already existed and belongs to a different trajectory
    if target_best_exists and source_best_exists and target_best_path != source_best_path:
        existing_target_ckpt = torch.load(target_best_path, map_location="cpu", weights_only=False)
        target_ep = int(existing_target_ckpt.get("round" if is_fedavg else "epoch", existing_target_ckpt.get("epoch", -1)))
        target_score = float(existing_target_ckpt.get("best_val_f1_raw", existing_target_ckpt.get("best_val_f1", -1.0)))
        if target_ep != best_epoch_or_round or abs(target_score - best_val_f1_raw) > 1e-4:
            raise ValueError(
                f"Pre-existing target checkpoint_best.pt in '{out_dir}' belongs to a different run/trajectory "
                f"(epoch={target_ep}, score={target_score:.4f}) than resume source (epoch={best_epoch_or_round}, score={best_val_f1_raw:.4f})."
            )

    # 5. ONLY AFTER ALL VALIDATION PASSES: import full verified transaction to output_dir if needed
    if needs_copy:
        out_dir.mkdir(parents=True, exist_ok=True)
        # 1. Import best checkpoint
        temp_target = out_dir / f".tmp_best_import_{uuid.uuid4().hex[:8]}.pt"
        shutil.copy2(source_best_path, temp_target)
        copied_sha = compute_bytes_sha256(temp_target)
        source_sha = compute_bytes_sha256(source_best_path)
        if copied_sha != source_sha:
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Integrity verification failed during checkpoint_best.pt import into {out_dir}")
        try:
            os.replace(temp_target, target_best_path)
        except OSError:
            shutil.copy2(temp_target, target_best_path)
            try:
                temp_target.unlink()
            except OSError:
                pass

        # 2. Also import last checkpoint if present
        target_last_path = out_dir / "checkpoint_last.pt"
        if resume_last.is_file() and target_last_path != resume_last:
            temp_last = out_dir / f".tmp_last_import_{uuid.uuid4().hex[:8]}.pt"
            shutil.copy2(resume_last, temp_last)
            try:
                os.replace(temp_last, target_last_path)
            except OSError:
                shutil.copy2(temp_last, target_last_path)
                try:
                    temp_last.unlink()
                except OSError:
                    pass

        # 3. Locate source root directory containing commit_manifest.json or checkpoints/
        source_root = None
        for cand_root in [resume_last.parent] + list(resume_last.parents):
            if cand_root and (cand_root / "commit_manifest.json").is_file():
                source_root = cand_root
                break

        if source_root is not None and (source_root / "commit_manifest.json").is_file():
            source_manifest_path = source_root / "commit_manifest.json"
            try:
                manifest_data = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest_data = {}

            # Copy checkpoints/ generation directory if referenced in manifest
            last_rel = manifest_data.get("last_checkpoint_path")
            best_rel = manifest_data.get("best_checkpoint_path")
            if last_rel:
                source_last_blob = source_root / last_rel
                dest_last_blob = out_dir / last_rel
                if source_last_blob.is_file():
                    dest_last_blob.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_last_blob, dest_last_blob)
            if best_rel:
                source_best_blob = source_root / best_rel
                dest_best_blob = out_dir / best_rel
                if source_best_blob.is_file():
                    dest_best_blob.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_best_blob, dest_best_blob)

            # Copy commit_marker.json in generation dirs
            for rel_path_item in [last_rel, best_rel]:
                if rel_path_item:
                    src_marker = (source_root / rel_path_item).parent / "commit_marker.json"
                    if src_marker.is_file():
                        dest_marker = (out_dir / rel_path_item).parent / "commit_marker.json"
                        dest_marker.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_marker, dest_marker)

            # Copy commit_manifest.json
            shutil.copy2(source_manifest_path, out_dir / "commit_manifest.json")

            # Copy resolved_config.json if present
            if (source_root / "resolved_config.json").is_file():
                shutil.copy2(source_root / "resolved_config.json", out_dir / "resolved_config.json")
            if (source_root / "progress.json").is_file():
                shutil.copy2(source_root / "progress.json", out_dir / "progress.json")
        else:
            # Synthetic or legacy source: construct a valid committed generation transaction in out_dir
            last_epoch = int(last_ckpt.get("round" if is_fedavg else "epoch", last_ckpt.get("epoch", best_epoch_or_round))) if resume_last.is_file() else best_epoch_or_round
            step = last_epoch if last_epoch >= 0 else 1
            best_step = best_epoch_or_round if best_epoch_or_round >= 0 else step
            gen_name = f"gen_{step:06d}"
            gen_dir = out_dir / "checkpoints" / gen_name
            gen_dir.mkdir(parents=True, exist_ok=True)

            gen_last = gen_dir / "checkpoint_last.pt"
            shutil.copy2(target_last_path if target_last_path.is_file() else target_best_path, gen_last)

            if step == best_step:
                gen_best = gen_dir / "checkpoint_best.pt"
                shutil.copy2(target_best_path, gen_best)
                best_rel = f"checkpoints/{gen_name}/checkpoint_best.pt"
            else:
                best_gen_name = f"gen_{best_step:06d}"
                best_gen_dir = out_dir / "checkpoints" / best_gen_name
                best_gen_dir.mkdir(parents=True, exist_ok=True)
                gen_best = best_gen_dir / "checkpoint_best.pt"
                shutil.copy2(target_best_path, gen_best)
                best_rel = f"checkpoints/{best_gen_name}/checkpoint_best.pt"

            last_sha = compute_bytes_sha256(gen_last)
            best_sha = compute_bytes_sha256(gen_best)

            new_manifest = {
                "status": "COMMITTED",
                "generation": step,
                "gen_name": gen_name,
                "is_best": (step == best_step),
                "is_fedavg": is_fedavg,
                "last_checkpoint_path": f"checkpoints/{gen_name}/checkpoint_last.pt",
                "last_checkpoint_sha256": last_sha,
                "best_checkpoint_path": best_rel,
                "best_checkpoint_sha256": best_sha,
                "best_score": best_val_f1_raw,
                "best_generation": best_step,
                "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (gen_dir / "commit_marker.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
            (out_dir / "commit_manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
            (gen_dir / "meta.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

        logger.info("Imported verified transaction from %s into %s", resume_last.parent, out_dir)

    return target_best_path, best_val_f1_raw, best_epoch_or_round


def check_fresh_overwrite_guard(output_dir: Path, is_resume: bool, overwrite: bool) -> None:
    """Prevent accidentally overwriting existing checkpoints or outputs when starting a fresh run."""
    if not is_resume and output_dir.exists():
        has_last = (output_dir / "checkpoint_last.pt").is_file()
        has_summary = (output_dir / "summary.json").is_file()
        has_best = (output_dir / "checkpoint_best.pt").is_file()
        if (has_last or has_summary or has_best) and not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already contains existing checkpoints or summary.json. "
                "To continue an existing run, pass --resume. To overwrite, pass --overwrite."
            )
