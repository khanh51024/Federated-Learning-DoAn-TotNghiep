"""Package and manifest integrity verification for Stage 2 Scratch FedAvg-only releases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_package_manifest(
    package_dir: Path | str,
    manifest_path: Path | str | None = None,
    allowlist: Sequence[str] | None = None,
    strict_allowlist: bool = False,
) -> dict:
    """
    Verify the integrity of a package directory against its release manifest.

    Args:
        package_dir: Root directory of extracted package or source tree.
        manifest_path: Path to manifest JSON file. If None, looks for release_manifest_v4.json or release_manifest_v3.json in package_dir.
        allowlist: Optional list of allowed relative paths.
        strict_allowlist: If True, asserts no extra files exist beyond the manifest entries.

    Returns:
        dict containing verified file count, total bytes, manifest hash, and status.
    """
    pkg_dir = Path(package_dir).resolve()
    if not pkg_dir.is_dir():
        raise FileNotFoundError(f"Package directory does not exist or is not a directory: {pkg_dir}")

    if manifest_path is None:
        for candidate in ("release_manifest_v4.json", "release_manifest_v3.json"):
            if (pkg_dir / candidate).is_file():
                manifest_path = pkg_dir / candidate
                break
        if manifest_path is None:
            raise FileNotFoundError(f"No release manifest found in {pkg_dir}")
    else:
        manifest_path = Path(manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse manifest JSON at {manifest_path}: {e}")

    # Validate manifest schema
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")

    if "files" not in manifest or not isinstance(manifest["files"], dict):
        raise ValueError("Manifest missing required 'files' dictionary")

    files_map = manifest["files"]
    if len(files_map) == 0:
        raise ValueError("Manifest 'files' dictionary is empty; at least one file must be verified")

    if "total_files" in manifest and manifest["total_files"] != len(files_map):
        raise ValueError(f"Manifest total_files mismatch: declared {manifest['total_files']} vs {len(files_map)} entries")

    total_bytes = 0
    verified_count = 0

    for rel_path, meta in files_map.items():
        if not isinstance(meta, dict) or "size_bytes" not in meta or "sha256" not in meta:
            raise ValueError(f"Invalid manifest entry schema for '{rel_path}': must have 'size_bytes' and 'sha256'")

        target_file = pkg_dir / rel_path
        if not target_file.is_file():
            raise FileNotFoundError(f"Manifest file missing on disk: {target_file}")

        actual_size = target_file.stat().st_size
        if actual_size != meta["size_bytes"]:
            raise ValueError(
                f"File size mismatch for '{rel_path}': expected {meta['size_bytes']} bytes, found {actual_size} bytes"
            )

        actual_sha = compute_file_sha256(target_file)
        if actual_sha != meta["sha256"]:
            raise ValueError(
                f"File SHA-256 hash mismatch for '{rel_path}': expected {meta['sha256']}, found {actual_sha}"
            )

        total_bytes += actual_size
        verified_count += 1

    if strict_allowlist:
        # Check that no unauthorized files exist in the package dir (excluding common caches)
        ignored_parts = {"__pycache__", ".git", ".pytest_cache", ".venv"}
        for p in pkg_dir.rglob("*"):
            if p.is_file() and not any(part in ignored_parts for part in p.parts):
                rel = p.relative_to(pkg_dir).as_posix()
                if rel != manifest_path.relative_to(pkg_dir).as_posix() and rel not in files_map:
                    raise ValueError(f"Unauthorized file found in package directory: {rel}")

    return {
        "status": "VERIFIED",
        "verified_files": verified_count,
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_sha,
        "protocol": manifest.get("protocol", "unknown"),
        "scope": manifest.get("scope", "unknown"),
    }
