"""Internal partition suite acquisition and cryptographic verification tooling (v4 / r10).

Guarantees non-destructive acquisition, strict path containment, collision rejection,
reparse point / symlink immunity on Windows, cross-namespace identity normalization,
and 100% cryptographic integrity matching pinned suite hashes.

Platform Note: Atomic non-overwrite report publishing via os.rename collision rejection
is verified for Windows filesystems. POSIX / Linux native execution is marked NOT_RUN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PINNED_TREE_HASH = "eea84a26962845f64141803c81eee82aa1cc3841a0dacdc7e94c8fcbac772c15"
EXPECTED_FILE_COUNT = 84
EXPECTED_TOTAL_BYTES = 264037275

PINNED_ROOT_FILES: Dict[str, str] = {
    "suite.json": "109d8b4ddc97c2f31d7c07a5a9328266226908b0ccedb63820689a11873f9350",
    "duplicate_review.json": "53bbb477f71e0512535f58d17835e77cd727c65f4354e034ec1761120f0c2f74",
    "visually_verified_pairs.json": "257cbcc1b8a77af1c59ab26eff71e191c0763db0cb6c1b5050e2019500a626f1",
}

PINNED_CONDITION_MANIFESTS: Dict[str, str] = {
    "feature01": "3255e9cade00da36f2f61f7f54a071d7197bac3bdc8a29855d35fdf2517ecf66",
    "feature100": "47497ca66a1b9e039d6a50b2ede5b07ae25df269a3c741be8e327cfa8802fe2b",
    "label01": "96b6538185969974b6d4f2ba0b7565b19cf61f96a8a9ab99c53141e9ace7b889",
    "label1": "951387f8fd3f45712a57908161daefe1a87a9a769cc8eccbc3f720050906bf6a",
    "label100": "bf7df208557b2fdfda5edfb62a5b0e167b84b2e82e53575ca5500d2ffa860417",
    "label_quantity01": "f2b4c43a6d7777630beba652fa1666c9b08b593fd18c6c03a67331dd4e831157",
    "quantity01": "9fd2cacf5f3317b69e5327e0360651d92593f0c677de59fa94cf6ac7299dc0d8",
    "quantity100": "f1e876649d508c34f108dac817f1cbcaa756116db8739bc9072b2f06f87232a9",
}

DOS_DEVICE_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def is_link_or_reparse(path: Path | str) -> bool:
    """Detect if path is a symlink, Windows directory junction, or reparse point.
    
    Fail-closed: raises ValueError on unexpected OSError (e.g. PermissionError),
    and only returns False on valid FileNotFoundError (non-existent leaf).
    """
    p_str = str(path)
    try:
        st = os.lstat(p_str)
    except FileNotFoundError:
        return False
    except OSError as e:
        raise ValueError(f"Cannot verify path attributes for '{p_str}' (fail-closed): {e}")
    if os.path.islink(p_str):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    # 0x400 = stat.FILE_ATTRIBUTE_REPARSE_POINT (includes junctions and symlinks on Windows)
    if attrs and (attrs & 0x400):
        return True
    return False


def normalize_path_identity(p: Path | str) -> Path:
    """Normalize path identity lexically for cross-namespace consistency on Windows and POSIX.
    
    1. Rejects empty paths.
    2. Rejects unsupported Win32 device namespaces and NT namespaces.
    3. Normalizes Windows extended prefix (\\\\?\\X:\\ -> X:\\, extended UNC -> standard UNC).
    4. Rejects unsupported extended namespace formats (e.g. Volume GUIDs).
    5. Rejects path traversal '..' components.
    6. Rejects reserved DOS device names (CON, NUL, COM1, etc.) and trailing dots/spaces in components.
    7. Anchors relative paths to lexical absolute CWD prior to ancestor checks (R8-01).
    8. Computes absolute path and normalized case (on Windows) WITHOUT resolving symlinks or junctions.
    """
    raw_str = str(p)
    if not raw_str.strip():
        raise ValueError("Path string cannot be empty.")

    s = raw_str.replace("/", "\\")

    # Reject Win32 device and NT namespaces
    if s.startswith("\\\\.\\") or s.startswith("//./"):
        raise ValueError(f"Win32 device namespace is not supported: '{raw_str}'")
    if s.startswith("\\??\\") or s.startswith("/??/"):
        raise ValueError(f"NT object manager namespace is not supported: '{raw_str}'")

    # Handle \\?\\ extended prefix on Windows
    if s.startswith("\\\\?\\"):
        rest = s[4:]
        if rest.upper().startswith("UNC\\"):
            s = "\\\\" + rest[4:]
        elif len(rest) >= 2 and rest[1] == ":" and rest[0].isalpha():
            s = rest
        elif rest.startswith("Volume{"):
            raise ValueError(f"Volume GUID namespace is not supported: '{raw_str}'")
        else:
            raise ValueError(f"Unsupported extended namespace format: '{raw_str}'")

    p_obj = Path(s)

    if ".." in p_obj.parts:
        raise ValueError(f"Path traversal component '..' is strictly prohibited: '{raw_str}'")

    for part in p_obj.parts:
        stem = part.rstrip("\\/").upper()
        stem_no_ext = stem.split(".")[0]
        if stem_no_ext in DOS_DEVICE_NAMES or stem in DOS_DEVICE_NAMES:
            raise ValueError(f"Reserved Windows device name '{part}' is prohibited: '{raw_str}'")
        if part.endswith(".") or part.endswith(" "):
            raise ValueError(f"Path component with trailing dot or space '{part}' is prohibited: '{raw_str}'")

    if not p_obj.is_absolute():
        p_obj = Path.cwd() / p_obj

    abs_str = os.path.abspath(str(p_obj))
    norm_case = os.path.normcase(abs_str)
    return Path(norm_case)


def check_path_and_ancestors_for_links(path: Path | str, label: str = "Path") -> Path:
    """Check if the path itself or any of its existing ancestors is a symlink or reparse point.
    
    Anchors relative paths to lexical absolute CWD prior to ancestor checks (R8-01).
    Returns the normalized Path.
    """
    curr = normalize_path_identity(path)
    # Check the path itself if it exists on disk
    if curr.exists() or os.path.lexists(str(curr)):
        if is_link_or_reparse(curr):
            raise ValueError(
                f"{label} '{curr}' is a symlink or reparse point (e.g. junction). "
                "Symlinks and junctions are strictly prohibited."
            )
    # Check all existing ancestors up to the drive root
    for ancestor in curr.parents:
        if ancestor.exists() or os.path.lexists(str(ancestor)):
            if is_link_or_reparse(ancestor):
                raise ValueError(
                    f"{label} ancestor '{ancestor}' is a symlink or reparse point (e.g. junction). "
                    "Symlinks and junctions are strictly prohibited."
                )
    return curr


def check_for_links_in_tree(dir_path: Path, label: str = "Suite directory") -> None:
    """Check root, ancestors, and all descendants within directory tree for symlinks/reparse points."""
    # 1. Check root and all existing ancestors
    check_path_and_ancestors_for_links(dir_path, label=label)

    # 2. Check all children and descendants before descending
    for root, dirs, files in os.walk(str(dir_path), followlinks=False):
        for d in list(dirs):
            child_dir = Path(root) / d
            if is_link_or_reparse(child_dir):
                raise ValueError(
                    f"Reparse point or symlink detected at directory '{child_dir}'. "
                    "Symlinks/junctions are not permitted within suite trees."
                )
        for f in files:
            child_file = Path(root) / f
            if is_link_or_reparse(child_file):
                raise ValueError(
                    f"Reparse point or symlink detected at file '{child_file}'. "
                    "Symlinks/junctions are not permitted within suite trees."
                )


def _normalize_path_lexical(p: Path | str) -> Path:
    """Normalize path lexically into absolute form with normalized case on Windows, without following symlinks."""
    return normalize_path_identity(p)


def is_same_or_descendant(child: Path | str, parent: Path | str) -> bool:
    """Check if child is the same path or a descendant of parent (lexical, non-resolving)."""
    c = normalize_path_identity(child)
    p = normalize_path_identity(parent)
    if c == p:
        return True
    c_str = str(c)
    p_str = str(p)
    sep = os.sep
    p_prefix = p_str if p_str.endswith(sep) else p_str + sep
    return c_str.startswith(p_prefix) or p in c.parents


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(obj: object) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_directory_tree_hash(target_dir: Path) -> Tuple[str, int, int, List[Tuple[str, int, str]]]:
    records: List[Tuple[str, int, str]] = []
    total_bytes = 0
    for root, _, files in os.walk(target_dir):
        for f in files:
            full_path = Path(root) / f
            rel_posix = full_path.relative_to(target_dir).as_posix()
            sz = full_path.stat().st_size
            total_bytes += sz
            sha = compute_file_sha256(full_path)
            records.append((rel_posix, sz, sha))

    records.sort(key=lambda x: x[0])
    manifest_lines = "".join(f"{p}:{sha}\n" for p, _, sha in records)
    tree_hash = hashlib.sha256(manifest_lines.encode("utf-8")).hexdigest()
    return tree_hash, len(records), total_bytes, records


def compute_condition_manifest_hash(cond_dir: Path) -> str:
    files = list(cond_dir.glob("*.csv")) + list((cond_dir / "clients").glob("*.csv"))
    files += [cond_dir / "partition_config.json", cond_dir / "image_content.json"]
    files = [p for p in sorted(files) if p.is_file()]
    entries = {p.relative_to(cond_dir).as_posix(): compute_file_sha256(p) for p in files}
    return canonical_digest(entries)


def validate_paths_pre_mutation(
    source_dir: Optional[Path],
    target_dir: Path,
    json_out: Optional[Path],
    verify_only: bool,
) -> Tuple[Optional[Path], Path, Optional[Path]]:
    """Strict pre-mutation path, collision, containment, and link validation.
    
    Returns tuple of (validated_source, validated_target, validated_json_out).
    """
    # 1. Normalize and check target path and existing ancestors for reparse/symlinks
    val_target = check_path_and_ancestors_for_links(target_dir, label="Target suite path")

    # 2. Normalize and check source path and existing ancestors for reparse/symlinks if specified
    val_source: Optional[Path] = None
    if source_dir is not None:
        val_source = check_path_and_ancestors_for_links(source_dir, label="Source suite path")

    # 3. Mode-specific checks
    if not verify_only:
        # COPY MODE
        if val_source is None:
            raise ValueError("In COPY mode, --source path is required.")

        if not val_source.exists():
            raise FileNotFoundError(f"Source suite directory does not exist: {val_source}")
        if not val_source.is_dir():
            raise ValueError(f"Source suite path is not a directory: {val_source}")
        if not os.access(val_source, os.R_OK):
            raise PermissionError(f"Source suite directory is not readable: {val_source}")

        # Check source tree for reparse points / symlinks
        check_for_links_in_tree(val_source, label="Source suite directory")

        # Target MUST NOT exist (even if empty) in COPY mode
        if val_target.exists():
            raise FileExistsError(
                f"Target directory '{val_target}' already exists. "
                "In COPY mode, target must not exist prior to acquisition."
            )

        # Source and Target must not be the same or overlapping
        if is_same_or_descendant(val_target, val_source):
            raise ValueError(
                f"Target directory '{val_target}' cannot be inside or identical to source '{val_source}'."
            )
        if is_same_or_descendant(val_source, val_target):
            raise ValueError(
                f"Source directory '{val_source}' cannot be inside target '{val_target}'."
            )
    else:
        # VERIFY-ONLY MODE
        if not val_target.exists():
            raise FileNotFoundError(f"Target suite directory does not exist for verification: {val_target}")
        if not val_target.is_dir():
            raise ValueError(f"Target suite path is not a directory: {val_target}")
        if not os.access(val_target, os.R_OK):
            raise PermissionError(f"Target suite directory is not readable: {val_target}")

        # Check target tree for reparse points / symlinks
        check_for_links_in_tree(val_target, label="Target suite directory")

    # 4. Report output path checks (Common to both modes)
    val_json_out: Optional[Path] = None
    if json_out is not None:
        val_json_out = check_path_and_ancestors_for_links(json_out, label="Report output path")

        # Report must NOT be inside target directory
        if is_same_or_descendant(val_json_out, val_target):
            raise ValueError(
                f"Report output path '{json_out}' cannot be inside target suite directory '{target_dir}'."
            )

        # Report must NOT be an ancestor of target directory
        if is_same_or_descendant(val_target, val_json_out):
            raise ValueError(
                f"Report output path '{json_out}' cannot be an ancestor of target directory '{target_dir}'."
            )

        # Report must NOT be inside declared source directory
        if val_source is not None:
            if is_same_or_descendant(val_json_out, val_source):
                raise ValueError(
                    f"Report output path '{json_out}' cannot be inside declared source directory '{source_dir}'."
                )
            if is_same_or_descendant(val_source, val_json_out):
                raise ValueError(
                    f"Report output path '{json_out}' cannot be an ancestor of declared source directory '{source_dir}'."
                )

        # Report file must NOT already exist (prevent check-then-truncate overwrite)
        if val_json_out.exists():
            try:
                st = os.stat(str(val_json_out))
                if st.st_nlink > 1:
                    raise ValueError(
                        f"Report output file '{json_out}' already exists as a hardlink (nlink={st.st_nlink})."
                    )
            except OSError:
                pass
            raise FileExistsError(
                f"Report output file '{json_out}' already exists. Overwriting reports is strictly forbidden."
            )

        # Validate that every existing ancestor of report is a directory (not a regular file)
        for ancestor in val_json_out.parents:
            if ancestor.exists():
                if not ancestor.is_dir():
                    raise ValueError(
                        f"Report destination ancestor '{ancestor}' exists but is a regular file, not a directory."
                    )

        # Report destination's closest existing parent must be writable
        closest = val_json_out.parent
        while not closest.exists() and closest != closest.parent:
            closest = closest.parent
        if closest.exists() and not os.access(closest, os.W_OK):
            raise PermissionError(f"Report parent directory is not writable: {closest}")

    return val_source, val_target, val_json_out


def acquire_suite(
    source_dir: Optional[Path],
    target_dir: Path,
    verify_only: bool = False,
    json_out: Optional[Path] = None,
) -> Dict[str, object]:
    """Acquires and verifies the partition suite into target_dir."""
    # Pre-mutation validation: all paths checked before any I/O
    val_source, val_target, val_json_out = validate_paths_pre_mutation(
        source_dir=source_dir,
        target_dir=target_dir,
        json_out=json_out,
        verify_only=verify_only,
    )

    if not verify_only:
        assert val_source is not None
        print(f"[Acquire] Copying suite non-destructively: {val_source} -> {val_target}")
        try:
            # Must not use dirs_exist_ok=True; target_dir must be created cleanly by copytree
            shutil.copytree(val_source, val_target, symlinks=False)
        except Exception as exc:
            print(
                f"[Acquire] FAILED/PARTIAL: Copy failed mid-stream into '{val_target}': {exc}",
                file=sys.stderr,
            )
            raise

    print(f"[Verify] Auditing integrity of target suite: {val_target}")
    tree_hash, file_count, total_bytes, records = compute_directory_tree_hash(val_target)

    errors: List[str] = []
    if file_count != EXPECTED_FILE_COUNT:
        errors.append(f"File count mismatch: expected {EXPECTED_FILE_COUNT}, got {file_count}")
    if total_bytes != EXPECTED_TOTAL_BYTES:
        errors.append(f"Total uncompressed bytes mismatch: expected {EXPECTED_TOTAL_BYTES}, got {total_bytes}")
    if tree_hash != PINNED_TREE_HASH:
        errors.append(f"Tree fingerprint mismatch: expected {PINNED_TREE_HASH}, got {tree_hash}")

    rec_map = {p: sha for p, _, sha in records}
    for root_f, expected_sha in PINNED_ROOT_FILES.items():
        actual_sha = rec_map.get(root_f)
        if actual_sha != expected_sha:
            errors.append(f"Root file '{root_f}' SHA mismatch: expected {expected_sha}, got {actual_sha}")

    for cond, expected_manifest_sha in PINNED_CONDITION_MANIFESTS.items():
        cond_dir = val_target / cond
        if not cond_dir.is_dir():
            errors.append(f"Condition directory missing: {cond}")
            continue
        actual_manifest_sha = compute_condition_manifest_hash(cond_dir)
        if actual_manifest_sha != expected_manifest_sha:
            errors.append(
                f"Condition '{cond}' manifest SHA mismatch: expected {expected_manifest_sha}, got {actual_manifest_sha}"
            )

    if errors:
        raise ValueError("Acquisition verification failed:\n" + "\n".join(f"- {e}" for e in errors))

    report: Dict[str, object] = {
        "status": "VERIFIED",
        "mode": "VERIFY_ONLY" if verify_only else "COPY",
        "source_dir": str(val_source) if val_source and not verify_only else "VERIFY_ONLY",
        "target_dir": str(val_target),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_fingerprint_sha256": tree_hash,
        "pinned_root_files_verified": list(PINNED_ROOT_FILES.keys()),
        "pinned_conditions_verified": list(PINNED_CONDITION_MANIFESTS.keys()),
    }

    # Write report file atomically: staging outside suite -> flush/sync/close -> publish without overwrite
    if val_json_out is not None:
        report_parent = val_json_out.parent
        try:
            report_parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(
                f"[Acquire] FAILED/PARTIAL: Cannot create report directory '{report_parent}': {exc}",
                file=sys.stderr,
            )
            raise

        staging_path = report_parent / f".tmp_report_{os.getpid()}_{time.time_ns()}.json"
        try:
            with open(staging_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            print(
                f"[Acquire] FAILED/PARTIAL: Failed writing/flushing staging report '{staging_path}': {exc}",
                file=sys.stderr,
            )
            raise

        # Publish staging report to final name without overwrite
        try:
            if val_json_out.exists():
                raise FileExistsError(
                    f"Report output file '{val_json_out}' already exists. Overwriting reports is strictly forbidden."
                )
            os.rename(staging_path, val_json_out)
            print(f"[Acquire] Verification report published exclusively to: {val_json_out}")
        except FileExistsError as exc:
            print(
                f"[Acquire] FAILED/PARTIAL: Destination collision publishing report to '{val_json_out}': {exc}",
                file=sys.stderr,
            )
            raise
        except Exception as exc:
            print(
                f"[Acquire] FAILED/PARTIAL: Failed publishing report to '{val_json_out}': {exc}",
                file=sys.stderr,
            )
            raise

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and verify internal partition suite (non-destructive)."
    )
    parser.add_argument(
        "--source",
        default=r"D:\university\do-an-tot-nghiep\training-data\stage2\partitions_stage2_scratch_v3",
        help="Path to source partition suite (default: internal training-data store)",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Destination directory to receive or verify the acquired suite",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing target directory without copying (read-only audit)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write verification JSON report (must be outside suite/source)",
    )
    args = parser.parse_args()

    source_path = Path(args.source) if args.source else None
    target_path = Path(args.target_dir)
    json_out_path = Path(args.json_out) if args.json_out else None

    try:
        report = acquire_suite(
            source_dir=source_path,
            target_dir=target_path,
            verify_only=args.verify_only,
            json_out=json_out_path,
        )
        # SUCCESS is ONLY printed after verification AND report output have completely succeeded
        print("[Acquire] SUCCESS:", json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"[Acquire] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
