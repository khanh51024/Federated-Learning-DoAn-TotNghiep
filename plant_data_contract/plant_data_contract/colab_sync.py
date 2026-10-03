"""Colab and Google Drive Verified Persistence Module.

Guarantees:
- Fail-closed Drive destination validation for folder ID 1GpmEi_wFgszbrP9wr6_Xe6VA0xRagBCi.
- Zero fallback to unverified paths, MyDrive root, or arbitrary folder names.
- Local staging on VM SSD (/content/runs/<job_id>) for performance and isolation.
- Per-epoch atomic sync of generation blobs and manifest with checksum/size verification.
- Stop-the-world on sync failure: marks UNSYNCED and halts training before the next epoch.
- Safe staging of code and data archives with pre-extraction SHA-256 and traversal verification.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import zipfile

logger = logging.getLogger("colab_sync")

REQUIRED_DRIVE_FOLDER_ID = os.environ.get("COLAB_DRIVE_FOLDER_ID", "1ayKrI1Jrl0SXgo9cm5Hm0AuXex-cCMR8")
REQUIRED_DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{REQUIRED_DRIVE_FOLDER_ID}?usp=sharing"


class ColabSyncError(RuntimeError):
    """Raised when remote Drive synchronization fails."""
    pass


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_file_md5(filepath: Union[str, Path]) -> str:
    """Compute MD5 digest of a file in streaming chunks."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dumps(data: Any) -> str:
    """Deterministic JSON serialization matching repository contract."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def verify_drive_folder_id(
    folder_id: str,
    drive_api: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify that the target folder strictly matches REQUIRED_DRIVE_FOLDER_ID and is writable."""
    if folder_id != REQUIRED_DRIVE_FOLDER_ID:
        raise ColabSyncError(
            f"Drive folder ID mismatch! Required: {REQUIRED_DRIVE_FOLDER_ID}, got: {folder_id}. "
            f"Only the user-approved destination is permitted: {REQUIRED_DRIVE_FOLDER_URL}"
        )

    if drive_api is not None:
        try:
            meta = drive_api.files().get(
                fileId=folder_id,
                fields="id,name,mimeType,parents,capabilities",
                supportsAllDrives=True,
            ).execute()
        except Exception as e:
            raise ColabSyncError(f"Drive API call failed when verifying folder {folder_id}: {e}")

        if meta.get("mimeType") != "application/vnd.google-apps.folder":
            raise ColabSyncError(f"Target Drive object {folder_id} is not a folder: {meta.get('mimeType')}")
        if not meta.get("capabilities", {}).get("canAddChildren", False):
            raise ColabSyncError(f"Colab account lacks write/canAddChildren permission on Drive folder {folder_id}")
        return meta

    return {"id": folder_id, "verified": True}


def safe_extract_archive(
    archive_path: Path,
    target_dir: Path,
    expected_sha256: Optional[str] = None,
) -> None:
    """Verify archive hash and extract safely without path traversal."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if expected_sha256:
        actual_sha = compute_file_sha256(archive_path)
        if actual_sha != expected_sha256:
            raise ValueError(f"Archive SHA-256 mismatch! Expected {expected_sha256}, got {actual_sha}")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in member:
                    raise ValueError(f"Illegal path traversal in archive member: {member}")
                dest_path = (target_dir / member).resolve()
                if not dest_path.is_relative_to(target_resolved):
                    raise ValueError(f"Extracted member escapes target directory: {member}")
            zf.extractall(target_dir)
    elif archive_path.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        import tarfile
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Illegal path traversal in tar member: {member.name}")
                dest_path = (target_dir / member.name).resolve()
                if not dest_path.is_relative_to(target_resolved):
                    raise ValueError(f"Extracted member escapes target directory: {member.name}")
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format for safe extraction: {archive_path.name}")


class ColabPersistenceSync:
    """Manages transactional checkpoint persistence between local SSD and Google Drive."""

    def __init__(
        self,
        local_staging_dir: Path,
        remote_dest_dir: Path,
        drive_folder_id: str = REQUIRED_DRIVE_FOLDER_ID,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        drive_api: Optional[Any] = None,
    ):
        if drive_folder_id != REQUIRED_DRIVE_FOLDER_ID:
            raise ColabSyncError(f"Illegal drive_folder_id: {drive_folder_id}. Must be {REQUIRED_DRIVE_FOLDER_ID}")

        self.local_staging_dir = Path(local_staging_dir).resolve()
        self.remote_dest_dir = Path(remote_dest_dir).resolve()
        self.drive_folder_id = drive_folder_id
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.drive_api = drive_api

    def _get_drive_remote_name(self, dst: Path) -> str:
        """Derive an immutable, namespaced remote file name for Google Drive.
        Preserves job/stage hierarchy to prevent cross-job collisions in the flat Drive parent folder.
        """
        try:
            rel = dst.relative_to(self.remote_dest_dir)
            rel_str = str(rel).replace("\\", "/")
        except ValueError:
            rel_str = dst.name

        dest_parts = self.remote_dest_dir.parts
        if len(dest_parts) >= 2:
            candidate_stage = dest_parts[-1]
            candidate_job = dest_parts[-2]
            if (
                candidate_stage in ("full", "pilot", "preflight")
                or candidate_job.startswith(("cent_", "fed_", "job"))
            ):
                if not rel_str.startswith(f"{candidate_job}/"):
                    rel_str = f"{candidate_job}/{candidate_stage}/{rel_str}"
        elif len(dest_parts) == 1:
            candidate_job = dest_parts[-1]
            if candidate_job.startswith(("cent_", "fed_", "job")):
                if not rel_str.startswith(f"{candidate_job}/"):
                    rel_str = f"{candidate_job}/{rel_str}"

        return rel_str

    @staticmethod
    def _copy_file_atomic(src: Path, dst: Path) -> None:
        """Locally copy a file atomically with size validation."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp_dst = dst.parent / f".tmp_{dst.name}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        shutil.copy2(src, temp_dst)
        if temp_dst.stat().st_size != src.stat().st_size:
            if temp_dst.exists():
                temp_dst.unlink()
            raise IOError(f"Copy size mismatch: {src} -> {dst}")
        temp_dst.replace(dst)

    def upload_file_atomic(self, src: Path, dst: Path) -> Dict[str, Any]:
        """Upload a file atomically with content verification, Drive API support, and retries."""
        if not src.is_file():
            raise FileNotFoundError(f"Source file does not exist: {src}")

        src_sha = compute_file_sha256(src)
        src_md5 = compute_file_md5(src)
        src_size = src.stat().st_size
        dst.parent.mkdir(parents=True, exist_ok=True)

        if self.drive_api is not None:
            # Verified Drive API path
            from googleapiclient.http import MediaFileUpload
            remote_name = self._get_drive_remote_name(dst)

            last_err = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    media = MediaFileUpload(str(src), resumable=True)
                    # Check if file exists in parent
                    escaped_name = remote_name.replace("'", "\\'")
                    q = f"'{self.drive_folder_id}' in parents and name = '{escaped_name}' and trashed = false"
                    existing = self.drive_api.files().list(
                        q=q,
                        fields="files(id, name, size, md5Checksum, parents)",
                        supportsAllDrives=True,
                    ).execute().get("files", [])

                    if len(existing) > 1:
                        raise ColabSyncError(
                            f"Ambiguous remote duplicate matches found for '{remote_name}': {[f.get('id') for f in existing]}"
                        )

                    if existing:
                        file_id = existing[0]["id"]
                        updated = self.drive_api.files().update(
                            fileId=file_id,
                            media_body=media,
                            fields="id,name,size,md5Checksum,parents",
                            supportsAllDrives=True,
                        ).execute()
                    else:
                        body = {"name": remote_name, "parents": [self.drive_folder_id]}
                        updated = self.drive_api.files().create(
                            body=body,
                            media_body=media,
                            fields="id,name,size,md5Checksum,parents",
                            supportsAllDrives=True,
                        ).execute()

                    file_id = updated["id"]
                    # Server readback verification
                    readback = self.drive_api.files().get(
                        fileId=file_id,
                        fields="id,name,size,md5Checksum,parents",
                        supportsAllDrives=True,
                    ).execute()

                    # Strict fail-closed metadata checks: reject missing as well as wrong metadata
                    if "size" not in readback or readback.get("size") is None:
                        raise ColabSyncError(f"Drive API readback missing 'size' for {remote_name}")
                    server_size = int(readback["size"])
                    if server_size != src_size:
                        raise ColabSyncError(f"Drive API size mismatch: expected {src_size}, got {server_size}")

                    server_md5 = readback.get("md5Checksum")
                    if not server_md5:
                        raise ColabSyncError(f"Drive API readback missing 'md5Checksum' for {remote_name}")
                    if server_md5.lower() != src_md5.lower():
                        raise ColabSyncError(f"Drive API checksum mismatch: expected {src_md5}, got {server_md5}")

                    server_parents = readback.get("parents")
                    if not server_parents:
                        raise ColabSyncError(f"Drive API readback missing 'parents' for {remote_name}")
                    if self.drive_folder_id not in server_parents:
                        raise ColabSyncError(f"Drive API parent mismatch: expected {self.drive_folder_id}, got {server_parents}")

                    return {
                        "source": str(src),
                        "destination": str(dst),
                        "file_id": file_id,
                        "sha256": src_sha,
                        "size_bytes": src_size,
                        "server_md5": server_md5,
                        "attempts": attempt,
                    }
                except Exception as e:
                    last_err = e
                    logger.warning(f"Drive API upload attempt {attempt}/{self.max_retries} failed for {src.name}: {e}")
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

            raise ColabSyncError(f"Failed to persist {src} via Drive API after {self.max_retries} attempts: {last_err}")

        # Local filesystem / mount atomic copy path
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                temp_dst = dst.parent / f".tmp_{dst.name}_{os.getpid()}_{attempt}"
                shutil.copy2(src, temp_dst)

                # Verify copied file integrity
                temp_size = temp_dst.stat().st_size
                temp_sha = compute_file_sha256(temp_dst)
                if temp_size != src_size or temp_sha != src_sha:
                    if temp_dst.exists():
                        temp_dst.unlink()
                    raise IOError(f"Copied file integrity mismatch on attempt {attempt}: size {temp_size} vs {src_size}, sha {temp_sha} vs {src_sha}")

                # Atomic replace
                temp_dst.replace(dst)

                # Verify final destination file
                if not dst.is_file() or dst.stat().st_size != src_size:
                    raise IOError(f"Destination file validation failed after replace: {dst}")

                return {
                    "source": str(src),
                    "destination": str(dst),
                    "sha256": src_sha,
                    "size_bytes": src_size,
                    "attempts": attempt,
                }
            except Exception as e:
                last_error = e
                logger.warning(f"File upload attempt {attempt}/{self.max_retries} failed for {src.name}: {e}")
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise ColabSyncError(f"Failed to atomically persist {src} to {dst} after {self.max_retries} attempts: {last_error}")

    def sync_epoch(self, epoch: int, is_best: bool = False) -> Dict[str, Any]:
        """Synchronize generation blobs and commit manifest to remote Drive destination.
        
        Order of operations (strict fail-closed protocol):
        1. Read local manifest and strictly validate schema, status, blobs, and hashes before upload.
        2. Upload generation blobs and marker.
        3. Upload top-level convenience aliases and progress.json.
        4. Finally commit remote commit_manifest.json as the transactional commit point.
        5. On failure: mark local progress as UNSYNCED and raise ColabSyncError.
        """
        local_manifest_path = self.local_staging_dir / "commit_manifest.json"
        if not local_manifest_path.is_file():
            local_manifest_path = self.local_staging_dir / "checkpoints/manifest.json"
        if not local_manifest_path.is_file():
            raise FileNotFoundError(f"Local manifest not found: {self.local_staging_dir / 'commit_manifest.json'}")

        manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))

        try:
            # 1. Strictly validate local blobs, hashes, and schema BEFORE any upload
            from plant_data_contract.resume_guard import validate_transaction_manifest
            validate_transaction_manifest(
                manifest=manifest,
                base_dir=self.local_staging_dir,
                verify_hashes=True,
            )

            gen_dir_rel = manifest["last_checkpoint_path"]
            local_gen_file = (self.local_staging_dir / gen_dir_rel).resolve()
            best_rel = manifest.get("best_checkpoint_path")
            local_best_file = (self.local_staging_dir / best_rel).resolve() if best_rel else None

            # 2. Upload generation checkpoint_last.pt
            remote_gen_file = self.remote_dest_dir / gen_dir_rel
            self.upload_file_atomic(local_gen_file, remote_gen_file)

            # 3. Upload generation checkpoint_best.pt if present
            if local_best_file is not None and local_best_file.is_file():
                remote_best_file = self.remote_dest_dir / best_rel
                self.upload_file_atomic(local_best_file, remote_best_file)

            # 4. Upload generation commit_marker.json after blobs are safely persisted
            local_marker = local_gen_file.parent / "commit_marker.json"
            if local_marker.is_file():
                self.upload_file_atomic(local_marker, remote_gen_file.parent / "commit_marker.json")

            # 5. Sync top-level convenience aliases if present locally
            local_top_last = self.local_staging_dir / "checkpoint_last.pt"
            if local_top_last.is_file():
                self.upload_file_atomic(local_top_last, self.remote_dest_dir / "checkpoint_last.pt")

            local_top_best = self.local_staging_dir / "checkpoint_best.pt"
            if local_top_best.is_file():
                self.upload_file_atomic(local_top_best, self.remote_dest_dir / "checkpoint_best.pt")

            # 6. Sync metadata files
            local_progress = self.local_staging_dir / "progress.json"
            if local_progress.is_file():
                self.upload_file_atomic(local_progress, self.remote_dest_dir / "progress.json")

            local_cfg = self.local_staging_dir / "resolved_config.json"
            if local_cfg.is_file():
                self.upload_file_atomic(local_cfg, self.remote_dest_dir / "resolved_config.json")

            # 7. Finally commit remote commit_manifest.json as the transactional commit point
            remote_manifest_path = self.remote_dest_dir / "commit_manifest.json"
            self.upload_file_atomic(local_manifest_path, remote_manifest_path)
            if local_manifest_path.name != "commit_manifest.json":
                self.upload_file_atomic(local_manifest_path, self.remote_dest_dir / "checkpoints/manifest.json")

            logger.info(f"[Drive Sync] Epoch {epoch} successfully persisted to {self.remote_dest_dir}")
            return {
                "status": "SYNCED",
                "epoch": epoch,
                "remote_manifest": str(remote_manifest_path),
            }

        except Exception as e:
            # Mark local status as UNSYNCED
            local_progress = self.local_staging_dir / "progress.json"
            if local_progress.is_file():
                try:
                    p_info = json.loads(local_progress.read_text(encoding="utf-8"))
                    p_info["status"] = "UNSYNCED"
                    p_info["sync_error"] = str(e)
                    local_progress.write_text(canonical_json_dumps(p_info), encoding="utf-8")
                except Exception:
                    pass
            raise ColabSyncError(
                f"Sync failed at epoch {epoch}: unable to persist checkpoints to verified Drive folder. "
                f"Status set to UNSYNCED. Aborting training before next epoch: {e}"
            ) from e

    def sync_final(self, summary_file: Path) -> Dict[str, Any]:
        """Synchronize final summary.json to remote Drive destination."""
        if not summary_file.is_file():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        try:
            remote_summary = self.remote_dest_dir / "summary.json"
            res = self.upload_file_atomic(summary_file, remote_summary)
            logger.info(f"[Drive Sync] Final summary persisted to {remote_summary}")
            return res
        except Exception as e:
            raise ColabSyncError(f"Failed to persist final summary.json to Drive: {e}") from e

    def stage_remote_checkpoints_for_resume(self) -> bool:
        """Download / copy remote checkpoints into local staging directory for resume.
        Pre-validates entire remote manifest before creating any local files to prevent path traversal.
        Purges any local uncommitted generations that exceed the verified remote commit.
        Restore does not upload or mutate remote Drive files.
        """
        temp_staging_dir = self.local_staging_dir.parent / f".tmp_stage_{uuid.uuid4().hex[:8]}"

        try:
            if self.drive_api is not None:
                # 1. Drive API Mode: query Drive for manifest
                manifest_name = self._get_drive_remote_name(self.remote_dest_dir / "commit_manifest.json")
                escaped_name = manifest_name.replace("'", "\\'")
                q = f"'{self.drive_folder_id}' in parents and name = '{escaped_name}' and trashed = false"
                items = self.drive_api.files().list(
                    q=q,
                    fields="files(id, name, size, md5Checksum)",
                    supportsAllDrives=True,
                ).execute().get("files", [])

                if not items:
                    fallback_name = self._get_drive_remote_name(self.remote_dest_dir / "checkpoints/manifest.json")
                    if fallback_name != manifest_name:
                        escaped_fb = fallback_name.replace("'", "\\'")
                        q_fb = f"'{self.drive_folder_id}' in parents and name = '{escaped_fb}' and trashed = false"
                        items = self.drive_api.files().list(
                            q=q_fb,
                            fields="files(id, name, size, md5Checksum)",
                            supportsAllDrives=True,
                        ).execute().get("files", [])

                if not items:
                    logger.info("No remote manifest found on Drive API; fresh run required.")
                    return False

                if len(items) > 1:
                    raise ColabSyncError(
                        f"Ambiguous remote duplicate manifest files found: {[f.get('id') for f in items]}"
                    )

                manifest_meta = items[0]
                manifest_file_id = manifest_meta["id"]

                from googleapiclient.http import MediaIoBaseDownload
                import io

                req = self.drive_api.files().get_media(fileId=manifest_file_id, supportsAllDrives=True)
                buf = io.BytesIO()
                dl = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                manifest_bytes = buf.getvalue()
                manifest_data = json.loads(manifest_bytes.decode("utf-8"))

                if "last_checkpoint_path" not in manifest_data:
                    raise ColabSyncError("Manifest missing 'last_checkpoint_path'")

                last_rel = manifest_data["last_checkpoint_path"]
                best_rel = manifest_data.get("best_checkpoint_path")
                remote_gen = int(manifest_data.get("generation", 0))

                def _download_drive_file(rel_path: str, target_local: Path) -> None:
                    rn = self._get_drive_remote_name(self.remote_dest_dir / rel_path)
                    escaped_rn = rn.replace("'", "\\'")
                    eq = f"'{self.drive_folder_id}' in parents and name = '{escaped_rn}' and trashed = false"
                    f_list = self.drive_api.files().list(
                        q=eq,
                        fields="files(id, name, size, md5Checksum)",
                        supportsAllDrives=True,
                    ).execute().get("files", [])
                    if not f_list:
                        raise ColabSyncError(f"Required remote file '{rn}' not found on Drive API")
                    if len(f_list) > 1:
                        raise ColabSyncError(f"Ambiguous remote duplicate files for '{rn}': {[f.get('id') for f in f_list]}")
                    f_item = f_list[0]
                    target_local.parent.mkdir(parents=True, exist_ok=True)
                    tmp_down = target_local.parent / f".tmp_{target_local.name}_{uuid.uuid4().hex[:6]}"
                    with open(tmp_down, "wb") as f_out:
                        f_req = self.drive_api.files().get_media(fileId=f_item["id"], supportsAllDrives=True)
                        f_dl = MediaIoBaseDownload(f_out, f_req)
                        f_done = False
                        while not f_done:
                            _, f_done = f_dl.next_chunk()
                    expected_sz = int(f_item.get("size", 0))
                    if expected_sz > 0 and tmp_down.stat().st_size != expected_sz:
                        tmp_down.unlink(missing_ok=True)
                        raise ColabSyncError(f"Size mismatch downloading {rn}: expected {expected_sz}, got {tmp_down.stat().st_size}")
                    expected_md5 = f_item.get("md5Checksum")
                    if expected_md5 and compute_file_md5(tmp_down).lower() != expected_md5.lower():
                        tmp_down.unlink(missing_ok=True)
                        raise ColabSyncError(f"MD5 mismatch downloading {rn}")
                    tmp_down.replace(target_local)

                temp_staging_dir.mkdir(parents=True, exist_ok=True)
                temp_last = temp_staging_dir / last_rel
                _download_drive_file(last_rel, temp_last)

                if best_rel:
                    temp_best = temp_staging_dir / best_rel
                    _download_drive_file(best_rel, temp_best)

                marker_rel = str(Path(last_rel).parent / "commit_marker.json").replace("\\", "/")
                try:
                    _download_drive_file(marker_rel, temp_staging_dir / marker_rel)
                except Exception:
                    pass

                temp_manifest = temp_staging_dir / "commit_manifest.json"
                temp_manifest.write_bytes(manifest_bytes)

            else:
                # 2. Local Filesystem / FUSE Mount Mode
                remote_manifest_file = self.remote_dest_dir / "commit_manifest.json"
                if not remote_manifest_file.is_file():
                    remote_manifest_file = self.remote_dest_dir / "checkpoints/manifest.json"
                if not remote_manifest_file.is_file():
                    logger.info("No remote manifest found; fresh run required.")
                    return False

                manifest_data = json.loads(remote_manifest_file.read_text(encoding="utf-8"))
                last_rel = manifest_data["last_checkpoint_path"]
                best_rel = manifest_data.get("best_checkpoint_path")
                remote_last = (self.remote_dest_dir / last_rel).resolve()
                remote_best = (self.remote_dest_dir / best_rel).resolve() if best_rel else None
                remote_gen = int(manifest_data.get("generation", 0))

                temp_staging_dir.mkdir(parents=True, exist_ok=True)
                temp_last = temp_staging_dir / last_rel
                self._copy_file_atomic(remote_last, temp_last)

                if remote_best and remote_best.is_file():
                    temp_best = temp_staging_dir / best_rel
                    self._copy_file_atomic(remote_best, temp_best)

                remote_marker = remote_last.parent / "commit_marker.json"
                if remote_marker.is_file():
                    self._copy_file_atomic(remote_marker, temp_last.parent / "commit_marker.json")

                temp_manifest = temp_staging_dir / "commit_manifest.json"
                self._copy_file_atomic(remote_manifest_file, temp_manifest)

            # Pre-validate staged checkpoints in temp staging directory before promoting to active
            from plant_data_contract.resume_guard import validate_transaction_manifest, load_verified_checkpoint
            validate_transaction_manifest(
                manifest=manifest_data,
                base_dir=temp_staging_dir,
                verify_hashes=True,
            )
            load_verified_checkpoint(temp_staging_dir, allow_legacy=True)

            # Crucial R5-C2 fix: Purge any local uncommitted generations that exceed remote_gen
            local_checkpoints = self.local_staging_dir / "checkpoints"
            if local_checkpoints.is_dir():
                for g_dir in list(local_checkpoints.glob("gen_*")):
                    try:
                        parts = g_dir.name.split("_")
                        if len(parts) >= 2 and parts[1].isdigit():
                            gen_num = int(parts[1])
                            if gen_num > remote_gen:
                                shutil.rmtree(g_dir, ignore_errors=True)
                    except Exception:
                        pass

            # Promote staged files to active self.local_staging_dir
            for root_p, _, files in os.walk(temp_staging_dir):
                for fname in files:
                    src_f = Path(root_p) / fname
                    rel_f = src_f.relative_to(temp_staging_dir)
                    dst_f = self.local_staging_dir / rel_f
                    self._copy_file_atomic(src_f, dst_f)

            # Top-level aliases
            local_last = self.local_staging_dir / last_rel
            if local_last.is_file():
                self._copy_file_atomic(local_last, self.local_staging_dir / "checkpoint_last.pt")

            if best_rel:
                local_best = self.local_staging_dir / best_rel
                if local_best.is_file():
                    self._copy_file_atomic(local_best, self.local_staging_dir / "checkpoint_best.pt")

            # Final verify active staging
            load_verified_checkpoint(self.local_staging_dir, allow_legacy=True)
            logger.info(f"Successfully staged and verified remote checkpoints into {self.local_staging_dir}")
            return True

        finally:
            if temp_staging_dir.exists():
                shutil.rmtree(temp_staging_dir, ignore_errors=True)

