"""Canonical Data Contract for Mixed PlantVillage and PlantDoc Dataset."""

__version__ = "0.1.0"

from plant_data_contract.schema import (
    ImageRecord,
    AnnotationRecord,
    ClassificationSample,
    ReleaseManifest,
    MappingStatus,
    QuarantineStatus,
    SourceDomain,
)
from plant_data_contract.taxonomy import (
    PLANTVILLAGE_38_CLASSES,
    PLANTDOC_DETECTION_29_CLASSES,
    TaxonomyContract,
    load_taxonomy,
)
from plant_data_contract.path_resolver import PathResolver
from plant_data_contract.crop_geometry import crop_bounding_box, extract_leaf_crop
from plant_data_contract.transforms import (
    get_eval_transforms,
    get_train_transforms,
    get_transforms,
    get_transform_metadata,
)
from plant_data_contract.dataset import (
    CanonicalClassificationDataset,
    build_domain_balanced_sampler,
)
from plant_data_contract.models import (
    create_model,
    find_pretrained_weights_file,
    PINNED_PRETRAINED_FILE_SHA256,
    PINNED_PRETRAINED_BASE_STATE_SHA256,
)
from plant_data_contract.integrity import (
    verify_release_integrity,
    verify_manifest_records_contract,
    verify_image_dataset_integrity,
    canonical_json_dumps,
    compute_model_w0_fingerprint,
    compute_file_sha256,
    EXPECTED_RELEASE_MANIFEST_SHA256,
)
from plant_data_contract.resume_guard import (
    validate_config_identity,
    validate_and_restore_rng,
    import_and_validate_best_checkpoint,
    check_fresh_overwrite_guard,
    save_atomic_checkpoint,
    save_crash_safe_checkpoint,
    load_verified_checkpoint,
    load_verified_best_checkpoint,
    compute_bytes_sha256,
    compute_job_protocol_sha256,
)
from plant_data_contract.partitions import verify_client_partitions, verify_partition_directory
from plant_data_contract.colab_sync import (
    ColabPersistenceSync,
    ColabSyncError,
    REQUIRED_DRIVE_FOLDER_ID,
    REQUIRED_DRIVE_FOLDER_URL,
    verify_drive_folder_id,
    safe_extract_archive,
)

__all__ = [
    "ImageRecord",
    "AnnotationRecord",
    "ClassificationSample",
    "ReleaseManifest",
    "MappingStatus",
    "QuarantineStatus",
    "SourceDomain",
    "PLANTVILLAGE_38_CLASSES",
    "PLANTDOC_DETECTION_29_CLASSES",
    "TaxonomyContract",
    "load_taxonomy",
    "PathResolver",
    "crop_bounding_box",
    "extract_leaf_crop",
    "get_eval_transforms",
    "get_train_transforms",
    "get_transforms",
    "get_transform_metadata",
    "CanonicalClassificationDataset",
    "build_domain_balanced_sampler",
    "create_model",
    "find_pretrained_weights_file",
    "PINNED_PRETRAINED_FILE_SHA256",
    "PINNED_PRETRAINED_BASE_STATE_SHA256",
    "verify_release_integrity",
    "verify_manifest_records_contract",
    "verify_image_dataset_integrity",
    "verify_client_partitions",
    "verify_partition_directory",
    "canonical_json_dumps",
    "compute_model_w0_fingerprint",
    "compute_file_sha256",
    "EXPECTED_RELEASE_MANIFEST_SHA256",
    "validate_config_identity",
    "validate_and_restore_rng",
    "import_and_validate_best_checkpoint",
    "check_fresh_overwrite_guard",
    "save_atomic_checkpoint",
    "save_crash_safe_checkpoint",
    "load_verified_checkpoint",
    "load_verified_best_checkpoint",
    "compute_bytes_sha256",
    "compute_job_protocol_sha256",
    "ColabPersistenceSync",
    "ColabSyncError",
    "REQUIRED_DRIVE_FOLDER_ID",
    "REQUIRED_DRIVE_FOLDER_URL",
    "verify_drive_folder_id",
    "safe_extract_archive",
]


