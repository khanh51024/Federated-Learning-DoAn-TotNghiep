"""Schema definitions for Canonical PlantVillage and PlantDoc Release."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SourceDomain(str, Enum):
    PLANTVILLAGE = "plantvillage"
    PLANTDOC = "plantdoc"


class MappingStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    AMBIGUOUS_ALIAS = "ambiguous_alias"
    ASSUMED_HEALTHY = "assumed_healthy"
    UNMAPPED = "unmapped_in_classification_yaml"
    OUT_OF_SCOPE = "unknown_or_out_of_scope"


class QuarantineStatus(str, Enum):
    CLEAN = "clean"
    LABEL_CONFLICT = "quarantined_label_conflict"
    TEST_OVERLAP = "quarantined_test_overlap"
    UNRESOLVED_NEAR_DUP = "quarantined_unresolved_near_duplicate"


@dataclass
class ImageRecord:
    """Record for an original scene or specimen image."""
    image_id: str
    source_dataset: str
    source_domain: str
    relative_path: str
    original_split: str
    width: int
    height: int
    byte_sha256: str
    pixel_sha256: str
    decode_status: str = "valid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationRecord:
    """Record for a bounding box annotation on an image."""
    annotation_id: str
    image_id: str
    bbox_xyxy: List[float]
    source_label: str
    mapping_status: str
    target_class: Optional[str]
    class_id: Optional[int]
    detection_eligible: bool = True
    supervised_eligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationSample:
    """Input sample record for image classification."""
    sample_id: str
    image_id: str
    annotation_id: Optional[str]
    group_id: str
    parent_scene_id: str
    relative_path: str
    original_path: str
    source_dataset: str
    source_domain: str
    class_id: int
    target_class: str
    mapping_status: str
    supervised_eligible: bool
    split: str
    crop_policy_version: str
    byte_sha256: str
    quarantine_status: str = "clean"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReleaseManifest:
    """Frozen release metadata and integrity hashes."""
    release_name: str
    schema_version: str
    taxonomy_sha256: str
    mapping_sha256: str
    group_graph_sha256: str
    split_sha256: str
    source_inventory_sha256: str
    crop_policy: str
    preprocessing_version: str
    num_classes: int
    counts: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
