"""
Data partitioning and loading package for the PlantVillage Federated Learning
simulation (GĐ2 -- mô phỏng non-IID).

Public surface
--------------
Partitioning : DatasetPartitioner, partition, group_samples, attach_group_ids
Loading      : PlantVillageDataset, DataLoaderSimple, build_loader
FedAvg       : FedAvgPartition
Transforms   : get_default_transform, get_client_transform, get_mobilenetv3_transforms
Metrics      : calculate_partition_metrics, compute_client_class_matrix,
               audit_group_integrity, audit_sample_integrity
"""

from .dataset import (
    TORCH_AVAILABLE,
    DataLoaderSimple,
    PlantVillageDataset,
    build_loader,
    load_manifest,
    quick_stats,
    sample_rng_seed,
)
from .dirichlet_split import (
    SCENARIOS,
    partition,
    partition_iid,
    partition_label_quantity_skew,
    partition_label_skew,
    partition_quantity_skew,
)
from .fedavg import FedAvgPartition
from .leaf_groups import (
    attach_group_ids,
    group_samples,
    load_leaf_map,
    resolve_group_id,
)
from .metrics import (
    audit_group_integrity,
    audit_sample_integrity,
    calculate_partition_metrics,
    compute_client_class_matrix,
)
from .partitioner import MANIFEST_FIELDS, DatasetPartitioner
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILENETV3_IMAGE_SIZE,
    BaseTransform,
    ClientDomainTransform,
    ClientFeatureProfile,
    create_deterministic_client_profile,
    get_client_transform,
    get_default_transform,
    get_mobilenetv3_transforms,
    profiles_from_config,
)

__all__ = [
    "TORCH_AVAILABLE",
    "DataLoaderSimple",
    "PlantVillageDataset",
    "build_loader",
    "load_manifest",
    "quick_stats",
    "sample_rng_seed",
    "SCENARIOS",
    "partition",
    "partition_iid",
    "partition_label_skew",
    "partition_quantity_skew",
    "partition_label_quantity_skew",
    "FedAvgPartition",
    "attach_group_ids",
    "group_samples",
    "load_leaf_map",
    "resolve_group_id",
    "audit_group_integrity",
    "audit_sample_integrity",
    "calculate_partition_metrics",
    "compute_client_class_matrix",
    "MANIFEST_FIELDS",
    "DatasetPartitioner",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MOBILENETV3_IMAGE_SIZE",
    "BaseTransform",
    "ClientDomainTransform",
    "ClientFeatureProfile",
    "create_deterministic_client_profile",
    "get_client_transform",
    "get_default_transform",
    "get_mobilenetv3_transforms",
    "profiles_from_config",
]
