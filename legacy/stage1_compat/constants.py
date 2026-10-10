"""Hằng số cố định cho profile tương thích GĐ1 (stage1_compat)."""

from pathlib import Path

# Upstream Git provenance
UPSTREAM_REPO = "khanh51024/Federated-Learning-DoAn-TotNghiep"
UPSTREAM_COMMIT = "79a537060322d3e83e0ebef2d394e2a39e0aa959"
UPSTREAM_BRANCH = "Federated-Learning-DoAn-TotNghiep-Stage-1"

# Hyperparameters & Protocol đã khóa
SEED = 42
DEFAULT_ALPHAS = [100.0, 1.0]
NUM_CLASSES = 38
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_CLIENTS = 5
DEFAULT_ROUNDS = 10
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
OPTIMIZER_NAME = "AdamW"
MODEL_NAME = "MobileNetV3-Small"

# Split protocol
SPLIT_TRAIN_RATIO = 0.72
SPLIT_VAL_RATIO = 0.08
SPLIT_TEST_RATIO = 0.20
TOTAL_DATASET_SAMPLES = 54305
TRAIN_SAMPLES = 39084
VAL_SAMPLES = 4328
TEST_SAMPLES = 10893

# Quota & Budget
RESERVE_HOURS = 3.0
MAX_SESSION_HOURS = 8.0
STOP_BEFORE_MINUTES = 15.0
SAFETY_MARGIN = 1.5

# Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "PlantVillage-Dataset" / "raw" / "color"
UPSTREAM_DIR = PACKAGE_ROOT / "upstream"
UPSTREAM_MANIFEST_PATH = PACKAGE_ROOT / "upstream_manifest.json"
DATASET_IDENTITY_PATH = PACKAGE_ROOT / "dataset_identity.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "output" / "fedavg-stage1-antigravity-work"
