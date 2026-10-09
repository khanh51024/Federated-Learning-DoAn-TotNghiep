from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset" / "dataset" / "PlantVillage-Dataset" / "raw" / "color"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
SPLITS_DIR = PROJECT_ROOT / "experiments" / "splits"
PARTITIONS_DIR = PROJECT_ROOT / "experiments" / "partitions"

SEED = 42
NUM_CLASSES = 38
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_CLIENTS = 5


def partition_path(alpha: float, seed: int, clients: int) -> Path:
	alpha_label = f"{alpha:g}".replace(".", "_")
	return PARTITIONS_DIR / f"plantvillage_alpha{alpha_label}_clients{clients}_seed{seed}.json"

