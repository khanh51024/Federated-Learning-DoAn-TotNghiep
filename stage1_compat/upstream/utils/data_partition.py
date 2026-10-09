"""Chia train set theo Dirichlet để mô phỏng non-IID."""

import json
from pathlib import Path

import numpy as np


def dirichlet_partition(labels, num_clients: int, alpha: float, seed: int = 42) -> list[list[int]]:
    if num_clients < 1 or alpha <= 0:
        raise ValueError("num_clients phải >= 1 và alpha phải > 0.")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in np.unique(labels):
        indices = np.where(labels == class_id)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client_id, shard in enumerate(np.split(indices, cuts)):
            client_indices[client_id].extend(shard.tolist())

    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def partition_summary(labels, client_indices: list[list[int]]) -> list[dict]:
    labels = np.asarray(labels)
    class_count = int(labels.max()) + 1
    result = []
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        counts = np.bincount(client_labels, minlength=class_count)
        probabilities = counts[counts > 0] / max(len(indices), 1)
        result.append({
            "client_id": client_id,
            "samples": len(indices),
            "classes_present": int(len(np.unique(client_labels))),
            "largest_class_fraction": float(probabilities.max()) if len(probabilities) else 0.0,
            "label_entropy": float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0,
        })
    return result


def save_partition(path: Path | str, labels, client_indices: list[list[int]], alpha: float, seed: int) -> None:
    """Persist a partition so every baseline reuses the exact same clients."""
    labels = np.asarray(labels)
    validate_partition(client_indices, len(labels))
    payload = {
        "seed": seed,
        "alpha": alpha,
        "num_clients": len(client_indices),
        "total_samples": len(labels),
        "partitions": client_indices,
        "summary": partition_summary(labels, client_indices),
        "class_distribution": class_distribution(labels, client_indices),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_partition(path: Path | str, labels, expected_clients: int | None = None) -> list[list[int]]:
    """Load and validate a previously persisted partition."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = np.asarray(labels)
    partitions = payload["partitions"]
    if payload["total_samples"] != len(labels):
        raise ValueError("Partition đã lưu không khớp số lượng mẫu train hiện tại.")
    if expected_clients is not None and len(partitions) != expected_clients:
        raise ValueError("Partition đã lưu không khớp số client yêu cầu.")
    validate_partition(partitions, len(labels))
    return partitions


def validate_partition(client_indices: list[list[int]], total_samples: int) -> None:
    """Ensure every training example belongs to exactly one client."""
    flattened = [index for indices in client_indices for index in indices]
    if len(flattened) != total_samples or len(set(flattened)) != total_samples:
        raise ValueError("Client partition phải bao phủ mỗi mẫu train đúng một lần.")


def class_distribution(labels, client_indices: list[list[int]]) -> list[list[int]]:
    """Return client-by-class counts for reproducible non-IID reporting."""
    labels = np.asarray(labels)
    class_count = int(labels.max()) + 1
    return [np.bincount(labels[indices], minlength=class_count).tolist() for indices in client_indices]
