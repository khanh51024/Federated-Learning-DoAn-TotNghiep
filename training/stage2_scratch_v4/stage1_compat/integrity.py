"""Atomic and portable artifact primitives (no training imports)."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def finite(value, name, minimum=0, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))


def read_json(path):
    def reject(value):
        raise ValueError(f"Invalid JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def validate_indices(groups, total):
    flat = [i for group in groups for i in group]
    if (any(type(i) is not int or not 0 <= i < total for i in flat)
            or len(flat) != total or len(set(flat)) != total):
        raise ValueError("Indices must cover [0,total) exactly once")
