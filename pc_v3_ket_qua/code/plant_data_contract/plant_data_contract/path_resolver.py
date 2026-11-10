"""Safe cross-platform path resolver for dataset paths."""

from pathlib import Path
from typing import Union


class PathResolver:
    """Resolves relative dataset paths safely against a canonical dataset root."""

    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {self.dataset_root}")

    def resolve(self, relative_path: Union[str, Path], must_exist: bool = False) -> Path:
        """Resolve a relative path, preventing directory traversal escapes."""
        # Normalize slashes
        clean_rel = str(relative_path).replace("\\", "/").lstrip("/")
        resolved = (self.dataset_root / clean_rel).resolve()

        # Strict escape guard
        try:
            resolved.relative_to(self.dataset_root)
        except ValueError:
            raise ValueError(f"Security error: Path '{relative_path}' escapes dataset root '{self.dataset_root}'")

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Image or file does not exist: {resolved}")

        return resolved

    def relative(self, absolute_path: Union[str, Path]) -> str:
        """Convert an absolute path within dataset_root to POSIX relative path."""
        p = Path(absolute_path).resolve()
        try:
            rel = p.relative_to(self.dataset_root)
            return rel.as_posix()
        except ValueError:
            raise ValueError(f"Path '{absolute_path}' is not within dataset root '{self.dataset_root}'")
