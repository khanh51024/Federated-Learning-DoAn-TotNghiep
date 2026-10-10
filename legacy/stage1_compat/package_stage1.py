"""Đóng gói stage1_compat thành file ZIP bàn giao gọn gàng, không chứa ảnh thô hay .venv."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from stage1_compat.constants import (
    DEFAULT_OUTPUT_DIR,
    PACKAGE_ROOT,
    PROJECT_ROOT,
)


def create_stage1_compat_zip(
    output_zip_path: Path | str | None = None,
) -> Path:
    """Tạo file ZIP gọn cho stage1_compat:

    Bao gồm:
      - gd2_federated_learning/stage1_compat/ (toàn bộ code và upstream metadata)
      - gd2_federated_learning/kaggle_fedavg_stage1_compat.ipynb
      - gd2_federated_learning/tests/test_stage1_compat_*.py
      - gd2_federated_learning/requirements.txt
      - gd2_federated_learning/pyproject.toml
    Loại trừ:
      - .venv, __pycache__, .pytest_cache, checkpoint cũ, dataset ảnh thô.
    """
    resolved_zip = (
        Path(output_zip_path).resolve()
        if output_zip_path
        else (DEFAULT_OUTPUT_DIR / "stage1_compat_package.zip").resolve()
    )
    resolved_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        pkg_dest = tmp_dir / "gd2_federated_learning"
        pkg_dest.mkdir(parents=True, exist_ok=True)

        # 1. Copy stage1_compat
        shutil.copytree(
            PACKAGE_ROOT,
            pkg_dest / "stage1_compat",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp", "*.pth"),
        )

        # 2. Copy tests
        test_dest = pkg_dest / "tests"
        test_dest.mkdir(parents=True, exist_ok=True)
        for test_file in (PROJECT_ROOT / "tests").glob("test_stage1_compat_*.py"):
            shutil.copy2(test_file, test_dest / test_file.name)

        # 3. Copy notebook
        nb_src = PROJECT_ROOT / "kaggle_fedavg_stage1_compat.ipynb"
        if nb_src.exists():
            shutil.copy2(nb_src, pkg_dest / nb_src.name)

        # 4. Copy config / requirements
        for req in ["requirements-stage1.txt"]:
            req_src = PROJECT_ROOT / req
            if req_src.exists():
                shutil.copy2(req_src, pkg_dest / req)

        (test_dest / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dest / "README.md").write_text("Run commands from this directory: python -m stage1_compat --help. "
            "Install requirements-stage1.txt in the tested runtime. Preserve entire output on resume. "
            "Historical baselines are reference-only; scientific_stage2_complete=false.\n", encoding="utf-8")
        # 5. Tạo ZIP
        with zipfile.ZipFile(resolved_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in pkg_dest.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(tmp_dir)
                    zf.write(file_path, arcname)

    print(f"Đã tạo ZIP package tại: {resolved_zip} ({resolved_zip.stat().st_size / 1024:.1f} KB)")
    return resolved_zip


if __name__ == "__main__":
    create_stage1_compat_zip()
