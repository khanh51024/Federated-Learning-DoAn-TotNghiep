"""requirements-train.txt phải đủ cho mọi thư viện ngoài mà scripts/train_v4.py nạp (trực tiếp hoặc qua mã trong repo).

Lý do: máy GPU thuê chỉ cài torch/torchvision cộng tệp này. Lần kiểm gói ở thư mục tách rời dùng venv của repo nên không bắt được việc thiếu `pyyaml`
(`src/common/config.py` nạp yaml); lỗi chỉ lộ khi chạy thử trên máy thuê.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts" / "train_v4.py"
INSTALLED_SEPARATELY = {"torch", "torchvision"}      # cài theo bản CUDA bởi scripts/remote/setup_env.sh
IMPORT_NAME = {"pyyaml": "yaml", "pillow": "PIL"}     # tên gói pip khác tên mô-đun nạp


def local_file(module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] not in {"src", "scripts"}:
        return None
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return base / "__init__.py" if (base / "__init__.py").is_file() else None


def guarded_import(node: ast.AST) -> bool:
    """`try: import x / except ImportError:` — thư viện tùy chọn, mã vẫn chạy khi thiếu."""
    return isinstance(node, ast.Try) and any(
        isinstance(h.type, ast.Name) and h.type.id in {"ImportError", "ModuleNotFoundError"} for h in node.handlers)


def walk(tree: ast.AST, into_functions: bool):
    """Mọi nút của cây; không vào thân hàm khi into_functions=False (nạp trễ, chỉ chạy khi hàm được gọi) và không vào thân `try ... except ImportError`."""
    for node in ast.iter_child_nodes(tree):
        yield node
        if guarded_import(node):
            for part in (*node.handlers, *node.orelse, *node.finalbody):
                yield from walk(part, into_functions)
        elif into_functions or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from walk(node, into_functions)


def runs_in_trainer(path: Path) -> bool:
    return path == ENTRY or ROOT / "src" / "training" in path.parents


def imports_of(path: Path, into_functions: bool = True) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in walk(tree, into_functions):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)    # `from src import registry` nạp mô-đun src.registry
        elif isinstance(node, ast.ImportFrom) and node.level > 0:
            pkg = path.relative_to(ROOT).with_suffix("").parts[:-1]
            base = ".".join(pkg[: len(pkg) - node.level + 1])
            names.add(f"{base}.{node.module}" if node.module else base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def reachable_third_party() -> set[str]:
    seen: set[Path] = set()
    todo = [ENTRY]
    third: set[str] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        # thư viện ngoài chỉ tính khi nạp ngay lúc import mô-đun, hoặc nằm trong mã của chính bộ train (hàm của nó luôn chạy);
        # `flwr` chỉ được nạp trễ trong hàm của đường Flower, bộ train thống nhất không dùng
        eager = imports_of(path, into_functions=runs_in_trainer(path))
        for name in imports_of(path):
            found = local_file(name)
            if found is not None:
                todo.append(found)
                # nạp `src.common.image_io` cũng chạy `src/common/__init__.py` (và `src/__init__.py`)
                parts = name.split(".")
                todo.extend(p for p in (local_file(".".join(parts[:i])) for i in range(1, len(parts))) if p is not None)
            elif name.split(".")[0] not in {"src", "scripts"} and name in eager:
                third.add(name.split(".")[0])
    return {name for name in third if name not in sys.stdlib_module_names}


def required_modules() -> set[str]:
    out = set()
    for line in (ROOT / "requirements-train.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            package = re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower()
            out.add(IMPORT_NAME.get(package, package))
    return out


def test_training_requirements_cover_every_third_party_import():
    missing = reachable_third_party() - required_modules() - INSTALLED_SEPARATELY
    assert not missing, f"scripts/train_v4.py nạp thư viện chưa có trong requirements-train.txt: {sorted(missing)}"


def test_the_scan_finds_the_libraries_that_the_trainer_really_uses():
    found = reachable_third_party()
    assert {"torch", "numpy", "PIL", "yaml"} <= found      # nếu quét không thấy các thư viện hiển nhiên thì bộ quét hỏng
