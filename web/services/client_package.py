"""Nút "Tải bộ cài" trên trang #flower: laptop khác tự tải và cài client, không cần chép file.

Gói nằm ở ~/client-package trên máy chủ (ngoài thư mục phát hành nên còn qua các lần deploy), dựng bằng scripts/build_client_package.py.
File cài là khuôn deploy/client_package/CAI_TU_WEB.bat.in điền địa chỉ và mã băm; nó kiểm mã băm trước khi giải nén.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "deploy" / "client_package" / "CAI_TU_WEB.bat.in"
ZIP_NAME = "goi_client.zip"
MANIFEST_NAME = "manifest.json"
SAFE_BASE = re.compile(r"^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?$")     # địa chỉ này được ghi vào file .bat nên không nhận ký tự nào khác
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def package_dir() -> Path:
    return Path.home() / "client-package"


def zip_path() -> Path:
    return package_dir() / ZIP_NAME


def info() -> Optional[dict[str, Any]]:
    """Mã băm và cỡ gói, None nếu chưa có gói hoặc gói không khớp bản kê."""
    try:
        manifest = json.loads((package_dir() / MANIFEST_NAME).read_text(encoding="utf-8"))
        size = zip_path().stat().st_size
    except (OSError, ValueError):
        return None
    sha = manifest.get("sha256") if isinstance(manifest, dict) else None
    if not isinstance(sha, str) or not SHA256.fullmatch(sha) or manifest.get("bytes") != size:
        return None
    return {"sha256": sha, "bytes": size}


def installer(base_url: str, package: dict[str, Any]) -> bytes:
    """File .bat đã điền sẵn. Windows cần xuống dòng CRLF."""
    base = base_url.rstrip("/")
    if not SAFE_BASE.fullmatch(base):
        raise ValueError(f"địa chỉ không hợp lệ: {base!r}")
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in (("@@BASE@@", base), ("@@SHA256@@", package["sha256"]), ("@@SHORT@@", package["sha256"][:12]), ("@@SIZE_MB@@", str(round(package["bytes"] / 1_000_000)))):
        text = text.replace(key, value)
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
