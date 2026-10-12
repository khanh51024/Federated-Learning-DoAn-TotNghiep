"""Tải một lần phông Source Serif 4 (tiêu đề) từ Google Fonts về web/static/fonts/ và cập nhật fonts.css.

Chỉ chạy tay khi dựng bộ phông; giao diện KHÔNG gọi Google Fonts lúc chạy (xem web/static/fonts/README.md).

    python -I scripts/fetch_webfonts.py

Lấy các subset latin, latin-ext, vietnamese; báo lỗi nếu thiếu vietnamese. Thân chữ vẫn là Be Vietnam Pro.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "web" / "static" / "fonts"
FAMILY = "Source Serif 4"
SLUG = "source-serif-4"
SUBSETS = ("latin", "latin-ext", "vietnamese")
# Google trả woff2 + chú thích subset chỉ khi thấy User-Agent của trình duyệt hiện đại.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
SIZE_LIMIT = 90 * 1024
BEGIN, END = f"/* BEGIN {SLUG} */", f"/* END {SLUG} */"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 — URL cố định của Google Fonts
        return response.read()


def css_url(with_opsz: bool) -> str:
    axes = "opsz,wght@8..60,600..700" if with_opsz else "wght@600..700"
    return f"https://fonts.googleapis.com/css2?family=Source+Serif+4:{axes}&display=swap"


def parse_faces(css: str) -> dict[str, dict[str, str]]:
    faces: dict[str, dict[str, str]] = {}
    for subset, body in re.findall(r"/\*\s*([\w-]+)\s*\*/\s*@font-face\s*\{(.*?)\}", css, flags=re.S):
        src = re.search(r"url\((https://[^)]+\.woff2)\)", body)
        rng = re.search(r"unicode-range:\s*([^;]+);", body)
        if src and rng:
            faces[subset] = {"url": src.group(1), "range": rng.group(1).strip()}
    return faces


def build(with_opsz: bool) -> tuple[dict[str, bytes], dict[str, dict[str, str]]]:
    faces = parse_faces(fetch(css_url(with_opsz)).decode("utf-8"))
    missing = [s for s in SUBSETS if s not in faces]
    if missing:
        raise SystemExit(f"thiếu subset {missing} (đã có: {sorted(faces)})")
    blobs = {}
    for subset in SUBSETS:
        data = fetch(faces[subset]["url"])
        if data[:4] != b"wOF2":
            raise SystemExit(f"{subset}: không phải woff2")
        blobs[subset] = data
    return blobs, faces


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    blobs, faces = build(with_opsz=True)
    if any(len(b) > SIZE_LIMIT for b in blobs.values()):
        print("[fonts] tệp > 90 KB với trục opsz, thử lại không có opsz")
        blobs, faces = build(with_opsz=False)
    block = [BEGIN]
    for subset in SUBSETS:
        name = f"{SLUG}-600-700-{subset}.woff2"
        (FONTS / name).write_bytes(blobs[subset])
        print(f"[fonts] {name}: {len(blobs[subset]) / 1024:.1f} KB")
        block.append(
            "@font-face {\n"
            f"  font-family: '{FAMILY}';\n  font-style: normal;\n  font-weight: 600 700;\n  font-display: swap;\n"
            f"  src: url('/static/fonts/{name}') format('woff2');\n"
            f"  unicode-range: {faces[subset]['range']};\n}}"
        )
    block.append(END)
    text = "\n".join(block)

    path = FONTS / "fonts.css"
    css = path.read_text(encoding="utf-8")
    if BEGIN in css and END in css:
        css = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _m: text, css, flags=re.S)
    else:
        css = css.rstrip("\n") + "\n\n" + text + "\n"
    path.write_text(css, encoding="utf-8", newline="\n")
    print("[fonts] fonts.css đã cập nhật; nhớ đổi ?v= của fonts.css trong index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
