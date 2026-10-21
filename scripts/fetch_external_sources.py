"""Tải các bộ ảnh ngoài đồng công khai khai báo trong configs/external_sources.yaml.

Mỗi nguồn tải vào data/external/sources/<id>/raw/ (gitignored). Ghi `fetch_record.json` gồm URL, giấy phép khai báo, thời điểm, SHA-256 tệp
đã tải; bản ghi đó là bằng chứng nguồn gốc cho phần nhập dữ liệu. Chỉ tải qua API công khai, không cần tài khoản; nguồn đòi đăng nhập
(Kaggle...) không nằm trong danh sách.

Các loại nguồn (`kind`):
  mendeley   : dataset_id Mendeley Data; tải gói zip công khai (có tên thư mục = nhãn). `files` tùy chọn để chỉ tải một số tệp.
  zenodo     : record_id + files (khóa tệp); dùng khi bộ quá lớn: `zip_members_prefix` + `max_members` đọc từng phần tệp zip bằng HTTP Range
               thay vì tải hết (ASDID có tệp zip 7 GB).
  url        : một hoặc nhiều URL trực tiếp (`urls`).
  fourtu     : item UUID của 4TU.ResearchData.
  hf_parquet : kho dataset Hugging Face; tải `files` (tệp parquet) rồi giải nén ảnh theo cột `image` và nhãn `label_column`.

Chạy:
    python scripts/fetch_external_sources.py --list
    python scripts/fetch_external_sources.py --source mendeley_tomato_bd_field [--source ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import registry as reg  # noqa: E402

CONFIG = "configs/external_sources.yaml"
SOURCES_DIR = "data/external/sources"
USER_AGENT = "CropFed-FL-research/1.0 (university thesis; contact: repo owner)"


def load_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(reg.abs_path(CONFIG).read_text(encoding="utf-8"))["sources"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def http_get(url: str, dest: Path, retries: int = 3, headers: dict[str, str] | None = None) -> int:
    """Tải một URL về `dest` (theo luồng, ghi tạm rồi đổi tên). Trả số byte. Tệp đã có (khác rỗng) thì dùng lại."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.stat().st_size
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
            with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
                total = 0
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    total += len(block)
            tmp.replace(dest)
            return total
        except Exception as exc:  # noqa: BLE001 — thử lại, báo lỗi cuối
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"không tải được {url}: {last}")


def http_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


class RemoteZip(io.RawIOBase):
    """Tệp zip ở xa đọc bằng HTTP Range: chỉ tải mục lục và các thành viên cần, không tải cả tệp hàng GB."""

    def __init__(self, url: str):
        self.url = url
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    BLOCK = 8 * 1024 * 1024  # mỗi yêu cầu Range lấy 8 MB: zipfile đọc từng khúc 64 KB, nếu không đệm sẽ gọi mạng hàng nghìn lần mỗi ảnh
    _block_start = -1
    _block = b""

    def _fetch(self, start: int, length: int) -> bytes:
        request = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{start + length - 1}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Range lỗi: {self.url}")

    def read(self, size: int = -1) -> bytes:
        if size < 0 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        end = self.pos + size
        if not (self._block_start <= self.pos and end <= self._block_start + len(self._block)):
            length = min(max(self.BLOCK, size), self.size - self.pos)
            self._block, self._block_start = self._fetch(self.pos, length), self.pos
        offset = self.pos - self._block_start
        data = self._block[offset: offset + size]
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


# --------------------------------------------------------------------------- từng loại nguồn
def fetch_mendeley(spec: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    meta = http_json(f"https://data.mendeley.com/public-api/datasets/{spec['dataset_id']}")
    records = [{"api_license": (meta.get("data_licence") or {}).get("short_name"), "api_name": meta.get("name"), "api_doi": (meta.get("doi") or {}).get("id")}]
    wanted = spec.get("files")
    if wanted:  # tải riêng một số tệp (bộ chứa vài zip lớn)
        for entry in meta["files"]:
            if entry["filename"] in wanted:
                url = entry["content_details"]["download_url"]
                size = http_get(url, raw / entry["filename"])
                records.append({"file": entry["filename"], "url": url, "bytes": size, "sha256": sha256_file(raw / entry["filename"])})
    else:
        zip_path = raw / f"mendeley_{spec['dataset_id']}.zip"
        size = http_get(f"https://data.mendeley.com/public-api/zip/{spec['dataset_id']}/download/{spec.get('version', 1)}", zip_path)
        records.append({"file": zip_path.name, "bytes": size, "sha256": sha256_file(zip_path)})
    return records


def fetch_zenodo(spec: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    meta = http_json(f"https://zenodo.org/api/records/{spec['record_id']}")
    records = [{"api_license": (meta["metadata"].get("license") or {}).get("id"), "api_title": meta["metadata"].get("title"), "api_doi": meta.get("doi")}]
    for entry in meta["files"]:
        if entry["key"] not in spec["files"]:
            continue
        url = entry["links"]["self"]
        if spec.get("max_members"):
            remote = RemoteZip(url)
            with zipfile.ZipFile(remote) as archive:
                members = [m for m in archive.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png")) and m.startswith(spec.get("zip_members_prefix", ""))]
                # Lấy cách đều trên toàn bộ danh sách (không lấy liền một đoạn): ảnh liền nhau thường cùng một cây/ngày, lấy cách đều cho đa dạng hơn
                limit = int(spec["max_members"])
                step = max(1, len(members) // limit)
                members = members[::step][:limit]
                out_dir = raw / Path(entry["key"]).stem
                out_dir.mkdir(parents=True, exist_ok=True)
                for member in members:
                    target = out_dir / Path(member).name
                    if not target.exists():
                        target.write_bytes(archive.read(member))
            records.append({"file": entry["key"], "url": url, "members_extracted": len(members), "partial_read_by_range": True})
        else:
            size = http_get(url, raw / entry["key"])
            records.append({"file": entry["key"], "url": url, "bytes": size, "sha256": sha256_file(raw / entry["key"])})
    return records


def fetch_urls(spec: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    records = []
    for url in spec["urls"]:
        name = url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "download.bin"
        size = http_get(url, raw / name)
        records.append({"file": name, "url": url, "bytes": size, "sha256": sha256_file(raw / name)})
    return records


def fetch_fourtu(spec: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    url = f"https://data.4tu.nl/ndownloader/items/{spec['item_uuid']}/versions/{spec.get('version', 1)}"
    size = http_get(url, raw / f"4tu_{spec['item_uuid']}.zip")
    return [{"file": f"4tu_{spec['item_uuid']}.zip", "url": url, "bytes": size, "sha256": sha256_file(raw / f"4tu_{spec['item_uuid']}.zip")}]


def fetch_hf_parquet(spec: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    info = http_json(f"https://huggingface.co/api/datasets/{spec['repo']}")
    records = [{"api_license": (info.get("cardData") or {}).get("license"), "api_repo": spec["repo"], "api_sha": info.get("sha")}]
    names: list[str] | None = None
    for filename in spec["files"]:
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{filename}"
        path = raw / Path(filename).name
        size = http_get(url, path)
        table = pq.read_table(path)
        if names is None and spec.get("label_names"):
            names = spec["label_names"]
        out_dir = raw / Path(filename).stem
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = table.to_pylist()
        for index, row in enumerate(rows):
            image = row[spec.get("image_column", "image")]
            label = row[spec["label_column"]]
            if isinstance(label, list):  # nhiều nhãn (Plant Pathology): chỉ ảnh có đúng MỘT nhãn mới có nhãn rõ ràng
                label = label[0] if len(label) == 1 else None
            label_text = "multi_label" if label is None else (names[label] if (names and isinstance(label, int)) else str(label))
            folder = out_dir / label_text.replace("/", "_")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{Path(filename).stem}_{index:05d}.jpg").write_bytes(image["bytes"])
        records.append({"file": filename, "url": url, "bytes": size, "sha256": sha256_file(path), "rows": len(rows)})
        path.unlink()  # parquet lớn: chỉ giữ ảnh đã giải nén, SHA ở trên là bằng chứng nguồn
    return records


FETCHERS = {"mendeley": fetch_mendeley, "zenodo": fetch_zenodo, "url": fetch_urls, "fourtu": fetch_fourtu, "hf_parquet": fetch_hf_parquet}


MAX_PATH_LEN = 235  # giới hạn 260 ký tự của Windows, chừa chỗ cho phần ghi thêm


def safe_member_path(target: Path, member_name: str, taken: set[str]) -> Path:
    """Đường dẫn đích cho một thành viên zip: giữ nhiều cấp thư mục nhất mà vẫn < MAX_PATH_LEN, nếu không thì rút gọn tên tệp.

    Luôn giữ ít nhất thư mục cha trực tiếp vì nó thường là nhãn lớp."""
    # Windows không tạo được thư mục/tệp có tên kết thúc bằng dấu cách hoặc dấu chấm: bỏ chúng khỏi từng phần của đường dẫn
    parts = [p.rstrip(" .") or "_" for p in member_name.replace("\\", "/").split("/") if p and p not in (".", "..")]
    candidate: Path | None = None
    for depth in range(len(parts), 0, -1):
        trial = target.joinpath(*parts[-depth:])
        if len(str(trial)) < MAX_PATH_LEN:
            candidate = trial
            break
    if candidate is None or len(parts) < 2 and len(str(candidate)) >= MAX_PATH_LEN:
        stem, dot, suffix = parts[-1].rpartition(".")
        room = max(12, MAX_PATH_LEN - len(str(target)) - (len(parts[-2]) if len(parts) > 1 else 0) - len(suffix) - 6)
        short = f"{stem[: room // 2]}~{stem[-(room // 2):]}.{suffix}"
        candidate = target.joinpath(*(parts[-2:-1] + [short])) if len(parts) > 1 else target / short
    final, counter = candidate, 1
    while str(final) in taken:
        final = candidate.with_name(f"{candidate.stem}__{counter}{candidate.suffix}")
        counter += 1
    taken.add(str(final))
    return final


def extract_zips(raw: Path) -> int:
    """Giải nén mọi zip trong raw/ (kể cả zip lồng zip, tối đa 4 cấp) bằng đường dẫn đã rút gọn; không cho thoát khỏi thư mục đích."""
    count = 0
    for _ in range(4):
        pending = [p for p in sorted(raw.rglob("*.zip")) if not p.with_suffix(".done").exists()]
        if not pending:
            break
        for index, archive_path in enumerate(pending):
            target = archive_path.parent / (archive_path.stem if archive_path.parent == raw else f"_z{index}")
            archive_path.with_suffix(".done").write_text("extracted", encoding="utf-8")
            taken: set[str] = set()
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    destination = safe_member_path(target, info.filename, taken)
                    if not str(destination.resolve()).startswith(str(target.resolve())):
                        raise RuntimeError(f"đường dẫn zip không an toàn: {info.filename}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("wb") as out:
                        while True:
                            block = source.read(1 << 20)
                            if not block:
                                break
                            out.write(block)
            count += 1
    return count


def extract_rars(raw: Path) -> int:
    """Giải nén .rar bằng tar của Windows (libarchive đọc được RAR); trên máy không có thì báo rõ thay vì bỏ qua."""
    import shutil
    import subprocess

    tar = shutil.which("bsdtar") or ("C:/Windows/System32/tar.exe" if Path("C:/Windows/System32/tar.exe").is_file() else None)
    count = 0
    for archive_path in sorted(raw.rglob("*.rar")):
        target = archive_path.with_suffix("")
        if target.exists():
            continue
        if tar is None:
            raise RuntimeError(f"cần bsdtar hoặc 7-Zip để giải nén {archive_path.name}")
        target.mkdir(parents=True)
        subprocess.run([tar, "-xf", str(archive_path), "-C", str(target)], check=True, capture_output=True)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--keep-zips", action="store_true", help="giữ tệp zip sau khi giải nén (mặc định xóa để tiết kiệm đĩa)")
    parser.add_argument("--extract-only", action="store_true", help="chỉ giải nén các zip còn sót trong raw/ của nguồn đã chọn")
    args = parser.parse_args()

    sources = load_config()
    if args.list:
        for name, spec in sources.items():
            print(f"{name:34s} {spec['kind']:11s} {spec.get('license', '?'):28s} {spec.get('title', '')}")
        return 0
    for name in args.source:
        spec = sources[name]
        base = reg.abs_path(SOURCES_DIR) / name
        raw = base / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {name}: {spec.get('title', '')}", flush=True)
        started = time.time()
        if args.extract_only:
            extracted = extract_zips(raw)
            for archive_path in raw.rglob("*.zip"):
                archive_path.unlink()
            for marker in raw.rglob("*.done"):
                marker.unlink()
            print(f"[fetch] {name}: giải nén thêm {extracted} zip")
            continue
        records = FETCHERS[spec["kind"]](spec, raw)
        extracted = extract_zips(raw) + extract_rars(raw)
        if extracted and not args.keep_zips:
            for archive_path in [*raw.rglob("*.zip"), *raw.rglob("*.rar")]:
                archive_path.unlink()
            for marker in raw.rglob("*.done"):
                marker.unlink()
        (base / "fetch_record.json").write_text(json.dumps({
            "source": name, "title": spec.get("title"), "declared_license": spec.get("license"), "url": spec.get("url"),
            "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seconds": round(time.time() - started, 1),
            "records": records,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        n_images = sum(1 for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"[fetch] {name}: xong, {n_images} ảnh trong raw/ ({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
