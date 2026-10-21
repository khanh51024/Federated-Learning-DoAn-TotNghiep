"""scripts/fetch_external_sources.py: tải Mendeley theo thư mục, đọc một phần zip ở xa, khúc Range phải đủ byte (không dùng mạng thật)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import scripts.fetch_external_sources as fetch


def _tree() -> tuple[dict, dict]:
    folders = [{"id": "root", "name": "Tomato Leaf", "parent_id": None},
               {"id": "a", "name": "Spider Mites", "parent_id": "root"},
               {"id": "b", "name": "Healthy", "parent_id": "root"},
               {"id": "c", "name": "Leaf Miner", "parent_id": "root"}]
    files = [{"filename": "m1.jpg", "folder_id": "a", "content_details": {"download_url": "http://x/m1"}},
             {"filename": "m2.JPG", "folder_id": "a", "content_details": {"download_url": "http://x/m2"}},
             {"filename": "aug_m3.jpg", "folder_id": "a", "content_details": {"download_url": "http://x/m3"}},
             {"filename": "notes.csv", "folder_id": "a", "content_details": {"download_url": "http://x/csv"}},
             {"filename": "h1.png", "folder_id": "b", "content_details": {"download_url": "http://x/h1"}},
             {"filename": "l1.jpg", "folder_id": "c", "content_details": {"download_url": "http://x/l1"}}]
    return {"files": files}, {f["id"]: f for f in folders}


def test_mendeley_folders_download_only_matching_image_files_and_keep_the_folder_as_label(tmp_path, monkeypatch):
    meta, folders = _tree()
    monkeypatch.setattr(fetch, "http_json", lambda url: meta if url.endswith("/ds1") else list(folders.values()))
    got = []

    def fake_get(url, dest, **kwargs):
        got.append((url, Path(dest)))
        return 10

    monkeypatch.setattr(fetch, "http_get", fake_get)
    spec = {"dataset_id": "ds1", "folder_regex": "^Tomato Leaf/(Spider Mites|Healthy)$", "exclude_name_regex": "^aug_"}
    (record,) = fetch.fetch_mendeley_folders(spec, tmp_path)
    paths = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for _, p in got)
    assert paths == ["Tomato Leaf/Healthy/h1.png", "Tomato Leaf/Spider Mites/m1.jpg", "Tomato Leaf/Spider Mites/m2.JPG"]   # bỏ csv, ảnh aug_, thư mục không khớp
    assert record["files"] == 3 and record["by_folder"] == {"Tomato Leaf/Spider Mites": 2, "Tomato Leaf/Healthy": 1}


def _zip_bytes(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"img:" + name.encode())
    return buffer.getvalue()


def test_extract_remote_members_filters_by_regex_samples_evenly_per_folder_and_keeps_folders(tmp_path, monkeypatch):
    names = [f"Pack/Black_Rot/img_{i:03d}.jpg" for i in range(10)] + [f"Pack/Healthy/img_{i:03d}.jpg" for i in range(10)] + ["Pack/Black_Rot/readme.txt"]
    data = _zip_bytes(names)
    monkeypatch.setattr(fetch, "RemoteZip", lambda url, refresh=None: io.BytesIO(data))
    spec = {"zip_member_regex": "^Pack/(Black_Rot|Healthy)/", "max_members": 4}
    record = fetch.extract_remote_members("http://x/zip", "pack.zip", spec, tmp_path)
    out = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in tmp_path.rglob("*.jpg"))
    assert record["members_matching"] == 20 and record["members_extracted"] == 8 and record["partial_read_by_range"] is True
    assert len(out) == 8 and all(p.startswith(("Pack/Black_Rot/", "Pack/Healthy/")) for p in out)
    assert sum(p.startswith("Pack/Black_Rot/") for p in out) == 4                       # tối đa max_members mỗi thư mục
    assert not list(tmp_path.rglob("*.txt"))                                              # chỉ ảnh
    unlimited = fetch.extract_remote_members("http://x/zip", "pack.zip", {"zip_member_regex": "^Pack/Healthy/"}, tmp_path / "all")
    assert unlimited["members_extracted"] == 10


class _Response:
    def __init__(self, data: bytes, status: int):
        self.data, self.status = data, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.data


def _remote(monkeypatch, responses: list, refresh=None):
    seen = []

    def fake_urlopen(request, timeout=0):
        seen.append(request.full_url)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    remote = fetch.RemoteZip.__new__(fetch.RemoteZip)
    remote.url, remote.refresh, remote.size, remote.pos = "http://old", refresh, 1000, 0
    return remote, seen


def test_range_chunk_with_missing_bytes_or_wrong_status_is_retried_not_returned(monkeypatch):
    remote, seen = _remote(monkeypatch, [_Response(b"abc", 206), _Response(b"x" * 10, 200), _Response(b"y" * 10, 206)])
    assert remote._fetch(0, 10) == b"y" * 10      # khúc thiếu byte và khúc không phải 206 bị bỏ, khúc đủ mới được dùng
    assert len(seen) == 3


def test_range_refreshes_an_expired_url_and_gives_up_after_the_retries(monkeypatch):
    remote, seen = _remote(monkeypatch, [OSError("403"), _Response(b"z" * 5, 206)], refresh=lambda: "http://fresh")
    assert remote._fetch(0, 5) == b"z" * 5 and seen == ["http://old", "http://fresh"]
    remote, _ = _remote(monkeypatch, [_Response(b"", 206)] * 5)
    with pytest.raises(RuntimeError):
        remote._fetch(0, 5)
