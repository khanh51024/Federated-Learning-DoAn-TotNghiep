"""Nút tải bộ cài trên trang #flower: file cài điền đúng mã băm, gói chỉ có ảnh của client, chưa có gói thì báo rõ."""
from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from scripts.build_client_package import plain_members
from web.services import client_package as cp


def _package(tmp_path, content=b"noi dung goi"):
    (tmp_path / cp.ZIP_NAME).write_bytes(content)
    (tmp_path / cp.MANIFEST_NAME).write_text(json.dumps({"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}), encoding="utf-8")


def _client(monkeypatch, tmp_path):
    from web.app import app
    from web.services import admin_auth
    monkeypatch.setattr(cp, "package_dir", lambda: tmp_path)
    monkeypatch.setattr(admin_auth.LIMITER, "allow", lambda *_a, **_k: (True, 0))
    return TestClient(app)


def test_routes_say_missing_until_the_package_exists(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/flower/client-package/status").json() == {"available": False}
    assert client.get("/api/flower/client-package/install.bat").status_code == 503
    assert client.get("/api/flower/client-package/file").status_code == 503


def test_installer_carries_the_hash_and_the_file_downloads_whole(monkeypatch, tmp_path):
    content = b"noi dung goi" * 100
    _package(tmp_path, content)
    client = _client(monkeypatch, tmp_path)
    sha = hashlib.sha256(content).hexdigest()
    assert client.get("/api/flower/client-package/status").json() == {"available": True, "bytes": len(content)}
    reply = client.get("/api/flower/client-package/install.bat")
    assert reply.status_code == 200 and "attachment" in reply.headers["content-disposition"]
    body = reply.content.decode("utf-8")
    assert f'set "SHA={sha}"' in body and "@@" not in body and "http://testserver/api/flower/client-package/file" in body
    assert body.count("\n") == body.count("\r\n")
    assert client.get("/api/flower/client-package/file").content == content
    part = client.get("/api/flower/client-package/file", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == content[10:20]


def test_package_that_does_not_match_its_manifest_is_not_offered(monkeypatch, tmp_path):
    _package(tmp_path)
    (tmp_path / cp.ZIP_NAME).write_bytes(b"bi cat")
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/flower/client-package/status").json() == {"available": False}


@pytest.mark.parametrize("base", ['https://x.vn" & calc & "', "https://x.vn/duong/dan", "file:///c:/x"])
def test_installer_refuses_an_address_that_could_break_out_of_the_bat_file(base):
    with pytest.raises(ValueError):
        cp.installer(base, {"sha256": "a" * 64, "bytes": 1})


def _data(root, clients):
    (root / "partition" / "clients").mkdir(parents=True)
    (root / "partition" / "partition_config.json").write_text("{}", encoding="utf-8")
    (root / "partition" / "global_val.csv").write_text("x", encoding="utf-8")
    (root / "images_eval" / "lop").mkdir(parents=True)
    (root / "images_eval" / "lop" / "bi_mat.jpg").write_bytes(b"x")
    for client in clients:
        (root / "partition" / "clients" / f"client_{client:02d}.csv").write_text("x", encoding="utf-8")
        (root / f"images_client_{client:02d}" / "lop").mkdir(parents=True)
        (root / f"images_client_{client:02d}" / "lop" / "anh.jpg").write_bytes(b"x")


def test_package_holds_client_images_only_never_the_servers_evaluation_set(tmp_path):
    net10, mini10 = tmp_path / "net10", tmp_path / "mini10"
    _data(net10, range(10))
    _data(mini10, range(10))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "deploy" / "lan_client").mkdir(parents=True)
    (tmp_path / "deploy" / "lan_client" / "CHAY_CLIENT.bat").write_text("", encoding="utf-8")
    names = {name for name, _path in plain_members(tmp_path, net10, mini10)}
    assert {"src/a.py", "CHAY_CLIENT.bat", "images_client_05/lop/anh.jpg", "mini10/images_client_09/lop/anh.jpg", "mini10/partition/clients/client_00.csv"} <= names
    assert "images_client_04/lop/anh.jpg" not in names and "images_client_07/lop/anh.jpg" not in names
    assert not [name for name in names if "global_" in name or "images_eval" in name]
