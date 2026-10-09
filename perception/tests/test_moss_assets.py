"""Pinned MOSS cache install never contacts the network by default."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from plugins import moss_assets


def _archive(extra=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in moss_assets.REQUIRED_FILES:
            payload = name.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
    return buffer.getvalue()


def _fake_internal_response(monkeypatch, payload):
    class Opener:
        calls = 0

        def open(self, url, timeout):
            assert url.startswith("http://172.28.4.81:34567/")
            self.calls += 1
            return io.BytesIO(payload)

    opener = Opener()
    monkeypatch.setattr(moss_assets.urllib.request, "build_opener", lambda *_: opener)
    return opener


def test_no_url_means_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(moss_assets.urllib.request, "build_opener",
                        lambda *_: pytest.fail("unexpected network request"))
    with pytest.raises(FileNotFoundError, match="MOSS_ASSET_URL"):
        moss_assets.ensure_moss_assets(tmp_path / "assets")


def test_verified_archive_installed_once_and_reused(tmp_path, monkeypatch):
    payload = _archive()
    opener = _fake_internal_response(monkeypatch, payload)
    destination = tmp_path / "assets"
    kwargs = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    result = moss_assets.ensure_moss_assets(
        destination, url="http://172.28.4.81:34567/moss-assets.tar", **kwargs,
    )
    assert result == destination.resolve()
    assert (destination / moss_assets.REQUIRED_FILES[0]).is_file()
    assert moss_assets.ensure_moss_assets(destination, **kwargs) == result
    assert opener.calls == 1
    assert list(tmp_path.glob("*.partial")) == []


def test_rejects_path_traversal_even_when_archive_hash_matches(tmp_path, monkeypatch):
    member = tarfile.TarInfo("../escape")
    member.size = 1
    payload = _archive(member)
    _fake_internal_response(monkeypatch, payload)
    with pytest.raises(ValueError, match="Unsafe MOSS archive member"):
        moss_assets.ensure_moss_assets(
            tmp_path / "assets", url="http://172.28.4.81:34567/moss-assets.tar",
            sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
        )
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "assets").exists()


def test_bad_hash_never_installs_archive(tmp_path, monkeypatch):
    payload = _archive()
    _fake_internal_response(monkeypatch, payload)
    with pytest.raises(ValueError, match="integrity mismatch"):
        moss_assets.ensure_moss_assets(
            tmp_path / "assets", url="http://172.28.4.81:34567/moss-assets.tar",
            sha256="0" * 64, size=len(payload),
        )
    assert not (tmp_path / "assets").exists()
