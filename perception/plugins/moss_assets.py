"""Install the pinned MOSS asset archive into the shared /models cache.

The archive contains original ONNX assets and reference voices, the verified
Orin TensorRT 10.4 plans, their provenance, and the tested Chinese TN cache.
No network request is made unless a URL is explicitly supplied.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)


# SHA and byte count of the reviewed portable bundle. Rebuilding the bundle
# requires reviewing these pins in the project code before deployment.
ASSET_SHA256 = "6f42daded458b3dedffb10b52e216aaf71dd5d56dbc9647c71522db9cb13e6c2"
ASSET_BYTES = 1558394880
ASSET_HOST = "172.28.4.81"
ASSET_PORT = 34567
READY_FILE = ".moss-assets-ready.json"
REQUIRED_FILES = (
    "analysis.json",
    "engine-map.json",
    "onnx/codec/codec_browser_onnx_meta.json",
    "source/MOSS-TTS-Nano/onnx_tts_runtime.py",
    "release-manifest.json",
    "frame-candidate.json",
    "plans/global_unified/global_unified.plan",
    "plans/codec_decode_step/codec_decode_step.plan",
    "plans/local_fixed_sampled_frame/local_fixed_sampled_frame.plan",
    "plans/local_fixed_sampled_frame/build_report.json",
)


def _required_assets_present(asset_dir: Path) -> bool:
    """Resolve the packaged frame candidate without fixing its release layout."""
    try:
        if not all((asset_dir / name).is_file() for name in REQUIRED_FILES):
            return False
        candidate = json.loads((asset_dir / "frame-candidate.json").read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            return False
        root = asset_dir.resolve()
        for field in ("plan", "contract", "onnx"):
            name = candidate.get(field)
            if not isinstance(name, str) or not name or "\\" in name:
                return False
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                return False
            target = asset_dir.joinpath(*relative.parts).resolve()
            target.relative_to(root)
            if not target.is_file():
                return False
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _ready(asset_dir: Path, sha256: str, size: int) -> bool:
    try:
        marker = json.loads((asset_dir / READY_FILE).read_text(encoding="utf-8"))
        return (marker == {"archive_sha256": sha256, "archive_bytes": size}
                and _required_assets_present(asset_dir))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def _download(url: str, destination: Path, sha256: str, size: int) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname != ASSET_HOST
            or parsed.port != ASSET_PORT or not parsed.path.startswith("/")
            or parsed.username is not None or parsed.password is not None):
        raise ValueError(f"MOSS_ASSET_URL must be under http://{ASSET_HOST}:{ASSET_PORT}/")
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    next_progress = 10
    log.info("[moss-assets] downloading pinned archive (%.1f MiB)", size / (1024 * 1024))
    # The internal JuiceFS endpoint must bypass HTTP(S)_PROXY in the image.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response, destination.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > size:
                raise ValueError("MOSS archive exceeds its pinned byte count")
            digest.update(chunk)
            output.write(chunk)
            percent = received * 100 // size
            if percent >= next_progress:
                log.info("[moss-assets] download %d%% (%.1f/%.1f MiB, %.1fs)",
                         min(percent, 100), received / (1024 * 1024),
                         size / (1024 * 1024), time.monotonic() - started)
                next_progress = (percent // 10 + 1) * 10
        output.flush()
        os.fsync(output.fileno())
    if received != size or digest.hexdigest() != sha256:
        raise ValueError(f"MOSS archive integrity mismatch: bytes={received}, sha256={digest.hexdigest()}")
    log.info("[moss-assets] archive size and SHA256 verified in %.1fs", time.monotonic() - started)


def _extract_regular_files(archive_path: Path, staging: Path) -> None:
    started = time.monotonic()
    extracted = 0
    next_progress = 256 * 1024 * 1024
    log.info("[moss-assets] extracting verified archive")
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            name = member.name
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or ".." in path.parts
                    or name.startswith("\\") or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe MOSS archive member: {name!r}")
            target = staging.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read MOSS archive member: {name!r}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != member.size:
                raise ValueError(f"Truncated MOSS archive member: {name!r}")
            extracted += member.size
            if extracted >= next_progress:
                log.info("[moss-assets] extracted %.1f MiB in %.1fs",
                         extracted / (1024 * 1024), time.monotonic() - started)
                next_progress = (extracted // (256 * 1024 * 1024) + 1) * (256 * 1024 * 1024)
    log.info("[moss-assets] extracted %.1f MiB in %.1fs",
             extracted / (1024 * 1024), time.monotonic() - started)


def ensure_moss_assets(asset_dir: str | Path, *, url: str | None = None,
                       sha256: str = ASSET_SHA256, size: int = ASSET_BYTES) -> Path:
    """Return the verified cache, or install it once under an interprocess lock.

    `sha256` and `size` are injectable for small offline tests. Production callers
    use the pinned defaults; only the URL is configured at deployment time.
    """
    destination = Path(asset_dir).resolve()
    if _ready(destination, sha256, size):
        log.info("[moss-assets] verified asset cache is ready")
        return destination
    if not url:
        raise FileNotFoundError(
            f"MOSS assets are not installed at {destination}; set MOSS_ASSET_URL "
            "to the pinned JuiceFS HTTP archive or stage and verify the archive"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.download.lock"
    with lock_path.open("a+b") as lock:
        waiting_since = time.monotonic()
        log.info("[moss-assets] waiting for asset installation lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        log.info("[moss-assets] asset installation lock acquired after %.1fs",
                 time.monotonic() - waiting_since)
        if _ready(destination, sha256, size):
            log.info("[moss-assets] verified asset cache was installed by another process")
            return destination
        if destination.exists():
            raise RuntimeError(f"Existing MOSS asset directory is incomplete: {destination}")
        token = uuid.uuid4().hex
        partial = destination.parent / f".{destination.name}.{token}.partial"
        staging = destination.parent / f".{destination.name}.{token}.unpack"
        try:
            _download(url, partial, sha256, size)
            staging.mkdir()
            _extract_regular_files(partial, staging)
            if not _required_assets_present(staging):
                raise ValueError("MOSS archive is missing required model assets")
            (staging / READY_FILE).write_text(
                json.dumps({"archive_sha256": sha256, "archive_bytes": size}) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
            log.info("[moss-assets] asset installation complete")
            return destination
        finally:
            partial.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
