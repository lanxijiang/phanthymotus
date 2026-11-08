"""SHA-pinned V6 GPU bundle installation. No model inference or backend fallback.

Production downloads use a fixed ModelScope repository and immutable revision.
An internal HTTP mirror is available only through the explicitly named test opt-in.
Existing bundles are rehashed before use; unknown/corrupt destinations are preserved.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import time
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

MANIFEST = 'runtime_manifest.json'
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, name):
    value = str(value or '').lower()
    if not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('A fixed SHA256 is required for ' + name)
    return value


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid bundle relative path')
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ('', '.', '..') for part in path.parts):
        raise ValueError('Bundle path escapes its directory')
    if path.as_posix() != name or len(path.parts) > 8 or len(name) > 512:
        raise ValueError('Noncanonical bundle path')
    return name


def _verified_bundle(root, expected_manifest_sha256):
    root = Path(root)
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError('Missing V6 runtime_manifest.json')
    if _sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('V6 runtime manifest SHA mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema_version') != 1
            or manifest.get('project_version') != 'xiaoyou_v6'
            or manifest.get('model_id') != 'xiaoyou01'
            or manifest.get('voice_id') != 'xiaoyou01'
            or manifest.get('backend') != 'tensorrt_fp32'
            or manifest.get('precision') != 'FP32'
            or manifest.get('sample_rate') != 24000
            or manifest.get('global_step') != 16500):
        raise ValueError('Wrong V6 GPU runtime bundle identity')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or MANIFEST in files:
        raise ValueError('Missing or circular V6 file manifest')
    declared = {MANIFEST}
    total = 0
    for name, item in files.items():
        name = _relative(name)
        if not isinstance(item, dict):
            raise ValueError('Invalid V6 file declaration')
        size = item.get('bytes')
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError('Invalid V6 file size')
        total += size
        expected = _digest(item.get('sha256'), name)
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Missing or escaping V6 file')
        if path.stat().st_size != size or _sha(path) != expected:
            raise ValueError('V6 file integrity mismatch: ' + name)
        declared.add(name)
    if total > MAX_BUNDLE_BYTES:
        raise ValueError('V6 bundle exceeds the fixed extraction bound')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks are not accepted in V6 bundles')
        if path.is_file() and path.relative_to(root).as_posix() not in declared:
            raise ValueError('Undeclared file in V6 bundle')
    # Schema/weights/frontend/typed IO and GPU long-profile acceptance are checked
    # again by the GPU backend; a successful download is not numerical acceptance.
    return root


@contextmanager
def _lock(path):
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)



def _packaged_release():
    path = Path(__file__).with_name('release_spec.json')
    if not path.is_file():
        return {}
    release = json.loads(path.read_text())
    if (release.get('schema_version') != 1 or release.get('model_id') != 'xiaoyou01'
            or release.get('voice_id') != 'xiaoyou01'):
        raise ValueError('Wrong packaged public model release identity')
    return release



def resolve_runtime_manifest_pin(explicit=None):
    """Resolve only an external cfg/env/fixed-Git release pin, never bundle content."""
    release = _packaged_release()
    supplied = [value for value in (explicit, os.environ.get('XIAOYOU_V6_RUNTIME_MANIFEST_SHA256'),
                                   release.get('runtime_manifest_sha256')) if value]
    pins = {_digest(value, 'runtime manifest') for value in supplied}
    if not pins:
        raise ValueError('An external fixed V6 runtime manifest SHA is required')
    if len(pins) != 1:
        raise ValueError('Configured runtime manifest SHA declarations disagree')
    return pins.pop()


def _release_from_environment():
    release = _packaged_release()
    expected = _digest(os.environ.get('XIAOYOU_V6_ASSET_SHA256') or release.get('asset_sha256'), 'model archive')
    try:
        size = int(os.environ.get('XIAOYOU_V6_ASSET_BYTES') or release.get('asset_bytes') or '')
    except ValueError:
        raise ValueError('A fixed model archive byte count is required') from None
    if not 0 < size <= MAX_BUNDLE_BYTES:
        raise ValueError('Model archive byte count is outside the supported bound')
    mirror = os.environ.get('XIAOYOU_V6_INTERNAL_TEST_ASSET_URL')
    if mirror:
        if os.environ.get('XIAOYOU_V6_ALLOW_INTERNAL_TEST_ASSETS') != '1':
            raise ValueError('Internal model mirrors require the explicit test opt-in')
        url = mirror
    else:
        repo = os.environ.get('XIAOYOU_V6_MODELSCOPE_REPO') or release.get('modelscope_repo') or 'lanxijiang/xiaoyou01'
        revision = os.environ.get('XIAOYOU_V6_MODELSCOPE_REVISION') or release.get('modelscope_revision') or ''
        name = os.environ.get('XIAOYOU_V6_MODELSCOPE_FILE') or release.get('modelscope_file') or 'xiaoyou01.tar.gz'
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
            raise ValueError('A fixed ModelScope repository is required')
        if not re.fullmatch(r'[0-9a-fA-F]{40,64}', revision):
            raise ValueError('ModelScope revision must be an immutable commit, never a branch')
        name = _relative(name)
        url = ('https://modelscope.cn/api/v1/models/' + quote(repo, safe='/')
               + '/repo?' + urlencode({'Revision': revision, 'FilePath': name}))
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Model archive URL must be HTTP(S) without embedded credentials')
    return url, size, expected


def _download(url, path, expected_size, expected_sha256):
    request = Request(url, headers={'User-Agent': 'xiaoyou-v6-pinned-assets/1'})
    last_error = None
    for attempt in range(3):
        digest = hashlib.sha256()
        count = 0
        try:
            with urlopen(request, timeout=30) as response, path.open('xb') as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > expected_size:
                        raise ValueError('Archive exceeds its pinned byte count')
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if count != expected_size or digest.hexdigest() != expected_sha256:
                raise ValueError('Downloaded V6 archive size/SHA mismatch')
            return
        except Exception as exc:
            last_error = type(exc).__name__
            if path.exists():
                path.unlink()  # Only this invocation's exclusive temporary path.
            if attempt < 2:
                time.sleep(attempt + 1)
    # Never expose signed URLs, credentials, or HTTP request details in exceptions.
    raise RuntimeError('V6 archive download failed after3 attempts: ' + str(last_error))


def _extract(archive_path, destination):
    total = 0
    seen = set()
    with tarfile.open(archive_path, 'r:*') as archive:
        for member_number, member in enumerate(archive, 1):
            if member_number > 4096:
                raise ValueError('Too many archive members')
            name = member.name.rstrip('/') if member.isdir() else member.name
            name = _relative(name)
            if name in seen:
                raise ValueError('Duplicate archive member')
            seen.add(name)
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('Only ordinary V6 files/directories are accepted')
            if not 0 < member.size <= MAX_FILE_BYTES:
                raise ValueError('Archive member size is outside the fixed bound')
            total += member.size
            if total > MAX_BUNDLE_BYTES:
                raise ValueError('Archive uncompressed size exceeds the fixed bound')
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError('Missing archive member data')
            with source, target.open('xb') as stream:
                remaining = member.size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('Truncated archive member')
                    stream.write(chunk)
                    remaining -= len(chunk)
                stream.flush()
                os.fsync(stream.fileno())


def ensure_runtime_bundle(model_dir=None, expected_manifest_sha256=None):
    """Return a fully rehashed local bundle; fail closed on any missing pin or mismatch.

    ModelScope repo/revision/archive size/SHA are required only for a cold download.
    A preprovisioned offline bundle is accepted only with the same external manifest
    SHA and complete per-file verification. Existing unknown content is never erased.
    """
    expected = resolve_runtime_manifest_pin(expected_manifest_sha256)
    root = Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/xiaoyou01').expanduser()
    if root.is_symlink():
        raise ValueError('V6 model root must not be a symlink')
    root = root.resolve()
    if root == root.parent or root.name in ('models', '', '.', '..'):
        raise ValueError('A dedicated V6 model subdirectory is required')
    if root.exists() and any(root.iterdir()):
        return _verified_bundle(root, expected)
    url, size, archive_sha = _release_from_environment()
    root.parent.mkdir(parents=True, exist_ok=True)
    with _lock(root.parent / ('.' + root.name + '.xiaoyou-v6.lock')):
        if root.exists() and any(root.iterdir()):
            return _verified_bundle(root, expected)
        with tempfile.TemporaryDirectory(prefix='.' + root.name + '.install-', dir=root.parent) as temp:
            temp = Path(temp)
            archive = temp / 'model.tar'
            staging = temp / 'bundle'
            staging.mkdir()
            _download(url, archive, size, archive_sha)
            _extract(archive, staging)
            _verified_bundle(staging, expected)
            os.replace(staging, root)  # Complete verified tree, same filesystem.
    return _verified_bundle(root, expected)
