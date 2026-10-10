"""Offline checks for exact frontend reconstruction and fail-fast GPU startup."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lock_tool():
    return load_tool("moss_frontend_lock")


@pytest.fixture
def preflight():
    return load_tool("moss_runtime_preflight")


def write_metadata(folder):
    folder.mkdir()
    for name, version, build in [("python", "3.10.22", "test_0"),
                                 ("pynini", "2.1.7", "py310h0992a49_2"),
                                 ("openfst", "1.8.4", "test_0"),
                                 ("importlib_resources", "7.1.0", "test_0")]:
        record = dict(name=name, version=version, build=build, sha256="a" * 64,
                      url=f"https://conda.anaconda.org/conda-forge/linux-aarch64/{name}-{version}-{build}.conda")
        (folder / f"{name}.json").write_text(json.dumps(record))
    return folder


def test_frontend_metadata_lock_roundtrip_and_installed_mismatch(lock_tool, tmp_path):
    metadata = write_metadata(tmp_path / "conda-meta")
    path = tmp_path / "frontend.lock"
    lock_tool.export_lock(metadata, path)
    original = path.read_bytes()
    assert lock_tool.read_lock(path) == lock_tool.installed_records(metadata)
    lock_tool.export_lock(metadata, path)
    assert path.read_bytes() == original
    changed = metadata / "python.json"
    record = json.loads(changed.read_text())
    record["sha256"] = "b" * 64
    changed.write_text(json.dumps(record))
    result = subprocess.run(["python3", str(TOOLS / "moss_frontend_lock.py"), "check",
                             "--lock", str(path), "--metadata", str(metadata)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not match the lock: ['python']" in result.stderr


@pytest.mark.parametrize("corruption", ["banner", "hash", "architecture", "duplicate", "version", "inference_stack"])
def test_frontend_lock_rejects_unreplayable_or_unvalidated_artifacts(lock_tool, tmp_path, corruption):
    metadata = write_metadata(tmp_path / "conda-meta")
    path = tmp_path / "frontend.lock"
    lock_tool.export_lock(metadata, path)
    text = path.read_text()
    if corruption == "banner":
        text = text.replace("@EXPLICIT", 'List of packages in environment: "/moss/frontend/env"')
    elif corruption == "hash":
        text = text.replace("#sha256:" + "a" * 64, "")
    elif corruption == "architecture":
        text = text.replace("/linux-aarch64/", "/linux-64/")
    elif corruption == "duplicate":
        text += text.splitlines()[-1] + "\n"
    elif corruption == "version":
        text = text.replace("pynini-2.1.7-", "pynini-2.1.6-")
    else:
        text += "https://conda.anaconda.org/conda-forge/linux-aarch64/torch-2.5.0-test_0.conda#sha256:" + "a" * 64 + "\n"
    path.write_text(text)
    with pytest.raises(ValueError):
        lock_tool.read_lock(path)


def test_checked_in_frontend_lock_is_complete_and_small(lock_tool):
    path = TOOLS / "moss_frontend_aarch64.lock"
    records = lock_tool.read_lock(path)
    assert len(records) > 4  # Includes the resolved transitive artifacts.
    assert records["pynini"][1] == "fea0f7c4cc4d6cd8a020bab417255b00c1ab6477d02a0c78788ae11520439b6d"
    assert path.stat().st_size < 1_000_000


@pytest.mark.parametrize("prefix,lock,expected", [
    ("/opt/moss-frontend", None, "requires --lock"),
    ("/opt/somebody-else", None, "--prefix must be"),
    ("/moss/../../etc/foreign", None, "resolves outside"),
    ("/opt/moss-frontend", "bad", "must begin with @EXPLICIT"),
])
def test_installer_rejects_wrong_prefix_or_missing_production_lock_before_install(tmp_path, prefix, lock, expected):
    command = ["bash", str(TOOLS / "moss_prepare_frontend.sh"), "--prefix", prefix]
    if lock:
        path = tmp_path / "bad.lock"
        path.write_text("not a replayable lock\n")
        command += ["--lock", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr


def dependency_fixture(preflight, monkeypatch):
    monkeypatch.setattr(preflight.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(preflight.importlib.metadata, "version", lambda name: preflight.EXPECTED[name])
    imported = []

    class MessageMeta(type):
        _TYPE_SUPPORT = None
        @classmethod
        def __import_type_support__(cls):
            cls._TYPE_SUPPORT = object()

    class Message(metaclass=MessageMeta):
        pass

    def loader(name):
        imported.append(name)
        assert not name.startswith("torch"), "Startup must not add torch's RAM cost"
        return SimpleNamespace(__file__=str(preflight.FRONTEND_ROOT / f"lib/python3.10/site-packages/{name}/__init__.py"),
                               __version__="10.4.0", String=Message, AudioChunk=Message)

    monkeypatch.setattr(preflight.importlib, "import_module", loader)
    return imported


def test_preflight_checks_original_frontend_without_torch_import(preflight, monkeypatch):
    imported = dependency_fixture(preflight, monkeypatch)
    assert preflight.check_dependencies() == preflight.EXPECTED
    assert "plugins.moss_deploy.runtime" in imported
    assert "tn.chinese.normalizer" in imported and "tn.english.normalizer" in imported
    assert {"plugins.tts", "onnxruntime", "rclpy", "std_msgs.msg", "audio_msgs.msg"} <= set(imported)


@pytest.mark.parametrize("failure", ["trt_version", "frontend_prefix", "architecture", "python"])
def test_dependency_preflight_fails_explicitly(preflight, monkeypatch, failure):
    dependency_fixture(preflight, monkeypatch)
    if failure == "trt_version":
        original = preflight.importlib.import_module
        def wrong_version(name):
            module = original(name)
            if name == "tensorrt":
                module.__version__ = "8.5.2"
            return module
        monkeypatch.setattr(preflight.importlib, "import_module", wrong_version)
    elif failure == "frontend_prefix":
        monkeypatch.setattr(preflight.importlib, "import_module", lambda name: SimpleNamespace(
            __file__="/moss/frontend/foreign.py", __version__="10.4.0"))
    elif failure == "architecture":
        monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    else:
        monkeypatch.setattr(preflight.sys, "version_info", (3, 8, 10))
    with pytest.raises(RuntimeError):
        preflight.check_dependencies()


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


def cuda_fixture(preflight, monkeypatch, *, count=1, capability=(8, 7), init_status=0):
    calls = []

    def record(name, *args):
        calls.append(name)
        if name == "cuInit":
            return init_status
        if name == "cuDeviceGetCount":
            args[0]._obj.value = count
        elif name == "cuDeviceGet":
            args[0]._obj.value = 0
        else:
            args[0]._obj.value = capability[0 if args[1] == 75 else 1]
        return 0

    cuda = SimpleNamespace(**{name: FakeFunction(lambda *args, name=name: record(name, *args))
                             for name in ("cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetAttribute")})
    monkeypatch.setattr(preflight.ct, "CDLL", lambda name: cuda)
    return calls


def test_gpu_preflight_checks_orin_without_allocating_context_or_weights(preflight, monkeypatch):
    calls = cuda_fixture(preflight, monkeypatch)
    assert preflight.check_gpu() == {"device": 0, "compute_capability": "8.7"}
    assert calls == ["cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetAttribute", "cuDeviceGetAttribute"]


@pytest.mark.parametrize("parameters,match", [({"count": 0}, "visible GPU"),
                                             ({"capability": (8, 0)}, "Orin SM 8.7"),
                                             ({"init_status": 100}, "CUDA status 100")])
def test_gpu_preflight_never_falls_back_to_cpu(preflight, monkeypatch, parameters, match):
    cuda_fixture(preflight, monkeypatch, **parameters)
    with pytest.raises(RuntimeError, match=match):
        preflight.check_gpu()


def test_image_scripts_parse_as_bash():
    for name in ("moss_prepare_frontend.sh", "moss_entrypoint.sh"):
        subprocess.run(["bash", "-n", str(TOOLS / name)], check=True)


def test_vendor_tensorrt_needs_no_pip_metadata(preflight, monkeypatch):
    dependency_fixture(preflight, monkeypatch)
    def distribution_version(name):
        if name == "tensorrt":
            raise preflight.importlib.metadata.PackageNotFoundError(name)
        return preflight.EXPECTED[name]
    monkeypatch.setattr(preflight.importlib.metadata, "version", distribution_version)
    assert preflight.check_dependencies()["tensorrt"] == "10.4.0"


@pytest.mark.parametrize("with_metadata", [True, False])
def test_build_preflight_checks_trt_installation_without_linking_driver(
        preflight, monkeypatch, tmp_path, with_metadata):
    imported = dependency_fixture(preflight, monkeypatch)
    trt_file = tmp_path / "tensorrt.py"
    trt_file.write_text("raise ImportError('libnvos.so unavailable during docker build')\n")
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name:
                        SimpleNamespace(origin=str(trt_file)))
    original = preflight.importlib.import_module

    def loader(name):
        if name == "tensorrt":
            raise ImportError("libnvos.so unavailable during docker build")
        return original(name)

    def version(name):
        if name == "tensorrt" and not with_metadata:
            raise preflight.importlib.metadata.PackageNotFoundError(name)
        return preflight.EXPECTED[name]

    monkeypatch.setattr(preflight.importlib, "import_module", loader)
    monkeypatch.setattr(preflight.importlib.metadata, "version", version)
    report = preflight.check_dependencies(build_check=True)
    assert report["tensorrt"] == {
        "path": str(trt_file), "distribution_version": "10.4.0" if with_metadata else None,
        "imported": False, "runtime_version_check_pending": True,
    }
    assert "tensorrt" not in imported
    assert {"tn.chinese.normalizer", "rclpy", "audio_msgs.msg"} <= set(imported)
    # Deferring the build import must never weaken the default startup gate.
    with pytest.raises(ImportError, match="libnvos.so"):
        preflight.check_dependencies()


@pytest.mark.parametrize("spec", [None, SimpleNamespace(origin=None),
                                   SimpleNamespace(origin="/missing/tensorrt/__init__.py")])
def test_build_preflight_rejects_missing_trt_artifacts(preflight, monkeypatch, spec):
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: spec)
    with pytest.raises(RuntimeError, match="TensorRT Python package is missing"):
        preflight.inspect_trt_installation()


def test_missing_ros_type_support_fails_before_server_start(preflight, monkeypatch):
    dependency_fixture(preflight, monkeypatch)
    original = preflight.importlib.import_module
    class BrokenMeta(type):
        _TYPE_SUPPORT = None
        @classmethod
        def __import_type_support__(cls):
            pass
    class BrokenMessage(metaclass=BrokenMeta):
        pass
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name:
                        SimpleNamespace(AudioChunk=BrokenMessage) if name == "audio_msgs.msg" else original(name))
    with pytest.raises(RuntimeError, match="audio_msgs.msg.AudioChunk"):
        preflight.check_dependencies()


def test_overlay_pins_validated_base_and_copies_whole_application():
    text = (TOOLS.parent / "Dockerfile.jetson").read_text()
    assert "@sha256:0c8d533b94a858fd6c8571c188ead15b95645ead0f79f589272db40dff662f3a" in text
    for source in ("main.py", "plugins/", "utils/", "config.yaml", "deploy/"):
        assert f"COPY perception/{source}" in text
    assert "COPY perception/tools/moss_platform_smoke.py" in text
    assert "COPY perception/tools/moss_benchmark.py" in text
    assert "source /opt/ros/humble/install/setup.bash" in text
    assert "source /ros_ws/install/setup.bash" in text
    assert "moss_runtime_preflight.py --build-check" in text
    assert "--lock /opt/moss-install/moss_frontend_aarch64.lock --build-check" in text
    assert "COPY perception/tools/moss_runtime_preflight.py /opt/moss-install/" in text
    assert "moss_build_trt.py" not in text
