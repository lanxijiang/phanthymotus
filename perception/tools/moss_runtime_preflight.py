"""Check the self-contained JP6.1 MOSS image without loading model weights."""
from __future__ import annotations

import argparse
import ctypes as ct
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys


EXPECTED = {"numpy": "1.26.4", "tensorrt": "10.4.0", "sentencepiece": "0.2.0",
            "pynini": "2.1.7", "WeTextProcessing": "1.2.0"}
FRONTEND_ROOT = Path("/opt/moss-frontend/env")


def inspect_trt_installation() -> dict:
    """Check installed artifacts without linking Jetson's driver libraries.

    Docker build does not expose the host libnvos/driver mounts. A module spec
    is installation evidence only; normal startup must import and version-check
    the actual TensorRT module before any server can become available.
    """
    spec = importlib.util.find_spec("tensorrt")
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise RuntimeError("TensorRT Python package is missing from the image")
    try:
        version = importlib.metadata.version("tensorrt")
    except importlib.metadata.PackageNotFoundError:
        version = None  # NVIDIA's vendor/apt package may omit pip dist-info.
    return {"path": spec.origin, "distribution_version": version,
            "imported": False, "runtime_version_check_pending": True}


def check_dependencies(*, build_check: bool = False) -> dict:
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("MOSS deployment requires Jetson Orin Linux ARM64")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("MOSS deployment requires the existing CPython 3.10 interpreter")
    report = {}
    for distribution, expected in EXPECTED.items():
        if distribution == "tensorrt" and build_check:
            report[distribution] = inspect_trt_installation()
            continue
        # NVIDIA's vendor/apt TensorRT may have no pip dist-info record. Its
        # imported module version identifies the runtime used to load plans.
        actual = (importlib.import_module("tensorrt").__version__
                  if distribution == "tensorrt" else importlib.metadata.version(distribution))
        if actual != expected:
            raise RuntimeError(f"MOSS requires {distribution}=={expected}; found {actual}")
        report[distribution] = actual
    for name in ("numpy", "tensorrt", "sentencepiece", "pynini", "pywrapfst",
                 "tn.chinese.normalizer", "tn.english.normalizer"):
        if name == "tensorrt" and build_check:
            continue
        module = importlib.import_module(name)
        if name in {"pynini", "pywrapfst", "tn.chinese.normalizer", "tn.english.normalizer"}:
            if not Path(module.__file__).resolve().is_relative_to(FRONTEND_ROOT):
                raise RuntimeError(f"MOSS frontend imported outside its image prefix: {name}")
    # Import the adapter glue too: an import failure must stop startup before
    # the evaluator repeatedly issues requests to an unusable TTS instance.
    importlib.import_module("plugins.moss_deploy.runtime")
    importlib.import_module("onnxruntime")
    importlib.import_module("rclpy")
    importlib.import_module("plugins.tts")
    # Generated ROS message modules can import while their native type support
    # is missing. Construct each actual TTS transport type and load its support.
    for package, name in (("std_msgs.msg", "String"), ("audio_msgs.msg", "AudioChunk")):
        message_type = getattr(importlib.import_module(package), name)
        message = message_type()
        type(message).__class__.__import_type_support__()
        if type(message).__class__._TYPE_SUPPORT is None:
            raise RuntimeError(f"ROS message type support is unavailable: {package}.{name}")
    return report


def check_gpu() -> dict:
    # Do not import torch for this check. The TRT inference process otherwise
    # avoids it, and loading it in three containers would add unnecessary RAM.
    cuda = ct.CDLL("libcuda.so.1")
    signatures = {"cuInit": [ct.c_uint], "cuDeviceGetCount": [ct.POINTER(ct.c_int)],
                  "cuDeviceGet": [ct.POINTER(ct.c_int), ct.c_int],
                  "cuDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]}
    for name, arguments in signatures.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = arguments, ct.c_int

    def call(name, *args):
        status = getattr(cuda, name)(*args)
        if status:
            raise RuntimeError(f"MOSS requires a working GPU: {name} returned CUDA status {status}")

    call("cuInit", 0)
    count, device, major, minor = (ct.c_int() for _ in range(4))
    call("cuDeviceGetCount", ct.byref(count))
    if count.value < 1:
        raise RuntimeError("MOSS requires a visible GPU; expose the Jetson NVIDIA runtime")
    call("cuDeviceGet", ct.byref(device), 0)
    call("cuDeviceGetAttribute", ct.byref(major), 75, device)
    call("cuDeviceGetAttribute", ct.byref(minor), 76, device)
    if (major.value, minor.value) != (8, 7):
        raise RuntimeError(f"Packaged MOSS plans require Orin SM 8.7; found {major.value}.{minor.value}")
    return {"device": 0, "compute_capability": "8.7"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-check", action="store_true",
                        help="Check image dependencies without linking TRT driver libraries or requesting a GPU")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report = {"check_mode": "build_installation" if args.build_check else "runtime_import_and_gpu",
              "dependencies": check_dependencies(build_check=args.build_check)}
    if not args.build_check:
        report["gpu"] = check_gpu()
    print("[moss-preflight] " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
