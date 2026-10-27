#!/usr/bin/env bash
# Prepare Jinghong's original WeText frontend in an isolated ARM64 prefix.
# Run after TRT compilation; this installs/imports classes, not TN graphs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash moss_prepare_frontend.sh --prefix /moss/frontend [options]

  --prefix PATH    /opt/moss-frontend in the image, or below /moss for testing.
  --python PATH    Existing CPython 3.10 with torch/TRT (default /usr/bin/python3).
  --lock FILE      Replay a previously generated conda-explicit.txt.
  --build-check    Defer TensorRT driver linking to mandatory GPU startup checks.
  -h, --help       Show this help.

Installs pinned micromamba, Pynini 2.1.7 and WeTextProcessing 1.2.0 in PATH.
Does not activate conda, alter the shell profile, install torch, or use the GPU.
Writes runtime-python: use this wrapper to run the existing Python interpreter
with only the frontend import/library paths added for that command.
The first solve records all transitive packages in conda-explicit.txt; retain
that lock for exact environment reconstruction. TN graph creation is separate.
EOF
}

die() { printf 'moss_prepare_frontend: %s\n' "$*" >&2; exit 1; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix=''
system_python=/usr/bin/python3
lock_file=''
build_check=0
while (($#)); do
    case "$1" in
        --prefix|--python|--lock)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --prefix) prefix=$2 ;;
                --python) system_python=$2 ;;
                --lock) lock_file=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --build-check) build_check=1; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n "$prefix" ]] || { usage >&2; exit 1; }
[[ "$prefix" = /moss/* || "$prefix" = /opt/moss-frontend ]] \
    || die '--prefix must be /opt/moss-frontend or below our /moss mount'
for command_name in realpath curl tar sha256sum; do
    command -v "$command_name" >/dev/null || die "Missing command: $command_name"
done
prefix=$(realpath -m -- "$prefix")
[[ "$prefix" = /moss/* || "$prefix" = /opt/moss-frontend ]] \
    || die '--prefix resolves outside the permitted frontend directories'
[[ -x "$system_python" ]] || die "Python is not executable: $system_python"
system_python=$(realpath -- "$system_python")
[[ -z "$lock_file" || -f "$lock_file" ]] || die "Missing lock file: $lock_file"
[[ "$prefix" != /opt/moss-frontend || -n "$lock_file" ]] \
    || die 'The deployment prefix requires --lock; do not solve dependencies during image builds'
if [[ -n "$lock_file" ]]; then
    "$system_python" "$script_dir/moss_frontend_lock.py" validate --lock "$lock_file"
fi

# Fail before installing anything on the wrong architecture/interpreter.
"$system_python" - <<'PY'
import platform
import sys
if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
    raise SystemExit("This frontend package set requires Linux ARM64")
if sys.version_info[:2] != (3, 10):
    raise SystemExit("Existing runtime Python must be CPython 3.10")
if platform.python_implementation() != "CPython":
    raise SystemExit("Existing runtime Python must be CPython")
PY

mkdir -p "$prefix/downloads" "$prefix/mamba" "$prefix/work"
env_prefix="$prefix/env"
baseline="$prefix/runtime-before.json"
"$system_python" - "$baseline" "$build_check" "$script_dir" <<'PY'
import importlib
import json
from pathlib import Path
import sys
modules = {}
build_check = sys.argv[2] == "1"
sys.path.insert(0, sys.argv[3])
for name in ("torch", "tensorrt"):
    if name == "tensorrt" and build_check:
        from moss_runtime_preflight import inspect_trt_installation
        modules[name] = inspect_trt_installation()
        continue
    module = importlib.import_module(name)
    modules[name] = {"version": module.__version__, "path": module.__file__}
Path(sys.argv[1]).write_text(json.dumps(modules, indent=2) + "\n")
print("Existing runtime checked; TRT import deferred to GPU startup." if build_check
      else "Existing runtime imports verified; no CUDA context requested.")
PY

download_verified() {
    local url=$1 destination=$2 expected=$3
    if [[ ! -f "$destination" ]]; then
        curl --fail --location --show-error --retry 3 --connect-timeout 20 \
            --output "$destination.part" "$url"
        printf '%s  %s\n' "$expected" "$destination.part" | sha256sum --check --status \
            || die "SHA256 mismatch: $destination.part"
        mv -- "$destination.part" "$destination"
    fi
    printf '%s  %s\n' "$expected" "$destination" | sha256sum --check --status \
        || die "SHA256 mismatch: $destination"
}

# Fixed official conda-forge artifact (metadata from api.anaconda.org).
mamba_archive="$prefix/downloads/micromamba-2.5.0-1.tar.bz2"
download_verified \
    'https://conda.anaconda.org/conda-forge/linux-aarch64/micromamba-2.5.0-1.tar.bz2' \
    "$mamba_archive" \
    'cbe498d65b4173d68875634985cad1b7f2561d4bee58126d084899d3ba3d9cd8'
tar -xjf "$mamba_archive" -C "$prefix/mamba" bin/micromamba
micromamba="$prefix/mamba/bin/micromamba"

if [[ ! -f "$env_prefix/conda-meta/history" ]]; then
    # All selected packages are frontend dependencies; no torch/numpy/CUDA
    # package is requested. --no-rc ignores the container's conda configuration.
    create_args=(--no-rc create -y -r "$prefix/mamba-root" -p "$env_prefix")
    if [[ -n "$lock_file" ]]; then
        create_args+=(--file "$lock_file")
    else
        create_args+=(-c conda-forge --override-channels --strict-channel-priority
            'python=3.10' 'pynini=2.1.7=py310h0992a49_2' 'openfst=1.8.4'
            importlib_resources pip)
    fi
    TMPDIR="$prefix/work" "$micromamba" "${create_args[@]}"
fi

if [[ -n "$lock_file" ]]; then
    "$system_python" "$script_dir/moss_frontend_lock.py" check \
        --lock "$lock_file" --metadata "$env_prefix/conda-meta"
fi

"$system_python" - "$env_prefix" <<'PY'
import json
from pathlib import Path
import sys
metadata = Path(sys.argv[1]) / "conda-meta"
packages = [json.loads(p.read_text()) for p in metadata.glob("*.json")]
records = {p["name"]: p for p in packages}
for forbidden in ("torch", "pytorch", "libtorch", "numpy", "tensorrt"):
    if forbidden in records:
        raise SystemExit(f"Frontend prefix unexpectedly contains {forbidden}")
pynini = records.get("pynini", {})
expected = ("2.1.7", "py310h0992a49_2", "linux-aarch64",
            "fea0f7c4cc4d6cd8a020bab417255b00c1ab6477d02a0c78788ae11520439b6d")
actual = tuple(pynini.get(key) for key in ("version", "build", "subdir", "sha256"))
if actual != expected:
    raise SystemExit(f"Pynini artifact mismatch: {actual!r}")
if records.get("openfst", {}).get("version") != "1.8.4":
    raise SystemExit("Expected OpenFST 1.8.4")
PY

wheel="$prefix/downloads/wetextprocessing-1.2.0-py3-none-any.whl"
download_verified \
    'https://files.pythonhosted.org/packages/d9/81/351722ec61dbdf96cca59edac14ce620be273fa98a9fab9e1a2aa524c3e1/wetextprocessing-1.2.0-py3-none-any.whl' \
    "$wheel" '2c7c1e048ede69b67e8a0921093c08a591bc9d7566ad70f5343686196f97aea9'
TMPDIR="$prefix/work" env -u PYTHONPATH -u PYTHONHOME \
    "$env_prefix/bin/python" -m pip --isolated install \
    --disable-pip-version-check --no-cache-dir --no-index --no-deps "$wheel"

# A subprocess wrapper leaves the caller's environment and original runtime
# packages unchanged. Keep the original interpreter for NVIDIA's torch/TRT.
"$system_python" - "$prefix" "$system_python" <<'PY'
from pathlib import Path
import shlex
import sys
root = Path(sys.argv[1])
site = shlex.quote(str(root / "env/lib/python3.10/site-packages"))
lib = shlex.quote(str(root / "env/lib"))
python = shlex.quote(sys.argv[2])
wrapper = root / "runtime-python"
wrapper.write_text(
    "#!/usr/bin/env bash\nset -euo pipefail\n"
    f"frontend_site={site}\nfrontend_lib={lib}\n"
    'exec env "PYTHONPATH=$frontend_site${PYTHONPATH:+:$PYTHONPATH}" '
    '"LD_LIBRARY_PATH=$frontend_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" '
    f'{python} "$@"\n'
)
wrapper.chmod(0o755)
PY

"$prefix/runtime-python" - "$baseline" "$prefix" "$build_check" "$script_dir" <<'PY'
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

# Check that the selected C++ runtime supports the existing torch and OpenFST.
# TensorRT links driver libraries unavailable during docker build; in that
# explicit mode only compare installation evidence, then require runtime import.
expected = json.loads(Path(sys.argv[1]).read_text())
build_check = sys.argv[3] == "1"
sys.path.insert(0, sys.argv[4])
report = {"runtime": {}, "frontend": {}, "tn_graphs_built": False,
          "trt_import_verified": not build_check,
          "check_mode": "build_installation" if build_check else "runtime_import"}
for name in ("torch", "tensorrt"):
    if name == "tensorrt" and build_check:
        from moss_runtime_preflight import inspect_trt_installation
        actual = inspect_trt_installation()
    else:
        module = importlib.import_module(name)
        actual = {"version": module.__version__, "path": module.__file__}
    if actual != expected[name]:
        raise SystemExit(f"Existing {name} changed: {actual!r} != {expected[name]!r}")
    report["runtime"][name] = actual
for name in ("pynini", "pywrapfst", "importlib_resources",
             "tn.chinese.normalizer", "tn.english.normalizer"):
    module = importlib.import_module(name)
    report["frontend"][name] = module.__file__
for distribution, version in (("pynini", "2.1.7"), ("WeTextProcessing", "1.2.0")):
    actual = importlib.metadata.version(distribution)
    if actual != version:
        raise SystemExit(f"{distribution}: expected {version}, found {actual}")
root = Path(sys.argv[2])
(root / "frontend-import-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

"$system_python" "$script_dir/moss_frontend_lock.py" export \
    --metadata "$env_prefix/conda-meta" --output "$prefix/conda-explicit.txt"
printf 'Frontend imports verified. Run smoke with: %s/runtime-python <moss_smoke.py> ...\n' "$prefix"
printf 'Retain dependency lock: %s/conda-explicit.txt\n' "$prefix"
