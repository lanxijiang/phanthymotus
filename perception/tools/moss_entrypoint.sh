#!/usr/bin/env bash
# Full perception startup using the image-owned MOSS frontend and GPU plans.
set -eo pipefail

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
source /opt/ros/humble/install/setup.bash
source /ros_ws/install/setup.bash
source /etc/dla-fallback.env

frontend=/opt/moss-frontend/runtime-python
if [[ ! -x "$frontend" ]]; then
    printf '%s\n' '[moss-startup] Image frontend is missing; rebuild the JP6.1 image.' >&2
    exit 1
fi

# This checks imports/versions and CUDA availability; it does not deserialize
# weights or build engines. The adapter downloads and verifies packaged assets.
"$frontend" /work/tools/moss_runtime_preflight.py
if [[ -n "${MCP_PORT:-}" || -n "${WS_PORT:-}" ]]; then
    runtime_config_dir=$(mktemp -d /tmp/moss-port-config.XXXXXX)
    runtime_config="$runtime_config_dir/config.yaml"
    # Resolve relative CONFIG_PATH values from the same directory as main.
    if ! (cd /work && "$frontend" /work/tools/moss_runtime_config.py --port-config "$runtime_config"); then
        rm -f -- "$runtime_config"
        rmdir -- "$runtime_config_dir"
        exit 1
    fi
    export CONFIG_PATH="$runtime_config"
fi
cd /work
# Keep Python thread stacks in the existing container log if a native library
# faults. This is diagnostic only; it does not recover or hide a failed process.
exec "$frontend" -X faulthandler /work/main.py "$@"
