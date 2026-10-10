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
cd /work
exec "$frontend" /work/main.py "$@"
