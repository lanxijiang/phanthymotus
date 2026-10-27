#!/usr/bin/env bash
# Dedicated leaderboard image: select MOSS on prepared JP6 unless an operator
# explicitly opts out. JP5 without MOSS retains the official VITS2 path. This
# must not become the default command of a general JP6 Perception image.
set -eo pipefail
if [[ -z "${MOSS_ENABLE+x}" ]]; then
    if [[ "${MOSS_BUILT_FOR_JP:-}" == 61 && "${MOSS_BUILT_ENABLED:-}" == 1 ]]; then
        export MOSS_ENABLE=1
    else
        export MOSS_ENABLE=0
    fi
fi

case "${MOSS_ENABLE:-0}" in
    0|'')
        export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
        source /opt/ros/humble/install/setup.bash
        source /ros_ws/install/setup.bash
        source /etc/dla-fallback.env
        exec python3 /work/main.py
        ;;
    1)
        test -x /opt/moss-frontend/runtime-python || {
            printf '%s\n' '[moss-startup] MOSS is available only in the prepared JP6 image.' >&2
            exit 1
        }
        cd /work
        config_dir=$(mktemp -d /tmp/moss-deployment.XXXXXX)
        config="$config_dir/config.yaml"
        # This entrypoint is specific to the TTS leaderboard image. Avoid
        # starting unrelated perception plugins in each benchmark container;
        # an operator can explicitly request the full service with value 0.
        export MOSS_TTS_ONLY="${MOSS_TTS_ONLY:-1}"
        /usr/bin/python3 /work/tools/moss_runtime_config.py --moss-config "$config"
        export CONFIG_PATH="$config"
        model_dir="${MOSS_MODEL_DIR:-/models/moss-zh5-fp32-trt-runtime-20261028/assets}"
        export MOSS_TRT_FRAME_PLAN="${MOSS_TRT_FRAME_PLAN:-${model_dir}/plans/local_fixed_sampled_frame/local_fixed_sampled_frame.plan}"
        export MOSS_TRT_FRAME_CONTRACT="${MOSS_TRT_FRAME_CONTRACT:-${model_dir}/candidates/frame-fp32-v2/frame-contract.json}"
        export MOSS_TRT_FRAME_CUDA_GRAPH="${MOSS_TRT_FRAME_CUDA_GRAPH:-1}"
        export MOSS_TRT_PROFILE="${MOSS_TRT_PROFILE:-0}"
        exec /bin/bash /work/tools/moss_entrypoint.sh
        ;;
    *)
        printf '%s\n' '[moss-startup] MOSS_ENABLE must be 0 or 1.' >&2
        exit 1
        ;;
esac
