#!/usr/bin/env bash
# build_perception.sh — 构建 perception-stack（感知层）镜像并推送
#
# Usage:
#   ./build_perception.sh                           # CPU 版（默认），交互选源
#   ./build_perception.sh --variant jetson          # Jetson GPU 版
#   ./build_perception.sh --variant jetson --mirror tuna
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/build_common.sh"

ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
fi

eval "$(parse_mirror_arg "$@")"

# ── 解析参数 ─────────────────────────────────────────────────────────
VARIANT="cpu"
JP_VERSION="${JP_VERSION:-6}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --jp-version) JP_VERSION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "${JP_VERSION}" in
    5|511|5.11|5.1.1)
        JP_TAG="511"; JP_LABEL="5.1.1"
        VITS2_MODEL_URL="${VITS2_MODEL_URL:-http://172.28.4.81:34567/liaoqianqian/models/lc1-jetson/lc1_male_v6_frontend_146k_jp5_nocudnn_nojit_runtime.tar.gz}"
        VITS2_MODEL_SHA256="${VITS2_MODEL_SHA256:-4a006ceb74325d11c506f2e054faffbfce6e0ca80946ffaaa8ef5e03fc26f96a}"
        ;;
    6|61|6.1)
        JP_TAG="61"; JP_LABEL="6.1"
        VITS2_MODEL_URL="${VITS2_MODEL_URL:-http://172.28.4.81:34567/liaoqianqian/models/lc1-jetson/lc1_male_v6_frontend_146k_jp6_runtime.tar.gz}"
        VITS2_MODEL_SHA256="${VITS2_MODEL_SHA256:-e3a26c913285ba26d30624be50751fa27c8d6ecb8458ac1ad3b0f0d196794b42}"
        ;;
    *) echo "Unknown JP_VERSION=${JP_VERSION} (supported: 5, 5.11, 6, 6.1)" >&2; exit 1 ;;
esac
VITS_MODEL_RELEASE="${VITS_MODEL_RELEASE:-lc1_male_v6_frontend_146k}"

RESOURCE_CENTER_URL="${RESOURCE_CENTER_URL:-https://motus.phanthy.com}"

# If registry not configured, build locally only
PUSH_ENABLED=true
if [ -z "${REGISTRY:-}" ] || [ -z "${REGISTRY_USER:-}" ] || [ -z "${REGISTRY_PASSWORD:-}" ] || [ -z "${IMAGE_NAMESPACE:-}" ]; then
    echo "[info] Registry not configured — building locally only (no push)."
    PUSH_ENABLED=false
    REGISTRY="${REGISTRY:-local}"
    IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-phanthy-motus}"
fi

DATE="$(date +%y%m%d)"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD)"

# ── 根据 variant 选择 Dockerfile、context、tag ────────────────────────
case "${VARIANT}" in
    cpu)
        DOCKERFILE="${REPO_ROOT}/perception/Dockerfile"
        BUILD_CONTEXT="${REPO_ROOT}/perception"
        TAG="release.${DATE}.${COMMIT}"
        ;;
    jetson)
        DOCKERFILE="${REPO_ROOT}/perception/Dockerfile.jetson"
        BUILD_CONTEXT="${REPO_ROOT}"
        TAG="release.${DATE}.${COMMIT}-jetson-jp${JP_TAG}"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}  (supported: cpu, jetson)"
        exit 1
        ;;
esac

FULL_IMAGE="${REGISTRY}/${IMAGE_NAMESPACE}/perception:${TAG}"

echo "============================================"
echo "Building perception-stack image"
echo "Variant: ${VARIANT}"
if [[ "${VARIANT}" == "jetson" ]]; then
    echo "JetPack: ${JP_LABEL} (base tag jp${JP_TAG})"
    echo "Model  : ${VITS_MODEL_RELEASE}"
fi
echo "Image  : ${FULL_IMAGE}"
echo "Arch   : ${ARCH} (native=${IS_ARM64})"
echo "Push   : ${PUSH_ENABLED}"
echo "============================================"

if ${PUSH_ENABLED}; then
    echo "${REGISTRY_PASSWORD}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
fi

select_mirror

if [[ "${VARIANT}" == "jetson" ]]; then
    do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}" \
        "JP_VERSION=${JP_TAG}" \
        "VITS_MODEL_RELEASE=${VITS_MODEL_RELEASE}" \
        "VITS2_MODEL_URL=${VITS2_MODEL_URL}" \
        "VITS2_MODEL_SHA256=${VITS2_MODEL_SHA256}"
else
    do_build "${DOCKERFILE}" "${BUILD_CONTEXT}" "${FULL_IMAGE}"
fi

if ${PUSH_ENABLED}; then
    do_push "${FULL_IMAGE}"
    echo ""
    echo "Done. Image pushed: ${FULL_IMAGE}"
else
    echo ""
    echo "Done. Image built locally: ${FULL_IMAGE}"
fi

# ── 注册到 resource-center（可选）────────────────────────────────────────────
if ${PUSH_ENABLED} && [ -n "${RESOURCE_CENTER_API_KEY:-}" ]; then
    SYNC_CONFIRM="y"
    if [ -t 0 ] || [ -e /dev/tty ]; then
        printf "Sync to resource-center (%s)? [Y/n]: " "${RESOURCE_CENTER_URL}" >/dev/tty
        read -r SYNC_CONFIRM </dev/tty || SYNC_CONFIRM="y"
    fi
    if [[ ! "${SYNC_CONFIRM}" =~ ^[Nn] ]]; then
        echo "Registering image to resource-center (${RESOURCE_CENTER_URL})..."
        HTTP_STATUS=$(curl -s -o /tmp/rc_register_resp.json -w "%{http_code}" \
            -X POST "${RESOURCE_CENTER_URL}/api/admin/register" \
            -H "Content-Type: application/json" \
            -H "x-api-key: ${RESOURCE_CENTER_API_KEY}" \
            -d "{
                \"imageRef\": \"${FULL_IMAGE}\",
                \"registryImage\": \"perception\",
                \"tag\": \"${TAG}\",
                \"category\": \"perception\",
                \"name\": \"Perception Stack\",
                \"description\": \"语音感知套件 — ASR 语音识别 + TTS 语音合成 + VAD 静音检测 + 唤醒词检测\"
            }")

        if [ "${HTTP_STATUS}" = "200" ] || [ "${HTTP_STATUS}" = "201" ]; then
            echo "Registered: $(cat /tmp/rc_register_resp.json)"
        else
            echo "Warning: registration failed (HTTP ${HTTP_STATUS}): $(cat /tmp/rc_register_resp.json)"
        fi
    else
        echo "跳过同步。"
    fi
fi
