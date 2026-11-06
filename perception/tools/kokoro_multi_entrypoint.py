#!/usr/bin/env python3
"""Collect offline Darvin audio through Perception's normal JP6 main module."""

import logging
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile


WORK = Path(__file__).resolve().parents[1]
# Filled with the published, immutable release before submission. Missing pins
# fail explicitly; an unverified or unrelated model is never substituted.
ASSET_URL = "https://www.modelscope.cn/api/v1/models/lanxijiang/Kokoro-Multi-TRT-JP61/repo?Revision=e9a6d1d9ac8fa3d75bff8178cb3312fda2d6f547&FilePath=kokoro-multi-jp61-fp32-v2.tar.gz"
ASSET_SHA256 = "df319fa3827abd3e7b1d0779e7b043485becabcdaa570ecc7174ba8baf611d43"
ASSET_SIZE = 231646880


def enabled():
    jp = os.environ.get("KOKORO_MULTI_BUILT_FOR_JP", "")
    value = os.environ.get("KOKORO_MULTI_ENABLE", "1" if jp == "61" else "0")
    if value not in ("0", "1"):
        raise ValueError("KOKORO_MULTI_ENABLE must be 0 or 1")
    if value == "1" and jp != "61":
        raise RuntimeError("Kokoro Multi requires the prepared JetPack 6.1 image")
    return value == "1"


def prepare_config():
    import yaml
    from utils.model_downloader import ensure_verified_archive, require_models_subpath

    url = os.environ.get("KOKORO_MULTI_ASSET_URL", ASSET_URL)
    sha256 = os.environ.get("KOKORO_MULTI_ASSET_SHA256", ASSET_SHA256)
    size = int(os.environ.get("KOKORO_MULTI_ASSET_SIZE", str(ASSET_SIZE)))
    if not url or not re.fullmatch(r"[0-9a-f]{64}", sha256) or size <= 0:
        raise RuntimeError(
            "Kokoro Multi release is not pinned: set KOKORO_MULTI_ASSET_URL, "
            "KOKORO_MULTI_ASSET_SHA256 and KOKORO_MULTI_ASSET_SIZE"
        )
    model_dir = require_models_subpath(
        os.environ.get("KOKORO_MULTI_MODEL_DIR", "/models/kokoro-multi-trt")
    )
    # A mounted /models cache is reused. Without a mount this is the container's
    # writable layer, and each fresh container needs network access to the URL.
    ensure_verified_archive(
        "kokoro-multi-jp61", model_dir, url, {"sha256": sha256, "size": size}
    )
    if not (Path(model_dir) / "manifest.json").is_file():
        raise RuntimeError("Kokoro Multi release must contain manifest.json at its root")

    source = Path(os.environ.get("CONFIG_PATH", str(WORK / "config.yaml")))
    with source.open() as handle:
        cfg = yaml.safe_load(handle)
    cfg["plugins"] = {"tts": {
        "enabled": True,
        "engine": "kokoro-multi",
        "backend": "trt",
        # Offline collection only: the benchmark drains audio without live playback.
        "benchmark": True,
        "device": "gpu",
        "model_dir": model_dir,
        "language": "auto",
        "warmup": True,
        "speaker_id": 0,
        "speed": 1.0,
    }, "echo_cancellation": {"enabled": False},
       "audio_mixer": {"enabled": False}}
    for key, default in (("mcp_port", 15720), ("ws_port", 15721)):
        port = int(os.environ.get(key.upper(), cfg.get(key, default)))
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid {key}: {port}")
        cfg[key] = port
    return cfg


def main():
    sys.path.insert(0, str(WORK))
    if not enabled():
        # No Kokoro Multi imports, model downloads, or configuration changes.
        runpy.run_path(str(WORK / "main.py"), run_name="__main__")
        return

    import yaml
    logging.basicConfig(level=logging.INFO)
    cfg = prepare_config()
    with tempfile.TemporaryDirectory(prefix="kokoro-multi-config-") as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        os.environ["CONFIG_PATH"] = str(config)
        logging.info("Kokoro Multi JP6: MCP port %s, WS port %s, cache %s",
                     cfg["mcp_port"], cfg["ws_port"], cfg["plugins"]["tts"]["model_dir"])
        runpy.run_path(str(WORK / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
