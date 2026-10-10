#!/usr/bin/env python3
"""Smoke test the real MOSS adapter without starting perception or ROS.

Run from the repository root (or anywhere with this checkout available):

    python3 perception/tools/moss_smoke.py
    python3 perception/tools/moss_smoke.py --instances 3 --concurrent

The model directory and asset URL default to perception/config.yaml; explicit
arguments override them. The script writes one JSON result and exits nonzero
if initialization or any synthesis fails. Timings start at the adapter call,
after model initialization and optional warmup. The default one-runtime check
matches one judgeflow perception container. The three-runtime option tests only
one Python process; it does not prove that three separate containers fit on the
shared Orin GPU.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


PERCEPTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_DIR))

from plugins.moss_tts import MossTTSAdapter  # noqa: E402


SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
DEFAULT_TEXT = "你好，欢迎使用第四范式。"
CONCURRENT_TEXTS = ("你好。", "请注意前方。", "欢迎使用第四范式。")


def _tts_config(path: Path) -> dict[str, str]:
    """Read only the simple TTS scalar settings used by this repository."""
    values: dict[str, str] = {}
    in_tts = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^  tts:\s*(?:#.*)?$", line):
            in_tts = True
            continue
        if in_tts and re.match(r"^  [a-zA-Z_][a-zA-Z_0-9]*:", line):
            break
        if not in_tts:
            continue
        match = re.match(r"^    (model_dir|moss_asset_url|moss_language|moss_instances|speaker_id):\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(2)
        if value.startswith('"') and value.endswith('"'):
            value = json.loads(value)
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[match.group(1)] = value
    if not in_tts or "model_dir" not in values:
        raise ValueError(f"Could not find TTS model_dir in {path}")
    return values


def _synthesize(adapter: MossTTSAdapter, text: str) -> dict:
    start = time.perf_counter()
    first_pcm_s = None
    total_bytes = 0
    has_signal = False
    chunks = 0
    for pcm in adapter.synthesize_stream(text):
        if not isinstance(pcm, bytes):
            raise TypeError(f"MOSS returned {type(pcm).__name__}, expected PCM bytes")
        if not pcm:
            continue
        if first_pcm_s is None:
            first_pcm_s = time.perf_counter() - start
        total_bytes += len(pcm)
        has_signal |= any(pcm)
        chunks += 1
    elapsed_s = time.perf_counter() - start
    if total_bytes == 0 or not has_signal:
        raise RuntimeError(f"MOSS returned no audible PCM for {text!r}")
    if total_bytes % BYTES_PER_SAMPLE:
        raise RuntimeError(f"MOSS returned an odd number of PCM16 bytes: {total_bytes}")
    audio_s = total_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    return {
        "text": text,
        "first_pcm_s": round(first_pcm_s, 4),
        "elapsed_s": round(elapsed_s, 4),
        "audio_s": round(audio_s, 4),
        "rtf": round(elapsed_s / audio_s, 4),
        "pcm_bytes": total_bytes,
        "chunks": chunks,
        "nonempty_audio": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PERCEPTION_DIR / "config.yaml")
    parser.add_argument("--model-dir", help="Override TTS model_dir from config.yaml")
    parser.add_argument("--asset-url", help="Override moss_asset_url from config.yaml")
    parser.add_argument("--speaker-id", type=int, help="Override speaker_id from config.yaml")
    parser.add_argument("--language", help="Override moss_language from config.yaml")
    parser.add_argument("--instances", type=int, help="Override moss_instances from config.yaml (default: 1)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Short single-case text")
    parser.add_argument("--concurrent", action="store_true", help="Single-process stress test; requires explicit --instances 3")
    parser.add_argument("--skip-warmup", action="store_true", help="Diagnose cold runtime behavior")
    args = parser.parse_args()

    stage = "config"
    result: dict = {"ok": False}
    adapter = None
    try:
        cfg = _tts_config(args.config)
        model_dir = args.model_dir or os.environ.get("MOSS_MODEL_DIR") or cfg["model_dir"]
        asset_url = args.asset_url or os.environ.get("MOSS_ASSET_URL") or cfg.get("moss_asset_url")
        speaker_id = args.speaker_id if args.speaker_id is not None else int(cfg.get("speaker_id", "0"))
        language = args.language or cfg.get("moss_language", "auto")
        instances = args.instances if args.instances is not None else int(cfg.get("moss_instances", "1"))
        if args.concurrent and args.instances != 3:
            raise ValueError("--concurrent requires explicit --instances 3; judgeflow uses three separate one-runtime containers")
        result.update(model_dir=model_dir, language=language, speaker_id=speaker_id,
                      instances=instances, scope="single_process")

        stage = "initialize"
        start = time.perf_counter()
        adapter = MossTTSAdapter(
            model_dir, speaker_id=speaker_id, language=language,
            instances=instances, asset_url=asset_url,
        )
        result["initialize_s"] = round(time.perf_counter() - start, 4)

        if not args.skip_warmup:
            stage = "warmup"
            start = time.perf_counter()
            result["warmup_pcm_bytes"] = adapter.warmup()
            result["warmup_s"] = round(time.perf_counter() - start, 4)

        stage = "single"
        result["single"] = _synthesize(adapter, args.text)

        if args.concurrent:
            stage = "concurrent"
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(_synthesize, adapter, text) for text in CONCURRENT_TEXTS]
                result["concurrent"] = [future.result() for future in futures]

        result["ok"] = True
    except Exception as error:
        result.update(stage=stage, error_type=type(error).__name__, error=str(error))
        traceback.print_exc(file=sys.stderr)
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                result.update(ok=False, stage="close", error_type=type(error).__name__, error=str(error))
                traceback.print_exc(file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
