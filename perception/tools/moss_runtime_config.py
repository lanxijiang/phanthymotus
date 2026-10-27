"""Apply MOSS container port overrides through main's existing CONFIG_PATH."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def write_port_config(destination: str | Path, *, environ=None,
                      default_config: str | Path = "/work/config.yaml") -> Path | None:
    """Write a private config for main's existing CONFIG_PATH interface.

    Empty port variables retain the normal startup path without reading or
    writing configuration. The selected source is never changed or logged.
    """
    environ = os.environ if environ is None else environ

    def port(value, name):
        if (isinstance(value, bool) or not isinstance(value, (str, int))
                or isinstance(value, str) and (not value.isascii() or not value.isdecimal())):
            raise ValueError(f"{name} must be an integer port")
        number = int(value)
        if not 1 <= number <= 65535:
            raise ValueError(f"{name} must be between 1 and 65535")
        return number

    overrides = {key: port(environ[name], name)
                 for name, key in (("MCP_PORT", "mcp_port"), ("WS_PORT", "ws_port"))
                 if environ.get(name)}
    if not overrides:
        return None

    import yaml
    source = Path(environ.get("CONFIG_PATH", default_config))
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("MOSS port config must not overwrite its source")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MOSS source configuration must be a YAML mapping")
    config.update(overrides)
    mcp = port(config.get("mcp_port", 15720), "mcp_port")
    websocket = port(config.get("ws_port", 15721), "ws_port")
    if mcp == websocket:
        raise ValueError("MCP and WebSocket ports must be different")
    content = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def write_moss_config(destination: str | Path, *, environ=None,
                      default_config: str | Path = "/work/config.yaml") -> Path:
    """Create an opt-in MOSS config without changing the original config file."""
    import yaml
    environ = os.environ if environ is None else environ
    source = Path(environ.get("CONFIG_PATH", default_config))
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("MOSS config must not overwrite its source")
    count = environ.get("MOSS_INSTANCES", "1")
    if count not in ("1", "2", "3"):
        raise ValueError("MOSS_INSTANCES must be 1, 2 or 3")
    transport = environ.get("MOSS_TRANSPORT_MODE", "paced")
    if transport not in ("paced", "as_generated"):
        raise ValueError("MOSS_TRANSPORT_MODE must be paced or as_generated")
    language = environ.get("MOSS_LANGUAGE", "auto")
    if language not in ("auto", "zh", "ja", "ru", "en", "es", "fr", "th", "de"):
        raise ValueError("unsupported MOSS_LANGUAGE")
    model_dir = environ.get("MOSS_MODEL_DIR", "/models/moss-zh5-fp32-trt-runtime-20261028/assets")
    if not model_dir.startswith("/models/") or ".." in Path(model_dir).parts:
        raise ValueError("MOSS_MODEL_DIR must be under /models")
    only = environ.get("MOSS_TTS_ONLY", "0")
    if only not in ("0", "1"):
        raise ValueError("MOSS_TTS_ONLY must be 0 or 1")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("plugins"), dict):
        raise ValueError("MOSS source configuration must contain plugins")
    plugins = config["plugins"]
    tts = plugins.get("tts")
    if not isinstance(tts, dict):
        raise ValueError("MOSS source configuration must contain tts settings")
    if only == "1":
        for name, row in plugins.items():
            if name != "tts" and isinstance(row, dict) and "enabled" in row:
                row["enabled"] = False
    tts.update({"enabled": True, "engine": "moss-multi", "backend": "trt",
                "model_dir": model_dir,
                "moss_asset_url": environ.get("MOSS_ASSET_URL", "http://172.28.4.81:34567/i-lanxijiang/phanthymotus-tts/moss-orin-jp61-zh5-fp32-trt-runtime-20261028.tar"),
                "moss_language": language, "moss_transport_mode": transport,
                "moss_instances": int(count), "moss_subscriber_settle_s": 0.5,
                "speaker_id": 2, "speed": 1.0, "warmup": True})
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port-config", type=Path,
                       help="Write a private config with validated MCP_PORT/WS_PORT overrides")
    group.add_argument("--moss-config", type=Path,
                       help="Write an opt-in MOSS TTS config from the existing configuration")
    args = parser.parse_args(argv)
    output = write_moss_config(args.moss_config) if args.moss_config else write_port_config(args.port_config)
    if output is not None:
        print(f"[moss-startup] Port configuration written to {output}", flush=True)


if __name__ == "__main__":
    main()
