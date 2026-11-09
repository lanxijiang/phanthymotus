"""Private JP61 startup profile for the existing public ``tts`` tool.

The original main.py is executed without editing, module aliases or import hooks.
Release pins are resolved by assets.py; an unpublished/unpinned bundle fails closed.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

_VENV_ROOT = Path('/opt/xiaoyou-v6')


def _flag(name, default):
    value = os.environ.get(name, default)
    if value not in ('0', '1'):
        raise ValueError(name + ' must be 0 or 1')
    return value == '1'


def _prepare_ros_log_dir():
    value = os.environ.setdefault('ROS_LOG_DIR', '/tmp/xiaoyou01/ros-log')
    try:
        if not value:
            raise ValueError('ROS_LOG_DIR must name a directory')
        directory = Path(value)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(prefix='.xiaoyou01-write-', dir=directory):
            pass
    except (OSError, ValueError) as error:
        raise RuntimeError('Cannot prepare xiaoyou01 ROS log directory '
                           + repr(value) + ': ' + str(error)) from error
    return directory


def _isolated_python():
    marker = _VENV_ROOT / 'jetpack-version'
    built_for = os.environ.get('XIAOYOU01_BUILT_FOR_JP', '61')
    if (built_for != '61' or not marker.is_file()
            or marker.read_text().strip() != '61'):
        raise RuntimeError('xiaoyou01 requires its provisioned JP61 GPU environment; '
                           'no JP511, CPU or older engine fallback is allowed')
    python = _VENV_ROOT / 'bin/python3'
    if not python.is_file():
        raise RuntimeError('Missing isolated xiaoyou01 interpreter')
    return python


def _profile_config(perception_root):
    # Imported only after re-exec into the isolated JP61 environment.
    import yaml
    from plugins.xiaoyou_v6.assets import ensure_runtime_bundle, resolve_runtime_manifest_pin

    source = Path(os.environ.get('CONFIG_PATH') or perception_root / 'config.yaml')
    with source.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('CONFIG_PATH must contain a configuration mapping')
    # Darvin assigns each host-network container its own server ports via env.
    # Bridge only explicit overrides into the private generated configuration.
    for env_name, key in (('MCP_PORT', 'mcp_port'), ('WS_PORT', 'ws_port')):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(env_name + ' must be an integer port in 1..65535') from None
        if not 1 <= port <= 65535:
            raise ValueError(env_name + ' must be an integer port in 1..65535')
        config[key] = port
    plugins = config.get('plugins', {})
    if not isinstance(plugins, dict) or any(not isinstance(v, dict) for v in plugins.values()):
        raise ValueError('plugins configuration must contain mappings')
    original_tts = plugins.get('tts', {})
    pin = resolve_runtime_manifest_pin(original_tts.get('runtime_manifest_sha256'))
    model_dir = os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/xiaoyou01'
    verified_root = ensure_runtime_bundle(model_dir, pin)
    benchmark = _flag('XIAOYOU01_TTS_BENCHMARK', '1')
    # Preserve non-plugin server settings; never modify the source config file.
    config['plugins'] = {name: {**value, 'enabled': False} for name, value in plugins.items()}
    # Both default to enabled in main.py, so they must be explicit even if absent.
    config['plugins']['echo_cancellation'] = {'enabled': False}
    config['plugins']['audio_mixer'] = {'enabled': False}
    config['plugins']['tts'] = {
        'enabled': True,
        'engine': 'xiaoyou01',
        'backend': 'tensorrt_fp32',
        'model_dir': str(verified_root),
        'runtime_manifest_sha256': pin,
        'language': 'auto',
        'speaker_id': 0,
        'speed': 1.0,
        'cpu_threads': 2,
        'warmup': True,
        'benchmark': benchmark,
        # Greedy delivery is only the explicitly named Darvin offline profile.
        'pace_audio': not benchmark,
        'subscriber_wait_s': 3.0,
    }
    return config


def _bootstrap(main_path):
    perception_root = main_path.parent
    original_path = list(sys.path)
    original_argv = sys.argv
    original_config = os.environ.get('CONFIG_PATH')
    sys.path.insert(0, str(perception_root))
    try:
        config = _profile_config(perception_root)
        with tempfile.TemporaryDirectory(prefix='xiaoyou01-config-') as directory:
            config_path = Path(directory) / 'config.json'
            # JSON is accepted by the original main.py YAML safe_load reader.
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
            config_path.chmod(0o600)
            os.environ['CONFIG_PATH'] = str(config_path)
            sys.argv = [str(main_path)] + original_argv[1:]
            runpy.run_path(str(main_path), run_name='__main__')
    finally:
        sys.path[:] = original_path
        sys.argv = original_argv
        if original_config is None:
            os.environ.pop('CONFIG_PATH', None)
        else:
            os.environ['CONFIG_PATH'] = original_config


def main():
    main_path = Path(__file__).resolve().parents[2] / 'main.py'
    if not main_path.is_file():
        raise RuntimeError('Missing original perception/main.py')
    if not _flag('XIAOYOU01_TTS_ENABLE', '1'):
        # Explicit opt-out keeps the caller's interpreter/config and original main.
        os.execv(sys.executable, [sys.executable, '-B', str(main_path)] + sys.argv[1:])
        raise RuntimeError('execv unexpectedly returned')
    python = _isolated_python()
    _prepare_ros_log_dir()
    if Path(sys.prefix).resolve() != _VENV_ROOT.resolve():
        # Re-enter this same file; no extra bootstrap module or public entry is added.
        os.execv(str(python), [str(python), '-B', str(Path(__file__).resolve())] + sys.argv[1:])
        raise RuntimeError('execv unexpectedly returned')
    _bootstrap(main_path)


if __name__ == '__main__':
    main()
