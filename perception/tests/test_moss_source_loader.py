"""The built-in TRT voice path must not import custom-audio dependencies."""

from __future__ import annotations

import builtins
import sys

import pytest

from plugins.moss_deploy.assets import load_original_source


SOURCE = '''from __future__ import annotations
import torch
import torchaudio
from ort_cpu_runtime import OrtCpuRuntime

class OnnxTtsRuntime(OrtCpuRuntime):
    def builtin_codes(self):
        return [1, 2, 3]

    def _load_reference_audio(self, path):
        """Keep the original audio loader and its docstring."""
        return torchaudio.load(path), torch.float32
'''


@pytest.fixture
def source_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    # Register restoration even if the caller's process has no module yet.
    for name in ("onnx_tts_runtime", "ort_cpu_runtime", "torch", "torchaudio"):
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    (tmp_path / "onnx_tts_runtime.py").write_text(SOURCE)
    (tmp_path / "ort_cpu_runtime.py").write_text("class OrtCpuRuntime: pass\n")
    return tmp_path


def test_builtin_voice_import_does_not_require_torch_or_torchaudio(source_dir, monkeypatch):
    original_import = builtins.__import__
    attempted = []

    def reject_audio_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torchaudio"}:
            attempted.append(name)
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_audio_import)
    original_bytes = (source_dir / "onnx_tts_runtime.py").read_bytes()
    module, ort_module = load_original_source(source_dir)
    assert module.OnnxTtsRuntime().builtin_codes() == [1, 2, 3]
    assert issubclass(module.OnnxTtsRuntime, ort_module.OrtCpuRuntime)
    assert attempted == []
    assert (source_dir / "onnx_tts_runtime.py").read_bytes() == original_bytes
    assert load_original_source(source_dir)[0] is module

    with pytest.raises(ModuleNotFoundError) as error:
        module.OnnxTtsRuntime()._load_reference_audio("custom.wav")
    assert error.value.name == "torch"
    assert attempted == ["torch"]


def test_custom_audio_imports_real_modules_on_demand(source_dir):
    # Small on-disk fixture dependencies make their import timing observable.
    (source_dir / "torch.py").write_text("float32 = 'fixture-float32'\n")
    (source_dir / "torchaudio.py").write_text("def load(path): return 'loaded:' + path\n")
    module, _ = load_original_source(source_dir)
    assert "torch" not in sys.modules
    assert "torchaudio" not in sys.modules
    loader = module.OnnxTtsRuntime()._load_reference_audio
    assert loader.__doc__ == "Keep the original audio loader and its docstring."
    assert loader("custom.wav") == ("loaded:custom.wav", "fixture-float32")
    assert sys.modules["torch"].__file__ == str(source_dir / "torch.py")
    assert sys.modules["torchaudio"].__file__ == str(source_dir / "torchaudio.py")


@pytest.mark.parametrize("change", [
    lambda text: text + "\nmodel_dtype = torch.float32\n",
    lambda text: text.replace("return [1, 2, 3]", "return torchaudio.load('other.wav')"),
    lambda text: text.replace("import torch\n", "import torch as torch\n"),
    lambda text: text + "\nimport torchaudio.functional\n",
    lambda text: text.replace("return torchaudio.load(path), torch.float32", "torch = None\n        return torchaudio.load(path), torch.float32"),
    lambda text: text.replace("def _load_reference_audio(self, path):", "def _load_reference_audio(self, path=torch.float32):"),
])
def test_changed_source_dependency_usage_is_rejected(source_dir, change):
    (source_dir / "onnx_tts_runtime.py").write_text(change(SOURCE))
    with pytest.raises(ValueError, match="original MOSS|Original MOSS"):
        load_original_source(source_dir)
    assert "onnx_tts_runtime" not in sys.modules


def test_failed_module_initialization_does_not_cache_partial_runtime(source_dir):
    (source_dir / "onnx_tts_runtime.py").write_text(SOURCE + "\nraise RuntimeError('source failed')\n")
    with pytest.raises(RuntimeError, match="source failed"):
        load_original_source(source_dir)
    assert "onnx_tts_runtime" not in sys.modules
