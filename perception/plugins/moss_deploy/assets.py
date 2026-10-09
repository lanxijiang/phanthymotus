"""Read-only local assets for the original MOSS multilingual checkpoint."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def _literal_assignment(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Original source does not define {name}: {path}")


def load_original_source(source_dir: Path):
    """Import reviewed local code without constructing its auto-download path."""
    source_dir = source_dir.resolve()
    for name in ("ort_cpu_runtime", "onnx_tts_runtime"):
        existing = sys.modules.get(name)
        if existing is not None and Path(existing.__file__).resolve().parent != source_dir:
            raise RuntimeError(f"Another MOSS source directory is already imported for {name}")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return importlib.import_module("onnx_tts_runtime"), importlib.import_module("ort_cpu_runtime")


def load_original_prompting(path: Path):
    """Execute original helper bodies; skip only their unused config import.

    All function bodies/constants come from the local checkpoint's prompting
    source. Future annotations let a read-only config namespace stand in for
    the model's Transformers configuration object; no model is loaded here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "configuration_moss_tts_nano")]
    namespace = {"__name__": "moss_deploy.original_prompting"}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class ModelAssets:
    def __init__(self, analysis_path: str | Path, source_dir: str | Path, *, streaming_codec_meta: str | Path | None = None):
        self.analysis_path = Path(analysis_path).resolve()
        self.analysis = json.loads(self.analysis_path.read_text(encoding="utf-8"))
        source = Path(source_dir).resolve()
        self.source_dir = source if (source / "ort_cpu_runtime.py").is_file() else source / "MOSS-TTS-Nano"
        for filename in ("ort_cpu_runtime.py", "onnx_tts_runtime.py", "moss_tts_nano_runtime.py", "app_gpu.py"):
            if not (self.source_dir / filename).is_file():
                raise FileNotFoundError(f"Original local MOSS code is missing: {self.source_dir / filename}")
        self.graphs = {graph["name"]: graph for graph in self.analysis["graphs"]}
        required = {"global_unified", "local_cached_step", "codec_encoder", "codec_decoder"}
        if not required.issubset(self.graphs):
            raise ValueError(f"Missing original-model export graphs: {sorted(required - self.graphs.keys())}")
        self.graph_paths = {}
        for name, graph in self.graphs.items():
            path = Path(graph["file"])
            path = path if path.is_absolute() else self.analysis_path.parent / path
            if not path.is_file():
                raise FileNotFoundError(path)
            self.graph_paths[name] = path.resolve()
        # Compatible analyses can point to alternate same-IO graphs. Metadata
        # and tokenizer belong to the canonical compact graph directory.
        global_dir = self.graph_paths["global_unified"].parent
        meta_candidates = [global_dir / "tts_browser_onnx_meta.json", self.analysis_path.parent / "tts_browser_onnx_meta.json"]
        self.tts_meta_path = next((path for path in meta_candidates if path.is_file()), None)
        if self.tts_meta_path is None:
            raise FileNotFoundError("Original MOSS tts_browser_onnx_meta.json is required")
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        frontend = self.analysis.get("frontend", {}).get("directory", "frontend")
        self.frontend_dir = global_dir / frontend
        if not self.frontend_dir.is_dir():
            self.frontend_dir = self.tts_meta_path.parent / "frontend"
        self.config_path = self.frontend_dir / "config.json"
        self.tokenizer_path = self.frontend_dir / "tokenizer.model"
        self.prompting_path = self.frontend_dir / "prompting.py"
        for path in (self.config_path, self.tokenizer_path, self.prompting_path):
            if not path.is_file():
                raise FileNotFoundError(f"Local original checkpoint frontend asset is missing: {path}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        model = self.tts_meta["model_config"]
        if int(model["n_vq"]) != 16 or int(model["row_width"]) != 17:
            raise ValueError("Runtime expects this MOSS checkpoint's 16-codebook row format")
        if any(int(self.config[key]) != int(model[key]) for key in ("audio_pad_token_id", "audio_start_token_id", "audio_end_token_id", "audio_user_slot_token_id", "audio_assistant_slot_token_id")):
            raise ValueError("Tokenizer/control metadata does not match the original exported checkpoint")
        self.voice_files = _literal_assignment(self.source_dir / "moss_tts_nano_runtime.py", "_DEFAULT_VOICE_FILES")
        self.default_voice = _literal_assignment(self.source_dir / "moss_tts_nano_runtime.py", "DEFAULT_VOICE")
        self.languages = _literal_assignment(self.source_dir / "app_gpu.py", "SUPPORTED_LANGUAGES")
        script = ast.parse((self.source_dir / "app_gpu.py").read_text(encoding="utf-8"))
        names = {"_LATIN_STOPWORDS", "_LATIN_CHAR_HINTS"}
        selected = []
        for node in script.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id in names for target in targets) or isinstance(node, ast.FunctionDef) and node.name == "detect_language" or isinstance(node, ast.ImportFrom) and node.module == "__future__":
                selected.append(node)
        namespace = {"re": re}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(self.source_dir / "app_gpu.py"), "exec"), namespace)
        self.detect_language = namespace["detect_language"]
        self.voice_paths = {name: (self.source_dir / "assets/audio" / row[0]).resolve() for name, row in self.voice_files.items()}
        for path in self.voice_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Original reference voice asset is missing; no download attempted: {path}")
        metadata = self.analysis["metadata"]
        if (int(metadata["sample_rate"]), int(metadata["channels"]), int(metadata["num_quantizers"])) != (48000, 2, 16):
            raise ValueError("Codec metadata does not match original 48 kHz stereo MOSS codec")
        self.codec_meta = {"codec_config": {"sample_rate": 48000, "channels": 2, "num_quantizers": 16}}
        self.streaming_path = None
        self.streaming_source_step_path = None
        self.precomputed_reference_codes = {}
        if streaming_codec_meta is None and "codec_decode_step" in self.graph_paths:
            raise ValueError("Analysis includes codec_decode_step; its original streaming codec metadata is required")
        if streaming_codec_meta is not None:
            self.streaming_path = Path(streaming_codec_meta).resolve()
            streaming = json.loads(self.streaming_path.read_text(encoding="utf-8"))
            if streaming["codec_config"] != self.codec_meta["codec_config"]:
                # The original metadata can also include extra codec fields.
                for key, value in self.codec_meta["codec_config"].items():
                    if int(streaming["codec_config"][key]) != value:
                        raise ValueError("Streaming codec metadata belongs to a different codec")
            self.codec_meta.update(streaming)
            step = self.streaming_path.parent / streaming["files"]["decode_step"]
            if not step.is_file():
                raise FileNotFoundError(step)
            self.streaming_source_step_path = step.resolve()
            chosen_step = self.graph_paths.get("codec_decode_step")
            if chosen_step is None:
                self.graph_paths["codec_decode_step"] = self.streaming_source_step_path
            elif chosen_step != self.streaming_source_step_path:
                # A TensorRT-compatible graph may have a static branch rewrite.
                # Require the exact source/candidate hashes and CPU equivalence
                # report; never silently replace the analysis-selected graph.
                proof_path = chosen_step.parent / "streaming_codec_compatibility_report.json"
                if not proof_path.is_file():
                    raise FileNotFoundError(f"Streaming codec compatibility proof is missing: {proof_path}")
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                if not proof.get("all_outputs_exact") or proof.get("source_sha256") != hashlib.sha256(self.streaming_source_step_path.read_bytes()).hexdigest() or proof.get("compatible_sha256") != hashlib.sha256(chosen_step.read_bytes()).hexdigest():
                    raise ValueError("Selected codec_decode_step graph is not covered by the original-model compatibility proof")
            report_path = self.streaming_path.parent / "streaming_validation.json"
            if report_path.is_file():
                import numpy as np
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for row in report.get("reference_audio", []):
                    voice = row["voice"]
                    if voice not in self.voice_paths or not row.get("padding_original_codes_exact"):
                        continue
                    if hashlib.sha256(self.voice_paths[voice].read_bytes()).hexdigest() != row["source_sha256"]:
                        raise ValueError(f"Precomputed reference codes do not match original voice WAV: {voice}")
                    code_path = self.streaming_path.parent / row["codes_file"]
                    with np.load(code_path, allow_pickle=False) as data:
                        codes = data["audio_codes"]
                        length = int(data["audio_codes_lengths"][0])
                        if codes.ndim != 3 or codes.shape[:2] != (16, 1) or not 0 < length <= codes.shape[-1] or length != int(row["valid_code_frames"]):
                            raise ValueError(f"Invalid native reference code shape/valid length: {voice}")
                        valid = codes[:, 0, :length]
                        if np.any(valid < 0) or np.any(valid >= 1024):
                            raise ValueError(f"Reference code exceeds this original codec's codebook: {voice}")
                        self.precomputed_reference_codes[voice] = tuple(tuple(int(value) for value in frame) for frame in valid.T)

    def summary(self) -> dict:
        return {"model": "original MOSS-TTS-Nano-100M with original MOSS-Audio-Tokenizer-Nano", "graphs": sorted(self.graph_paths), "sample_rate": 48000, "channels": 2, "languages": list(self.languages), "default_voice": self.default_voice, "voices": list(self.voice_files), "tokenizer_sha256": hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest(), "streaming_codec_available": self.streaming_path is not None, "precomputed_original_reference_voices": len(self.precomputed_reference_codes), "automatic_download": False}
