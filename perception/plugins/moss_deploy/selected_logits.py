"""Explicit adapter for a validated local graph computing only consumed logits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected_contract(plan_path, contract_path, assets):
    plan_path, contract_path = Path(plan_path).resolve(), Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    model = assets.tts_meta["model_config"]
    expected_ids = [int(model["audio_assistant_slot_token_id"]), int(model["audio_end_token_id"])]
    if (contract.get("version") != 1 or contract.get("text_token_ids") != expected_ids
            or contract.get("audio_channels") != int(model["n_vq"])
            or contract.get("audio_vocab_size") != 1024
            or ("audio_codebook_sizes" in model and
                model["audio_codebook_sizes"] != [1024] * int(model["n_vq"]))):
        raise ValueError("Selected local projection contract does not match original MOSS")
    if contract.get("source_onnx_sha256") != _sha256(assets.graph_paths["local_cached_step"]):
        raise ValueError("Selected local projection comes from another original ONNX graph")
    report = json.loads((plan_path.parent / "build_report.json").read_text())
    engines = [entry for entry in report.get("engines", []) if entry.get("name") == "local_cached_step"]
    if report.get("status") != "built" or len(engines) != 1:
        raise ValueError("Selected local plan has no successful numerical build report")
    entry = engines[0]
    if (entry.get("onnx_sha256") != contract.get("selected_onnx_sha256")
            or entry.get("sha256") != _sha256(plan_path)):
        raise ValueError("Selected local plan/contract hashes do not match")
    if not entry.get("validation") or not all(
        sample.get("outputs")
        and {"text_logits", "audio_logits"} <= sample["outputs"].keys()
        and all(output.get("numerical_pass") for output in sample["outputs"].values())
        for sample in entry["validation"]
    ):
        raise ValueError("Selected local plan has not passed output validation")
    return contract


class SelectedTextScores:
    """Expose only the two scores the original text sampler actually reads."""
    def __init__(self, values, token_ids):
        self.values = np.asarray(values, dtype=np.float32).reshape(-1)
        self.token_ids = np.asarray(token_ids, dtype=np.int32)
        if self.values.shape != (2,):
            raise ValueError("Selected text logits must contain two candidates")

    def reshape(self, *shape):
        if shape != (-1,):
            raise ValueError("Selected text logits only support the original flatten call")
        return self

    def __getitem__(self, indices):
        if not np.array_equal(np.asarray(indices), self.token_ids):
            raise ValueError("An uncomputed text logit was requested")
        return self.values


class SelectedAudioScores:
    def __init__(self, values, channel):
        self.values = np.asarray(values, dtype=np.float32).reshape(-1)
        self.channel = int(channel)
        if self.values.shape != (1024,):
            raise ValueError("Selected audio logits must contain 1024 candidates")

    def for_channel(self, channel):
        if int(channel) != self.channel:
            raise ValueError("An uncomputed audio channel was requested")
        return self.values


def wrap_selected_outputs(outputs, feeds, contract):
    step = int(np.asarray(feeds["step_type"]).item())
    previous_channel = int(np.asarray(feeds["channel_index"]).item())
    if step not in (0, 1, 2):
        raise ValueError("Unknown selected local step type")
    if (step in (0, 1) and previous_channel != 0) or (step == 2 and previous_channel < 0):
        raise ValueError("Invalid selected local input channel for this step")
    channel = previous_channel + 1 if step == 2 else 0
    if not 0 <= channel < contract["audio_channels"]:
        raise ValueError("Selected local audio channel is outside the original codebooks")
    return {**outputs,
            "text_logits": SelectedTextScores(outputs["text_logits"], contract["text_token_ids"]),
            "audio_logits": SelectedAudioScores(outputs["audio_logits"], channel)}
