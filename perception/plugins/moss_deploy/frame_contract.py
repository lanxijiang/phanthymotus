"""Require provenance and real GPU validation before enabling a fused frame."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .selected_logits import _sha256


FRAME_GENERATION_DEFAULTS = {
    "text_temperature": 1.0, "text_top_p": 1.0, "text_top_k": 50,
    "audio_temperature": 0.8, "audio_top_p": 0.95, "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
}
REPETITION_HISTORY_INPUT = {
    "name": "repetition_seen_mask", "dtype": "float32", "shape": [1, 16, 1024],
    "semantics": "positive means previously generated token in this audio channel; excludes current frame",
}
HISTORY_PATTERNS = {
    "empty", "full", "even_tokens", "odd_tokens", "checkerboard", "inverse_checkerboard",
    "channel_stride", "alternating_channels",
}


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _valid_logit_check(row, step):
    if not isinstance(row, dict) or row.get("step") != step:
        return False
    if not all(row.get(key) is True for key in ("numerical_pass", "shape_match", "finite", "reference_finite")):
        return False
    maximum, relative = row.get("max_abs_error"), row.get("relative_rms_error")
    if (not isinstance(maximum, (float, int)) or not isinstance(relative, (float, int))
            or not math.isfinite(maximum) or not math.isfinite(relative) or min(maximum, relative) < 0):
        return False
    return maximum <= 1e-4 or relative <= 0.05


def _valid_sample(row):
    required = ("model_logits_pass", "sampling_exact", "finite", "outputs_valid")
    if not isinstance(row, dict) or not all(row.get(key) is True for key in required):
        return False
    sampling, logits = row.get("sampling_checks"), row.get("logit_checks")
    if not isinstance(sampling, list) or not isinstance(logits, list) or len(sampling) != 17 or len(logits) != 17:
        return False
    if type(row.get("should_continue")) is not int or row["should_continue"] not in (0, 1):
        return False
    for step, check in enumerate(sampling):
        if (not isinstance(check, dict) or check.get("step") != step or check.get("exact") is not True
                or type(check.get("actual")) is not int or type(check.get("expected")) is not int
                or check["actual"] != check["expected"]
                or not 0 <= check["actual"] < (2 if step == 0 else 1024)):
            return False
        counts = [check.get(key) for key in ("history_seen_tokens", "history_penalized_positive_logits",
                                             "history_penalized_negative_logits")]
        if (any(type(value) is not int or value < 0 for value in counts)
                or counts[0] > (0 if step == 0 else 1024)
                or sum(counts[1:]) > counts[0]):
            return False
    if row["should_continue"] != int(sampling[0]["actual"] == 0):
        return False
    return all(_valid_logit_check(check, step) for step, check in enumerate(logits))


def load_frame_contract(plan_path, contract_path, assets):
    plan_path, contract_path = Path(plan_path).resolve(), Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    model = assets.tts_meta["model_config"]
    if (contract.get("version") != 2
            or contract.get("model_precision") != "fp32"
            or contract.get("graph_name") != "local_fixed_sampled_frame"
            or contract.get("n_vq") != 16 or int(model["n_vq"]) != 16
            or contract.get("audio_channels") != 16 or contract.get("audio_vocab_size") != 1024
            or contract.get("batch_size") != 1 or contract.get("steps_per_frame") != 17
            or contract.get("text_token_ids") != [int(model["audio_assistant_slot_token_id"]), int(model["audio_end_token_id"])]
            or model.get("audio_codebook_sizes") != [1024] * 16
            or contract.get("generation_defaults") != FRAME_GENERATION_DEFAULTS
            or contract.get("sampling") != {"do_sample": True, **FRAME_GENERATION_DEFAULTS}
            or contract.get("repetition_history_input") != REPETITION_HISTORY_INPUT
            or contract.get("sampling_semantics", {}).get("cdf_normalization") != "divide_by_last_cumulative_probability"
            or contract.get("random_input", {}).get("name") != "random_u"
            or contract.get("random_input", {}).get("dtype") != "float32"
            or contract.get("random_input", {}).get("shape") != [1, 17]):
        raise ValueError("Fused frame contract does not match the current MOSS model and sampling parameters")
    if not _is_sha256(contract.get("onnx_sha256")):
        raise ValueError("Fused frame contract has no valid candidate ONNX hash")
    if contract.get("source_onnx_sha256") != _sha256(assets.graph_paths["local_cached_step"]):
        raise ValueError("Fused frame was exported from another local MOSS graph")
    if contract.get("source_metadata_sha256") != _sha256(assets.tts_meta_path):
        raise ValueError("Fused frame was exported with another MOSS model metadata file")
    report = json.loads((plan_path.parent / "build_report.json").read_text())
    entries = [row for row in report.get("engines", []) if row.get("name") == "local_fixed_sampled_frame"]
    if report.get("status") != "built" or report.get("validation_pass") is not True or len(entries) != 1:
        raise ValueError("Fused frame requires a successful GPU numerical and sampling validation report")
    entry = entries[0]
    if (report.get("model_precision") != "fp32" or entry.get("model_precision") != "fp32"
            or entry.get("requested_precision") != "fp32" or entry.get("tf32_enabled") is not False
            or entry.get("strongly_typed") is not True
            or entry.get("precision") != "onnx_explicit_types"):
        raise ValueError("Fused frame requires a full FP32 engine with TF32 disabled")
    if (entry.get("sha256") != _sha256(plan_path)
            or entry.get("bytes") != plan_path.stat().st_size
            or entry.get("onnx_sha256") != contract.get("onnx_sha256")
            or entry.get("contract_sha256") != _sha256(contract_path)
            or report.get("contract_sha256") != _sha256(contract_path)
            or report.get("source_onnx_sha256") != contract["source_onnx_sha256"]
            or report.get("source_metadata_sha256") != contract["source_metadata_sha256"]
            or not _is_sha256(report.get("reference_plan_sha256"))
            or report.get("sampling") != contract["sampling"]
            or report.get("cdf_normalization") != contract["sampling_semantics"]["cdf_normalization"]
            or report.get("repetition_history_input") != REPETITION_HISTORY_INPUT):
        raise ValueError("Fused frame engine, ONNX and contract hashes do not match")
    validation = entry.get("validation", [])
    if (not isinstance(validation, list) or len(validation) != 32
            or report.get("sample_count") != len(validation)
            or not all(_valid_sample(row) for row in validation)):
        raise ValueError("Fused frame numerical or exact sampling validation is incomplete")
    labels = [row.get("sample") for row in validation]
    if (any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
            or sorted({row["should_continue"] for row in validation}) != [0, 1]
            or report.get("continuation_values_covered") != [0, 1]
            or report.get("audio_channels_covered") != list(range(16))):
        raise ValueError("Fused frame validation lacks distinct samples or full continuation/channel coverage")
    if any(not isinstance(row.get("history_pattern"), str) for row in validation):
        raise ValueError("Fused frame validation has no repetition-history pattern")
    history_patterns = {row["history_pattern"] for row in validation}
    channels = sorted({check["step"] - 1 for row in validation for check in row["sampling_checks"]
                       if check["step"] > 0 and check["history_seen_tokens"] > 0})
    signs = sorted({sign for sign in ("negative", "positive")
                    if any(check["history_penalized_" + sign + "_logits"] > 0
                           for row in validation for check in row["sampling_checks"])})
    if (history_patterns != HISTORY_PATTERNS
            or report.get("history_patterns_covered") != sorted(HISTORY_PATTERNS)
            or channels != list(range(16)) or report.get("history_nonempty_channels_covered") != channels
            or signs != ["negative", "positive"] or report.get("history_penalty_signs_covered") != signs
            or any(not _is_sha256(row.get("repetition_history_sha256")) for row in validation)):
        raise ValueError("Fused frame repetition-history validation is incomplete")
    for row in validation:
        pattern = row["history_pattern"]
        for check in row["sampling_checks"][1:]:
            channel = check["step"] - 1
            expected = (0 if pattern == "empty" else 1024 if pattern == "full" else
                        1024 * int(channel % 2 == 0) if pattern == "alternating_channels" else
                        64 if pattern == "channel_stride" else 512)
            if check["history_seen_tokens"] != expected:
                raise ValueError("Fused frame repetition-history fixture coverage is inconsistent")
    return contract
