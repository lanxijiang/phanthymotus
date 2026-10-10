"""Require provenance and real GPU validation before enabling a fused frame."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .selected_logits import _sha256


FRAME_GENERATION_DEFAULTS = {
    "text_temperature": 1.5, "text_top_p": 1.0, "text_top_k": 50,
    "audio_temperature": 1.7, "audio_top_p": 0.8, "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
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
    if row["should_continue"] != int(sampling[0]["actual"] == 0):
        return False
    return all(_valid_logit_check(check, step) for step, check in enumerate(logits))


def load_frame_contract(plan_path, contract_path, assets):
    plan_path, contract_path = Path(plan_path).resolve(), Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    model = assets.tts_meta["model_config"]
    if (contract.get("version") != 1
            or contract.get("graph_name") != "local_fixed_sampled_frame"
            or contract.get("n_vq") != 16 or int(model["n_vq"]) != 16
            or contract.get("audio_channels") != 16 or contract.get("audio_vocab_size") != 1024
            or contract.get("batch_size") != 1 or contract.get("steps_per_frame") != 17
            or contract.get("text_token_ids") != [int(model["audio_assistant_slot_token_id"]), int(model["audio_end_token_id"])]
            or model.get("audio_codebook_sizes") != [1024] * 16
            or contract.get("generation_defaults") != FRAME_GENERATION_DEFAULTS
            or contract.get("sampling") != {"do_sample": True, **FRAME_GENERATION_DEFAULTS}
            or contract.get("sampling_semantics", {}).get("cdf_normalization") != "divide_by_last_cumulative_probability"
            or contract.get("random_input", {}).get("name") != "random_u"
            or contract.get("random_input", {}).get("dtype") != "float32"
            or contract.get("random_input", {}).get("shape") != [1, 17]):
        raise ValueError("Fused frame contract does not match the current MOSS model and sampling parameters")
    if not _is_sha256(contract.get("onnx_sha256")):
        raise ValueError("Fused frame contract has no valid candidate ONNX hash")
    if contract.get("source_onnx_sha256") != _sha256(assets.graph_paths["local_cached_step"]):
        raise ValueError("Fused frame was exported from another local MOSS graph")
    report = json.loads((plan_path.parent / "build_report.json").read_text())
    entries = [row for row in report.get("engines", []) if row.get("name") == "local_fixed_sampled_frame"]
    if report.get("status") != "built" or report.get("validation_pass") is not True or len(entries) != 1:
        raise ValueError("Fused frame requires a successful GPU numerical and sampling validation report")
    entry = entries[0]
    if (entry.get("sha256") != _sha256(plan_path)
            or entry.get("onnx_sha256") != contract.get("onnx_sha256")
            or entry.get("contract_sha256") != _sha256(contract_path)
            or report.get("contract_sha256") != _sha256(contract_path)
            or report.get("source_onnx_sha256") != contract["source_onnx_sha256"]
            or not _is_sha256(report.get("reference_plan_sha256"))
            or report.get("sampling") != contract["sampling"]
            or report.get("cdf_normalization") != contract["sampling_semantics"]["cdf_normalization"]):
        raise ValueError("Fused frame engine, ONNX and contract hashes do not match")
    validation = entry.get("validation", [])
    if (not isinstance(validation, list) or len(validation) < 8
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
    return contract
