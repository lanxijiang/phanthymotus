"""The fused runtime rejects stale, incomplete or failed GPU validation evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from plugins.moss_deploy.frame_contract import FRAME_GENERATION_DEFAULTS, load_frame_contract


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    source, plan = tmp_path / "original.onnx", tmp_path / "local_fixed_sampled_frame.plan"
    source.write_bytes(b"original step graph")
    plan.write_bytes(b"validated fused engine")
    contract_path, report_path = tmp_path / "frame-contract.json", tmp_path / "build_report.json"
    contract = {
        "version": 1, "graph_name": "local_fixed_sampled_frame", "n_vq": 16,
        "audio_channels": 16, "audio_vocab_size": 1024, "batch_size": 1, "steps_per_frame": 17,
        "text_token_ids": [9, 7], "generation_defaults": dict(FRAME_GENERATION_DEFAULTS),
        "sampling": {"do_sample": True, **FRAME_GENERATION_DEFAULTS},
        "sampling_semantics": {"cdf_normalization": "divide_by_last_cumulative_probability"},
        "source_onnx_sha256": digest(source.read_bytes()), "onnx_sha256": digest(b"candidate ONNX"),
        "random_input": {"name": "random_u", "dtype": "float32", "shape": [1, 17]},
    }
    rows = []
    for index in range(32):
        decision = index % 2
        rows.append({"sample": f"hidden-{index // 8}-uniform-{index % 8}", "model_logits_pass": True,
                     "sampling_exact": True, "finite": True, "outputs_valid": True,
                     "should_continue": decision,
                     "sampling_checks": [{"step": step, "exact": True,
                         "actual": (1 - decision) if step == 0 else 20,
                         "expected": (1 - decision) if step == 0 else 20} for step in range(17)],
                     "logit_checks": [{"step": step, "numerical_pass": True, "shape_match": True,
                         "finite": True, "reference_finite": True,
                         "max_abs_error": 0.003, "relative_rms_error": 0.001} for step in range(17)]})
    entry = {"name": "local_fixed_sampled_frame", "sha256": digest(plan.read_bytes()),
             "onnx_sha256": contract["onnx_sha256"], "validation": rows}
    report = {"status": "built", "validation_pass": True, "engines": [entry],
              "source_onnx_sha256": contract["source_onnx_sha256"],
              "reference_plan_sha256": digest(b"verified reference step engine"),
              "sampling": copy.deepcopy(contract["sampling"]), "sample_count": len(rows),
              "cdf_normalization": contract["sampling_semantics"]["cdf_normalization"],
              "continuation_values_covered": [0, 1], "audio_channels_covered": list(range(16))}
    model = {"n_vq": 16, "audio_codebook_sizes": [1024] * 16,
             "audio_assistant_slot_token_id": 9, "audio_end_token_id": 7}
    assets = SimpleNamespace(tts_meta={"model_config": model}, graph_paths={"local_cached_step": source})

    def save(*, rebind_contract=True):
        contract_path.write_text(json.dumps(contract))
        if rebind_contract:
            entry["contract_sha256"] = digest(contract_path.read_bytes())
            report["contract_sha256"] = entry["contract_sha256"]
        report_path.write_text(json.dumps(report))

    save()
    return SimpleNamespace(source=source, plan=plan, contract_path=contract_path, report_path=report_path,
                           contract=contract, report=report, entry=entry, rows=rows,
                           model=model, assets=assets, save=save)


def test_complete_builder_report_loads(bundle):
    assert load_frame_contract(bundle.plan, bundle.contract_path, bundle.assets) == bundle.contract


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("graph_name", "another"), ("n_vq", 8), ("audio_channels", 8),
    ("audio_vocab_size", 512), ("batch_size", 2), ("steps_per_frame", 16), ("text_token_ids", [7, 9]),
    ("onnx_sha256", None), ("onnx_sha256", "not-a-sha"),
])
def test_wrong_contract_interface_or_hash_rejected(bundle, field, value):
    bundle.contract[field] = value
    if field == "onnx_sha256":
        bundle.entry[field] = value  # Matching missing hashes must also fail.
    bundle.save()
    with pytest.raises(ValueError):
        load_frame_contract(bundle.plan, bundle.contract_path, bundle.assets)


@pytest.mark.parametrize("field", list(FRAME_GENERATION_DEFAULTS))
def test_modified_sampling_controls_rejected_even_with_rebound_hash(bundle, field):
    bundle.contract["generation_defaults"][field] += .1
    bundle.contract["sampling"][field] += .1
    bundle.save()
    with pytest.raises(ValueError, match="sampling parameters"):
        load_frame_contract(bundle.plan, bundle.contract_path, bundle.assets)


@pytest.mark.parametrize("fault", [
    "plan_bytes", "source_bytes", "source_report", "candidate_hash", "contract_bytes", "entry_contract_hash",
    "top_contract_hash", "missing_reference_hash", "failed_build", "failed_validation", "missing_engine", "duplicate_engine",
    "model_codebook", "sampling_conflict", "sampling_off", "rng_dtype", "rng_shape", "rng_name",
    "old_cdf_contract", "old_cdf_report",
])
def test_hash_provenance_and_configuration_faults_rejected(bundle, fault):
    if fault == "plan_bytes":
        bundle.plan.write_bytes(b"different engine")
    elif fault == "source_bytes":
        bundle.source.write_bytes(b"different original model")
    elif fault == "source_report":
        bundle.report["source_onnx_sha256"] = digest(b"different source")
    elif fault == "candidate_hash":
        bundle.entry["onnx_sha256"] = digest(b"another candidate")
    elif fault == "contract_bytes":
        bundle.contract["new_unsupported_metadata"] = True
    elif fault == "entry_contract_hash":
        bundle.entry["contract_sha256"] = digest(b"stale contract")
    elif fault == "top_contract_hash":
        bundle.report["contract_sha256"] = digest(b"stale contract")
    elif fault == "missing_reference_hash":
        bundle.report.pop("reference_plan_sha256")
    elif fault == "failed_build":
        bundle.report["status"] = "failed"
    elif fault == "failed_validation":
        bundle.report["validation_pass"] = False
    elif fault == "missing_engine":
        bundle.report["engines"] = []
    elif fault == "duplicate_engine":
        bundle.report["engines"].append(dict(bundle.entry))
    elif fault == "model_codebook":
        bundle.model["audio_codebook_sizes"][0] = 512
    elif fault == "sampling_conflict":
        bundle.report["sampling"]["audio_top_p"] = .95
    elif fault == "sampling_off":
        bundle.contract["sampling"]["do_sample"] = False
    elif fault == "rng_dtype":
        bundle.contract["random_input"]["dtype"] = "float64"
    elif fault == "rng_shape":
        bundle.contract["random_input"]["shape"] = [1, 16]
    elif fault == "rng_name":
        bundle.contract["random_input"]["name"] = "other"
    elif fault == "old_cdf_contract":
        bundle.contract["sampling_semantics"].pop("cdf_normalization")
    elif fault == "old_cdf_report":
        bundle.report.pop("cdf_normalization")
    bundle.save(rebind_contract=fault not in {"contract_bytes", "entry_contract_hash", "top_contract_hash"})
    with pytest.raises(ValueError):
        load_frame_contract(bundle.plan, bundle.contract_path, bundle.assets)


@pytest.mark.parametrize("fault", [
    "no_samples", "too_few_samples", "wrong_count", "summary_false", "missing_sample_field", "missing_logit_checks",
    "short_logit_checks", "out_of_order_logits", "failed_logit", "nonfinite_logit", "numeric_tolerance_failure",
    "missing_sampling_checks", "wrong_step", "mismatched_token", "false_exact", "out_of_vocab_token",
    "conflicting_continuation", "duplicate_sample", "missing_sample_label", "missing_channels", "missing_stop_coverage",
    "missing_stop_actual", "nonboolean_pass", "missing_reference_finite", "nan_metric",
])
def test_incomplete_or_failed_validation_evidence_rejected(bundle, fault):
    first = bundle.rows[0]
    if fault == "no_samples":
        bundle.entry["validation"] = []
    elif fault == "too_few_samples":
        bundle.entry["validation"] = bundle.rows[:7]
        bundle.report["sample_count"] = 7
    elif fault == "wrong_count":
        bundle.report["sample_count"] = 100
    elif fault == "summary_false":
        first["sampling_exact"] = False
    elif fault == "missing_sample_field":
        first.pop("model_logits_pass")
    elif fault == "missing_logit_checks":
        first.pop("logit_checks")
    elif fault == "short_logit_checks":
        first["logit_checks"].pop()
    elif fault == "out_of_order_logits":
        first["logit_checks"].reverse()
    elif fault == "failed_logit":
        first["logit_checks"][9]["numerical_pass"] = False
    elif fault == "nonfinite_logit":
        first["logit_checks"][9]["finite"] = False
    elif fault == "numeric_tolerance_failure":
        first["logit_checks"][9].update(max_abs_error=1, relative_rms_error=.9)
    elif fault == "missing_sampling_checks":
        first.pop("sampling_checks")
    elif fault == "wrong_step":
        first["sampling_checks"][16]["step"] = 15
    elif fault == "mismatched_token":
        first["sampling_checks"][16]["actual"] += 1
    elif fault == "false_exact":
        first["sampling_checks"][16]["exact"] = False
    elif fault == "out_of_vocab_token":
        first["sampling_checks"][16].update(actual=1024, expected=1024)
    elif fault == "conflicting_continuation":
        first["should_continue"] = 1 - first["should_continue"]
    elif fault == "duplicate_sample":
        bundle.rows[1]["sample"] = first["sample"]
    elif fault == "missing_sample_label":
        first.pop("sample")
    elif fault == "missing_channels":
        bundle.report["audio_channels_covered"].pop()
    elif fault == "missing_stop_coverage":
        bundle.report["continuation_values_covered"] = [1]
    elif fault == "missing_stop_actual":
        for row in bundle.rows:
            row["should_continue"] = 1
            row["sampling_checks"][0].update(actual=0, expected=0)
    elif fault == "nonboolean_pass":
        first["sampling_exact"] = 1
    elif fault == "missing_reference_finite":
        first["logit_checks"][0].pop("reference_finite")
    elif fault == "nan_metric":
        first["logit_checks"][0]["max_abs_error"] = float("nan")
    bundle.save()
    with pytest.raises(ValueError):
        load_frame_contract(bundle.plan, bundle.contract_path, bundle.assets)
