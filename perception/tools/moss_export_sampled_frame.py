#!/usr/bin/env python3
"""Unroll one MOSS frame and sample all 16 channels inside a candidate ONNX graph.

This is graph transformation only: no CPU/GPU model inference, no weight training,
and no implicit change to the deployed sampling configuration. The candidate is
NOT deployment ready until its logits and sampled tokens are validated on Orin.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PREFIX = "moss_frame/"
SAMPLING = {
    "do_sample": True,
    "text_temperature": 1.5,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prune_to_outputs(model):
    producers = {value: i for i, node in enumerate(model.graph.node) for value in node.output}
    wanted, used = set(), set()
    pending = [value.name for value in model.graph.output]
    while pending:
        value = pending.pop()
        if value in used:
            continue
        used.add(value)
        if value in producers:
            index = producers[value]
            wanted.add(index)
            pending.extend(model.graph.node[index].input)
    nodes = [node for i, node in enumerate(model.graph.node) if i in wanted]
    weights = [value for value in model.graph.initializer if value.name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    del model.graph.value_info[:]


class GraphBuilder:
    def __init__(self, model):
        self.model = model
        self.constants = {}

    def constant(self, name, value, dtype):
        array = np.asarray(value, dtype=dtype)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.constants:
            full_name = PREFIX + "const/" + name
            self.model.graph.initializer.append(numpy_helper.from_array(array, full_name))
            self.constants[key] = full_name
        return self.constants[key]

    def node(self, op, inputs, name, *, outputs=None, **attrs):
        outputs = outputs or [name]
        self.model.graph.node.append(helper.make_node(op, inputs, outputs, name=name, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    def prefix_scan(self, values, *, vocabulary, name, exclusive=False):
        """Static Hillis-Steele FP32 prefix sum, without ONNX/TensorRT loops.

        Every stage reads the entire preceding stage, so additions cannot depend
        on host scheduling. Tree order differs from sequential FP32 CumSum by
        rounding only; exact sampled-token validation remains mandatory.
        """
        start = self.constant("scan_start", [0], np.int64)
        axis = self.constant("scan_axis", [1], np.int64)
        stride = self.constant("scan_stride", [1], np.int64)
        current = values
        offset = 1
        while offset < vocabulary:
            end = self.constant("scan_end_" + str(vocabulary - offset), [vocabulary - offset], np.int64)
            zero = self.constant("scan_zero_" + str(offset), np.zeros((1, offset)), np.float32)
            prefix = name + "/offset_" + str(offset) + "/"
            clipped = self.node("Slice", [current, start, end, axis, stride], prefix + "previous")
            shifted = self.node("Concat", [zero, clipped], prefix + "shifted", axis=1)
            current = self.node("Add", [current, shifted], prefix + "sum")
            offset *= 2
        if exclusive:
            end = self.constant("scan_end_" + str(vocabulary - 1), [vocabulary - 1], np.int64)
            zero = self.constant("scan_zero_1", np.zeros((1, 1)), np.float32)
            clipped = self.node("Slice", [current, start, end, axis, stride], name + "/exclusive_previous")
            return self.node("Concat", [zero, clipped], name, axis=1)
        return self.node("Identity", [current], name)

    def sample(self, logits, random_u, *, vocabulary, temperature, top_k, top_p, prefix):
        """Return int32 [1] sampled ids, with host filtering and token-ID CDF order.

        FP32 math is deliberate; this does not claim bitwise equivalence to the
        original host's FP64 softmax/subtraction near sampling boundaries.
        """
        def add(op, inputs, suffix, **attrs):
            return self.node(op, inputs, prefix + suffix, **attrs)

        temperature_name = self.constant("temperature_" + str(temperature), temperature, np.float32)
        scores = add("Div", [logits, temperature_name], "temperature")
        axes = self.constant("axis_one", [1], np.int64)
        negative_inf = self.constant("negative_inf", -np.inf, np.float32)
        if (0 < top_k < vocabulary) or (0 < top_p < 1):
            # ONNX TopK breaks equal scores by lower original index. Sorting the
            # full vocabulary preserves all ties at the kth threshold, unlike
            # taking exactly k entries. TRT supports k=1024 on our target.
            count = self.constant("vocabulary_" + str(vocabulary), [vocabulary], np.int64)
            sorted_scores, sorted_ids = self.node(
                "TopK", [scores, count], prefix + "sort",
                outputs=[prefix + "sorted_scores", prefix + "sorted_ids"],
                axis=1, largest=1, sorted=1,
            )
            filtered = sorted_scores
            if 0 < top_k < vocabulary:
                kth = self.constant("top_k_index_" + str(top_k), [top_k - 1], np.int64)
                threshold = add("Gather", [sorted_scores, kth], "kth_threshold", axis=1)
                too_small = add("Less", [sorted_scores, threshold], "below_kth")
                filtered = add("Where", [too_small, negative_inf, sorted_scores], "top_k")
            if 0 < top_p < 1:
                probabilities = add("Softmax", [filtered], "sorted_probabilities", axis=1)
                previous = self.prefix_scan(probabilities, vocabulary=vocabulary,
                                            name=prefix + "previous_probability", exclusive=True)
                cutoff = self.constant("top_p_" + str(top_p), top_p, np.float32)
                remove = add("Greater", [previous, cutoff], "past_top_p")
                filtered = add("Where", [remove, negative_inf, filtered], "top_p")
            # Every output index is written exactly once by this full sort.
            # Restore original token order before CDF, matching the host loop.
            scores = add("ScatterElements", [scores, sorted_ids, filtered], "token_order_scores", axis=1)
        probabilities = add("Softmax", [scores], "probabilities", axis=1)
        cdf_raw = self.prefix_scan(probabilities, vocabulary=vocabulary, name=prefix + "cdf_raw")
        last_index = self.constant("last_index_" + str(vocabulary), [vocabulary - 1], np.int64)
        cdf_total = add("Gather", [cdf_raw, last_index], "cdf_total", axis=1)
        # Normalize the entire cumulative distribution. The final support and
        # every trailing masked position become exactly one, so rounding cannot
        # send a near-one uniform to an unrelated argmax fallback.
        normalized = add("Div", [cdf_raw, cdf_total], "cdf_normalized")
        zero_f32 = self.constant("zero_f32", 0, np.float32)
        one_f32 = self.constant("one_f32", 1, np.float32)
        zero_i32 = self.constant("zero_i32", [0], np.int32)
        minus_one = self.constant("minus_one_i32", -1, np.int32)
        positions = self.constant("token_positions_" + str(vocabulary),
                                  np.arange(vocabulary, dtype=np.int32).reshape(1, vocabulary), np.int32)
        positive = add("Greater", [probabilities, zero_f32], "positive_probability")
        supported_positions = add("Where", [positive, positions, minus_one], "supported_positions")
        last_support = add("ReduceMax", [supported_positions], "last_supported_token", axes=[1], keepdims=1)
        tail = add("GreaterOrEqual", [positions, last_support], "cdf_tail")
        # Parallel FP32 sums can differ by an ulp even across a plateau of zero
        # probabilities. Set the complete terminal plateau to its exact value,
        # then search only positive-probability tokens. This changes no ideal
        # categorical distribution and cannot assign mass to a masked token.
        cdf = add("Where", [tail, one_f32, normalized], "cdf")
        reached = add("GreaterOrEqual", [cdf, random_u], "cdf_reached_u")
        eligible = add("And", [positive, reached], "eligible_token")
        eligible_f32 = add("Cast", [eligible], "eligible_f32", to=TensorProto.FLOAT)
        selected_i64 = add("ArgMax", [eligible_f32], "first_eligible", axis=1, keepdims=0, select_last_index=0)
        selected = add("Cast", [selected_i64], "sampled_position", to=TensorProto.INT32)
        zero_uniform_2d = add("Equal", [random_u, zero_f32], "zero_uniform_2d")
        zero_uniform = add("Squeeze", [zero_uniform_2d, axes], "zero_uniform")
        # The original host subtracts probability[0] then checks <= 0, so u=0
        # returns token zero even when it was masked by top-k/top-p.
        return add("Where", [zero_uniform, zero_i32, selected], "token_id")


def _external_locations(model):
    locations = set()
    for value in model.graph.initializer:
        if value.data_location != TensorProto.EXTERNAL:
            continue
        info = {entry.key: entry.value for entry in value.external_data}
        name = info.get("location", "")
        location = Path(name)
        if not name or location.is_absolute() or ".." in location.parts:
            raise ValueError("Only safe relative external weights are supported")
        locations.add(name)
    return sorted(locations)


def rewrite(source, config):
    """Return a fixed-batch frame graph with traceable, shared original weights."""
    channels = int(config["n_vq"])
    hidden = int(config["hidden_size"])
    sizes = list(map(int, config["audio_codebook_sizes"]))
    text_ids = [int(config["audio_assistant_slot_token_id"]), int(config["audio_end_token_id"])]
    if channels < 1 or len(sizes) != channels or len(set(sizes)) != 1:
        raise ValueError("Equal codebook sizes are required")
    if len(set(text_ids)) != 2 or min(text_ids) < 0 or max(text_ids) >= int(config["vocab_size"]):
        raise ValueError("Invalid text candidate ids")
    if any(value.name.startswith(PREFIX) for value in source.graph.initializer):
        raise ValueError("Already transformed frame graph")
    nodes = {node.name: node for node in source.graph.node}
    heads = [nodes["/text_lm_head/MatMul"]] + [nodes[f"/audio_lm_heads.{i}/MatMul"] for i in range(channels)]
    if any(node.op_type != "MatMul" or len(node.input) != 2 or node.input[0] != heads[0].input[0] for node in heads):
        raise ValueError("Expected shared last hidden vector for all heads")
    weights = {value.name: value for value in source.graph.initializer}
    for index, head in enumerate(heads):
        expected = [hidden, int(config["vocab_size"]) if index == 0 else sizes[index - 1]]
        if list(weights[head.input[1]].dims) != expected:
            raise ValueError("Unexpected projection weight dimensions")
    inputs = {value.name: value for value in source.graph.input}
    past_names = [value.name for value in source.graph.input if value.name.startswith("local_past_")]
    if not past_names or len(past_names) % 2:
        raise ValueError("Expected local key/value input pairs")
    expected_inputs = {"global_hidden", "text_token_id", "audio_token_id", "channel_index", "step_type", "past_valid_lengths", *past_names}
    if set(inputs) != expected_inputs:
        raise ValueError("Unexpected local decoder interface")
    for name in past_names:
        shape = inputs[name].type.tensor_type.shape.dim
        if len(shape) != 4 or not shape[2].dim_value or not shape[3].dim_value:
            raise ValueError("Expected local KV tensors [batch,past,heads,head_dim]")
    model = copy.deepcopy(source)
    del model.graph.node[:]
    del model.graph.input[:]
    del model.graph.output[:]
    del model.graph.value_info[:]
    model.graph.name = "moss_local_fixed_sampled_frame"
    model.graph.input.extend([
        helper.make_tensor_value_info("global_hidden", TensorProto.FLOAT, [1, hidden]),
        helper.make_tensor_value_info("random_u", TensorProto.FLOAT, [1, channels + 1]),
    ])
    build = GraphBuilder(model)
    init_names = set(weights)
    zero = build.constant("zero_i32", [0], np.int32)
    text_candidates = build.constant("text_candidate_ids", text_ids, np.int32)
    text_candidates_i64 = build.constant("text_candidate_ids_i64", text_ids, np.int64)
    axis_one = build.constant("axes_one", [1], np.int64)
    text_weight = build.node("Gather", [heads[0].input[1], text_candidates_i64], PREFIX + "text_weight", axis=1)
    previous = {}
    for name in past_names:
        dims = inputs[name].type.tensor_type.shape.dim
        previous[name] = build.constant("empty_" + name, np.zeros((1, 0, dims[2].dim_value, dims[3].dim_value)), np.float32)
    next_text, next_audio = zero, zero
    audio_logits, audio_tokens = [], []
    text_logits = None
    selected_continue = None
    # Specialize the original embedding switch when its known producer exists.
    # The original graph computes all sixteen embeddings and multiplies fifteen
    # of them by zero. Static selection allows those unused branches to vanish.
    embedding_switch = "/Add_1_output_0"
    can_specialize_embedding = (nodes.get("/Add_1") is not None
                                and list(nodes["/Add_1"].output) == [embedding_switch]
                                and "model.transformer.wte.weight" in {v for n in source.graph.node for v in n.output}
                                and all("model.audio_embeddings.%d.weight" % i in {v for n in source.graph.node for v in n.output}
                                        for i in range(channels)))
    for step in range(channels + 1):
        prefix = PREFIX + f"step_{step:02d}/"
        mapping = {name: name for name in init_names}
        mapping.update(global_hidden="global_hidden", text_token_id=next_text, audio_token_id=next_audio,
                       channel_index=build.constant(f"channel_{max(0, step - 2)}", [max(0, step - 2)], np.int32),
                       step_type=build.constant(f"step_type_{min(step, 2)}", [min(step, 2)], np.int32),
                       past_valid_lengths=build.constant(f"past_length_{step}", [step], np.int32), **previous)
        for node in source.graph.node:
            for output in node.output:
                mapping[output] = prefix + output
        if can_specialize_embedding:
            if step == 0:
                embedded = build.node("Cast", ["global_hidden"], prefix + "chosen_embedding", to=TensorProto.FLOAT16)
            else:
                # The canonical transpose producing this value occurs before
                # the original embedding switch; insert Gather at that point.
                embedded = prefix + "chosen_embedding"
            mapping[embedding_switch] = embedded
        for node in source.graph.node:
            if can_specialize_embedding and embedding_switch in node.output:
                if step:
                    embedding = "model.transformer.wte.weight" if step == 1 else f"model.audio_embeddings.{step - 2}.weight"
                    token = next_text if step == 1 else next_audio
                    build.node("Gather", [mapping[embedding], token], embedded, axis=0)
                continue
            cloned = copy.deepcopy(node)
            cloned.name = prefix + node.name
            for i, value in enumerate(cloned.input):
                cloned.input[i] = mapping.get(value, value)
            for i, value in enumerate(cloned.output):
                cloned.output[i] = mapping[value]
            model.graph.node.append(cloned)
        vector = mapping[heads[0].input[0]]
        weight = text_weight if step == 0 else heads[step].input[1]
        projection = build.node("MatMul", [vector, weight], prefix + "selected_projection")
        rounded = build.node("Cast", [projection], prefix + "projection_fp16", to=TensorProto.FLOAT16)
        logits = build.node("Cast", [rounded], prefix + "logits", to=TensorProto.FLOAT)
        random_index = build.constant(f"random_index_{step}", [step], np.int64)
        random = build.node("Gather", ["random_u", random_index], prefix + "random_u", axis=1)
        token = build.sample(logits, random, vocabulary=2 if step == 0 else sizes[step - 1],
                             temperature=SAMPLING["text_temperature"] if step == 0 else SAMPLING["audio_temperature"],
                             top_k=SAMPLING["text_top_k"] if step == 0 else SAMPLING["audio_top_k"],
                             top_p=SAMPLING["text_top_p"] if step == 0 else SAMPLING["audio_top_p"],
                             prefix=f"sample/step_{step:02d}/")
        if step == 0:
            text_logits = logits
            is_continue = build.node("Equal", [token, zero], prefix + "is_continue")
            selected_continue = build.node("Cast", [is_continue], prefix + "should_continue", to=TensorProto.INT32)
            next_text = build.node("Gather", [text_candidates, token], prefix + "text_token", axis=0)
        else:
            audio_logits.append(build.node("Unsqueeze", [logits, axis_one], prefix + "logits_channel"))
            audio_tokens.append(build.node("Unsqueeze", [token, axis_one], prefix + "token_channel"))
            next_audio = token
        previous = {name: mapping[name.replace("local_past_", "local_present_", 1)] for name in past_names}
    build.node("Unsqueeze", [selected_continue, axis_one], "should_continue")
    build.node("Concat", audio_tokens, "frame_token_ids", axis=1)
    build.node("Identity", [text_logits], "text_logits")
    build.node("Concat", audio_logits, "audio_logits", axis=1)
    model.graph.output.extend([
        helper.make_tensor_value_info("should_continue", TensorProto.INT32, [1, 1]),
        helper.make_tensor_value_info("frame_token_ids", TensorProto.INT32, [1, channels]),
        helper.make_tensor_value_info("text_logits", TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info("audio_logits", TensorProto.FLOAT, [1, channels, sizes[0]]),
    ])
    prune_to_outputs(model)
    # Every original weight retained in the new graph must still reference the
    # exact original bytes, and each name is defined once across all 17 steps.
    actual = {value.name: value for value in model.graph.initializer}
    if len(actual) != len(model.graph.initializer):
        raise ValueError("Duplicate initializer names")
    if any(value.SerializeToString() != weights[name].SerializeToString()
           for name, value in actual.items() if name in weights):
        raise ValueError("An original weight was modified")
    contract = {
        "version": 1,
        "graph_name": "local_fixed_sampled_frame",
        "batch_size": 1,
        "steps_per_frame": channels + 1,
        "audio_channels": channels,
        "audio_vocab_size": sizes[0],
        "text_token_ids": text_ids,
        "sampling": copy.deepcopy(SAMPLING),
        "generation_defaults": {key: value for key, value in SAMPLING.items() if key != "do_sample"},
        "n_vq": channels,
        "random_input": {"name": "random_u", "dtype": "float32", "shape": [1, channels + 1],
                         "range": "0 <= u < 1; caller must clamp float64 rounding to nextafter(float32(1),0)",
                         "order": "assistant text, audio channel 0 through channel 15"},
        "sampling_semantics": {"top_k": "retain all scores tied at kth threshold",
                               "top_p": "stable score descending, token id ascending ties; retain previous CDF <= top_p",
                               "cdf_order": "original token id ascending",
                               "cdf_normalization": "divide_by_last_cumulative_probability",
                               "cdf_accumulation": "parallel_prefix_scan_fp32",
                               "cdf_terminal_plateau": "last_positive_probability_token_and_suffix_set_to_one",
                               "inverse_cdf": "first_positive_probability_token_with_cdf_greater_equal_u; u_zero_returns_token_zero",
                               "math": "float32 softmax/CDF, original host uses float64; boundary tokens may differ",
                               "rng": "caller supplies uniforms; no random generator inside graph",
                               "eos": "audio outputs are computed but must be discarded when should_continue is zero"},
        "diagnostic_outputs": ["text_logits", "audio_logits"],
        "validation": "NOT VALIDATED: replay emitted token history through original GPU step engine, compare continuous logits, then independently validate sampling from emitted logits and supplied uniforms",
        "original_weight_initializers_unchanged": True,
        "embedding_switch_specialized": can_specialize_embedding,
        "weights_shared_across_steps": True,
        "local_kv": "reset to empty inside every frame, carried entirely on device across 17 steps",
    }
    return model, contract


def export(assets: Path, output: Path, *, copy_weights=False):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    analysis = json.loads((assets / "analysis.json").read_text())
    original = next(graph for graph in analysis["graphs"] if graph["name"] == "local_cached_step")
    source_path = (assets / original["file"]).resolve()
    metadata_path = source_path.parent / "tts_browser_onnx_meta.json"
    config = json.loads(metadata_path.read_text())["model_config"]
    source = onnx.load(source_path, load_external_data=False)
    rewritten, contract = rewrite(source, config)
    locations = _external_locations(rewritten)
    output.mkdir(parents=True, exist_ok=False)
    contract.update(source_onnx_sha256=sha256(source_path), source_onnx=str(source_path),
                    source_metadata_sha256=sha256(metadata_path), external_weights={})
    for location in locations:
        weight = (source_path.parent / location).resolve()
        destination = output / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_weights:
            shutil.copy2(weight, destination)
        else:
            try:
                destination.symlink_to(weight)
            except OSError:
                shutil.copy2(weight, destination)
        contract["external_weights"][location] = {"bytes": weight.stat().st_size, "sha256": sha256(weight)}
    graph_path = output / "local_fixed_sampled_frame.onnx"
    onnx.save_model(rewritten, graph_path)
    try:
        onnx.checker.check_model(str(graph_path))
    except onnx.checker.ValidationError as error:
        if "symbolic link" not in str(error):
            raise
        for location in locations:
            path = output / location
            if path.is_symlink():
                path.unlink()
                shutil.copy2(source_path.parent / location, path)
        onnx.checker.check_model(str(graph_path))
    contract["onnx_sha256"] = sha256(graph_path)
    contract["onnx_bytes"] = graph_path.stat().st_size
    contract["node_count"] = len(rewritten.graph.node)
    contract["initializer_count"] = len(rewritten.graph.initializer)
    contract["model_inference_performed_by_export"] = False
    candidate = {"name": "local_fixed_sampled_frame", "file": graph_path.name,
                 "precision": "fp16_transformer_with_fp32_sampling", "profiles": {}, "validation_samples": []}
    candidate_analysis = {"format_version": 1, "model": analysis.get("model"), "graphs": [candidate],
                          "sampled_frame_contract": "frame-contract.json",
                          "status": "needs_gpu_reference_capture_and_numerical_validation"}
    (output / "analysis.json").write_text(json.dumps(candidate_analysis, indent=2) + "\n")
    (output / "frame-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--copy-weights", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.assets.resolve(), args.output_dir.resolve(), copy_weights=args.copy_weights), indent=2))


if __name__ == "__main__":
    main()
