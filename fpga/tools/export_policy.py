#!/usr/bin/env python3
"""Export a Gemm/Elu/Concat ONNX policy for the RL FPGA accelerator.

Weight placement is selected with --weight-mode:

  cache   v1 behaviour. Every GEMM reads the on-chip weight cache, so the
          policy must satisfy sum(ceil(K/8)*ceil(N/8)) <= 1280.
  stream  RoboAccel-v2. Every GEMM reads its weights from DDR through the
          accelerator's AXI4 master, which removes the on-chip capacity limit.
  auto    cache when it fits, otherwise stream.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import shape_inference
from onnx.helper import get_attribute_value
from onnx.numpy_helper import to_array


LANES = 8
VECTOR_DEPTH = 512
WEIGHT_DEPTH = 1280
INSTRUCTION_DEPTH = 32
ACT_FRAC = 8
MAX_WEIGHT_FRAC = 14


def round_away(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def quantize(values: np.ndarray, frac: int) -> np.ndarray:
    scaled = round_away(np.asarray(values, dtype=np.float64) * (1 << frac))
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def sat16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -32768, 32767).astype(np.int16)


def round_shift(values: np.ndarray, shift: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if shift == 0:
        return values
    half = np.int64(1 << (shift - 1))
    return (values + np.where(values < 0, -half, half)) >> shift


def elu_q8_8(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.int32)
    negative = x < 0
    saturated = x <= -(8 << ACT_FRAC)
    magnitude = np.maximum(-x, 0)
    index = np.minimum(magnitude >> 3, 255)
    remainder = magnitude & 7
    table = round_away(
        (np.exp(-np.arange(256, dtype=np.float64) / 32.0) - 1.0) * 256.0
    ).astype(np.int16)
    next_table = np.concatenate((table[1:], np.asarray([-256], dtype=np.int16)))
    base = table[index].astype(np.int32)
    following = next_table[index].astype(np.int32)
    interpolated = base + (((following - base) * remainder + 4) >> 3)
    result = np.where(negative, interpolated, x)
    result = np.where(saturated, -(1 << ACT_FRAC), result)
    return sat16(result)


def choose_weight_frac(weight: np.ndarray) -> int:
    maximum = float(np.max(np.abs(weight))) if weight.size else 0.0
    if maximum == 0.0:
        return MAX_WEIGHT_FRAC
    frac = int(math.floor(math.log2(32767.0 / maximum)))
    return min(max(frac, 0), MAX_WEIGHT_FRAC)


def pack_word(values: np.ndarray) -> int:
    word = 0
    flat = np.asarray(values, dtype=np.int16).reshape(-1)
    for lane in range(LANES):
        value = int(flat[lane]) if lane < flat.size else 0
        word |= (value & 0xFFFF) << (16 * lane)
    return word


def word_u32(word: int) -> tuple[int, int, int, int]:
    return tuple((word >> (32 * index)) & 0xFFFFFFFF for index in range(4))


def descriptor_u32(descriptor: dict[str, Any]) -> tuple[int, int, int, int]:
    """Pack one operator into the PL instruction RAM's 128-bit format."""
    opcode = {"GEMM": 1, "ELU": 2, "NORM": 3, "CONCAT": 4}[descriptor["opcode"]]
    fields = {
        "src_base": (descriptor["src_base"], 0x1FF),
        "dst_base": (descriptor["dst_base"], 0x1FF),
        "aux0_base": (descriptor["aux0_base"], 0x1FF),
        "aux1_base": (descriptor["aux1_base"], 0x1FF),
        "weight_base": (descriptor["weight_base"], 0x7FF),
        "dim_k": (descriptor["dim_k"], 0xFFFF),
        "dim_n": (descriptor["dim_n"], 0xFFFF),
        "shift": (descriptor["shift"], 0x3F),
    }
    for name, (value, maximum) in fields.items():
        if not 0 <= int(value) <= maximum:
            raise ValueError(f"{descriptor['node']}: {name}={value} exceeds descriptor field")
    word0 = (opcode | (descriptor["src_base"] << 3) |
             (descriptor["dst_base"] << 12) | (descriptor["aux0_base"] << 21))
    word1 = (descriptor["aux1_base"] | (descriptor["weight_base"] << 9) |
             (descriptor["shift"] << 20))
    word2 = descriptor["dim_k"] | (descriptor["dim_n"] << 16)
    # word3 was reserved-zero in v1, so a cache-resident program is unchanged.
    if descriptor.get("weight_stream"):
        weight_word = int(descriptor.get("weight_word", 0))
        if not 0 <= weight_word < (1 << 31):
            raise ValueError(f"{descriptor['node']}: weight_word={weight_word} "
                             f"exceeds the 31-bit descriptor field")
        word3 = 1 | (weight_word << 1)
    else:
        word3 = 0
    return word0, word1, word2, word3 & 0xFFFFFFFF


def descriptor_word(descriptor: dict[str, Any]) -> int:
    return sum(value << (32 * index)
               for index, value in enumerate(descriptor_u32(descriptor)))


def write_hex(path: Path, words: list[int], depth: int) -> None:
    if len(words) > depth:
        raise ValueError(f"{path.name}: {len(words)} words exceeds depth {depth}")
    padded = words + [0] * (depth - len(words))
    path.write_text("".join(f"{word:032x}\n" for word in padded), encoding="ascii")


def _feature_count(value_info: onnx.ValueInfoProto) -> int:
    dimensions = value_info.type.tensor_type.shape.dim
    if len(dimensions) != 2 or dimensions[1].dim_value <= 0:
        raise ValueError(f"{value_info.name}: expected [batch, features]")
    return int(dimensions[1].dim_value)


def load_policy(path: Path, max_inputs: int = 2) -> dict[str, Any]:
    model = shape_inference.infer_shapes(onnx.load(path))
    onnx.checker.check_model(model)
    initializers = {item.name: np.asarray(to_array(item), dtype=np.float64)
                    for item in model.graph.initializer}
    input_specs = [{"name": item.name, "count": _feature_count(item)}
                   for item in model.graph.input if item.name not in initializers]
    # v1 required exactly two inputs. Every Unitree locomotion actor examined
    # has a single concatenated observation vector, and nothing in PL cares how
    # many graph inputs there are, so 1..max_inputs is accepted.
    if not 1 <= len(input_specs) <= max_inputs:
        raise ValueError(f"expected 1..{max_inputs} policy inputs, "
                         f"found {len(input_specs)}")

    value_counts: dict[str, int] = {item["name"]: item["count"] for item in input_specs}
    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        attributes = {item.name: get_attribute_value(item) for item in node.attribute}
        if node.op_type == "Gemm":
            if int(attributes.get("transA", 0)) != 0:
                raise ValueError("Gemm transA is not supported")
            if float(attributes.get("alpha", 1.0)) != 1.0 or float(attributes.get("beta", 1.0)) != 1.0:
                raise ValueError("Gemm alpha/beta other than 1 are not supported")
            weight = initializers[node.input[1]]
            if int(attributes.get("transB", 0)) == 0:
                weight = weight.T
            bias = initializers[node.input[2]].reshape(-1)
            if node.input[0] not in value_counts:
                raise ValueError(f"Gemm input {node.input[0]} has no known shape")
            if weight.ndim != 2 or weight.shape[1] != value_counts[node.input[0]]:
                raise ValueError(f"{node.name}: incompatible weight shape {weight.shape}")
            if bias.size != weight.shape[0]:
                raise ValueError(f"{node.name}: incompatible bias shape {bias.shape}")
            value_counts[node.output[0]] = int(weight.shape[0])
            nodes.append({"op": "GEMM", "name": node.name or f"gemm_{index}",
                          "inputs": [node.input[0]], "output": node.output[0],
                          "weight": weight, "bias": bias})
        elif node.op_type == "Elu":
            if float(attributes.get("alpha", 1.0)) != 1.0:
                raise ValueError("only ELU alpha=1 is supported")
            value_counts[node.output[0]] = value_counts[node.input[0]]
            nodes.append({"op": "ELU", "name": node.name or f"elu_{index}",
                          "inputs": [node.input[0]], "output": node.output[0]})
        elif node.op_type == "Concat":
            if int(attributes.get("axis", -1)) not in (-1, 1):
                raise ValueError("only feature-axis Concat is supported")
            if len(node.input) != 2:
                raise ValueError("the PL CONCAT operator accepts exactly two inputs")
            value_counts[node.output[0]] = sum(value_counts[name] for name in node.input)
            nodes.append({"op": "CONCAT", "name": node.name or f"concat_{index}",
                          "inputs": list(node.input), "output": node.output[0]})
        else:
            raise ValueError(f"unsupported ONNX operator {node.op_type} ({node.name})")

    if len(model.graph.output) != 1:
        raise ValueError("expected one policy output")
    output_name = model.graph.output[0].name
    return {"path": str(path), "inputs": input_specs, "nodes": nodes,
            "output_name": output_name, "output_count": value_counts[output_name],
            "value_counts": value_counts}


def build_images(policy: dict[str, Any], weight_mode: str = "cache") -> dict[str, Any]:
    vector_words = [0] * VECTOR_DEPTH
    tensor_layout: dict[str, dict[str, int]] = {}
    cursor = 0

    def allocate(name: str, count: int) -> int:
        nonlocal cursor
        words = math.ceil(count / LANES)
        if cursor + words > VECTOR_DEPTH:
            raise ValueError(f"vector cache exhausted while allocating {name}")
        tensor_layout[name] = {"base": cursor, "elements": count, "words": words}
        cursor += words
        return tensor_layout[name]["base"]

    def place(base: int, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.int16).reshape(-1)
        for group in range(math.ceil(flat.size / LANES)):
            vector_words[base + group] = pack_word(flat[group * LANES:(group + 1) * LANES])

    for item in policy["inputs"]:
        allocate(item["name"], item["count"])

    weight_banks: list[list[int]] = [[] for _ in range(LANES)]
    descriptors: list[dict[str, Any]] = []
    quant_layers: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    gemm_index = 0

    for node in policy["nodes"]:
        op = node["op"]
        if op == "GEMM":
            source = tensor_layout[node["inputs"][0]]
            output_count = int(node["weight"].shape[0])
            output_base = allocate(node["output"], output_count)
            bias_name = f"bias_{gemm_index}"
            bias_base = allocate(bias_name, output_count)
            weight_frac = choose_weight_frac(node["weight"])
            weight_q = quantize(node["weight"], weight_frac)
            bias_q = quantize(node["bias"], ACT_FRAC)
            place(bias_base, bias_q)
            quant_layers[node["output"]] = (weight_q, bias_q, weight_frac)

            layer_base = len(weight_banks[0])
            out_count, in_count = weight_q.shape
            for out_group in range(math.ceil(out_count / LANES)):
                for in_group in range(math.ceil(in_count / LANES)):
                    for bank in range(LANES):
                        output_index = out_group * LANES + bank
                        chunk = np.zeros(LANES, dtype=np.int16)
                        if output_index < out_count:
                            start = in_group * LANES
                            valid = weight_q[output_index, start:min(start + LANES, in_count)]
                            chunk[:valid.size] = valid
                        weight_banks[bank].append(pack_word(chunk))
            descriptors.append({"opcode": "GEMM", "src_base": source["base"],
                                "dst_base": output_base, "aux0_base": bias_base,
                                "aux1_base": 0, "weight_base": layer_base,
                                "dim_k": in_count, "dim_n": out_count,
                                "shift": weight_frac, "node": node["name"],
                                "weight_stream": False, "weight_word": layer_base})
            gemm_index += 1
        elif op == "ELU":
            source = tensor_layout[node["inputs"][0]]
            tensor_layout[node["output"]] = dict(source)
            descriptors.append({"opcode": "ELU", "src_base": source["base"],
                                "dst_base": source["base"], "aux0_base": 0,
                                "aux1_base": 0, "weight_base": 0, "dim_k": 0,
                                "dim_n": source["elements"], "shift": 0,
                                "node": node["name"]})
        elif op == "CONCAT":
            left = tensor_layout[node["inputs"][0]]
            right = tensor_layout[node["inputs"][1]]
            output_base = allocate(node["output"], left["elements"] + right["elements"])
            descriptors.append({"opcode": "CONCAT", "src_base": left["base"],
                                "dst_base": output_base, "aux0_base": right["base"],
                                "aux1_base": 0, "weight_base": 0,
                                "dim_k": left["elements"], "dim_n": right["elements"],
                                "shift": 0, "node": node["name"]})

    weight_used = len(weight_banks[0])
    if weight_mode not in ("cache", "stream", "auto"):
        raise ValueError(f"unknown weight mode {weight_mode}")
    stream = (weight_mode == "stream" or
              (weight_mode == "auto" and weight_used > WEIGHT_DEPTH))
    if stream:
        # weight_base addresses the on-chip cache and is meaningless while
        # streaming; the DDR word index lives in word3 instead.
        for descriptor in descriptors:
            if descriptor["opcode"] == "GEMM":
                descriptor["weight_stream"] = True
                descriptor["weight_base"] = 0
    elif weight_used > WEIGHT_DEPTH:
        raise ValueError(f"policy needs {weight_used} weight words/bank; "
                         f"limit is {WEIGHT_DEPTH}. Use --weight-mode stream.")
    if len(descriptors) > INSTRUCTION_DEPTH:
        raise ValueError(f"policy needs {len(descriptors)} instructions; limit is {INSTRUCTION_DEPTH}")
    for descriptor in descriptors:
        descriptor_u32(descriptor)
    return {"vector_words": vector_words, "weight_banks": weight_banks,
            "tensor_layout": tensor_layout, "descriptors": descriptors,
            "quant_layers": quant_layers, "vector_words_used": cursor,
            "weight_words_used": weight_used, "weight_stream": stream}


def ddr_weight_words(weight_banks: list[list[int]]) -> list[int]:
    """One 1024-bit word per weight tile row, bank 0 in the least significant
    128 bits -- the same order rl_weight_cache presents on eng_rd_data."""
    depth = len(weight_banks[0])
    words = []
    for index in range(depth):
        word = 0
        for bank in range(LANES):
            word |= weight_banks[bank][index] << (128 * bank)
        words.append(word)
    return words


def write_ddr_image(path: Path, words: list[int]) -> None:
    path.write_bytes(b"".join(word.to_bytes(128, "little") for word in words))


def write_ddr_hex(path: Path, words: list[int]) -> None:
    path.write_text("".join(f"{word:0256x}\n" for word in words), encoding="ascii")


def fixed_inference(policy: dict[str, Any], images: dict[str, Any],
                    inputs_q: dict[str, np.ndarray]) -> np.ndarray:
    values = {name: np.asarray(value, dtype=np.int16).reshape(-1)
              for name, value in inputs_q.items()}
    for node in policy["nodes"]:
        if node["op"] == "GEMM":
            weight_q, bias_q, weight_frac = images["quant_layers"][node["output"]]
            accum = weight_q.astype(np.int64) @ values[node["inputs"][0]].astype(np.int64)
            values[node["output"]] = sat16(
                round_shift(accum, weight_frac) + bias_q.astype(np.int64))
        elif node["op"] == "ELU":
            values[node["output"]] = elu_q8_8(values[node["inputs"][0]])
        elif node["op"] == "CONCAT":
            values[node["output"]] = np.concatenate(
                [values[name] for name in node["inputs"]]).astype(np.int16)
    return values[policy["output_name"]]


def float_inference(policy: dict[str, Any], inputs: dict[str, np.ndarray]) -> np.ndarray:
    values = {name: np.asarray(value, dtype=np.float64).reshape(-1)
              for name, value in inputs.items()}
    for node in policy["nodes"]:
        if node["op"] == "GEMM":
            values[node["output"]] = node["weight"] @ values[node["inputs"][0]] + node["bias"]
        elif node["op"] == "ELU":
            source = values[node["inputs"][0]]
            values[node["output"]] = np.where(source >= 0.0, source, np.exp(source) - 1.0)
        elif node["op"] == "CONCAT":
            values[node["output"]] = np.concatenate([values[name] for name in node["inputs"]])
    return values[policy["output_name"]]


def make_test_inputs(policy: dict[str, Any], variant: int = 0) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for input_index, item in enumerate(policy["inputs"]):
        x = np.arange(item["count"], dtype=np.float64)
        phase = 0.31 * variant + 0.47 * input_index
        result[item["name"]] = (0.70 * np.sin((0.09 + 0.01 * variant) * x + phase) +
                                0.25 * np.cos(0.037 * x + 0.5 * phase))
    return result


def evaluate(policy: dict[str, Any], images: dict[str, Any], samples: int,
             seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    errors = []
    fixed_outputs = []
    float_outputs = []
    for _ in range(samples):
        floating_inputs = {item["name"]: rng.normal(0.0, 1.0, item["count"])
                           for item in policy["inputs"]}
        fixed_inputs = {name: quantize(value, ACT_FRAC)
                        for name, value in floating_inputs.items()}
        floating = float_inference(policy, floating_inputs)
        fixed = fixed_inference(policy, images, fixed_inputs).astype(np.float64) / (1 << ACT_FRAC)
        errors.append(np.abs(floating - fixed))
        fixed_outputs.append(fixed)
        float_outputs.append(floating)
    errors_array = np.asarray(errors)
    fixed_array = np.asarray(fixed_outputs)
    float_array = np.asarray(float_outputs)
    return {"samples": samples, "mean_absolute_error": float(errors_array.mean()),
            "max_absolute_error": float(errors_array.max()),
            "rmse": float(np.sqrt(np.mean((fixed_array - float_array) ** 2))),
            "argmax_agreement": float(np.mean(
                np.argmax(fixed_array, axis=1) == np.argmax(float_array, axis=1)))}


def write_c_images(path: Path, policy: dict[str, Any], images: dict[str, Any],
                   selftest_q: dict[str, np.ndarray], actions_q: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if [item["name"] for item in policy["inputs"]] != ["obs", "obs_history"]:
        raise ValueError("the public PS API requires inputs named obs and obs_history")
    input0, input1 = policy["inputs"]
    input0_layout = images["tensor_layout"][input0["name"]]
    input1_layout = images["tensor_layout"][input1["name"]]
    output_layout = images["tensor_layout"][policy["output_name"]]
    header = f"""#ifndef RL_POLICY_DATA_H
#define RL_POLICY_DATA_H

#include <stdint.h>
#include "rl_accel.h"

#define RL_POLICY_VECTOR_WORDS {VECTOR_DEPTH}u
#define RL_POLICY_VECTOR_WORDS_USED {images['vector_words_used']}u
#define RL_POLICY_WEIGHT_BANKS {LANES}u
#define RL_POLICY_WEIGHT_WORDS {WEIGHT_DEPTH}u
#define RL_POLICY_WEIGHT_WORDS_USED {images['weight_words_used']}u
#define RL_POLICY_OBS_COUNT {input0['count']}u
#define RL_POLICY_HISTORY_COUNT {input1['count']}u
#define RL_POLICY_ACTIONS {policy['output_count']}u
#define RL_POLICY_OBS_BASE {input0_layout['base']}u
#define RL_POLICY_HISTORY_BASE {input1_layout['base']}u
#define RL_POLICY_ACTION_BASE {output_layout['base']}u
#define RL_POLICY_COMMAND_COUNT {len(images['descriptors'])}u
#define RL_POLICY_OBS_WORDS {input0_layout['words']}u
#define RL_POLICY_HISTORY_WORDS {input1_layout['words']}u

extern const uint32_t rl_policy_vector_image[RL_POLICY_VECTOR_WORDS][4];
extern const uint32_t rl_policy_weight_image[RL_POLICY_WEIGHT_BANKS][RL_POLICY_WEIGHT_WORDS][4];
extern const rl_accel_cmd_t rl_policy_commands[RL_POLICY_COMMAND_COUNT];
extern const uint32_t rl_policy_selftest_obs[RL_POLICY_OBS_WORDS][4];
extern const uint32_t rl_policy_selftest_history[RL_POLICY_HISTORY_WORDS][4];
extern const int16_t rl_policy_selftest_actions[RL_POLICY_ACTIONS];

#endif
"""
    (path / "rl_policy_data.h").write_text(header, encoding="ascii")

    def row(word: int) -> str:
        values = word_u32(word)
        return "    {" + ", ".join(f"0x{value:08x}u" for value in values) + "}"

    lines = ['#include "rl_policy_data.h"', "",
             "const uint32_t rl_policy_vector_image[RL_POLICY_VECTOR_WORDS][4] = {"]
    lines.extend(row(word) + ("," if index + 1 < VECTOR_DEPTH else "")
                 for index, word in enumerate(images["vector_words"]))
    lines.extend(["};", "",
                  "const uint32_t rl_policy_weight_image[RL_POLICY_WEIGHT_BANKS][RL_POLICY_WEIGHT_WORDS][4] = {"])
    for bank, words in enumerate(images["weight_banks"]):
        padded = words + [0] * (WEIGHT_DEPTH - len(words))
        lines.append("  {")
        lines.extend("  " + row(word) + ("," if index + 1 < WEIGHT_DEPTH else "")
                     for index, word in enumerate(padded))
        lines.append("  }" + ("," if bank + 1 < LANES else ""))
    lines.extend(["};", "", "const rl_accel_cmd_t rl_policy_commands[RL_POLICY_COMMAND_COUNT] = {"])
    for index, descriptor in enumerate(images["descriptors"]):
        suffix = "," if index + 1 < len(images["descriptors"]) else ""
        lines.append(
            "    {RL_OP_%s, %uu, %uu, %uu, %uu, %uu, %uu, %uu, %uu}%s" % (
                descriptor["opcode"], descriptor["src_base"], descriptor["dst_base"],
                descriptor["aux0_base"], descriptor["aux1_base"], descriptor["weight_base"],
                descriptor["dim_k"], descriptor["dim_n"], descriptor["shift"], suffix))
    lines.extend(["};", ""])

    for array_name, item, macro in (
            ("rl_policy_selftest_obs", input0, "RL_POLICY_OBS_WORDS"),
            ("rl_policy_selftest_history", input1, "RL_POLICY_HISTORY_WORDS")):
        values = selftest_q[item["name"]]
        words = [pack_word(values[group * LANES:(group + 1) * LANES])
                 for group in range(math.ceil(values.size / LANES))]
        lines.append(f"const uint32_t {array_name}[{macro}][4] = {{")
        lines.extend(row(word) + ("," if index + 1 < len(words) else "")
                     for index, word in enumerate(words))
        lines.extend(["};", ""])
    lines.append("const int16_t rl_policy_selftest_actions[RL_POLICY_ACTIONS] = {")
    lines.append("    " + ", ".join(str(int(value)) for value in actions_q))
    lines.extend(["};", ""])
    (path / "rl_policy_data.c").write_text("\n".join(lines), encoding="ascii")


def export_policy(model_path: Path, out_path: Path, samples: int, seed: int,
                  c_out: Path | None = None,
                  weight_mode: str = "cache") -> tuple[dict[str, Any], dict[str, Any]]:
    policy = load_policy(model_path)
    images = build_images(policy, weight_mode)
    out_path.mkdir(parents=True, exist_ok=True)
    write_hex(out_path / "vector_cache.hex", images["vector_words"], VECTOR_DEPTH)
    instruction_words = [descriptor_word(item) for item in images["descriptors"]]
    write_hex(out_path / "instruction_program.hex", instruction_words,
              INSTRUCTION_DEPTH)
    if images["weight_stream"]:
        words = ddr_weight_words(images["weight_banks"])
        write_ddr_image(out_path / "weights_ddr.bin", words)
        write_ddr_hex(out_path / "weights_ddr.hex", words)
    else:
        for bank, words in enumerate(images["weight_banks"]):
            write_hex(out_path / f"weight_bank{bank}.hex", words, WEIGHT_DEPTH)
            write_hex(out_path / f"weight_bank{bank}_main.hex", words[:1024], 1024)
            write_hex(out_path / f"weight_bank{bank}_tail.hex", words[1024:], 256)

    floating_inputs = make_test_inputs(policy)
    selftest_q = {name: quantize(value, ACT_FRAC) for name, value in floating_inputs.items()}
    actions_q = fixed_inference(policy, images, selftest_q)
    for index, item in enumerate(policy["inputs"]):
        values = selftest_q[item["name"]]
        words = [pack_word(values[group * LANES:(group + 1) * LANES])
                 for group in range(math.ceil(values.size / LANES))]
        write_hex(out_path / f"selftest_input{index}.hex", words, len(words))
    write_hex(out_path / "selftest_actions.hex",
              [pack_word(actions_q[group * LANES:(group + 1) * LANES])
               for group in range(math.ceil(actions_q.size / LANES))],
              math.ceil(actions_q.size / LANES))
    if c_out is not None:
        write_c_images(c_out, policy, images, selftest_q, actions_q)

    metrics = evaluate(policy, images, samples, seed)
    manifest = {
        "model": model_path.name,
        "format": {"input_activation_and_bias": "signed Q8.8",
                   "weight": "signed INT16 with per-GEMM fractional bits",
                   "accumulator": "signed INT48"},
        "inputs": policy["inputs"],
        "output": {"name": policy["output_name"], "count": policy["output_count"]},
        "tensor_layout_words": images["tensor_layout"],
        "vector_words_used": images["vector_words_used"],
        "vector_depth_words": VECTOR_DEPTH,
        "weight_words_per_bank": images["weight_words_used"],
        "weight_depth_words_per_bank": WEIGHT_DEPTH,
        "weight_mode": "stream" if images["weight_stream"] else "cache",
        "weight_ddr_bytes": (images["weight_words_used"] * 128
                             if images["weight_stream"] else 0),
        "instruction_count": len(images["descriptors"]),
        "instruction_depth": INSTRUCTION_DEPTH,
        "instruction_encoding": {
            "word0": "opcode[2:0], src[11:3], dst[20:12], aux0[29:21]",
            "word1": "aux1[8:0], weight[19:9], shift[25:20]",
            "word2": "dim_k[15:0], dim_n[31:16]",
            "word3": "weight_stream[0], weight_ddr_word[31:1]",
        },
        "instruction_program_u32": [list(descriptor_u32(item))
                                    for item in images["descriptors"]],
        "selftest_actions_q8_8": actions_q.astype(int).tolist(),
        "descriptors": images["descriptors"],
        "random_standard_normal_input_metrics": metrics}
    (out_path / "policy_map.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, {"policy": policy, "images": images,
                      "selftest_q": selftest_q, "actions_q": actions_q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("fudan_policy.onnx"))
    parser.add_argument("--out", type=Path, default=Path("generated"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--c-out", type=Path, default=None)
    parser.add_argument("--weight-mode", choices=("cache", "stream", "auto"),
                        default="cache",
                        help="where GEMM weights live: the on-chip cache (v1), "
                             "DDR through the v2 AXI master, or cache-if-it-fits")
    args = parser.parse_args()
    manifest, _ = export_policy(args.model, args.out, args.samples, args.seed,
                                args.c_out, args.weight_mode)
    print(json.dumps({"model": manifest["model"], "inputs": manifest["inputs"],
                      "output": manifest["output"],
                      "commands": len(manifest["descriptors"]),
                      "vector_words_used": manifest["vector_words_used"],
                      "weight_words_per_bank": manifest["weight_words_per_bank"],
                      "weight_mode": manifest["weight_mode"],
                      "metrics": manifest["random_standard_normal_input_metrics"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
