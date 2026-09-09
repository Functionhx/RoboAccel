#!/usr/bin/env python3
"""Report whether an ONNX policy maps onto the rl_accel PL without RTL changes.

The accelerator's limits are read from analysis/accelerator_constraints.json so
that this tool and the documentation cannot drift apart.  Every capacity rule
mirrors tools/export_policy.py; the cycle model is the one measured by
tb/tb_cycle_model.sv.

    python tools/analyze_policy.py models/go2/go2_robot_lab_actor.onnx
    python tools/analyze_policy.py --json report.json policy.onnx
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

ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS_PATH = ROOT / "analysis" / "accelerator_constraints.json"


def load_constraints(path: Path = CONSTRAINTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Finding:
    """One reason a policy does or does not map, with the number behind it."""

    def __init__(self, category: str, severity: str, message: str,
                 required: float | None = None, available: float | None = None) -> None:
        self.category = category
        self.severity = severity          # "fail" | "warn"
        self.message = message
        self.required = required
        self.available = available

    @property
    def overage(self) -> float | None:
        if self.required is None or not self.available:
            return None
        return self.required / self.available

    def as_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {"category": self.category, "severity": self.severity,
                                "message": self.message}
        if self.required is not None:
            item["required"] = self.required
            item["available"] = self.available
            if self.overage is not None:
                item["overage_ratio"] = round(self.overage, 4)
        return item

    def __str__(self) -> str:
        if self.overage is not None and self.overage > 1.0:
            return (f"{self.message} "
                    f"(needs {self.required:g}, capacity {self.available:g}, "
                    f"{self.overage:.2f}x over)")
        return self.message


def feature_count(value_info: onnx.ValueInfoProto) -> int | None:
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != 2:
        return None
    return int(dims[1].dim_value) if dims[1].dim_value > 0 else None


def analyze(model_path: Path, constraints: dict[str, Any],
            label: str | None = None) -> dict[str, Any]:
    lanes = constraints["datapath"]["lanes"]
    cap = constraints["capacity"]
    fields = constraints["descriptor_fields"]
    cyc = constraints["latency_model_cycles"]
    clock = constraints["device"]["pl_clock_hz"]

    model = onnx.load(model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:                                   # noqa: BLE001
        pass

    initializers = {item.name: np.asarray(to_array(item), dtype=np.float64)
                    for item in model.graph.initializer}
    graph_inputs = [item for item in model.graph.input if item.name not in initializers]

    findings: list[Finding] = []
    supported_ops = set(constraints["operators"])

    # ---- topology -----------------------------------------------------------
    if len(graph_inputs) != constraints["topology"]["graph_inputs_required"]:
        findings.append(Finding(
            "topology", "fail",
            f"exporter requires exactly "
            f"{constraints['topology']['graph_inputs_required']} graph inputs; "
            f"this policy has {len(graph_inputs)} "
            f"({', '.join(i.name for i in graph_inputs)})"))
    if len(model.graph.output) != 1:
        findings.append(Finding("topology", "fail",
                                f"expected 1 graph output, found {len(model.graph.output)}"))

    counts: dict[str, int] = {}
    inputs_report = []
    for item in graph_inputs:
        n = feature_count(item)
        inputs_report.append({"name": item.name, "features": n})
        if n is None:
            findings.append(Finding("topology", "fail",
                                    f"input {item.name} is not a static [batch, features] tensor"))
        else:
            counts[item.name] = n

    # ---- operator walk ------------------------------------------------------
    op_histogram: dict[str, int] = {}
    op_status: dict[str, str] = {}
    layers: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for index, node in enumerate(model.graph.node):
        op_histogram[node.op_type] = op_histogram.get(node.op_type, 0) + 1
        attrs = {a.name: get_attribute_value(a) for a in node.attribute}
        name = node.name or f"{node.op_type.lower()}_{index}"

        if node.op_type == "Gemm":
            weight = initializers.get(node.input[1])
            if weight is None:
                op_status["Gemm"] = "FAIL"
                unsupported.append(node.op_type)
                findings.append(Finding("operator", "fail",
                                        f"{name}: Gemm B operand is not a constant"))
                continue
            if int(attrs.get("transB", 0)) == 0:
                weight = weight.T
            if int(attrs.get("transA", 0)) != 0:
                findings.append(Finding("operator", "fail", f"{name}: Gemm transA != 0"))
            if float(attrs.get("alpha", 1.0)) != 1.0 or float(attrs.get("beta", 1.0)) != 1.0:
                findings.append(Finding("operator", "fail",
                                        f"{name}: Gemm alpha/beta must both be 1"))
            if len(node.input) < 3:
                findings.append(Finding("operator", "fail",
                                        f"{name}: the GEMM operator always adds a bias; "
                                        f"this Gemm has none"))
                bias_size = int(weight.shape[0])
            else:
                bias_size = int(initializers[node.input[2]].size)
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            counts[node.output[0]] = out_features
            op_status.setdefault("Gemm", "PASS")
            layers.append({"op": "GEMM", "name": name,
                           "in_features": in_features, "out_features": out_features,
                           "params": int(weight.size) + bias_size,
                           "macs": in_features * out_features,
                           "weight_words": math.ceil(in_features / lanes) *
                                           math.ceil(out_features / lanes),
                           "weight_abs_max": float(np.max(np.abs(weight))) if weight.size else 0.0})
            descriptors.append({"op": "GEMM", "k": in_features, "n": out_features})

        elif node.op_type == "Elu":
            if float(attrs.get("alpha", 1.0)) != 1.0:
                findings.append(Finding("operator", "fail",
                                        f"{name}: only ELU alpha=1 is implemented"))
            src = node.input[0]
            counts[node.output[0]] = counts.get(src, 0)
            op_status.setdefault("Elu", "PASS")
            layers.append({"op": "ELU", "name": name, "elements": counts.get(src, 0)})
            descriptors.append({"op": "ELU", "k": 0, "n": counts.get(src, 0)})

        elif node.op_type == "Concat":
            if int(attrs.get("axis", -1)) not in (-1, 1):
                findings.append(Finding("operator", "fail",
                                        f"{name}: only feature-axis Concat is supported"))
            if len(node.input) != 2:
                findings.append(Finding("operator", "fail",
                                        f"{name}: the CONCAT operator takes exactly two "
                                        f"inputs, this one has {len(node.input)}"))
            left = counts.get(node.input[0], 0)
            right = counts.get(node.input[1], 0) if len(node.input) > 1 else 0
            counts[node.output[0]] = left + right
            op_status.setdefault("Concat", "PASS")
            layers.append({"op": "CONCAT", "name": name, "left": left, "right": right})
            descriptors.append({"op": "CONCAT", "k": left, "n": right})

        else:
            op_status[node.op_type] = "FAIL"
            if node.op_type not in unsupported:
                unsupported.append(node.op_type)
            # Keep walking so capacity is still reported: assume shape-preserving.
            if node.input and node.input[0] in counts:
                counts[node.output[0]] = counts[node.input[0]]

    for op in unsupported:
        findings.append(Finding("operator", "fail",
                                f"{op} is not implemented in PL "
                                f"(supported: {', '.join(sorted(supported_ops))})"))

    # ---- capacity, mirroring the exporter's bump allocator ------------------
    vector_words = sum(math.ceil(c / lanes) for c in
                       (counts[i["name"]] for i in inputs_report if i["features"]))
    for layer in layers:
        if layer["op"] == "GEMM":
            vector_words += 2 * math.ceil(layer["out_features"] / lanes)   # output + bias
        elif layer["op"] == "CONCAT":
            vector_words += math.ceil((layer["left"] + layer["right"]) / lanes)
        # ELU is in place and allocates nothing.

    weight_words = sum(layer["weight_words"] for layer in layers if layer["op"] == "GEMM")
    instructions = len(descriptors)

    if vector_words > cap["vector_cache_words"]:
        findings.append(Finding("vector_memory", "fail",
                                "vector cache exceeded", vector_words,
                                cap["vector_cache_words"]))
    if weight_words > cap["weight_cache_words_per_bank"]:
        findings.append(Finding("weight_memory", "fail",
                                "weight cache exceeded", weight_words,
                                cap["weight_cache_words_per_bank"]))
    if instructions > cap["instruction_depth"]:
        findings.append(Finding("instruction_memory", "fail",
                                "instruction RAM exceeded", instructions,
                                cap["instruction_depth"]))

    # ---- descriptor field ranges -------------------------------------------
    for layer in layers:
        if layer["op"] == "GEMM":
            for key, value in (("dim_k", layer["in_features"]), ("dim_n", layer["out_features"])):
                if value > fields[key]["max"]:
                    findings.append(Finding("descriptor", "fail",
                                            f"{layer['name']}: {key}={value} exceeds "
                                            f"the {fields[key]['bits']}-bit field",
                                            value, fields[key]["max"]))
    if vector_words > fields["src_base"]["max"] + 1:
        findings.append(Finding("descriptor", "fail",
                                "tensor bases would exceed the 9-bit vector address field",
                                vector_words, fields["src_base"]["max"] + 1))
    if weight_words > fields["weight_base"]["max"] + 1:
        findings.append(Finding("descriptor", "fail",
                                "weight bases would exceed the 11-bit weight address field",
                                weight_words, fields["weight_base"]["max"] + 1))

    # ---- quantization headroom ---------------------------------------------
    max_frac = constraints["datapath"]["weight_frac_bits_max"]
    for layer in layers:
        if layer["op"] != "GEMM":
            continue
        amax = layer["weight_abs_max"]
        if amax > 0.0:
            frac = min(max(int(math.floor(math.log2(32767.0 / amax))), 0), max_frac)
            layer["weight_frac_bits"] = frac
            if frac == 0:
                findings.append(Finding("quantization", "warn",
                                        f"{layer['name']}: |w|max={amax:.3g} leaves no "
                                        f"fractional bits at int16"))

    # ---- latency, using the measured cycle model ---------------------------
    cycles = cyc["sequence_constant"] + cyc["per_instruction_dispatch"] * instructions
    for d in descriptors:
        kg, ng = math.ceil(d["k"] / lanes), math.ceil(d["n"] / lanes)
        if d["op"] == "GEMM":
            cycles += ng * (kg + 12)
        elif d["op"] == "ELU":
            cycles += 7 * ng
        elif d["op"] == "CONCAT":
            total = d["k"] + d["n"]
            cycles += 3 * total + math.ceil(total / lanes)

    macs = sum(layer.get("macs", 0) for layer in layers)
    params = sum(layer.get("params", 0) for layer in layers)
    ideal = math.ceil(macs / constraints["datapath"]["macs_per_cycle"])

    hard = [f for f in findings if f.severity == "fail"]
    report = {
        "model": str(model_path),
        "label": label or model_path.stem,
        "inputs": inputs_report,
        "output": {"name": model.graph.output[0].name if model.graph.output else None,
                   "features": counts.get(model.graph.output[0].name) if model.graph.output else None},
        "op_histogram": op_histogram,
        "op_status": op_status,
        "unsupported_ops": unsupported,
        "layers": layers,
        "parameters": params,
        "macs": macs,
        "weight_words_per_bank": weight_words,
        "weight_capacity_words": cap["weight_cache_words_per_bank"],
        "weight_utilization": round(weight_words / cap["weight_cache_words_per_bank"], 4),
        "vector_words": vector_words,
        "vector_capacity_words": cap["vector_cache_words"],
        "vector_utilization": round(vector_words / cap["vector_cache_words"], 4),
        "instructions": instructions,
        "instruction_capacity": cap["instruction_depth"],
        "estimated_cycles": cycles,
        "estimated_latency_us": round(cycles / clock * 1e6, 3),
        "ideal_cycles_at_64_mac_per_cycle": ideal,
        "mac_array_utilization": round(ideal / cycles, 4) if cycles else 0.0,
        "maps_without_rtl_change": not hard,
        "findings": [f.as_dict() for f in findings],
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"MODEL: {report['label']}")
    print(f"  file        : {report['model']}")
    ins = ", ".join(f"{i['name']}[{i['features']}]" for i in report["inputs"])
    print(f"  inputs      : {ins}")
    print(f"  output      : {report['output']['name']}[{report['output']['features']}]")
    print(f"  Parameters  : {report['parameters']:,}")
    print(f"  MACs        : {report['macs']:,}")
    print(f"  Weight storage : {report['weight_words_per_bank']:,} words/bank "
          f"of {report['weight_capacity_words']:,} "
          f"({report['weight_utilization'] * 100:.1f}%)")
    print(f"  Vector storage : {report['vector_words']:,} words "
          f"of {report['vector_capacity_words']:,} "
          f"({report['vector_utilization'] * 100:.1f}%)")
    print(f"  Instructions   : {report['instructions']} of {report['instruction_capacity']}")
    print()
    print("  Layers:")
    for layer in report["layers"]:
        if layer["op"] == "GEMM":
            print(f"    GEMM   {layer['in_features']:>4} -> {layer['out_features']:<4} "
                  f"weights={layer['weight_words']:>5} words  "
                  f"frac={layer.get('weight_frac_bits', '-')}")
        elif layer["op"] == "ELU":
            print(f"    ELU    {layer['elements']:>4}")
        else:
            print(f"    CONCAT {layer['left']} + {layer['right']}")
    print()
    print("  Supported ops:")
    for op, status in sorted(report["op_status"].items()):
        print(f"    {op:<10} {status}")
    print()
    print(f"  Estimated latency: {report['estimated_cycles']:,} cycles "
          f"= {report['estimated_latency_us']:.2f} us at 100 MHz")
    print(f"  MAC-array utilization: {report['mac_array_utilization'] * 100:.1f}% "
          f"(ideal {report['ideal_cycles_at_64_mac_per_cycle']:,} cycles)")
    print()
    verdict = "PASS" if report["maps_without_rtl_change"] else "FAIL"
    print(f"  Overall mapping: {verdict}")
    hard = [f for f in report["findings"] if f["severity"] == "fail"]
    warn = [f for f in report["findings"] if f["severity"] == "warn"]
    if hard:
        print("\n  Reasons:")
        for number, item in enumerate(hard, start=1):
            extra = ""
            if "overage_ratio" in item and item["overage_ratio"] > 1.0:
                extra = (f" (needs {item['required']:g}, capacity {item['available']:g}, "
                         f"{item['overage_ratio']:.2f}x over)")
            print(f"  {number}. [{item['category']}] {item['message']}{extra}")
    if warn:
        print("\n  Warnings:")
        for number, item in enumerate(warn, start=1):
            print(f"  {number}. [{item['category']}] {item['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, help="write the machine-readable report here")
    parser.add_argument("--constraints", type=Path, default=CONSTRAINTS_PATH)
    parser.add_argument("--label", action="append", default=None)
    args = parser.parse_args()

    constraints = load_constraints(args.constraints)
    reports = []
    for index, path in enumerate(args.model):
        label = args.label[index] if args.label and index < len(args.label) else None
        report = analyze(path, constraints, label)
        reports.append(report)
        print_report(report)
        print("=" * 72)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports if len(reports) > 1 else reports[0],
                                        indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if all(r["maps_without_rtl_change"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
