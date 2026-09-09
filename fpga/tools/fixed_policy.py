#!/usr/bin/env python3
"""Run a policy with exactly the arithmetic RoboAccel executes.

This mirrors tools/export_policy.py: Q8.8 activations and bias, per-GEMM INT16
weight fractional bits chosen from |w|max, round-half-away-from-zero requantize,
INT16 saturation, and the Q8.8 ELU lookup with its 3-bit interpolation. A policy
wrapped here produces the same actions the FPGA produces, so it can be put
straight into the closed-loop harness.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx.helper import get_attribute_value
from onnx.numpy_helper import to_array

ACT_FRAC = 8
MAX_WEIGHT_FRAC = 14


def _round_away(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5))


def _quantize(v, frac):
    return np.clip(_round_away(np.asarray(v, np.float64) * (1 << frac)),
                   -32768, 32767).astype(np.int64)


def _sat16(v):
    return np.clip(v, -32768, 32767).astype(np.int64)


def _round_shift(v, shift):
    v = np.asarray(v, dtype=np.int64)
    if shift == 0:
        return v
    half = np.int64(1 << (shift - 1))
    return (v + np.where(v < 0, -half, half)) >> shift


_TABLE = _round_away((np.exp(-np.arange(256, dtype=np.float64) / 32.0) - 1.0)
                     * 256.0).astype(np.int64)
_NEXT = np.concatenate([_TABLE[1:], np.array([-256], dtype=np.int64)])


def _elu_q8_8(x):
    x = np.asarray(x, dtype=np.int64)
    neg = x < 0
    sat = x <= -(8 << ACT_FRAC)
    mag = np.maximum(-x, 0)
    idx = np.minimum(mag >> 3, 255)
    rem = mag & 7
    base, nxt = _TABLE[idx], _NEXT[idx]
    interp = base + (((nxt - base) * rem + 4) >> 3)
    out = np.where(neg, interp, x)
    out = np.where(sat, -(1 << ACT_FRAC), out)
    return _sat16(out)


WEIGHT_QMAX = {4: 7, 8: 127, 16: 32767}


def _weight_frac(w, bits=16):
    m = float(np.max(np.abs(w))) if w.size else 0.0
    if m == 0.0:
        return MAX_WEIGHT_FRAC
    qmax = float(WEIGHT_QMAX[bits])
    return int(min(max(math.floor(math.log2(qmax / m)), 0), MAX_WEIGHT_FRAC))


def _quantize_weight(v, frac, bits=16):
    """INT8 weights ride in the same INT16 lanes; only the clip range narrows,
    so the MAC array and the requantizer are untouched."""
    lo, hi = -WEIGHT_QMAX[bits] - 1, WEIGHT_QMAX[bits]
    return np.clip(_round_away(np.asarray(v, np.float64) * (1 << frac)),
                   lo, hi).astype(np.int64)


ACT_QMAX = {4: 7, 8: 127, 16: 32767}


def _sat_act(v, bits):
    return np.clip(v, -ACT_QMAX[bits] - 1, ACT_QMAX[bits]).astype(np.int64)


class FixedPointPolicy:
    """Callable with the same signature the closed-loop harness expects.

    weight_bits/act_bits select the datapath being modelled. The defaults are
    the shipped v2/v3 hardware -- INT16 weights, Q8.8 INT16 activations -- and
    that path is untouched by the mixed-precision additions.

    act_bits=8 stores each inter-layer activation as INT8 with a per-layer
    fractional count (act_fracs, calibrated during QAT). ELU still evaluates on
    the Q8.8 table, because that is what the SFU ROM holds; the value is shifted
    into Q8.8, looked up, and shifted back. In hardware that is one extra shift
    in a requantizer that already has a barrel shifter, not a new table.
    """

    def __init__(self, onnx_path: Path, weight_bits: int = 16,
                 act_bits: int = 16, act_fracs: list | None = None):
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.act_fracs = act_fracs
        model = onnx.load(onnx_path)
        init = {i.name: to_array(i) for i in model.graph.initializer}
        self.layers = []
        for node in model.graph.node:
            if node.op_type != "Gemm":
                continue
            w = init[node.input[1]]
            attrs = {a.name: get_attribute_value(a) for a in node.attribute}
            if int(attrs.get("transB", 0)) == 0:
                w = w.T
            b = np.asarray(init[node.input[2]], np.float64).reshape(-1)
            frac = _weight_frac(np.asarray(w, np.float64), weight_bits)
            # The bias is kept in float: it quantizes on the OUTPUT activation
            # grid, which is only known per layer once act_fracs is applied.
            self.layers.append((_quantize_weight(w, frac, weight_bits), b, frac))
        self.saturations = 0
        self.samples = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ab = self.act_bits
        # act_fracs[i] is the format of layer i's OUTPUT; index -1 is the input.
        fr = self.act_fracs if self.act_fracs else [ACT_FRAC] * (len(self.layers) + 1)
        f_in = fr[0]
        x = np.clip(_round_away(np.asarray(obs, np.float64) * (1 << f_in)),
                    -ACT_QMAX[ab] - 1, ACT_QMAX[ab]).astype(np.int64)
        for i, (wq, bq, frac) in enumerate(self.layers):
            f_out = fr[i + 1]
            acc = wq @ x
            # shift = frac_in + frac_w - frac_out, exactly the 6-bit field the
            # GEMM descriptor already encodes.
            shift = f_in + frac - f_out
            raw = _round_shift(acc, shift) if shift >= 0 else acc << (-shift)
            raw = raw + _quantize(bq, f_out)
            self.saturations += int(np.sum((raw > ACT_QMAX[ab]) | (raw < -ACT_QMAX[ab] - 1)))
            self.samples += int(raw.size)
            x = _sat_act(raw, ab)
            if i < len(self.layers) - 1:
                if f_out == ACT_FRAC:
                    x = _sat_act(_elu_q8_8(x), ab)
                else:
                    # Into Q8.8 for the table, then back to this layer's format.
                    q88 = _sat16(_round_shift(x << 8, f_out) if f_out <= 8
                                 else _round_shift(x, f_out - ACT_FRAC))
                    e = _elu_q8_8(q88)
                    x = _sat_act(_round_shift(e << f_out, ACT_FRAC) if f_out <= 8
                                 else e << (f_out - ACT_FRAC), ab)
            f_in = f_out
        return (x.astype(np.float64) / (1 << f_in)).astype(np.float32)


def main() -> int:
    import argparse, json, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
    from locomotion_eval import evaluate
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=Path)
    ap.add_argument("--robot", required=True, choices=["go2", "g1"])
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--keep-frames", type=int, default=None)
    ap.add_argument("--weight-bits", type=int, default=16, choices=[8, 16])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    pol = FixedPointPolicy(args.policy, weight_bits=args.weight_bits)
    m = evaluate(pol, args.robot, seconds=args.seconds, seeds=args.seeds,
                 keep_frames=args.keep_frames)
    m["saturation_pct"] = 100.0 * pol.saturations / max(pol.samples, 1)
    print(json.dumps({"policy": str(args.policy), **m}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
