#!/usr/bin/env python3
"""goal.md §20 -- what does the hardware's rounding rule actually cost?

`rl_round_shift_sat16.sv` adds half an LSB *away from zero* then arithmetic
shifts. Combined with the two's-complement shift that follows, negative values
land one LSB low of true round-half-away-from-zero 99.6% of the time
(measured in scripts/verify_against_rtl.py). The W16A16 diagnostics show this
as a systematic negative errMEAN on every tensor, -0.12 to -0.75 LSB.

This is a DC bias, not noise, so the question is whether it matters. Here the
integer reference is run twice over the same on-policy observations -- once
with the hardware rule, once with exact round-half-away-from-zero -- and the
action difference is measured.

This is an OPEN-LOOP measurement. It bounds the per-step perturbation; it does
not tell you what the closed loop does with it, because a small persistent
bias can either wash out or accumulate depending on the controller. The
closed-loop version needs a GPU rollout and is noted as outstanding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QAT_ROOT))

import numpy as np
import torch

import hwq.fixed_ref as fr
from hwq.quant_policy import build_from_checkpoint
from hwq.torch_hw import QuantConfig


def ideal_round_shift(v, shift):
    """Exact round-half-away-from-zero, with no shift-direction asymmetry."""
    v = np.asarray(v, dtype=np.int64)
    if shift == 0:
        return v
    if shift < 0:
        return v << (-shift)
    scale = np.int64(1) << shift
    half = scale // 2
    out = np.where(v >= 0, (v + half) // scale, -((-v + half) // scale))
    return out.astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--act-bits", type=int, default=16)
    ap.add_argument("--act-fracs", type=Path, default=None)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    kw = {}
    if a.act_fracs:
        d = json.loads(a.act_fracs.read_text())
        kw["act_fracs"] = d.get("act_fracs", d)
        if "obs" in kw["act_fracs"]:
            kw["obs_frac"] = kw["act_fracs"]["obs"]
    cfg = QuantConfig(weight_bits=a.weight_bits, act_bits=a.act_bits, **kw)
    net = build_from_checkpoint(a.checkpoint, cfg, strict_critic=False)
    ref = net.to_fixed_ref()

    d = np.load(a.obs)
    o = d["obs"][:a.n].astype(np.float32).astype(np.float64)
    h = d["obs_history"][:a.n].astype(np.float32).astype(np.float64)

    hw = np.stack([ref(o[i], h[i]) for i in range(len(o))])
    saved, fr.round_shift = fr.round_shift, ideal_round_shift
    try:
        ideal = np.stack([ref(o[i], h[i]) for i in range(len(o))])
    finally:
        fr.round_shift = saved

    lsb = 2.0 ** -ref.act[-1].f_out
    diff = hw.astype(np.float64) - ideal.astype(np.float64)
    same = float((diff == 0).mean()) * 100

    print(f"W{a.weight_bits}A{a.act_bits}, n={len(o)} on-policy observations, "
          f"output LSB = {lsb:.6f}")
    print(f"  actions identical under both rules : {same:.2f}%")
    print(f"  bias  (mean signed) : {diff.mean():+.6f}  = {diff.mean()/lsb:+.3f} LSB")
    print(f"  spread (RMS)        : {np.sqrt((diff**2).mean()):.6f}  "
          f"= {np.sqrt((diff**2).mean())/lsb:.3f} LSB")
    print(f"  worst |difference|  : {np.abs(diff).max():.6f}  "
          f"= {np.abs(diff).max()/lsb:.3f} LSB")
    print(f"  per-action bias (LSB): "
          f"{np.array2string(diff.mean(0)/lsb, precision=3)}")

    if a.json:
        a.json.write_text(json.dumps({
            "n": len(o), "weight_bits": a.weight_bits, "act_bits": a.act_bits,
            "output_lsb": lsb, "identical_pct": same,
            "bias_lsb": float(diff.mean()/lsb),
            "rms_lsb": float(np.sqrt((diff**2).mean())/lsb),
            "max_lsb": float(np.abs(diff).max()/lsb),
            "per_action_bias_lsb": (diff.mean(0)/lsb).tolist(),
            "closed_loop": "not measured; open-loop bound only",
        }, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
