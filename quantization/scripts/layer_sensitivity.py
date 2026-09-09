#!/usr/bin/env python3
"""goal.md §10 -- attribute the W8A8 collapse to individual layers.

The W16A16 diagnostics predicted, before any low-precision run existed, that
the latent GEMM `enc2` would be the weak point: it wastes 3.4 of its 16 weight
bits (|w|max 0.1831 against a frac capped at 14, p99.9/max 0.912 so uniformly
small rather than outlier-driven), and at INT8 the latent receives about six
levels within one sigma.

This tests that prediction directly. Each layer is quantized ALONE, every
other layer left in FP32, and the resulting action error measured against the
fully-FP32 policy. A layer that dominates the collapse should dominate here.

Open-loop on stored on-policy observations: it ranks the layers by the
perturbation they inject per step, which is not the same as their closed-loop
cost. A layer feeding a fast feedback path can matter more than its
open-loop error suggests.
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

from hwq.quant_policy import build_from_checkpoint
from hwq.torch_hw import QuantConfig

LAYERS = ["obs", "enc0", "enc1", "enc2", "act0", "act1", "act2", "act3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--act-fracs", type=Path, default=None)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    base_kw = {}
    if a.act_fracs:
        d = json.loads(a.act_fracs.read_text())
        base_kw["act_fracs"] = d.get("act_fracs", d)
        if "obs" in base_kw["act_fracs"]:
            base_kw["obs_frac"] = base_kw["act_fracs"]["obs"]

    d = np.load(a.obs)
    o = torch.as_tensor(d["obs"][:a.n], dtype=torch.float32)
    h = torch.as_tensor(d["obs_history"][:a.n], dtype=torch.float32)

    off = QuantConfig(quant_weights=False, quant_obs=False,
                      quant_hidden=False, quant_output=False)
    with torch.no_grad():
        ref = build_from_checkpoint(a.checkpoint, off,
                                    strict_critic=False).act_inference(o, h)[0]

    rows = []
    # ALL, then each layer alone, then each layer EXCLUDED. One-at-a-time
    # over-credits layers whose error partly cancels downstream; leave-one-out
    # over-credits layers other layers can compensate for. Agreeing rankings
    # are trustworthy, disagreeing ones are the interesting case.
    combos = ([None] + [[L] for L in LAYERS]
              + [[x for x in LAYERS if x != L] for L in LAYERS])
    for sel in combos:
        cfg = QuantConfig(weight_bits=a.weight_bits, act_bits=a.act_bits,
                          only_layers=None if sel is None else frozenset(sel),
                          **base_kw)
        with torch.no_grad():
            act = build_from_checkpoint(a.checkpoint, cfg,
                                        strict_critic=False).act_inference(o, h)[0]
        e = (act - ref).abs()
        if sel is None:
            label = "ALL"
        elif len(sel) == 1:
            label = sel[0]
        else:
            label = "-" + [x for x in LAYERS if x not in sel][0]
        rows.append({"layers": label,
                     "rmse": float(((act - ref) ** 2).mean().sqrt()),
                     "max": float(e.max()), "bias": float((act - ref).mean())})

    allr = rows[0]["rmse"]
    solo = {r["layers"]: r for r in rows[1:1 + len(LAYERS)]}
    loo = {r["layers"][1:]: r for r in rows[1 + len(LAYERS):]}
    print(f"W{a.weight_bits}A{a.act_bits}, n={len(o)}; action error vs the fully-FP32 policy")
    print(f"ALL layers quantized: RMSE {allr:.5f}\n")
    print(f"{'layer':8s} {'alone':>10s} {'share':>8s} {'all-but':>10s} "
          f"{'drop vs ALL':>12s}")
    print("-" * 52)
    order = sorted(LAYERS, key=lambda L: -solo[L]["rmse"])
    for L in order:
        s_, l_ = solo[L]["rmse"], loo[L]["rmse"]
        print(f"{L:8s} {s_:>10.5f} {s_/allr*100:>7.1f}% {l_:>10.5f} "
              f"{(allr - l_)/allr*100:>11.1f}%")
    top_solo = order[0]
    top_loo = max(LAYERS, key=lambda L: allr - loo[L]["rmse"])
    print(f"\nalone      -> {top_solo}")
    print(f"leave-out  -> {top_loo}")
    print("agree" if top_solo == top_loo else
          "DISAGREE: errors interact; neither ranking alone is attribution")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
