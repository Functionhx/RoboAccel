#!/usr/bin/env python3
"""Fold codex's base+residual actor into the single-actor form the PL runs.

goal.md §5 says to extend the existing datapath representation rather than
build a new inference framework. Codex's promoted controllers are
`action = base(x) + residual(x)`, two parallel actor branches, and the PL has
GEMM / ELU / NORM / CONCAT but no element-wise ADD.

Both branches end in a GEMM producing the action vector, from 32 and 64 inputs
respectively, so their sum is exactly one GEMM over the concatenated
penultimate activations:

    W_b @ a + c_b  +  W_r @ b + c_r  ==  [W_b | W_r] @ [a ; b] + (c_b + c_r)

The missing ADD becomes a CONCAT the hardware already has. This rewrites a
checkpoint into that form and verifies the rewrite numerically rather than
trusting the algebra.

The constraint it introduces: PL CONCAT cannot requantize, so `base2_out` and
`residual1_out` must share an activation grid. Calibrating the branches
independently would silently violate that, so it is asserted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

import torch


def fuse(sd: dict) -> tuple[dict, dict]:
    if "actor.residual.0.weight" not in sd:
        raise SystemExit("checkpoint has no actor.residual branch; nothing to fuse")
    b_last = max(int(k.split(".")[2]) for k in sd if k.startswith("actor.base.")
                 and k.endswith(".weight"))
    r_last = max(int(k.split(".")[2]) for k in sd if k.startswith("actor.residual.")
                 and k.endswith(".weight"))
    Wb, cb = sd[f"actor.base.{b_last}.weight"], sd[f"actor.base.{b_last}.bias"]
    Wr, cr = sd[f"actor.residual.{r_last}.weight"], sd[f"actor.residual.{r_last}.bias"]
    if Wb.shape[0] != Wr.shape[0]:
        raise SystemExit(f"branches disagree on action width: {Wb.shape[0]} vs {Wr.shape[0]}")

    out = {k: v for k, v in sd.items()
           if not k.startswith("actor.base.") and not k.startswith("actor.residual.")}
    # base trunk keeps its indices; residual trunk is appended after it.
    for k, v in sd.items():
        if k.startswith("actor.base.") and int(k.split(".")[2]) != b_last:
            out["actor." + k.split("actor.base.")[1]] = v
    out["actor_residual_trunk"] = {k.split("actor.residual.")[1]: v
                                   for k, v in sd.items()
                                   if k.startswith("actor.residual.")
                                   and int(k.split(".")[2]) != r_last}
    out["actor_fused_out.weight"] = torch.cat([Wb, Wr], dim=1)
    out["actor_fused_out.bias"] = cb + cr
    meta = {"base_last": b_last, "residual_last": r_last,
            "fused_in": int(Wb.shape[1] + Wr.shape[1]),
            "base_in": int(Wb.shape[1]), "residual_in": int(Wr.shape[1]),
            "actions": int(Wb.shape[0])}
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--base-frac", type=int, default=None,
                    help="calibrated output frac of the base branch's penultimate layer")
    ap.add_argument("--residual-frac", type=int, default=None,
                    help="same for the residual branch; must match --base-frac")
    a = ap.parse_args()

    ck = torch.load(str(a.checkpoint), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    fused, meta = fuse(sd)

    # Verify on the real weights: fused GEMM over the concatenation must equal
    # the two separate GEMMs summed.
    torch.manual_seed(0)
    x_b = torch.randn(a.n, meta["base_in"])
    x_r = torch.randn(a.n, meta["residual_in"])
    Wb = sd[f"actor.base.{meta['base_last']}.weight"]
    cb = sd[f"actor.base.{meta['base_last']}.bias"]
    Wr = sd[f"actor.residual.{meta['residual_last']}.weight"]
    cr = sd[f"actor.residual.{meta['residual_last']}.bias"]
    sep = (x_b @ Wb.T + cb) + (x_r @ Wr.T + cr)
    fus = torch.cat([x_b, x_r], 1) @ fused["actor_fused_out.weight"].T \
        + fused["actor_fused_out.bias"]
    err = (fus - sep).abs().max().item()
    print(f"fusion check over {a.n} inputs: max|fused - separate| = {err:.3e}")
    if err > 1e-4:
        raise SystemExit("fusion does not reproduce the residual sum")
    print(f"  base {meta['base_in']}->{meta['actions']}, "
          f"residual {meta['residual_in']}->{meta['actions']}, "
          f"fused {meta['fused_in']}->{meta['actions']}")
    print("  CONSTRAINT: PL CONCAT cannot requantize, so frac(base_penultimate) "
          "must equal frac(residual_penultimate)")
    if a.base_frac is not None and a.residual_frac is not None:
        from roboaccel_quant.fixed_ref import FudanFixedPolicy
        FudanFixedPolicy.check_residual_concat(a.base_frac, a.residual_frac)
        print(f"  grid check: base frac {a.base_frac} == residual frac "
              f"{a.residual_frac} -- fusion is representable")

    if a.out:
        ck["model_state_dict"] = fused
        ck["fusion_meta"] = meta
        a.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, str(a.out))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
