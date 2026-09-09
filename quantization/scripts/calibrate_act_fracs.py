#!/usr/bin/env python3
"""Pick per-layer activation fractional bits from real activation ranges.

Needed only when act_bits < 16. The shipped hardware runs a fixed Q8.8 grid
whose +-128 window is never troubled by this network (measured: 0.00 %
saturation, largest activation 22.3), so W*A16 needs no calibration and gets
none -- calibrating it would silently change the format under study.

Min-max observer on the FP32 network, which is the standard PTQ starting
point: quantization perturbs the activations it is measuring, so the fracs are
an initialisation that QAT then adapts around, not a fixed point.

Two hardware constraints are enforced rather than assumed:
  * scales are powers of two (the requantizer is a shifter, not a multiplier);
  * frac(latent) must equal frac(obs), because the PL CONCAT moves data
    without requantizing and the actor's first GEMM reads both halves.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.fixed_ref import QMAX  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

sys.path.insert(0, str(QUANT_ROOT / "scripts"))
from quant_diagnostics import OFF, load_net  # noqa: E402

# GEMM output -> the module whose output sits on that layer's grid.
TAP = {"enc0": "encoder.0", "enc1": "encoder.2", "enc2": "encoder.4",
       "act0": "actor.0", "act1": "actor.2", "act2": "actor.4", "act3": "actor.6"}


def frac_for(peak: float, bits: int, headroom: float) -> int:
    """Largest power-of-two frac whose range still covers peak*headroom."""
    peak = max(peak * headroom, 1e-6)
    return int(np.clip(np.floor(np.log2((QMAX[bits] + 1) / peak)), -4, 15))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--obs", type=Path, required=True)
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--headroom", type=float, default=1.25,
                    help="margin over the observed peak, for unseen states")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.obs)
    obs = torch.as_tensor(d["obs"], dtype=torch.float32)
    hist = torch.as_tensor(d["obs_history"], dtype=torch.float32)

    net = load_net(args, QuantConfig(**OFF))
    got: dict = {}
    hooks = [dict(net.named_modules())[m].register_forward_hook(
        lambda mo, i, o, s=k: got.__setitem__(s, o.detach().cpu().numpy()))
        for k, m in TAP.items()]
    with torch.no_grad():
        net.act_inference(obs, hist)
    for h in hooks:
        h.remove()

    peaks = {k: float(np.abs(v).max()) for k, v in got.items()}
    peaks["obs"] = float(max(np.abs(d["obs"]).max(), np.abs(d["obs_history"]).max()))

    fracs = {k: frac_for(v, args.act_bits, args.headroom) for k, v in peaks.items()}
    # CONCAT constraint: one grid for obs and the latent, so take the tighter.
    shared = min(fracs["obs"], fracs["enc2"])
    fracs["obs"] = fracs["enc2"] = shared

    span = (QMAX[args.act_bits] + 1)
    print(f"INT{args.act_bits} activations, headroom x{args.headroom}")
    print(f"{'tensor':<8} {'peak':>9} {'frac':>5} {'range':>12} {'LSB':>9}  {'note':<28}")
    for k in ("obs", "enc0", "enc1", "enc2", "act0", "act1", "act2", "act3"):
        f = fracs[k]
        note = ""
        if peaks[k] > span / (2.0 ** f):
            note = "SATURATES"
        elif k in ("obs", "enc2"):
            note = "shared grid (CONCAT)"
        print(f"{k:<8} {peaks[k]:>9.3f} {f:>5d} {span / 2.0 ** f:>12.3f} "
              f"{1.0 / 2.0 ** f:>9.5f}  {note:<28}")

    out = {"act_bits": args.act_bits, "headroom": args.headroom,
           "source": str(args.checkpoint or args.onnx), "obs_file": str(args.obs),
           "peaks": peaks, "act_fracs": fracs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
