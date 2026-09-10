#!/usr/bin/env python3
"""Pick per-layer activation fractional bits from real activation ranges.

Needed only when act_bits < 16. The shipped hardware runs a fixed Q8.8 grid
whose +-128 window is never troubled by this network (measured: 0.00 %
saturation, largest activation 22.3), so W*A16 needs no calibration and gets
none -- calibrating it would silently change the format under study.

Two observers. `--percentile` off gives the min-max observer, which is the
conservative default: no sample is ever clipped. `--percentile P` clips at the
P-th percentile of |activation| instead.

Min-max is the wrong default below 16 bits, and goal 6 measured how wrong.
Because the requantizer is a shifter, the scale itself is a power of two, so a
single outlier that pushes a tensor just past a binade boundary costs a whole
bit for every other sample. Under the shipped min-max fracs no tensor uses even
40% of INT8's range and one uses 11%. Giving up 0.9% of `enc0`'s peak magnitude
buys a full bit back.

That trade is worth 4.8x in post-training error and takes closed-loop success
from 0.285 to 0.820 across three policies -- with no retraining at all. See
docs/qat_failure/goal6_root_cause.md section 5.7.

Saturation is reported per tensor rather than assumed, because the point of
clipping is to spend a measured amount of it.

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
                    help="margin over the observed range, for unseen states")
    ap.add_argument("--percentile", type=float, default=None,
                    help="clip at this percentile of |activation| instead of "
                         "the peak, e.g. 99.9. Off by default so existing "
                         "calibrations reproduce exactly.")
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

    raw = {k: np.abs(v).ravel() for k, v in got.items()}
    raw["obs"] = np.concatenate([np.abs(d["obs"]).ravel(),
                                 np.abs(d["obs_history"]).ravel()])
    peaks = {k: float(v.max()) for k, v in raw.items()}
    if args.percentile is not None:
        limits = {k: float(np.percentile(v, args.percentile)) for k, v in raw.items()}
    else:
        limits = dict(peaks)

    fracs = {k: frac_for(v, args.act_bits, args.headroom) for k, v in limits.items()}
    # CONCAT constraint: one grid for obs and the latent, so take the tighter.
    shared = min(fracs["obs"], fracs["enc2"])
    fracs["obs"] = fracs["enc2"] = shared

    span = (QMAX[args.act_bits] + 1)
    # What fraction of real samples the chosen grid actually clips. Clipping is
    # the price being paid for resolution, so it is measured, not inferred from
    # the peak ratio.
    saturation, levels = {}, {}
    for k, v in raw.items():
        rng = span / 2.0 ** fracs[k]
        saturation[k] = float((v > rng).mean())
        levels[k] = int(min(peaks[k], rng) * 2.0 ** fracs[k])

    obs_mode = ("min-max" if args.percentile is None
                else f"p{args.percentile:g} clipping")
    print(f"INT{args.act_bits} activations, {obs_mode}, headroom x{args.headroom}")
    print(f"{'tensor':<8} {'peak':>9} {'limit':>9} {'frac':>5} {'range':>9} "
          f"{'LSB':>8} {'clipped':>9} {'levels':>7}  {'note':<22}")
    for k in ("obs", "enc0", "enc1", "enc2", "act0", "act1", "act2", "act3"):
        f = fracs[k]
        note = "shared grid (CONCAT)" if k in ("obs", "enc2") else ""
        print(f"{k:<8} {peaks[k]:>9.3f} {limits[k]:>9.3f} {f:>5d} "
              f"{span / 2.0 ** f:>9.3f} {1.0 / 2.0 ** f:>8.5f} "
              f"{saturation[k]:>8.3%} {levels[k]:>7d}  {note:<22}")
    print(f"\n{'':<8} {'':>9} {'':>9} {'':>5} {'':>9} {'':>8} "
          f"{'':>9} {'/' + str(span - 1):>7}  of INT{args.act_bits}'s range")

    out = {"act_bits": args.act_bits, "headroom": args.headroom,
           "percentile": args.percentile, "observer": obs_mode,
           "source": str(args.checkpoint or args.onnx), "obs_file": str(args.obs),
           "peaks": peaks, "clip_limits": limits,
           "saturation_fraction": saturation, "levels_used": levels,
           "act_fracs": fracs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
