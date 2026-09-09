#!/usr/bin/env python3
"""Per-tensor quantization statistics and layer sensitivity.

goal.md section 12 (saturation and error statistics for every deployed tensor)
and the second half of section 7 (which layers PTQ hurts most).

Runs the same observations through the FP32 network and the quantized network,
capturing every intermediate, and reports for each:

  min / max / mean / std        does the tensor fit the +-128 Q8.8 window
  saturated %                   how often it does not
  quant error RMS / max         in LSB, so it can be read against the 1/256 grid
  weight headroom               how many of the 16 integer bits the tensor uses

The weight column matters because the format is per-tensor power-of-two: one
outlier weight sets the resolution of every other weight in the layer, and the
"bits used" number says how much of the range that outlier is wasting.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.fixed_ref import ACT_FRAC, QMAX, choose_weight_frac  # noqa: E402
from roboaccel_quant.quant_policy import (QuantActorCriticSequence, build_from_onnx,
                              build_from_checkpoint)  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

LSB = 1.0 / (1 << ACT_FRAC)
TAPS = ["encoder.0", "encoder.1", "encoder.2", "encoder.3", "encoder.4",
        "actor.0", "actor.1", "actor.2", "actor.3", "actor.4", "actor.5", "actor.6"]
PRETTY = {"encoder.0": "enc0 pre", "encoder.1": "enc0 ELU", "encoder.2": "enc1 pre",
          "encoder.3": "enc1 ELU", "encoder.4": "latent", "actor.0": "act0 pre",
          "actor.1": "act0 ELU", "actor.2": "act1 pre", "actor.3": "act1 ELU",
          "actor.4": "act2 pre", "actor.5": "act2 ELU", "actor.6": "action"}
# Each tap sits on the output grid of a named layer, so its LSB is that
# layer's frac -- NOT the module-level Q8.8 default. Reporting saturation and
# LSB-normalised error against a single fixed LSB was wrong whenever
# --act-fracs was supplied: it showed 80-96% saturation for a calibrated INT8
# run that in fact clips nothing.
FRAC_KEY = {"encoder.0": "enc0", "encoder.1": "enc0", "encoder.2": "enc1",
            "encoder.3": "enc1", "encoder.4": "enc2", "actor.0": "act0",
            "actor.1": "act0", "actor.2": "act1", "actor.3": "act1",
            "actor.4": "act2", "actor.5": "act2", "actor.6": "act3"}
PRESETS = {"W16A16": QuantConfig(16, 16), "W8A16": QuantConfig(8, 16),
           "W8A8": QuantConfig(8, 8), "W4A8": QuantConfig(4, 8)}
OFF = dict(quant_weights=False, quant_obs=False, quant_hidden=False, quant_output=False)


def capture(net, obs, hist):
    got = {}
    hooks = [dict(net.named_modules())[k].register_forward_hook(
        lambda m, i, o, s=k: got.__setitem__(s, o.detach().cpu().numpy().astype(np.float64)))
        for k in TAPS]
    with torch.no_grad():
        net.act_inference(obs, hist)
    for h in hooks:
        h.remove()
    return got


def load_net(args, cfg):
    """One loader for both sources; the checkpoint path handles the Codex
    log_std rename and the shape assertions in build_from_checkpoint."""
    if args.onnx:
        return build_from_onnx(args.onnx, cfg)
    return build_from_checkpoint(args.checkpoint, cfg, strict_critic=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--obs", type=Path, required=True, help="npz from eval --dump-obs")
    ap.add_argument("--quant", default="W16A16", choices=sorted(PRESETS))
    ap.add_argument("--act-fracs", type=Path, default=None,
                    help="calibrated per-layer fracs; without this an A8 run\n                         measures the default Q8.8 grid, where everything\n                         saturates, and says nothing about the real config")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    d = np.load(args.obs)
    obs = torch.as_tensor(d["obs"], dtype=torch.float32)
    hist = torch.as_tensor(d["obs_history"], dtype=torch.float32)
    print(f"{len(obs)} observations from {args.obs.name}")

    qcfg = PRESETS[args.quant]

    if args.act_fracs:

        _d = json.loads(args.act_fracs.read_text())

        _f = _d.get('act_fracs', _d)

        qcfg = replace(qcfg, act_fracs=_f,

                       obs_frac=int(_f.get('obs', qcfg.obs_frac)))

    elif qcfg.act_bits < 16:

        raise SystemExit('A%d needs --act-fracs' % qcfg.act_bits)

    q_net = load_net(args, qcfg)
    f_net = load_net(args, QuantConfig(**{**PRESETS[args.quant].__dict__, **OFF}))
    qa = capture(q_net, obs, hist)
    fa = capture(f_net, obs, hist)

    rows = []
    for k in TAPS:
        f, q = fa[k], qa[k]
        lsb_k = 2.0 ** -qcfg.frac_of(FRAC_KEY[k])
        qmax_k = QMAX[qcfg.act_bits] * lsb_k
        signed = (q - f) / lsb_k        # signed: exposes systematic offset
        err = np.abs(signed)
        rows.append({
            "tensor": PRETTY[k], "module": k, "lsb": lsb_k, "grid_max": qmax_k,
            "min": float(f.min()), "max": float(f.max()),
            "mean": float(f.mean()), "std": float(f.std()),
            "sat_pct": 100.0 * float(np.mean(np.abs(f) > qmax_k)),
            "sat_count": int(np.sum(np.abs(f) > qmax_k)),
            "n": int(f.size),
            "err_mean_lsb": float(signed.mean()),
            "err_rms_lsb": float(np.sqrt((err ** 2).mean())),
            "err_max_lsb": float(err.max()),
        })

    src = "calibrated per-layer" if qcfg.act_fracs else "fixed Q8.8"
    print(f"\nactivations   ({src} grids)   arithmetic={args.quant}")
    print(f"{'tensor':<10} {'min':>9} {'max':>9} {'std':>8} {'LSB':>7} {'grid±':>8} "
          f"{'sat %':>8} {'errMEAN':>8} {'errRMS':>8} {'errMax':>8}   (err in that layer's LSB)")
    for r in rows:
        flag = (f"  <== {r['sat_count']} of {r['n']:,} clipped"
                if r["sat_count"] else "")
        print(f"{r['tensor']:<10} {r['min']:>9.2f} {r['max']:>9.2f} {r['std']:>8.2f} "
              f"{r['lsb']:>7.3f} {r['grid_max']:>8.2f} "
              f"{r['sat_pct']:>8.4f} {r['err_mean_lsb']:>+8.2f} {r['err_rms_lsb']:>8.2f} "
              f"{r['err_max_lsb']:>8.1f}{flag}")

    wrows = []
    wb = PRESETS[args.quant].weight_bits
    for mod, short in q_net.LAYER_NAMES.items():
        w = dict(q_net.named_modules())[mod].weight.detach().cpu().numpy().astype(np.float64)
        f_w = choose_weight_frac(w, wb)
        wq = np.clip(np.round(w * (1 << f_w)), -QMAX[wb] - 1, QMAX[wb])
        used = float(np.log2(max(np.abs(wq).max(), 1.0))) + 1.0
        rel = np.abs(wq / (1 << f_w) - w)
        wrows.append({"layer": short, "absmax": float(np.abs(w).max()), "f_w": f_w,
                      "bits_used": used, "wasted_bits": float(np.log2(QMAX[wb] + 1) + 1 - used),
                      "p999_over_max": float(np.percentile(np.abs(w), 99.9) / np.abs(w).max()),
                      "err_rms": float(np.sqrt((rel ** 2).mean()))})
    print(f"\nweights (INT{wb}, per-tensor power-of-two scale)")
    print(f"{'layer':<6} {'|w|max':>9} {'f_w':>4} {'bits used':>10} {'wasted':>7} "
          f"{'p99.9/max':>10}   (outlier cost)")
    for r in wrows:
        print(f"{r['layer']:<6} {r['absmax']:>9.4f} {r['f_w']:>4d} {r['bits_used']:>10.1f} "
              f"{r['wasted_bits']:>7.1f} {r['p999_over_max']:>10.3f}")

    out = {"source": str(args.checkpoint or args.onnx), "quant": args.quant,
           "n_obs": int(len(obs)), "activations": rows, "weights": wrows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
