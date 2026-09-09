#!/usr/bin/env python3
"""goal.md section 10: is the torch fake-quant forward the integer pipeline?

Compares three things on the same inputs:
  torch  QuantActorCriticSequence.act_inference  (float64 integer emulation)
  numpy  FudanFixedPolicy                        (int64, the arbiter)
  fp32   onnxruntime                             (the unquantized policy)

Reported per goal.md: exact-match rate, max/mean LSB error, layer by layer.
Anything short of bit-identical between the first two is a bug in this
package, not a quantization effect, and is printed as such.
"""
from __future__ import annotations

import argparse
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hwq.fixed_ref import ACT_FRAC
from hwq.quant_policy import (QuantActorCriticSequence, build_from_onnx,
                              build_from_checkpoint)
from hwq.torch_hw import QuantConfig

LSB = 1.0 / (1 << ACT_FRAC)


def sample_obs(n: int, seed: int = 0):
    """Observations spanning the ranges the env actually produces.

    Scaled per wheel_legged_gym: ang_vel x0.25, gravity unit vector, commands
    scaled, dof_pos deltas, dof_vel x0.05, previous action. Deliberately wider
    than nominal so saturation behaviour is exercised.
    """
    rng = np.random.default_rng(seed)
    o = np.zeros((n, 25), np.float64)
    o[:, 0:3] = rng.normal(0, 0.5, (n, 3))            # base ang vel * 0.25
    g = rng.normal(0, 1, (n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    o[:, 3:6] = g                                      # projected gravity
    o[:, 6:9] = rng.uniform(-2, 2, (n, 3))             # commands
    o[:, 9:13] = rng.normal(0, 0.4, (n, 4))            # dof pos deltas
    o[:, 13:19] = rng.normal(0, 1.5, (n, 6))           # dof vel * 0.05
    o[:, 19:25] = rng.normal(0, 1.0, (n, 6))           # previous action
    h = np.repeat(o[:, None, :], 5, axis=1)
    h += rng.normal(0, 0.05, h.shape)                  # history: nearby states
    return np.clip(o, -100, 100), np.clip(h.reshape(n, 125), -100, 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", type=Path)
    src.add_argument("--checkpoint", type=Path,
                     help="a SOLID_FP32-style .pt from the frozen Codex infra")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--act-bits", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--act-fracs", type=Path, default=None,
                    help="calibrated per-layer fracs; required below A16, and "
                         "the QAT policy ships at W8A8 with these")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    _kw = {}
    if args.act_fracs:
        _d = json.loads(args.act_fracs.read_text())
        _kw["act_fracs"] = _d.get("act_fracs", _d)
        if "obs" in _kw["act_fracs"]:
            _kw["obs_frac"] = int(_kw["act_fracs"]["obs"])
    elif args.act_bits < 16:
        raise SystemExit(f"A{args.act_bits} needs --act-fracs")
    cfg = QuantConfig(weight_bits=args.weight_bits, act_bits=args.act_bits, **_kw)
    net = (build_from_onnx(args.onnx, cfg, device=args.device) if args.onnx
           else build_from_checkpoint(args.checkpoint, cfg, device=args.device,
                                      strict_critic=False))
    ref = net.to_fixed_ref()

    obs, hist = sample_obs(args.n)
    # Both paths must see the SAME numbers. The policy is fed float32 on the
    # robot, so float32 is the input of record; handing numpy the float64
    # originals would let values within half an LSB of a rounding boundary
    # land on different integers and look like a datapath mismatch.
    obs = obs.astype(np.float32).astype(np.float64)
    hist = hist.astype(np.float32).astype(np.float64)
    with torch.no_grad():
        t_act, t_lat = net.act_inference(
            torch.as_tensor(obs, dtype=torch.float32, device=args.device),
            torch.as_tensor(hist, dtype=torch.float32, device=args.device))
    t_act = t_act.cpu().numpy().astype(np.float64)
    t_lat = t_lat.cpu().numpy().astype(np.float64)

    n_act = t_act.shape[1]
    r_act = np.zeros_like(t_act)
    r_lat = np.zeros_like(t_lat)
    traces = []
    for i in range(args.n):
        tr = {}
        a = ref(obs[i], hist[i], trace=tr)
        r_act[i] = a
        r_lat[i] = tr["enc2"] * (2.0 ** -cfg.frac_of("enc2"))
        traces.append(tr)

    def cmp(name, t, r, lsb=None):
        # Each tensor lives on its OWN layer's grid. Normalising every one by a
        # fixed Q8.8 LSB inflated differences by 2**(8-frac) once per-layer
        # calibration existed, and reported mismatches where there were none.
        d = np.abs(t - r) / (LSB if lsb is None else lsb)
        return {"tensor": name, "exact_pct": 100.0 * float(np.mean(d < 1e-9)),
                "max_lsb": float(d.max()), "mean_lsb": float(d.mean())}

    rows = [cmp("latent", t_lat, r_lat), cmp("action", t_act, r_act)]

    # Layer by layer: hook the torch modules and compare integer codes.
    caught: dict = {}
    hooks = []
    for key, short in net.LAYER_NAMES.items():
        mod = dict(net.named_modules())[key]
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, s=short: caught.__setitem__(s, o.detach().cpu().numpy())))
    # ELU outputs are what the next GEMM consumes; catch those too.
    for key in ("encoder.1", "encoder.3", "actor.1", "actor.3", "actor.5"):
        mod = dict(net.named_modules())[key]
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, s=key: caught.__setitem__(s, o.detach().cpu().numpy())))
    with torch.no_grad():
        net.act_inference(torch.as_tensor(obs, dtype=torch.float32, device=args.device),
                          torch.as_tensor(hist, dtype=torch.float32, device=args.device))
    for h in hooks:
        h.remove()

    # The numpy trace stores post-ELU values under the GEMM's own name, so the
    # comparable torch tensor is the ELU output where one follows.
    post = {"enc0": "encoder.1", "enc1": "encoder.3", "enc2": "enc2",
            "act0": "actor.1", "act1": "actor.3", "act2": "actor.5", "act3": "act3"}
    for short, tkey in post.items():
        t = np.asarray(caught[tkey], np.float64)
        lsb = 2.0 ** -cfg.frac_of(short)
        r = np.stack([tr[short] for tr in traces]) * lsb
        rows.append(cmp(short, t, r, lsb))

    # FP32 comparison, for the quantization damage this is about to measure.
    fp32 = None
    try:
        if not args.onnx:
            raise ImportError("no ONNX source; skipping onnxruntime cross-check")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        fp32 = sess.run(None, {"obs": obs.astype(np.float32),
                               "obs_history": hist.astype(np.float32)})[0]
    except Exception as exc:                                  # noqa: BLE001
        print(f"(onnxruntime unavailable: {exc})")

    w = 10
    print(f"torch fake-quant vs numpy integer reference   n={args.n}  "
          f"W{args.weight_bits}A{args.act_bits}  device={args.device}")
    print(f"{'tensor':<10} {'exact %':>9} {'max LSB':>9} {'mean LSB':>10}")
    for r in rows:
        print(f"{r['tensor']:<10} {r['exact_pct']:>9.4f} {r['max_lsb']:>9.4f} "
              f"{r['mean_lsb']:>10.6f}")
    bit_exact = all(r["exact_pct"] == 100.0 for r in rows)
    print(f"\nBIT_EXACT: {'PASS' if bit_exact else 'FAIL'}")

    out = {"n": args.n, "weight_bits": args.weight_bits,
           "act_bits": args.act_bits, "bit_exact": bit_exact, "layers": rows}
    if fp32 is not None:
        err = np.abs(r_act - fp32.astype(np.float64))
        out["vs_fp32"] = {"max_abs": float(err.max()), "rmse": float(np.sqrt((err ** 2).mean())),
                          "max_lsb": float(err.max() / LSB),
                          "mean_lsb": float(err.mean() / LSB)}
        print(f"\nquantized vs FP32 action: RMSE {out['vs_fp32']['rmse']:.6f}  "
              f"max {out['vs_fp32']['max_abs']:.6f} ({out['vs_fp32']['max_lsb']:.1f} LSB)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
    return 0 if bit_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
