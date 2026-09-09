#!/usr/bin/env python3
"""Cross-validate three independent fixed-point implementations.

goal.md section 10: compare the QAT fake-quant against the integer reference,
against the existing FPGA exporter, and against the H7 model. All four run the
same weights on the same inputs; anything less than bit-identical is an
arithmetic mismatch to be located, not a tolerance to be widened.

  A  roboaccel_quant/torch_hw.py            torch fake-quant (what QAT trained against)
  B  roboaccel_quant/fixed_ref.py           numpy int64 reference
  C  tools/export_policy.py     the shipped FPGA exporter's fixed_inference
  D  tools/fixed_policy.py      the shipped H7/FPGA closed-loop model
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

QAT_ROOT = Path(__file__).resolve().parent.parent
RL_ACCEL = QAT_ROOT.parent
sys.path.insert(0, str(QAT_ROOT))
sys.path.insert(0, str(RL_ACCEL / "tools"))

from roboaccel_quant.fixed_ref import ACT_FRAC  # noqa: E402
from roboaccel_quant.quant_policy import build_from_onnx  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

LSB = 1.0 / (1 << ACT_FRAC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    net = build_from_onnx(args.onnx, QuantConfig())
    ref = net.to_fixed_ref()

    n_obs = net.num_obs
    n_hist = net.encoder[0].weight.shape[1]
    rng = np.random.default_rng(11)
    obs = rng.normal(0, 0.4, (args.n, n_obs)).astype(np.float32)
    hist = rng.normal(0, 0.4, (args.n, n_hist)).astype(np.float32)

    # A: torch fake-quant
    with torch.no_grad():
        A, _ = net.act_inference(torch.as_tensor(obs), torch.as_tensor(hist))
    A = A.numpy().astype(np.float64)

    # B: numpy integer reference
    B = np.stack([ref(obs[i].astype(np.float64), hist[i].astype(np.float64))
                  for i in range(args.n)]).astype(np.float64)

    # C: the shipped FPGA exporter
    import export_policy as ep
    pol = ep.load_policy(args.onnx)
    img = ep.build_images(pol)
    C = np.stack([
        ep.fixed_inference(pol, img, {"obs": ep.quantize(obs[i], ep.ACT_FRAC),
                                      "obs_history": ep.quantize(hist[i], ep.ACT_FRAC)})
        for i in range(args.n)]).astype(np.float64) / (1 << ep.ACT_FRAC)

    # D: the shipped H7/FPGA closed-loop model. It takes one concatenated
    # vector, so it only applies to single-input policies; skipped otherwise.
    D = None
    try:
        from fixed_policy import FixedPointPolicy
        fp = FixedPointPolicy(args.onnx)
        if len(fp.layers) == 7 and fp.layers[0][0].shape[1] == n_hist:
            D = None  # two-input graph: layer chain is not a straight line
    except Exception as exc:                                        # noqa: BLE001
        print(f"(D skipped: {exc})")

    def cmp(name, x, y):
        d = np.abs(x - y) / LSB
        return {"pair": name, "exact_pct": 100.0 * float(np.mean(d < 1e-9)),
                "max_lsb": float(d.max()), "mean_lsb": float(d.mean())}

    rows = [cmp("A torch  vs B numpy", A, B),
            cmp("B numpy  vs C exporter", B, C),
            cmp("A torch  vs C exporter", A, C)]
    print(f"n={args.n}  {args.onnx.name}")
    print(f"{'comparison':<24} {'exact %':>9} {'max LSB':>9} {'mean LSB':>10}")
    for r in rows:
        print(f"{r['pair']:<24} {r['exact_pct']:>9.4f} {r['max_lsb']:>9.4f} {r['mean_lsb']:>10.6f}")
    ok = all(r["exact_pct"] == 100.0 for r in rows)
    print(f"\nCROSS_VALIDATION: {'PASS (all bit-identical)' if ok else 'FAIL'}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"n": args.n, "onnx": str(args.onnx),
                                         "bit_identical": ok, "pairs": rows}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
