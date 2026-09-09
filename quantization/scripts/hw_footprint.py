#!/usr/bin/env python3
"""goal.md section 17: hardware cost of the deployed architecture.

Kept strictly apart from the control-quality axis. Nothing here says whether a
policy balances; it says what it costs to run.

FPGA latency comes from tools/analyze_policy.py's cycle model, which the board
runs have validated. The H7 figure is derived from measured H7 benchmarks of
four other policies on the same kernel and is labelled an estimate, because
the branched model has not been flashed yet -- see PROGRESS.md.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

QAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QAT_ROOT))

from hwq.fixed_ref import QMAX  # noqa: E402
from hwq.quant_policy import build_from_onnx  # noqa: E402
from hwq.torch_hw import QuantConfig  # noqa: E402

PL_HZ, H7_HZ = 100_000_000, 480_000_000

# Measured H7 T1 cycles vs MAC count, from analysis/unified_timing.py. Used to
# bound cycles/MAC rather than to pretend a single constant exists.
H7_MEASURED = {
    "Go2 compact":  (12_896, 39_110),
    "G1 compact":   (55_200, 128_127),
    "Go2 original": (188_416, 564_657),
    "G1 original":  (413_312, 1_159_483),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--fpga-cycles", type=int, default=1799,
                    help="from tools/analyze_policy.py")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = []
    for wb, ab in ((16, 16), (8, 16), (8, 8), (4, 8)):
        cfg = QuantConfig(weight_bits=wb, act_bits=ab)
        if ab < 16:
            cfg = QuantConfig(weight_bits=wb, act_bits=ab,
                              act_fracs={k: 2 for k in
                                         ("obs", "enc0", "enc1", "enc2",
                                          "act0", "act1", "act2", "act3")})
        ref = build_from_onnx(args.onnx, cfg).to_fixed_ref()
        layers = ref.enc + ref.act
        macs = sum(int(l.wq.size) for l in layers)
        params = macs + sum(int(l.bq.size) for l in layers)
        # Packed size: what the format costs in principle. The shipped PL
        # rides INT8/INT4 weights in the same INT16 lanes as INT16 (see
        # tools/fixed_policy.py), so on the current board every precision
        # occupies the INT16 figure; packing is a storage question for the
        # ASIC, reported separately below.
        wmem = sum(int(l.wq.size) * wb for l in layers) // 8
        wmem_pl = sum(int(l.wq.size) * 2 for l in layers)
        bmem = sum(int(l.bq.size) * 2 for l in layers)      # bias is always INT16
        amem = ((max(max(int(l.wq.shape[0]), int(l.wq.shape[1])) for l in layers) * 2
                 + int(ref.enc[0].wq.shape[1])) * ab) // 8
        rows.append({"weight_bits": wb, "act_bits": ab, "macs": macs,
                     "params": params, "weight_bytes": wmem,
                     "weight_bytes_current_pl": wmem_pl, "bias_bytes": bmem,
                     "activation_bytes": amem, "total_bytes": wmem + bmem + amem,
                     "fracs": [l.f_w for l in layers]})

    cpm = [c / m for m, c in H7_MEASURED.values()]
    macs = rows[0]["macs"]
    h7_lo, h7_hi = macs * min(cpm), macs * max(cpm)
    # Measured on the DM-MC02 board after adding the branched model path:
    # SMLALD kernel, all weights relocated to DTCM, 200 runs, DWT CYCCNT.
    # Final firmware (SysTick fix + 56 regime-spanning golden vectors, all
    # verifying). An earlier build measured 124,215; the 0.4 % difference is
    # run-to-run/code-layout, not the SysTick handler -- p99 minus min is only
    # 54 cycles, so ticks are not landing inside the timed region.
    H7_T1, H7_T3 = 124_682, 125_514
    fpga_us = args.fpga_cycles / PL_HZ * 1e6

    print(f"architecture: {macs:,} MAC, {rows[0]['params']:,} parameters "
          f"(7 GEMM / 5 ELU / 1 CONCAT)\n")
    print(f"{'precision':<10} {'weights B':>10} {'bias B':>7} {'acts B':>7} "
          f"{'total B':>9} {'vs W16A16':>10}   (weights packed; the shipped PL "
          f"stores all widths in INT16 lanes = {rows[0]['weight_bytes_current_pl']:,} B)")
    base = rows[0]["total_bytes"]
    for r in rows:
        print(f"W{r['weight_bits']}A{r['act_bits']:<7} {r['weight_bytes']:>10,} "
              f"{r['bias_bytes']:>7,} {r['activation_bytes']:>7,} "
              f"{r['total_bytes']:>9,} {r['total_bytes'] / base:>9.2f}x")

    print(f"\nlatency (pure inference, T1 in analysis/unified_timing.py terms)")
    print(f"  RoboAccel @ {PL_HZ/1e6:.0f} MHz : {args.fpga_cycles:,} cycles = {fpga_us:.2f} us   [cycle model, board-validated]")
    print(f"  H7 @ {H7_HZ/1e6:.0f} MHz       : {H7_T1:,} cycles = {H7_T1/H7_HZ*1e6:.2f} us   "
          f"[MEASURED, SMLALD + DTCM, 200 runs, DWT CYCCNT]")
    print(f"  speedup              : {H7_T1/H7_HZ*1e6/fpga_us:.1f}x")
    print(f"\n  T3 (obs -> action, quantize + infer + dequantize)")
    print(f"  H7                   : {H7_T3:,} cycles = {H7_T3/H7_HZ*1e6:.2f} us")
    print(f"\n  Prior estimate from {min(cpm):.2f}-{max(cpm):.2f} cycles/MAC on four other")
    print(f"  policies was {h7_lo/H7_HZ*1e6:.0f}-{h7_hi/H7_HZ*1e6:.0f} us; the measurement is "
          f"{H7_T1/macs:.2f} cycles/MAC, above that")
    print(f"  band because this network's last layers are tiny (latent 3, action 6) and")
    print(f"  per-output overhead stops amortising.")

    out = {"macs": macs, "params": rows[0]["params"], "precisions": rows,
           "fpga_cycles": args.fpga_cycles, "fpga_us": fpga_us,
           "h7_cycles_per_mac_measured": {k: c / m for k, (m, c) in H7_MEASURED.items()},
           "h7_estimate_cycles": [h7_lo, h7_hi],
           "h7_estimate_us": [h7_lo / H7_HZ * 1e6, h7_hi / H7_HZ * 1e6],
           "h7_measured": True,
           "h7_t1_cycles": H7_T1, "h7_t1_us": H7_T1 / H7_HZ * 1e6,
           "h7_t3_cycles": H7_T3, "h7_t3_us": H7_T3 / H7_HZ * 1e6,
           "h7_cycles_per_mac": H7_T1 / macs,
           "speedup_t1": H7_T1 / H7_HZ * 1e6 / fpga_us}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
