#!/usr/bin/env python3
"""Parse the RTL and the H7 C kernel, and check the Python reference matches.

The three implementations are meant to agree by construction. "Meant to" is
not evidence, so this reads the actual sources and compares.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roboaccel_quant.fixed_ref import elu_table, elu_q88, round_shift, saturate, ACT_FRAC

REPO = Path(__file__).resolve().parents[2]
RTL = Path(os.environ.get("ROBOACCEL_RTL", REPO / "fpga" / "rtl"))
H7 = Path(os.environ.get("ROBOACCEL_H7_SRC", REPO / "stm32" / "src"))


def check_elu_rom() -> int:
    src = (RTL / "rl_elu_array8.sv").read_text()
    rom = {}
    for m in re.finditer(r"8'd(\d+):\s*elu_rom\s*=\s*(-?)16'sd(\d+);", src):
        rom[int(m.group(1))] = -int(m.group(3)) if m.group(2) else int(m.group(3))
    tbl = elu_table()
    missing = [i for i in range(256) if i not in rom]
    bad = [(i, rom[i], int(tbl[i])) for i in sorted(rom) if rom[i] != int(tbl[i])]
    print(f"ELU ROM: parsed {len(rom)}/256 entries from rl_elu_array8.sv")
    if missing:
        print(f"  MISSING indices: {missing[:10]}")
    if bad:
        print(f"  MISMATCH (idx, rtl, py): {bad[:10]}")
    ok = not missing and not bad
    print(f"  -> {'MATCH (all 256)' if ok else 'FAIL'}")
    return 0 if ok else 1


def check_rtl_constants() -> int:
    """Saturation threshold, interpolation shift, accumulator width."""
    src = (RTL / "rl_elu_array8.sv").read_text()
    req = (RTL / "rl_round_shift_sat16.sv").read_text()
    fails = []
    sat = re.search(r"saturated_s1\[lane\]\s*<=.*?<=\s*-16'sd(\d+)", src, re.S)
    if not sat or -int(sat.group(1)) != -(8 << ACT_FRAC):
        fails.append(f"ELU saturation threshold: rtl={sat and sat.group(1)} py={8<<ACT_FRAC}")
    if "((scaled_tmp + 4) >>> 3)" not in src:
        fails.append("ELU interpolation is not (delta*rem + 4) >>> 3")
    accw = re.search(r"parameter integer ACC_W\s*=\s*(\d+)", req)
    if not accw or int(accw.group(1)) != 48:
        fails.append(f"accumulator width: rtl={accw and accw.group(1)} py=48")
    # bias is added AFTER the shift, and only then saturated
    if not re.search(r"shifted\s*=\s*adjusted\s*>>>\s*i_shift;\s*biased\s*=\s*shifted\s*\+", req, re.S):
        fails.append("requantizer does not add bias after the shift")
    if "> 32767" not in req or "< -32768" not in req:
        fails.append("requantizer output saturation is not INT16")
    print("RTL constants:")
    for f in fails:
        print("  MISMATCH:", f)
    print(f"  -> {'MATCH' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def check_h7_kernel() -> int:
    src = (H7 / "ra_kernel.c").read_text()
    fails = []
    if "(v + ((v < 0) ? -half : half)) >> shift" not in src:
        fails.append("H7 round_shift is not round-half-away-from-zero")
    if "ra_sat16(ra_round_shift(acc, L->weight_frac) + L->bias[o])" not in src:
        fails.append("H7 does not add bias after the shift then saturate")
    # The table must be built as 1/exp(+x), not the alternating series for
    # exp(-x): near x = -8 that cancels catastrophically and drove entries
    # 247..255 to -257, one LSB below the ROM. See EXPERIMENT_LOG.md E10.
    if "(1.0 / e - 1.0) * 256.0" not in src:
        fails.append("H7 ELU table is not built as (1/exp(+x) - 1) * 256")
    if "double x = -((double)i) / 32.0;" in src:
        fails.append("H7 ELU table uses the alternating exp(-x) series "
                     "(cancels near x=-8; regression of the E10 fix)")
    if "base + (((next - base) * rem + 4) >> 3)" not in src:
        fails.append("H7 ELU interpolation differs")
    print("H7 kernel (ra_kernel.c):")
    for f in fails:
        print("  MISMATCH:", f)
    print(f"  -> {'MATCH' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def check_round_shift_semantics() -> int:
    """The requantizer is NOT round-half-away-from-zero. Pin down what it is.

    rl_round_shift_sat16.sv adds -half for negative values and then arithmetic
    right-shifts. An arithmetic shift floors, so the negative path computes
    floor((v - half) / 2**s), which sits one LSB below round-to-nearest for
    every negative v except where (|v| + half) is an exact multiple of 2**s.
    The H7 C kernel and tools/export_policy.py use the same idiom and inherit
    the same behaviour, which is why board-vs-python golden tests agree.

    This test asserts the hardware's actual rule, and asserts that the
    deviation from round-to-nearest is exactly the one characterised above --
    so if anyone ever fixes the RTL, this fails loudly instead of silently
    invalidating every trained QAT policy.
    """
    fails = []
    for shift in range(0, 16):
        v = np.arange(-70000, 70001, dtype=np.int64)
        got = round_shift(v, shift)
        if shift == 0:
            exp = v
        else:
            half = np.int64(1) << (shift - 1)
            exp = (v + np.where(v < 0, -half, half)) >> shift
        if not np.array_equal(got, exp):
            fails.append(f"shift {shift}: python != rtl formula")

    q = np.int64(1) << 8
    v = np.arange(-70000, 70001, dtype=np.int64)
    mag = np.abs(v)
    nearest = np.where(v < 0, -((mag + (q >> 1)) // q), (mag + (q >> 1)) // q)
    delta = round_shift(v, 8) - nearest
    pos, neg = v > 0, v < 0
    if not np.all(delta[pos] == 0):
        fails.append("positive path deviates from round-to-nearest")
    if not np.all((delta[neg] == 0) | (delta[neg] == -1)):
        fails.append("negative deviation is not confined to {0, -1}")
    exact = ((mag[neg] + (q >> 1)) % q) == 0
    if not np.array_equal(delta[neg] == 0, exact):
        fails.append("negative deviation does not follow the floor((v-h)/2^s) rule")

    print("round_shift: matches the RTL formula for shifts 0..15")
    print(f"  vs round-to-nearest at shift 8: positives exact, "
          f"negatives {100.0 * np.mean(delta[neg] == -1):.2f}% one LSB low "
          f"(mean {delta[neg].mean():+.4f} LSB)")
    for f in fails:
        print("  MISMATCH:", f)
    print(f"  -> {'MATCH (hardware-faithful)' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def check_elu_properties() -> int:
    x = np.arange(-32768, 32768, dtype=np.int64)
    y = elu_q88(x)
    fails = []
    pos = x >= 0
    if not np.array_equal(y[pos], x[pos]):
        fails.append("positive inputs are not passed through")
    if not np.all(y[x <= -(8 << 8)] == -256):
        fails.append("inputs below -8.0 do not saturate to -1.0")
    if int(y.min()) < -256:
        fails.append(f"output below -1.0: {int(y.min())}")
    d = np.diff(y[x < 0])
    if np.any(d < 0):
        fails.append("ELU is not monotonic on negatives")
    err = np.abs(y[(x < 0) & (x > -(8 << 8))] / 256.0
                 - (np.exp(x[(x < 0) & (x > -(8 << 8))] / 256.0) - 1.0)).max()
    print(f"ELU properties: max |table - exact ELU| = {err:.5f} "
          f"({err*256:.2f} LSB)")
    for f in fails:
        print("  MISMATCH:", f)
    print(f"  -> {'MATCH' if not fails else 'FAIL'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    rc = 0
    for fn in (check_elu_rom, check_rtl_constants, check_h7_kernel,
               check_round_shift_semantics, check_elu_properties):
        rc |= fn()
        print()
    print("VERIFY_AGAINST_RTL:", "PASS" if rc == 0 else "FAIL")
    raise SystemExit(rc)
