#!/usr/bin/env python3
"""Bit-exact integer reference for the RoboAccel / H7 datapath.

This is the arbiter. Everything else in this package -- the torch fake
quantizer, the QAT training loop, the exported C arrays -- is checked against
what this file computes, because this file is a transcription of three
implementations that already agree with each other on the board:

  rtl/rl_round_shift_sat16.sv   the PL requantizer  (ACC_W = 48)
  rtl/rl_elu_array8.sv          the PL ELU ROM + 3-bit interpolation
  h7_bench/src/ra_kernel.c      the Cortex-M7 SMLALD kernel
  tools/export_policy.py        the exporter that feeds both

Nothing here is a design choice. Where a comment explains "why", it is
explaining the hardware, not this code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- format constants, from rtl/rl_accel_params.vh -------------------------
ACT_FRAC = 8            # `RL_ACT_FRAC -- activations and bias are Q8.8
ACC_W = 48              # accumulator width in the PL requantizer
MAX_WEIGHT_FRAC = 14    # exporter cap on the per-GEMM weight fractional bits

QMAX = {4: 7, 8: 127, 16: 32767}
QMIN = {4: -8, 8: -128, 16: -32768}


def round_away(v):
    """Round half away from zero -- the exporter's rule for float -> integer.

    Distinct from numpy's round (half to even) and from C's cast (toward
    zero); both would disagree with the shipped weights.
    """
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5))


def quantize(v, frac, bits=16):
    """`2.0 ** frac`, not `1 << frac`: a calibrated activation grid can have a
    NEGATIVE fractional count. At INT8 this network needs frac = -1 on enc1
    (peak 130.7, so an LSB of 2.0), and a shift would raise ValueError."""
    return np.clip(round_away(np.asarray(v, np.float64) * (2.0 ** frac)),
                   QMIN[bits], QMAX[bits]).astype(np.int64)


def saturate(v, bits=16):
    return np.clip(np.asarray(v, np.int64), QMIN[bits], QMAX[bits]).astype(np.int64)


def round_shift(v, shift):
    """Add half an LSB away from zero, then arithmetic-shift right.

    Mirrors rl_round_shift_sat16.sv: the half-LSB is *subtracted* for negative
    values, so the rule stays symmetric about zero rather than biasing toward
    -inf the way a bare `>>` would.
    """
    v = np.asarray(v, dtype=np.int64)
    if shift == 0:
        return v
    if shift < 0:
        return v << (-shift)
    half = np.int64(1) << (shift - 1)
    return (v + np.where(v < 0, -half, half)) >> shift


def elu_table() -> np.ndarray:
    """256 entries of ELU(alpha=1) over [-8, 0) on a 1/32 grid, in Q8.8.

    Index i is |x| = i/32, so entry i is round_away((exp(-i/32) - 1) * 256).
    Checked against the literal case statement in rl_elu_array8.sv.
    """
    return round_away((np.exp(-np.arange(256, dtype=np.float64) / 32.0) - 1.0)
                      * 256.0).astype(np.int64)


_TABLE = elu_table()
_NEXT = np.concatenate([_TABLE[1:], np.array([-256], dtype=np.int64)])
ELU_SAT_INPUT = -(8 << ACT_FRAC)   # -2048: below this the PL returns -1.0


def elu_q88(x) -> np.ndarray:
    """Q8.8 ELU. Positive values pass through untouched (identity costs nothing
    in hardware); negatives index the ROM and interpolate on the low 3 bits."""
    x = np.asarray(x, dtype=np.int64)
    neg = x < 0
    sat = x <= ELU_SAT_INPUT
    mag = np.maximum(-x, 0)
    idx = np.minimum(mag >> 3, 255)
    rem = mag & 7
    base, nxt = _TABLE[idx], _NEXT[idx]
    interp = base + (((nxt - base) * rem + 4) >> 3)
    out = np.where(neg, interp, x)
    out = np.where(sat, -(1 << ACT_FRAC), out)
    return saturate(out)


def choose_weight_frac(w, bits: int = 16, cap: int = MAX_WEIGHT_FRAC) -> int:
    """Largest fractional count that keeps |w|max inside the integer range.

    Per tensor, power-of-two only: the requantizer is an arithmetic shifter,
    not a multiplier, so a general scale cannot be represented.
    """
    m = float(np.max(np.abs(w))) if np.size(w) else 0.0
    if m == 0.0:
        return MAX_WEIGHT_FRAC
    return int(min(max(math.floor(math.log2(QMAX[bits] / m)), 0), cap))


# --- layer / network description -------------------------------------------

@dataclass
class LayerRef:
    """One GEMM as the sequencer executes it, plus its optional ELU."""
    wq: np.ndarray        # (N, K) int64, already quantized
    bq: np.ndarray        # (N,)  int64, Q(f_out), added AFTER the shift
    f_w: int              # weight fractional bits
    f_in: int             # input activation fractional bits
    f_out: int            # output activation fractional bits
    elu: bool
    name: str = ""
    act_bits: int = 16

    @property
    def shift(self) -> int:
        return self.f_in + self.f_w - self.f_out

    def __call__(self, x: np.ndarray, stats: "SatStats | None" = None) -> np.ndarray:
        acc = self.wq @ x                      # int48 in hardware; int64 here
        raw = round_shift(acc, self.shift) + self.bq
        if stats is not None:
            stats.observe(self.name, raw, self.act_bits)
        y = saturate(raw, self.act_bits)
        if self.elu:
            y = _elu_in_format(y, self.f_out, self.act_bits)
        return y


def _elu_in_format(y: np.ndarray, frac: int, act_bits: int) -> np.ndarray:
    """The SFU ROM is addressed in Q8.8. At any other activation format the
    requantizer's barrel shifter moves the value in and back out again."""
    if frac == ACT_FRAC:
        return saturate(elu_q88(y), act_bits)
    # round_shift already handles a negative count as a left shift, so one
    # call covers frac above and below Q8.8 -- including negative frac.
    q88 = saturate(round_shift(y, frac - ACT_FRAC))
    e = elu_q88(q88)
    return saturate(round_shift(e, ACT_FRAC - frac), act_bits)


@dataclass
class SatStats:
    """Per-tensor saturation and range, for goal.md section 12."""
    counts: dict = field(default_factory=dict)

    def observe(self, name: str, raw: np.ndarray, bits: int) -> None:
        d = self.counts.setdefault(name, {"n": 0, "sat": 0, "min": None, "max": None})
        d["n"] += int(raw.size)
        d["sat"] += int(np.sum((raw > QMAX[bits]) | (raw < QMIN[bits])))
        lo, hi = int(np.min(raw)), int(np.max(raw))
        d["min"] = lo if d["min"] is None else min(d["min"], lo)
        d["max"] = hi if d["max"] is None else max(d["max"], hi)

    def report(self) -> dict:
        return {k: {**v, "sat_pct": 100.0 * v["sat"] / max(v["n"], 1)}
                for k, v in self.counts.items()}


class FudanFixedPolicy:
    """The deployed encoder+actor, run entirely in integers.

    Graph (verified against the repo, not the article):

        obs_history[125] -> enc0 -> ELU -> enc1 -> ELU -> enc2 -> latent[3]
        concat(obs[25], latent[3])[28]
            -> act0 -> ELU -> act1 -> ELU -> act2 -> ELU -> act3 -> action[6]

    The concat is pure data movement: both operands are already on the same
    activation grid, so the PL CONCAT descriptor carries shift = 0.
    """

    def __init__(self, weights: dict, weight_bits: int = 16, act_bits: int = 16,
                 act_fracs: dict | None = None, obs_frac: int | None = None,
                 max_weight_frac: int = MAX_WEIGHT_FRAC):
        """`weights` maps layer name -> (W float (N,K), b float (N,)).

        Names must be enc0..enc2 and act0..act3.
        """
        self.weight_bits, self.act_bits = weight_bits, act_bits
        af = act_fracs or {}
        self.obs_frac = obs_frac if obs_frac is not None else af.get("obs", ACT_FRAC)
        self.enc, self.act = [], []
        order = [("enc0", True), ("enc1", True), ("enc2", False),
                 ("act0", True), ("act1", True), ("act2", True), ("act3", False)]
        f_prev = {"enc0": self.obs_frac}
        for name, has_elu in order:
            w, b = weights[name]
            f_w = choose_weight_frac(w, weight_bits, max_weight_frac)
            f_in = f_prev.get(name, ACT_FRAC)
            f_out = af.get(name, ACT_FRAC)
            layer = LayerRef(quantize(w, f_w, weight_bits),
                             quantize(b, f_out, 16),
                             f_w, f_in, f_out, has_elu, name, act_bits)
            (self.enc if name.startswith("enc") else self.act).append(layer)
            nxt = {"enc0": "enc1", "enc1": "enc2", "act0": "act1",
                   "act1": "act2", "act2": "act3"}.get(name)
            if nxt:
                f_prev[nxt] = f_out
        # act0 consumes concat(obs, latent); both sit on the same grid.
        self.latent_frac = self.enc[-1].f_out
        assert self.latent_frac == self.obs_frac, (
            "concat needs obs and latent on one grid; PL CONCAT cannot requantize")
        self.act[0].f_in = self.obs_frac

    @staticmethod
    def check_residual_concat(base_frac: int, residual_frac: int) -> None:
        """Second CONCAT constraint, for Codex's base+residual actor.

        The two branches are folded into one GEMM over
        concat(base_penultimate, residual_penultimate) -- see
        scripts/fuse_residual_actor.py, which turns the element-wise ADD the PL
        lacks into a CONCAT it has. That CONCAT cannot requantize either, so the
        branches must share an activation grid.

        Calibrating them independently is the natural thing to do and silently
        produces a policy whose fixed-point output does not match its float
        one, so it fails loudly here instead.
        """
        if base_frac != residual_frac:
            raise ValueError(
                f"residual fusion needs one grid for both actor branches: base "
                f"frac {base_frac} != residual frac {residual_frac}. PL CONCAT "
                f"cannot requantize; calibrate the branches jointly (take the "
                f"tighter of the two) rather than separately.")

    # -- inference ---------------------------------------------------------
    def quantize_obs(self, obs) -> np.ndarray:
        return quantize(obs, self.obs_frac, self.act_bits)

    def run_int(self, obs_q: np.ndarray, hist_q: np.ndarray,
                stats: SatStats | None = None, trace: dict | None = None):
        """Integers in, integers out. obs_q/hist_q are already quantized."""
        x = hist_q
        for layer in self.enc:
            x = layer(x, stats)
            if trace is not None:
                trace[layer.name] = x.copy()
        latent_q = x
        x = np.concatenate([obs_q, latent_q], axis=0)
        if trace is not None:
            trace["concat"] = x.copy()
        for layer in self.act:
            x = layer(x, stats)
            if trace is not None:
                trace[layer.name] = x.copy()
        return x, latent_q

    def __call__(self, obs, obs_history, stats=None, trace=None):
        obs_q = self.quantize_obs(np.asarray(obs, np.float64).reshape(-1))
        hist_q = self.quantize_obs(np.asarray(obs_history, np.float64).reshape(-1))
        act_q, latent_q = self.run_int(obs_q, hist_q, stats, trace)
        f = self.act[-1].f_out
        return (act_q.astype(np.float64) / (2.0 ** f)).astype(np.float32)

    # -- introspection -----------------------------------------------------
    def layer_table(self) -> list[dict]:
        return [{"name": l.name, "K": int(l.wq.shape[1]), "N": int(l.wq.shape[0]),
                 "f_w": l.f_w, "f_in": l.f_in, "f_out": l.f_out,
                 "shift": l.shift, "elu": l.elu,
                 "wq_absmax": int(np.max(np.abs(l.wq))),
                 "bq_absmax": int(np.max(np.abs(l.bq)))}
                for l in self.enc + self.act]

    def weight_bytes(self) -> int:
        wb = 2 if self.weight_bits > 8 else 1
        return sum(l.wq.size * wb + l.bq.size * 2 for l in self.enc + self.act)
