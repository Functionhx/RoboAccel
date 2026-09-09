#!/usr/bin/env python3
"""Hardware-exact fake quantization with straight-through gradients.

The forward pass is not an approximation of the RoboAccel/H7 datapath -- it is
that datapath, evaluated on tensors. Every integer stage from
`hwq.fixed_ref` is reproduced here, including the requantizer's asymmetric
negative rounding, so a QAT policy trains against the errors the board will
actually make rather than against a tidier model of them.

How the integers stay exact
---------------------------
Torch has no int64 matmul on CUDA, so the accumulation runs in float64.
That is exact, not a compromise: every product of two INT16 operands is an
integer below 2**30, and the longest layer sums 125 of them, so no partial
sum leaves 2**37 -- far inside float64's 2**53 exactly-representable range.
The same argument covers the shift (a power-of-two divide) and the floor.
`scripts/bitexact_check.py` verifies the claim against the numpy reference
rather than leaving it as an argument.

Gradients
---------
Each stage's backward is the derivative of the *unquantized* operation, which
is the straight-through estimator. Two masks make it clip-aware: gradients are
dropped where the INT16 output saturated and where a weight hit the edge of
its integer range, because in both places the true derivative is zero and
pretending otherwise pushes parameters further into the clip.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .fixed_ref import ACT_FRAC, MAX_WEIGHT_FRAC, QMAX, QMIN, elu_table

# ---------------------------------------------------------------- primitives


def round_away(v: torch.Tensor) -> torch.Tensor:
    """Round half away from zero, matching the exporter."""
    return torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))


# goal.md §20. The deployed requantizer adds half an LSB away from zero and
# then FLOORS, which lands negatives one LSB low of true round-half-away-from-
# zero 99.6% of the time and biases the two wheels differently (§20 measured a
# 3.84 rad/s differential, a standing yaw torque). Setting this to True swaps in
# exact round-half-away-from-zero so a QAT run can be trained against a
# corrected requantizer, isolating whether the rounding rule causes the turning
# failure. It does NOT match the current RTL; it models the RTL we could build.
IDEAL_ROUNDING = False


def ideal_round_shift(acc: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Exact round-half-away-from-zero, with no floor asymmetry."""
    two_s = torch.pow(2.0, shift)
    q = acc / two_s
    return torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5))


def hw_round_shift(acc: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """The PL requantizer's shift, quirk included.

    Adds -half for negative accumulators and then floors, exactly as
    rl_round_shift_sat16.sv does. See fixed_ref.round_shift and
    scripts/verify_against_rtl.py for why that is not round-to-nearest.
    """
    two_s = torch.pow(2.0, shift)
    if IDEAL_ROUNDING:
        return ideal_round_shift(acc, shift)
    half = torch.where(shift > 0, torch.pow(2.0, shift - 1),
                       torch.zeros_like(two_s))
    adj = acc + torch.where(acc < 0, -half, half)
    return torch.floor(adj / two_s)


def weight_frac(w: torch.Tensor, bits: int, cap: int = MAX_WEIGHT_FRAC) -> torch.Tensor:
    """Per-tensor power-of-two fractional count, recomputed every forward.

    Dynamic on purpose: tools/export_policy.py derives the frac from the final
    weights, so a frac frozen at initialisation would deploy a different grid
    from the one training saw. It only moves when |w|max crosses a power of
    two, which is rare and is logged.
    """
    m = w.detach().abs().amax()
    m = torch.clamp(m, min=torch.finfo(torch.float32).tiny)
    f = torch.floor(torch.log2(QMAX[bits] / m))
    return torch.clamp(f, 0.0, float(cap))


# ------------------------------------------------------------- autograd ops


class _QuantizeAct(torch.autograd.Function):
    """float -> integer grid, with a clip-aware STE."""

    @staticmethod
    def forward(ctx, x, frac, bits):
        scale = torch.pow(2.0, frac)
        q = round_away(x.double() * scale)
        lo, hi = QMIN[bits], QMAX[bits]
        ctx.save_for_backward((q >= lo) & (q <= hi))
        return torch.clamp(q, lo, hi).to(x.dtype) / scale.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (inside,) = ctx.saved_tensors
        return g * inside.to(g.dtype), None, None


def quantize_act(x: torch.Tensor, frac: torch.Tensor, bits: int = 16):
    return _QuantizeAct.apply(x, frac, bits)


class _HwGemm(torch.autograd.Function):
    """One GEMM descriptor: INT48 accumulate, requantize, bias, INT saturate.

    Forward is the integer datapath. Backward is the derivative of
    `y = x @ w.T + b` evaluated on the dequantized operands -- the STE -- with
    saturated outputs and clipped weights masked out.
    """

    @staticmethod
    def forward(ctx, x_deq, w, b, f_in, f_w, f_out, w_bits, a_bits):
        sw, si, so = (torch.pow(2.0, t) for t in (f_w, f_in, f_out))
        wlo, whi = QMIN[w_bits], QMAX[w_bits]
        alo, ahi = QMIN[a_bits], QMAX[a_bits]

        wq_raw = round_away(w.double() * sw)
        wq = torch.clamp(wq_raw, wlo, whi)
        xq = round_away(x_deq.double() * si)          # already on the grid
        bq = torch.clamp(round_away(b.double() * so), QMIN[16], QMAX[16])

        acc = xq @ wq.t()                             # exact: see module docstring
        raw = hw_round_shift(acc, f_w + f_in - f_out) + bq
        y = torch.clamp(raw, alo, ahi)

        ctx.save_for_backward(x_deq, (wq / sw).to(x_deq.dtype),
                              ((raw >= alo) & (raw <= ahi)).to(x_deq.dtype),
                              ((wq_raw >= wlo) & (wq_raw <= whi)).to(x_deq.dtype))
        ctx.sat_count = int((raw != y).sum().item()) if _COUNT_SAT else 0
        ctx.n_out = raw.numel()
        return (y / so).to(x_deq.dtype)

    @staticmethod
    def backward(ctx, g):
        x_deq, w_deq, live, w_live = ctx.saved_tensors
        g = g * live
        gx = g @ w_deq
        gw = (g.reshape(-1, g.shape[-1]).t() @ x_deq.reshape(-1, x_deq.shape[-1]))
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw * w_live, gb, None, None, None, None, None


_COUNT_SAT = False   # counting forces a device sync; off during training


def set_saturation_counting(on: bool) -> None:
    global _COUNT_SAT
    _COUNT_SAT = on


class _HwElu(torch.autograd.Function):
    """Q8.8 ELU: 256-entry ROM over [-8, 0) with 3-bit linear interpolation.

    Backward is d/dx ELU(x) = exp(x) for x < 0 and 1 otherwise, evaluated on
    the dequantized input. The table is a piecewise-linear approximation of
    that curve, so its true derivative is a staircase with zero gradient
    almost everywhere -- useless for learning, and the reason an STE is used.
    """

    _tbl: dict = {}

    @classmethod
    def table(cls, device):
        key = str(device)
        if key not in cls._tbl:
            t = torch.tensor(elu_table(), dtype=torch.float64, device=device)
            nxt = torch.cat([t[1:], torch.tensor([-256.0], dtype=torch.float64,
                                                 device=device)])
            cls._tbl[key] = (t, nxt)
        return cls._tbl[key]

    @staticmethod
    def forward(ctx, x_deq, frac, a_bits):
        s = torch.pow(2.0, frac)
        xq = round_away(x_deq.double() * s)
        # The SFU ROM is addressed in Q8.8; other formats shift in and back out.
        d = frac - ACT_FRAC
        q88 = torch.where(d > 0, hw_round_shift(xq, d), xq * torch.pow(2.0, -d))
        q88 = torch.clamp(q88, QMIN[16], QMAX[16])

        tbl, nxt = _HwElu.table(x_deq.device)
        mag = torch.clamp(-q88, min=0.0)
        idx = torch.clamp(torch.floor(mag / 8.0), max=255.0).long()
        rem = mag - torch.floor(mag / 8.0) * 8.0
        base, nx = tbl[idx], nxt[idx]
        interp = base + torch.floor(((nx - base) * rem + 4.0) / 8.0)
        out = torch.where(q88 < 0, interp, q88)
        out = torch.where(q88 <= -(8 << ACT_FRAC), torch.full_like(out, -256.0), out)
        out = torch.clamp(out, QMIN[16], QMAX[16])

        back = torch.where(d < 0, hw_round_shift(out, -d), out * torch.pow(2.0, d))
        back = torch.clamp(back, QMIN[a_bits], QMAX[a_bits])
        ctx.save_for_backward(x_deq)
        return (back / s).to(x_deq.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.where(x < 0, torch.exp(x.clamp(min=-30.0)),
                               torch.ones_like(x)), None, None


# ------------------------------------------------------------------ modules


@dataclass
class QuantConfig:
    """Which parts of the datapath are quantized, and at what width.

    The defaults are the shipped hardware. The booleans exist for the staged
    schedule in goal.md section 8 and the ablations in section 15; every
    non-default combination is an experiment that must be named as such.
    """
    weight_bits: int = 16
    act_bits: int = 16
    act_frac: int = ACT_FRAC
    obs_frac: int = ACT_FRAC
    # Per-layer output fractional bits, keyed "obs" and enc0..act3. None means
    # the shipped fixed Q8.8 grid. Required for act_bits < 16: an INT8 tensor
    # at frac 8 spans only +-0.5, while real activations here reach 22, so an
    # uncalibrated A8 run measures saturation and nothing else.
    act_fracs: dict | None = None
    quant_weights: bool = True
    quant_obs: bool = True
    quant_hidden: bool = True     # hidden activations + the ELU table
    quant_output: bool = True     # the final GEMM's output grid
    # goal.md section 15 ablations G/H/I: which half of the deployed network
    # carries the quantization. Both true is the real deployment.
    quant_encoder: bool = True
    quant_actor: bool = True
    # goal.md section 15 ablation F. 14 is the value tools/export_policy.py
    # uses; it is a software constant, not a hardware limit, and it costs
    # real resolution on the two small-magnitude output layers.
    max_weight_frac: int = MAX_WEIGHT_FRAC
    # goal.md section 10: restrict quantization to a named subset of layers so
    # each one's contribution to the collapse can be attributed. None means
    # every layer, which is the real deployment; anything else is an ablation.
    only_layers: frozenset | None = None

    def layer_on(self, name: str) -> bool:
        return self.only_layers is None or name in self.only_layers

    def frac_of(self, name: str) -> int:
        if self.act_fracs and name in self.act_fracs:
            return int(self.act_fracs[name])
        return self.obs_frac if name == "obs" else self.act_frac

    def tag(self) -> str:
        parts = [f"W{self.weight_bits}A{self.act_bits}"]
        off = [n for n, v in (("w", self.quant_weights), ("o", self.quant_obs),
                              ("h", self.quant_hidden), ("y", self.quant_output),
                              ("E", self.quant_encoder), ("A", self.quant_actor))
               if not v]
        if off:
            parts.append("no" + "".join(off))
        return "_".join(parts)


class HwLinear(nn.Module):
    """nn.Linear with the deployed datapath in the forward pass."""

    def __init__(self, in_f: int, out_f: int, cfg: QuantConfig,
                 is_output: bool = False, name: str = "", prev: str = "obs"):
        super().__init__()
        self.name, self.prev = name, prev
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.cfg, self.is_output = cfg, is_output
        self.register_buffer("last_frac", torch.zeros(()), persistent=False)
        self.sat_count = 0
        self.sat_total = 0

    def active(self) -> bool:
        if not self.cfg.layer_on(self.name):
            return False
        half = (self.cfg.quant_encoder if self.name.startswith("enc")
                else self.cfg.quant_actor)
        return half and self.cfg.quant_weights and (
            self.cfg.quant_output if self.is_output else self.cfg.quant_hidden)

    def forward(self, x, f_in: torch.Tensor | None = None):
        if not self.active():
            return nn.functional.linear(x, self.weight, self.bias)
        cfg = self.cfg
        f_w = weight_frac(self.weight, cfg.weight_bits, cfg.max_weight_frac)
        self.last_frac = f_w.detach()
        f_out = torch.tensor(float(cfg.frac_of(self.name)), device=x.device,
                             dtype=torch.float64)
        if f_in is None:
            f_in = torch.tensor(float(cfg.frac_of(self.prev)), device=x.device,
                                dtype=torch.float64)
        y = _HwGemm.apply(x, self.weight, self.bias, f_in, f_w, f_out,
                          cfg.weight_bits, cfg.act_bits)
        return y


class HwElu(nn.Module):
    """ELU on the grid of the layer whose output it consumes."""

    def __init__(self, cfg: QuantConfig, name: str = ""):
        super().__init__()
        self.cfg, self.name = cfg, name

    def forward(self, x):
        half = (self.cfg.quant_encoder if self.name.startswith("enc")
                else self.cfg.quant_actor)
        if not (half and self.cfg.quant_hidden and self.cfg.layer_on(self.name)):
            return nn.functional.elu(x)
        f = torch.tensor(float(self.cfg.frac_of(self.name)), device=x.device,
                         dtype=torch.float64)
        return _HwElu.apply(x, f, self.cfg.act_bits)


class HwObsQuant(nn.Module):
    """Sensor float -> Q8.8, where the MCU does it before the first GEMM.

    Two instances exist because the two halves can be ablated separately: the
    history feeds the encoder, the current observation feeds the actor.
    """

    def __init__(self, cfg: QuantConfig, feeds: str = "actor"):
        super().__init__()
        self.cfg, self.feeds = cfg, feeds

    def forward(self, x):
        half = (self.cfg.quant_encoder if self.feeds == "encoder"
                else self.cfg.quant_actor)
        if not (half and self.cfg.quant_obs and self.cfg.layer_on("obs")):
            return x
        f = torch.tensor(float(self.cfg.frac_of("obs")), device=x.device,
                         dtype=torch.float64)
        return quantize_act(x, f, self.cfg.act_bits)
