# Solid policy → deployment datapath mapping

Required by `goal.md` §5. Companion to `QAT_HANDOFF_AUDIT.md`.

## Verdict

**The actor and history-encoder topology is unchanged by the Codex pass for the
retained `robust_v1` configuration.** The existing quantization / export /
deployment infrastructure is reused as-is; nothing in the fixed-point
reference, the exporter, or the PL descriptor program needs adapting.

`ActorCriticRobust` subclasses `ActorCriticSequence` and, at the default
`encoder_type="mlp"`, inherits `self.encoder` and `self.actor` unmodified.
`robust_config.py:experiment_configs` sets `causal_conv` only for `A4` and
`A5`, neither of which is in `robust_v1`.

Machine-readable: `artifacts/handoff_audit.json` →
`"compatible_with_existing_datapath": true`.

## Layer-by-layer mapping

Observation 25, history 5×25 = 125, latent 3, actor input 28, actions 6.
Weight fractional bits below are SOLID_FP32's, **measured** rather than
inherited: `[12, 13, 14, 13, 14, 14, 14]`. The pre-handoff model used
`[11, 12, 14, 13, 14, 14, 14]`; enc0 and enc1 gained a bit because
SOLID_FP32's weights there are smaller. Activations are Q8.8
(`ACT_FRAC = 8`) at W16A16.

| # | Op | Torch module | K → N | Weight words `⌈K/8⌉·⌈N/8⌉` | W frac | Notes |
|---|---|---|---|---|---|---|
| 0 | GEMM | `encoder.0` | 125 → 128 | 16·16 = 256 | 12 | history in |
| 1 | ELU | `encoder.1` | 128 | — | — | SFU ROM |
| 2 | GEMM | `encoder.2` | 128 → 64 | 16·8 = 128 | 13 | |
| 3 | ELU | `encoder.3` | 64 | — | — | |
| 4 | GEMM | `encoder.4` | 64 → 3 | 8·1 = 8 | 14 | latent out, **no ELU** |
| 5 | CONCAT | `torch.cat` | 25 ⧺ 3 → 28 | — | — | see constraint below |
| 6 | GEMM | `actor.0` | 28 → 128 | 4·16 = 64 | 13 | |
| 7 | ELU | `actor.1` | 128 | — | — | |
| 8 | GEMM | `actor.2` | 128 → 64 | 16·8 = 128 | 14 | |
| 9 | ELU | `actor.3` | 64 | — | — | |
| 10 | GEMM | `actor.4` | 64 → 32 | 8·4 = 32 | 14 | |
| 11 | ELU | `actor.5` | 32 | — | — | |
| 12 | GEMM | `actor.6` | 32 → 6 | 4·1 = 4 | 14 | action mean, **no ELU** |

Totals: **7 GEMM, 5 ELU, 1 CONCAT — program length 13** of the 32-descriptor
instruction RAM; **620 weight words** of the 1280-word per-bank budget;
**38,400 MAC**. All three are unchanged from the pre-handoff model, and all
three are within the capacity limits the exporter enforces.

### The CONCAT constraint still binds

PL `CONCAT` moves words; it cannot requantize. Descriptor 5 therefore requires

```
frac(obs) == frac(latent)
```

`hwq/fixed_ref.py:FudanFixedPolicy` asserts exactly this
(`assert latent_frac == obs_frac`). Nothing in the handoff changes it, and the
per-layer activation calibration must keep honouring it at every precision
point in the matrix.

### What is *not* on the datapath

- The critic (`[256, 128, 64]`) is training-only and stays FP32 — `goal.md` §12.
- `log_std` (renamed from `std` by the Codex pass) is exploration-only. The
  deployed path is the actor mean, so the rename does not reach the export.
- The velocity-estimation auxiliary MSE head is the latent itself; it is
  already descriptor 4.

## A4 / A5 would break the datapath — recorded, not adopted

`goal.md` §5 says not to revert an improved policy merely to avoid extending
the deployment infrastructure. That trade-off does not arise here, because the
retained configuration is already compatible. For the record, if `A4`/`A5` were
later promoted, `CausalHistoryEncoder` would require:

```
ConstantPad1d((2,0)) → Conv1d(25, 32, k=3) → ELU
ConstantPad1d((2,0)) → Conv1d(32, 32, k=3) → ELU
mean over the time axis → Linear(32, 3)
```

Three operators the current PL has no descriptor for: causal zero padding, 1-D
convolution, and a mean-reduction over time. The convolutions can be lowered to
GEMM by im2col (25·3 = 75 → 32 and 32·3 = 96 → 32 per output frame, ×5 frames),
and the mean is a fixed 1/5 scale that folds into the following requantizer's
shift — but the im2col staging buffer and the per-frame loop are host-side work
the AXI4-Lite-only transport would pay for five times over. That is a real
extension, not a repack, and it is out of scope unless a measured `A4`/`A5`
control benefit justifies it.

## Consequence for the experiment tree

Because the mapping is identical, the entire QAT apparatus built before the
handoff — `hwq/fixed_ref.py`, `hwq/torch_hw.py`, `hwq/quant_policy.py`, the
exporter, the golden-vector generator, the H7 kernel and the RTL cross-check —
carries over without modification. The only thing the handoff invalidates is
the *checkpoint*: there is no converged `robust_v1` policy yet, so
`SOLID_FP32` has to be trained before the tree in `goal.md` §1 can branch.


## Confirmed against the exporter, not just asserted

`tools/export_policy.py` was run on `artifacts/solid_fp32/SOLID_FP32.onnx`.
It reads only the ONNX graph and shares no code with this document or with
`hwq/`, so its output is an independent check of every claim above:

| Quantity | Predicted here | Exporter |
|---|---|---|
| Program length | 13 | `instruction_count: 13` of 32 |
| Descriptor order | 7 GEMM / 5 ELU / 1 CONCAT, CONCAT at slot 5 | identical |
| CONCAT shift | 0 (cannot requantize) | `shift: 0` |
| Weight words per bank | 620 | 620 of 1280 |
| Weight fracs | `[12,13,14,13,14,14,14]` | `[12,13,14,13,14,14,14]` |
| Weight residency | fits on-chip | `weight_mode: cache` |

Vector cache use is 132 words of 512. The exporter's own FP32-vs-fixed check
over 1000 samples reports MAE 0.0062, RMSE 0.0081, max 0.0378 — the last being
about 9.7 LSB of the Q8.8 output grid.

`hwq/fixed_ref.py:choose_weight_frac` and the exporter's independent
implementation agree on all seven layers.


---

## Forward look: Codex's 27-observation expansion (checked 2026-09-08 12:45)

Codex's step-6 work adds `commanded_jump_config.py` with
`num_observations = 27` and an `expand_known_observations(sd, 25, 27, 5)` path
that grows a pretrained policy in place, preserving the actor, encoder and
critic (their `test_policy_expansion.py` asserts the pre-expansion latent and
value are reproduced to 1e-6).

This is the first change since the handoff that would move the datapath
geometry, so it was checked rather than assumed:

| | locomotion_v2 (obs 25) | commanded_jump (obs 27) | limit |
|---|---|---|---|
| history / actor input | 125 / 28 | 135 / 30 | — |
| weight words per bank | 620 | **636** | 1280 |
| MACs | 38,400 | **39,936** | — |
| vector cache words | 79 | 80 | 512 |
| program length | 13 | 13 | 32 |
| operators | 7 GEMM, 5 ELU, 1 CONCAT | unchanged | — |

**It still fits, comfortably.** Only `enc0` and `act0` change shape
(125→135 and 28→30 inputs); `⌈135/8⌉ = 17` versus `⌈125/8⌉ = 16` adds 16 weight
words, and `⌈30/8⌉ = ⌈28/8⌉ = 4` leaves `act0` unchanged. The operator sequence,
the CONCAT-at-slot-5 constraint and the ELU ROM are all untouched.

So if the deployed policy ever becomes the jump-capable one, the exporter and
`hwq/` need a re-export, not a redesign. What would *not* survive is A4/A5's
Conv1D encoder — that remains the only known topology in this project the PL
has no descriptor for.


---

# Codex final baseline: the residual-actor policy (2026-09-08 20:50)

Codex finished at ~25 h. Its `selected_controllers.json` promotes commanded-jump
controllers built as "calibrated reference plus trained bounded residual", and
that construction is **not a single feed-forward actor**:

```
encoder        : 135 -> 128 -> 64 -> 3        (obs 27, history 27x5)
actor.base     :  30 -> 128 -> 64 -> 32 -> 6
actor.residual :  30 -> 128 -> 64 -> 6        <- parallel second branch
action         = base(x) + residual(x)
```

The PL implements GEMM, ELU, NORM and CONCAT. **There is no element-wise ADD**,
so the naive mapping does not exist.

## It maps anyway, by fusing the two output GEMMs

Both branches end in a GEMM producing 6 outputs, from 32 and 64 inputs
respectively. Their sum is one GEMM over the concatenated penultimate
activations:

```
W_b @ a + c_b  +  W_r @ b + c_r  ==  [W_b | W_r] @ [a ; b] + (c_b + c_r)
```

Verified numerically on the actual weights over 256 random inputs:
max |fused − separate| = **4.8e-07**, i.e. float32 round-off. The fused weight
is 6x96 with a summed bias.

That replaces the missing ADD with a CONCAT the PL already has.

## Cost

| | current single-actor | residual-actor (fused) | limit |
|---|---|---|---|
| GEMM / ELU / CONCAT | 7 / 5 / 1 | **9 / 7 / 2** | — |
| program length | 13 | **18** | 32 |
| weight words per bank | 620 | **836** | 1280 |
| vector cache words | 79 | **116** | 512 |
| MACs | 38,400 | **52,352** | — |

Everything fits with room to spare. MACs rise 36%, so pure-inference latency
scales with them: RoboAccel ~1,799 -> ~2,450 cycles (18.0 -> ~24.5 µs) and the
H7 ~259 -> ~353 µs, both estimates pending measurement.

## The constraint this adds

The fused GEMM consumes `CONCAT(base2_out[32], residual1_out[64])`, and PL
CONCAT cannot requantize. So, exactly as for `obs`/`latent`:

```
frac(base2_out) == frac(residual1_out)
```

`hwq/fixed_ref.py` asserts the obs/latent case today; the residual case needs
the same assertion before this policy is exported. Calibrating the two branches
independently would silently violate it.
