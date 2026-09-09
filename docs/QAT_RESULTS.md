# QAT results

> ## ⚠ Validity notice — added 2026-09-08 09:40
>
> Codex's second infrastructure pass has found that the `robust_v1` task these
> results are measured on admits a **degenerate solution**: the robot rests on
> its chassis and legs instead of balancing on its wheels.
>
> * `physical_probe.json` — "zero leg actions settle onto the chassis; most
>   cases show 91–100% chassis contact while the old failure logic records no
>   falls."
> * `baseline_1500_sequence.json` — a 1,500-update baseline holds ~0.5 m/s
>   forward RMSE "with **zero valid wheel support**"; turning "partly improves
>   through ground contact on leg links."
> * `asset_audit.json` — at the configured initial height z=0.10 m the wheel
>   bottoms sit ~0.079 m below ground and the lower-leg mesh ~0.052 m below,
>   i.e. the episode begins inside the floor.
>
> **What this invalidates:** the "zero falls in all 13 scenarios" claim, which
> reflects failure logic that does not treat chassis contact as a fall, and
> every absolute control number here as a statement about *locomotion*.
> `SOLID_FP32_PLUS`'s 0.0657 m/s forward RMSE is a real measurement of that
> task, but the task is not wheel-legged locomotion.
>
> **What it does not invalidate:** the quantization comparison. FP32, PTQ and
> QAT were all measured on the identical task, evaluator, scenarios, seeds and
> horizons, so the *relative* result — that W8A8 PTQ destroys the policy and
> QAT recovers 86–101% of it — remains a valid statement about quantization
> sensitivity. It is a claim about arithmetic, not about the robot.
>
> Everything below is being re-run against `locomotion_v1`, Codex's corrected
> task (z=0.18 m, explicit non-wheel contact/tilt/clearance failures), once
> that infrastructure is finished.

Answers the central question of `goal.md`: can a policy trained with the
deployment's fixed-point arithmetic simulated in the loop retain substantially
more control performance than post-training quantization of the same policy?

**Yes — 86–101% of the tracking performance PTQ destroys is recovered, with
training length held fixed.** Two caveats qualify it, both below: attitude and
action smoothness recover far less well than tracking, and the headline number
only appears once the training-length control is in place.

All numbers come from the Codex robustness evaluator (`goal.md` §6 forbids a
competing benchmark): 13 scenarios, eval seeds 11/12/13 independent of the
training seed, 256 envs, full 20 s horizons, 768 episodes per scenario.

## The control changed the answer

The first comparison available was QAT (5500 iterations) against
`SOLID_FP32` (5000). On that basis QAT appeared to *beat* FP32 —
0.1334 versus 0.2898 nominal forward RMSE. That reading was wrong.

`SOLID_FP32_PLUS`, the §7 control — the same 500 extra iterations with no
quantization — reaches **0.0657**, better than QAT on every metric. The
apparent QAT gain was training length.

| arm | iters | nominal fwd | vel-est | action rate | peak roll |
|---|---|---|---|---|---|
| SOLID_FP32 | 5000 | 0.2898 | 0.036 | 0.265 | 0.40 |
| SOLID_FP32_PLUS | 5500 | **0.0657** | **0.015** | **0.087** | **0.12** |
| PTQ W8A8 of the 5000 ckpt | 5000 | 0.4906 | 0.126 | 0.197 | 0.35 |
| QAT W8A8 | 5500 | 0.1334 | 0.044 | 0.385 | 0.43 |

This is exactly the confound §7 exists to catch, and it is why every number
below compares QAT against **PTQ and FP32 at the same 5500 iterations**.

## Main table — training length matched at 5500

Recovery = `(PTQ − QAT) / (PTQ − FP32)`. 100% means QAT fully recovers the
FP32 control; 0% means QAT is no better than post-training quantization.

| scenario | metric | FP32 | PTQ | QAT | recovery |
|---|---|---|---|---|---|
| nominal | fwd RMSE | 0.0657 | 0.5524 | 0.1334 | **86.1%** |
| nominal | yaw RMSE | 0.0533 | 0.9398 | 0.1464 | **89.5%** |
| nominal | vel-est RMSE | 0.0154 | 0.2527 | 0.0441 | **87.9%** |
| nominal | peak roll | 0.1178 | 0.5006 | 0.4266 | 19.3% |
| nominal | action rate | 0.0867 | 0.8035 | 0.3853 | 58.3% |
| delay:2 | fwd RMSE | 0.1475 | 0.7244 | 0.2125 | **88.7%** |
| delay:2 | yaw RMSE | 0.2246 | 1.7007 | 0.3775 | **89.6%** |
| delay:2 | peak roll | 0.2272 | **3.1415** | 0.3907 | **94.4%** |
| delay:2 | action rate | 0.4365 | 1.1123 | 0.7951 | 46.9% |
| friction:0.4 | fwd RMSE | 0.1168 | 0.5082 | 0.1154 | **100.4%** |
| mass:2 | fwd RMSE | 0.0619 | 0.5538 | 0.1249 | **87.2%** |
| mass:2 | peak roll | 0.2002 | 0.4650 | 0.5180 | **−20.0%** |
| push:1.0 | fwd RMSE | 0.4411 | 0.6413 | 0.4386 | **101.2%** |
| push:1.0 | yaw RMSE | 0.2211 | 1.0968 | 0.2137 | **100.9%** |

Falls and survival: QAT is 0.000 falls and a full 20 s in all 13 scenarios.
PTQ at 5500 falls 0.7% of the time under `delay:2` and reaches **3.1415 rad**
of roll — the robot is inverted.

## What QAT does not fix

Tracking recovers; attitude and smoothness do not, and the table should not be
read as a uniform win.

* **Action rate is the real cost** (below). Attitude, contrary to an earlier
  revision of this document, is *not*: see the correction immediately after
  this list.
* **Action rate recovers only 38–68%.** QAT is consistently jerkier than the
  FP32 control — 0.3853 vs 0.0867 nominal, a 4.4× increase. On hardware that is
  actuator wear and power draw, and it is a real cost of the method, not a
  measurement artefact.

### Correction: attitude does recover — the earlier figure used a max

An earlier revision reported "peak roll recovers only 19–94%, and at `mass:2`
QAT is 20% *worse* than PTQ". That used `peak_roll_rad`, which the evaluator
defines as the **worst** episode peak across the whole condition — a maximum
over 768 episodes, dominated by a single outlier. It is the wrong statistic for
comparing seeds or arms, and multi-seed made that visible: its recovery swings
19.3% → 39.1% (nominal) and 32.8% → 108.2% (push) between two seeds that agree
to 0.1 points on every tracking metric.

The evaluator also reports `mean_episode_peak_roll_rad`, the mean of per-episode
peaks, which is the stable companion. On that statistic:

| scenario | FP32 | PTQ | QAT | recovery | (max-stat said) |
|---|---|---|---|---|---|
| nominal | 0.0387 | 0.2939 | 0.0463 | **97.0%** | 19.3% |
| delay:2 | 0.0724 | 0.4255 | 0.0705 | **100.5%** | 94.4% |
| friction:0.4 | 0.0401 | 0.2625 | 0.0434 | **98.5%** | 48.5% |
| mass:2 | 0.0403 | 0.2909 | 0.0498 | **96.2%** | −20.0% |
| push:1.0 | 0.1280 | 0.3903 | 0.1066 | **108.2%** | 32.8% |

and it replicates across seeds: 97.0/97.3 nominal, 100.5/95.7 delay, 108.2/91.2
push.

So **QAT recovers attitude as thoroughly as it recovers tracking, ~96–108%.**
The `mass:2` "QAT is worse than PTQ" result was one unlucky episode out of 768,
not a property of the method. Both statistics are kept in the artefacts; the
max is still worth reporting as a worst case, but not as a recovery fraction.

## A better FP32 policy is harder to quantize

The same W8A8 post-training quantization applied to the two FP32 checkpoints:

| FP32 checkpoint | FP32 fwd RMSE | after PTQ | degradation |
|---|---|---|---|
| SOLID_FP32 (5000) | 0.2898 | 0.4906 | 1.7× |
| SOLID_FP32_PLUS (5500) | 0.0657 | 0.5524 | **8.4×** |

PTQ of the *better* policy is absolutely worse (0.5524 vs 0.4906) and
relatively far worse. Sharpening the FP32 policy adds detail that INT8
activations cannot represent, so improving the float baseline makes
quantization-aware training **more** necessary, not less. Any project that
tunes its FP32 policy and assumes its PTQ margin is preserved is mistaken.

## Mechanism — corrected by measurement

An earlier revision of this document claimed the collapse was driven by the
latent: `enc2` wastes 3.4 of its 16 weight bits, at INT8 the latent gets about
six levels within one sigma, it is the velocity estimate, and it reaches the
actor through a CONCAT that cannot requantize. Velocity-estimation RMSE does
degrade monotonically across the ladder — 0.036, 0.034, 0.044, 0.126, 0.560 —
so the story was consistent with the evidence available at the time.

**It does not survive a direct test.** Quantizing one layer at a time at W8A8
and measuring action error against the fully-FP32 policy:

| layer | quantized alone | share of full | all-but-this | drop vs ALL |
|---|---|---|---|---|
| act0 | 0.34666 | 53.2% | 0.57564 | 11.6% |
| act1 | 0.32563 | 50.0% | 0.56561 | 13.2% |
| act2 | 0.23897 | 36.7% | 0.53207 | **18.3%** |
| obs | 0.17027 | 26.1% | 0.65136 | **0.0%** |
| enc0 | 0.15662 | 24.0% | 0.61851 | 5.0% |
| **enc2** | 0.08593 | **13.2%** | 0.60052 | 7.8% |
| act3 | 0.07394 | 11.4% | 0.63664 | 2.3% |
| enc1 | 0.05089 | 7.8% | 0.66063 | **−1.4%** |

The latent contributes 13.2%, not the majority. The actor's wide hidden layers
dominate — which makes sense on reflection: `enc2` emits only 3 values into a
28-wide concat, so its perturbation is diluted, while `act0` perturbs 128
channels feeding everything downstream.

**And no single layer explains the collapse.** The two attributions disagree:
`act0` is worst alone but removing it recovers only 11.6%, while `obs`
contributes 26.1% alone yet removing it recovers *nothing* — downstream
quantization reproduces the same error. Removing `enc1` makes the result
slightly *worse*. Single-layer shares sum to 222%, so they are a ranking, not
a decomposition.

The defensible statement is therefore weaker and more useful than the original
one: **the W8A8 collapse is distributed across the actor's three hidden layers
with strong error interaction, and no per-layer fix addresses it** — removing
any single layer's quantization leaves at least 82% of the damage.

That is also the best available explanation for why QAT succeeds where
per-layer frac calibration fails. Calibration optimises each layer's grid in
isolation, which is precisely the thing the interaction structure defeats.
Training in the loop co-adapts all of them against the composed error.

### It is not saturation, and QAT cannot rescale the grid

Measured directly at the calibrated fracs, on both the FP32 control and the
QAT policy:

| layer | frac | grid ± | FP32+ peak / clip% | QAT peak / clip% |
|---|---|---|---|---|
| enc0 | 1 | 63.50 | 29.32 / **0.00%** | 33.52 / **0.00%** |
| enc1 | 1 | 63.50 | 37.02 / 0.00% | 61.98 / 0.00% |
| enc2 | 3 | 15.88 | 2.18 / 0.00% | 1.52 / 0.00% |
| act0 | 3 | 15.88 | 7.72 / 0.00% | 6.32 / 0.00% |
| act2 | 2 | 31.75 | 21.81 / 0.00% | 17.67 / 0.00% |
| act3 | 4 | 7.94 | 7.07 / 0.00% | 4.70 / 0.00% |

**Nothing clips.** The calibration is doing its job, and the W8A8 collapse is
pure resolution and rounding noise, not saturation. That rules out the obvious
remedy — widening the grids will not help.

It also constrains what QAT can possibly be doing. The activation fracs are
**fixed inputs** to the QAT run; the training cannot move them. So QAT is not
"adapting the network to the grid" in the sense of rescaling activations into
range — they are already in range. What it can do, and by elimination is doing,
is learn weights whose input-output behaviour is less sensitive to the rounding
noise those fixed grids inject.

(An earlier reading of this used `quant_diagnostics.py`'s saturation column,
which reported 80-96% saturation. That column was computed against a fixed
Q8.8 LSB and ignored the calibrated per-layer fracs entirely. Fixed; it now
derives each tap's LSB from its own layer's frac, and reports 0.0000%.)

Velocity-estimation RMSE remains a genuine and useful *symptom* — it tracks
the damage monotonically and QAT restores it to 0.0441 — but it is not the
cause of the action error.

## Caveats

* **The baseline is not converged.** The task's configured schedule is
  `max_iterations = 50000`; SOLID_FP32 is 5000, i.e. 10% of it. (An earlier
  revision said the configured value was 5000 — the audit script was reading
  the base `LeggedRobotCfgPPO`, which the task overrides.) This is why 500
  extra FP32 iterations improved tracking 4.4x. The QAT/PTQ/FP32 contrast is
  unaffected because all three arms sit at a matched 5500 iterations, but the
  recovery fractions are measured at an under-trained operating point and may
  differ at convergence. Nothing here should be read as a final number for
  this robot.
* `--init-from` restores weights only, not Adam moments or adaptive LR/KL
  state, so both 500-iteration arms are warm starts rather than exact
  continuations (Codex's docs use the same language for their resume path).
  This is identical across QAT and its control, so the comparison holds; but
  part of the 5000→5500 improvement may be the optimizer restart escaping a
  plateau rather than training length alone.
* The command curriculum was **not** a confound: `a_flat_max_command_x` is
  2.0000 at the end of SOLID_FP32 and throughout the fine-tune, so no arm
  trained on an easier command distribution.
* One seed. `goal.md` §32 asks for multi-seed validation; not yet run.
* PTQ and QAT share one activation-frac calibration, taken from on-policy
  observations of SOLID_FP32. Recalibrating per checkpoint might improve PTQ
  of the 5500 policy specifically.

## Provenance

SOLID_FP32 `75e00dc5`, `robust_v1`, 8192 envs, seed 1, 5000 iterations, frozen
read-only. Infrastructure pinned by a 56-file SHA256 manifest because the
Codex pass is uncommitted working-tree state. Quantization confirmed active at
construction by an isinstance assertion, logged as
`quantized policy confirmed: factory W8A8`.

---

# Deployment: which arithmetic should actually ship

The result above is stated at W8A8, but **W8A8 is not what this hardware runs**.
RoboAccel's PL is INT16 weights with Q8.8 activations, and the H7 kernel is the
same. W8A8 belongs to the narrow-arithmetic regime the ASIC sweep explores.

That distinction could not be assumed, because of the finding that a better
FP32 policy quantizes worse: W16A16 was lossless on the 5000-iteration
checkpoint, but the 5500-iteration policy degrades 8.4× at W8A8 against the
earlier 1.7×. So the shipping configuration was re-measured on the sharper
policy rather than inherited.

## W16A16 is still effectively lossless on the sharper policy

| scenario | FP32 | W16A16 PTQ | Δ |
|---|---|---|---|
| nominal fwd RMSE | 0.0657 | 0.0717 | +9% |
| nominal yaw RMSE | 0.0533 | 0.0529 | −0.8% |
| nominal action rate | 0.0867 | 0.0867 | 0 |
| delay:2 fwd RMSE | 0.1475 | 0.1468 | −0.5% |
| push:1.0 fwd RMSE | 0.4411 | 0.4446 | +0.8% |
| falls, all scenarios | 0.000 | 0.000 | — |

The nominal +9% is the same magnitude seen between FP32 and W16A16 on the
5000-iteration policy (0.2898 → 0.3216, +11%), and the perturbed scenarios
agree to well under 1%. Nominal has no disturbance forcing the trajectories
together, so small arithmetic differences diverge chaotically over 2000 steps;
this is run-to-run variation, not a quantization cost.

## Recommendation

**Ship `SOLID_FP32_PLUS` with W16A16 post-training quantization.** QAT is not
required for the current board — PTQ at the shipped precision costs nothing
measurable, and the QAT policy is 4.4× jerkier for no tracking benefit at this
width.

**QAT is what makes a narrower part viable.** Its value is unlocking W8A8,
where PTQ produces a robot that inverts under a two-step delay:

| precision | total bytes | vs W16A16 | PTQ viable? | QAT viable? |
|---|---|---|---|---|
| W16A16 | 78,412 | 1.00× | yes | not needed |
| W8A16 | 40,012 | 0.51× | yes | not needed |
| W8A8 | 39,631 | 0.51× | **no** (3.14 rad roll, 0.7% falls) | **yes** (86–101% recovery) |
| W4A8 | 20,431 | 0.26× | no | untested |

So QAT buys a **2× reduction in model storage** and the halved activation
width an 8-bit datapath needs — on a part that does not exist yet. On the part
that does, it buys nothing. Both statements are worth making plainly, because
only the first is the interesting one and only the second governs what to flash
this week.

## On-silicon verification of the shipping policy

`SOLID_FP32_PLUS` was compiled into the H7 firmware and flashed to the board
(STM32H723VGT6, ST-LINK/V2). Read back over SWD from the JTAG mailbox:

| | measured on silicon |
|---|---|
| golden-vector verify | **scalar PASS, SIMD PASS** |
| weights / MACs | 78,418 B / 38,400 — identical to host generation |
| `simd_dtcm` (pure inference) | 124,301 cyc = **258.96 µs** (p99 259.04) |
| obs→action end to end | 125,122 cyc = **260.67 µs** |
| `scalar_flash` | 161,526 cyc = 336.51 µs |
| DTCM weight copy | 78,418 B |

This closes the chain: the same weights are now bit-exact in five independent
implementations — torch fake-quant, the numpy integer reference, the FPGA
exporter's descriptor program, H7 scalar C and H7 SMLALD — with the last two
verified on the physical part rather than only on the host.

Timing is within 0.31% of the previously measured model (124,682 cycles), as
expected: the topology is unchanged, so 500 iterations of fine-tuning cost
nothing at run time.

**RoboAccel 17.99 µs vs H7 258.96 µs = 14.39×**, at 3.24 cycles/MAC on the H7.

## Measured hardware cost, unchanged by any of this

38,400 MAC, 38,825 parameters, 7 GEMM / 5 ELU / 1 CONCAT.

* RoboAccel @ 100 MHz: 1,799 cycles = **17.99 µs** (cycle model, board-validated)
* STM32H723 @ 480 MHz: 124,682 cycles = **259.75 µs** (measured, SMLALD + DTCM,
  200 runs, DWT CYCCNT)
* **14.4× speedup**; H7 obs→action including quantize/dequantize is 261.49 µs

At 3.25 cycles/MAC the H7 sits above the 2.32–3.03 band measured on four other
policies, because this network's last layers are tiny (latent 3, action 6) and
per-output overhead stops amortising.

---

# §20 — what the hardware's rounding rule costs

`rl_round_shift_sat16.sv` adds half an LSB away from zero and then arithmetic
shifts. For negatives the shift floors *after* the half has already been added
away from zero, so the result lands one LSB below true round-half-away-from-zero
99.6% of the time. The W16A16 diagnostics show it as a negative errMEAN on every
tensor. `hwq/` reproduces it deliberately, because matching the board matters
more than being correct.

The integer reference was run twice over the same 4000 on-policy observations —
once with the hardware rule, once with exact round-half-away-from-zero, verified
against ground truth on a scalar test vector.

| | W16A16 | W8A8 |
|---|---|---|
| output LSB | 0.003906 | 0.0625 |
| actions identical under both rules | 19.6% | **4.1%** |
| mean bias | −0.785 LSB (−0.003) | −1.777 LSB (−0.111) |
| RMS difference | 1.98 LSB | **9.48 LSB** |
| worst difference | 9 LSB | **45 LSB** (2.81 action units) |

At the shipped W16A16 the bias is under one LSB and worth ignoring. At W8A8 it
is not: the rule changes 96% of actions and biases them by nearly two LSB.

## The bias is asymmetric, and that is the interesting part

Per-action bias at W8A8, converted through each channel's action scale:

| action | bias | physical |
|---|---|---|
| lf0 | −1.62 LSB | −0.051 rad |
| lf1 | −4.55 LSB | −0.142 rad |
| **l_wheel** | **−6.32 LSB** | **−3.95 rad/s** |
| rf0 | −6.07 LSB | −0.190 rad |
| rf1 | +8.09 LSB | +0.253 rad |
| **r_wheel** | **−0.19 LSB** | **−0.117 rad/s** |

The two wheels are biased differently by **3.84 rad/s**. A persistent
differential wheel command is a yaw torque, and it predicts that yaw should
degrade worse than forward tracking at W8A8. It does:

| | FP32 | PTQ W8A8 | degradation |
|---|---|---|---|
| forward RMSE | 0.0657 | 0.5524 | 8.4× |
| yaw RMSE | 0.0533 | 0.9398 | **17.6×** |

Yaw degrades twice as badly as forward, which is what a differential bias
would do and is not what uniform resolution loss alone would produce.

**This is consistent with, not proof of, causation.** The measurement is
open-loop: it bounds the per-step perturbation without showing what the closed
loop does with it. Confirming it needs a closed-loop run of the ideal-rounding
variant, which is outstanding.

If it holds, it has a concrete implication for the RTL: correcting the
requantizer's negative-rounding path is a small change that would recover part
of the W8A8 PTQ collapse for free — reducing, though not removing, how much
work QAT has to do at 8-bit activations. That is a hardware fix competing with
a training fix, and it should be measured before the next silicon spin.

---

# §32 — three-seed validation (final)

Seeds 2 and 3 rerun both arms from the same frozen `SOLID_FP32`, changing only
the fine-tuning seed, and evaluate FP32 control / PTQ of that control / QAT.
Mean and range, not mean and standard deviation: three points do not support a
sample standard deviation, and the spread a reader needs is how far apart the
seeds actually landed.

Recovery = `(PTQ − QAT) / (PTQ − FP32)`.

| scenario | metric | seed 1 | seed 2 | seed 3 | mean | range |
|---|---|---|---|---|---|---|
| nominal | fwd RMSE | 86.1% | 83.7% | 90.6% | **86.8%** | 83.7–90.6 |
| nominal | yaw RMSE | 89.5% | 89.6% | 93.0% | **90.7%** | 89.5–93.0 |
| nominal | vel-est RMSE | 87.9% | 88.0% | 88.8% | **88.2%** | 87.9–88.8 |
| nominal | mean peak roll | 97.0% | 97.3% | 98.6% | **97.6%** | 97.0–98.6 |
| nominal | action rate | 58.3% | 57.1% | 46.9% | 54.1% | 46.9–58.3 |
| delay:2 | fwd RMSE | 88.7% | 73.0% | 90.5% | **84.1%** | 73.0–90.5 |
| delay:2 | yaw RMSE | 89.6% | 88.0% | 94.1% | **90.6%** | 88.0–94.1 |
| delay:2 | vel-est RMSE | 69.1% | 77.6% | 77.1% | 74.6% | 69.1–77.6 |
| delay:2 | mean peak roll | 100.5% | 95.7% | 103.0% | **99.8%** | 95.7–103.0 |
| delay:2 | action rate | 46.9% | 53.6% | 50.0% | 50.2% | 46.9–53.6 |
| push:1.0 | fwd RMSE | 101.2% | 84.5% | 77.3% | **87.7%** | 77.3–101.2 |
| push:1.0 | yaw RMSE | 100.9% | 93.4% | 91.8% | **95.4%** | 91.8–100.9 |
| push:1.0 | vel-est RMSE | 90.8% | 91.9% | 86.5% | 89.7% | 86.5–91.9 |
| push:1.0 | mean peak roll | 108.2% | 91.2% | 73.8% | 91.1% | 73.8–108.2 |
| push:1.0 | action rate | 67.8% | 63.1% | 63.9% | 65.0% | 63.1–67.8 |

**The result replicates.** Tracking recovery is 84–88% on average with no seed
below 73%; yaw is 90–95%; attitude, on the stable mean-episode statistic, is
91–100%. Velocity estimation is 75–90%.

**Action rate does not, and is the method's one real cost.** It recovers only
50–65% across every seed and scenario, consistently the lowest row in the
table. QAT policies are roughly 4× jerkier than the FP32 control — on hardware,
actuator wear and power. This is not seed noise; it reproduces tightly.

`max peak roll` is reported in the artefacts but excluded from conclusions: its
recovery ranges 19.3–60.8% on nominal across three seeds that agree to under a
point on everything else, because it is a maximum over 768 episodes.

## Seeds are comparable despite mid-run infrastructure changes

Codex modified `robust_robot.py` at 09:40, between seed 3's FP32 control and
its PTQ/QAT arms, plus `ppo.py` and `envs/__init__.py` earlier. Seed 1's QAT
evaluation was therefore re-run against the changed code: all 21
metric/scenario combinations came back **bit-identical**, 0.000% difference.
The changes are inert for `robust_v1` — the new hooks are identity functions in
the base class, and the added observation-delay reset does nothing at
`observation_delay_steps = [0, 0]`. `ppo.py` cannot matter because evaluation
does no training.

---

# §25 — golden-vector dataset (regenerated for the shipping policy)

The previous 168-vector set was built from `fp32_baseline.onnx`, the
pre-handoff model, and was stale. Regenerating it surfaced two things.

**The calibration dump cannot serve as the golden pool.** Rebuilt from
`calib_obs.npz`, only 2 of 7 regimes appear — 19,699 `balancing` and 301
`disturbance`. That is not a classifier fault: the calibration dump holds one
fixed command in the nominal scenario, which is correct for measuring
activation ranges but produces a steady-state trajectory with no transients.
`dump_regime_obs.py` sweeps eight command/scenario segments instead, stepping
the command between them, because `acceleration` and `braking` are transients.

Pooled over 122,880 samples, 6 of 7 regimes are covered:

| regime | available | selected |
|---|---|---|
| balancing | 15,657 | 24 |
| acceleration | 7,781 | 24 |
| turning | 14,435 | 24 |
| disturbance | 1,294 | 24 |
| near_saturation | 677 | 24 |
| braking | 156 | 24 |
| **large_attitude** | **0** | **0** |

**The empty regime is evidence, not a sampling failure.** `large_attitude`
requires tilt > 0.35 rad (~20° from upright). Across all 122,880 samples,
spanning eight command/scenario segments *including 1.0 m/s pushes*, the tilt
channels peak at 0.346 and 0.233 — the policy never reaches 20°. A robot
resting on its chassis has a stable base and does not tilt. This is an
independent corroboration, from the quantization side of the pipeline, of the
chassis-support exploit in the validity notice at the top of this document.

The 24-vector regime-spanning golden set was compiled into the H7 firmware and
flashed. On silicon: **scalar PASS, SIMD PASS**, 124,458 cycles = 259.29 µs
(p99 259.36), obs→action 261.03 µs — within 0.13% of the previous build, as
expected for identical topology.


---

# Re-run on `locomotion_v2` — the result does not fully survive

Preliminary: one evaluation seed, 128 envs, 500 QAT iterations. Both arms
branch from Codex's `locomotion_v2` seed-1 checkpoint at matched training
length, with activation fracs recalibrated for that policy.

`locomotion_v2` is a real controller — `support_fraction` 1.000 and
`body_contact_fraction` 0.000 — so unlike `robust_v1` its failures are pass/fail
rather than soft tracking degradation.

| segment | metric | FP32 | PTQ | QAT | recovery |
|---|---|---|---|---|---|
| forward | success | 1.00 | 0.00 | **0.98** | **98.4%** |
| stop | success | 1.00 | 0.92 | **1.00** | **100%** |
| every segment | support_fraction | 1.000 | 0.92–0.96 | **0.999–1.000** | **97–100%** |
| reverse | success | 1.00 | 0.00 | 0.17 | 17.2% |
| height | success | 1.00 | 0.02 | 0.02 | −0.8% |
| **turn_left** | success | 1.00 | 0.18 | **0.00** | **−21.9%** |
| **turn_right** | success | 1.00 | 0.41 | **0.00** | **−68.4%** |
| **turn_right** | yaw RMSE | 0.027 | 0.151 | **0.255** | **−83.4%** |
| **combined** | success | 1.00 | 0.00 | 0.00 | 0% |

**What survives.** QAT fully restores wheel support (97–100% on every segment)
and forward and stop locomotion (98–100% success). PTQ drops `support_fraction`
to 0.92–0.96 — the robot is losing its wheels — and QAT puts it back to ~1.0.
Velocity estimation recovers 38–69% throughout.

**What does not.** QAT is *worse than doing nothing* on turning: both
directions go to 0.00 success where PTQ retained 0.18 and 0.41, and yaw RMSE
is 60–83% worse than PTQ. The full `combined` command sequence fails for both.

So the honest statement is narrower than the `robust_v1` headline:
**QAT recovers straight-line locomotion and wheel support at W8A8, and
degrades turning.** The 86–101% figure came from a task where the robot slid on
its chassis, where every metric was a soft degradation with no pass/fail floor
and "turning" was never real.

**A candidate mechanism, untested.** §20 measured the requantizer's rounding
bias as *asymmetric across the wheels* — a 3.84 rad/s differential, which is a
standing yaw torque. QAT trains against that biased arithmetic, and both turn
directions collapsing to exactly zero is consistent with over-fitting a yaw
compensation that does not generalise. Testable by re-running QAT against the
ideal-rounding variant.

**Caveats.** One seed, 128 envs. The budget question is answered below.

## The turning failure is structural, not under-training

QAT was rerun at 2000 iterations — 4x the budget — with its own matched FP32
control, to test whether the harder task simply needed longer.

| segment | 500: FP32/PTQ/QAT | 2000: FP32/PTQ/QAT |
|---|---|---|
| forward | 1.00 / 0.00 / 0.984 | 1.00 / 0.00 / **1.000** |
| reverse | 1.00 / 0.00 / 0.172 | 1.00 / 0.00 / **0.570** |
| height | 1.00 / 0.02 / 0.016 | 1.00 / 0.54 / **1.000** |
| **turn_left** | 1.00 / 0.18 / 0.000 | 1.00 / 0.76 / **0.000** |
| **turn_right** | 1.00 / 0.41 / 0.000 | 1.00 / 0.52 / **0.000** |
| combined | 1.00 / 0.00 / 0.000 | 1.00 / 0.00 / **0.000** |

More budget fixed everything it could: forward reached 1.000, height went
0.016 to 1.000, reverse 0.17 to 0.57. **Turning stayed at exactly 0.000 in both
directions.**

The failure is specifically yaw, and it is not a balance failure. At 2000
iterations QAT's `support_fraction` on both turn segments is **1.0000** — the
robot is upright on its wheels — while yaw RMSE is 0.237 and 0.277 against
PTQ's 0.147 and 0.144. QAT is about **1.9x worse than doing nothing** on yaw.

The control makes it sharper: PTQ's turning *improved* with more training
(0.18 to 0.76) because the FP32 base improved. QAT's did not move at all.

This is what §20 predicted. The requantizer's rounding bias is asymmetric
across the wheels — −6.32 LSB on `l_wheel` against −0.19 LSB on `r_wheel`, a
3.84 rad/s differential, which is a standing yaw torque. QAT trains against
that biased arithmetic and cannot compensate; it over-fits a correction that
fails in both directions.

**That hypothesis was tested and is wrong.** See the next section.

## The rounding rule is not the cause — hypothesis retracted

QAT was retrained for 2000 iterations against a *corrected* requantizer
(`IDEAL_ROUNDING`, exact round-half-away-from-zero) from the same checkpoint,
calibration and seed, and evaluated with that same arithmetic. Only the
rounding rule differs.

| segment | FP32 | PTQ | QAT hw-round | QAT ideal-round |
|---|---|---|---|---|
| forward | 1.000 | 0.000 | 1.000 | 1.000 |
| reverse | 1.000 | 0.000 | 0.570 | **0.992** |
| stop | 1.000 | 0.984 | 0.938 | **1.000** |
| height | 1.000 | 0.539 | 1.000 | 1.000 |
| **turn_left** | 1.000 | 0.758 | 0.000 | **0.000** |
| **turn_right** | 1.000 | 0.523 | 0.000 | **0.000** |
| **combined** | 1.000 | 0.000 | 0.000 | **0.000** |

**Turning did not move.** Both directions stay at exactly 0.000, as does the
combined sequence. The §20 mechanism — that the requantizer's 3.84 rad/s
differential wheel bias causes the turning failure — is **refuted**, despite
having fitted three independent observations beforehand.

It was not a null result, though. Correcting the rounding improved everything
else: reverse 0.570 → 0.992, stop 0.938 → 1.000, and yaw RMSE fell roughly 20%
across every segment (forward 0.0751 → 0.0501, turn_right 0.2766 → 0.2287).
The bias is real and worth fixing in the RTL; it simply is not what breaks
turning.

## A better-supported explanation

With perfect rounding, QAT's yaw RMSE on turns (0.193, 0.229) is still far
worse than PTQ's (0.147, 0.144) — while QAT's forward success is 1.000 against
PTQ's 0.000. QAT is not uniformly better or worse; it **trades yaw accuracy for
forward accuracy**.

That fits every observation: the reward is dominated by linear-velocity
tracking, so QAT spends the scarce INT8 capacity where the objective rewards it
and starves yaw. It explains why neither 4× the training budget nor a corrected
requantizer moves turning — both leave the objective untouched — and it is a
property of *what QAT optimises*, not of the arithmetic.

The prediction it makes, and the next test: reweighting the QAT fine-tune
toward yaw tracking should recover turning at some cost to forward RMSE. If
reweighting does nothing either, the limit is INT8 yaw resolution itself.


---

# Re-run on Codex's final frozen baseline (`SOLID_FP32_V2`)

Codex finished at ~25 h with zero pending items. `SOLID_FP32_V2` is its
`locomotion_v2` seed-1 checkpoint — the configuration its own
`nominal_three_seed_summary.json` validates across three seeds with every
observed segment passing. Frozen read-only, sha256 `ab20c190`, with 145
infrastructure files hashed.

## §8/§9 — the cliff is exactly at 8-bit activations

| arm | success (all 7 segments) | support_fraction |
|---|---|---|
| FP32 | **1.000** | **1.000** |
| W16A16 | **1.000** | **1.000** |
| W8A16 | **1.000** | **1.000** |
| W8A8 | 0.000–0.164 | 0.915–0.950 |
| W4A8 | **0.000** | **0.239–0.480** |

INT8 weights are free; INT8 activations are not. At W4A8 `support_fraction`
falls to 0.239 — the robot is off its wheels most of the time.

This is the sharpest form of the result the project has produced. On
`robust_v1` every metric was a soft tracking degradation on a policy already
resting on its chassis; here `success` and `support_fraction` are pass/fail on
one that genuinely balances, and the deployed W16A16 arithmetic is **exactly**
lossless including the combined command sequence.

## §23–§29 — deployment chain, re-verified on silicon

| gate | result |
|---|---|
| ONNX vs checkpoint | 2.4e-06 |
| FPGA program | 13 instructions, 620/1280 words, CONCAT shift 0 |
| torch vs integer reference | exact at W16A16 **and** W8A8 |
| RTL cross-check | PASS |
| H7 host, 24 regime vectors | 0 scalar / 0 SIMD mismatches |
| **STM32H723, physical** | **scalar PASS, SIMD PASS**, 258.55 µs |

## §12/§10 — diagnostics

Zero saturation on every tensor at W16A16. Layer attribution at W8A8 again
disagrees between one-at-a-time (`enc1`, 57.7%) and leave-one-out (`act0`,
12.4% drop), and removing `obs` alone recovers **0.0%** — downstream
quantization reproduces the same error. As on `robust_v1`, the damage is
distributed with strong interaction and no single-layer fix applies.

## §20 — the rounding bias is differential, not merely negative

| | W16A16 | W8A8 |
|---|---|---|
| actions changed by the rule | 86.5% | **91.8%** |
| mean bias | −0.052 LSB | +0.133 LSB |
| **left–right wheel differential** | — | **+4.05 rad/s** |

An earlier revision described this as a *systematic negative* bias. On this
baseline the mean is near zero because the per-action biases have mixed signs
and cancel — while the wheel differential is 4.05 rad/s, comparable to the 3.84
measured on `robust_v1` but with the opposite sign assignment.

**The differential, not the mean, is the meaningful quantity**, and it is large
and policy-independent in magnitude. It is still worth fixing in the RTL — the
ideal-rounding experiment recovered `reverse` 0.570 → 0.992 and cut yaw RMSE
~20% — even though it is *not* what causes the turning failure.


## §25 — golden vectors, and a confirmed prediction

Regenerated from 61,440 samples across eight command/scenario segments on the
frozen baseline.

| regime | available on `robust_v1` | available on `locomotion_v2` |
|---|---|---|
| balancing | 19,699 | 9,590 |
| disturbance | 301 | 14,332 |
| acceleration | **0** | 637 |
| turning | **0** | 581 |
| near_saturation | **0** | 5,048 |
| braking | **0** | 5 |
| **large_attitude** | **0** | **9,807** |

When `large_attitude` came back empty on `robust_v1` I argued it was evidence
rather than a sampling gap: the regime needs tilt above 0.35 rad (~20°), and a
robot resting on its chassis has a stable base that never tilts. The prediction
made then was that a genuinely balancing policy would reach it, and that a
still-empty result would itself be a finding.

It is populated with 9,807 states. The chassis-exploit diagnosis is confirmed
from an independent measurement path — a regime classifier that simply could
not find high-tilt states before, and now finds them in abundance.

All seven regimes are covered (`braking` has only 5 states available, so 5 were
taken). The 24-vector set compiled into the H7 firmware passes on silicon:
scalar PASS, SIMD PASS, 259.09 µs.


## §30 — control quality and hardware cost are separate axes

They must not be traded off in one number, so they are stated separately.

**Hardware cost** (identical for every arm — quantization changes storage and
arithmetic width, not topology):

| precision | total bytes | vs W16A16 | control quality on the frozen baseline |
|---|---|---|---|
| W16A16 | 78,412 | 1.00× | **1.000 success, 1.000 support — lossless** |
| W8A16 | 40,012 | 0.51× | **1.000 success, 1.000 support — lossless** |
| W8A8 | 39,631 | 0.51× | 0.000–0.164 success (QAT recovers straight-line only) |
| W4A8 | 20,431 | 0.26× | 0.000 success, support 0.239 — off its wheels |

**Latency**, measured not estimated:

* RoboAccel @ 100 MHz: 1,799 PL cycles = **17.99 µs**
* STM32H723 @ 480 MHz: 124,363 cycles = **259.09 µs**, measured on the physical
  part over 200 runs with DWT CYCCNT, SMLALD + DTCM
* **14.4×**, at 3.25 cycles/MAC on the H7

The H7 sits above the 2.32–3.03 cycles/MAC band measured on four other policies
because this network's last layers are tiny — latent 3, action 6 — so per-output
overhead stops amortising.

**The decision this supports.** W8A16 halves model storage at *zero* measured
control cost, which is the free win and is available today with plain PTQ.
W8A8 halves activation storage as well but costs the robot its legs unless QAT
is applied, and even then only straight-line motion returns. W4A8 is not
viable at any training budget tested.

## Deployment recommendation (unchanged by the re-run, now better evidenced)

**Ship `SOLID_FP32_V2` at W16A16.** It is exactly lossless on a
three-seed-validated policy that genuinely balances, across all seven command
segments including the combined sequence, and it is what the exporter actually
emits. QAT is not needed for this board.

> **Correction (2026-09-09, from the release audit).** An earlier revision said
> "or W8A16 if weight storage matters". **That is not currently executable.**
> `tools/export_policy.py` is INT16/Q8.8 **by construction**: `ACT_FRAC = 8` is
> hardcoded (`:32`), `choose_weight_frac` hardcodes `32767.0` = INT16 max
> (`:77-82`), and the tool has **no precision CLI flags**. Upstream,
> `scripts/export_to_onnx.py` deliberately exports an FP32 twin
> (`OFF = dict(quant_weights=False, ...)`), so `weight_bits` / `act_bits` /
> `act_fracs` **never cross the ONNX boundary** and the exporter re-derives
> INT16/Q8.8 on its own.
>
> W8A16's measured control result stands — it was produced through `hwq/`, the
> arbiter, which never touches the exporter. But **shipping it requires adding a
> weight-bit-width parameter to the export path first.** Until then the
> deployable configuration is W16A16.
>
> Same reasoning applies to the "QAT enables an 8-bit-activation part" framing:
> that part does not exist *and neither does an exporter that could target it*.

**QAT's value is enabling an 8-bit-activation datapath**, where PTQ produces a
robot that cannot stand. That remains a real result — with the documented limit
that QAT recovers straight-line locomotion and wheel support but not turning.


---

# §31/§32 — final three-seed tables (frozen baseline)

Three independently seeded QAT runs and their matched FP32 controls, all from
`SOLID_FP32_V2` at 2000 iterations, scored by Codex's locomotion sequence
evaluator. Cells are `mean[min,max]` across seeds; recovery is
`(PTQ − QAT) / (PTQ − FP32)`.

## The robust result: QAT restores wheel support

| segment | FP32 | PTQ | QAT | recovery |
|---|---|---|---|---|
| forward | 1.000 | 0.954 | 0.998 | **95.3%** |
| reverse | 1.000 | 0.969 | 0.991 | 71.9% |
| turn_left | 1.000 | 0.959 | 1.000 | **99.8%** |
| turn_right | 1.000 | 0.964 | 1.000 | **99.9%** |
| stop | 1.000 | 0.962 | 0.998 | 95.7% |
| height | 1.000 | 0.961 | 0.999 | 98.6% |

PTQ costs the robot its wheels — `support_fraction` 0.954–0.969 — and QAT puts
it back to 0.991–1.000 on every segment, with ranges no wider than 0.02. This
is the one finding that replicates tightly.

## Straight-line success recovers, with wide seed variance

| segment | PTQ | QAT | recovery |
|---|---|---|---|
| forward | 0.000 | 0.841 [0.52, 1.00] | 84.1% |
| height | 0.273 | 0.711 [0.13, 1.00] | 60.2% |
| reverse | 0.000 | 0.523 [0.12, 0.88] | 52.3% |
| stop | 0.706 | 0.820 [0.53, 0.99] | 38.9% |

The means are favourable but the ranges are wide — `reverse` spans 0.12 to 0.88
across seeds. A single-seed number here would have been misleading in either
direction.

## Turning fails categorically, in every seed

| segment | PTQ | QAT | recovery |
|---|---|---|---|
| turn_left | 0.544 [0.00, 0.88] | **0.000 [0.00, 0.00]** | **−119.4%** |
| turn_right | 0.469 [0.20, 0.68] | **0.000 [0.00, 0.00]** | **−88.2%** |
| combined | 0.000 | 0.000 | 0.0% |

`0.000 [0.00, 0.00]` in both directions across three independently seeded runs.
This is not seed variance, and QAT is decisively *worse than not quantizing
aware* — PTQ retains 0.47–0.54 mean success where QAT retains none.

## The yaw signature

| segment | FP32 | PTQ | QAT | recovery |
|---|---|---|---|---|
| forward | 0.022 | 0.145 | 0.095 | +40.9% |
| stop | 0.006 | 0.140 | 0.054 | +64.2% |
| **turn_left** | 0.026 | 0.153 | **0.232** | **−62.0%** |
| **turn_right** | 0.025 | 0.142 | **0.278** | **−116.1%** |

QAT improves yaw on straight-line segments and degrades it on turns, by
comparable margins. That is the signature of a policy reallocating limited
INT8 capacity toward the reward-dominant axis — linear-velocity tracking — and
starving yaw, and it holds across all three seeds.

## Final statement

**QAT at W8A8 reliably restores wheel support (95–100%) and substantially
restores straight-line locomotion (39–84%, seed-dependent). It does not restore
turning at any budget, rounding rule, or seed tested, and is worse than PTQ
there.** No arm recovers the full combined command sequence.

For deployment this is moot: W16A16 and W8A16 are exactly lossless on this
policy, so the board ships without QAT. The result matters for a future
8-bit-activation part, where it says the arithmetic is viable for straight-line
driving but not yet for steering.
