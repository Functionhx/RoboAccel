# Goal 5 — Confirm the PPO–QAT Failure Mechanism and Freeze Report-Ready Evidence

We now have a concrete mechanism:

**W8A8 quantization → extreme policy sensitivity → KL amplification → PPO adaptive-LR collapse → optimizer cannot meaningfully step → reward interventions become ineffective.**

Current key evidence:

* A weight perturbation of only `1e-5` produces:

  * FP32 KL: `4.06e-07`
  * W8A8 KL: `6.73e-02`
  * amplification: `165,737×`
* PPO desired KL threshold is exceeded by ~`6.7×` even at this tiny perturbation.
* W8A8 adaptive LR stays at its floor for `2000/2000` iterations.
* FP32 control hits the floor only `10/2000` iterations.
* Earlier reward interventions, including yaw reward ×4, barely changed behavior, consistent with an optimizer that cannot move.

## Primary objective

Finish the two confirmation experiments and determine whether the mechanism is causally supported:

1. `qat_v2_fixlr`

   * W8A8 QAT
   * disable/bypass adaptive KL-based LR control
   * use the same effective LR schedule / step size as the FP32 control as closely as possible

2. `qat_v2_w8a16`

   * same PPO/QAT setup
   * W8 weights, A16 activations
   * retain the normal adaptive KL controller

Both jobs are currently relaunched under systemd user units. Do not accidentally restart or duplicate them if healthy.

## Required analysis after completion

For FP32, original W8A8, `qat_v2_fixlr`, and `qat_v2_w8a16`, report:

* final reward / task metrics
* yaw RMSE and other existing evaluation metrics
* KL statistics over training
* LR trajectory
* fraction/count of iterations at LR floor
* whether policy parameters actually changed meaningfully
* checkpoint hashes and evaluation provenance

Explicitly answer:

### Hypothesis A

Does fixed LR rescue W8A8 training?

If yes, this supports:

> the dominant failure is the interaction between quantization-induced policy sensitivity and PPO's KL-based step-size controller, rather than insufficient W8A8 policy capacity.

### Hypothesis B

Does W8A16 substantially reduce KL sensitivity and/or restore adaptive-LR training?

If yes, this supports:

> activation quantization is a major contributor to the pathological policy sensitivity.

## Add one mechanism figure/table

Create a compact perturbation-response analysis for:

* FP32
* W8A16
* W8A8

Measure policy KL under identical weight perturbation magnitudes, ideally across several scales around:

`1e-7, 3e-7, 1e-6, 3e-6, 1e-5`

Do not cherry-pick. Use the exact same perturbation procedure, states, policy outputs, and KL definition for all variants.

Produce data suitable for a figure:

`weight perturbation magnitude → policy KL`

The purpose is to show whether quantization changes the local sensitivity of the policy.

## Guardrails

* Do not modify the already-running confirmation jobs unless they are demonstrably broken.
* Do not silently change PPO, reward, environment, seeds, evaluation protocol, or quantization settings.
* Verify every checkpoint actually contains different deployed weights where expected.
* Treat suspiciously identical results as a bug until proven otherwise.
* Record exact commands, configs, commit hashes, checkpoint hashes, and seeds.
* No unsupported interpretation: separate observation, inference, and causal claim.
* If a confirmation result contradicts the proposed mechanism, investigate and report it rather than forcing the story.

## Deliverable

Update `goal5.md` with:

1. experiment status
2. exact results
3. mechanism confirmation/refutation
4. perturbation-vs-KL table
5. report-ready figures/tables or scripts to generate them
6. concise final conclusion
7. remaining uncertainties

At the end, state whether the evidence is sufficient to freeze a v0 technical report titled approximately:

**“Why W8A8 QAT Stalls in PPO: KL Amplification and Learning-Rate Collapse”**

Do not broaden the scope unnecessarily. The goal is to **confirm or falsify this mechanism cleanly**, then freeze the evidence.


---

# STATUS · 2026-09-09

## 1. Experiment status

| Arm | State | Iterations | Unit |
|---|---|---|---|
| `solid_fp32_plus_v2_long` (FP32 control) | complete | 2000 | — |
| `qat_w8a8_v2_long` (original W8A8) | complete | 2000 | — |
| **`qat_v2_fixlr`** | **running** | see below | `systemd --user ra-fixlr` |
| **`qat_v2_w8a16`** | **running** | see below | `systemd --user ra-w8a16` |

Both confirmation jobs are `active` and were **not** touched, per the goal-5
guardrail. They were relaunched under systemd user units after two earlier
launches (plain `nohup`, then `setsid`) were killed by session teardown at
~1990 and ~250 iterations. Progress is read with
`experiments/goal4/collect_results.py`, which parses each run's own
`metrics.jsonl` and tolerates the truncated final line a killed run leaves.

### Exact commands

```bash
systemd-run --user --unit=ra-fixlr --collect \
  --working-directory=$QAT \
  --setenv=LD_LIBRARY_PATH=$ENV/lib --setenv=PYTHONPATH=$PLANE \
  --property=StandardOutput=append:$QAT/artifacts/v2/train_qat_v2_fixlr.log \
  --property=StandardError=append:$QAT/artifacts/v2/train_qat_v2_fixlr.log \
  $ENV/bin/python scripts/train_policy.py \
    --run qat_v2_fixlr --task locomotion_v2 --quant W8A8 \
    --act-fracs $QAT/artifacts/v2/act_fracs_a8.json \
    --schedule fixed --learning-rate 2.563e-4 \
    --init-from $QAT/artifacts/v2/frozen/SOLID_FP32_V2.pt \
    --iters 2000 --num-envs 4096 --seed 1 --headless

# identical, except:  --quant W8A16   (and no --schedule/--learning-rate)
```

`2.563e-4` is the **median learning rate the FP32 control actually used** over
its own 2000 iterations, not a guess. Both overrides are asserted after the
algorithm is constructed, because mutating a config object after its consumer
has read it is a silent no-op — that exact failure produced three
bit-identical policies from three different reward weights earlier in this
project.

```
learning_rate override VERIFIED in optimizer: 2.563e-04 schedule=fixed
quantized policy confirmed: factory W8A8   /   factory W8A16
```

### Provenance

| Item | SHA-256 (first 16) |
|---|---|
| `SOLID_FP32_V2.pt` — warm start for **every** arm | `ab20c19071c4be2c` |
| `qat_w8a8_v2_long/model_2000.pt` | `e42ec7065abb5611` |
| `solid_fp32_plus_v2_long/model_2000.pt` | `343a027646c660db` |

Every arm shares: `locomotion_v2`, seed 1, 4096 envs, 2000 iterations, the same
warm start, the same `act_fracs_a8.json`, `desired_kl = 0.005` (threshold
`0.01`), learning-rate floor `1e-5`. Nothing else was changed.

## 2. Perturbation-vs-KL table  **[complete]**

The required mechanism measurement. Same perturbation procedure, same 4096
observations, same KL definition, 12 trials per point, actor parameters only.
A step is one Adam update with unit-RMS gradients, so `|dw| ≈ lr` elementwise.

**`codes` is the number of the actor's 14,016 INT8 weight codes that actually
changed** — measured, not assumed. It is identical for W8A16 and W8A8 by
construction: both are 8-bit weights.

### QAT checkpoint (`e42ec706…`)

| \|dw\| | codes moved | float | W16A16 | W8A16 | W8A8 | W8A8/W8A16 |
|---|---:|---|---|---|---|---:|
| `1e-07` | 0.9 | 3.89e-11 | 9.22e-06 | 2.66e-05 | 1.84e-03 | 69× |
| `3e-07` | 0.9 | 3.30e-10 | 2.24e-05 | 2.66e-05 | 1.84e-03 | 69× |
| `1e-06` | 4.2 | 3.66e-09 | 5.64e-05 | 5.19e-05 | 1.80e-02 | 346× |
| `3e-06` | 8.2 | 3.30e-08 | 8.96e-05 | 7.75e-05 | 2.02e-02 | 260× |
| `1e-05` | 30.2 | 3.66e-07 | 1.42e-04 | 8.61e-04 | 6.28e-02 | 73× |
| `3e-05` | 89.2 | 3.30e-06 | 1.69e-04 | 1.00e-03 | 8.01e-02 | 80× |
| `0.0001` | 299.7 | 3.66e-05 | 2.63e-04 | 1.94e-03 | 1.03e-01 | 53× |
| `0.0002563` | 764.7 | 2.41e-04 | 6.53e-04 | 3.15e-03 | 1.25e-01 | 40× |

### FP32 checkpoint (`ab20c190…`)

| \|dw\| | codes moved | float | W16A16 | W8A16 | W8A8 | W8A8/W8A16 |
|---|---:|---|---|---|---|---:|
| `1e-07` | 0.0 | 1.09e-10 | 2.01e-05 | **0** | **0** | — |
| `3e-07` | 0.3 | 9.27e-10 | 4.91e-05 | 4.56e-06 | 1.31e-04 | 29× |
| `1e-06` | 0.9 | 1.03e-08 | 1.15e-04 | 2.72e-05 | 1.54e-03 | 57× |
| `3e-06` | 5.2 | 9.26e-08 | 2.11e-04 | 2.22e-03 | 6.97e-02 | 31× |
| `1e-05` | 17.3 | 1.03e-06 | 3.14e-04 | 2.36e-03 | 9.84e-02 | 42× |
| `3e-05` | 55.1 | 9.26e-06 | 4.23e-04 | 3.08e-03 | 1.44e-01 | 47× |
| `0.0001` | 182.8 | 1.03e-04 | 8.19e-04 | 8.11e-03 | 2.50e-01 | 31× |
| `0.0002563` | 467.6 | 6.76e-04 | 1.84e-03 | 1.68e-02 | 3.39e-01 | 20× |

Reproduce: `quantization/scripts/kl_amplification.py --checkpoint <ckpt>
--lrs 1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,2.563e-4 --trials 12`.
Figure: `assets/perturbation_response.svg`, generated by
`assets/make_figures.py` from `docs/results/kl_perturbation_qat.json`.

## 3. What the perturbation data establishes

**Observation 1 — the float policy is exactly quadratic in step size.**
Over a **2563×** span of `|dw|`, float KL rises **6.19e6×**. Quadratic would be
2563² = 6.57e6. This is the behaviour PPO's controller is designed around:
halve the step, quarter the KL.

**Observation 2 — the quantized policy is not.** Over the same span, W8A8 KL
rises **68×** and W8A16 **118×** (QAT checkpoint). Fitting an exponent,
W8A8 responds as `|dw|^0.54` against float's `|dw|^2.00`.

> *Inference.* The controller's only lever moves the quantity it regulates with
> an exponent of ~0.5 instead of 2. Reducing the learning rate by 10× reduces
> the measured KL by ~3.5× under W8A8, against 100× in float.

**Refinement — flatness alone is not the discriminator.** Fitting the exponent
for every arm on the QAT checkpoint:

| arm | exponent, ±lr steps | exponent, Gaussian steps |
|---|---:|---:|
| float | **1.99** | **1.99** |
| W16A16 | 0.54 | 0.55 |
| W8A16 | 0.61 | 0.61 |
| W8A8 | 0.54 | 0.52 |
| W4A8 | 0.51 | 0.60 |

**Every quantized arm is flat**, including the two that train perfectly well.
So the sub-quadratic response is a property of quantization in general, and it
is *necessary but not sufficient* for the collapse. What separates W8A8 is the
**absolute level** of its floor against the fixed 0.01 threshold, not the shape
of its curve. Both facts are needed: the floor sits above the threshold, *and*
the controller cannot lower it.

> An earlier draft of this section attributed the collapse to the exponent
> alone. That was wrong, and the W16A16 column is what shows it.

**Robustness — the exponent is not an artifact of the step model.** The table
above repeats the whole sweep with `dw ~ N(0, lr²)` instead of `dw = ±lr`, so
a weight sitting just inside a boundary is no longer guaranteed either to cross
it or not to. Float stays at 1.99; W8A8 moves from 0.54 to 0.52. The flatness
is the quantizer, not the sign structure. **This retires uncertainty 4 below.**
Reproduce with `--perturbation gauss`.

**Observation 3 — there is a dead zone.** On the FP32 checkpoint at
`|dw| = 1e-7`, **0.0** weight codes move and KL is **exactly 0**. On the QAT
checkpoint, `1e-7` and `3e-7` move the *same* 0.9 codes and give the *same*
KL, `1.843e-03`, to the last digit.

> Those identical values were treated as a suspected bug, as goal-5 requires.
> They are not one: the crossing counter shows the two perturbations produce a
> **bit-identical quantized network**, so an identical KL is arithmetic. The
> response is piecewise constant — flat until a boundary is crossed, then a
> jump — which is the discontinuity stated directly.

**Observation 4 — the amplifier is the activation path, not the weight path.**
W8A16 and W8A8 have identical weights and, at every magnitude, move an
**identical number of weight codes**. Their KL nevertheless differs by
**40×–350×** (typically ~70×).

> *Causal claim, with its control.* Weight quantization creates the
> discontinuity; **8-bit activations amplify each crossing's effect on the
> policy output by roughly two orders of magnitude**. Since the weight-side
> perturbation is held identical and measured to be identical, activation width
> is the only remaining difference between the two arms.

Verified at **all 16 measurement points across both checkpoints**: W8A16 and
W8A8 move the same number of codes every time.

**Observation 5 — it is not the number of crossings.** W16A16 has a 16-bit
weight grid, so its boundaries are far denser. At `|dw| = 1e-5` it moves
**2,272** codes against W8A8's **30** — **75× more crossings** — and yet its KL
is **1.42e-04** against W8A8's **6.28e-02**, i.e. **440× lower**.

> *Inference.* Crossing count and KL move in **opposite** directions across the
> precision ladder, which rules out "more boundary crossings" as the driver.
> What separates the arms is how much each crossing perturbs the policy output,
> and that is set by the width of the activation path it propagates through.

This is the cleanest support for **Hypothesis B** available without waiting for
training, and it is a *sensitivity* result rather than a *trainability* result —
the running `qat_v2_w8a16` arm tests the trainability half.

## 4. Hypothesis A — does fixed LR rescue W8A8?  **NO — falsified**

`qat_v2_fixlr` completed 2000/2000. The intervention worked exactly as
intended: learning rate constant at **2.563e-4** (min == max over all 2000
iterations), **0.0%** of iterations at the floor, against the baseline's
100.0%. The optimizer could step, and was verified in the optimizer at
construction.

**Turning did not come back.**

| arm | forward | reverse | turn_L | turn_R | stop | height | combined | **mean** |
|---|---|---|---|---|---|---|---|---|
| FP32 control | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| PTQ W8A8 | 0.000 | 0.000 | 0.758 | 0.523 | 0.984 | 0.539 | 0.000 | 0.401 |
| QAT W8A8 baseline | 1.000 | 0.570 | **0.000** | **0.000** | 0.938 | 1.000 | 0.000 | 0.501 |
| **QAT W8A8 fixed-LR** | 0.984 | 0.438 | **0.000** | **0.000** | 0.023 | 0.000 | 0.000 | **0.206** |

`yaw_rmse_rad_s` on the turning segments: baseline 0.2371 / 0.2766, fixed-LR
**0.2470 / 0.2545**, against the FP32 control's 0.0224 / 0.0295.

> **The hypothesis is falsified, and the result is worse than the null.**
> Fixed LR did not merely fail to restore turning — overall success fell from
> 0.501 to **0.206**, below even PTQ. `stop` collapsed from 0.938 to 0.023 and
> `height` from 1.000 to 0.000.

*Inference.* Restoring the step size without adding any behavior-preservation
term let the policy drift **further** from the reference. The learning-rate
floor had been acting as an accidental brake. Removing the brake, on an
objective with nothing anchoring it to the reference policy, made things worse
— which is what the objective audit predicted: PPO surrogate + value − entropy
contains no term that refers to the reference at all.

Arm A (float execution of the fixed-LR parameters) scores **0.143**. It is
still quantizer-dependent, exactly like the baseline.

## 5. Hypothesis B — does W8A16 restore adaptive-LR training?  **YES — confirmed**

`qat_v2_w8a16` completed 2000/2000 under the **normal** adaptive controller.
Learning rate at floor **2.8%** of iterations (baseline 100.0%), median KL
**0.0055** against the 0.01 threshold — the controller behaved.

| arm | forward | reverse | turn_L | turn_R | stop | height | combined | **mean** |
|---|---|---|---|---|---|---|---|---|
| **QAT W8A16, fake-quant (B)** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| **QAT W8A16, float (A)** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

`yaw_rmse_rad_s` 0.0194 / 0.0261 — **better than the FP32 control's**
0.0224 / 0.0295.

> **Arm A equals arm B equals 1.000.** W8A16 QAT does *not* become
> quantizer-dependent. It learned a policy that is genuinely robust to
> quantization rather than one that requires it. This is the sharpest possible
> contrast with W8A8, where arm A collapses to 0.143–0.000 while arm B still
> passes several segments.

The sensitivity half of Hypothesis B was already supported by the perturbation
study (§3, Observation 4). The trainability half is now confirmed directly.

## 5b. What this does to the proposed mechanism

**The mechanism is real but it is not the cause.** Both halves have to be
stated:

| Claim | Status |
|---|---|
| W8A8 fake-quant amplifies policy KL by ~1.6e5× at the LR floor | **measured**, §2 |
| That pins PPO's learning rate at its floor for 2000/2000 iterations | **measured**, §1 |
| **That collapse is why W8A8 loses turning** | **FALSIFIED** — §4 |
| Activation precision is the binding constraint | **supported** — §5 |
| Current QAT has no behavior-preservation term | **established** (objective audit) |

The learning-rate collapse is a genuine, reproducible, previously undocumented
pathology of QAT under KL-adaptive on-policy RL, and the noise-floor criterion
still predicts which precisions trigger it. But it is a **co-symptom** of
8-bit activations, not the mechanism by which control is lost. Removing it in
isolation makes the policy worse.

*Why the earlier reasoning was wrong.* The LR collapse and the control failure
were perfectly correlated across the precision ladder, and correlation across
a ladder that varies one thing is easy to mistake for causation. Only the
intervention separated them.

## 6. Remaining uncertainties

1. ~~Both confirmation arms are incomplete.~~ **Closed.** Both completed
   2000/2000; §4 and §5 are answered.
2. **One seed.** Every arm here is seed 1. The mechanism measurement is
   deterministic given a checkpoint, but the training outcomes are not.
3. **One model, one task.** The exponent 0.54 and the ~70× activation
   amplification are properties of this 38,400-MAC policy on `locomotion_v2`.
   Nothing here shows they generalise.
4. **The perturbation is isotropic, not a real gradient.** ~~`|dw| ≈ lr`
   elementwise with random signs models an Adam step with unit-RMS
   gradients.~~ **Substantially retired**: repeating the sweep with Gaussian
   steps of the same RMS changes the W8A8 exponent from 0.54 to 0.52 and
   leaves float at 1.99. What remains open is *correlation* — a real gradient
   is correlated across weights, and neither step model reproduces that.
5. **The encoder is out of scope of the perturbation study.** It is stepped by a
   *separate* optimizer at a fixed 1e-3 that the adaptive controller never
   touches (`ppo.py` writes only `self.optimizer.param_groups`), and its
   `Encoder/policy_kl` is ~100× the control's. That interaction is observed but
   not isolated.
6. **`ladder_pooled.json` is single-seed.** The precision ladder in §5 comes
   from one seed; the three-seed evidence covers the separate QAT-versus-PTQ
   comparison.

## 7. Is the v0 report ready to freeze?

**Yes — but not under the proposed title.**

*"Why W8A8 QAT Stalls in PPO: KL Amplification and Learning-Rate Collapse"*
asserts that the collapse is the reason W8A8 fails. §4 falsifies that. Writing
it under that title would be forcing the story, which goal-5 explicitly
forbids.

The evidence supports a different and, in the end, more useful paper:

> **"Learning-Rate Collapse in Quantization-Aware RL: A Real Pathology That Is
> Not the Cause"**

with three results, each with its controlled comparison:

1. **The pathology.** Fake-quant makes the policy discontinuous in its weights.
   One 1e-5 step — the algorithm's own floor — produces KL 6.73e-02 against
   4.06e-07 in float. PPO's controller reads that as divergence and pins the
   rate at its floor for 2000/2000 iterations, against 10/2000 for the FP32
   control. The reward curve stays healthy throughout (28.64 vs 29.30), so
   only the learning-rate trace reveals it.
2. **A predictive criterion.** The noise floor at the LR floor separates the
   precision ladder exactly: W16A16 and W8A16 sit 70× and 12× below the
   controller's threshold, W8A8 sits 6.3× above it. One forward pass, no
   training run.
3. **The falsification.** Removing the collapse does not fix the policy — it
   makes it worse (0.501 → 0.206, below PTQ). Activation width is the binding
   constraint: W8A16 under the *normal* controller reaches 1.000 on all seven
   segments in **both** float and fake-quant execution, with yaw RMSE better
   than the FP32 control's.

Result 3 is what makes the paper worth writing. A paper reporting only 1 and 2
would have been wrong in exactly the way this project keeps finding: a
mechanism that explains a phenomenon is not the mechanism that produces it.

**Deployment recommendation is unchanged and now better supported.** W16A16
remains the verified deployment configuration. W8A16 is now demonstrated to be
trainable *and* quantizer-independent, so the only barrier to shipping it
remains the exporter's INT16-by-construction weight path.

**Next experiments.** §4 points directly at behavior preservation rather than
step size: the frozen-anchor residual and the FP32-teacher KL designs in
`goal4_experiment_designs.md`. Their stated precondition — that they must run
with a fixed learning rate — is now known to be **necessary but not
sufficient**, which is itself a result those designs should absorb.
