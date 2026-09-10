# Isolating the actual cause of W8A8 QAT control failure

Goal 5 left the project with a real pathology and no cause. W8A8 quantization
amplifies policy KL by five orders of magnitude, PPO's adaptive controller reads
that as divergence and pins the learning rate at its `1e-5` floor for 2000 of
2000 iterations — all measured, all reproducible. But removing the collapse by
fixing the learning rate did not restore turning, and made the policy *worse*
(mean success 0.501 → 0.206). Meanwhile W8A16 trained to 1.000 on every segment
under the ordinary controller.

So the learning-rate collapse is a symptom that travels with the disease, not
the disease. This document isolates what the disease is.

Every arm here shares `locomotion_v2`, seed 1, 4096 environments, 2000
iterations, the warm start `SOLID_FP32_V2.pt`
(`sha256 ab20c19071c4be2c…`), the calibrated `act_fracs_a8.json`,
`desired_kl = 0.005` (controller threshold `0.01`) and the learning-rate floor
`1e-5`. Arms differ from their named control in exactly one respect.

---

## 1. Hypotheses

**H1 — encoder drift.** The history encoder is optimized by a *separate* Adam
instance at a *fixed* learning rate that the adaptive KL controller never
touches. When W8A8 pins the actor at the floor, the encoder keeps moving at full
speed, and its output is an input to the actor. The actor would then be chasing a
non-stationary observation representation with a step size 100× too small to
follow it.

*Prediction.* Freezing the encoder removes the disturbance and turning returns.

**H2 — missing behaviour preservation.** PPO's objective — clipped surrogate,
value loss, entropy bonus — contains no term that refers to the warm-start
policy. A quantized student is therefore free to drift anywhere the reward
permits, and Goal 5 showed that giving it a *larger* step size let it drift
further and score worse.

*Prediction.* Adding `beta * KL(pi_teacher || pi_student)` against a frozen FP32
teacher restores turning.

**H3 — W8A8 is intrinsically insufficient.** If neither intervention rescues
turning, the failure is a property of 8-bit activations in this architecture and
this QAT formulation, not of the optimizer or the objective.

*Prediction.* Both interventions fail, and W8A16 remains the minimal change that
works.

These are mutually exclusive as stated only if one intervention succeeds
outright. Partial success from both is the case Experiment 3 exists for.

---

## 2. Code audit: how the actor and the encoder are actually optimized

Read from the frozen upstream at
`plane/wheel_legged_gym/rsl_rl/`, which is imported by path and never modified.

### 2.1 Two optimizers, one controller

`algorithms/ppo.py` constructs **two** independent Adam instances:

| | parameters | learning rate | adjusted by the KL controller |
|---|---|---|---|
| `self.optimizer` | `actor` + `critic` + `std` | `algorithm.learning_rate`, initial `1e-3` | **yes** |
| `self.extra_optimizer` | `encoder` only | `algorithm.extra_learning_rate`, **fixed `1e-3`** | **no** |

The controller is this, and it writes to one of them:

```python
if kl_mean > self.desired_kl * 2.0:
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
...
for param_group in self.optimizer.param_groups:      # <- self.optimizer only
    param_group["lr"] = self.learning_rate
```

`locomotion_v2` inherits `extra_learning_rate = 1e-3` from
`LeggedRobotCfgPPO.algorithm` — no task config on this path overrides it. So in
the W8A8 run the actor stepped at `1e-5` and the encoder stepped at `1e-3`
for all 2000 iterations: **a 100:1 ratio, created entirely by the controller
regulating one optimizer and not the other.**

### 2.2 The policy gradient never reaches the encoder

`modules/actor_critic_robust.py`:

```python
def update_distribution(self, observations, observation_history):
    self.latent = self.encoder(observation_history)
    mean = self.actor(torch.cat((observations, self.latent.detach()), dim=-1))
```

The `.detach()` is upstream behaviour, reproduced in
`QuantActorCriticSequence`. Consequences:

* the encoder is trained *only* by the auxiliary regression onto
  `base_lin_vel` (plus observation denoising), never by reward;
* nothing in the PPO objective can push back on an encoder change that hurts
  the policy;
* the encoder is, from the actor's point of view, an **exogenous, unregulated,
  non-stationary input**.

### 2.3 There is no guard on this path

`ppo.py` supports an `encoder_max_policy_kl` guard that reverts encoder updates
which move the policy too far. It is enabled in `locomotion_guard` and
`commanded_jump` — and **not** in `locomotion_v2`, where it defaults to `None`.
Every QAT arm in Goals 4, 5 and 6 ran with the guard off, exactly as the FP32
control did.

### 2.3.1 The encoder's disturbance is invisible to the only regulator

This is structural, not incidental. In `PPO.update()` the minibatch loop that
measures `kl_mean` and adjusts the learning rate runs at lines 216–325; the
encoder block runs at 330 onward. Within any single `update()` call the encoder
is frozen — both the rollout that produced `old_mu_batch` and every forward pass
in the loop use the same encoder state. The encoder then steps once, at the end,
after the controller has already made its decision and will not look again.

So the adaptive controller does not throttle the encoder, and it also never
*sees* the encoder's effect. `Encoder/policy_kl` is computed a few lines later,
logged, and fed to nothing. The encoder is not merely unregulated; it is outside
the loop that regulates anything.

This predicts that freezing the encoder will not relieve the learning-rate
collapse. §3 confirms it: with every encoder parameter frozen, the W8A8 run
under the ordinary controller still sits at the `1e-5` floor for 100% of
iterations, with a measured PPO KL median of 0.087.

### 2.4 The disturbance is already measured, in every run

`ppo.py` logs `Encoder/policy_kl`: the policy KL induced by the encoder update
*alone*, probed before and after the auxiliary optimizer's step on the same
states. It has been in every `metrics.jsonl` this project has ever written.

| arm | median | p90 | max | frac > 0.01 | frac > 1.0 |
|---|---|---|---|---|---|
| FP32 control | 0.0024 | 0.0071 | 0.091 | 0.045 | 0.000 |
| W8A16, adaptive | 0.0430 | 0.118 | 0.610 | 0.983 | 0.000 |
| **W8A8, adaptive (baseline)** | **0.9478** | 2.272 | 16.96 | 1.000 | **0.475** |
| W8A8, fixed LR | 0.5296 | 1.295 | 17.81 | 1.000 | 0.201 |

The actor's entire per-iteration KL budget is `0.01`. Under W8A8 the encoder
update alone moves the policy by a median KL of **0.95 — about 95× that budget,
399× the FP32 control** — and does so on every one of 2000 iterations. In the
FP32 control the same quantity sits *below* `desired_kl` and is negligible.

### 2.5 Why the same encoder movement is not equally harmful in every arm

Encoder *displacement* does not separate the working arm from the broken one
(§7): W8A16 and W8A8 move their encoders by an almost identical amount. What
separates them is how much policy KL a given encoder step produces.

`scripts/kl_amplification.py --stepped encoder` measures exactly that: one Adam
step of size `lr` applied to the encoder block only, same weights, same 4096
calibration states, quantizer on and off.

| encoder step | float | W16A16 | W8A16 | W8A8 | W4A8 |
|---|---|---|---|---|---|
| `1e-6` | 6.26e-9 | 3.52e-4 | 8.70e-4 | 5.99e-2 | 0.0 |
| `1e-5` | 6.27e-7 | 5.83e-4 | 1.86e-3 | 3.15e-1 | 1.29e-1 |
| `1e-4` | 6.27e-5 | 9.25e-4 | 1.05e-2 | 1.09 | 4.98e-1 |
| `2.56e-4` | 4.11e-4 | 1.55e-3 | 1.48e-2 | 1.40 | 1.74 |
| **`1e-3`** (the real encoder LR) | **6.26e-3** | 7.50e-3 | **4.65e-2** | **2.12** | 5.36 |

At the learning rate the encoder actually runs at, one step produces policy KL
**2.12 under W8A8 against 0.0063 in float — 339× — and 212× the controller's
own `0.01` threshold.** W8A16 sits at 0.0465: still 4.6× over threshold, but 46×
below W8A8.

This offline instrument and the in-training `Encoder/policy_kl` of §2.4 are
independent measurements of the same quantity, and they agree in ordering and
within about 2× in magnitude across three arms. That is the mechanistic bridge
H1 needs: the encoder's *step size* is identical in every arm because nothing
regulates it, and the *policy disturbance* that step produces scales with
activation width.

### 2.5.1 What the audit does **not** establish

All of the above is correlational. It shows a large, unregulated, quantization-
amplified disturbance exists on a path that plausibly matters. It does not show
that removing it recovers turning — Goal 5's whole lesson was that a real
pathology need not be the cause. That is what Experiment 1 tests.

### 2.6 How the freeze is implemented

Upstream already supports the intervention: `ppo.py` reads
`actor_critic.encoder_frozen` **in its constructor** to decide whether to build
`extra_optimizer` at all, and refuses a nonzero auxiliary learning rate when the
flag is set. `--freeze-encoder` therefore changes nothing upstream; it sets that
attribute on the class the runner is about to instantiate (setting it afterwards
would be a silent no-op — the optimizer would already exist) and sets
`extra_learning_rate = 0.0`.

Four independent facts are asserted after construction, because any one can hold
while the encoder still moves:

```
encoder freeze VERIFIED: encoder_frozen=True, extra_optimizer=None,
24579 encoder params requires_grad=False, extra_learning_rate=0.0
```

and a runtime guard re-hashes all encoder parameters around the first ten
updates and every hundredth thereafter, aborting the run on any change — a
start/end comparison cannot tell "never moved" from "moved and came back", and
reports 2000 iterations too late. A matched negative control (identical command,
flag omitted) confirms the digest is sensitive:

| run | encoder digest after 3 updates | actor |
|---|---|---|
| `--freeze-encoder` | `f6ca963c…` **unchanged** | changed |
| flag omitted | `ceb113c8…` **changed** | changed |

### 2.7 Encoder step versus actor step, at the learning rates each actually runs at

The two blocks are not comparable by amplification factor alone, because they
run at different learning rates. `--stepped actor` gives the matched control on
the same checkpoint and the same states:

| step size | block | float | W16A16 | W8A16 | W8A8 | W4A8 |
|---|---|---|---|---|---|---|
| `1e-5` | actor | 1.16e-6 | 3.17e-4 | 1.69e-3 | 8.18e-2 | 3.15e-2 |
| `1e-3` | encoder | 6.26e-3 | 7.50e-3 | 4.65e-2 | **2.12** | 5.36 |

In float the encoder is the *less* influential block per unit step (0.54× the
actor's KL at every step size tested). Under W8A8 that inverts to 3.3–4.6×.
Combined with the 100:1 learning-rate ratio the controller creates, the
operating point is:

> **Under W8A8, per iteration, the encoder injects about 26× more policy KL
> than the actor is permitted to take** (2.12 against 0.0818). In the FP32
> control the same comparison is 0.0063 against 0.00076 — the encoder is
> larger, but both are near or below the controller's `0.01` threshold and the
> policy is not being dragged anywhere.

### 2.8 The encoder's own task degrades under A8

The evaluator reports `velocity_estimation_rmse_m_s`, the encoder's error on the
quantity it is trained to regress, measured on the turning segments:

| arm | turn L | turn R |
|---|---|---|
| FP32 control | 0.0051 | 0.0056 |
| QAT W8A16 | 0.0071 | 0.0081 |
| QAT W8A8 adaptive | 0.0571 | 0.0519 |
| QAT W8A8 fixed-LR | 0.0994 | 0.0716 |

The latent is quantized at `enc2` fractional bits = 3, so one latent LSB is
`2^-3 = 0.125` in latent units — the encoder cannot represent its target more
finely than that, and its measured error under W8A8 is an order of magnitude
worse than in float. An encoder chasing a target it cannot represent does not
converge; it keeps stepping. That is the same picture from the other side.

### 2.9 Where the finished policies actually ended up — block attribution

The retraining experiments ask what happens when the encoder is not allowed to
move. The complementary question can be answered on the runs that already exist:
of the total action-mean change a run produced, how much is carried by each
block? `scripts/goal6_block_attribution.py` builds four policies from two
checkpoints by swapping whole blocks — legitimate here precisely because the
latent is detached, so encoder and actor are separately parameterized and the
mixed policies are well-formed rather than chimeric — and measures KL against
the shared warm start on 4096 identical states, in each arm's own execution
precision, with `std` held at the warm start's value.

| arm | execution | both blocks | actor only | encoder only | encoder/actor |
|---|---|---|---|---|---|
| FP32 control | float | 4.06 | 4.11 | 0.70 | 0.17 |
| QAT W8A16 | W8A16 | 2.96 | 1.60 | 1.36 | 0.85 |
| QAT W8A8 adaptive | W8A8 | **33.69** | 27.82 | 24.38 | 0.88 |
| QAT W8A8 fixed-LR | W8A8 | 31.18 | 19.21 | 26.76 | 1.39 |

Three things follow.

1. **The FP32 control's behaviour change is almost entirely the actor's**
   (encoder/actor = 0.17). Under W8A8 the encoder carries as much of it as the
   actor does. So the encoder's role is not a constant of the setup — 8-bit
   activations promote it from a minor contributor to an equal one.

2. **The W8A8 runs end up ~8× further from the warm start than the FP32 control
   does** (33.7 against 4.06) while scoring 0.501 against 1.000. More drift,
   worse policy. This is what an objective with no reference term looks like
   when the function it is optimizing has become hypersensitive to its own
   parameters.

3. **The actor alone accounts for KL 27.8 despite sitting at the `1e-5`
   learning-rate floor for all 2000 iterations.** A pinned learning rate is not
   a pinned policy: under fake-quant, 2000 sub-LSB steps still accumulate into
   an enormous functional change whenever they cross code boundaries.

**Prediction recorded before Experiment 1 finished.** If the drift is split
roughly evenly between the blocks, freezing the encoder should remove
approximately half of the disturbance and should *not* by itself restore
turning — and it should not relieve the learning-rate collapse at all, because
§2.7 shows the actor's own `1e-5` step already produces KL 0.082, eight times
the controller's threshold, with the encoder held still. H1 as stated would then
be partially supported and insufficient, and H2 would become the live
hypothesis.

---

## 3. Experiment 1 — freeze the encoder

Two arms, each differing from an existing control in exactly one respect: the
encoder is frozen. `g6_e1_frzenc_fixlr` is `qat_v2_fixlr` plus the freeze;
`g6_e1_frzenc_adapt` is the goal-5 W8A8 baseline plus the freeze, added because
the fixed-LR control is itself a degraded arm and an intervention should be
testable against the ordinary controller too.

### 3.1 The intervention did exactly what it claimed

| | encoder ‖Δ‖ | encoder SHA-256, iter 0 → 2000 | actor ‖Δ‖ |
|---|---:|---|---:|
| W8A8 adaptive, control | 66.69 | changed | 2.51 |
| **+ frozen encoder** | **0.00** | `f6ca963c…` → `f6ca963c…` | 2.90 |
| W8A8 fixed-LR, control | 62.71 | changed | 11.81 |
| **+ frozen encoder** | **0.00** | `f6ca963c…` → `f6ca963c…` | 12.66 |

Bit-identical, not approximately unchanged, and checked around thirty separate
updates rather than only at the end. The actor moved slightly *more* with the
encoder held still, which is the expected sign: it is no longer chasing.

### 3.2 The learning-rate collapse is untouched

| | lr median | iterations at the `1e-5` floor | median PPO KL |
|---|---:|---:|---:|
| W8A8 adaptive, control | 1.000e-05 | 100% | 0.0650 |
| **+ frozen encoder** | 1.000e-05 | **100%** | 0.0491 |
| W8A8 fixed-LR, control | 2.563e-04 | 0% | 0.0665 |
| **+ frozen encoder** | 2.563e-04 | 0% | 0.0557 |

Removing a disturbance of median KL 0.95 per iteration lowered the KL the
controller measures by 0.016 and moved the floor fraction by nothing at all.
This is §2.3.1 confirmed by intervention: the controller measures KL inside the
minibatch loop, the encoder steps after it, and the two never meet. Whatever
the encoder was doing, it was not what pinned the learning rate.

### 3.3 Turning did not come back

Arm B, the deployed fake-quant datapath:

| arm | forward | reverse | turn_L | turn_R | stop | height | combined | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 control | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| W8A8 adaptive, control | 1.000 | 0.570 | 0.000 | 0.000 | 0.938 | 1.000 | 0.000 | **0.501** |
| **+ frozen encoder** | 0.680 | 0.062 | **0.000** | **0.000** | 0.898 | 0.938 | 0.000 | **0.368** |
| W8A8 fixed-LR, control | 0.984 | 0.438 | 0.000 | 0.000 | 0.023 | 0.000 | 0.000 | **0.206** |
| **+ frozen encoder** | 0.922 | 1.000 | **0.000** | **0.000** | 1.000 | 0.000 | 0.000 | **0.417** |

> **H1 is falsified.** `turn_left` and `turn_right` are exactly 0.000 in both
> frozen arms, as they are in both controls. Freezing the encoder — completely,
> verifiably, for 2000 iterations — changes nothing about the behaviour the
> project set out to recover.

Under the ordinary controller the freeze made the arm *worse* (0.501 → 0.368).
Under fixed LR it helped everything except turning (0.206 → 0.417, with
`reverse` 0.438 → 1.000 and `stop` 0.023 → 1.000), which says the freeze does
remove a real source of drift — just not the one that costs turning.

### 3.4 And the policies became more quantizer-dependent, not less

Arm A, the same parameters executed in float:

| arm | mean success, arm A | mean success, arm B |
|---|---:|---:|
| W8A16 QAT | 1.000 | 1.000 |
| W8A8 fixed-LR, control | 0.143 | 0.206 |
| **+ frozen encoder** | **0.055** | 0.417 |
| **W8A8 adaptive + frozen encoder** | **0.002** | 0.368 |

The frozen-adaptive arm scores 0.002 in float against 0.368 under its own
quantizer. It has not learned a policy that tolerates quantization; it has
learned one that *requires* it, more completely than any previous arm.

### 3.5 The encoder drift was partly repair, not only disturbance

`velocity_estimation_rmse_m_s` on the turning segments — the encoder's own
regression target:

| arm | turn L | turn R |
|---|---:|---:|
| FP32 control | 0.0051 | 0.0056 |
| PTQ W8A8, untrained encoder | 0.1423 | 0.1499 |
| W8A8 adaptive, encoder free to move | 0.0571 | 0.0519 |
| **W8A8 adaptive, encoder frozen** | **0.1785** | **0.1772** |

Freezing pins the encoder at its post-quantization value, and the frozen arms
sit at the PTQ error level. The unfrozen arms are 3× better, so the encoder
drift §2 measured was not only injecting disturbance — it was also *recovering*
most of the velocity-estimation accuracy that 8-bit activations destroyed.

That reframes §2.9's reading. The encoder does move a long way, and that motion
does produce a large unregulated policy KL. But it is doing useful work while
it does so, and removing it costs more than it saves. A disturbance and a
repair can be the same signal.

---

## 4. Experiment 2 — FP32-teacher behaviour preservation

### 4.1 The specified objective is degenerate, and the run proved it

goal6 specifies `L = L_PPO + beta * KL(pi_teacher || pi_student)` with
beta = 1.0. Run exactly as written, that arm does not test H2.

The full Gaussian KL between two diagonal Gaussians is

```
KL(t || s) = log(s_s/s_t) + (s_t^2 + d^2) / (2 s_s^2) - 1/2      d = mu_t - mu_s
```

and minimising it over the student's own sigma gives `s_s^2 = s_t^2 + d^2`. A
student that *cannot* match the teacher's mean can therefore shrink the
objective by widening its action distribution instead. The teacher term does
not have to be satisfied by imitation; it can be satisfied by getting vaguer.

That is what happened, within 97 iterations:

| iteration | exploration std | teacher KL | mean term's share |
|---:|---:|---:|---:|
| 1 | 0.2871 | 17.221 | 1.000 |
| 25 | 0.3841 | 2.542 | 0.793 |
| 49 | 0.4462 | 3.134 | 0.641 |
| 97 | 0.4601 | 3.357 | 0.587 |

The std rose **1.60×** while the KL plateaued near 3.0, and the fraction of
that KL attributable to the mean mismatch fell from 1.000 to 0.587. The policy
was not moving toward the teacher; it was spreading out around its own mean.
The arm is preserved at `checkpoints/g6_e2_teacher_fullkl_DEGENERATE/`.

### 4.2 The correction, and why it costs nothing

Hold sigma at the teacher's value and penalise the mean alone:

```
L_teacher = beta * sum_a (mu_t - mu_s)^2 / (2 * s_t^2)
```

This is the mean-matching half of the same KL with the escape hatch removed,
and it gives up nothing that matters. The exploration std is a **training-only**
parameter — it is explicitly listed among the tensors that are never deployed,
and `act_inference`, the function the FPGA and the Cortex-M7 both reproduce,
returns the mean. Preserving the teacher's *behaviour* means preserving its
mean; preserving its exploration noise is preserving an artefact of PPO.

`--teacher-kl-mode full` still reproduces the degenerate objective, because a
defect that can no longer be demonstrated stops being evidence.

Both modes are verified before any run starts by
`quantization/scripts/check_teacher_kl.py`, which compares the gradient of the
injected objective against a directly constructed `L_PPO + beta * L_teacher`
and requires them to match elementwise. Both pass with a maximum absolute
gradient difference of 0.0, and both confirm beta = 1.0 is not a no-op.

### 4.3 The correction reduced the inflation without removing it

Exploration std over the first 89 iterations, all three arms sharing the same
warm start, learning rate and seed:

| iteration | control, no teacher | teacher, full KL | teacher, mean only |
|---:|---:|---:|---:|
| 1 | 0.2863 | 0.2871 | 0.2859 |
| 41 | 0.3239 | 0.4339 | 0.3585 |
| 81 | 0.3231 | 0.4599 | 0.4082 |
| **growth** | **×1.13** | **×1.60** | **×1.48** |

The mean-only objective has no gradient path to the student's sigma at all, so
the residual growth is not the escape hatch — it is the ordinary entropy bonus
winning more often. A teacher term this strong drags the action mean far enough
per update that many samples land outside PPO's clip range, which zeroes the
surrogate's counter-pressure on sigma and leaves the entropy bonus unopposed.

That is a side effect of beta = 1.0 being large, not a second degeneracy, and
it does not touch the evaluation: `act_inference` returns the mean, and sigma
is never deployed. It is recorded because a reader comparing the two arms will
see the number and should not have to guess which mechanism produced it.

### 4.4 Result

`g6_e2_teacher_fixlr` completed 2000/2000 with the corrected objective, fixed
learning rate 2.563e-4, 0% of iterations at the floor.

| | teacher KL |
|---|---:|
| at warm start (PTQ) | 17.509 |
| best reached | 2.916 |
| median over the last 200 iterations | **6.708** |
| floor from supervised distillation (§5.3) | **0.317** |

The teacher term worked — KL fell from 17.5 and the gradient it contributed was
verified at 5,678× the entropy term — and it never came close to the floor that
distillation alone reaches. PPO holds the student roughly **21× further from the
teacher** than the same objective achieves with PPO removed.

Closed-loop:

| arm | fwd | rev | turn_L | turn_R | stop | height | comb | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-LR control, no teacher | 0.984 | 0.438 | 0.000 | 0.000 | 0.023 | 0.000 | 0.000 | 0.206 |
| **+ FP32 teacher, beta = 1.0** | 0.258 | 0.000 | **0.000** | **0.000** | 0.828 | 0.000 | 0.000 | **0.155** |
| the same arm in float (A) | 0.000 | 0.000 | 0.000 | 0.000 | 0.023 | 0.000 | 0.000 | 0.003 |

> **H2 is falsified.** Explicitly anchoring the student to the FP32 policy did
> not restore turning, and left the arm below the control it modifies. The
> prediction recorded in §5.4 before this run — that Experiment 2 could not
> drive the teacher KL below roughly 0.32 — held with a large margin.

---

## 5. Experiment 3, and the measurement that replaced it

### 5.1 Why the combination arm is not justified

goal6 reserves Experiment 3 — frozen encoder *plus* teacher KL — for the case
where each intervention *partially* improves the result. That is not what
happened. Freezing the encoder left `turn_left` and `turn_right` at exactly
0.000, the same value both controls produce. There is no partial recovery to
combine with anything, so the arm is not run.

### 5.2 H3 deserved a measurement, not an elimination

goal6 reaches H3 by exhaustion: if neither intervention rescues turning, the
failure "may be intrinsic to W8A8". That is weak evidence for the claim that
decides the deployment baseline, and it is weak for a specific reason — an RL
run confounds three things that fail identically from the outside:

* what the quantized network **can represent**,
* what the PPO objective **asks for**,
* what the adaptive controller **lets the optimizer do**.

Experiments 1 and 2 both operate on the second and third. Neither can see the
first. So the first was measured on its own.

`quantization/scripts/goal6_capacity_probe.py` removes the RL entirely. The
frozen FP32 policy is the teacher; a fake-quant student initialised from the
same weights is fitted to its action mean by plain supervised regression on
stored on-policy observations, with gradients reaching the quantized weights
through the straight-through estimator exactly as in QAT. No reward, no PPO,
no controller, no exploration. 80/20 train/holdout split, encoder and actor
trained jointly, best holdout error over the run.

> capacity = how close supervised distillation can get.
> Everything QAT adds sits on top of that floor.

### 5.3 The floor

24,000 Adam steps, batch 2,048, cosine-decayed learning rate:

| arm | KL at PTQ | best achievable KL | vs PPO's 0.01 threshold |
|---|---:|---:|---:|
| float | 0 | 0 | 0.0× |
| W16A16 | 0.0069 | 0.0043 | 0.4× |
| W8A16 | 0.3347 | **0.0096** | **1.0×** |
| W8A8 | 14.276 | **0.3167** | **31.7×** |
| W4A8 | 56.72 | 0.6369 | 63.7× |

And the descent is finished, not merely slow. W8A8's holdout KL by step:

```
     0   2880   5760   8640  11520  14400  17280  20160  23040
14.276 0.4143 0.3457 0.3167 0.3410 0.3685 0.3582 0.3483 0.3441
```

Flat from step 2,880 onward, oscillating around 0.34. Running six times longer
than the first pass bought 24%, and the curve turns back up as often as it goes
down. This is a floor, not an unfinished descent.

> **H3 is confirmed, directly.** W8A16 can be fitted to *inside* the
> learning-rate controller's own threshold. W8A8 cannot get within **32×** of
> it, with the teacher as the only objective and nothing else in the way. The
> W8A8 datapath cannot represent this policy closely enough for PPO to regard
> it as converged, before any reinforcement learning is attempted.

### 5.4 This bounds Experiment 2 in advance

Supervised distillation is the `beta → infinity` limit of Experiment 2's
objective: the teacher term alone, with PPO contributing nothing. Its floor is
0.3167. Adding PPO back can only move the student *away* from the teacher, so
**Experiment 2 cannot drive the teacher KL below roughly 0.32**, and a policy
0.32 in KL from the reference is one the controller would treat as diverged on
every iteration.

Recorded here before Experiment 2 finished, so that the prediction is testable
rather than retrofitted.

---

### 5.5 What the floor is actually made of, and one way to lower it

The capacity result says W8A8 cannot represent this policy *at the activation
scales this project ships*. That is narrower than "8-bit activations cannot do
it", and the difference is worth chasing, because the scales were chosen by a
heuristic rather than optimized against anything.

`act_fracs_a8.json` picks each tensor's fractional bits from its calibrated
peak with 1.25× headroom. Comparing what each tensor got against what its own
range would allow:

| tensor | peak | frac used | frac its range allows |
|---|---:|---:|---:|
| enc0 | 12.81 | 2 | 2 |
| enc1 | 16.60 | 2 | 2 |
| **enc2** (latent) | **3.56** | **3** | **4** |
| act0 | 3.91 | 4 | 4 |
| act1 | 6.31 | 4 | 4 |
| act2 | 9.38 | 3 | 3 |
| act3 | 7.63 | 3 | 3 |
| obs | 7.63 | 3 | 3 |

Seven of eight tensors are at their limit. **The latent is not.** It is held one
bit coarser than its range permits, and the reason is architectural rather than
numerical: PL's CONCAT cannot requantize, so the latent must arrive on the same
grid as `obs`, and `obs` has a peak of 7.63 that forces frac 3.

That is the tensor this study has repeatedly landed on. It is the encoder's
velocity estimate; quantizing it takes the velocity RMSE from 0.0051 to 0.1423
(§2.8, §3.5); and it enters the actor through the one operator in the ISA that
cannot rescale its inputs. One bit of avoidable precision loss sits exactly
where the measurements say the damage is.

#### The fix needs no hardware change

The latent's grid is fixed, but its *magnitude* is not. Multiply `encoder.4`'s
weight and bias by `S`, and divide the trailing `latent_dim` columns of
`actor.0`'s weight — the ones that consume the latent — by the same `S`. In
FP32 this is exactly the identity. Under quantization the latent now occupies
`S` times more of the same integer grid, so its effective resolution improves
by `S` at an unchanged fractional-bit setting.

This is cross-layer equalization applied to the one boundary the hardware
constrains, and it touches only exported weights: no RTL, no descriptor format,
no exporter change, no new operator. The headroom bounds it —
`3.56 · S · 1.25 ≤ 127/2³` gives `S ≤ 3.57`.

It also has a cost that has to be measured rather than assumed: `actor.0`'s
weight scale is set per-GEMM from its largest magnitude, so shrinking the latent
columns by `S` spends weight precision on them to buy activation precision. The
capacity probe is the right instrument for that trade because it scores a
configuration in minutes with no RL in the loop.

### 5.6 The rescaling fix was tested and does not work

| latent scale S | best achievable KL, W8A8 |
|---:|---:|
| 1.0 (shipped) | 0.4155 |
| 2.0 | 0.4018 |
| 3.0 | 0.4206 |

A 3.3% improvement at S = 2, and *worse* at S = 3. Non-monotonic and
negligible against a floor that needs to fall by 32×. The hypothesis was
reasonable and it is wrong: the precision gained on the latent is paid for in
`actor.0`'s weight scale, which is set per-GEMM from its largest magnitude, and
the two roughly cancel.

More usefully, it localises the problem. The floor is **not** concentrated in
the latent. Counting how much of the INT8 range each tensor actually occupies:

| tensor | peak | frac | LSB | representable range | levels used of 255 |
|---|---:|---:|---:|---:|---:|
| enc0 | 12.81 | 2 | 0.2500 | 31.75 | 51 |
| enc1 | 16.60 | 2 | 0.2500 | 31.75 | 66 |
| enc2 | 3.56 | 3 | 0.1250 | 15.88 | 28 |
| act0 | 3.91 | 4 | 0.0625 | 7.94 | 62 |
| act1 | 6.31 | 4 | 0.0625 | 7.94 | 100 |
| act2 | 9.38 | 3 | 0.1250 | 15.88 | 75 |
| act3 | 7.63 | 3 | 0.1250 | 15.88 | 61 |
| obs | 7.63 | 3 | 0.1250 | 15.88 | 61 |

**No tensor uses even 40% of the range INT8 offers, and one uses 11%.** That is
the cost of a power-of-two scale: the requantizer is an arithmetic shift, so the
scale itself is quantized to powers of two and up to a full bit of range is
stranded, which the 1.25× headroom then compounds.

Recovering it by rescaling weights only works across a *linear* boundary, and
ELU sits between every pair of layers except one. The encoder→actor boundary is
the single rescalable seam in this network, which is why it was the one tested —
and it is worth 3%.

So the loss is structural given shift-based requantization and ELU. Removing it
properly means an arbitrary-scale multiplier in the requantizer, which is an RTL
change and out of scope here. What remains in software is the choice of
fractional bits itself.

### 5.7 Trading clipping for resolution does work

The remaining software lever is the fractional-bit choice itself.
`calibrate_act_fracs.py` sets each tensor from its calibrated **peak** with 1.25×
headroom, so a single outlier sample can cost a whole bit for every sample. The
standard alternative in post-training quantization is to clip at a percentile
instead — accept a little saturation, halve the step size everywhere else.

How much magnitude each tensor would have to give up to earn one more bit:

| tensor | peak | frac | needs peak ≤ | magnitude clipped |
|---|---:|---:|---:|---:|
| **enc0** | 12.81 | 2 → 3 | 12.70 | **0.9%** |
| **enc2** | 3.56 | 3 → 4 | 6.35 | **0%** (pinned by CONCAT, not by range) |
| act3 | 7.63 | 3 → 4 | 6.35 | 16.8% |
| obs | 7.63 | 3 → 4 | 6.35 | 16.8% |
| enc1 | 16.60 | 2 → 3 | 12.70 | 23.5% |
| act2 | 9.38 | 3 → 4 | 6.35 | 32.3% |
| act1 | 6.31 | 4 → 5 | 3.17 | 49.7% |

Scored with the capacity probe, 4,000 steps each, W8A8 only:

| assignment | KL at PTQ | best achievable KL | vs 0.01 threshold |
|---|---:|---:|---:|
| shipped (peak + 1.25×) | 14.28 | 0.4155 | 41.5× |
| `enc0`=3 alone | 12.02 | 0.4330 | 43.3× |
| `enc0`=3, `obs`/`enc2`=4 | 9.14 | 0.3327 | 33.3× |
| + `act3`=4 | 8.62 | 0.2813 | 28.1× |
| + `enc1`=3, `act2`=4 | **4.88** | **0.2478** | **24.8×** |

Two results, pointing different ways.

**The scales the project ships are not the best available ones.** The
representational floor falls 40%, and the PTQ error — what you get with *no
retraining whatsoever* — improves **2.9×**, from KL 14.28 to 4.88. The
calibration heuristic is leaving that on the table, and the fix is a change to
one script.

**And it does not rescue W8A8.** Even the best assignment sits 24.8× above the
threshold that decides whether the controller regards a policy as converged.
Better scales move the floor from "hopeless" to "still hopeless", which sharpens
§5.3 rather than overturning it: the W8A8 gap is not an artefact of a lazy
calibration rule.

Note also that `enc0`=3 *alone* is slightly worse than the baseline (0.4330 vs
0.4155). The assignment interacts, so tensors cannot be tuned one at a time —
which is exactly why a fast RL-free scorer is worth having.

### 5.8 Better scales recover the behaviour that QAT never did

The scale search was run to sharpen §5.3. It did something larger.

Applying the tuned assignment as **plain post-training quantization** — no QAT,
no retraining, no fine-tuning of any kind — to the same trained FP32 control,
scored by the same evaluator on the same scenarios:

| arm | fwd | rev | turn_L | turn_R | stop | height | comb | **mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 control | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| PTQ W8A8, shipped scales | 0.000 | 0.000 | 0.758 | 0.523 | 0.984 | 0.539 | 0.000 | **0.401** |
| **PTQ W8A8, tuned scales** | 1.000 | 0.008 | **1.000** | **1.000** | 1.000 | 1.000 | 1.000 | **0.858** |
| QAT W8A8, best arm in goals 4–6 | 1.000 | 0.570 | 0.000 | 0.000 | 0.938 | 1.000 | 0.000 | 0.501 |

Yaw RMSE on the turning segments falls from 0.1465 / 0.1444 to 0.0866 / 0.0971.

> **Turning comes back.** `turn_left` and `turn_right` are 0.000 in every QAT
> arm this project has run — three seeds in goal 4, the fixed-LR and W8A16 arms
> in goal 5, both frozen-encoder arms in goal 6. Changing eight integers in a
> calibration file returns them to **1.000**, with no training at all.

Only `reverse` remains broken (0.008).

### 5.8.1 Two confounds, checked rather than assumed

**The evaluator changed between runs.** The published 0.401 was produced by
evaluator `ae47970433e6`; the new number by `10fe85451096`, because the upstream
harness is under active development. Attributing an evaluator change to a
calibration change would have been wrong, so the shipped-scale arm was re-scored
under the new evaluator: it returns **0.000 / 0.000 / 0.758 / 0.523 / 0.984 /
0.539 / 0.000**, identical to the published values. The evaluator update is
behaviourally neutral here and the comparison is clean.

**The evaluation seed does nothing.** Runs at `--seed 11`, `12` and `13` return
*bit-identical* summaries — SHA-256 `da31b272…` for all three shipped runs and
`b2a6e6fb…` for all three tuned ones — while each output faithfully records its
own `evaluation_seed`. `evaluate_locomotion.py` is deterministic given a
checkpoint: fixed command sequence, fixed initial states, and `act_inference`
returns the action mean with no sampling.

So an eval-seed sweep measures nothing, and reporting "three seeds" here would
have been **n = 1 presented as n = 3**. Independent variation for this evaluator
means independently *trained* policies, which is the axis goals 4 and 5 used.

### 5.8.2 The gain holds across independently trained policies

Since eval seeds vary nothing, robustness was measured the only way this
evaluator allows: the same two scale assignments applied to **three
independently trained FP32 control policies** — the seed-1, seed-2 and seed-3
controls from goal 4's multi-seed matrix — each scored by the same evaluator.

| policy | PTQ, shipped scales | PTQ, tuned scales | Δ |
|---|---:|---:|---:|
| control, training seed 1 | 0.401 | **0.858** | +0.458 |
| control, training seed 2 | 0.068 | **0.746** | +0.677 |
| control, training seed 3 | 0.385 | **0.857** | +0.472 |
| **mean** | **0.285** | **0.820** | **+0.535** |

Per-segment, `turn_left` and `turn_right` reach **1.000 on all three policies**
with the tuned scales, against 0.758/0.523, 0.000/0.203 and 0.875/0.680 with the
shipped ones. The seed-2 control is the clearest case: it collapses to 0.068
under the shipped scales and reaches 0.746 under the tuned ones.

The effect is large, consistent in sign across every policy, and larger than the
spread between policies. It is not a seed artefact.

---

## 6. A/B execution comparison

Every arm scored twice on identical scenarios: **A** executes the trained
parameters in float, **B** executes them through the fake-quant datapath they
were trained with. The gap between the two is the arm's *dependence* on the
quantizer, and it separates two things that a single score cannot.

| arm | A, float | B, fake-quant | B − A | reading |
|---|---:|---:|---:|---|
| QAT W8A16 | **1.000** | **1.000** | 0.000 | learned a policy that tolerates quantization |
| QAT W8A8, adaptive | — | 0.501 | — | |
| QAT W8A8, fixed LR | 0.143 | 0.206 | +0.063 | depends on the quantizer |
| QAT W8A8, frozen encoder, fixed LR | 0.055 | 0.417 | **+0.362** | depends on it heavily |
| QAT W8A8, frozen encoder, adaptive | 0.002 | 0.368 | **+0.366** | depends on it almost totally |

W8A16 is the only arm where A equals B. Every W8A8 arm scores worse in float
than through its own quantizer, and the two Experiment-1 arms are the most
extreme cases the project has produced: a policy that scores 0.002 without the
quantizer it was trained with is not a quantization-robust controller, it is a
controller that has folded the quantizer's error into its own weights.

This is the sharpest available evidence that W8A8 QAT is not converging on the
FP32 behaviour and then discretising it. It is converging on something else.

---

## 7. Parameter-drift measurements

All displacements are ‖Δ‖₂ against the shared warm start `SOLID_FP32_V2.pt`
(`ab20c19071c4be2c…`), computed by `quantization/scripts/goal6_table.py` from
each run's own final checkpoint. Encoder block: 24,579 parameters. Actor block:
14,246.

| arm | ‖Δ encoder‖ | rel | ‖Δ actor‖ | rel | median encoder→policy KL |
|---|---:|---:|---:|---:|---:|
| FP32 control | 26.49 | 0.705 | 4.29 | 0.247 | 0.0024 |
| QAT W8A16 | 66.16 | 1.760 | 1.75 | 0.101 | 0.0430 |
| QAT W8A8, adaptive | 66.69 | 1.775 | 2.51 | 0.145 | 0.9478 |
| QAT W8A8, fixed LR | 62.71 | 1.669 | 11.81 | 0.679 | 0.5296 |
| QAT W8A8, frozen enc, adaptive | **0.00** | 0.000 | 2.90 | 0.167 | — |
| QAT W8A8, frozen enc, fixed LR | **0.00** | 0.000 | 12.66 | 0.728 | — |

Three readings.

**Encoder displacement does not separate success from failure.** W8A16 works
and W8A8 fails, and their encoders move by 66.16 and 66.69 — a 0.8% difference.
Whatever distinguishes them, it is not how far the encoder travelled.

**A pinned learning rate is not a pinned policy.** The adaptive W8A8 arm sat at
the `1e-5` floor for all 2000 iterations and still moved its actor by ‖Δ‖ 2.51,
ending 33.7 in policy KL from the warm start (§2.9) — eight times further than
the FP32 control, which was free to take full-size steps the whole time. Under
fake-quant, 2000 sub-LSB steps accumulate into a large functional change
whenever they cross code boundaries.

**Freezing removed the drift and not the failure.** ‖Δ encoder‖ = 0.00 exactly,
and turning stayed at 0.000. Drift was real, measurable, and not the cause.

---

## 8. Causal conclusion

**H1 falsified. H2 falsified. H3 confirmed by direct measurement — and then
substantially narrowed.**

Every intervention this project has aimed at the *optimizer* or the *objective*
has failed to restore turning, and each failed with the encoder, the learning
rate, or the reference behaviour verifiably under control:

| intervention | what it fixed | turn_L / turn_R |
|---|---|---:|
| fixed learning rate (goal 5) | 0% of iterations at the floor | 0.000 / 0.000 |
| frozen encoder, adaptive | ‖Δ encoder‖ = 0.00 | 0.000 / 0.000 |
| frozen encoder, fixed LR | both of the above | 0.000 / 0.000 |
| FP32 teacher, beta = 1.0 | gradient 5,678× the entropy term | 0.000 / 0.000 |

They failed for a reason that none of them could have addressed. With RL removed
entirely — teacher as the only objective, 24,000 supervised steps, held-out
scoring — a W8A16 student fits the FP32 policy to KL **0.0096**, inside the
controller's own threshold, and a W8A8 student cannot get below **0.3167**, 32×
outside it. The constraint is representational. No learning rate, no encoder
policy and no auxiliary loss can move a floor that exists before learning
starts.

**But the floor is not where the shipped configuration puts it.** The activation
scales come from a peak-plus-headroom heuristic that leaves every tensor using
under 40% of INT8's range. Searching them against the same RL-free score lowers
the floor by 45% and the post-training error by **4.8×**, and applied as plain
PTQ with no retraining at all it takes closed-loop success from **0.285 to
0.820** averaged over three independently trained policies, returning
`turn_left` and `turn_right` to **1.000 on every one of them**.

So the causal chain is:

1. Eight-bit activations genuinely cannot represent this policy to the accuracy
   PPO's controller demands. That is real, measured without RL, and unmoved by
   every intervention tried.
2. **The shipped calibration made that limit look far worse than it is.** Most
   of the observed W8A8 control collapse — including all of the turning failure
   — was a calibration artefact, not a precision limit.
3. QAT was therefore attacking the wrong thing for the entire study. The
   behaviour it spent six goals failing to recover was available all along from
   eight integers in a JSON file.

The learning-rate collapse (goal 4), the KL amplification (goal 5) and the
encoder drift (goal 6 §2) are all real and all reproducible. None of them is the
cause.

---

## 9. Remaining uncertainties

**`reverse` is still broken under tuned scales** — 0.008, 0.000 and 0.000 on the
three policies, while every other segment reaches 1.000. It is the one behaviour
that neither better scales nor any QAT arm recovers, and it has not been
diagnosed.

**The scale search was greedy and is not converged.** Assignments were tried in
a hand-guided sequence, not optimized; `enc0`=3 *alone* is worse than baseline
while the same change inside a fuller assignment helps, so the coordinates
interact and a proper search would likely do better than 0.2275. No claim is
made that the tuned assignment is optimal.

**Clipping was never measured directly.** The tuned scales were chosen from peak
ratios, and the fraction of *samples* actually saturating at each setting was
not recorded. A percentile-based calibration would be the principled version.

**The capacity probe optimizes a proxy.** It fits the action mean on stored
observations. A policy matched in mean on the calibration distribution can still
diverge in closed loop, and the 0.858-vs-0.317 gap between the tuned PTQ result
and the distillation floor shows the two are not interchangeable.

**Everything downstream of the tuned scales is untested on hardware.** The
exporter emits INT16 weights by construction, so no W8A8 configuration —
shipped or tuned — has ever run on the FPGA or the Cortex-M7. These are
simulation results.

**One training seed for every goal-6 arm.** E1 and E2 are seed 1 only, per
goal6's instruction to expand only after a meaningful positive result. There was
none to expand on.

---

## 10. Does the technical-report story have to change?

**Yes, and more than goal 5 changed it.**

Goal 5 retracted the *mechanism*: the learning-rate collapse is real but is not
the cause. The story stayed "W8A8 collapses and QAT cannot fix it."

That headline is now wrong as stated. The repository's own precision table
reports W8A8 at 0.041 and describes a cliff at 8-bit activations. Under a
calibration the project never tried, the same arithmetic reaches **0.820 mean
across three policies with no training at all** — better than the best QAT arm
in six goals (0.501) and better than every W8A8 number this project has
published.

What must change:

* **"W8A8 collapses" becomes "W8A8 collapses at the shipped calibration."** The
  cliff is real but far shallower than reported, and the published figure
  measures a calibration choice as much as a precision limit.
* **The QAT negative result stands, and its explanation is replaced.** QAT does
  fail at W8A8. It fails because the target is not representable at that
  precision, not because PPO's controller misbehaves — and the gap it was trying
  to close was mostly not a precision gap at all.
* **The deployment recommendation needs revisiting.** W16A16 remains the
  shipped baseline and W8A16 remains the measured-lossless option. But W8A8 PTQ
  at 0.820 is no longer the non-option the tables imply, and it halves model
  bytes. The exporter cannot emit INT8 weights today, which is now the binding
  constraint rather than control quality.
* **`calibrate_act_fracs.py` should stop using peak-plus-headroom.** It is the
  single highest-value change identified in this study, it costs one script, and
  it needs no retraining and no RTL.

The honest one-line version, replacing the one in `docs/qat_failure/README.md`:

> QAT did not fail because PPO's controller misreads quantization noise — that
> was a real pathology and a false lead. It failed because 8-bit activations
> cannot represent this policy to the accuracy the controller demands. And most
> of the collapse that motivated the investigation was not a precision limit at
> all: it was an activation-scale heuristic that left three quarters of the
> integer range unused.

---

## 11. The fix, implemented

§10 named `calibrate_act_fracs.py` as the highest-value change this study
identified. It is now made rather than recommended.

`--percentile P` clips each tensor at the P-th percentile of `|activation|`
instead of its peak. Min-max remains the default, and reproduces the shipped
calibration **bit-for-bit** — verified as a regression, because a calibration
change that silently alters existing results would invalidate every number in
this repository.

The script now also reports what the clipping actually costs, which §9 listed as
an uncertainty:

| rule | tensors clipping any sample | worst-case samples clipped | INT8 levels used |
|---|---|---:|---:|
| min-max (shipped) | none | 0.000% | 28 – 100 |
| p99.9 | none | **0.000%** | 56 – 128 |
| p99.5 | 4 of 8 | **0.115%** | 56 – 128 |
| p99.0 | 5 of 8 | 0.188% | 56 – 128 |

p99.9 costs **nothing at all** — the 1.25× headroom still covers every observed
sample — and nearly doubles the range in use. The heuristic was not trading
accuracy for safety; it was paying for safety it already had.

**The principled rule rediscovers the hand-tuned assignment.** The best
assignment found by the guided search in §5.7 was
`{enc0:4, enc1:4, enc2:4, act0:5, act1:4, act2:4, act3:4, obs:4}`. p99.5 returns
`{enc0:4, enc1:4, enc2:4, act0:5, act1:5, act2:4, act3:4, obs:4}` — different in
one tensor, derived from a single percentile argument rather than from a dozen
scored experiments. That is the result worth keeping: not the specific integers,
but that a one-line rule change reaches them.

### 11.1 The principled rule beats the search that motivated it

Scored by the capacity probe, W8A8:

| rule | PTQ KL | best achievable KL | vs 0.01 threshold |
|---|---:|---:|---:|
| min-max (shipped) | 14.276 | 0.4155 | 41.5× |
| p99.9 | 5.041 | 0.3109 | 31.1× |
| **p99.5** | **2.614** | **0.2291** | **22.9×** |
| p99.0 | 2.231 | 0.2253 | 22.5× |
| guided search, best of twelve | 2.948 | 0.2275 | 22.8× |

And in closed loop, as plain PTQ with no retraining, across the same three
independently trained policies:

| rule | seed 1 | seed 2 | seed 3 | **mean** |
|---|---:|---:|---:|---:|
| FP32 control (upper bound) | 1.000 | 1.000 | 1.000 | **1.000** |
| shipped, min-max | 0.401 | 0.068 | 0.385 | **0.285** |
| guided search | 0.858 | 0.746 | 0.857 | 0.820 |
| **p99.5** | **0.985** | **0.875** | **0.857** | **0.906** |

**A 3.2× improvement over the shipped calibration, with no training of any
kind, from one argument.** The one-line rule also beats the twelve hand-scored
assignments that motivated it, which is the outcome to prefer: the finding is
the rule, not the integers.

Per segment, p99.5 reaches **1.000 on six of seven segments for every policy**:

| policy | fwd | rev | turn_L | turn_R | stop | height | comb |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 1 | 1.000 | 0.898 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed 2 | 1.000 | 0.125 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| seed 3 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

`reverse` is the whole remaining gap, and it is **policy-dependent** — 0.898,
0.125, 0.000 across three policies that are otherwise identical at 1.000. A
purely numerical limit would not vary that way between seeds, so `reverse` is
probably not, or not only, a quantization failure. §9's open question is now
sharper and smaller: it is one segment, on some policies.

### 11.2 Recommended default

`--percentile 99.5`. It matches the best hand-found assignment, beats it in
closed loop, clips **0.115%** of samples at worst, and needs no retraining, no
exporter change and no RTL. p99.9 is the zero-clipping option and still worth
1.3× on PTQ error; p99.0 buys almost nothing beyond p99.5 for more saturation.

Min-max stays the default in the script, because changing a calibration default
silently would invalidate every published number in this repository. The
recommendation belongs in the documentation and in the export recipe, not in a
flag flip.


---

## 12. Can any of this actually be deployed?

§11 recommends a calibration. Before that recommendation means anything, the
export path has to be able to express it. It cannot — and the reason is not the
one already recorded in `EVIDENCE.md`.

### 12.1 The exporter has no per-layer activation format

`fpga/tools/export_policy.py` defines `ACT_FRAC = 8` as a module constant and
uses it for every activation and every bias. Every tensor is Q8.8, so `f_in`
always equals `f_out`, and the GEMM descriptor is emitted as

```python
"shift": weight_frac
```

which is the correct value **only** under that assumption. The integer
reference, which is what the RTL and both Cortex-M7 kernels are verified
against, computes

```python
shift = f_in + f_w - f_out
```

Comparing the two for the two calibrations in question:

| layer | shipped Q8.8 shift needed | p99.5 shift needed | exporter emits |
|---|---:|---:|---:|
| enc0 | **6** | 5 | 5 |
| enc1 | 8 | 8 | 8 |
| enc2 | **8** | 9 | 9 |
| act0 | **5** | **5** | 6 |
| act1 | 7 | 7 | 7 |
| act2 | **9** | **9** | 8 |
| act3 | 8 | 8 | 8 |

The shipped W8A8 calibration needs **four of seven** shifts the exporter would
get wrong; p99.5 needs **two**. So this is not a limitation introduced by the
new calibration — **no calibrated activation configuration has ever been
exportable**, including the W8A8 and W8A16 rows this repository has published
since goal 4. Every one of them is a simulation result.

`EVIDENCE.md` records "cannot emit INT8 weights" as the blocker on shipping
W8A16. That is true and it is not the binding one. The binding one is that the
exporter cannot emit a per-layer activation format at all.

### 12.2 The hardware can express it; the toolchain cannot

Every shift required by either calibration is a small non-negative integer, and
the descriptor's shift field is 6-bit unsigned — range 0 to 63. Nothing about
these configurations is outside what the RTL already accepts.

The one genuine obstacle is ELU. `rl_elu_engine.sv` and `rl_elu_array8.sv` take
no shift input: the SFU ROM is addressed in Q8.8 and cannot be moved. An
activation on any other grid has to be converted into Q8.8 before the ELU and
back afterwards.

The ISA already has the instruction for that. `rl_vector_engine.sv` implements

```
AFFINE   dst = sat((src * a + b) >> shift)
```

which converts in either direction: a right shift for `f > 8`, or `a = 2^(8-f)`
for `f < 8`. Two AFFINE descriptors around each of the five ELUs takes the
program from 13 descriptors to **23**, against an instruction RAM depth of
**32**. It fits.

**So this is a toolchain limit, not a hardware limit** — the same verdict this
project already reached for element-wise ADD, arrived at independently.

### 12.3 What it would cost

*Estimated, not measured.* The vector engine processes one lane per cycle by
deliberate design, so an AFFINE over `N` elements costs about `N` cycles. The
five ELU sites carry 128, 64, 128, 64 and 32 elements, so ten conversions add
roughly `2 × 416 = 832` cycles.

| | cycles | at 100 MHz | share of the 10 ms control period |
|---|---:|---:|---:|
| today, Q8.8 only | 1,799 | 17.99 µs | 0.18% |
| estimated, per-layer fracs | ~2,631 | ~26.3 µs | ~0.26% |

A 46% increase in inference latency, on a budget where the current design uses
under a fifth of one percent. That is not the constraint.

### 12.4 What would have to change

1. Thread `act_fracs` through `export_policy.py`; emit `f_in + f_w - f_out`.
2. Quantize each bias at its layer's `f_out` rather than at `ACT_FRAC`.
3. Emit AFFINE pairs around each ELU when `f_out != 8`.
4. Re-run the full chain required by `docs/07_verification_guide.md`: re-export,
   Python golden, and the policy testbench, since this changes the descriptor
   program.

None of this is implemented or tested here. It is scoped, not done — and the
sizing above is arithmetic from the RTL, not a measurement.

