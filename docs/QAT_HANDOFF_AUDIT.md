# QAT handoff audit

Required by `goal.md` §0. Everything below was read from the Codex worktree's
source and artifacts, not from the previous Fudan-era notes. Audited
2026-09-07 22:0x local, while the Codex pass was still running (see
"Handoff is partial" at the end).

## 1. What the Codex pass is

| | |
|---|---|
| Worktree | `/home/as/vllm/rm-cortex-references/wheel-legged/fudan_rl_wheel_leg/.worktrees/robust-training` |
| Branch | `robust-training` |
| Base commit | `8204e85` "修复若干问题，训练改为初始化竖直立着的" |
| Committed on top | nothing — the entire pass is uncommitted working-tree state |
| Packages touched | `plane` **and** `jump`, symmetrically |

`actuatex/reference/wheel_legged/fudan_rl_wheel_leg` is a symlink to the same
repository, so `goal.md`'s path and this one are one tree.

Modified in place (9 files per package): `envs/__init__.py`,
`envs/base/base_config.py`, `rsl_rl/algorithms/ppo.py`,
`rsl_rl/runners/on_policy_runner.py`, `rsl_rl/storage/rollout_storage.py`,
`scripts/play.py`, `scripts/train.py`, `utils/helpers.py`,
`utils/task_registry.py`.

Added: `envs/base/robust_robot.py`, `envs/wheel_legged/robust_config.py`,
`rsl_rl/modules/actor_critic_robust.py`, `rsl_rl/utils/{evaluation,robustness}.py`,
`scripts/{evaluate_robustness,validate_training}.py`, plus top-level
`docs/`, `scripts/`, `tests/`, `pytest.ini`, `artifacts/`.

The design is additive and removable: `WheelLeggedCfg` itself is untouched, and
`BaselineCfg` re-declares the original behaviour explicitly
(`correctness_fixes = False`, no delay, no curriculum) so that "original" is a
runnable configuration rather than a git revert.

## 2. The experiment ladder Codex defined

| Task | Delta from the previous stage | Actor topology |
|---|---|---|
| `baseline` / `A0` | original behaviour, optional diagnostics | MLP |
| `A1` | fresh reset obs/history, terminal bootstrap, episode counters, PPO sampling/numerics/grad fixes | MLP |
| `A2` | A1 + per-env action-delay curriculum, 0..2 control steps | MLP |
| `A3` | A2 + corrected physical-property labels, planar mass-scaled pushes, disturbance curriculum | MLP |
| `A4` | A3 + causal Conv1D velocity encoder | **Conv1D** |
| `A5` | A4 + smooth orientation reward | **Conv1D** |
| `robust_v1` | the conservative retained selection: `correctness_fixes` on, `finite_checks` on, disturbance curriculum on. Plane stays push-disabled; A2/A3's new delay/push distributions are *not* included | MLP |

`robust_v1` is the intended solid configuration (`goal.md` §0.5). Read from
`robust_config.py:experiment_configs`, it does **not** set
`encoder_type = "causal_conv"` and does **not** set `action_delay_steps`, so on
`plane` it inherits `[0, 0]` — the delay buffer is a pass-through.

## 3. Current architecture, verified from code

Read by `scripts/handoff_audit.py` → `artifacts/handoff_audit.json`.

| Quantity | Value | Same as old Fudan policy? |
|---|---|---|
| observation dim | 25 | yes |
| history length | 5 frames, `obs_history_dec = 1` | yes |
| history/encoder input | 125 | yes |
| encoder hidden | `[128, 64]` → latent | yes |
| latent dim | 3 | yes |
| actor input | 28 (25 obs ⧺ 3 latent) | yes |
| actor hidden | `[128, 64, 32]` | yes |
| action dim | 6 | yes |
| activation | ELU (both encoder and actor) | yes |
| critic hidden | `[256, 128, 64]` | yes |
| control dt | 0.01 s (`sim.dt` 0.005 × `decimation` 2) | yes |
| policy rate | 100 Hz | yes |
| `pos_action_scale` | 0.5 | yes |
| `vel_action_scale` | 10.0 | yes |
| `clip_actions` | 100.0 | yes |
| PPO rollout | `num_steps_per_env = 48`, 5 epochs, 4 minibatches, adaptive LR, `desired_kl = 0.005` | yes |
| configured schedule | `max_iterations = 50000` (`WheelLeggedCfgPPO`) | yes |

`ActorCriticRobust` **subclasses** `ActorCriticSequence`. With the default
`encoder_type="mlp"` it inherits the parent's `self.encoder` and `self.actor`
verbatim and only overrides `update_distribution`, which still computes

```python
self.latent = self.encoder(observation_history)
mean = self.actor(torch.cat((observations, self.latent.detach()), dim=-1))
```

The `latent.detach()` that the QAT rollout code (`qat/hwq/quant_policy.py`)
reproduces is therefore still correct.

**Previous-action semantics are unchanged.** `robust_robot.py:step` clips the
raw policy output into `self.actions` and pushes it through
`control_action_buffer`; with `action_delay_steps = [0, 0]` on `plane`, the
applied action equals the requested action, and `last_actions[:, :, 0]` still
feeds the observation. The QAT closed-loop recursion (`goal.md` §13) needs no
change for `robust_v1`.

**One parameter-level change:** the unconstrained `std` parameter became
`log_std`, with a `_load_from_state_dict` migration hook. This affects
exploration and checkpoint loading only. The deployed path is the actor mean,
so the export and fixed-point reference are unaffected.

## 4. Available FP32 checkpoints

**There is no converged checkpoint.** Every run Codex produced is a short
validation run:

| Package | Runs | Max iterations reached |
|---|---|---|
| `plane` | 60 run directories across `baseline`, `A1`…`A5`, `robust_v1` | **48** |
| `jump` | 6 run directories | **48** |

`WheelLeggedCfgPPO.runner.max_iterations` is **50000**; the ladder ran 12- and
48-iteration smoke runs, i.e. 0.1% of the configured schedule. (An earlier
revision of this document said 5000: `scripts/handoff_audit.py` was reading
`LeggedRobotCfgPPO`, the base class, which the task overrides. Fixed.) Codex's own `docs/training.md` says the driver
"does not launch full training" and that "short horizons are not final
policy-quality evidence".

Consequence for `goal.md` §1: `SOLID_FP32` **cannot** be selected from the
handoff. §1 requires "the SAME converged FP32 policy", so a full-length
training on the frozen `robust_v1` configuration has to be run before any of
PTQ / QAT / `SOLID_FP32_PLUS` can branch. This is the single largest change
the handoff makes to the plan.

## 5. Evaluation infrastructure

`scripts/evaluate_robustness.py` is the harness `goal.md` §6 requires be used
instead of a competing benchmark. Default scenario matrix: `nominal`,
action delay 0/1/2 control steps, friction 0.4/0.8/1.2, added base mass
−1/0/+2 kg, planar push Δv 0.2/0.5/1.0 m/s. It evaluates a frozen checkpoint
with fresh episodes, noise off unless `--eval_noise`, fixed commands
(0.5 m/s forward, 0.5 rad/s yaw, 0.15 m height).

Reported per seed: observed fall rate, failure termination, right-censored
survival, forward/yaw RMSE, peak absolute roll/pitch, action first-difference
RMS, last-substep torque saturation, per-component velocity-estimation RMSE,
plus checkpoint hash and iteration. This covers every §6 minimum except
episode return, which is available from the training `metrics.jsonl`.

The `plane` sweep is complete: `artifacts/full_robustness/evaluations.json`,
6 evaluations (`baseline` and `robust_v1` × training seeds 1/2/3), eval seeds
11/12/13, 64 envs, all `exit_code: 0`. It is a valid *methodology* artifact
but not a policy result — the checkpoints are `model_48.pt`.

## 6. Bugs Codex confirmed in the original source

Recorded in `docs/training_audit.md` §"Bugs and fragile behavior verified in
source". The ones that bear on QAT:

- partial resets leak the previous episode's velocity/gravity/acceleration and
  do not clear the action FIFO — this is why the QAT rollout must use the
  fixed reset path, not the original one;
- no terminal observation before auto-reset, so timeouts bootstrap `γV(s_t)`
  instead of `γV(s_terminal)`, and timeouts are not separated from falls —
  the same early-termination evaluation bias `goal.md` §18 calls out;
- `RolloutStorage.mini_batch_generator` never samples the tail of a
  non-divisible rollout and reuses indices across epochs;
- `PPO.update` clips over *all* parameters in both calls, so the encoder's
  gradient clipping can be dominated by stale PPO gradients;
- `ActorCriticSequence.update_distribution` allows a non-positive std and
  assigns `Normal.set_default_validate_args = False` instead of calling it.

These are Codex's findings about the RL stack. They are distinct from, and do
not overlap with, the three arithmetic bugs this QAT effort found and owns
(`goal.md` §4): the requantizer's asymmetric negative rounding, the 9 wrong
H7 ELU ROM entries, and the unimplemented `SysTick_Handler`. All four
arithmetic regression tests still pass against the current tree.

## 7. Deployment compatibility

Unchanged topology ⇒ the existing pipeline is reused, not adapted. Layer map,
weight-word budget and the negative-result on A4/A5 are in
`SOLID_POLICY_DEPLOYMENT_MAPPING.md`. Machine-readable:
`artifacts/handoff_audit.json` → `"compatible_with_existing_datapath": true`.

## 8. Handoff is partial

At the time of writing, the Codex process (PID 2106511, 2 h 32 m) had finished
the `plane` package end to end and had just started the same robustness sweep
on `jump` (`artifacts/jump_robustness/`, started 22:00). `jump` is a separate
task configuration and is not the deployment target, so it does not gate the
QAT work — but this audit will be re-confirmed against the final tree before
`SOLID_FP32` is frozen, because the audit is only as good as the snapshot it
read.
