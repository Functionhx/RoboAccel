# SOLID_FP32 — the frozen FP32 baseline

`goal.md` §1 requires one official converged FP32 policy, preserved
permanently, from which FP32 evaluation / PTQ / FP32+extra-training / QAT all
branch. This file is that record.

## Why it had to be trained rather than selected

The Codex handoff contains **no converged checkpoint**. Every one of its 66 run
directories across both packages stops at 12 or 48 iterations, against a
configured `max_iterations` of 5000. This is deliberate, not an oversight —
`docs/training.md` states the experiment driver "does not launch full training",
and `docs/training_report.md` §F concludes:

> Tracking errors remain large, and the plane policies mostly hold position:
> 48 updates do not establish competent locomotion or a meaningful robustness
> advantage.

Codex delivered validated infrastructure plus an ablation ladder. Producing the
converged policy is this side of the handoff.

## Configuration

| | |
|---|---|
| Task | `robust_v1` |
| Package | `plane` |
| Seed | 1 |
| Environments | 8192 (repository default) |
| Iterations | 5000 — **10% of the configured 50000** |
| Rollout | `num_steps_per_env = 48` |
| History | 5 frames, `obs_history_dec = 1` |
| Encoder | MLP (`encoder_type` left at default) |
| Launcher | `scripts/train_solid_fp32.sh` |
| Run directory | `plane/logs/robust_v1/20260907_224501_717847_seed1_solid_fp32` |

`robust_v1` was chosen because it is the configuration Codex retained as its
conservative selection *and* the only rung of the A0–A5 ladder that stays
compatible with the deployed datapath — `A4`/`A5` swap in a causal Conv1D
encoder the PL has no descriptor for (see `SOLID_POLICY_DEPLOYMENT_MAPPING.md`).

Concretely, `robust_v1` enables `correctness_fixes`, `finite_checks` and the
disturbance curriculum. On `plane` it inherits `action_delay_steps = [0, 0]` and
pushes stay disabled, so the new delay/push distributions introduced in A2/A3
are *not* active. That matters for QAT: the closed-loop previous-action
recursion (`goal.md` §13) sees an undelayed action path, identical to what the
pre-handoff QAT rollout code already assumed.

## Frozen infrastructure identity

The Codex pass is entirely uncommitted working-tree state, so a git SHA alone
does not identify it. Both are recorded:

| | |
|---|---|
| Worktree | `.worktrees/robust-training` on branch `robust-training` |
| Base commit | `8204e853dfd2ed06d85a322e1a998c3d20a3be2c` |
| Tracked worktree dirty | `true` |
| Infrastructure manifest | `artifacts/solid_fp32/infra_manifest.sha256`, 56 files |
| Manifest digest | `9c15b533913ae1db3157177bbd4e1cb86885b6c95b0cd1cdb43a1bfd0f62b420` |

The training run independently records its own `source_sha256` over 44 modules
in `config.json`, so provenance is captured twice by two different tools.

Runtime: Python 3.8.20, torch 2.1.1, CUDA device 0, `sim_dt` 0.005,
`control_dt` 0.01, 25 observation features, 141 privileged features, DOF order
`lf0, lf1, l_wheel, rf0, rf1, r_wheel`.

## Do not overwrite

The final checkpoint is copied into `artifacts/solid_fp32/` on completion and is
read-only from that point. Every downstream arm loads it; none of them writes
back to it. `SOLID_FP32_PLUS` (`goal.md` §7) continues training *from a copy*
for exactly the iteration count QAT uses, and lands in its own directory.

## Measured throughput

673,403 steps/s at 8192 envs, 0.58 s/iteration (collection 0.363 s, learning
0.222 s). Codex's independent measurement of the same configuration was 676,014
transitions/s. Full 5000-iteration wall time ≈ 49 min.


## Correction: SOLID_FP32 is not converged

An earlier revision of this file called SOLID_FP32 converged. That was wrong.
The task's configured schedule is `max_iterations = 50000`
(`WheelLeggedCfgPPO`), so 5000 iterations is **10%** of it. The audit script
had been reading `LeggedRobotCfgPPO`, the base class, whose 5000 the task
overrides.

The evidence for the correction is direct: 500 further FP32 iterations cut
nominal forward RMSE from 0.2898 to 0.0657, a 4.4x improvement. A converged
policy does not do that.

What this does and does not affect:

* **Unaffected** — the QAT vs PTQ vs FP32 comparison. All three arms sit at a
  matched 5500 iterations, so the controlled contrast holds regardless of
  where 5500 falls on the schedule.
* **Affected** — the absolute performance level, and any claim that these are
  final numbers. The recovery fractions in `QAT_RESULTS.md` are measured at an
  under-trained operating point and could differ at true convergence.
* **Strengthened** — the reading of the handoff. Codex's 48-iteration runs are
  0.1% of the configured schedule, not 1%.
