# Re-run plan for `locomotion_v2`

`QAT_RESULTS.md` carries a validity notice: `robust_v1` admits a chassis-support
exploit, so its absolute control numbers are not statements about locomotion.
Codex's `locomotion_v2` corrects it — z=0.18 m, explicit non-wheel
contact/tilt/clearance failures — and now passes every 64-second nominal
segment across three training seeds.

## What carries over unchanged

Verified 2026-09-08 11:45 against
`locomotion_v2/…_seed1_short_locomotion_v2_h5_s1/model_1500.pt`:

| check | result |
|---|---|
| actor/encoder shapes | identical: 125→128→64→3, 28→128→64→32→6 |
| deployment mapping | identical: 7 GEMM, 5 ELU, 1 CONCAT, 38,400 MAC, 620/1280 words |
| `compatible_with_existing_datapath` | true |
| torch vs integer reference, W16A16 | **PASS**, 9/9 tensors exact |
| torch vs integer reference, W8A8 | **PASS** |

`LocomotionV2Cfg` inherits `RobustV1Cfg` and changes rewards, commands, control
and initial state — not the network. So the whole apparatus (`hwq/`, the
exporter, the H7 kernel, the RTL cross-check, `SOLID_POLICY_DEPLOYMENT_MAPPING.md`)
carries over without modification, and the re-run is a re-execution rather than
a port.

## What must be redone, and why

1. **Activation-frac calibration.** The W8A8 bit-exactness above only shows
   torch and numpy agree; it says nothing about whether the *old* fracs suit
   `locomotion_v2`'s activation ranges. A policy that balances on its wheels
   visits different states than one resting on its chassis. Recalibrate from
   fresh on-policy rollouts before any A8 point.
2. **Every control number.** FP32 evaluation, the PTQ ladder, QAT, the §7
   control, multi-seed. These are the numbers the validity notice invalidates.
3. **Golden vectors.** Regenerate; `large_attitude` should now be reachable,
   since a wheel-balancing policy does tilt. If it is still empty, that is a
   finding about `locomotion_v2` worth reporting.
4. **§33 from-scratch**, stopped at 474/5000 on the old task.

## Baseline choice

Prefer **Codex's own `locomotion_v2` checkpoint** as `SOLID_FP32` over training
a fresh one. goal.md §0/§1 ask for the Codex baseline to be *identified and
frozen*, not reproduced, and codex has three-seed nominal evidence for theirs.
Wait for their final held-out evaluation — `selected_development_policy.json`
is explicitly labelled "development candidate selected before final held-out
evaluation".

## Two evaluators, not one

`robust_v1` and `locomotion_v2` need different evaluators, and using the wrong
one fails immediately rather than silently:

* `evaluate_robustness.py` hardcodes `make_env("robust_v1")`, so it builds
  `RobustLeggedRobot` regardless of the checkpoint and dies on a locomotion
  policy with `'RobustLeggedRobot' object has no attribute '_reward_body_contact'`.
* `scripts/evaluate_locomotion.py` builds `sequence_evaluation` and scores
  command sequences with `support_fraction`, `body_contact_fraction` and
  per-segment `success` — the metrics that actually detect the chassis exploit.
  It lives in the worktree's top-level `scripts/`, not in the package, so it is
  loaded by path; its entry point is `evaluate()`, not `main()`; and it needs an
  explicit `--seed` because the base parser defaults it to `None` and
  `torch.manual_seed(None)` raises.

`scripts/eval_quant_locomotion.py` wraps it with the same one-global rebinding
used for the robustness evaluator.

## Commands

Every script is now parameterised; nothing needs editing.

```bash
export TASK=locomotion_v2
CKPT=<codex checkpoint>            # or: TASK=$TASK bash scripts/train_solid_fp32.sh

$ISAAC_PY scripts/dump_calib_obs.py --checkpoint $CKPT \
    --out artifacts/v2/calib_obs.npz --steps 400 --headless --num_envs 256
CKPT=$CKPT OBS=artifacts/v2/calib_obs.npz OUTD=artifacts/v2/ptq \
    bash scripts/run_ptq_ladder.sh
CKPT=$CKPT FRACS=artifacts/v2/ptq/act_fracs_a8.json TASK=$TASK TAG=_v2 \
    bash scripts/run_qat_and_control.sh
bash scripts/dump_regimes.sh          # golden vectors
bash scripts/run_multiseed.sh         # §32
```

Estimated GPU time: ~1 h PTQ ladder, ~40 min QAT + control, ~2 h multi-seed,
~15 min golden vectors. Roughly four hours, unattended.


---

## Smoke-test result (2026-09-08 13:50, preliminary)

One evaluation seed, 128 envs, no QAT arm. Enough to confirm the pipeline runs
and to see the shape of the answer.

| segment | metric | FP32 | W16A16 | W8A8 |
|---|---|---|---|---|
| forward | fwd RMSE | 0.0119 | 0.0138 | 0.1429 |
| forward | support_fraction | 1.000 | 1.000 | 0.946 |
| forward | success | 1.00 | 1.00 | **0.00** |
| forward | vel-est RMSE | 0.0064 | 0.0058 | 0.1413 |
| stop | success | 1.00 | 1.00 | 0.05 |
| turn_left | success | 1.00 | 1.00 | 0.16 |

* The FP32 policy is genuinely balancing — `support_fraction` 1.000 and
  `body_contact_fraction` 0.000 throughout.
* W16A16 remains lossless, so the deployment recommendation survives the task
  change.
* W8A8 still collapses, now with an unambiguous signal: `success` → 0.00 and
  `support_fraction` below 1.0, i.e. the robot loses wheel support rather than
  merely tracking worse.
* Degradation is **12×** here against **8.4×** on `robust_v1`, confirming
  "a better FP32 policy quantizes worse" on an independent task — and refuting
  the prediction that `locomotion_v2`'s 2–4× finer activation LSBs would
  protect it.

The QAT arm is the open question: whether it still recovers 86–101% when the
baseline is a real controller and the failure metric is loss of wheel support.
