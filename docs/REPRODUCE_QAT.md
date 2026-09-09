# REPRODUCE_QAT

Everything here runs on environments already present on this machine. Do not
create new ones — see `actuatex/docs/LOCAL_ENVIRONMENTS.md` for why.

> **The pipeline was rebased on 2026-09-08.** Every result in `QAT_RESULTS.md`
> branches from `SOLID_FP32`, trained on the frozen Codex `robust-training`
> infrastructure. Sections 0–1 below still describe the shared arithmetic
> checks; the end-to-end recipe is in **"Reproducing the current results"** at
> the bottom. The older `env_isaac.sh` / ONNX path is kept because the
> arithmetic regressions still run through it, but it is not how the reported
> numbers were produced.

## 0. Environments

```bash
source scripts/env_isaac.sh      # sets ISAAC_PY, MJ_PY, FUDAN_PLANE, PYTHONPATH
```

| Variable | Points at | Used for |
|---|---|---|
| `ISAAC_PY` | `envs/fudan-py38` (py3.8, torch 2.1.1 CUDA, Isaac Gym Preview 4) | training, closed-loop eval |
| `MJ_PY` | `envs/fudan-mujoco-py39` (py3.9, torch 2.4.0 CUDA, mujoco 3.2.7, onnx) | analysis, export, verification |
| `FUDAN_PLANE` | `rm-cortex-references/wheel-legged/fudan_rl_wheel_leg/plane` | the upstream env, **imported by path, never modified or copied** |

The upstream repo has no licence (`WHEEL_LEGGED_REPOSITORY_MAP.zh-CN.md`
sections 2 and 3.3). It is read-only reference. Nothing from it is vendored
here; `PYTHONPATH` does the work.

## 1. Verify the arithmetic before trusting any result

```bash
$MJ_PY scripts/verify_against_rtl.py                       # RTL + H7 C vs Python
$MJ_PY scripts/bitexact_check.py --onnx <policy.onnx> --n 2000
```

Both must print `PASS`. The first parses `rtl/rl_elu_array8.sv` and
`rtl/rl_round_shift_sat16.sv` and `../h7_bench/src/ra_kernel.c`; the second
compares the torch fake-quant against the numpy integer reference layer by
layer. If either fails, stop — every downstream number is meaningless.

## 2. FP32 baseline

```bash
cd $FUDAN_PLANE
$ISAAC_PY $QAT_ROOT/scripts/train_policy.py --run fp32_baseline \
    --iters 6000 --num-envs 4096 --seed 1 --save-interval 250
```

~40 min on an RTX 4070 Ti SUPER. Checkpoints land in
`qat/checkpoints/fp32_baseline/`.

**Why not the released weights.** The repo ships two ONNX actors but no `.pt`,
so there is no critic to fine-tune from. More importantly the released actors
were trained against the *previous* `default_joint_angles` — the values
commented out directly above the live ones in `wheel_legged_config.py`, which
match `PLANE_DEFAULT_DOF_POS` in the MuJoCo script. Evaluating them in the
current env measures a config mismatch, not quantization. They are still used
for pipeline cross-validation and MuJoCo golden vectors.

## 3. The full matrix

```bash
BASE=checkpoints/fp32_baseline/model_6000.pt bash scripts/run_matrix.sh
```

Resumable: every step is skipped if its JSON artifact already exists. Runs
FP32 eval, PTQ at four precisions, QAT fine-tune at four precisions, and QAT
eval both with its own arithmetic and in FP32 (to separate "adapted to
quantization" from "trained 1500 iterations longer").

Individual steps:

```bash
# PTQ: the same weights, no retraining
$ISAAC_PY scripts/eval_policy.py --checkpoint <ckpt> --quant W16A16 --check-int 3

# INT8 activations need a calibrated grid first, and refuse to run without it
$MJ_PY scripts/calibrate_act_fracs.py --checkpoint <ckpt> \
    --obs golden_vectors/obs_fp32.npz --act-bits 8 --out artifacts/act_fracs_a8.json

# QAT fine-tune
$ISAAC_PY scripts/train_policy.py --run qat_W8A16 --quant W8A16 \
    --init-from <fp32 ckpt> --iters 1500
```

## 4. Diagnostics, golden vectors, export

```bash
$MJ_PY scripts/quant_diagnostics.py --checkpoint <ckpt> \
       --obs golden_vectors/obs_fp32.npz --quant W16A16

$MJ_PY scripts/make_golden_vectors.py --onnx <exported.onnx> \
       --obs golden_vectors/obs_fp32.npz golden_vectors/obs_push.npz

$MJ_PY scripts/export_to_onnx.py <ckpt> --out artifacts/qat_policy.onnx
$MJ_PY ../tools/export_policy.py --model artifacts/qat_policy.onnx \
       --out ../generated --samples 1000 --seed 7 --c-out ../03_vitis/src
$MJ_PY scripts/verify_export.py --onnx artifacts/qat_policy.onnx --n 1000
```

## 5. Results

```bash
$MJ_PY scripts/make_results_table.py --md QAT_RESULTS_TABLE.md
```

## Determinism

Training is seeded (`--seed`) but Isaac Gym on GPU is not bit-reproducible
across runs; reward curves land within a few percent. Evaluation is fixed-seed
and averages over ~1000 envs x 30 s, which is what makes the arms comparable.
Trajectories still diverge between arms — a policy that acts differently
resets at different times — so what is controlled is the distribution, not the
individual episode. Fixed-point results (`bitexact_check`, `verify_export`,
golden vectors) are exactly reproducible.


---

# Reproducing the current results

## Environment

```bash
source scripts/env_solid.sh   # SOLID_WT, SOLID_PLANE, ISAAC_PY, QAT_ROOT, PYTHONPATH
```

`env_solid.sh` points at the Codex worktree
`fudan_rl_wheel_leg/.worktrees/robust-training`, not the pristine upstream.
Same licensing rule: imported by path, never vendored.

That worktree's changes are **uncommitted working-tree state**, so a git SHA
does not identify them. `artifacts/solid_fp32/infra_manifest.sha256` pins all
56 infrastructure files; check it before trusting a rerun:

```bash
( cd "$SOLID_WT" && sha256sum -c "$QAT_ROOT/artifacts/solid_fp32/infra_manifest.sha256" )
```

## 1. Arithmetic gates — run these first

```bash
$ISAAC_PY scripts/verify_against_rtl.py                    # ELU ROM + rounding vs RTL
bash scripts/hosttest/build_and_run.sh artifacts/deploy/SOLID_FP32_PLUS_h7
$ISAAC_PY scripts/bitexact_check.py \
    --checkpoint artifacts/solid_fp32/SOLID_FP32.pt --n 4000
```

All three must print `PASS`. If they do not, no control result below means
anything.

## 2. SOLID_FP32 — the frozen baseline (~51 min)

```bash
bash scripts/train_solid_fp32.sh          # robust_v1, 8192 envs, seed 1, 5000 iters
```

Then freeze it read-only. sha256 `75e00dc5…`; do not overwrite.

## 3. Calibration and diagnostics

```bash
$ISAAC_PY scripts/dump_calib_obs.py --checkpoint artifacts/solid_fp32/SOLID_FP32.pt \
    --out artifacts/solid_fp32/calib_obs.npz --steps 400 --headless --num_envs 256
$ISAAC_PY scripts/quant_diagnostics.py --checkpoint artifacts/solid_fp32/SOLID_FP32.pt \
    --obs artifacts/solid_fp32/calib_obs.npz --quant W16A16
```

Calibration **must** come from on-policy rollouts. Fracs fitted to Gaussian
noise put the encoder input far outside its real range and waste half the INT8
grid, which then looks like a quantization cliff that is really a scaling bug.

## 4. PTQ ladder (~1 h)

```bash
bash scripts/run_ptq_ladder.sh            # W16A16, W8A16, W8A8, W4A8
```

## 5. QAT and its mandatory control (~40 min)

```bash
bash scripts/run_qat_and_control.sh       # SOLID_FP32_PLUS first, then QAT W8A8
```

The control runs **first** on purpose. Without it the QAT number is
uninterpretable — on this data QAT appeared to beat FP32 until the control
showed the gain was training length.

## 6. Evaluation — always through the Codex evaluator

```bash
bash scripts/eval_qat_arms.sh
python3 scripts/summarize_eval.py <arm>.json ... --labels A,B,C
```

`eval_quant_robustness.py` rebinds one module global inside Codex's
`evaluate_robustness.py` so quantized policies are scored by identical code.
Never stand up a second benchmark (`goal.md` §6).

## 7. Multi-seed and ablations

```bash
bash scripts/run_multiseed.sh                                  # seeds 2, 3 (~2 h)
python3 scripts/multiseed_table.py --triples 1:...,2:...,3:...
$ISAAC_PY scripts/rounding_ablation.py --checkpoint <ckpt> \
    --obs artifacts/solid_fp32/calib_obs.npz --weight-bits 8 --act-bits 8 \
    --act-fracs artifacts/ptq/act_fracs_a8.json
```

## 8. Deployment export

```bash
PYTHONPATH=$PWD $MJ_PY scripts/export_to_onnx.py <ckpt> --out <model>.onnx
$MJ_PY ../tools/export_policy.py --model <model>.onnx --out <dir> --samples 1000 --seed 7
PYTHONPATH=$PWD $MJ_PY scripts/gen_h7_branched.py <model>.onnx --name <n> --out <dir> \
    --golden 16 --golden-npz artifacts/solid_fp32/calib_obs.npz
bash scripts/hosttest/build_and_run.sh <dir>
```

## Traps that cost time here

* `nohup cmd > dir/log` fails if `dir` does not exist yet — the shell opens the
  redirect before the script's own `mkdir` runs, the job dies instantly, and
  the wrapper still reports exit 0. Create the directory first, then verify the
  process exists rather than trusting an exit status.
* **Never edit a shell script while it is running.** Bash reads scripts
  incrementally by byte offset; inserting lines mid-run shifts the offset and
  execution resumes mid-command. This silently skipped an entire evaluation arm.
* `until ! pgrep -f foo.sh` matches its own command line and waits forever.
  Grep a completion marker in the log instead.
* Substituting a policy class into the runner must be a **class**, not a
  factory function: the runner reads `actor_critic_class.is_sequence` before
  instantiating.
* Patch **both** `ActorCriticSequence` and `ActorCriticRobust`; `robust_v1`
  names the latter, and patching only the former trains an unquantized policy
  while labelling it QAT.
