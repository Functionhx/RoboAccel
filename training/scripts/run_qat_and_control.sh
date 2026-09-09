#!/usr/bin/env bash
# goal.md §7 + §13/§14 -- QAT fine-tuning and its mandatory FP32 control.
#
# Both arms branch from the SAME frozen SOLID_FP32 checkpoint and run the SAME
# number of iterations in the SAME environment, so any difference is
# attributable to the fake quantization and not to extra training. §7 calls
# this control non-optional and it is run first, so a failure there cannot be
# discovered after the QAT result is already in hand.
#
# Target is W8A8: the PTQ ladder puts the cliff at 8-bit activations, not at
# 8-bit weights (W8A16 is intact), so that is where recovery is worth measuring.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_solid.sh"

CKPT=${CKPT:-$QAT_ROOT/artifacts/solid_fp32/SOLID_FP32.pt}
FRACS=${FRACS:-$QAT_ROOT/artifacts/ptq/act_fracs_a8.json}
TASK=${TASK:-robust_v1}
TAG=${TAG:-}
ITERS=${ITERS:-500}
ENVS=${ENVS:-8192}
SEED=${SEED:-1}

cd "$SOLID_WT"

echo "=== SOLID_FP32_PLUS: $ITERS FP32 iterations (the §7 control) ==="
"$ISAAC_PY" "$QAT_ROOT/scripts/train_policy.py" \
  --run "solid_fp32_plus$TAG" --task "$TASK" \
  --init-from "$CKPT" --iters "$ITERS" --num-envs "$ENVS" --seed "$SEED" \
  --headless 2>&1 | tail -4

echo "=== QAT W8A8: $ITERS fake-quant iterations from the same checkpoint ==="
"$ISAAC_PY" "$QAT_ROOT/scripts/train_policy.py" \
  --run "qat_w8a8$TAG" --task "$TASK" \
  --quant W8A8 --act-fracs "$FRACS" \
  --init-from "$CKPT" --iters "$ITERS" --num-envs "$ENVS" --seed "$SEED" \
  --headless 2>&1 | tail -4

echo "QAT_AND_CONTROL_DONE"
