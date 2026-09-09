#!/usr/bin/env bash
# goal.md sections 8/9 on the frozen codex baseline, scored by the locomotion
# evaluator. Separate from the multi-seed driver so the two can share the GPU.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_solid.sh"
CK=$QAT_ROOT/artifacts/v2/frozen/SOLID_FP32_V2.pt
F8=$QAT_ROOT/artifacts/v2/act_fracs_a8.json
OUT=$QAT_ROOT/artifacts/v2
cd "$SOLID_WT"
ev () { local tag=$1; shift
  [ -f "$OUT/$tag" ] && { echo "skip $tag"; return; }
  timeout 2400 "$ISAAC_PY" "$QAT_ROOT/scripts/eval_quant_locomotion.py" "$@" \
    --headless --model="$CK" --output="$OUT/$tag" --num_envs=128 --seed=11 \
    >/dev/null 2>&1 && echo "  $tag OK" || echo "  $tag FAILED"; }
ev ladder_fp32   --fp32
ev ladder_w16a16 --weight-bits 16 --act-bits 16
ev ladder_w8a16  --weight-bits 8  --act-bits 16
ev ladder_w8a8   --weight-bits 8  --act-bits 8 --act-fracs "$F8"
ev ladder_w4a8   --weight-bits 4  --act-bits 8 --act-fracs "$F8"
echo "V2_LADDER_DONE"
