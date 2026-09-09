#!/usr/bin/env bash
# One Isaac Gym simulator per process (a second sim in the same process
# segfaults), then merge the partial dumps into one observation pool.
set -euo pipefail
source /home/as/vllm/fpga/projects/rl_accel/qat/scripts/env_solid.sh
OUT=${OUT:-$QAT_ROOT/artifacts/deploy/regimes}
mkdir -p "$OUT"
cd "$SOLID_WT"
for i in 0 1 2 3 4 5 6 7; do
  echo "=== segment $i ==="
  "$ISAAC_PY" "$QAT_ROOT/scripts/dump_regime_obs.py" \
    --checkpoint "${CKPT:-$QAT_ROOT/checkpoints/solid_fp32_plus/model_500.pt}" \
    --out "$OUT/seg$i.npz" --segment "$i" --steps-per-segment 120 \
    --headless --num_envs ${ENVS:-128} 2>&1 | grep -E "segment|wrote" || true
done
"$ISAAC_PY" - <<'PY'
import numpy as np, glob, os
from pathlib import Path
out = Path(os.environ.get("OUT") or (os.environ["QAT_ROOT"] + "/artifacts/deploy/regimes"))
fs = sorted(out.glob("seg*.npz"))
o = np.concatenate([np.load(f)["obs"] for f in fs], 0)
h = np.concatenate([np.load(f)["obs_history"] for f in fs], 0)
idx = np.random.default_rng(0).choice(o.shape[0], size=min(40000, o.shape[0]), replace=False)
np.savez_compressed(out.parent / "regime_obs.npz", obs=o[idx], obs_history=h[idx])
print(f"merged {len(fs)} segments -> {o.shape} pooled, {o[idx].shape} sampled")
PY
echo "REGIME_DUMP_DONE"
