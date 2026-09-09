#!/usr/bin/env python3
"""Build the permanent golden-vector dataset. goal.md section 11.

Observations are pulled from real closed-loop rollouts and sorted into the
seven regimes the spec asks for, using the observation's own contents rather
than by tagging the simulator -- so the same classifier works on data captured
from the robot later.

For every vector the full datapath is recorded: float observation, quantized
observation, each hidden activation, the latent, and the action in both float
and integer form. Emitted three ways so the same numbers can be replayed
anywhere:

  golden_vectors/golden.npz   Python
  golden_vectors/*.hex        $readmemh for the RTL testbenches
  golden_vectors/qat_golden.h C, for the H7 and the Vitis app
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.fixed_ref import ACT_FRAC, SatStats  # noqa: E402
from roboaccel_quant.quant_policy import build_from_onnx  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

# Observation layout, verified in QAT_DESIGN.md section 1.
ANG_VEL, GRAV, CMD, DOF_POS, DOF_VEL, PREV_ACT = (
    slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 13), slice(13, 19), slice(19, 25))


# Observation scales, from legged_robot.py:1082 and normalization.obs_scales.
# The observation stores scaled quantities; the classifier works in physical
# units so its thresholds mean something.
S_ANG_VEL, S_DOF_VEL = 0.25, 0.05
S_CMD_VX, S_CMD_WZ, S_CMD_H = 2.0, 0.25, 5.0


def classify(o: np.ndarray) -> str:
    """Regime of a single observation. Order matters: the rarer and more
    safety-relevant states win, so a tilted robot is not filed as 'turning'."""
    g = o[GRAV]
    tilt = float(np.arccos(np.clip(-g[2] / max(np.linalg.norm(g), 1e-9), -1, 1)))
    prev_act = float(np.abs(o[PREV_ACT]).max())
    cmd_vx = float(o[CMD][0]) / S_CMD_VX                 # m/s
    cmd_wz = float(o[CMD][1]) / S_CMD_WZ                 # rad/s
    roll_rate = float(o[ANG_VEL][0]) / S_ANG_VEL         # rad/s
    pitch_rate = float(o[ANG_VEL][1]) / S_ANG_VEL        # rad/s
    wheel = float(np.mean(o[DOF_VEL][[2, 5]])) / S_DOF_VEL   # rad/s, wheel proxy

    if prev_act > 8.0:
        return "near_saturation"
    # Disturbance before attitude: being pushed shows up as angular RATE,
    # while a large static tilt is a different (slower) regime. Checking tilt
    # first swallowed almost every pushed state.
    #
    # Threshold from the measured distribution rather than picked: over 13,728
    # observations max(|roll_rate|,|pitch_rate|) has p95 = 0.41 and p99 = 0.76
    # rad/s nominal, so 1.5 is comfortably in the tail. It fires on 0.46 % of a
    # --push rollout against 0.20 % nominal -- 2.3x, which is what makes it a
    # disturbance signature and not just noise. The earlier 3.0 caught 0.01 %,
    # about one vector.
    if max(abs(roll_rate), abs(pitch_rate)) > 1.5:
        return "disturbance"
    if tilt > 0.35:                                      # ~20 deg from upright
        return "large_attitude"
    if abs(cmd_wz) > 1.0:
        return "turning"
    if cmd_vx > 0.5 and abs(wheel) < 8.0:                # commanded, not yet moving
        return "acceleration"
    if abs(cmd_vx) < 0.2 and abs(wheel) > 8.0:           # moving, commanded to stop
        return "braking"
    return "balancing"


REGIMES = ["balancing", "acceleration", "braking", "turning",
           "large_attitude", "disturbance", "near_saturation"]


def hexlines(v: np.ndarray) -> list[str]:
    return [f"{int(x) & 0xFFFF:04x}" for x in np.asarray(v).reshape(-1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--obs", type=Path, nargs="+", required=True)
    ap.add_argument("--per-regime", type=int, default=24)
    ap.add_argument("--out", type=Path, default=QUANT_ROOT / "golden_vectors")
    args = ap.parse_args()

    obs = np.concatenate([np.load(p)["obs"] for p in args.obs]).astype(np.float64)
    hist = np.concatenate([np.load(p)["obs_history"] for p in args.obs]).astype(np.float64)
    obs = obs.astype(np.float32).astype(np.float64)     # float32 is the input of record
    hist = hist.astype(np.float32).astype(np.float64)
    print(f"{len(obs)} candidate observations from {len(args.obs)} dump(s)")

    labels = np.array([classify(o) for o in obs])
    counts = {r: int((labels == r).sum()) for r in REGIMES}
    print("available per regime:", counts)

    rng = np.random.default_rng(3)
    pick = []
    for r in REGIMES:
        idx = np.flatnonzero(labels == r)
        if len(idx) == 0:
            print(f"  WARNING: no '{r}' states in these rollouts")
            continue
        take = min(args.per_regime, len(idx))
        pick += rng.choice(idx, size=take, replace=False).tolist()
    pick = np.array(sorted(pick))
    print(f"selected {len(pick)} vectors")

    net = build_from_onnx(args.onnx, QuantConfig())
    ref = net.to_fixed_ref()
    stats = SatStats()

    rec: dict[str, list] = {k: [] for k in
                            ("obs", "obs_history", "obs_q", "hist_q", "enc0", "enc1",
                             "latent", "concat", "act0", "act1", "act2", "action_q",
                             "action", "regime")}
    for i in pick:
        tr: dict = {}
        a = ref(obs[i], hist[i], stats=stats, trace=tr)
        rec["obs"].append(obs[i]); rec["obs_history"].append(hist[i])
        rec["obs_q"].append(ref.quantize_obs(obs[i]))
        rec["hist_q"].append(ref.quantize_obs(hist[i]))
        for k, src in (("enc0", "enc0"), ("enc1", "enc1"), ("latent", "enc2"),
                       ("concat", "concat"), ("act0", "act0"), ("act1", "act1"),
                       ("act2", "act2"), ("action_q", "act3")):
            rec[k].append(tr[src])
        rec["action"].append(a)
        rec["regime"].append(labels[i])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "golden.npz",
                        **{k: np.array(v) for k, v in rec.items()})

    for name in ("obs_q", "hist_q", "action_q"):
        (out / f"golden_{name}.hex").write_text(
            "\n".join(hexlines(np.array(rec[name]))) + "\n")

    n = len(pick)
    hdr = [
        "/* Generated by qat/scripts/make_golden_vectors.py -- do not edit. */",
        "#ifndef QAT_GOLDEN_H", "#define QAT_GOLDEN_H", "#include <stdint.h>", "",
        f"#define QAT_GOLDEN_COUNT   {n}",
        f"#define QAT_GOLDEN_OBS     {len(rec['obs_q'][0])}",
        f"#define QAT_GOLDEN_HIST    {len(rec['hist_q'][0])}",
        f"#define QAT_GOLDEN_ACT     {len(rec['action_q'][0])}", "",
    ]
    for cname, key in (("qat_golden_obs_q", "obs_q"), ("qat_golden_hist_q", "hist_q"),
                       ("qat_golden_act_q", "action_q")):
        flat = np.array(rec[key]).reshape(-1).astype(np.int64)
        hdr.append(f"static const int16_t {cname}[{flat.size}] = {{")
        for i in range(0, flat.size, 16):
            hdr.append("    " + ", ".join(str(int(v)) for v in flat[i:i + 16]) + ",")
        hdr += ["};", ""]
    hdr += ["#endif /* QAT_GOLDEN_H */", ""]
    (out / "qat_golden.h").write_text("\n".join(hdr))

    meta = {"onnx": str(args.onnx), "count": int(n),
            "per_regime": {r: int(sum(1 for x in rec["regime"] if x == r)) for r in REGIMES},
            "available": counts, "saturation": stats.report()}
    (out / "golden_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["per_regime"], indent=2))
    print(f"wrote {out}/golden.npz, *.hex, qat_golden.h, golden_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
