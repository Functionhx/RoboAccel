#!/usr/bin/env python3
"""goal.md §25 -- collect observations spanning the seven golden-vector regimes.

`dump_calib_obs.py` holds one fixed command in the nominal scenario, which is
right for activation calibration but produces a near-steady-state trajectory:
19699 of 20000 samples classify as `balancing` and five regimes are empty.
Golden vectors are meant to exercise the arithmetic where it is stressed, so
this sweeps commands and perturbations instead.

Commands are stepped mid-episode rather than held, because `acceleration` and
`braking` are transients -- a constant 2 m/s command produces a robot already
at 2 m/s, which classifies as balancing like everything else.

ONE SEGMENT PER PROCESS. Isaac Gym segfaults when a second simulator is created
in the same process after the first is released, which is why Codex's own
run_evaluation_experiments.py spawns a subprocess per scenario. `--segment`
selects which entry of PLAN to run; `scripts/dump_regimes.sh` loops over them
and merges the partial dumps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# roboaccel_quant lives under quantization/, one component over.
sys.path.insert(0, str(REPO / "quantization"))
QAT_ROOT = REPO / "training"

import isaacgym  # noqa: F401  must precede torch
import numpy as np
import torch

# (command_x, command_yaw, scenario) -- chosen to hit distinct regimes
PLAN = [
    (0.0, 0.0, "nominal"),      # balancing
    (2.0, 0.0, "nominal"),      # acceleration on the step, then cruise
    (0.0, 0.0, "nominal"),      # braking, entered from the 2.0 segment
    (0.5, 1.5, "nominal"),      # turning
    (1.0, -1.5, "nominal"),     # turning, opposite sign
    (0.5, 0.5, "push:1.0"),     # disturbance + large attitude
    (2.0, 1.5, "push:1.0"),     # near saturation
    (1.5, 0.0, "delay:2"),      # delayed actuation transient
]


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps-per-segment", type=int, default=120)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--segment", type=int, required=True,
                    help="index into PLAN; one simulator per process")
    known, rest = ap.parse_known_args()
    if not 0 <= known.segment < len(PLAN):
        raise SystemExit(f"--segment must be 0..{len(PLAN)-1}")

    from wheel_legged_gym.scripts import evaluate_robustness as ev
    from wheel_legged_gym.utils import get_args, task_registry
    from wheel_legged_gym.utils.helpers import set_seed
    from roboaccel_quant.quant_policy import build_from_checkpoint
    from roboaccel_quant.torch_hw import QuantConfig

    ck = torch.load(str(known.checkpoint), map_location="cpu", weights_only=False)
    obs_all, hist_all = [], []

    for cx, cyaw, scen in [PLAN[known.segment]]:
        sys.argv = [sys.argv[0]] + rest
        args = get_args([
            {"name": "--checkpoint_path", "type": str, "default": str(known.checkpoint)},
            {"name": "--scenarios", "type": str, "default": scen},
            {"name": "--eval_seconds", "type": float, "default": 20.},
            {"name": "--command_x", "type": float, "default": cx},
            {"name": "--command_yaw", "type": float, "default": cyaw},
            {"name": "--command_height", "type": float, "default": .15},
            {"name": "--eval_noise", "action": "store_true", "default": False},
            # --push_interval_s is already declared by the base get_args parser;
            # redeclaring it raises ArgumentError on the second segment.
        ])
        cfg = ev.configure(args, {"environment": ck["environment"],
                                  "training": ck["train_cfg"]}, scen)
        cfg.env.obs_history_length = ck["train_cfg"]["policy"]["num_encoder_obs"] // 25
        args.seed = known.seed
        env, _ = task_registry.make_env("robust_v1", args=args, env_cfg=cfg)
        off = QuantConfig(quant_weights=False, quant_obs=False,
                          quant_hidden=False, quant_output=False)
        net = build_from_checkpoint(known.checkpoint, off,
                                    device=env.device, strict_critic=False)
        set_seed(known.seed)
        env.reset()
        with torch.inference_mode():
            for _ in range(known.steps_per_segment):
                o, h = env.get_observations()
                a, _ = net.act_inference(o, h)
                obs_all.append(o.cpu().numpy().copy())
                hist_all.append(h.cpu().numpy().copy())
                env.step(a)
        print(f"  segment cx={cx} yaw={cyaw} {scen}: {len(obs_all)} batches so far",
              flush=True)
        del env, net
        torch.cuda.empty_cache()

    o = np.concatenate(obs_all, 0); h = np.concatenate(hist_all, 0)
    known.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(known.out, obs=o, obs_history=h)
    print(f"wrote {known.out}  obs={o.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
