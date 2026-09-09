#!/usr/bin/env python3
"""Collect on-policy observations for activation-fraction calibration.

goal.md §11 needs per-layer activation ranges measured on the distribution the
policy actually visits, not on random inputs -- a Gaussian probe puts the
encoder input far outside its real range and produces fractional bits that
waste half the INT8 grid.

The environment is built through Codex's own `configure()` and
`task_registry.make_env`, in the nominal scenario, so the observations here are
drawn from exactly the distribution the evaluator scores against.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

QAT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QAT_ROOT))

import isaacgym  # noqa: F401  must precede torch
import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=101)
    known, rest = ap.parse_known_args()

    from wheel_legged_gym.scripts import evaluate_robustness as ev
    from wheel_legged_gym.utils import get_args, task_registry
    from wheel_legged_gym.utils.helpers import set_seed
    from hwq.quant_policy import build_from_checkpoint
    from hwq.torch_hw import QuantConfig

    sys.argv = [sys.argv[0]] + rest
    args = get_args([
        {"name": "--checkpoint_path", "type": str, "default": str(known.checkpoint)},
        {"name": "--scenarios", "type": str, "default": "nominal"},
        {"name": "--eval_seconds", "type": float, "default": 20.},
        {"name": "--command_x", "type": float, "default": .5},
        {"name": "--command_yaw", "type": float, "default": .5},
        {"name": "--command_height", "type": float, "default": .15},
        {"name": "--eval_noise", "action": "store_true", "default": False},
    ])
    ckpt = torch.load(str(known.checkpoint), map_location="cpu", weights_only=False)
    cfg = ev.configure(args, {"environment": ckpt["environment"],
                              "training": ckpt["train_cfg"]}, "nominal")
    cfg.env.obs_history_length = ckpt["train_cfg"]["policy"]["num_encoder_obs"] // 25
    args.seed = known.seed
    env, _ = task_registry.make_env("robust_v1", args=args, env_cfg=cfg)

    off = QuantConfig(quant_weights=False, quant_obs=False,
                      quant_hidden=False, quant_output=False)
    net = build_from_checkpoint(known.checkpoint, off,
                                device=env.device, strict_critic=False)
    set_seed(known.seed)
    env.reset()

    obs_all, hist_all = [], []
    with torch.inference_mode():
        for _ in range(known.steps):
            obs, history = env.get_observations()
            actions, _ = net.act_inference(obs, history)
            obs_all.append(obs.cpu().numpy().copy())
            hist_all.append(history.cpu().numpy().copy())
            env.step(actions)

    o = np.concatenate(obs_all, 0)
    h = np.concatenate(hist_all, 0)
    # Subsample: 400 steps x num_envs is far more than calibration needs.
    idx = np.random.default_rng(0).choice(o.shape[0], size=min(20000, o.shape[0]),
                                          replace=False)
    known.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(known.out, obs=o[idx], obs_history=h[idx])
    print(f"wrote {known.out}  obs={o[idx].shape} hist={h[idx].shape}")
    print(f"  |obs|max={np.abs(o).max():.4f}  |hist|max={np.abs(h).max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
