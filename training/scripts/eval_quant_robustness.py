#!/usr/bin/env python3
"""Run a quantized policy through the *Codex* robustness evaluator.

goal.md §6 forbids standing up a competing second benchmark, and §31's main
table is only meaningful if every arm -- FP32, PTQ, QAT, exact integer -- is
scored by identical code on identical scenarios. So rather than reimplement the
harness, this rebinds one module global inside
`plane/wheel_legged_gym/scripts/evaluate_robustness.py` and lets it drive.

The seam is real, not a hack around a hostile interface:

  * the evaluator picks its policy class by name from a dict of module globals,
    resolved at call time, so rebinding `ActorCriticRobust` substitutes cleanly;
  * it then calls `model.act_inference(obs, history) -> (actions, latent)`,
    which is exactly `QuantActorCriticSequence`'s signature.

Codex's file is never modified. Scenarios, metrics, summarisation, seeding,
episode accounting and output format are all still its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# roboaccel_quant lives under quantization/, one component over.
sys.path.insert(0, str(REPO / "quantization"))
QAT_ROOT = REPO / "training"

import isaacgym  # noqa: F401  must precede torch
import torch

from roboaccel_quant.torch_hw import QuantConfig
from roboaccel_quant.quant_policy import QuantActorCriticSequence


def make_factory(cfg: QuantConfig, use_integer: bool):
    """Return a class-like callable the evaluator can instantiate."""

    class _Injected(QuantActorCriticSequence):
        def __init__(self, num_obs, num_critic_obs, num_actions, **kw):
            kw.pop("encoder_type", None)          # MLP only; A4/A5 are off-datapath
            kw.pop("class_name", None)
            super().__init__(num_obs, num_critic_obs, num_actions,
                             quant=cfg, **kw)
            self._fixed = None
            self._use_integer = use_integer

        def load_state_dict(self, state, strict=True):
            state = dict(state)
            if "log_std" in state:                # Codex renamed std -> log_std
                state["std"] = state.pop("log_std").exp()
            out = super().load_state_dict(state, strict=strict)
            if self._use_integer:
                self._fixed = self.to_fixed_ref()
            return out

        def act_inference(self, observations, observation_history):
            if not self._use_integer:
                return super().act_inference(observations, observation_history)
            # goal.md §23: the arbiter is the integer reference, one sample per
            # PL sequence, exactly as the board runs it.
            import numpy as np
            o = observations.detach().cpu().numpy().astype(np.float32).astype(np.float64)
            h = observation_history.detach().cpu().numpy().astype(np.float32).astype(np.float64)
            acts = np.stack([self._fixed(o[i], h[i]) for i in range(o.shape[0])])
            a = torch.as_tensor(acts, dtype=observations.dtype,
                                device=observations.device)
            return a, self.latent if self.latent is not None else \
                torch.zeros(o.shape[0], self.latent_dim, device=observations.device)

    return _Injected


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--weight-bits", type=int, default=16)
    ap.add_argument("--act-bits", type=int, default=16)
    ap.add_argument("--act-fracs", type=Path, default=None,
                    help="calibrated per-layer activation fracs (required below A16)")
    ap.add_argument("--integer", action="store_true",
                    help="run the bit-exact numpy integer reference instead of fake-quant")
    ap.add_argument("--fp32", action="store_true",
                    help="disable quantization entirely (sanity control)")
    known, rest = ap.parse_known_args()

    if known.fp32:
        cfg = QuantConfig(quant_weights=False, quant_obs=False,
                          quant_hidden=False, quant_output=False)
    else:
        kw = {}
        if known.act_fracs:
            data = json.loads(known.act_fracs.read_text())
            kw["act_fracs"] = data.get("act_fracs", data)
            if "obs" in kw["act_fracs"]:
                kw["obs_frac"] = kw["act_fracs"]["obs"]
        elif known.act_bits < 16:
            raise SystemExit(
                "--act-bits < 16 needs --act-fracs: a fixed Q8.8 grid spans +-128 "
                "with an LSB of 1/256, which INT8 cannot cover for this network. "
                "Run scripts/calibrate_act_fracs.py first.")
        cfg = QuantConfig(weight_bits=known.weight_bits,
                          act_bits=known.act_bits, **kw)

    # Import Codex's evaluator and substitute the policy class it looks up.
    from wheel_legged_gym.scripts import evaluate_robustness as ev
    injected = make_factory(cfg, known.integer)
    ev.ActorCriticRobust = injected
    ev.ActorCriticSequence = injected

    # Same parameter list the evaluator declares under its own __main__, so the
    # CLI surface and every default stay identical to a native invocation.
    parameters = [
        {"name": "--checkpoint_path", "type": str, "default": ""},
        {"name": "--output", "type": str, "default": "evaluation/results"},
        {"name": "--scenarios", "type": str, "default": ev.DEFAULT_SCENARIOS},
        {"name": "--eval_seeds", "type": str, "default": "1,2,3"},
        {"name": "--episodes_per_env", "type": int, "default": 2},
        {"name": "--eval_seconds", "type": float, "default": 20.},
        {"name": "--command_x", "type": float, "default": .5},
        {"name": "--command_yaw", "type": float, "default": .5},
        {"name": "--command_height", "type": float, "default": .15},
        {"name": "--eval_noise", "action": "store_true", "default": False},
    ]
    from wheel_legged_gym.utils import get_args
    sys.argv = [sys.argv[0]] + rest
    return ev.main(get_args(parameters))


if __name__ == "__main__":
    raise SystemExit(main())
