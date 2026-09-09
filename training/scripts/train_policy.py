#!/usr/bin/env python3
"""Train the wheel-legged policy in Isaac Gym, FP32 or hardware-aware QAT.

One entry point for both so the two arms differ only in the flag, never in the
environment, seed, reward, curriculum or PPO hyper-parameters. Anything else
would make the FP32-vs-QAT comparison an argument instead of a measurement.

The upstream repo is imported by path and never modified: the runner resolves
its policy class with `eval("ActorCriticSequence")` in its own module
namespace, so binding a factory to that name there is enough to swap in the
quantized network. Checkpoints stay state-dict compatible in both directions.

  # FP32 baseline (goal.md section 6)
  train_policy.py --run fp32_baseline --iters 6000

  # QAT fine-tune from that checkpoint (goal.md section 8)
  train_policy.py --run qat_w16a16 --quant W16A16 \
                  --init-from checkpoints/fp32_baseline/model_6000.pt --iters 1500
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import isaacgym  # noqa: F401  must precede torch

REPO = Path(__file__).resolve().parents[2]
# roboaccel_quant lives under quantization/, one component over.
sys.path.insert(0, str(REPO / "quantization"))
QAT_ROOT = REPO / "training"

import torch  # noqa: E402
from wheel_legged_gym.envs import *  # noqa: F401,F403,E402
from wheel_legged_gym.utils import get_args, task_registry  # noqa: E402
from wheel_legged_gym.utils.helpers import class_to_dict  # noqa: E402

from roboaccel_quant.quant_policy import QuantActorCriticSequence  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

PRESETS = {
    "W16A16": QuantConfig(weight_bits=16, act_bits=16),   # the shipped datapath
    "W8A16":  QuantConfig(weight_bits=8, act_bits=16),
    "W8A8":   QuantConfig(weight_bits=8, act_bits=8),
    "W4A8":   QuantConfig(weight_bits=4, act_bits=8),
}

# goal.md section 8: a controlled schedule, only if the direct formulation
# destabilises. Each stage is a separate named experiment.
STAGES = {
    1: dict(quant_weights=True, quant_obs=False, quant_hidden=False, quant_output=False),
    2: dict(quant_weights=True, quant_obs=True, quant_hidden=False, quant_output=False),
    3: dict(quant_weights=True, quant_obs=True, quant_hidden=True, quant_output=False),
    4: dict(quant_weights=True, quant_obs=True, quant_hidden=True, quant_output=True),
}


def parse_quant_off(spec: str) -> dict:
    """"--quant-off weights,actor" -> {"quant_weights": False, "quant_actor": False}"""
    valid = {"weights", "obs", "hidden", "output", "encoder", "actor"}
    out = {}
    for part in filter(None, (x.strip() for x in spec.split(","))):
        if part not in valid:
            raise SystemExit(f"--quant-off: unknown part {part!r}; pick from {sorted(valid)}")
        out[f"quant_{part}"] = False
    return out


def _jsonable(o):
    """class_to_dict can yield tensors (e.g. sampled ranges); json cannot."""
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def install_quant_policy(cfg: QuantConfig) -> None:
    import wheel_legged_gym.rsl_rl.runners.on_policy_runner as opr

    # Must be a CLASS, not a factory function: the runner reads
    # actor_critic_class.is_sequence before instantiating, to decide whether to
    # widen num_critic_obs by latent_dim and supply num_encoder_obs. A closure
    # has no such attribute, and a plain function would also silently skip that
    # branch if the runner ever used getattr with a default.
    class factory(QuantActorCriticSequence):
        def __init__(self, *a, **kw):
            kw.pop("quant", None)
            # A4/A5's causal_conv encoder has no PL descriptor; robust_v1 is MLP.
            if kw.pop("encoder_type", "mlp") != "mlp":
                raise SystemExit("encoder_type must be mlp to stay on the datapath; "
                                 "see SOLID_POLICY_DEPLOYMENT_MAPPING.md")
            super().__init__(*a, quant=cfg, **kw)

    # robust_v1's runner names ActorCriticRobust; the original names
    # ActorCriticSequence. Patch both so the task choice cannot silently
    # reinstate an unquantized policy.
    opr.ActorCriticSequence = factory
    if hasattr(opr, "ActorCriticRobust"):
        opr.ActorCriticRobust = factory


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ideal-rounding", action="store_true",
                    help="train/evaluate against exact round-half-away-from-zero "
                         "instead of the RTL's floor semantics (goal.md section 20)")
    ap.add_argument("--reward-ang-vel", type=float, default=None,
                    help="override rewards.scales.tracking_ang_vel. The task ships "
                         "tracking_lin_vel=2.0 against tracking_ang_vel=1.0, and the "
                         "three-seed result says QAT spends INT8 capacity on the "
                         "reward-dominant axis and starves yaw. Raising this tests "
                         "that directly (goal.md section 19).")
    ap.add_argument("--task", default="robust_v1",
                    help="must match SOLID_FP32 for a valid control")
    ap.add_argument("--quant", default=None, choices=sorted(PRESETS))
    ap.add_argument("--stage", type=int, default=None, choices=sorted(STAGES))
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--quant-off", default="", help="comma list of datapath parts to leave in FP32: weights,obs,hidden,output,encoder,actor (goal.md section 15)")
    ap.add_argument("--max-weight-frac", type=int, default=None)
    ap.add_argument("--act-fracs", type=Path, default=None,
                    help="calibrated per-layer fracs; required for A8")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-interval", type=int, default=250)
    known, rest = ap.parse_known_args()
    if known.ideal_rounding:
        import roboaccel_quant.torch_hw as _th
        _th.IDEAL_ROUNDING = True
        print('IDEAL_ROUNDING enabled: requantizer rounds half away from zero')

    # Hand Isaac Gym the arguments it expects, and only those.
    sys.argv = [sys.argv[0], "--task", known.task, "--headless",
                "--num_envs", str(known.num_envs), "--seed", str(known.seed),
                "--max_iterations", str(known.iters)] + rest
    args = get_args()

    cfg = None
    if known.quant:
        cfg = PRESETS[known.quant]
        over = {}
        if known.act_fracs:
            over["act_fracs"] = json.loads(known.act_fracs.read_text())["act_fracs"]
        elif cfg.act_bits < 16:
            raise SystemExit(
                f"{known.quant} needs --act-fracs: an INT{cfg.act_bits} tensor at "
                "the default frac 8 spans +-0.5 and every activation would clip. "
                "Run scripts/calibrate_act_fracs.py first.")
        if known.stage:
            over.update(STAGES[known.stage])
        over.update(parse_quant_off(known.quant_off))
        if known.max_weight_frac is not None:
            over["max_weight_frac"] = known.max_weight_frac
        cfg = QuantConfig(**{**cfg.__dict__, **over})
        install_quant_policy(cfg)

    log_root = QAT_ROOT / "checkpoints"
    log_root.mkdir(parents=True, exist_ok=True)
    # The override MUST happen before make_env. LeggedRobot builds its reward
    # function list and dt-scales the coefficients at construction, so mutating
    # env_cfg afterwards is a silent no-op -- it printed a convincing message and
    # produced three bit-identical policies from three different reward weights.
    pre_cfg, _ = task_registry.get_cfgs(known.task)
    if known.reward_ang_vel is not None:
        pre_cfg.rewards.scales.tracking_ang_vel = known.reward_ang_vel
    env, env_cfg = task_registry.make_env(name=known.task, args=args, env_cfg=pre_cfg)
    if known.reward_ang_vel is not None:
        # Read back from the CONSTRUCTED env, not from the config object we just
        # wrote, so a no-op fails loudly instead of printing success.
        got = env.reward_scales.get("tracking_ang_vel")
        want = known.reward_ang_vel * env.dt
        if got is None or abs(got - want) > 1e-9:
            raise SystemExit(
                f"reward override did not reach the environment: "
                f"env.reward_scales['tracking_ang_vel']={got}, expected {want} "
                f"(={known.reward_ang_vel} x dt={env.dt}). "
                f"Check that make_env received the modified cfg.")
        print(f"reward override VERIFIED in env: tracking_ang_vel raw="
              f"{known.reward_ang_vel} dt-scaled={got}")
    train_cfg = task_registry.train_cfgs[known.task]
    train_cfg.runner.save_interval = known.save_interval
    train_cfg.runner.max_iterations = known.iters
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=known.task, args=args, train_cfg=train_cfg,
        log_root=str(log_root))
    # make_alg_runner appends <date>_<run_name>; pin it to the run name so
    # reruns and resumes address the same directory.
    runner.log_dir = str(log_root / known.run)
    Path(runner.log_dir).mkdir(parents=True, exist_ok=True)

    # A quantized run that silently built an FP32 policy would produce a
    # plausible-looking QAT number that means nothing. Verify the substitution
    # actually took, rather than trusting that the patch found the right global.
    if cfg is not None:
        ac = runner.alg.actor_critic
        if not isinstance(ac, QuantActorCriticSequence):
            raise SystemExit(
                f"quantization requested but the runner built {type(ac).__name__}; "
                "install_quant_policy patched the wrong module global")
        print(f"quantized policy confirmed: {type(ac).__name__} "
              f"W{cfg.weight_bits}A{cfg.act_bits}")

    if known.init_from:
        sd = dict(torch.load(known.init_from,
                             map_location=runner.device)["model_state_dict"])
        if "log_std" in sd:      # Codex renamed the exploration parameter
            sd["std"] = sd.pop("log_std").exp()
        missing, unexpected = runner.alg.actor_critic.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"load_state_dict missing={missing} unexpected={unexpected}")
            if missing:
                raise SystemExit("refusing to fine-tune from a partial checkpoint")
        print(f"warm-started from {known.init_from}")

    meta = {"run": known.run, "task": known.task,
            "reward_ang_vel": known.reward_ang_vel,
            "quant": known.quant, "stage": known.stage,
            "quant_cfg": cfg.__dict__ if cfg else None,
            "init_from": str(known.init_from) if known.init_from else None,
            "iters": known.iters, "num_envs": known.num_envs, "seed": known.seed,
            "num_obs": env.num_obs, "num_actions": env.num_actions,
            "obs_history_length": env.obs_history_length,
            "latent_dim": train_cfg.policy.latent_dim}
    (Path(runner.log_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # Codex's scripts/evaluate_locomotion.py reads config.json beside the
    # checkpoint and keys off saved["task"], so a run without one cannot be
    # evaluated at all. Write the same shape their train.py does, sourcing the
    # effective configs rather than the CLI so an override cannot be lost.
    cfg_json = {
        "task": known.task,
        "seed": known.seed,
        "run_name": known.run,
        "revision": os.environ.get("SOLID_REVISION", ""),
        "tracked_worktree_dirty": True,
        "source_sha256": {},
        "command": sys.argv,
        "runtime": {"observation_features": int(env.num_obs),
                    "control_dt": float(env.dt),
                    "simulation_device": str(runner.device)},
        "environment": _jsonable(class_to_dict(env_cfg)),
        "training": _jsonable(class_to_dict(train_cfg)),
    }
    (Path(runner.log_dir) / "config.json").write_text(json.dumps(cfg_json, indent=2))
    print(json.dumps(meta, indent=2))

    runner.learn(num_learning_iterations=known.iters, init_at_random_ep_len=True)
    runner.save(os.path.join(runner.log_dir, f"model_{known.iters}.pt"))
    print(f"TRAIN_DONE {runner.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
