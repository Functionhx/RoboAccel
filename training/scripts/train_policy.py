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

from roboaccel_quant.quant_policy import QuantActorCriticSequence, build_from_checkpoint  # noqa: E402
from roboaccel_quant.teacher_kl import TeacherKL, install_teacher_kl_class  # noqa: E402
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


def install_encoder_freeze() -> None:
    """Freeze the history encoder the way the upstream residual controller does.

    ppo.py reads `actor_critic.encoder_frozen` *in its constructor* to decide
    whether to build `extra_optimizer` at all, and refuses a nonzero auxiliary
    learning rate when the flag is set. Setting the attribute on the finished
    runner would therefore be a silent no-op -- the auxiliary optimizer would
    already exist and would keep stepping. It has to be an attribute of the
    class the runner is about to instantiate.

    Must run AFTER install_quant_policy, so it wraps the quantized factory
    rather than the class the factory replaced.
    """
    import wheel_legged_gym.rsl_rl.runners.on_policy_runner as opr

    wrapped = {}
    for name in ("ActorCriticSequence", "ActorCriticRobust"):
        base = getattr(opr, name, None)
        if base is None:
            continue
        if base not in wrapped:
            class frozen(base):                       # noqa: N801  matches upstream style
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.encoder_frozen = True
                    self.encoder.requires_grad_(False)

            frozen.__name__ = f"FrozenEncoder_{base.__name__}"
            wrapped[base] = frozen
        # Both names must resolve to the SAME subclass: patching them
        # separately would build two distinct classes and make which one the
        # runner picked depend on policy_class_name.
        setattr(opr, name, wrapped[base])


def param_digest(module) -> str:
    """SHA-256 over the raw bytes of a module's parameters, in name order."""
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(module.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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
    ap.add_argument("--schedule", default=None, choices=["adaptive", "fixed"],
                    help="algorithm.schedule. 'fixed' disables the adaptive "
                         "learning-rate controller entirely, which is the only "
                         "way to give a W8A8 run the same step size as the FP32 "
                         "control -- under 'adaptive' the quantization KL noise "
                         "floor pins it at 1e-5 (goal4.md section 3).")
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="algorithm.learning_rate. Only meaningful with "
                         "--schedule fixed; otherwise the controller overwrites it.")
    ap.add_argument("--desired-kl", type=float, default=None,
                    help="override algorithm.desired_kl. rsl_rl divides the "
                         "actor learning rate by 1.5 whenever the measured "
                         "policy KL exceeds 2*desired_kl, with a hard floor of "
                         "1e-5. Under W8A8 fake-quant a single 1e-5 weight step "
                         "already produces KL 0.067 (scripts/kl_amplification.py), "
                         "which is 6.7x the shipped 0.005*2 threshold -- so the "
                         "controller pinned the learning rate at the floor for "
                         "2000/2000 iterations and no amount of reward shaping "
                         "could move the policy. Raising this above the "
                         "quantization KL noise floor tests that directly "
                         "(goal4.md section 3).")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="freeze every encoder parameter and delete its "
                         "optimizer (goal6.md Experiment 1). The encoder has "
                         "its own Adam at algorithm.extra_learning_rate and is "
                         "NEVER touched by the adaptive-KL controller, so when "
                         "W8A8 pins the actor at the 1e-5 floor the encoder "
                         "keeps stepping at 1e-3 -- a 100:1 ratio. The logged "
                         "Encoder/policy_kl says that update alone moves the "
                         "policy by a median KL of 0.95 per iteration, ~100x "
                         "the actor's entire 0.01 budget. This flag removes "
                         "that disturbance so it can be tested causally.")
    ap.add_argument("--teacher-kl", type=float, default=None, metavar="BETA",
                    help="add beta*KL(pi_teacher||pi_student) to the PPO loss, "
                         "with a frozen FP32 copy of the warm start as teacher "
                         "(goal6.md Experiment 2). PPO's objective contains no "
                         "term referring to the reference policy, so a QAT "
                         "student is free to drift anywhere the reward permits; "
                         "goal5 showed that removing the learning-rate brake "
                         "without adding such a term made the policy worse. "
                         "The teacher is evaluated on the STUDENT'S on-policy "
                         "states. See hwq/teacher_kl.py for the injection and "
                         "scripts/check_teacher_kl.py for its proof.")
    ap.add_argument("--teacher-kl-mode", default="mean", choices=["mean", "full"],
                    help="'mean' matches the teacher's action mean at the "
                         "teacher's sigma, which is what actually gets "
                         "deployed. 'full' is the literal Gaussian KL and is "
                         "degenerate: the student widens its exploration std "
                         "instead of matching the mean (see hwq/teacher_kl.py). "
                         "Kept so the degeneracy can be reproduced.")
    ap.add_argument("--teacher-from", type=Path, default=None,
                    help="teacher checkpoint; defaults to --init-from, which is "
                         "what makes it a behaviour-preservation term rather "
                         "than distillation from a different policy")
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
        import hwq.torch_hw as _th
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
    if known.desired_kl is not None:
        train_cfg.algorithm.desired_kl = known.desired_kl
    if known.schedule is not None:
        train_cfg.algorithm.schedule = known.schedule
    if known.learning_rate is not None:
        train_cfg.algorithm.learning_rate = known.learning_rate
    if known.freeze_encoder:
        # Order matters: this wraps whatever install_quant_policy left bound.
        install_encoder_freeze()
        # ppo.py raises if encoder_frozen is set while this is nonzero, so the
        # two always travel together and a half-applied freeze cannot start.
        train_cfg.algorithm.extra_learning_rate = 0.0
    if known.teacher_kl is not None:
        # Wraps last, so it composes with the quantized factory and the freeze
        # rather than replacing either.
        install_teacher_kl_class()
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

    # Same failure mode as the reward override: mutating a config object after
    # the consumer has read it is a silent no-op. Read back from the built
    # algorithm, not from the config we just wrote.
    if known.desired_kl is not None:
        got = runner.alg.desired_kl
        if got is None or abs(got - known.desired_kl) > 1e-12:
            raise SystemExit(
                f"desired_kl override did not reach the algorithm: "
                f"runner.alg.desired_kl={got}, expected {known.desired_kl}")
        print(f"desired_kl override VERIFIED in alg: {got} "
              f"(lr is throttled above KL {2 * got})")

    if known.schedule is not None and runner.alg.schedule != known.schedule:
        raise SystemExit(f"schedule override did not reach the algorithm: "
                         f"runner.alg.schedule={runner.alg.schedule!r}")
    if known.learning_rate is not None:
        got = [g["lr"] for g in runner.alg.optimizer.param_groups]
        if any(abs(x - known.learning_rate) > 1e-12 for x in got):
            raise SystemExit(f"learning_rate override did not reach the optimizer: "
                             f"param_group lrs={got}, expected {known.learning_rate}")
        print(f"learning_rate override VERIFIED in optimizer: {got[0]:.3e} "
              f"schedule={runner.alg.schedule}")

    if known.freeze_encoder:
        # Four independent facts, because any one of them alone can be true
        # while the encoder still moves: the flag reached the module, the
        # auxiliary optimizer was never built, every encoder parameter is
        # detached from autograd, and PPO's own encoder parameter list is
        # empty (it is what the auxiliary gradient clip would have used).
        ac = runner.alg.actor_critic
        if not getattr(ac, "encoder_frozen", False):
            raise SystemExit("freeze requested but actor_critic.encoder_frozen "
                             "is not set; install_encoder_freeze patched the "
                             "wrong module global")
        if runner.alg.extra_optimizer is not None:
            raise SystemExit("freeze requested but PPO still built "
                             "extra_optimizer; the encoder would keep stepping")
        trainable = [n for n, q in ac.encoder.named_parameters() if q.requires_grad]
        if trainable:
            raise SystemExit(f"encoder parameters still require grad: {trainable}")
        if runner.alg.encoder_parameters:
            raise SystemExit("PPO still holds encoder parameters for clipping")
        print(f"encoder freeze VERIFIED: encoder_frozen=True, extra_optimizer=None, "
              f"{sum(q.numel() for q in ac.encoder.parameters())} encoder params "
              f"requires_grad=False, extra_learning_rate="
              f"{train_cfg.algorithm.extra_learning_rate}")

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

    ac = runner.alg.actor_critic
    encoder_sha_start = param_digest(ac.encoder)
    actor_sha_start = param_digest(ac.actor)
    print(f"encoder digest at iteration 0: {encoder_sha_start}")
    print(f"actor   digest at iteration 0: {actor_sha_start}")

    if known.freeze_encoder:
        # A start/end comparison cannot distinguish "never moved" from "moved
        # and came back", and it reports the failure 2000 iterations too late.
        # Check around every update for the first ten, then every hundredth.
        _update = runner.alg.update
        _state = {"n": 0, "actor_moved": False}

        def guarded_update():
            n = _state["n"]
            watch = n < 10 or n % 100 == 0
            before = param_digest(ac.encoder) if watch else None
            a_before = param_digest(ac.actor) if watch else None
            out = _update()
            if watch:
                after = param_digest(ac.encoder)
                if after != before:
                    raise SystemExit(
                        f"ENCODER MOVED during update {n}: {before[:16]} -> "
                        f"{after[:16]}. The freeze is not effective; abort "
                        f"rather than produce an uninterpretable arm.")
                if param_digest(ac.actor) != a_before:
                    _state["actor_moved"] = True
            _state["n"] = n + 1
            if n == 9 and not _state["actor_moved"]:
                raise SystemExit(
                    "actor parameters did not change in ten updates: this arm "
                    "would measure a frozen network, not a frozen encoder")
            return out

        runner.alg.update = guarded_update

    if known.teacher_kl is not None:
        teacher_path = known.teacher_from or known.init_from
        if teacher_path is None:
            raise SystemExit("--teacher-kl needs a teacher: pass --init-from "
                             "(preferred) or --teacher-from")
        teacher = build_from_checkpoint(
            teacher_path,
            quant=QuantConfig(quant_weights=False, quant_obs=False,
                              quant_hidden=False, quant_output=False),
            device=runner.device)
        tk = TeacherKL(teacher, known.teacher_kl, runner.alg.entropy_coef,
                       list(ac.actor.parameters()) + [ac.std],
                       mode=known.teacher_kl_mode)

        # goal6.md: "Verify that beta=1.0 actually changes gradients. Treat a
        # no-op configuration as a bug." The offline proof in
        # scripts/check_teacher_kl.py runs on CPU against calibration states;
        # this repeats it on this run's device, this run's policy and this
        # run's own first observations, so a wiring mistake cannot survive.
        probe_obs, probe_hist = env.get_observations()
        probe_obs = probe_obs[:512].to(runner.device)
        probe_hist = probe_hist[:512].to(runner.device)

        def _actor_grad_norm(attached):
            ac._teacher_kl = tk if attached else None
            ac.update_distribution(probe_obs, probe_hist)
            ac.zero_grad(set_to_none=True)
            (-runner.alg.entropy_coef * ac.entropy.mean()).backward()
            n = torch.sqrt(sum((q.grad.square().sum() for q in tk.actor_parameters
                                if q.grad is not None),
                               start=torch.zeros((), device=runner.device)))
            ac.zero_grad(set_to_none=True)
            return float(n)

        off = _actor_grad_norm(False)
        on = _actor_grad_norm(True)
        if not on > off * 1.0001:
            raise SystemExit(
                f"teacher term is a no-op: actor gradient norm {off:.6g} with "
                f"beta=0 vs {on:.6g} with beta={known.teacher_kl}. Refusing to "
                f"run an arm that would look like a teacher experiment and be "
                f"an ordinary PPO run.")
        teacher_kl_at_start = tk._sum["teacher_kl"] / max(tk._n, 1)
        ac._teacher_kl = tk
        tk.reset_stats(); tk.calls = 0
        print(f"teacher KL VERIFIED: beta={known.teacher_kl} teacher={teacher_path} "
              f"actor grad-norm {off:.4g} -> {on:.4g} ({on / max(off, 1e-12):.1f}x), "
              f"KL(teacher||student) at warm start = {teacher_kl_at_start:.4f}")

        _pre_teacher_update = runner.alg.update

        def teacher_logged_update():
            out = _pre_teacher_update()
            # log() reads alg.diagnostics after update() returns, so injecting
            # here puts the teacher terms in metrics.jsonl beside PPO's own.
            stats = tk.drain()
            runner.alg.diagnostics.update(stats)
            # PPO averaged the HIJACKED entropy into Policy/entropy, which
            # would make that column mean something different in this arm than
            # in every other one. Keep the hijacked value under a name that
            # says what it is, and restore the column to the real entropy.
            if "Teacher/true_entropy" in stats:
                runner.alg.diagnostics["Teacher/injected_entropy"] = \
                    runner.alg.diagnostics["Policy/entropy"]
                runner.alg.diagnostics["Policy/entropy"] = stats["Teacher/true_entropy"]
            return out

        runner.alg.update = teacher_logged_update

    meta = {"run": known.run, "task": known.task,
            "reward_ang_vel": known.reward_ang_vel,
            "desired_kl": known.desired_kl,
            "schedule": known.schedule,
            "learning_rate": known.learning_rate,
            "freeze_encoder": bool(known.freeze_encoder),
            "teacher_kl_beta": known.teacher_kl,
            "teacher_kl_mode": known.teacher_kl_mode,
            "teacher_from": str(known.teacher_from or known.init_from)
                            if known.teacher_kl is not None else None,
            "extra_learning_rate": float(train_cfg.algorithm.extra_learning_rate),
            "encoder_sha256_start": encoder_sha_start,
            "actor_sha256_start": actor_sha_start,
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
    encoder_sha_end, actor_sha_end = param_digest(ac.encoder), param_digest(ac.actor)
    print(f"encoder digest at iteration {known.iters}: {encoder_sha_end} "
          f"({'UNCHANGED' if encoder_sha_end == encoder_sha_start else 'CHANGED'})")
    print(f"actor   digest at iteration {known.iters}: {actor_sha_end} "
          f"({'UNCHANGED' if actor_sha_end == actor_sha_start else 'CHANGED'})")
    if known.freeze_encoder and encoder_sha_end != encoder_sha_start:
        raise SystemExit("encoder changed over training despite --freeze-encoder")
    if actor_sha_end == actor_sha_start:
        raise SystemExit("actor did not change over training; nothing was learned")
    (Path(runner.log_dir) / "digests.json").write_text(json.dumps(
        {"encoder_sha256_start": encoder_sha_start,
         "encoder_sha256_end": encoder_sha_end,
         "actor_sha256_start": actor_sha_start,
         "actor_sha256_end": actor_sha_end,
         "freeze_encoder": bool(known.freeze_encoder)}, indent=2))
    print(f"TRAIN_DONE {runner.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
