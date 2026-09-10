#!/usr/bin/env python3
"""Prove the goal6 Experiment-2 teacher term is what it claims to be.

roboaccel_quant/teacher_kl.py injects `L_PPO + beta*KL(teacher||student)` by returning a
modified entropy from the policy module. That is an algebraic identity, and an
algebraic identity that is only asserted in a docstring is exactly the kind of
thing this project has already been burned by. So it is checked numerically,
against a directly constructed objective, before any training run starts:

  1. IDENTITY   grad of the seam objective == grad of surrogate + c_v*value
                - c_H*H + beta*KL, elementwise, to float tolerance.
  2. NOT A NO-OP  beta=1 gradients differ materially from beta=0 gradients.
  3. TEACHER IS FROZEN AND FP32  no teacher parameter receives gradient, and
                the teacher's own output is unquantized.
  4. STATES ARE THE STUDENT'S  the teacher is evaluated on the same tensors the
                student just consumed, not on a replayed or stale batch.

Exit code 0 only if all four hold.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

import isaacgym  # noqa: F401  must precede torch
import numpy as np  # noqa: E402
import torch  # noqa: E402

from roboaccel_quant.quant_policy import QuantActorCriticSequence, build_from_checkpoint  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402
from roboaccel_quant.teacher_kl import (TeacherKL, gaussian_kl, mean_only_kl,  # noqa: E402
                            install_teacher_kl_class)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/frozen/SOLID_FP32_V2.pt")
    ap.add_argument("--act-fracs", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/act_fracs_a8.json")
    ap.add_argument("--obs", type=Path, default=QUANT_ROOT / "artifacts/v2/calib_obs.npz")
    ap.add_argument("--mode", default="mean", choices=["mean", "full"],
                    help="must match the mode the run will use; the identity "
                         "is only meaningful against the same objective")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-loss-coef", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    torch.manual_seed(0)
    fr = json.loads(a.act_fracs.read_text())["act_fracs"]
    qcfg = QuantConfig(weight_bits=8, act_bits=8, act_fracs=fr, obs_frac=fr["obs"])
    fp32 = QuantConfig(quant_weights=False, quant_obs=False,
                       quant_hidden=False, quant_output=False)

    # The class the runner would actually build, with the same wrapper chain.
    install_teacher_kl_class()
    import wheel_legged_gym.rsl_rl.runners.on_policy_runner as opr
    Wrapped = opr.ActorCriticSequence
    if not issubclass(Wrapped, QuantActorCriticSequence):
        # Nothing quantized was installed first, so wrap by hand exactly as
        # train_policy.py would.
        class _Q(QuantActorCriticSequence):
            def __init__(self, *x, **k):
                k.pop("quant", None); k.pop("encoder_type", None)
                super().__init__(*x, quant=qcfg, **k)
        opr.ActorCriticSequence = _Q
        opr.ActorCriticRobust = _Q
        install_teacher_kl_class()
        Wrapped = opr.ActorCriticSequence

    ck = torch.load(str(a.checkpoint), map_location="cpu", weights_only=False)
    sd = dict(ck.get("model_state_dict", ck))
    if "log_std" in sd:
        sd["std"] = sd.pop("log_std").exp()
    dims = dict(num_obs=int(sd["actor.0.weight"].shape[1]) - int(sd["encoder.4.weight"].shape[0]),
                num_critic_obs=int(sd["critic.0.weight"].shape[1]),
                num_actions=int(sd["actor.6.weight"].shape[0]),
                num_encoder_obs=int(sd["encoder.0.weight"].shape[1]),
                latent_dim=int(sd["encoder.4.weight"].shape[0]))
    student = Wrapped(**dims)
    student.load_state_dict(sd, strict=True)
    student.train()
    teacher = build_from_checkpoint(a.checkpoint, quant=fp32, device="cpu")

    d = np.load(a.obs)
    obs = torch.as_tensor(d["obs"][: a.batch])
    hist = torch.as_tensor(d["obs_history"][: a.batch])

    actor_params = [p for p in student.actor.parameters()] + [student.std]
    tk = TeacherKL(teacher, a.beta, a.entropy_coef, actor_params,
                   grad_probe_stride=1, mode=a.mode)

    # A stand-in for the two PPO terms the teacher does not touch. Their exact
    # values are irrelevant: what matters is that they carry gradient into the
    # same parameters, so the identity is tested on a realistic graph.
    def ppo_like(net):
        net.update_distribution(obs, hist)
        mu = net.distribution.mean
        surrogate = (mu.square().mean())
        value = net.evaluate(torch.zeros(obs.shape[0], dims["num_critic_obs"])).square().mean()
        return surrogate, value

    def grads(objective):
        student.zero_grad(set_to_none=True)
        objective.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in actor_params]

    # ---- 1. identity ----------------------------------------------------
    student._teacher_kl = tk
    surrogate, value = ppo_like(student)
    seam_entropy = student.entropy                     # H - (beta/c_H)*KL
    seam_loss = surrogate + a.value_loss_coef * value - a.entropy_coef * seam_entropy.mean()
    g_seam = grads(seam_loss)

    student._teacher_kl = None
    surrogate, value = ppo_like(student)
    true_entropy = student.entropy
    with torch.no_grad():
        mu_t = teacher.act_inference(obs, hist)[0]
        sigma_t = teacher.std.expand_as(mu_t)
    kl = (mean_only_kl(mu_t, sigma_t, student.distribution.mean)
          if a.mode == "mean"
          else gaussian_kl(mu_t, sigma_t, student.distribution.mean,
                           student.distribution.stddev))
    direct_loss = (surrogate + a.value_loss_coef * value
                   - a.entropy_coef * true_entropy.mean() + a.beta * kl.mean())
    g_direct = grads(direct_loss)

    worst, scale = 0.0, 0.0
    for x, y in zip(g_seam, g_direct):
        worst = max(worst, float((x - y).abs().max()))
        scale = max(scale, float(y.abs().max()))
    identity_ok = worst <= 1e-6 * max(scale, 1.0)

    # ---- 2. not a no-op --------------------------------------------------
    surrogate, value = ppo_like(student)
    base_loss = surrogate + a.value_loss_coef * value - a.entropy_coef * student.entropy.mean()
    g_beta0 = grads(base_loss)
    delta = max(float((x - y).abs().max()) for x, y in zip(g_direct, g_beta0))
    rel = delta / max(scale, 1e-12)
    not_noop = delta > 1e-8 and rel > 1e-3

    # ---- 3. teacher frozen, and executing in FP32 ------------------------
    teacher_grads = [n for n, p in teacher.named_parameters() if p.grad is not None]
    teacher_frozen = not teacher_grads and not any(p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        mu_s_q = student.act_inference(obs, hist)[0]
    teacher_is_fp32 = float((mu_t - mu_s_q).abs().max()) > 0   # differs => not the same path

    # ---- 4. the teacher saw the student's own states ---------------------
    states_ok = student._last_obs is obs and student._last_hist is hist

    res = dict(identity_max_abs_grad_diff=worst, gradient_scale=scale,
               identity_ok=bool(identity_ok),
               beta0_vs_beta_max_abs_grad_diff=delta, relative_change=rel,
               not_a_noop=bool(not_noop),
               teacher_frozen=bool(teacher_frozen),
               teacher_is_separate_fp32_path=bool(teacher_is_fp32),
               teacher_sees_student_states=bool(states_ok),
               mean_teacher_kl=float(kl.mean()),
               mean_true_entropy=float(true_entropy.mean()),
               beta=a.beta, mode=a.mode, entropy_coef=a.entropy_coef,
               batch=a.batch)
    print(json.dumps(res, indent=2))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))

    ok = identity_ok and not_noop and teacher_frozen and teacher_is_fp32 and states_ok
    print("TEACHER_KL_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
