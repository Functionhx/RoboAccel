#!/usr/bin/env python3
"""FP32-teacher behaviour preservation for PPO QAT (goal6.md Experiment 2).

The objective is

    L = L_PPO + beta * KL(pi_teacher || pi_student)

with a frozen FP32 copy of the warm-start checkpoint as the teacher, evaluated
on the student's own on-policy states.

How it is injected, and why this way
------------------------------------
ppo.py builds its loss as

    loss = surrogate + c_v * value_loss - c_H * entropy.mean()

and takes `entropy` from the policy module. So returning

    H_hat = H - (beta / c_H) * KL(teacher || student)

from that one property makes the loss

    surrogate + c_v * value_loss - c_H * H.mean() + beta * KL.mean()

which is the objective above, exactly. Autograd differentiates the real KL --
no gradient is derived by hand.

The alternative was to vendor a copy of PPO.update() with one line added. That
copy would be ~100 lines of transcribed control flow that silently rots the
first time upstream touches its own file, and this project has already been
bitten once by a config mutation that looked applied and was not. Hooking a
single documented property modifies nothing upstream and copies nothing
upstream, so it cannot drift.

The cost is that the seam is not obvious from reading ppo.py. Two things pay
for that: the algebraic identity is checked numerically against a directly
constructed L_PPO + beta*KL by scripts/check_teacher_kl.py before any run
starts, and the true entropy is logged separately so the Policy/entropy
diagnostic still means what it says.
"""
from __future__ import annotations

import torch


def mean_only_kl(mu_t, sigma_t, mu_s):
    """The mean-matching half of KL(teacher||student), at the teacher's sigma.

    The full Gaussian KL has an escape hatch. Minimising

        log(s_s/s_t) + (s_t^2 + d^2) / (2 s_s^2) - 1/2

    over the student's sigma gives s_s^2 = s_t^2 + d^2, so a student that
    cannot match the teacher's mean can shrink the objective by widening its
    action distribution instead. Run literally, that is what happened: the
    exploration std rose 1.60x in 97 iterations and the mean term's share of
    the KL fell from 1.000 to 0.587, while the mean itself barely moved.

    Holding sigma at the teacher's value removes the hatch, and costs nothing
    that matters: the exploration std is a training-only parameter. What gets
    deployed is act_inference, which returns the mean. Preserving the
    teacher's *behaviour* means preserving its mean.
    """
    return ((mu_t - mu_s).square() / (2.0 * sigma_t.square())).sum(dim=-1)


def gaussian_kl(mu_t, sigma_t, mu_s, sigma_s):
    """KL(teacher || student) for diagonal Gaussians, summed over action dims.

    Same algebraic form and same argument order as ppo.py's own KL between the
    old and new policies, so "KL" means one thing throughout this project.
    """
    return (torch.log(sigma_s / sigma_t)
            + (sigma_t.square() + (mu_t - mu_s).square()) / (2.0 * sigma_s.square())
            - 0.5).sum(dim=-1)


class TeacherKL:
    """One attached teacher plus the running statistics for its term."""

    def __init__(self, teacher, beta, entropy_coef, actor_parameters=(),
                 grad_probe_stride=50, mode="mean"):
        if entropy_coef == 0:
            raise ValueError(
                "the teacher term is injected through -c_H * entropy, so a zero "
                "entropy coefficient would silently drop it entirely")
        if not (beta > 0):
            raise ValueError("beta must be positive; a zero-beta arm is the control")
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if mode not in ("mean", "full"):
            raise ValueError("teacher KL mode must be 'mean' or 'full'")
        self.mode = mode
        self.beta = float(beta)
        self.entropy_coef = float(entropy_coef)
        self.actor_parameters = list(actor_parameters)
        self.grad_probe_stride = int(grad_probe_stride)
        self.calls = 0
        self.reset_stats()

    def reset_stats(self):
        self._sum = dict(teacher_kl=0.0, teacher_kl_mean_term=0.0,
                         true_entropy=0.0, teacher_loss=0.0)
        self._n = 0
        self._grad_sum, self._grad_n = 0.0, 0

    def drain(self) -> dict:
        """Averages since the last drain, as diagnostics keys, then reset."""
        if self._n == 0:
            return {}
        out = {f"Teacher/{k}": v / self._n for k, v in self._sum.items()}
        if self._grad_n:
            out["Teacher/gradient_norm"] = self._grad_sum / self._grad_n
        self.reset_stats()
        return out

    def penalty(self, student):
        """Per-sample KL(teacher||student) on the student's current batch."""
        obs, hist = student._last_obs, student._last_hist
        if obs is None:
            raise RuntimeError("teacher KL requested before any forward pass")
        with torch.no_grad():
            mu_t = self.teacher.act_inference(obs, hist)[0]
            sigma_t = self.teacher.std.expand_as(mu_t)
        mu_s = student.distribution.mean
        sigma_s = student.distribution.stddev
        kl = (mean_only_kl(mu_t, sigma_t, mu_s) if self.mode == "mean"
              else gaussian_kl(mu_t, sigma_t, mu_s, sigma_s))

        with torch.no_grad():
            mean_term = ((mu_t - mu_s).square() / (2.0 * sigma_s.square())).sum(-1).mean()
            self._sum["teacher_kl"] += kl.mean().item()
            self._sum["teacher_kl_mean_term"] += mean_term.item()
            self._sum["teacher_loss"] += self.beta * kl.mean().item()
            self._n += 1

        # How much of the gradient the teacher term is actually responsible for.
        # Strided, because it costs one extra backward through the actor.
        if self.actor_parameters and self.calls % self.grad_probe_stride == 0:
            g = torch.autograd.grad(self.beta * kl.mean(), self.actor_parameters,
                                    retain_graph=True, allow_unused=True)
            norm = torch.sqrt(sum((x.square().sum() for x in g if x is not None),
                                  start=torch.zeros((), device=kl.device)))
            self._grad_sum += float(norm)
            self._grad_n += 1
        self.calls += 1
        return kl


def install_teacher_kl_class() -> None:
    """Wrap whatever policy class the runner is about to build.

    Runs AFTER install_quant_policy / install_encoder_freeze, so it composes
    with them rather than replacing them. The teacher itself is attached later,
    once --init-from has loaded; until then the property returns the plain
    entropy and the run is an ordinary PPO run.
    """
    import wheel_legged_gym.rsl_rl.runners.on_policy_runner as opr

    wrapped = {}
    for name in ("ActorCriticSequence", "ActorCriticRobust"):
        base = getattr(opr, name, None)
        if base is None:
            continue
        if base not in wrapped:
            class with_teacher(base):                 # noqa: N801  upstream style
                _teacher_kl = None
                _last_obs = None
                _last_hist = None

                def update_distribution(self, observations, observation_history):
                    # The teacher must see the STUDENT'S on-policy states, and
                    # the entropy property has no arguments, so they are stashed
                    # here rather than recomputed from storage.
                    self._last_obs = observations
                    self._last_hist = observation_history
                    super().update_distribution(observations, observation_history)

                @property
                def entropy(self):
                    base_entropy = super().entropy
                    tk = self._teacher_kl
                    if tk is None:
                        return base_entropy
                    with torch.no_grad():
                        tk._sum["true_entropy"] += base_entropy.mean().item()
                    return base_entropy - (tk.beta / tk.entropy_coef) * tk.penalty(self)

            with_teacher.__name__ = f"TeacherKL_{base.__name__}"
            wrapped[base] = with_teacher
        setattr(opr, name, wrapped[base])
