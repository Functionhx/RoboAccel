#!/usr/bin/env python3
"""Why the QAT run sat at the learning-rate floor for 100% of training.

rsl_rl adapts the actor/critic learning rate from the PPO policy KL:

    if kl_mean > desired_kl * 2.0:   lr = max(1e-5, lr / 1.5)
    elif kl_mean < desired_kl / 2.0: lr = min(1e-2, lr * 1.5)

with desired_kl = 0.005 for this task, so the downward branch fires whenever the
measured KL exceeds 0.01. In the W8A8 run it fired on every
one of 2000 iterations: median measured KL 0.065 against that 0.01 threshold, and
the learning rate stayed pinned at the 1e-5 floor for 2000/2000 iterations.
The FP32 control, same task, same seed, same everything else, hit the floor on
10/2000 and ran at a median 2.6e-4.

The controller assumes KL is monotone in step size. Under fake-quant it is
not. A weight step far below one quantization LSB changes nothing; a step that
crosses an LSB boundary snaps that weight by a whole LSB. So the KL the
controller measures is dominated by how many weights happened to cross a
boundary, which barely depends on the learning rate at all -- and lowering the
learning rate, its only lever, does not lower it.

This script measures that amplification directly, with no simulator: perturb
the weights by one Adam step (|dw| ~= lr elementwise, since Adam normalises by
the gradient RMS) and compare the resulting policy KL with quantization on and
off, on the same parameters and the same observations.

    KL = sum_a (mu0_a - mu1_a)^2 / (2 sigma_a^2)     (sigma is not perturbed)

which is the same expression ppo.py evaluates, with the sigma-ratio terms
dropped because std is a separate parameter and is held fixed here.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

# ppo.py cuts the learning rate whenever kl_mean > desired_kl * 2.0.
# This task's config.json sets 0.005, so the threshold is 0.01.
DESIRED_KL = 0.005

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.quant_policy import build_from_checkpoint          # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig                        # noqa: E402

# The parameters the adaptive controller actually steps: ppo.py builds
# self.optimizer over actor + critic + std, and the critic is never deployed
# and never affects the action mean, so the actor is what moves mu.
STEPPED = ("actor.",)


def action_mean(net, obs, hist):
    with torch.no_grad():
        return net.act_inference(obs, hist)[0]   # (action_mean, latent)


def integer_weight_codes(net, cfg) -> torch.Tensor:
    """The actor's weights as INTEGER CODES, exactly as _HwGemm computes them.

    A perturbation smaller than one quantization LSB that crosses no boundary
    leaves these codes bit-identical, which is precisely why two different step
    sizes can produce the same policy KL. Reporting how many codes moved turns
    an identical KL from a suspicious coincidence into a measurement.

    Returns an empty tensor when the arm has weight quantization disabled --
    a continuous weight always "changes", so the count would be meaningless.
    """
    if not cfg.quant_weights:
        return torch.zeros(0)
    from roboaccel_quant.fixed_ref import QMAX, QMIN
    from roboaccel_quant.torch_hw import round_away, weight_frac
    parts = []
    with torch.no_grad():
        for m in net.actor.modules():
            if not hasattr(m, "weight") or not hasattr(m, "active"):
                continue
            if not m.active():
                continue
            f_w = weight_frac(m.weight, cfg.weight_bits, cfg.max_weight_frac)
            codes = torch.clamp(round_away(m.weight.double() * 2.0 ** f_w),
                                QMIN[cfg.weight_bits], QMAX[cfg.weight_bits])
            parts.append(codes.reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(0)


def kl_of(mu0, mu1, sigma):
    return float((((mu0 - mu1) ** 2) / (2.0 * sigma ** 2)).sum(-1).mean())


def perturb_(net, lr, gen):
    """One Adam step with unit-RMS gradients: dw = -lr * sign(g), elementwise."""
    saved = {}
    for name, p in net.named_parameters():
        if not any(name.startswith(s) for s in STEPPED):
            continue
        saved[name] = p.detach().clone()
        step = (torch.randint(0, 2, p.shape, generator=gen,
                              dtype=torch.float32) * 2.0 - 1.0) * lr
        p.data.add_(step)
    return saved


def restore_(net, saved):
    for name, p in net.named_parameters():
        if name in saved:
            p.data.copy_(saved[name])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--obs", default=str(QUANT_ROOT / "artifacts/v2/calib_obs.npz"))
    ap.add_argument("--act-fracs", default=str(QUANT_ROOT / "artifacts/v2/act_fracs_a8.json"))
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--lrs", default="1e-6,1e-5,1e-4,2.56e-4,1e-3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.obs)
    obs = torch.from_numpy(d["obs"][: a.samples]).float()
    hist = torch.from_numpy(d["obs_history"][: a.samples]).float()

    fracs = json.loads(Path(a.act_fracs).read_text())["act_fracs"]
    # The precision ladder from goal.md section 7, plus the unquantized control.
    # A8 needs the calibrated per-layer fracs; A16 uses the shipped Q8.8 grid.
    arms = {
        "float":  QuantConfig(quant_weights=False, quant_obs=False,
                              quant_hidden=False, quant_output=False),
        "W16A16": QuantConfig(weight_bits=16, act_bits=16),
        "W8A16":  QuantConfig(weight_bits=8, act_bits=16),
        "W8A8":   QuantConfig(weight_bits=8, act_bits=8, act_fracs=fracs),
        "W4A8":   QuantConfig(weight_bits=4, act_bits=8, act_fracs=fracs),
    }

    # Provenance: goal5.md requires the checkpoint hash and the controller
    # constants to travel with the numbers, not alongside them.
    out = {"checkpoint": a.checkpoint,
           "checkpoint_sha256": hashlib.sha256(
               Path(a.checkpoint).read_bytes()).hexdigest(),
           "obs_file": a.obs, "act_fracs_file": a.act_fracs,
           "samples": int(obs.shape[0]), "trials": a.trials,
           "stepped_prefixes": list(STEPPED),
           "desired_kl": DESIRED_KL,
           "kl_decrease_threshold": DESIRED_KL * 2.0,
           "lr_floor": 1e-5,
           "results": {}}

    for arm, cfg in arms.items():
        net = build_from_checkpoint(a.checkpoint, quant=cfg, device="cpu")
        sigma = net.std.detach().reshape(-1)   # exploration std is not quantized
        mu0 = action_mean(net, obs, hist)
        w0 = integer_weight_codes(net, cfg)
        out.setdefault("sigma_mean", float(sigma.mean()))
        for lr in [float(x) for x in a.lrs.split(",")]:
            kls, dmus, crossed = [], [], []
            for t in range(a.trials):
                gen = torch.Generator().manual_seed(1000 + t)
                saved = perturb_(net, lr, gen)
                mu1 = action_mean(net, obs, hist)
                kls.append(kl_of(mu0, mu1, sigma))
                dmus.append(float((mu1 - mu0).abs().mean()))
                w1 = integer_weight_codes(net, cfg)
                crossed.append(int((w1 != w0).sum()) if w0.numel() else -1)
                restore_(net, saved)
            out["results"].setdefault(f"{lr:g}", {})[arm] = {
                "kl_mean": float(np.mean(kls)), "kl_std": float(np.std(kls)),
                "abs_dmu_mean": float(np.mean(dmus)),
                "codes_changed_mean": float(np.mean(crossed)),
                "codes_total": int(w0.numel()),
            }

    # ppo.py drops the lr whenever kl_mean > desired_kl * 2.0. The wheel-legged
    # config sets desired_kl = 0.005 (read back from the run's own config.json),
    # so the controller starts cutting at 0.01.
    KL_DECREASE_THRESHOLD = out["kl_decrease_threshold"]
    for lr, r in out["results"].items():
        fl = r["float"]
        print(f"\n--- one Adam step at lr={lr} "
              f"(|dw| = {float(lr):g} elementwise, actor only) ---")
        print(f"{'arm':10s} {'policy KL':>12s} {'amplification':>14s} "
              f"{'controller':>22s}")
        for arm in arms:
            e = r[arm]
            amp = e["kl_mean"] / fl["kl_mean"] if fl["kl_mean"] > 0 else float("nan")
            e["amplification_vs_float"] = amp
            verdict = ("LR DRIVEN TO FLOOR" if e["kl_mean"] > KL_DECREASE_THRESHOLD
                       else "lr free to rise")
            print(f"{arm:10s} {e['kl_mean']:>12.3e} {amp:>13.1f}x {verdict:>22s}")

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
