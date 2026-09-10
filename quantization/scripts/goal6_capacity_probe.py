#!/usr/bin/env python3
"""Can an INT8-activation datapath represent this policy at all? (goal6.md H3)

goal6 reaches H3 by elimination: "if neither encoder freezing nor behaviour
preservation rescues turning, the failure may be intrinsic to W8A8." Concluding
capacity from the failure of two RL interventions is weak, because an RL run
confounds three things at once -- what the network CAN represent, what the PPO
objective ASKS for, and what the adaptive controller LETS the optimizer do.

This measures the first one on its own. No RL, no reward, no PPO, no
controller: the frozen FP32 policy is the teacher, and a fake-quant student
initialised from the same weights is fitted to it by plain supervised
regression on stored on-policy observations. Gradients reach the quantized
weights through the straight-through estimator, exactly as in QAT.

    capacity  =  how close supervised distillation can get
    the rest  =  everything QAT adds on top

If W8A8 distils to near-teacher error, the datapath has the capacity and the
QAT failure belongs to the objective or the optimizer. If it plateaus far from
the teacher while W8A16 does not, H3 is supported directly rather than by
elimination.

The encoder is trained jointly with the actor, because act_inference does not
detach the latent and both blocks are deployed.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.quant_policy import build_from_checkpoint          # noqa: E402
from roboaccel_quant.teacher_kl import gaussian_kl                      # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig                        # noqa: E402


def arms(fracs):
    fp32 = dict(quant_weights=False, quant_obs=False,
                quant_hidden=False, quant_output=False)
    return {
        "float":  QuantConfig(**fp32),
        "W16A16": QuantConfig(weight_bits=16, act_bits=16),
        "W8A16":  QuantConfig(weight_bits=8, act_bits=16),
        "W8A8":   QuantConfig(weight_bits=8, act_bits=8, act_fracs=fracs),
        "W4A8":   QuantConfig(weight_bits=4, act_bits=8, act_fracs=fracs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/frozen/SOLID_FP32_V2.pt")
    ap.add_argument("--obs", type=Path, default=QUANT_ROOT / "artifacts/v2/calib_obs.npz")
    ap.add_argument("--act-fracs", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/act_fracs_a8.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--latent-scale", type=float, default=1.0,
                    help="rescale the encoder->actor boundary. The latent is "
                         "pinned to the obs fractional bits because PL's CONCAT "
                         "cannot requantize, which costs it a bit of resolution "
                         "its own range would allow. Multiplying encoder.4 by S "
                         "and dividing actor.0's latent columns by S is exact in "
                         "FP32 and buys S-fold finer latent steps at the same "
                         "frac -- cross-layer equalization at the one boundary "
                         "the hardware constrains. No RTL or exporter change.")
    ap.add_argument("--arms", default=None,
                    help="comma list to restrict the sweep, e.g. W8A8,W8A16")
    ap.add_argument("--trace", action="store_true",
                    help="record the holdout curve so a plateau can be "
                         "distinguished from an unfinished descent")
    ap.add_argument("--out", type=Path,
                    default=QUANT_ROOT / "artifacts/goal6/capacity_probe.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    fracs = json.loads(a.act_fracs.read_text())["act_fracs"]
    d = np.load(a.obs)
    obs = torch.as_tensor(d["obs"]).to(a.device)
    hist = torch.as_tensor(d["obs_history"]).to(a.device)
    n = obs.shape[0]
    n_tr = int(n * (1 - a.holdout))
    # Held out so a plateau cannot be confused with underfitting and a low
    # training error cannot be confused with memorising the calibration set.
    tr = slice(0, n_tr)
    te = slice(n_tr, n)

    fp32cfg = QuantConfig(quant_weights=False, quant_obs=False,
                          quant_hidden=False, quant_output=False)
    teacher = build_from_checkpoint(a.checkpoint, quant=fp32cfg, device=a.device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        mu_t_tr = teacher.act_inference(obs[tr], hist[tr])[0]
        mu_t_te = teacher.act_inference(obs[te], hist[te])[0]
        sigma = teacher.std.detach().reshape(-1)

    out = {"checkpoint": str(a.checkpoint),
           "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()[:16],
           "samples_train": int(n_tr), "samples_holdout": int(n - n_tr),
           "steps": a.steps, "batch": a.batch, "lr": a.lr,
           "objective": "supervised regression onto the FP32 teacher's action mean",
           "latent_scale": a.latent_scale,
           "arms": {}}

    print(f"{'arm':<8} {'MSE start':>11} {'MSE best':>11} {'KL start':>10} "
          f"{'KL best':>10} {'vs PPO thr':>12} {'sec':>6}")
    print("-" * 76)
    wanted = set(a.arms.split(",")) if a.arms else None
    for name, cfg in arms(fracs).items():
        if wanted and name not in wanted:
            continue
        t0 = time.time()
        net = build_from_checkpoint(a.checkpoint, quant=cfg, device=a.device)
        if a.latent_scale != 1.0:
            S, n_obs = a.latent_scale, net.num_obs
            with torch.no_grad():
                enc_last = net.encoder[-1]
                enc_last.weight.mul_(S)
                enc_last.bias.mul_(S)
                # act0 consumes cat(obs, latent); the latent occupies the
                # trailing latent_dim columns. Dividing them by S makes the
                # rescaling exactly identity in FP32.
                net.actor[0].weight[:, n_obs:].div_(S)
        net.train()
        params = list(net.encoder.parameters()) + list(net.actor.parameters())
        opt = torch.optim.Adam(params, lr=a.lr)

        def evaluate():
            net.eval()
            with torch.no_grad():
                mu = net.act_inference(obs[te], hist[te])[0]
                mse = float((mu - mu_t_te).square().mean())
                kl = float(gaussian_kl(mu_t_te, sigma.expand_as(mu_t_te),
                                       mu, sigma.expand_as(mu)).mean())
            net.train()
            return mse, kl

        mse0, kl0 = evaluate()
        # The question is how close the arm CAN get, so track the best holdout
        # error reached at any point, not the endpoint: a fixed learning rate
        # on an already-near-optimal init can end worse than it started.
        best_mse, best_kl = mse0, kl0
        curve = [(0, kl0)]
        every = max(1, a.steps // 25)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
        g = torch.Generator().manual_seed(7)
        for step in range(a.steps):
            idx = torch.randint(0, n_tr, (a.batch,), generator=g)
            mu = net.act_inference(obs[idx], hist[idx])[0]
            loss = (mu - mu_t_tr[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % every == 0:
                m, k = evaluate()
                curve.append((step + 1, k))
                if k < best_kl:
                    best_mse, best_kl = m, k
        mse1, kl1 = best_mse, best_kl

        # PPO cuts the learning rate whenever measured KL exceeds
        # desired_kl*2 = 0.01, so that threshold is the natural yardstick for
        # "close enough that the controller would not fight it".
        over = kl1 / 0.01
        out["arms"][name] = {"mse_start": mse0, "mse_best": mse1,
                             "kl_start": kl0, "kl_best": kl1,
                             "kl_best_over_ppo_threshold": over,
                             "seconds": time.time() - t0,
                             **({"holdout_kl_curve": curve} if a.trace else {})}
        print(f"{name:<8} {mse0:>11.3e} {mse1:>11.3e} {kl0:>10.4g} {kl1:>10.4g} "
              f"{over:>11.1f}x {time.time()-t0:>6.0f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
