#!/usr/bin/env python3
"""Attribute a trained policy's deviation from its warm start to encoder vs actor.

The retraining experiments ask what happens if the encoder is not allowed to
move. This asks the complementary, much cheaper question about the runs that
already exist: of the total action-mean change a QAT run produced, how much is
carried by the encoder block and how much by the actor block?

Four policies are built from two checkpoints by swapping whole blocks:

    (d) warm encoder + warm actor      the warm start itself, KL 0 by definition
    (c) warm encoder + final actor     the actor's contribution alone
    (b) final encoder + warm actor     the encoder's contribution alone
    (a) final encoder + final actor    the trained run

KL is measured against (d) on identical states, in the arm's own execution
precision, with the exploration std held at the warm start's value so the
comparison is about the mean only -- the same convention as
scripts/kl_amplification.py.

Block swapping is legitimate here only because the latent is detached: the
encoder and actor are separately parameterized and communicate through one
concatenated vector, so (b) and (c) are well-formed policies, not chimeras.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))
# runs are written under training/checkpoints/<run>/
TRAIN_ROOT = QUANT_ROOT.parent / "training"

from roboaccel_quant.quant_policy import build_from_checkpoint          # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig                        # noqa: E402


def state_dict(path):
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    d = dict(d.get("model_state_dict", d))
    if "log_std" in d:
        d["std"] = d.pop("log_std").exp()
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/frozen/SOLID_FP32_V2.pt")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run=ARM pairs, e.g. qat_w8a8_v2_long=W8A8")
    ap.add_argument("--obs", type=Path, default=QUANT_ROOT / "artifacts/v2/calib_obs.npz")
    ap.add_argument("--act-fracs", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/act_fracs_a8.json")
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=QUANT_ROOT / "artifacts/goal6/block_attribution.json")
    a = ap.parse_args()

    fracs = json.loads(a.act_fracs.read_text())["act_fracs"]
    ARMS = {
        "float":  QuantConfig(quant_weights=False, quant_obs=False,
                              quant_hidden=False, quant_output=False),
        "W8A16":  QuantConfig(weight_bits=8, act_bits=16),
        "W8A8":   QuantConfig(weight_bits=8, act_bits=8, act_fracs=fracs),
    }
    d = np.load(a.obs)
    obs = torch.from_numpy(d["obs"][: a.samples]).float()
    hist = torch.from_numpy(d["obs_history"][: a.samples]).float()

    warm_sd = state_dict(a.warm)
    out = {"warm": str(a.warm),
           "warm_sha256": hashlib.sha256(a.warm.read_bytes()).hexdigest()[:16],
           "samples": int(obs.shape[0]), "runs": {}}

    print(f"{'run':<26} {'exec':>7} {'both':>10} {'actor only':>11} "
          f"{'encoder only':>13} {'enc/act':>8}")
    print("-" * 80)
    for spec in a.runs:
        run, _, armname = spec.partition("=")
        armname = armname or "W8A8"
        ck = TRAIN_ROOT / "checkpoints" / run / "model_2000.pt"
        if not ck.exists():
            print(f"{run:<26} MISSING {ck}")
            continue
        final_sd = state_dict(ck)
        cfg = ARMS[armname]
        net = build_from_checkpoint(a.warm, quant=cfg, device="cpu")
        sigma = net.std.detach().reshape(-1)          # warm start's std, held fixed

        def load_blocks(enc_from, act_from):
            sd = dict(warm_sd)
            for k in warm_sd:
                if k.startswith("encoder."):
                    sd[k] = enc_from[k]
                elif k.startswith("actor."):
                    sd[k] = act_from[k]
            sd["std"] = warm_sd["std"]
            net.load_state_dict(sd, strict=False)
            with torch.no_grad():
                return net.act_inference(obs, hist)[0]

        mu_d = load_blocks(warm_sd, warm_sd)
        mu_c = load_blocks(warm_sd, final_sd)
        mu_b = load_blocks(final_sd, warm_sd)
        mu_a = load_blocks(final_sd, final_sd)

        def kl(m):
            return float((((mu_d - m) ** 2) / (2.0 * sigma ** 2)).sum(-1).mean())

        r = {"execution": armname,
             "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest()[:16],
             "kl_both": kl(mu_a), "kl_actor_only": kl(mu_c),
             "kl_encoder_only": kl(mu_b), "kl_baseline": kl(mu_d)}
        r["encoder_over_actor"] = (r["kl_encoder_only"] / r["kl_actor_only"]
                                   if r["kl_actor_only"] > 0 else float("nan"))
        out["runs"][run] = r
        print(f"{run:<26} {armname:>7} {r['kl_both']:>10.4g} {r['kl_actor_only']:>11.4g} "
              f"{r['kl_encoder_only']:>13.4g} {r['encoder_over_actor']:>8.2f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
