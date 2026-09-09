#!/usr/bin/env python3
"""Checkpoint -> the same two-input ONNX graph the upstream exporter emits.

goal.md section 16 says to prefer the existing exporter over a new format.
`tools/export_policy.py` consumes ONNX, so the integration point is here: emit
a graph byte-compatible with `plane/export_onnx/export_onnx.py` (inputs `obs`
and `obs_history`, output `actions`, 7 Gemm / 5 Elu / 1 Concat) and the whole
downstream chain -- descriptor program, cache images, `rl_policy_data.c`, the
H7 C model, the RTL testbenches -- works unchanged.

The exported network is FP32. Quantization happens in the exporter, from the
same `choose_weight_frac` rule the QAT forward used, so the deployed integers
are the ones training saw. `scripts/verify_export.py` proves that end to end
rather than assuming it.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import torch
import torch.nn as nn

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
sys.path.insert(0, str(QUANT_ROOT))

from roboaccel_quant.quant_policy import QuantActorCriticSequence  # noqa: E402
from roboaccel_quant.torch_hw import QuantConfig  # noqa: E402

OFF = dict(quant_weights=False, quant_obs=False, quant_hidden=False, quant_output=False)


class Wrapper(nn.Module):
    """encoder(obs_history) -> latent; actor(cat(obs, latent)) -> actions."""

    def __init__(self, enc, act):
        super().__init__()
        self.encoder, self.actor = enc, act

    def forward(self, obs, obs_history):
        return self.actor(torch.cat([obs, self.encoder(obs_history)], dim=-1))


def build_fp32_twin(sd) -> tuple[nn.Sequential, nn.Sequential]:
    """Plain nn.Linear/nn.ELU stacks carrying the checkpoint's weights.

    Exporting the HwLinear modules directly would bake the fake-quant ops into
    the graph; the deployment wants the float weights, which the exporter then
    quantizes with the identical rule.
    """
    enc_out = [sd[f"encoder.{i}.weight"].shape[0] for i in (0, 2, 4)]
    act_out = [sd[f"actor.{i}.weight"].shape[0] for i in (0, 2, 4, 6)]
    enc, act = [], []
    for i, idx in enumerate((0, 2, 4)):
        enc.append(nn.Linear(sd[f"encoder.{idx}.weight"].shape[1], enc_out[i]))
        if idx != 4:
            enc.append(nn.ELU())
    for i, idx in enumerate((0, 2, 4, 6)):
        act.append(nn.Linear(sd[f"actor.{idx}.weight"].shape[1], act_out[i]))
        if idx != 6:
            act.append(nn.ELU())
    enc_s, act_s = nn.Sequential(*enc), nn.Sequential(*act)
    with torch.no_grad():
        for idx in (0, 2, 4):
            enc_s[idx].weight.copy_(sd[f"encoder.{idx}.weight"])
            enc_s[idx].bias.copy_(sd[f"encoder.{idx}.bias"])
        for idx in (0, 2, 4, 6):
            act_s[idx].weight.copy_(sd[f"actor.{idx}.weight"])
            act_s[idx].bias.copy_(sd[f"actor.{idx}.bias"])
    return enc_s, act_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sd = dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
    if "log_std" in sd:   # the Codex pass renamed the exploration parameter
        sd["std"] = sd.pop("log_std").exp()
    enc, act = build_fp32_twin(sd)
    model = Wrapper(enc, act).eval()

    n_obs = sd["actor.0.weight"].shape[1] - sd["encoder.4.weight"].shape[0]
    n_hist = sd["encoder.0.weight"].shape[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (torch.zeros(1, n_obs), torch.zeros(1, n_hist)), str(args.out),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["obs", "obs_history"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "obs_history": {0: "batch"},
                      "actions": {0: "batch"}})

    # The exported graph must agree with the checkpoint run in FP32, or the
    # rest of the chain is quantizing a different network.
    import numpy as np, onnxruntime as ort
    net = QuantActorCriticSequence(
        num_obs=n_obs, num_critic_obs=sd["critic.0.weight"].shape[1],
        num_actions=sd["actor.6.weight"].shape[0], num_encoder_obs=n_hist,
        latent_dim=sd["encoder.4.weight"].shape[0],
        encoder_hidden_dims=tuple(sd[f"encoder.{i}.weight"].shape[0] for i in (0, 2)),
        actor_hidden_dims=tuple(sd[f"actor.{i}.weight"].shape[0] for i in (0, 2, 4)),
        critic_hidden_dims=tuple(sd[f"critic.{i}.weight"].shape[0] for i in (0, 2, 4)),
        quant=QuantConfig(**OFF))
    net.load_state_dict(sd, strict=True)
    net.eval()
    rng = np.random.default_rng(0)
    o = rng.normal(0, 0.5, (256, n_obs)).astype(np.float32)
    h = rng.normal(0, 0.5, (256, n_hist)).astype(np.float32)
    with torch.no_grad():
        a_t, _ = net.act_inference(torch.as_tensor(o), torch.as_tensor(h))
    a_o = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"]).run(
        None, {"obs": o, "obs_history": h})[0]
    d = float(np.abs(a_t.numpy() - a_o).max())
    print(f"wrote {args.out}")
    print(f"onnx vs checkpoint (FP32): max abs diff {d:.3e}  -> "
          f"{'MATCH' if d < 1e-4 else 'MISMATCH'}")
    return 0 if d < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
