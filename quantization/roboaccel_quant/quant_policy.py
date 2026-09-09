#!/usr/bin/env python3
"""Quantized drop-in for the Fudan ActorCriticSequence.

Parameter names, module indices and call signatures match the upstream class
exactly, so an FP32 checkpoint loads with strict=True and rsl_rl's runner and
PPO need no changes. What differs is only what happens inside the forward
pass of the two modules that get deployed.

Quantized (goal.md section 5):
    obs, obs_history           -> Q8.8 at the policy input, where the MCU does it
    encoder weights + acts     -> the deployed datapath
    actor weights + acts       -> the deployed datapath
    latent                     -> falls out of encoder.4 already on the grid
    action mean                -> the deployed datapath

Left in FP32 (never deployed):
    critic, log_std / exploration noise, PPO losses, optimizer state, grads

The latent detach in update_distribution is upstream behaviour, not a
quantization choice: PPO's policy gradient does not reach the encoder, which
is trained by the auxiliary regression onto base_lin_vel in ppo.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .fixed_ref import ACT_FRAC, FudanFixedPolicy
from .torch_hw import HwElu, HwLinear, HwObsQuant, QuantConfig


def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU,
            "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]()


class QuantActorCriticSequence(nn.Module):
    is_recurrent = False
    is_sequence = True

    def __init__(self, num_obs, num_critic_obs, num_actions, num_encoder_obs,
                 latent_dim, encoder_hidden_dims=(128, 64),
                 actor_hidden_dims=(128, 64, 32),
                 critic_hidden_dims=(256, 128, 64), activation="elu",
                 orthogonal_init=False, init_noise_std=1.0,
                 quant: QuantConfig | None = None, **kwargs):
        super().__init__()
        if activation != "elu":
            raise ValueError("the SFU ROM holds ELU only; see docs/primitive_isa.md")
        self.cfg = quant or QuantConfig()
        self.latent_dim = latent_dim
        self.num_obs = num_obs

        self.obs_quant = HwObsQuant(self.cfg, feeds="actor")
        self.hist_quant = HwObsQuant(self.cfg, feeds="encoder")

        enc: list[nn.Module] = [HwLinear(num_encoder_obs, encoder_hidden_dims[0],
                                        self.cfg, name="enc0", prev="obs")]
        for i in range(len(encoder_hidden_dims)):
            enc.append(HwElu(self.cfg, name=f"enc{i}"))
            nxt = latent_dim if i == len(encoder_hidden_dims) - 1 else encoder_hidden_dims[i + 1]
            enc.append(HwLinear(encoder_hidden_dims[i], nxt, self.cfg,
                                name=f"enc{i + 1}", prev=f"enc{i}"))
        self.encoder = nn.Sequential(*enc)

        # act0 consumes concat(obs, latent); the PL CONCAT cannot requantize,
        # so both operands must already share a grid -- enforced below.
        act: list[nn.Module] = [HwLinear(num_obs + latent_dim, actor_hidden_dims[0],
                                        self.cfg, name="act0", prev="obs")]
        for i in range(len(actor_hidden_dims)):
            act.append(HwElu(self.cfg, name=f"act{i}"))
            last = i == len(actor_hidden_dims) - 1
            nxt = num_actions if last else actor_hidden_dims[i + 1]
            act.append(HwLinear(actor_hidden_dims[i], nxt, self.cfg,
                                is_output=last, name=f"act{i + 1}", prev=f"act{i}"))
        self.actor = nn.Sequential(*act)
        if self.cfg.frac_of("enc2") != self.cfg.frac_of("obs"):
            raise ValueError(
                f"CONCAT cannot requantize: latent frac {self.cfg.frac_of('enc2')} "
                f"!= obs frac {self.cfg.frac_of('obs')}")

        # Critic: plain FP32, identical construction to upstream.
        cri: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0])]
        for i in range(len(critic_hidden_dims)):
            cri.append(_activation(activation))
            nxt = 1 if i == len(critic_hidden_dims) - 1 else critic_hidden_dims[i + 1]
            cri.append(nn.Linear(critic_hidden_dims[i], nxt))
        self.critic = nn.Sequential(*cri)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.latent = None
        Normal.set_default_validate_args = False

    # -- rsl_rl API --------------------------------------------------------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def encode(self, observation_history, **kwargs):
        return self.encoder(self.hist_quant(observation_history))

    def update_distribution(self, observations, observation_history):
        self.latent = self.encode(observation_history)
        obs_q = self.obs_quant(observations)
        mean = self.actor(torch.cat((obs_q, self.latent.detach()), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, observation_history, **kwargs):
        self.update_distribution(observations, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent(self, **kwargs):
        return self.latent

    def act_inference(self, observations, observation_history):
        """The deployed path. This is what gets compared against the board."""
        self.latent = self.encode(observation_history)
        obs_q = self.obs_quant(observations)
        return self.actor(torch.cat((obs_q, self.latent), dim=-1)), self.latent

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    # -- bridge to the integer reference -----------------------------------
    LAYER_NAMES = {"encoder.0": "enc0", "encoder.2": "enc1", "encoder.4": "enc2",
                   "actor.0": "act0", "actor.2": "act1", "actor.4": "act2",
                   "actor.6": "act3"}

    def weight_dict(self) -> dict:
        sd = self.state_dict()
        return {short: (sd[f"{k}.weight"].detach().cpu().double().numpy(),
                        sd[f"{k}.bias"].detach().cpu().double().numpy())
                for k, short in self.LAYER_NAMES.items()}

    def to_fixed_ref(self) -> FudanFixedPolicy:
        names = ["obs", "enc0", "enc1", "enc2", "act0", "act1", "act2", "act3"]
        return FudanFixedPolicy(self.weight_dict(),
                                weight_bits=self.cfg.weight_bits,
                                act_bits=self.cfg.act_bits,
                                act_fracs={n: self.cfg.frac_of(n) for n in names},
                                obs_frac=self.cfg.frac_of("obs"),
                                max_weight_frac=self.cfg.max_weight_frac)

    def frac_table(self) -> dict:
        return {short: int(dict(self.named_modules())[k].last_frac.item())
                for k, short in self.LAYER_NAMES.items()}


def build_from_onnx(onnx_path, quant: QuantConfig | None = None,
                    num_critic_obs: int = 1, device="cpu"):
    """Load the article's released encoder+actor into a quantized policy.

    The published artefacts are ONNX, not .pt -- there is no critic in them, so
    a model built this way can be evaluated and quantized but not PPO-trained
    until a critic is fitted. See PROGRESS.md.
    """
    import numpy as np, onnx
    from onnx.helper import get_attribute_value
    from onnx.numpy_helper import to_array

    model = onnx.load(str(onnx_path))
    init = {i.name: to_array(i) for i in model.graph.initializer}
    gemms = []
    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue
        w = init[node.input[1]]
        attrs = {a.name: get_attribute_value(a) for a in node.attribute}
        if int(attrs.get("transB", 0)) == 0:
            w = w.T
        gemms.append((node.name, np.asarray(w), np.asarray(init[node.input[2]]).reshape(-1)))
    if len(gemms) != 7:
        raise ValueError(f"expected 7 Gemm nodes, found {len(gemms)}")

    enc_dims = [int(g[1].shape[0]) for g in gemms[:3]]
    act_dims = [int(g[1].shape[0]) for g in gemms[3:]]
    net = QuantActorCriticSequence(
        num_obs=int(gemms[3][1].shape[1]) - enc_dims[-1],
        num_critic_obs=num_critic_obs, num_actions=act_dims[-1],
        num_encoder_obs=int(gemms[0][1].shape[1]), latent_dim=enc_dims[-1],
        encoder_hidden_dims=tuple(enc_dims[:-1]),
        actor_hidden_dims=tuple(act_dims[:-1]),
        quant=quant)
    with torch.no_grad():
        for key, (_, w, b) in zip(net.LAYER_NAMES, gemms):
            mod = dict(net.named_modules())[key]
            mod.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
            mod.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    return net.to(device).eval()


def build_from_checkpoint(ckpt_path, quant: QuantConfig | None = None,
                          device="cpu", strict_critic: bool = True):
    """Load a SOLID_FP32 `.pt` from the frozen Codex infrastructure.

    Unlike `build_from_onnx`, this carries the critic, so a policy built here
    can be PPO-fine-tuned (goal.md §14) as well as evaluated. Two shape-level
    facts are asserted rather than assumed, because §5 turns on them:

      * the actor/encoder key layout is still encoder.0/.2/.4 + actor.0/.2/.4/.6
        (`ActorCriticRobust` inherits them from `ActorCriticSequence`);
      * `latent_dim` equals what the CONCAT descriptor was built for.

    The Codex pass renamed the exploration parameter `std` -> `log_std`. That
    is training-only -- deployment uses the actor mean -- so it is converted
    here and never reaches the fixed-point reference.
    """
    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck.get("model_state_dict", ck)

    missing = [k for k in ("encoder.0.weight", "encoder.2.weight", "encoder.4.weight",
                           "actor.0.weight", "actor.2.weight", "actor.4.weight",
                           "actor.6.weight") if k not in sd]
    if missing:
        raise ValueError(
            "checkpoint is not the expected ActorCriticSequence layout; missing "
            + ", ".join(missing) + ". If the Codex pass changed the topology, "
            "SOLID_POLICY_DEPLOYMENT_MAPPING.md must be redone before quantizing.")

    enc_dims = [int(sd[f"encoder.{i}.weight"].shape[0]) for i in (0, 2, 4)]
    act_dims = [int(sd[f"actor.{i}.weight"].shape[0]) for i in (0, 2, 4, 6)]
    num_encoder_obs = int(sd["encoder.0.weight"].shape[1])
    latent_dim = enc_dims[-1]
    num_obs = int(sd["actor.0.weight"].shape[1]) - latent_dim
    has_critic = "critic.0.weight" in sd
    num_critic_obs = int(sd["critic.0.weight"].shape[1]) if has_critic else 1
    if strict_critic and not has_critic:
        raise ValueError("checkpoint has no critic; pass strict_critic=False to "
                         "evaluate anyway (PPO fine-tuning will not be possible)")

    net = QuantActorCriticSequence(
        num_obs=num_obs, num_critic_obs=num_critic_obs,
        num_actions=act_dims[-1], num_encoder_obs=num_encoder_obs,
        latent_dim=latent_dim,
        encoder_hidden_dims=tuple(enc_dims[:-1]),
        actor_hidden_dims=tuple(act_dims[:-1]),
        critic_hidden_dims=tuple(int(sd[f"critic.{i}.weight"].shape[0])
                                 for i in (0, 2, 4)) if has_critic else (256, 128, 64),
        quant=quant)

    state = {k: v for k, v in sd.items() if k != "log_std"}
    if "log_std" in sd:
        state["std"] = sd["log_std"].exp()
    elif "std" in sd:
        state["std"] = sd["std"]
    incompatible = net.load_state_dict(state, strict=has_critic)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    net.iteration = int(ck.get("iter", -1))
    return net.to(device).eval()
