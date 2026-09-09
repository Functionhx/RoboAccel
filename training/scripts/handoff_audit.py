#!/usr/bin/env python3
"""goal.md section 0/5: audit the RL handoff and check deployment compatibility.

Reads the CURRENT state of the RL repo and reports what the deployed policy
actually is -- observation and history dimensions, encoder, latent, actor
topology, activations, action scaling, control rate -- then checks that against
what the fixed-point/export/deployment path in this package can represent.

Everything is read from the live config classes and module sources rather than
from documentation, because the point of the audit is to catch the case where a
doc and the code disagree. Nothing is imported from the RL repo that would
build a simulator; configs are plain classes.

Usage:
  ISAAC_PY scripts/handoff_audit.py --repo <pkg dir containing wheel_legged_gym>
"""
from __future__ import annotations

import argparse, ast, json, subprocess, sys
from pathlib import Path

# What the deployment datapath in this package can express. Anything outside
# this list needs the exporter/descriptor work called for in goal.md section 5.
SUPPORTED_OPS = {"Gemm", "Elu", "Concat"}
PL_LIMITS = {"weight_words_per_bank": 1280, "vector_words": 512, "program_len": 32}


def git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return ""


def class_attrs(path: Path, cls: str) -> dict:
    """Literal attributes of a class, including nested config classes."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:                                            # noqa: BLE001
        return {}
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.ClassDef):
                    inner = {}
                    for st in sub.body:
                        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name):
                            try:
                                inner[st.targets[0].id] = ast.literal_eval(st.value)
                            except Exception:                    # noqa: BLE001
                                inner[st.targets[0].id] = "<expr>"
                    out[sub.name] = inner
                elif isinstance(sub, ast.Assign) and isinstance(sub.targets[0], ast.Name):
                    try:
                        out[sub.targets[0].id] = ast.literal_eval(sub.value)
                    except Exception:                            # noqa: BLE001
                        out[sub.targets[0].id] = "<expr>"
    return out


def find_policy_modules(pkg: Path) -> list[Path]:
    d = pkg / "wheel_legged_gym" / "rsl_rl" / "modules"
    return sorted(p for p in d.glob("*.py") if p.name != "__init__.py") if d.is_dir() else []


def describe_policy(path: Path) -> dict:
    """Layer sizes and activation of a policy module, from its source."""
    src = path.read_text()
    info = {"file": path.name, "classes": [], "activations": sorted(
        {w for w in ("elu", "relu", "selu", "tanh", "sigmoid", "lrelu", "crelu")
         if f'"{w}"' in src or f"'{w}'" in src})}
    try:
        tree = ast.parse(src)
    except Exception:                                            # noqa: BLE001
        return info
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            defaults = {}
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    args = fn.args
                    names = [a.arg for a in args.args][-len(args.defaults):] if args.defaults else []
                    for n, d in zip(names, args.defaults):
                        try:
                            defaults[n] = ast.literal_eval(d)
                        except Exception:                        # noqa: BLE001
                            defaults[n] = "<expr>"
            info["classes"].append({"name": node.name, "init_defaults": defaults})
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True,
                    help="package dir containing wheel_legged_gym (e.g. .../plane)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    pkg = args.repo
    root = pkg.parent

    base_cfg = pkg / "wheel_legged_gym/envs/base/legged_robot_config.py"
    wl_cfgs = sorted((pkg / "wheel_legged_gym/envs/wheel_legged").glob("*config*.py"))

    rep: dict = {"repo": str(root), "package": str(pkg)}
    rep["git"] = {"head": git(root, "rev-parse", "--short", "HEAD"),
                  "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                  "subject": git(root, "log", "-1", "--format=%s"),
                  "dirty_files": [l for l in git(root, "status", "--short").splitlines()][:40]}

    base = class_attrs(base_cfg, "LeggedRobotCfg")
    rep["env"] = base.get("env", {})
    rep["control"] = base.get("control", {})
    rep["sim"] = {k: v for k, v in base.get("sim", {}).items() if k == "dt"}
    rep["normalization"] = base.get("normalization", {})
    rep["obs_scales"] = class_attrs(base_cfg, "LeggedRobotCfg").get("normalization", {})
    # The base class is only a fallback. WheelLeggedCfgPPO overrides runner
    # fields the task actually uses -- notably max_iterations, 5000 in the base
    # but 50000 for this task. Reading the base alone understated the
    # configured schedule by 10x and made a 5000-iteration run look complete.
    ppo = class_attrs(base_cfg, "LeggedRobotCfgPPO")
    task_ppo = {}
    for f in wl_cfgs:
        got = class_attrs(f, "WheelLeggedCfgPPO")
        if got:
            task_ppo = got
            break
    def merged(section):
        out = dict(ppo.get(section, {}))
        out.update(task_ppo.get(section, {}))
        return out
    rep["policy_cfg"] = merged("policy")
    rep["runner_cfg"] = merged("runner")
    rep["algorithm_cfg"] = merged("algorithm")
    rep["runner_cfg_source"] = ("WheelLeggedCfgPPO over LeggedRobotCfgPPO"
                                if task_ppo else "LeggedRobotCfgPPO only")

    rep["task_configs"] = {}
    for f in wl_cfgs:
        for cls in ("WheelLeggedCfg", "RobustWheelLeggedCfg", "WheelLeggedCfgPPO"):
            a = class_attrs(f, cls)
            if a:
                rep["task_configs"][f"{f.name}:{cls}"] = a

    rep["policy_modules"] = [describe_policy(p) for p in find_policy_modules(pkg)]

    # Derived, with the arithmetic spelled out so a mismatch is obvious.
    env = rep["env"]
    n_obs = env.get("num_observations")
    hist_len = env.get("obs_history_length")
    n_act = env.get("num_actions")
    pol = rep["policy_cfg"]
    latent = pol.get("latent_dim")
    enc_h = pol.get("encoder_hidden_dims")
    act_h = pol.get("actor_hidden_dims")
    dt = rep["sim"].get("dt")
    dec = rep["control"].get("decimation")
    rep["derived"] = {
        "num_obs": n_obs, "history_len": hist_len,
        "encoder_in": (n_obs * hist_len) if (n_obs and hist_len) else None,
        "latent_dim": latent,
        "actor_in": (n_obs + latent) if (n_obs and latent) else None,
        "num_actions": n_act,
        "control_dt_s": (dt * dec) if (dt and dec) else None,
        "policy_hz": (1.0 / (dt * dec)) if (dt and dec) else None,
        "pos_action_scale": rep["control"].get("pos_action_scale"),
        "vel_action_scale": rep["control"].get("vel_action_scale"),
        "activation": pol.get("activation"),
    }
    # Action scales and the activation usually live on the task config, which
    # overrides the base; take the task value when it exists.
    for name, attrs in rep["task_configs"].items():
        ctl = attrs.get("control", {})
        for k in ("pos_action_scale", "vel_action_scale"):
            if ctl.get(k) is not None:
                rep["derived"][k] = ctl[k]
                rep["derived"].setdefault("action_scale_source", name)
        pcfg = attrs.get("policy", {})
        if pcfg.get("activation"):
            rep["derived"]["activation"] = pcfg["activation"]
        for k in ("latent_dim", "encoder_hidden_dims", "actor_hidden_dims"):
            if pcfg.get(k) is not None:
                rep["derived"].setdefault("policy_overrides", {})[k] = pcfg[k]

    # Deployment compatibility: does this still map onto the shipped datapath?
    notes, ok = [], True
    if enc_h and act_h and latent and n_obs and n_act and hist_len:
        gemms = len(enc_h) + 1 + len(act_h) + 1
        elus = len(enc_h) + len(act_h)
        macs = 0
        dims = [n_obs * hist_len, *enc_h, latent]
        for a, b in zip(dims[:-1], dims[1:]):
            macs += a * b
        dims = [n_obs + latent, *act_h, n_act]
        for a, b in zip(dims[:-1], dims[1:]):
            macs += a * b
        wwords = 0
        for a, b in zip([n_obs * hist_len, *enc_h], [*enc_h, latent]):
            wwords += -(-a // 8) * -(-b // 8)
        for a, b in zip([n_obs + latent, *act_h], [*act_h, n_act]):
            wwords += -(-a // 8) * -(-b // 8)
        rep["deployment"] = {"gemm": gemms, "elu": elus, "concat": 1,
                             "macs": macs, "weight_words_per_bank": wwords,
                             "program_len": gemms + elus + 1}
        if wwords > PL_LIMITS["weight_words_per_bank"]:
            ok = False
            notes.append(f"weight cache overflow: {wwords} > {PL_LIMITS['weight_words_per_bank']} words")
        if gemms + elus + 1 > PL_LIMITS["program_len"]:
            ok = False
            notes.append(f"program too long: {gemms+elus+1} > {PL_LIMITS['program_len']}")
        # The module sources list every activation get_activation() can build;
        # what matters is the one the config selects. Only ELU has an SFU ROM.
        configured = rep["derived"].get("activation")
        if configured and configured != "elu":
            ok = False
            notes.append(f"configured activation is {configured!r}, but the SFU ROM "
                         f"holds ELU only -- the datapath cannot run this policy")
        elif configured == "elu":
            notes.append("configured activation is ELU, which the SFU ROM implements")
        else:
            notes.append("could not determine the configured activation; check manually")
    else:
        ok = False
        notes.append("could not derive topology from configs; inspect manually")
    rep["compatible_with_existing_datapath"] = ok
    rep["notes"] = notes

    print(json.dumps({k: rep[k] for k in
                      ("git", "derived", "deployment", "compatible_with_existing_datapath",
                       "notes") if k in rep}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
