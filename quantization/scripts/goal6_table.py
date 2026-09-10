#!/usr/bin/env python3
"""Build the goal6 comparison tables from each arm's own evaluator output.

Nothing is transcribed by hand: success, yaw RMSE and velocity-estimation RMSE
all come out of the evaluate_locomotion.py JSON, and the parameter-drift and
learning-rate columns out of each run's own checkpoint and metrics.jsonl.
"""
from __future__ import annotations

import argparse, hashlib, json, statistics as st, sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parent.parent   # quantization/
SEGMENTS = ["forward", "reverse", "turn_left", "turn_right", "stop", "height", "combined"]

# label -> (checkpoint run dir, eval tag prefix, directory holding the tags)
ARMS = [
    ("FP32 control",              "solid_fp32_plus_v2_long", "m_L_fp32",  "v2"),
    # PTQ of the FP32 CONTROL's final weights -- the same 2000 iterations of
    # training as every QAT arm, quantized only afterwards. That is the honest
    # "no QAT" comparison and the row goal5 published. ladder_w8a8 is PTQ of the
    # untrained warm start (mean 0.041) and answers a different question.
    ("PTQ W8A8 (no QAT)",         None,                      "m_L_ptq",     "v2"),
    ("QAT W8A8 adaptive",         "qat_w8a8_v2_long",        "m_L_qat",   "v2"),
    ("QAT W8A8 fixed-LR",         "qat_v2_fixlr",            "g5_fixlr",  "v2"),
    ("QAT W8A16 adaptive",        "qat_v2_w8a16",            "g5_w8a16",  "v2"),
    ("QAT W8A8 frozen-enc adapt", "g6_e1_frzenc_adapt",      "g6_e1_frzenc_adapt", "goal6"),
    ("QAT W8A8 frozen-enc fixLR", "g6_e1_frzenc_fixlr",      "g6_e1_frzenc_fixlr", "goal6"),
    ("QAT W8A8 teacher-KL fixLR", "g6_e2_teacher_fixlr",     "g6_e2_teacher_fixlr", "goal6"),
    ("QAT W8A8 frozen+teacher",   "g6_e3_frzenc_teacher",    "g6_e3_frzenc_teacher", "goal6"),
]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def seg_row(d, key="success"):
    s = d["summary"]
    vals = [s[x][key] for x in SEGMENTS if x in s]
    return vals, (st.mean(vals) if vals else float("nan"))


def drift(ref_sd, sd, prefix):
    num = sum(((ref_sd[k].double() - sd[k].double()) ** 2).sum().item()
              for k in ref_sd if k.startswith(prefix) and k in sd)
    den = sum((ref_sd[k].double() ** 2).sum().item()
              for k in ref_sd if k.startswith(prefix))
    return num ** .5, (num / den) ** .5


def state_dict(path):
    import torch
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    d = dict(d.get("model_state_dict", d))
    if "log_std" in d:
        d["std"] = d.pop("log_std").exp()
    return d


def lr_stats(run_dir: Path):
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None
    lrs, kls, enc_kl = [], [], []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "Loss/learning_rate" in r:
            lrs.append(r["Loss/learning_rate"])
        if "Policy/approx_kl" in r:
            kls.append(r["Policy/approx_kl"])
        if "Encoder/policy_kl" in r:
            enc_kl.append(r["Encoder/policy_kl"])
    if not lrs:
        return None
    return dict(n=len(lrs), lr_median=st.median(lrs),
                lr_floor_frac=sum(abs(x - 1e-5) < 1e-12 for x in lrs) / len(lrs),
                kl_median=st.median(kls) if kls else float("nan"),
                encoder_policy_kl_median=st.median(enc_kl) if enc_kl else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path,
                    default=QUANT_ROOT / "artifacts/v2/frozen/SOLID_FP32_V2.pt")
    ap.add_argument("--json", type=Path, default=QUANT_ROOT / "artifacts/goal6/goal6_tables.json")
    a = ap.parse_args()

    ref_sd = state_dict(a.ref)
    out = {"reference": str(a.ref),
           "reference_sha256": hashlib.sha256(a.ref.read_bytes()).hexdigest(),
           "segments": SEGMENTS, "arms": {}}

    for label, run, tag, sub in ARMS:
        d = QUANT_ROOT / "artifacts" / sub
        entry = {"run": run, "tag": tag}
        for arm_letter, suffix in (("A", "_A"), ("B", "_B")):
            j = load(d / f"{tag}{suffix}")
            if j is None and arm_letter == "B":         # goal4/5 tags have no suffix
                j = load(d / tag)
            if j is None:
                continue
            succ, mean = seg_row(j, "success")
            yaw, _ = seg_row(j, "yaw_rmse_rad_s")
            vel, _ = seg_row(j, "velocity_estimation_rmse_m_s")
            entry[arm_letter] = {
                "checkpoint_sha256": j.get("checkpoint_sha256", "")[:16],
                "success": dict(zip(SEGMENTS, succ)), "success_mean": mean,
                "yaw_rmse": dict(zip(SEGMENTS, yaw)),
                "velocity_estimation_rmse": dict(zip(SEGMENTS, vel)),
            }
        if run:
            ck = TRAIN_ROOT / "checkpoints" / run / "model_2000.pt"
            if ck.exists():
                sd = state_dict(ck)
                for block in ("encoder.", "actor.", "critic."):
                    abs_, rel = drift(ref_sd, sd, block)
                    entry[f"drift_{block.strip('.')}"] = {"abs": abs_, "rel": rel}
                entry["training"] = lr_stats(TRAIN_ROOT / "checkpoints" / run)
                dg = TRAIN_ROOT / "checkpoints" / run / "digests.json"
                if dg.exists():
                    entry["digests"] = json.loads(dg.read_text())
        out["arms"][label] = entry

    # ---- printed tables -------------------------------------------------
    def table(letter, title):
        rows = [(l, e) for l, e in out["arms"].items() if letter in e]
        if not rows:
            return
        print(f"\n## {title}\n")
        print("| arm | " + " | ".join(SEGMENTS) + " | **mean** |")
        print("|---" * (len(SEGMENTS) + 2) + "|")
        for l, e in rows:
            s = e[letter]["success"]
            print(f"| {l} | " + " | ".join(f"{s.get(x, float('nan')):.3f}" for x in SEGMENTS)
                  + f" | **{e[letter]['success_mean']:.3f}** |")

    table("B", "Closed-loop success, arm B (fake-quant execution — the deployed datapath)")
    table("A", "Closed-loop success, arm A (float execution of the same parameters)")

    print("\n## Turning accuracy and encoder quality\n")
    print("| arm | yaw RMSE L | yaw RMSE R | vel-est RMSE L | vel-est RMSE R |")
    print("|---|---|---|---|---|")
    for l, e in out["arms"].items():
        if "B" not in e:
            continue
        y, v = e["B"]["yaw_rmse"], e["B"]["velocity_estimation_rmse"]
        print(f"| {l} | {y.get('turn_left', float('nan')):.4f} | {y.get('turn_right', float('nan')):.4f} "
              f"| {v.get('turn_left', float('nan')):.4f} | {v.get('turn_right', float('nan')):.4f} |")

    print("\n## Parameter drift from the shared warm start, and optimizer behaviour\n")
    print("| arm | ‖Δ encoder‖ | rel | ‖Δ actor‖ | rel | lr median | frac at floor | median PPO KL | median encoder→policy KL |")
    print("|---|---|---|---|---|---|---|---|---|")
    for l, e in out["arms"].items():
        if "drift_encoder" not in e:
            continue
        t = e.get("training") or {}
        ek = t.get("encoder_policy_kl_median")
        print(f"| {l} | {e['drift_encoder']['abs']:.2f} | {e['drift_encoder']['rel']:.3f} "
              f"| {e['drift_actor']['abs']:.2f} | {e['drift_actor']['rel']:.3f} "
              f"| {t.get('lr_median', float('nan')):.3e} | {t.get('lr_floor_frac', float('nan')):.1%} "
              f"| {t.get('kl_median', float('nan')):.4f} | "
              f"{'—' if ek is None else f'{ek:.4f}'} |")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
