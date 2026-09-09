#!/usr/bin/env python3
"""Collect every Goal 4 number from its source file into one table.

Nothing here is transcribed by hand. Each row names the file it came from, so
a claim in goal4_final_analysis.md can be checked against the artifact rather
than against another document.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QAT = ROOT / "qat"
OUT = Path(__file__).resolve().parent

# Every arm below shares: task locomotion_v2, seed 1, 4096 envs, 2000 iters,
# and warm start from SOLID_FP32_V2 (sha ab20c190...). Anything that differs is
# named in the "difference" column.
RUNS = {
    "solid_fp32_plus_v2_long": "FP32 control (no quantization)",
    "qat_w8a8_v2_long":        "QAT W8A8, shipped hyper-parameters",
    "qat_v2_yaw2":             "QAT W8A8, tracking_ang_vel x2",
    "qat_v2_yaw4":             "QAT W8A8, tracking_ang_vel x4",
    "qat_v2_fixlr":            "QAT W8A8, schedule=fixed lr=2.563e-4",
    "qat_v2_w8a16":            "QAT W8A16 (positive control)",
}


def load(run: str) -> list[dict]:
    p = QAT / "checkpoints" / run / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass          # a run killed mid-write leaves one truncated line
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    tail = rows[-50:]
    lrs = [r["Loss/learning_rate"] for r in rows]
    kls = sorted(r["Policy/approx_kl"] for r in rows)
    mean = lambda k: sum(r[k] for r in tail) / len(tail)
    return {
        "iterations": len(rows),
        "lr_floor_fraction": sum(1 for x in lrs if x <= 1.0001e-5) / len(lrs),
        "lr_median": sorted(lrs)[len(lrs) // 2],
        "kl_median": kls[len(kls) // 2],
        "kl_over_threshold_fraction": sum(1 for x in kls if x > 0.01) / len(kls),
        "yaw_rmse_rad_s": mean("Locomotion/yaw_rmse_rad_s"),
        "forward_rmse_m_s": mean("Locomotion/forward_rmse_m_s"),
        "angvel_reward_contribution": mean("Reward/tracking_ang_vel/contribution/mean"),
        "mean_reward": mean("Train/mean_reward"),
        "encoder_policy_kl": mean("Encoder/policy_kl"),
        "yaw_rmse_at_iter1": rows[0]["Locomotion/yaw_rmse_rad_s"],
    }


def main() -> int:
    out = {"training_runs": {}, "kl_amplification": {}}
    for run, label in RUNS.items():
        s = summarize(load(run))
        if s:
            out["training_runs"][run] = {"label": label, **s}

    for tag in ("qat", "fp32"):
        p = QAT / "artifacts/v2" / f"kl_amplification_{tag}.json"
        if p.exists():
            out["kl_amplification"][tag] = json.loads(p.read_text())

    (OUT / "goal4_results.json").write_text(json.dumps(out, indent=2))

    lines = ["| run | iters | lr@floor | lr median | KL median | KL>0.01 | "
             "yaw RMSE | fwd RMSE | angvel rew | mean rew |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run, r in out["training_runs"].items():
        lines.append(
            f"| `{run}` | {r['iterations']} | {100*r['lr_floor_fraction']:.1f}% | "
            f"{r['lr_median']:.2e} | {r['kl_median']:.4f} | "
            f"{100*r['kl_over_threshold_fraction']:.1f}% | "
            f"{r['yaw_rmse_rad_s']:.4f} | {r['forward_rmse_m_s']:.4f} | "
            f"{r['angvel_reward_contribution']:.5f} | {r['mean_reward']:.2f} |")
    table = "\n".join(lines)
    (OUT / "goal4_results_table.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {OUT/'goal4_results.json'} and {OUT/'goal4_results_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
