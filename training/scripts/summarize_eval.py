#!/usr/bin/env python3
"""Pool Codex evaluator output across eval seeds into one row per scenario.

The evaluator reports per seed. goal.md wants one comparable number per
scenario per arm, and the pooling has to respect how each metric is defined:

  * RMSE metrics are pooled as sqrt(mean of squares) weighted by episodes, not
    averaged -- averaging RMSEs understates spread and is simply the wrong
    statistic for a squared-error quantity;
  * rates (fall, failure) are episode-weighted means;
  * peak tilt is a worst case, so it is a max across seeds, matching the
    evaluator's own within-seed convention;
  * survival is right-censored at the horizon, so a 20.0 s mean means "no
    episode ended early", not "the policy survived exactly 20 s".

Nothing here recomputes a metric from raw episodes; it only aggregates what the
frozen evaluator already produced.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RMSE = ("linear_tracking_rmse_m_s", "yaw_tracking_rmse_rad_s",
        "velocity_estimation_rmse_m_s", "action_delta_rms")
MEAN = ("survival_s", "torque_saturation_fraction", "fall_rate",
        "failure_termination_rate", "timeout", "edge_reset",
        "mean_episode_peak_roll_rad", "mean_episode_peak_pitch_rad")
MAXV = ("peak_roll_rad", "peak_pitch_rad")


def pool(summaries: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for s in summaries:
        by.setdefault(s["scenario"], []).append(s)
    out = {}
    for scen, rows in by.items():
        n = sum(r["episodes"] for r in rows)
        rec = {"episodes": n, "seeds": len(rows)}
        for k in RMSE:
            rec[k] = math.sqrt(sum(r[k] ** 2 * r["episodes"] for r in rows) / n)
        for k in MEAN:
            rec[k] = sum(r[k] * r["episodes"] for r in rows) / n
        for k in MAXV:
            rec[k] = max(r[k] for r in rows)
        out[scen] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+",
                    help="evaluator .json files, one per arm")
    ap.add_argument("--labels", default=None,
                    help="comma-separated arm labels, defaulting to file stems")
    ap.add_argument("--scenarios", default="nominal,delay:2,friction:0.4,mass:2,push:1.0")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    labels = (a.labels.split(",") if a.labels
              else [p.stem for p in a.results])
    if len(labels) != len(a.results):
        raise SystemExit("--labels count must match the number of result files")

    arms = {}
    for lab, path in zip(labels, a.results):
        d = json.loads(path.read_text())
        arms[lab] = {"pooled": pool(d["summaries_by_seed"]),
                     "checkpoint_sha256": d.get("checkpoint_sha256", "")[:16],
                     "iteration": d.get("checkpoint_iteration")}

    want = [s for s in a.scenarios.split(",")]
    hdr = f"{'scenario':16s} {'arm':12s} {'fwd RMSE':>9s} {'yaw RMSE':>9s} " \
          f"{'fall':>6s} {'surv s':>7s} {'d act':>7s} {'vel RMSE':>9s} {'pk roll':>8s}"
    print(hdr); print("-" * len(hdr))
    for scen in want:
        for lab in labels:
            r = arms[lab]["pooled"].get(scen)
            if r is None:
                continue
            print(f"{scen:16s} {lab:12s} {r['linear_tracking_rmse_m_s']:>9.4f} "
                  f"{r['yaw_tracking_rmse_rad_s']:>9.4f} {r['fall_rate']:>6.3f} "
                  f"{r['survival_s']:>7.2f} {r['action_delta_rms']:>7.4f} "
                  f"{r['velocity_estimation_rmse_m_s']:>9.5f} {r['peak_roll_rad']:>8.4f}")
        print()

    if a.json:
        a.json.write_text(json.dumps(arms, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
