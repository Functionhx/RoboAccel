#!/usr/bin/env python3
"""goal.md §32 -- aggregate the QAT recovery across independent training seeds.

Reports mean +- range rather than mean +- stdev: with three seeds a sample
standard deviation is a nearly meaningless statistic, and quoting one would
imply more precision than three points support. The spread that matters to a
reader is simply how far apart the seeds landed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_eval import pool

# peak_roll_rad is the WORST episode peak over the whole condition -- a max
# over 768 episodes, dominated by one outlier, and its recovery swings 19->108%
# between seeds that agree to 0.1 points elsewhere. mean_episode_peak_roll_rad
# is the stable companion the evaluator also reports; both are kept, but only
# the mean supports a recovery fraction.
METRICS = [("linear_tracking_rmse_m_s", "fwd RMSE"),
           ("yaw_tracking_rmse_rad_s", "yaw RMSE"),
           ("velocity_estimation_rmse_m_s", "vel-est RMSE"),
           ("mean_episode_peak_roll_rad", "mean pk roll"),
           ("peak_roll_rad", "max pk roll"),
           ("action_delta_rms", "action rate")]


def load(p: Path) -> dict:
    return pool(json.loads(p.read_text())["summaries_by_seed"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", nargs="+", required=True,
                    help="seed:fp32.json,ptq.json,qat.json")
    ap.add_argument("--scenarios", default="nominal,delay:2,push:1.0")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    arms = {}
    for t in a.triples:
        seed, files = t.split(":", 1)
        f, p, q = (Path(x) for x in files.split(","))
        arms[seed] = (load(f), load(p), load(q))

    out = {}
    for scen in a.scenarios.split(","):
        print(f"\n=== {scen} ===")
        print(f"{'metric':14s} " + " ".join(f"{'seed '+s:>16s}" for s in arms)
              + f" {'mean':>8s} {'range':>14s}")
        for key, lab in METRICS:
            recs, cells = [], []
            for s, (f, p, q) in arms.items():
                fv, pv, qv = f[scen][key], p[scen][key], q[scen][key]
                gap = pv - fv
                r = (pv - qv) / gap * 100 if abs(gap) > 1e-9 else float("nan")
                recs.append(r)
                cells.append(f"{qv:>7.4f}/{r:>6.1f}%")
            m = sum(recs) / len(recs)
            print(f"{lab:14s} " + " ".join(f"{c:>16s}" for c in cells)
                  + f" {m:>7.1f}% [{min(recs):>5.1f},{max(recs):>5.1f}]")
            out.setdefault(scen, {})[key] = {
                "per_seed_recovery_pct": recs, "mean": m,
                "min": min(recs), "max": max(recs)}
    print("\ncells are QAT value / recovery%; recovery = (PTQ-QAT)/(PTQ-FP32)")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
