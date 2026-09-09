#!/usr/bin/env python3
"""goal.md §31/§32 tables from Codex's locomotion sequence evaluator.

Different schema from evaluate_robustness.py: results are per command segment
(forward / reverse / turn_left / turn_right / stop / height / combined) rather
than per perturbation scenario, and the decisive metric is a pass/fail
`success` plus `support_fraction`, not a tracking RMSE.

`success` is a rate over environments, so it is averaged across seeds. RMSE
columns are pooled as root-mean-square, since averaging RMSEs is the wrong
statistic for a squared-error quantity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RMSE = ("forward_rmse_m_s", "yaw_rmse_rad_s", "height_rmse_m",
        "velocity_estimation_rmse_m_s")
RATE = ("success", "support_fraction", "body_contact_fraction",
        "failure_count", "reset_count")
SEGS = ("forward", "reverse", "turn_left", "turn_right", "stop", "height", "combined")


def summ(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]


def pool(runs: list[dict]) -> dict:
    """Pool one arm across seeds, per segment."""
    out = {}
    for seg in SEGS:
        rows = [r[seg] for r in runs if seg in r]
        if not rows:
            continue
        rec = {"seeds": len(rows)}
        for k in RMSE:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                rec[k] = math.sqrt(sum(v * v for v in vals) / len(vals))
        for k in RATE:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vals:
                rec[k] = sum(vals) / len(vals)
                rec[k + "_min"] = min(vals)
                rec[k + "_max"] = max(vals)
        out[seg] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="LABEL=file1[,file2,...]  (multiple files = seeds)")
    ap.add_argument("--metrics", default="success,support_fraction,yaw_rmse_rad_s,forward_rmse_m_s")
    ap.add_argument("--recovery-from", default=None,
                    help="FP32_LABEL,PTQ_LABEL,QAT_LABEL for a recovery column")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    arms = {}
    for spec in a.arm:
        lab, files = spec.split("=", 1)
        paths = [Path(f) for f in files.split(",")]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  ! {lab}: missing {[str(m) for m in missing]}")
            paths = [p for p in paths if p.exists()]
        if paths:
            arms[lab] = pool([summ(p) for p in paths])

    mets = a.metrics.split(",")
    for m in mets:
        print(f"\n=== {m} ===")
        hdr = f"{'segment':11s}" + "".join(f"{l:>18s}" for l in arms)
        print(hdr); print("-" * len(hdr))
        for seg in SEGS:
            row = f"{seg:11s}"
            for lab in arms:
                v = arms[lab].get(seg, {}).get(m)
                if isinstance(v, (int, float)):
                    n = arms[lab][seg]["seeds"]
                    lo = arms[lab][seg].get(m + "_min")
                    cell = (f"{v:.3f}" if n == 1 or lo is None
                            else f"{v:.3f}[{lo:.2f},{arms[lab][seg][m+'_max']:.2f}]")
                else:
                    cell = "-"
                row += f"{cell:>18s}"
            print(row)

    if a.recovery_from:
        f, p, q = a.recovery_from.split(",")
        print(f"\n=== recovery: (PTQ-QAT)/(PTQ-FP32), arms {f}/{p}/{q} ===")
        print(f"{'segment':11s}" + "".join(f"{m[:16]:>18s}" for m in mets))
        for seg in SEGS:
            row = f"{seg:11s}"
            for m in mets:
                try:
                    fv, pv, qv = (arms[x][seg][m] for x in (f, p, q))
                    gap = pv - fv
                    row += f"{(pv-qv)/gap*100:>17.1f}%" if abs(gap) > 1e-9 else f"{'n/a':>18s}"
                except Exception:
                    row += f"{'-':>18s}"
            print(row)

    if a.json:
        a.json.write_text(json.dumps(arms, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
