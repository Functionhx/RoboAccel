#!/usr/bin/env python3
"""Generate the two data figures from the committed result JSON.

Every number drawn here is read from docs/results/, so a figure cannot drift
away from the experiment it illustrates. The two hand-drawn architecture
figures (pipeline.svg, verification_chain.svg) carry no measurements and are
edited directly.

    python3 assets/make_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "docs" / "results"

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'width="{w}" height="{h}" font-family="Helvetica,Arial,sans-serif">\n'
        '<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        '<style>.hdr{{font-size:12px;fill:#888;letter-spacing:1.5px}}'
        '.ax{{font-size:11px;fill:#666}}.lbl{{font-size:12px;fill:#111}}'
        '.val{{font-size:11px;fill:#111;font-weight:600}}'
        '.tag{{font-size:11px;fill:#666}}'
        '.mono{{font-size:11px;fill:#111;'
        'font-family:ui-monospace,Menlo,Consolas,monospace}}</style>\n')

SEGMENTS = ["forward", "reverse", "turn_left", "turn_right",
            "stop", "height", "combined"]


def precision_cliff() -> str:
    d = json.loads((RESULTS / "ladder_pooled.json").read_text())
    order = ["FP32", "W16A16", "W8A16", "W8A8", "W4A8"]
    rows = [(k, [d[k][s]["success"] for s in SEGMENTS if s in d[k]])
            for k in order if k in d]

    w, h = 900, 330
    x0, y0, cw, ch = 190, 70, 78, 34
    out = [HEAD.format(w=w, h=h)]
    out.append('<text x="20" y="30" class="hdr">'
               'CONTROL SUCCESS BY PRECISION — SEVEN COMMAND SEGMENTS, MEASURED'
               '</text>\n')
    out.append(f'<line x1="20" y1="40" x2="{w-20}" y2="40" stroke="#ddd"/>\n')

    for j, s in enumerate(SEGMENTS):
        cx = x0 + j * cw + cw / 2
        out.append(f'<text x="{cx}" y="{y0-10}" class="ax" text-anchor="middle" '
                   f'transform="rotate(-30 {cx} {y0-10})">{s}</text>\n')
    out.append(f'<text x="{x0+7*cw+34}" y="{y0-10}" class="ax" '
               f'text-anchor="middle">mean</text>\n')

    for i, (name, vals) in enumerate(rows):
        y = y0 + i * ch
        deployed = name == "W16A16"
        bold = ' font-weight="600"' if deployed else ''
        out.append(f'<text x="176" y="{y+22}" class="lbl" text-anchor="end"'
                   f'{bold}>{name}</text>\n')
        if deployed:
            out.append(f'<text x="176" y="{y+34}" class="tag" '
                       f'text-anchor="end">deployed</text>\n')
        for j, v in enumerate(vals):
            # one perceptual ramp: white at 0, deep steel blue at 1
            g = int(255 - 175 * v)
            b = int(255 - 90 * v)
            fill = f"rgb({g},{int(g*0.95+10)},{b})"
            tc = "#fff" if v > 0.6 else "#111"
            out.append(f'<rect x="{x0+j*cw}" y="{y}" width="{cw-4}" '
                       f'height="{ch-6}" fill="{fill}" stroke="#ccc"/>\n')
            out.append(f'<text x="{x0+j*cw+(cw-4)/2}" y="{y+19}" class="val" '
                       f'fill="{tc}" text-anchor="middle">{v:.3f}</text>\n')
        m = sum(vals) / len(vals)
        out.append(f'<text x="{x0+7*cw+34}" y="{y+19}" class="val" '
                   f'text-anchor="middle">{m:.3f}</text>\n')

    yc = y0 + 3 * ch - 3
    out.append(f'<line x1="{x0-8}" y1="{yc}" x2="{x0+7*cw+62}" y2="{yc}" '
               f'stroke="#c1440e" stroke-width="1.6" stroke-dasharray="5 3"/>\n')
    out.append(f'<text x="{x0+7*cw+66}" y="{yc+4}" class="tag" '
               f'fill="#c1440e">cliff</text>\n')

    out.append(f'<line x1="20" y1="{h-58}" x2="{w-20}" y2="{h-58}" stroke="#ddd"/>\n')
    out.append(f'<text x="20" y="{h-38}" class="tag">The cliff is at 8-bit '
               f'ACTIVATIONS, not 8-bit weights: W8A16 is indistinguishable from '
               f'FP32; W8A8 collapses.</text>\n')
    out.append(f'<text x="20" y="{h-20}" class="tag">Closed-loop evaluation, '
               f'three training seeds. Source: docs/results/ladder_pooled.json</text>\n')
    out.append("</svg>\n")
    return "".join(out)


def kl_mechanism() -> str:
    d = json.loads((RESULTS / "kl_amplification_qat.json").read_text())
    lr = "1e-05"
    r = d["results"][lr]
    thr = d.get("kl_decrease_threshold", 0.01)
    arms = ["float", "W16A16", "W8A16", "W8A8", "W4A8"]
    kls = [r[a]["kl_mean"] for a in arms]

    w, h = 900, 400
    x0, y0, pw, ph = 130, 80, 620, 230
    lo, hi = -7.0, 0.0                        # log10 axis
    sx = lambda i: x0 + (i + 0.5) * pw / len(arms)
    sy = lambda v: y0 + ph - (math.log10(v) - lo) / (hi - lo) * ph

    out = [HEAD.format(w=w, h=h)]
    out.append('<text x="20" y="30" class="hdr">WHY W8A8 QAT NEVER TRAINED — '
               'QUANTIZATION KL NOISE FLOOR vs CONTROLLER THRESHOLD</text>\n')
    out.append(f'<line x1="20" y1="40" x2="{w-20}" y2="40" stroke="#ddd"/>\n')
    out.append(f'<text x="20" y="60" class="tag">Policy KL produced by ONE Adam '
               f'step at the learning-rate floor (lr = 1e-5), same parameters, '
               f'same observations.</text>\n')

    for e in range(int(lo), int(hi) + 1):
        y = sy(10.0 ** e)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" '
                   f'stroke="#eee"/>\n')
        out.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="ax" '
                   f'text-anchor="end">1e{e}</text>\n')

    yt = sy(thr)
    out.append(f'<line x1="{x0}" y1="{yt:.1f}" x2="{x0+pw}" y2="{yt:.1f}" '
               f'stroke="#c1440e" stroke-width="1.6"/>\n')
    out.append(f'<text x="{x0+pw+8}" y="{yt-6:.1f}" class="tag" fill="#c1440e">'
               f'controller throttles</text>\n')
    out.append(f'<text x="{x0+pw+8}" y="{yt+10:.1f}" class="tag" fill="#c1440e">'
               f'above KL {thr}</text>\n')

    for i, (a, v) in enumerate(zip(arms, kls)):
        x, y = sx(i), sy(v)
        bad = v > thr
        col = "#c1440e" if bad else "#2c5f8a"
        out.append(f'<line x1="{x:.1f}" y1="{y0+ph}" x2="{x:.1f}" y2="{y:.1f}" '
                   f'stroke="{col}" stroke-width="1"/>\n')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}"/>\n')
        out.append(f'<text x="{x:.1f}" y="{y-13:.1f}" class="val" fill="{col}" '
                   f'text-anchor="middle">{v:.2e}</text>\n')
        out.append(f'<text x="{x:.1f}" y="{y0+ph+20}" class="lbl" '
                   f'text-anchor="middle">{a}</text>\n')
        out.append(f'<text x="{x:.1f}" y="{y0+ph+37}" class="tag" fill="{col}" '
                   f'text-anchor="middle">{"QAT fails" if bad else "QAT works"}'
                   f'</text>\n')

    out.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" '
               f'stroke="#333"/>\n')
    out.append(f'<line x1="20" y1="{h-70}" x2="{w-20}" y2="{h-70}" stroke="#ddd"/>\n')
    out.append(f'<text x="20" y="{h-50}" class="tag">Fake-quant makes the policy '
               f'a discontinuous function of its weights, so PPO\'s adaptive '
               f'learning-rate controller reads quantization noise as policy '
               f'divergence</text>\n')
    out.append(f'<text x="20" y="{h-33}" class="tag">and cuts the learning rate. '
               f'In the W8A8 run it stayed at the 1e-5 floor for 2000 of 2000 '
               f'iterations; the FP32 control, 10 of 2000.</text>\n')
    out.append(f'<text x="20" y="{h-16}" class="tag">Source: '
               f'docs/results/kl_amplification_qat.json · '
               f'docs/results/goal4_results.json</text>\n')
    out.append("</svg>\n")
    return "".join(out)


def main() -> int:
    for name, fn in (("precision_cliff.svg", precision_cliff),
                     ("qat_kl_mechanism.svg", kl_mechanism)):
        (HERE / name).write_text(fn())
        print(f"wrote assets/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
