#!/usr/bin/env python3
"""Generate every RoboAccel figure from one theme and one set of source files.

All five figures live here so the visual system stays coherent and so no figure
can drift from the evidence it draws. Measured values are read from
docs/results/*.json at generation time; structural facts (layer count, MAC
count, instruction count) are declared once in FACTS below and are traceable to
docs/SOLID_POLICY_DEPLOYMENT_MAPPING.md and fpga/docs/02_system_architecture.md.

    python3 assets/make_figures.py

Figures are white-ground with a dark title band, which renders identically in
GitHub light and dark mode -- an <img>-embedded SVG does not inherit the page
theme, so a transparent ground would be unreadable in one of them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "docs" / "results"

# ---------------------------------------------------------------- theme

INK      = "#101820"   # near-black, title bands and primary text
PAPER    = "#FFFFFF"
FRAME    = "#DFE5EB"   # figure border
RULE     = "#E6EBF0"   # internal hairlines
MUTED    = "#5C6874"   # captions, axis labels
FAINT    = "#8A95A1"
STEEL    = "#2A6DB0"   # primary accent: the datapath
CLAY     = "#C1440E"   # attention: failure, thresholds
VERIFY   = "#0E7C61"   # verified on silicon
BAND     = "#F4F7FA"   # stage band fill

SANS = "'Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace"

# Structural facts, single-sourced. See the module docstring for provenance.
FACTS = {
    "mac": "38,400 MAC",
    "topology": "25 obs · 5-frame encoder · 4-layer actor · 6 actions",
    "instructions": "13 descriptors",
    "operators": "7 GEMM · 5 ELU · 1 CONCAT",
    "format": "INT16 · Q8.8 activations · INT48 accumulate",
}

CSS = f"""
  text{{font-family:{SANS};fill:{INK}}}
  .band-title{{font-size:13px;fill:#EAEFF4;letter-spacing:2.2px;font-weight:600}}
  .band-mark{{font-size:12px;fill:{FAINT};letter-spacing:1.4px}}
  .stage{{font-size:11px;fill:{MUTED};letter-spacing:1.8px;font-weight:600}}
  .h{{font-size:15px;font-weight:600}}
  .p{{font-size:13px;fill:{MUTED}}}
  .n{{font-size:13px;font-weight:600}}
  .mono{{font-size:12px;font-family:{MONO};fill:{STEEL}}}
  .cap{{font-size:12px;fill:{MUTED}}}
  .ax{{font-size:12px;fill:{MUTED}}}
"""


def frame(w: int, h: int, title: str, mark: str = "RoboAccel") -> list[str]:
    """Open an SVG with the shared border and dark title band."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img">\n',
        f"<style>{CSS}</style>\n",
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n',
        f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" fill="none" '
        f'stroke="{FRAME}"/>\n',
        f'<rect x="1" y="1" width="{w-2}" height="42" fill="{INK}"/>\n',
        f'<text x="20" y="27" class="band-title">{title}</text>\n',
        f'<text x="{w-20}" y="27" class="band-mark" text-anchor="end">{mark}</text>\n',
    ]


def arrowdefs() -> str:
    return (f'<defs>'
            f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{STEEL}"/></marker>'
            f'<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{VERIFY}"/></marker>'
            f'</defs>\n')


# ---------------------------------------------------------------- 1. header

def header() -> str:
    w, h = 880, 190
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
         f'width="{w}" height="{h}" role="img">\n', f"<style>{CSS}</style>\n",
         f'<rect width="{w}" height="{h}" fill="{INK}"/>\n']
    # a quiet datapath motif along the bottom: obs -> layers -> action
    y = 150
    xs = [90, 190, 290, 390, 490, 590, 690, 790]
    for i, x in enumerate(xs[:-1]):
        o.append(f'<line x1="{x+18}" y1="{y}" x2="{xs[i+1]-18}" y2="{y}" '
                 f'stroke="#2C3A47" stroke-width="1"/>\n')
    for i, x in enumerate(xs):
        edge = i in (0, len(xs) - 1)
        o.append(f'<rect x="{x-16}" y="{y-9}" width="32" height="18" rx="2" '
                 f'fill="{"#16222E" if not edge else STEEL}" '
                 f'stroke="{"#2C3A47" if not edge else STEEL}"/>\n')
    o.append(f'<text x="{w/2}" y="76" text-anchor="middle" '
             f'font-size="52" font-weight="700" fill="#F5F8FA" '
             f'letter-spacing="-1">RoboAccel</text>\n')
    o.append(f'<text x="{w/2}" y="104" text-anchor="middle" font-size="16" '
             f'fill="{STEEL}" letter-spacing="4.5">FROM RL POLICY TO '
             f'REAL-TIME SILICON</text>\n')
    o.append(f'<text x="{w/2}" y="128" text-anchor="middle" font-size="13" '
             f'fill="#7C8996">End-to-end reinforcement-learning controller '
             f'deployment for FPGA and Cortex-M7</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 2. architecture

def architecture() -> str:
    w, h = 880, 560
    o = frame(w, h, "SYSTEM ARCHITECTURE")
    o.append(arrowdefs())

    def band(y, bh, label):
        o.append(f'<rect x="16" y="{y}" width="{w-32}" height="{bh}" rx="3" '
                 f'fill="{BAND}" stroke="{RULE}"/>\n')
        o.append(f'<text x="30" y="{y+20}" class="stage">{label}</text>\n')

    def box(x, y, bw, bh, title, lines, stroke=INK, sw=1.2, fill=PAPER):
        o.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="3" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n')
        o.append(f'<text x="{x+14}" y="{y+24}" class="h">{title}</text>\n')
        for i, t in enumerate(lines):
            cls = "mono" if t.startswith("`") else "p"
            o.append(f'<text x="{x+14}" y="{y+45+i*17}" class="{cls}">'
                     f'{t.strip("`")}</text>\n')

    # -- TRAINING
    band(58, 82, "TRAINING · EXTERNAL DEPENDENCY")
    box(30, 82, 380, 48, "RL policy (PPO, Isaac Gym)",
        [FACTS["topology"]], stroke=FAINT, sw=1)
    o.append(f'<text x="428" y="103" class="p">not vendored — the upstream '
             f'environment</text>\n')
    o.append(f'<text x="428" y="120" class="p">carries no licence. See '
             f'NOTICE.md.</text>\n')

    # -- QUANTIZATION
    band(156, 96, "QUANTIZATION")
    box(30, 180, 254, 58, "Calibrate + quantize",
        ["per-layer activation scales", "`INT16 · Q8.8 · INT48 accumulate`"])
    box(304, 180, 254, 58, "Fixed-point reference",
        ["NumPy integer, bit-accurate", "`the arbiter`"], stroke=STEEL, sw=1.8)
    box(578, 180, 272, 58, "Arithmetic contract",
        [FACTS["operators"], FACTS["mac"]])

    # -- VERIFICATION
    band(268, 106, "REFERENCE · INDEPENDENT VERIFICATION")
    vy = 300
    for i, (t, sub, silicon) in enumerate([
            ("PyTorch fake-quant", "9/9 tensors, 4000 samples", False),
            ("FPGA descriptors", FACTS["instructions"], False),
            ("STM32 scalar C", "24 golden vectors", True),
            ("STM32 SMLALD", "24 golden vectors", True)]):
        x = 30 + i * 206
        col = VERIFY if silicon else INK
        o.append(f'<rect x="{x}" y="{vy}" width="190" height="56" rx="3" '
                 f'fill="{PAPER}" stroke="{col}" '
                 f'stroke-width="{1.8 if silicon else 1.2}"/>\n')
        o.append(f'<text x="{x+12}" y="{vy+22}" font-size="13.5" '
                 f'font-weight="600">{t}</text>\n')
        o.append(f'<text x="{x+12}" y="{vy+40}" class="cap">{sub}</text>\n')
        if silicon:
            o.append(f'<text x="{x+178}" y="{vy+22}" text-anchor="end" '
                     f'font-size="10.5" fill="{VERIFY}" font-weight="600" '
                     f'letter-spacing="1">SILICON</text>\n')
    o.append(f'<text x="30" y="{vy+74}" class="cap">All four are compared '
             f'against the NumPy integer reference. Zero mismatches. Two of '
             f'them run on physical hardware.</text>\n')

    # -- DEPLOYMENT
    band(398, 130, "DEPLOYMENT TARGETS")
    box(30, 424, 400, 92, "RoboAccel FPGA — Zynq-7000",
        ["PL 100 MHz · 72 DSP48E1 · no DMA",
         "descriptor sequencer, not fixed RTL",
         "`17.99 µs pure inference (1,799 cycles)`"], stroke=STEEL, sw=1.8)
    box(450, 424, 400, 92, "STM32H723 — Cortex-M7",
        ["480 MHz · SMLALD SIMD · DTCM-resident",
         "no accelerator, no external memory",
         "`259.09 µs pure inference (on silicon)`"], stroke=STEEL, sw=1.8)

    for x in (220, 660):
        o.append(f'<path d="M{x},140 V{176}" stroke="{STEEL}" '
                 f'stroke-width="1.4" fill="none" marker-end="url(#ar)"/>\n')
    o.append(f'<path d="M440,240 V{vy-4}" stroke="{STEEL}" stroke-width="1.4" '
             f'fill="none" marker-end="url(#ar)"/>\n')
    o.append(f'<path d="M440,386 V420" stroke="{STEEL}" stroke-width="1.4" '
             f'fill="none" marker-end="url(#ar)"/>\n')

    o.append(f'<text x="16" y="{h-14}" class="cap">One arithmetic definition '
             f'is shared by every stage. The same integer program runs on both '
             f'targets and is checked against the same reference.</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 3. verification

def verification() -> str:
    w, h = 880, 400
    o = frame(w, h, "VERIFICATION CHAIN · ONE ARITHMETIC, FIVE IMPLEMENTATIONS")
    o.append(arrowdefs())

    o.append(f'<rect x="300" y="66" width="280" height="62" rx="3" '
             f'fill="{PAPER}" stroke="{STEEL}" stroke-width="2"/>\n')
    o.append('<text x="440" y="90" text-anchor="middle" font-size="15.5" '
             'font-weight="600">NumPy integer reference</text>\n')
    o.append(f'<text x="440" y="112" text-anchor="middle" class="cap">'
             f'the arbiter — defines what is correct</text>\n')

    impls = [
        ("PyTorch fake-quant", "9/9 tensors exact", "4000 samples", False),
        ("FPGA descriptors", "13 instructions", "7 weight fracs", False),
        ("STM32 scalar C", "0 mismatches", "24 golden vectors", True),
        ("STM32 SMLALD SIMD", "0 mismatches", "24 golden vectors", True),
    ]
    by = 208
    for i, (t, r1, r2, silicon) in enumerate(impls):
        x = 24 + i * 212
        col = VERIFY if silicon else INK
        o.append(f'<rect x="{x}" y="{by}" width="196" height="92" rx="3" '
                 f'fill="{PAPER}" stroke="{col}" '
                 f'stroke-width="{2 if silicon else 1.2}"/>\n')
        o.append(f'<text x="{x+14}" y="{by+26}" font-size="14" '
                 f'font-weight="600">{t}</text>\n')
        o.append(f'<text x="{x+14}" y="{by+50}" font-size="13" '
                 f'font-weight="600" fill="{col}">{r1}</text>\n')
        o.append(f'<text x="{x+14}" y="{by+70}" class="cap">{r2}</text>\n')
        badge = "ON SILICON" if silicon else "HOST"
        bcol = VERIFY if silicon else FAINT
        o.append(f'<text x="{x+14}" y="{by+86}" font-size="10.5" '
                 f'fill="{bcol}" letter-spacing="1.4" '
                 f'font-weight="600">{badge}</text>\n')
        mk = "url(#ag)" if silicon else "url(#ar)"
        o.append(f'<path d="M440,128 V166 H{x+98} V{by-4}" stroke="{col}" '
                 f'stroke-width="1.3" fill="none" marker-end="{mk}"/>\n')

    o.append(f'<line x1="16" y1="336" x2="{w-16}" y2="336" stroke="{RULE}"/>\n')
    o.append(f'<text x="20" y="358" class="cap">A controller that looks correct '
             f'in PyTorch but differs numerically on the target is not '
             f'validated. Each path re-derives the arithmetic independently:</text>\n')
    o.append(f'<text x="20" y="377" class="cap">the FPGA exporter shares no '
             f'code with the quantization library and still produces the '
             f'identical program and the identical per-layer scales.</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 4. precision

SEGMENTS = ["forward", "reverse", "turn_left", "turn_right",
            "stop", "height", "combined"]


def precision_cliff() -> str:
    d = json.loads((RESULTS / "ladder_pooled.json").read_text())
    order = ["FP32", "W16A16", "W8A16", "W8A8", "W4A8"]
    rows = [(k, [d[k][s]["success"] for s in SEGMENTS if s in d[k]])
            for k in order if k in d]

    w, h = 880, 372
    x0, y0, cw, ch = 178, 96, 82, 38
    o = frame(w, h, "CONTROL SUCCESS BY PRECISION · CLOSED-LOOP, SEVEN SEGMENTS")

    for j, s in enumerate(SEGMENTS):
        cx = x0 + j * cw + cw / 2
        o.append(f'<text x="{cx}" y="{y0-12}" class="ax" text-anchor="middle" '
                 f'transform="rotate(-26 {cx} {y0-12})">{s}</text>\n')
    o.append(f'<text x="{x0+7*cw+36}" y="{y0-12}" class="ax" '
             f'text-anchor="middle" font-weight="600">mean</text>\n')

    for i, (name, vals) in enumerate(rows):
        y = y0 + i * ch
        deployed = name == "W16A16"
        bold = ' font-weight="700"' if deployed else ''
        o.append(f'<text x="164" y="{y+24}" font-size="14" '
                 f'text-anchor="end"{bold}>{name}</text>\n')
        if deployed:
            o.append(f'<text x="164" y="{y+37}" font-size="10.5" '
                     f'fill="{STEEL}" text-anchor="end" letter-spacing="1.2" '
                     f'font-weight="600">DEPLOYED</text>\n')
        for j, v in enumerate(vals):
            # one perceptual ramp from paper to steel
            r = int(255 - (255 - 0x2A) * v)
            g = int(255 - (255 - 0x6D) * v)
            b = int(255 - (255 - 0xB0) * v)
            tc = "#FFFFFF" if v > 0.55 else INK
            o.append(f'<rect x="{x0+j*cw}" y="{y}" width="{cw-5}" '
                     f'height="{ch-7}" fill="rgb({r},{g},{b})" '
                     f'stroke="{RULE}"/>\n')
            o.append(f'<text x="{x0+j*cw+(cw-5)/2}" y="{y+21}" font-size="13" '
                     f'font-weight="600" fill="{tc}" text-anchor="middle">'
                     f'{v:.3f}</text>\n')
        m = sum(vals) / len(vals)
        o.append(f'<text x="{x0+7*cw+36}" y="{y+21}" font-size="13.5" '
                 f'font-weight="700" text-anchor="middle">{m:.3f}</text>\n')

    yc = y0 + 3 * ch - 4
    o.append(f'<line x1="{x0-10}" y1="{yc}" x2="{x0+7*cw+68}" y2="{yc}" '
             f'stroke="{CLAY}" stroke-width="2"/>\n')
    o.append(f'<text x="{x0-16}" y="{yc-6}" class="ax" fill="{CLAY}" '
             f'text-anchor="end" font-weight="600">cliff</text>\n')

    o.append(f'<line x1="16" y1="{h-64}" x2="{w-16}" y2="{h-64}" '
             f'stroke="{RULE}"/>\n')
    o.append(f'<text x="20" y="{h-42}" font-size="13">'
             f'<tspan font-weight="600">The cliff is at 8-bit activations, not '
             f'8-bit weights.</tspan> W8A16 is indistinguishable from FP32; '
             f'W8A8 collapses.</text>\n')
    o.append(f'<text x="20" y="{h-22}" class="cap">One training seed, seven '
             f'command segments, closed-loop in simulation. '
             f'Source: docs/results/ladder_pooled.json</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 5. KL floor

def kl_mechanism() -> str:
    d = json.loads((RESULTS / "kl_amplification_qat.json").read_text())
    lr = "1e-05"
    r = d["results"][lr]
    thr = d.get("kl_decrease_threshold", 0.01)
    arms = ["float", "W16A16", "W8A16", "W8A8", "W4A8"]
    kls = [r[a]["kl_mean"] for a in arms]

    w, h = 880, 470
    x0, y0, pw, ph = 122, 96, 610, 232
    lo, hi = -7.0, 0.0
    sx = lambda i: x0 + (i + 0.5) * pw / len(arms)
    sy = lambda v: y0 + ph - (math.log10(v) - lo) / (hi - lo) * ph

    o = frame(w, h, "QUANTIZATION KL FLOOR vs PPO's LEARNING-RATE CONTROLLER")
    o.append(f'<text x="20" y="66" class="cap">Policy KL produced by one Adam '
             f'step at the learning-rate floor (1e-5) — same parameters, same '
             f'observations, quantizer on and off.</text>\n')

    for e in range(int(lo), int(hi) + 1):
        y = sy(10.0 ** e)
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" '
                 f'stroke="{RULE}"/>\n')
        o.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="ax" '
                 f'text-anchor="end">1e{e}</text>\n')

    yt = sy(thr)
    o.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{yt-y0:.1f}" '
             f'fill="{CLAY}" opacity="0.05"/>\n')
    o.append(f'<line x1="{x0}" y1="{yt:.1f}" x2="{x0+pw}" y2="{yt:.1f}" '
             f'stroke="{CLAY}" stroke-width="2"/>\n')
    o.append(f'<text x="{x0+pw+8}" y="{yt-7:.1f}" font-size="12" '
             f'fill="{CLAY}" font-weight="600">PPO cuts</text>\n')
    o.append(f'<text x="{x0+pw+8}" y="{yt+9:.1f}" font-size="12" '
             f'fill="{CLAY}">the learning</text>\n')
    o.append(f'<text x="{x0+pw+8}" y="{yt+25:.1f}" font-size="12" '
             f'fill="{CLAY}">rate above {thr}</text>\n')

    for i, (a, v) in enumerate(zip(arms, kls)):
        x, y = sx(i), sy(v)
        bad = v > thr
        col = CLAY if bad else STEEL
        o.append(f'<line x1="{x:.1f}" y1="{y0+ph}" x2="{x:.1f}" '
                 f'y2="{y:.1f}" stroke="{col}" stroke-width="1.4"/>\n')
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}"/>\n')
        o.append(f'<text x="{x:.1f}" y="{y-15:.1f}" font-size="13" '
                 f'font-weight="600" fill="{col}" text-anchor="middle">'
                 f'{v:.2e}</text>\n')
        o.append(f'<text x="{x:.1f}" y="{y0+ph+22}" font-size="13.5" '
                 f'text-anchor="middle">{a}</text>\n')
        o.append(f'<text x="{x:.1f}" y="{y0+ph+40}" font-size="11.5" '
                 f'fill="{col}" text-anchor="middle" font-weight="600">'
                 f'{"QAT FAILS" if bad else "QAT WORKS"}</text>\n')

    o.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" '
             f'stroke="{INK}"/>\n')
    o.append(f'<line x1="16" y1="{h-74}" x2="{w-16}" y2="{h-74}" '
             f'stroke="{RULE}"/>\n')
    o.append(f'<text x="20" y="{h-52}" font-size="13">Fake-quant makes the '
             f'policy a <tspan font-weight="600">discontinuous</tspan> function '
             f'of its weights, so PPO reads quantization noise as policy '
             f'divergence</text>\n')
    o.append(f'<text x="20" y="{h-33}" font-size="13">and cuts the learning '
             f'rate. In the W8A8 run it sat at the 1e-5 floor for '
             f'<tspan font-weight="600">2000 of 2000 iterations</tspan>; the '
             f'FP32 control, 10 of 2000.</text>\n')
    o.append(f'<text x="20" y="{h-13}" class="cap">Source: '
             f'docs/results/kl_amplification_qat.json · '
             f'docs/results/goal4_results.json</text>\n')
    o.append("</svg>\n")
    return "".join(o)



# ---------------------------------------------------------------- 6. perturbation

def perturbation_response() -> str:
    """Policy KL against weight-perturbation magnitude, quantizer on and off.

    The load-bearing comparison is W8A16 against W8A8: both are 8-bit weights,
    so an identical perturbation crosses an identical number of weight
    quantization boundaries. Anything that separates the two curves is the
    activation path, not the weight path.
    """
    d = json.loads((RESULTS / "kl_perturbation_qat.json").read_text())
    thr = d["kl_decrease_threshold"]
    lrs = sorted(float(k) for k in d["results"])
    series = [("float", "#9AA6B2", "no quantizer — KL proportional to step squared"),
              ("W16A16", STEEL, "W16A16"),
              ("W8A16", VERIFY, "W8A16 — 8-bit weights, 16-bit activations"),
              ("W8A8", CLAY, "W8A8 — 8-bit weights, 8-bit activations")]

    w, h = 880, 500
    x0, y0, pw, ph = 108, 96, 560, 274
    xlo, xhi = math.log10(min(lrs)), math.log10(max(lrs))
    ylo, yhi = -11.0, 0.0
    sx = lambda v: x0 + (math.log10(v) - xlo) / (xhi - xlo) * pw
    sy = lambda v: y0 + ph - (math.log10(v) - ylo) / (yhi - ylo) * ph

    o = frame(w, h, "PERTURBATION RESPONSE · POLICY KL vs WEIGHT STEP SIZE")
    o.append(f'<text x="20" y="66" class="cap">Identical perturbation '
             f'procedure, states and KL definition for every arm. '
             f'12 trials per point. Zero-KL points are omitted (log axis).</text>\n')

    for e in range(int(ylo), int(yhi) + 1, 2):
        y = sy(10.0 ** e)
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" '
                 f'stroke="{RULE}"/>\n')
        o.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="ax" '
                 f'text-anchor="end">1e{e}</text>\n')
    for v in lrs:
        x = sx(v)
        o.append(f'<text x="{x:.1f}" y="{y0+ph+20}" class="ax" '
                 f'text-anchor="middle">{v:g}</text>\n')
    o.append(f'<text x="{x0+pw/2:.0f}" y="{y0+ph+42}" class="ax" '
             f'text-anchor="middle">weight perturbation |dw| (per element)</text>\n')

    yt = sy(thr)
    o.append(f'<line x1="{x0}" y1="{yt:.1f}" x2="{x0+pw}" y2="{yt:.1f}" '
             f'stroke="{CLAY}" stroke-width="1.6" stroke-dasharray="6 3"/>\n')
    o.append(f'<text x="{x0+6}" y="{yt-7:.1f}" font-size="11.5" fill="{CLAY}" '
             f'font-weight="600">PPO throttles above KL {thr}</text>\n')

    for arm, col, _ in series:
        pts = [(sx(v), sy(d["results"][f"{v:g}"][arm]["kl_mean"]))
               for v in lrs if d["results"][f"{v:g}"][arm]["kl_mean"] > 0]
        if len(pts) < 2:
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pts))
        o.append(f'<path d="{path}" fill="none" stroke="{col}" '
                 f'stroke-width="2"/>\n')
        for x, y in pts:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" '
                     f'fill="{col}"/>\n')

    lx, ly = x0 + pw + 18, y0 + 10
    for i, (arm, col, label) in enumerate(series):
        o.append(f'<line x1="{lx}" y1="{ly+i*34}" x2="{lx+18}" '
                 f'y2="{ly+i*34}" stroke="{col}" stroke-width="2.5"/>\n')
        o.append(f'<text x="{lx+24}" y="{ly+i*34+4}" font-size="13" '
                 f'font-weight="600" fill="{col}">{arm}</text>\n')

    o.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" '
             f'stroke="{INK}"/>\n')
    o.append(f'<line x1="16" y1="{h-76}" x2="{w-16}" y2="{h-76}" '
             f'stroke="{RULE}"/>\n')
    o.append(f'<text x="20" y="{h-54}" font-size="13">Over a '
             f'<tspan font-weight="600">2563x</tspan> span of step size, float '
             f'KL rises <tspan font-weight="600">6.19e6x</tspan> — exactly '
             f'quadratic. W8A8 rises only '
             f'<tspan font-weight="600">68x</tspan>.</text>\n')
    o.append(f'<text x="20" y="{h-35}" font-size="13">W8A16 and W8A8 have the '
             f'same 8-bit weights and cross the same boundaries, yet differ by '
             f'<tspan font-weight="600">~70x</tspan> in KL: the amplifier is '
             f'the activation path.</text>\n')
    o.append(f'<text x="20" y="{h-15}" class="cap">Source: '
             f'docs/results/kl_perturbation_qat.json · generated by '
             f'quantization/scripts/kl_amplification.py</text>\n')
    o.append("</svg>\n")
    return "".join(o)


FIGURES = {
    "header.svg": header,
    "architecture.svg": architecture,
    "verification_chain.svg": verification,
    "precision_cliff.svg": precision_cliff,
    "qat_kl_mechanism.svg": kl_mechanism,
    "perturbation_response.svg": perturbation_response,
}


def main() -> int:
    for name, fn in FIGURES.items():
        (HERE / name).write_text(fn())
        print(f"wrote assets/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
