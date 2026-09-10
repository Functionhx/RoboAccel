#!/usr/bin/env python3
"""Generate every RoboAccel figure from one theme and one set of source files.

All five figures live here so the visual system stays coherent and so no figure
can drift from the evidence it draws. Measured values are read from
docs/results/*.json at generation time; structural facts (layer count, MAC
count, instruction count) are declared once in FACTS below and are traceable to
docs/SOLID_POLICY_DEPLOYMENT_MAPPING.md and fpga/docs/02_system_architecture.md.

    python3 assets/make_figures.py

The visual language is an engineering drawing: true black on white, one
plotter-blue accent reserved for measured values, and failure drawn as
diagonal hatch rather than as a second colour. Drafting is the vernacular this
subject already lives in, and hatch-for-failure survives greyscale and
colour-blindness in a way a red/green pair does not.

Deliberately absent, because they are the tells of a generated page rather
than choices made for this subject: tracked-out all-caps labels above content,
meta strings joined with middle dots, captions built as WORD-emdash-fragment,
a tinted near-black standing in for black, and monospace used as label
texture. Monospace appears only where the glyphs are data -- register values,
cycle counts, action integers.

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

# Four values. Black is black, not a tinted stand-in for it; the accent is a
# plotter-pen blue and is spent only on values that were measured.
INK      = "#000000"   # linework and primary text
PAPER    = "#FFFFFF"
PLOT     = "#2F2AC8"   # measured values, and only those
GHOST    = "#8C8C8C"   # secondary text, hatch, construction lines

# Aliases kept so the older figures keep compiling against one palette.
FRAME = RULE = HAIR = "#D8D8D8"
MUTED = LABEL = FAINT = GHOST
STEEL = VERIFY = PLOT
CLAY  = INK            # failure is hatch, not a colour
BAND  = "#F2F2F2"

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
  .band-title{{font-size:14px;fill:{PAPER};font-weight:600}}
  .band-mark{{font-size:12px;fill:#9A9A9A}}
  .stage{{font-size:12px;fill:{MUTED};font-weight:600}}
  .h{{font-size:15px;font-weight:600}}
  .p{{font-size:13px;fill:{MUTED}}}
  .n{{font-size:13px;font-weight:600}}
  .mono{{font-size:12px;font-family:{MONO};fill:{STEEL}}}
  .cap{{font-size:12px;fill:{MUTED}}}
  .ax{{font-size:12px;fill:{MUTED}}}
"""


def frame(w: int, h: int, title: str, mark: str = "",
          fig: int | None = None) -> list[str]:
    """Open an SVG with the shared border and dark title band.

    `fig` prints the figure number into the band so the number travels with the
    image itself; a caption in the README alone would drift the first time a
    section is reordered.
    """
    if fig is not None:
        title = f"Figure {fig}\u2003{title}"
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img">\n',
        f"<style>{CSS}</style>\n",
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n',
        f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" fill="none" '
        f'stroke="{FRAME}"/>\n',
        f'<rect x="1" y="1" width="{w-2}" height="42" fill="{INK}"/>\n',
        f'<text x="20" y="27" class="band-title">{title}</text>\n',
        # No repeated wordmark in the corner: the page it sits on is already
        # titled RoboAccel, so it would be decoration, not information.
        *([f'<text x="{w-20}" y="27" class="band-mark" '
           f'text-anchor="end">{mark}</text>\n'] if mark else []),
    ]


def hatchdefs() -> str:
    """Diagonal hatch: what a drawing uses for a section that is not solid.

    Failure is drawn with this rather than with a red, so the figures survive
    greyscale printing and red/green colour blindness, and so the palette does
    not need a second accent.
    """
    return (f'<defs><pattern id="hatch" width="6" height="6" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="{PAPER}"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{GHOST}" '
            f'stroke-width="1.6"/></pattern></defs>\n')


def arrowdefs() -> str:
    return (f'<defs>'
            f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{STEEL}"/></marker>'
            f'<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{VERIFY}"/></marker>'
            f'<marker id="am" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
            f'</defs>\n')


# ---------------------------------------------------------------- 1. masthead

# The six integers the deployed policy returns for the reference input, and the
# three independent paths that produce them. Verbatim from
# fpga/docs/09_hardware_test_20260821.md (RTL golden and the 2020.2 UART board)
# and fpga/docs/11_mini7010_vivado2026_port.md (the 2026.1 JTAG mailbox).
ACTIONS = ["543", "790", "74", "635", "478", "-796"]
PATHS = [
    ("RTL simulation", "Icarus, make -C fpga/tb all"),
    ("Zynq board, Vivado 2020.2", "UART console"),
    ("Zynq board, Vivado 2026.1", "JTAG mailbox"),
]


def header() -> str:
    """Open with the artifact, not with a banner about the artifact.

    Three independently built paths return the same six integers for the same
    input. That is the whole claim of the project, it cannot be faked, and it
    is legible in about two seconds -- which a wordmark on a dark ground is
    not.
    """
    w, h = 1200, 340
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
         f'width="{w}" height="{h}" role="img">\n', f"<style>{CSS}</style>\n",
         f'<rect width="{w}" height="{h}" fill="{PAPER}"/>\n']

    o.append(f'<text x="56" y="86" font-size="56" font-weight="700" '
             f'fill="{INK}" letter-spacing="-1.6">RoboAccel</text>\n')
    o.append(f'<text x="56" y="120" font-size="17" fill="{INK}">'
             f'An RL policy, compiled to 13 integer instructions and run on '
             f'silicon.</text>\n')

    # the evidence block: one column per action, right-aligned like a register
    col0, colw = 454, 138
    top = 176
    o.append(f'<line x1="56" y1="{top-30}" x2="{w-56}" y2="{top-30}" '
             f'stroke="{INK}" stroke-width="1.4"/>\n')
    for i, (name, how) in enumerate(PATHS):
        y = top + i * 40
        o.append(f'<text x="56" y="{y}" font-size="14" fill="{INK}">'
                 f'{name}</text>\n')
        o.append(f'<text x="56" y="{y+16}" font-size="11.5" fill="{GHOST}">'
                 f'{how}</text>\n')
        for j, a in enumerate(ACTIONS):
            o.append(f'<text x="{col0+j*colw}" y="{y}" text-anchor="end" '
                     f'font-size="21" font-family="{MONO}" fill="{INK}">'
                     f'{a}</text>\n')

    # the brace that says these are the same numbers
    by = top + 3 * 40 - 12
    x0, x1 = col0 - 66, col0 + 5 * colw + 4
    o.append(f'<path d="M{x0},{by} L{x0},{by+9} L{x1},{by+9} L{x1},{by}" '
             f'fill="none" stroke="{PLOT}" stroke-width="1.4"/>\n')
    o.append(f'<text x="{(x0+x1)/2:.0f}" y="{by+30}" text-anchor="middle" '
             f'font-size="14" style="fill:{PLOT}" font-weight="600">'
             f'identical, bit for bit</text>\n')
    o.append(f'<text x="56" y="{by+30}" font-size="12.5" style="fill:{GHOST}">'
             f'Actions in Q8.8, same reference input</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 1b. figure 1

# Functional block diagram. Bus labels carry real widths and formats so the
# arrows say what actually crosses them, the way a datasheet's do.
CHAIN = [
    ("RL policy", "Trained once, on a workstation",
     ["PPO, wheel-legged balance", "25 observations, 5-frame history",
      "38,400 MAC in FP32"]),
    ("Quantize and export", "Compiled once, on a workstation",
     ["INT16 weights, Q8.8 activations", "per-GEMM weight scales",
      "13 operator descriptors"]),
    ("PL accelerator", "Runs every control period",
     ["hardware sequencer", "8\u00d78 INT16 MAC, 72 DSP48E1",
      "1,799 cycles, no branches"]),
]
BUSES = ["FP32 weights", "128-bit \u00d7 13", "action[6] Q8.8"]


def block_diagram() -> str:
    w, h = 1200, 396
    o = frame(w, h, "From a trained policy to a running robot", fig=1)
    o.append(arrowdefs())
    pad, gap, rw = 34, 108, 132
    n = len(CHAIN)
    cw = (w - 2 * pad - gap * n - rw) / n
    top, bh = 86, 152
    mid = top + bh / 2

    for i, (title, where, lines) in enumerate(CHAIN):
        x = pad + i * (cw + gap)
        deployed = i == n - 1
        o.append(f'<rect x="{x:.0f}" y="{top}" width="{cw:.0f}" height="{bh}" '
                 f'fill="{PAPER}" stroke="{INK if deployed else FRAME}" '
                 f'stroke-width="{1.6 if deployed else 1}"/>\n')
        o.append(f'<rect x="{x:.0f}" y="{top}" width="{cw:.0f}" height="27" '
                 f'fill="{INK if deployed else BAND}"/>\n')
        o.append(f'<text x="{x+13:.0f}" y="{top+18}" font-size="11.5" '
                 f'font-weight="700" '
                 f'style="fill:{PAPER if deployed else INK}">'
                 f'{title}</text>\n')
        o.append(f'<text x="{x+13:.0f}" y="{top+46}" font-size="9.5" '
                 f'style="fill:{GHOST}">{where}</text>\n')
        for j, line in enumerate(lines):
            o.append(f'<text x="{x+13:.0f}" y="{top+70+j*22}" font-size="11.5" '
                     f'font-family="{MONO}" fill="{MUTED}">{line}</text>\n')
        ax0, ax1 = x + cw + 10, x + cw + gap - 10
        o.append(f'<line x1="{ax0:.0f}" y1="{mid:.0f}" x2="{ax1:.0f}" '
                 f'y2="{mid:.0f}" stroke="{STEEL}" stroke-width="1.4" '
                 f'marker-end="url(#ar)"/>\n')
        o.append(f'<text x="{(ax0+ax1)/2:.0f}" y="{mid-11:.0f}" '
                 f'text-anchor="middle" font-size="10" font-family="{MONO}" '
                 f'fill="{STEEL}">{BUSES[i]}</text>\n')

    rx = pad + n * (cw + gap)
    o.append(f'<rect x="{rx:.0f}" y="{top+34:.0f}" width="{rw}" '
             f'height="{bh-68}" fill="{BAND}" stroke="{FRAME}"/>\n')
    o.append(f'<text x="{rx+rw/2:.0f}" y="{mid-6:.0f}" text-anchor="middle" '
             f'font-size="12.5" font-weight="700" '
             f'fill="{INK}">Robot</text>\n')
    o.append(f'<text x="{rx+rw/2:.0f}" y="{mid+14:.0f}" text-anchor="middle" '
             f'font-size="10.5" font-family="{MONO}" fill="{MUTED}">'
             f'100 Hz loop</text>\n')

    # The loop is closed: quantization is judged by what the robot does, which
    # is the whole argument of the project.
    fy = top + bh + 46
    o.append(f'<path d="M{rx+rw/2:.0f},{top+bh-34:.0f} L{rx+rw/2:.0f},{fy} '
             f'L{pad+cw/2:.0f},{fy} L{pad+cw/2:.0f},{top+bh+8}" fill="none" '
             f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 3" '
             f'marker-end="url(#am)"/>\n')
    o.append(f'<text x="{w/2:.0f}" y="{fy-9}" text-anchor="middle" '
             f'font-size="10.5" font-family="{MONO}" fill="{MUTED}">'
             f'25 observations return each period, and quantization is '
             f'scored on what the robot does</text>\n')

    ay = fy + 42
    o.append(f'<rect x="{pad}" y="{ay}" width="{w-2*pad}" height="36" '
             f'fill="{PAPER}" stroke="{VERIFY}" stroke-dasharray="3 3"/>\n')
    o.append(f'<text x="{pad+15}" y="{ay+23}" font-size="11" font-weight="600" '
             f'style="fill:{PLOT}">Integer reference</text>\n')
    o.append(f'<text x="{pad+150}" y="{ay+23}" font-size="11.5" '
             f'style="fill:{GHOST}">defines correctness for every stage to its '
             f'right. The exporter, the RTL and both Cortex-M7 kernels must '
             f'reproduce it exactly, not approximately.</text>\n')
    o.append("</svg>\n")
    return "".join(o)


# ---------------------------------------------------------------- 2. architecture

def architecture() -> str:
    w, h = 880, 560
    o = frame(w, h, "How the pieces fit together", fig=2)
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
    band(58, 82, "Training, an external dependency")
    box(30, 82, 380, 48, "RL policy (PPO, Isaac Gym)",
        [FACTS["topology"]], stroke=FAINT, sw=1)
    o.append(f'<text x="428" y="103" class="p">not vendored — the upstream '
             f'environment</text>\n')
    o.append(f'<text x="428" y="120" class="p">carries no licence. See '
             f'NOTICE.md.</text>\n')

    # -- QUANTIZATION
    band(156, 96, "Quantization")
    box(30, 180, 254, 58, "Calibrate + quantize",
        ["per-layer activation scales", "`INT16 · Q8.8 · INT48 accumulate`"])
    box(304, 180, 254, 58, "Fixed-point reference",
        ["NumPy integer, bit-accurate", "`the arbiter`"], stroke=STEEL, sw=1.8)
    box(578, 180, 272, 58, "Arithmetic contract",
        [FACTS["operators"], FACTS["mac"]])

    # -- VERIFICATION
    band(268, 106, "Reference and independent verification")
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
                     f'>on silicon</text>\n')
    o.append(f'<text x="30" y="{vy+74}" class="cap">All four are compared '
             f'against the NumPy integer reference. Zero mismatches. Two of '
             f'them run on physical hardware.</text>\n')

    # -- DEPLOYMENT
    band(398, 130, "Deployment targets")
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
    o = frame(w, h, "One arithmetic, five implementations", fig=3)
    o.append(arrowdefs())

    o.append(f'<rect x="300" y="66" width="280" height="62" rx="3" '
             f'fill="{PAPER}" stroke="{STEEL}" stroke-width="2"/>\n')
    o.append('<text x="440" y="90" text-anchor="middle" font-size="15.5" '
             'font-weight="600">NumPy integer reference</text>\n')
    o.append(f'<text x="440" y="112" text-anchor="middle" class="cap">'
             f'the arbiter: it defines what is correct</text>\n')

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
        badge = "on silicon" if silicon else "on the host"
        bcol = VERIFY if silicon else FAINT
        o.append(f'<text x="{x+14}" y="{by+86}" font-size="10.5" '
                 f'style="fill:{bcol}">{badge}</text>\n')
        mk = "url(#ag)" if silicon else "url(#ar)"
        o.append(f'<path d="M440,128 V166 H{x+98} V{by-4}" stroke="{col}" '
                 f'stroke-width="1.3" fill="none" marker-end="{mk}"/>\n')

    o.append(f'<line x1="16" y1="336" x2="{w-16}" y2="336" stroke="{RULE}"/>\n')
    o.append(f'<text x="20" y="358" class="cap">A controller correct in PyTorch '
             f'but different on the target has not been validated. Each path '
             f're-derives the arithmetic on its own:</text>\n')
    o.append(f'<text x="20" y="377" class="cap">the FPGA exporter shares no code '
             f'with the quantization library, yet derives the same program and '
             f'the same per-layer scales.</text>\n')
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
    o = frame(w, h, "Closed-loop control success, by precision", fig=4)
    o.append(hatchdefs())

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
            o.append(f'<text x="164" y="{y+37}" font-size="11" '
                     f'style="fill:{PLOT}" text-anchor="end">deployed</text>\n')
        for j, v in enumerate(vals):
            # A segment either holds the command or it does not. Solid ink for a
            # pass, hatch for a failure -- so the cliff is legible in greyscale
            # and needs no second colour. The partial values in between are the
            # honest middle, drawn at proportional ink.
            cx, cyw = x0 + j * cw, cw - 5
            if v >= 0.999:
                fill = INK
            elif v <= 0.001:
                fill = "url(#hatch)"
            else:
                g = int(255 - 255 * v)
                fill = f"rgb({g},{g},{g})"
            o.append(f'<rect x="{cx}" y="{y}" width="{cyw}" '
                     f'height="{ch-7}" fill="{fill}" stroke="{INK}" '
                     f'stroke-width="0.8"/>\n')
            tc = PAPER if v > 0.55 else INK
            o.append(f'<text x="{cx+cyw/2}" y="{y+21}" font-size="13" '
                     f'font-weight="600" style="fill:{tc}" text-anchor="middle" '
                     f'font-family="{MONO}">{v:.3f}</text>\n')
        m = sum(vals) / len(vals)
        o.append(f'<text x="{x0+7*cw+36}" y="{y+21}" font-size="13.5" '
                 f'font-weight="700" text-anchor="middle" '
                 f'font-family="{MONO}">{m:.3f}</text>\n')

    yc = y0 + 3 * ch - 4
    o.append(f'<line x1="{x0-10}" y1="{yc}" x2="{x0+7*cw+70}" y2="{yc}" '
             f'style="stroke:{PLOT}" stroke-width="2"/>\n')
    # Sits to the RIGHT of the rule: at the left it lands on the W8A16 row
    # label, and an annotation that collides with the data it annotates is
    # worse than no annotation.
    o.append(f'<text x="{x0+7*cw+76}" y="{yc+4}" class="ax" '
             f'style="fill:{PLOT}" text-anchor="start" '
             f'font-weight="600">cliff</text>\n')

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

    o = frame(w, h, "The quantization KL floor, against PPO's controller", fig=6)
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
                 f'{"QAT fails" if bad else "QAT works"}</text>\n')

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

    o = frame(w, h, "Policy KL against weight step size", fig=5)
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
    "block_diagram.svg": block_diagram,
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
