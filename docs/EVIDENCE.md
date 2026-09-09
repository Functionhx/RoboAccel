# README Evidence Audit

Goal 3 §9. Every claim the README makes is listed here with its source and a
classification. The rule this file enforces:

> **Never present an estimate as a physical measurement.
> Never present a single-seed observation as a universal conclusion.**

**Classification**

| Class | Meaning |
|---|---|
| **measured** | Read off an instrument or a raw log on real hardware, or produced by a deterministic tool and preserved in a file |
| **reproduced** | Measured, and independently re-obtained at least once (different seed, different implementation, or a re-run) |
| **estimated** | Derived from a model, not observed |
| **qualitative** | A design statement with no number attached |
| **hypothesis** | Stated as a hypothesis in the README, and labelled as one there |

---

## Hardware performance

| # | Claim | Class | Source |
|---|---|---|---|
| H1 | STM32H723 pure inference **259.09 µs / 124,363 cycles** | **measured** | DWT CYCCNT on silicon, 200 runs, JTAG mailbox read-back |
| H2 | STM32 obs→action **260.82 µs** | **measured** | same run, T3 boundary |
| H3 | **3.25 cycles/MAC** | **measured** | derived arithmetic from H1 and the 38,400-MAC model |
| H4 | RoboAccel pure inference **17.99 µs / 1,799 PL cycles** | **measured** | PL sequence cycle count, board-validated |
| H5 | **14.4× speedup** | **measured** | ratio of H4 to H1, both measured at the same T1 boundary |
| H6 | 72 DSP48E1 (64 GEMM + 8 NORM) | **measured** | Vivado utilization report |

**Boundary discipline.** H5 is a ratio of two numbers taken at the *same*
defined boundary (T1, "pure inference": integer arithmetic only, operands
already in the engine's local memory). `tools/unified_timing.py` defines T1/T2/T3
and applies them identically to both targets. Quoting a speedup across
different boundaries would be an argument, not a measurement.

---

## Numerical verification

| # | Claim | Class | Source |
|---|---|---|---|
| N1 | PyTorch fake-quant matches the integer reference, **9/9 tensors, 4000 samples** | **measured** | `scripts/bitexact_check.py` |
| N2 | STM32 scalar C: **0 mismatches, 24 golden vectors, on silicon** | **measured** | JTAG mailbox, `MAILBOX_ERRORS=0` |
| N3 | STM32 SMLALD SIMD: **0 mismatches, on silicon** | **measured** | same |
| N4 | FPGA exporter independently derives the identical 13-instruction program and identical 7 weight fracs | **measured** | `generated/policy_map.json` vs the quantization library's frac table, compared field by field |
| N5 | Five implementations agree bit-exactly | **reproduced** | N1–N4 are four independent derivations of the same arithmetic; the fifth (NumPy integer reference) is the arbiter they are compared against |
| N6 | float64 GPU emulation is exact (max partial sum 1.34e11 ≪ 2⁵³) | **measured** | worst-case accumulator magnitude over the calibration set, checked against the float64 mantissa limit |

---

## Control quality

| # | Claim | Class | Source |
|---|---|---|---|
| C1 | FP32 / W16A16 / W8A16 all score **1.000** across seven segments | **measured** | `docs/results/ladder_pooled.json` — `seeds: 1`. Single training seed |
| C2 | W8A8 scores **0.041 mean, 0.000–0.164 across segments** | **measured** | same file, single seed |
| C3 | W4A8 scores **0.000**; wheel support **0.394 [0.239, 0.480]** | **measured** | same file, single seed |
| C4 | "The quantization cliff is exactly at 8-bit activations" | **measured** | follows directly from C1 vs C2: W8A16 (8-bit weights, 16-bit activations) is undamaged; W8A8 collapses |
| C5 | Model bytes 78,412 / 40,012 / 39,631 / 20,431 | **measured** | exporter output sizes |

**Seed discipline — corrected.** C1–C3 are **single-seed**: `ladder_pooled.json`
reports `seeds: 1`. An earlier revision of this table and of the README called
the precision ladder three-seed; that was a conflation with a *different*
experiment — the QAT-versus-PTQ comparison in `QAT_RESULTS.md` §31/§32, which
genuinely is three-seed (R1–R3 below). The ladder's W8A8 collapse is
corroborated by that three-seed result (turning 0.000 in every seed), but the
ladder itself is one seed and is now labelled as such wherever it appears.
A second correction: W4A8 wheel support was previously quoted as **0.239**,
which is the minimum across segments, not the value; the mean is **0.394**.
Where a number is single-seed it is said so at the point of use. `peak_roll_rad` is deliberately **not** quoted
anywhere: it is a max over 768 episodes and swings 75 points across seeds — an
earlier claim built on it was retracted. `mean_episode_peak_roll_rad` is the
seed-stable statistic and is the one used.

---

## Research findings (W8A8 / QAT)

| # | Claim | Class | Source |
|---|---|---|---|
| R1 | QAT at W8A8 restores wheel support (**95–100%**, three seeds) | **reproduced** | multi-seed table |
| R2 | QAT partially restores straight locomotion (**39–84%, seed-dependent**) | **measured**, explicitly seed-dependent | same |
| R3 | QAT **never** restores turning (0.000 across all seeds) | **reproduced** | same |
| R4 | QAT parameters score **0.000 in floating-point execution** — they require the quantizer | **measured** | `docs/01_localization.md`; same evaluator, same seed, same 128 envs |
| R5 | W8A8 QAT held the learning rate at its **1e-5 floor for 2000/2000 iterations**; the FP32 control, 10/2000 | **measured** | `docs/results/goal4_results.json`, from each run's own `metrics.jsonl` |
| R6 | One 1e-5 weight step produces policy KL **6.73e-02** under W8A8 fake-quant vs **4.06e-07** in float — **165,737×** | **measured** | `docs/results/kl_amplification_qat.json`; reproduced on a second checkpoint (`kl_amplification_fp32.json`) |
| R7 | W8A8 is the only precision on the ladder whose quantization KL floor exceeds the controller threshold, and the only one where QAT fails | **measured** | R6 table vs C1–C3 |
| R8 | Raising the yaw reward 2× and 4× changes `yaw_rmse` by 0.8%, i.e. not at all | **measured** | `docs/results/goal4_results.json`; reward contribution verified to scale exactly 2.00×/4.00× |
| R9 | "Preserve a reference policy, learn only the correction" would avoid this | **hypothesis** | labelled as a hypothesis; untested |

**R7 is the strongest claim in the README and deserves its caveat**: it is a
correspondence across five precisions on one model and one task. It predicts
correctly here; it has not been tested on another network.

---

## Estimated — never presented as measured

| # | Claim | Why it is an estimate |
|---|---|---|
| E1 | ASIC logic area / Fmax from the sky130 sweep | Post-synthesis logic only, from `yosys stat -liberty`. **Excludes memory**, which is reported separately as bits rather than converted to area with an invented SRAM figure |
| E2 | DDR-streaming latency for the original (non-compact) policies | Cycle model, not a board run — labelled "DDR stream" in `tools/unified_timing.py` |
| E3 | Any "cycles/MAC" figure for a configuration not built | Derived from the array geometry |

---

## Retracted claims

Kept here on purpose. See the README's *"What this project got wrong"*.

| Retracted | Why |
|---|---|
| "QAT beats FP32" (MSE 0.1334 vs 0.2898) | The gain was extra training, not quantization awareness. Control metrics reached 0.0657. **Numerical MSE is not evidence of control quality.** |
| "QAT is worse than PTQ at mass +2" | Built on `peak_roll_rad`, a max over 768 episodes that swings 75 points across seeds |
| "Ship at W8A16" | The exporter is INT16/Q8.8 by construction and **cannot emit INT8 weights**. Not executable without a bit-width parameter in the export path |
| "The latent bottleneck causes the W8A8 collapse" | `enc2` contributes 13.2% of the error |
| "Saturation causes the W8A8 collapse" | Measured clipping is 0.0000% |
| "Rounding bias causes the turning failure" | Ideal round-half-away-from-zero leaves turning at 0.000 |
| "The PL has no element-wise ADD" | False on ISA v3: `OP_AFFINE` with a=1, shift=0 **is** ADD. Still true through the current exporter |

---

## Reproduced from a clean clone (2026-09-09, commit `8d1d1b9`)

Every row of the verification table above was re-run against a fresh
`git clone` of the published repository, not the working tree. See
[`RELEASE_CANDIDATE.md`](RELEASE_CANDIDATE.md) for the commands and output.
`BIT_EXACT: PASS` (9/9 tensors, 4000 samples), `CROSS_VALIDATION: PASS` (three
implementations bit-identical), `tb_policy_e2e` PASS at 1,799 cycles,
`H7_HOST_CHECK: PASS` (0/24 mismatches, ELU ROM 0/256 differing), and the STM32
codegen reports 38,400 MAC.

## Claim upgraded by the release audit

| Claim | Status |
|---|---|
| "Deploying a different policy is a re-export, not an RTL change" | **Measured, two policies.** A second checkpoint with different weights and different per-layer weight fractional bits passes the full RTL regression at 1,799 cycles after a testbench fix. The testbench had hardcoded one policy's descriptor program and never read the exported one, so programmability had never been tested. No RTL changed. `RELEASE_CANDIDATE.md` §4 |

## Provenance boundaries

* `fudan_policy.onnx` is a **historical filename, not upstream Fudan weights**.
  Its SHA-256 matches none of the six pinned external reference policies.
* The RL environment this project trains against carries **no upstream licence**
  and is therefore **not vendored** — see `NOTICE.md`. The repository ships the
  adapter that imports it by `PYTHONPATH`, and nothing downstream of a trained
  checkpoint depends on it.
* The accelerator is MIT, © 2026 DreamChaser, attribution preserved.
