# goal.md §37 — definition of DONE

Status against Codex's **final** frozen baseline `SOLID_FP32_V2`
(`locomotion_v2` seed 1, sha256 `ab20c190`, 145 infrastructure files hashed).
Everything previously done on `robust_v1` was redone here, because that task
admits a chassis-support exploit and its absolute numbers are not statements
about locomotion.

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Codex solid RL baseline identified and frozen | ✅ | `QAT_HANDOFF_AUDIT.md`, `artifacts/v2/frozen/`, sha256 `ab20c190` |
| 2 | Solid FP32 checkpoint evaluated | ✅ | 1.000 success / 1.000 support, all 7 segments |
| 3 | FP32+ equal-training control | ✅ | `solid_fp32_plus_v2_long`, matched 2000 iters |
| 4 | PTQ precision matrix | ✅ | W16A16 / W8A16 / W8A8 / W4A8, `artifacts/v2/ladder_*` |
| 5 | Actual fixed-point semantics verified | ✅ | `verify_against_rtl.py` PASS, ELU ROM 0/256 |
| 6 | Hardware-aware fake quant on the solid policy | ✅ | `hwq/`, quantization asserted at construction |
| 7 | Closed-loop on-policy QAT fine-tuning | ✅ | 500 and 2000 iteration runs |
| 8 | PTQ vs QAT control comparison | ✅ | matched 2000 iters, three arms |
| 9 | QAT recovery quantified | ✅ | straight-line recovered; turning does not |
| 10 | Quantization sensitivity diagnostics | ✅ | `layer_sensitivity_w8a8.json`, one-at-a-time + leave-one-out |
| 11 | QAT mechanism diagnostics | ✅ | three hypotheses tested, two refuted |
| 12 | Robustness FP32/PTQ/QAT comparison | ✅ | 7 command segments incl. combined |
| 13 | Exact integer reference | ✅ | `hwq/fixed_ref.py` |
| 14 | Fake-quant vs integer large-scale validation | ✅ | 9/9 tensors exact, W16A16 **and** W8A8 |
| 15 | Golden trajectory dataset | ✅ | 7/7 regimes, `artifacts/v2/deploy/golden/` |
| 16 | Deployment export | ✅ | ONNX 2.4e-06, FPGA 13 instr / 620 words |
| 17 | H7 numerical verification | ✅ | host 0/0, silicon scalar+SIMD PASS |
| 18 | H7 measured benchmark | ✅ | 124,363 cyc = 259.09 µs, 200 runs, DWT |
| 19 | FPGA numerical verification | ✅ | exporter agrees on program and fracs |
| 20 | FPGA measured benchmark | ✅ | 1,799 PL cycles = 17.99 µs |
| 21 | Cross-platform consistency | ✅ | 5 implementations bit-identical |
| 22 | Reproducibility instructions | ✅ | `REPRODUCE_QAT.md`, `RERUN_PLAN.md` |
| 23 | Final QAT/control table | ✅ | §31/§32 in `QAT_RESULTS.md`, `artifacts/v2/final_tables.txt` |
| 24 | Final hardware table | ✅ | §30 in `QAT_RESULTS.md` |
| — | Multi-seed validation (§32) | ✅ | three seeds, all arms, 2000 iters each |
| — | QAT-from-scratch (§33, optional) | ⏸ | run on `robust_v1` to 474 iters, stopped when that task was superseded |

## What the re-run changed

Three claims did not survive contact with a corrected environment, and are
recorded as retractions rather than quietly dropped:

1. **"QAT recovers 86–101%."** True on `robust_v1`, but that robot slid on its
   chassis and every metric was a soft degradation with no pass/fail floor. On
   `locomotion_v2` QAT recovers straight-line locomotion and wheel support, and
   makes turning *worse than doing nothing* (0.18 → 0.00, 0.41 → 0.00).
2. **"The latent is the bottleneck."** Layer ablation puts `enc2` at 13% of the
   error, not the majority; leave-one-out then refutes any single-layer story.
3. **"The requantizer's rounding bias causes the turning failure."** Retraining
   against a corrected requantizer left turning at exactly 0.000. The bias is
   real (~4 rad/s differential, and fixing it recovers `reverse` 0.57 → 0.99)
   but it is not the cause.

## What survived, and is the deliverable

* The **cliff is exactly at 8-bit activations**. INT8 weights are free:
  W8A16 halves model storage at 1.000 success and 1.000 support, identical to
  FP32. W8A8 costs the robot its legs; W4A8 leaves `support_fraction` at 0.239.
* **Ship `SOLID_FP32_V2` at W16A16, or W8A16 for half the weight storage.**
  QAT is not required for this board.
* **QAT's value is enabling an 8-bit-activation part**, with the documented
  limit that turning does not recover at any budget tested.
* **Five implementations agree bit-exactly** — torch fake-quant, numpy integer
  reference, FPGA descriptor program, H7 scalar C and H7 SMLALD — with the last
  two verified on the physical part.
* **RoboAccel 17.99 µs vs H7 259.09 µs = 14.4×**, both measured.
