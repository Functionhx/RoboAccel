<img src="assets/header.svg" alt="RoboAccel. An RL policy, compiled to 13 integer instructions and run on silicon. An RTL simulation, a Zynq board built with Vivado 2020.2, and a Zynq board built with Vivado 2026.1 all return the actions 543, 790, 74, 635, 478, -796 for the same input, identical bit for bit." width="100%">

## General description

A policy is trained in floating point. RoboAccel compiles it into a precisely
specified integer program, proves the arithmetic survived deployment through
several independently written implementations, and executes it deterministically
on a Zynq-7000 `XC7Z010` accelerator and on an STM32H723 Cortex-M7 — then asks
whether the robot still walks, rather than whether tensor error is small.

The controller is a wheel-legged balancing robot: 25 observations, a 5-frame
history encoder, a 4-layer actor, 6 actions, 38,400 MAC. The whole network runs
from one register write, in 1,799 PL cycles, with no data-dependent branches.

<img src="assets/block_diagram.svg" alt="Figure 1. A trained RL policy is quantized and exported to 13 operator descriptors, executed by the PL accelerator, and closed back through the robot. An integer reference defines correctness for every stage after the policy." width="100%">

---

## Contents

| | | | |
| --- | --- | --- | --- |
| **1** [Specifications](#1-specifications) | **3** [Verification](#3-verification) | **5** [Getting started](#5-getting-started) | **7** [Documentation](#7-documentation) |
| **2** [Architecture](#2-architecture) | **4** [Quantization results](#4-quantization-results) | **6** [Repository map](#6-repository-map) | **8** [Licence](#8-licence-and-attribution) |

---

## 1. Specifications

Measured on hardware. Nothing in this section is projected, and both timing
boundaries are given because quoting only the flattering one would make the
speedup an argument rather than a measurement.

**Table 1. Timing**

| Parameter | RoboAccel FPGA | STM32H723 | Method |
| --- | ---: | ---: | --- |
| Pure inference (T1) | **17.99 µs** | **259.09 µs** | PL sequence cycle count, board-validated; DWT CYCCNT on silicon, 200 runs |
| — in cycles | 1,799 PL cycles @ 100 MHz | 124,363 core cycles @ 480 MHz | |
| obs→action (T3) | 46.75 – 46.79 µs | 260.82 µs | 5 × 100 board inferences on MINI_7010 with Vivado 2026.1; same STM32 run, T3 boundary |
| Cycle-to-cycle variation, PL | **none** | — | fixed 13-descriptor program, no data-dependent branches |
| Control period available | 10 ms (100 Hz) | 10 ms (100 Hz) | `control_dt` of the deployed task |

**14.4× faster at pure inference.** At the full obs→action boundary the gap
narrows to **5.6×**, because this Zynq part has no DMA: the PS pushes all 150
input words and pulls every output over AXI4-Lite, and that traffic — not the
network — is the remainder. Both boundaries are given because quoting only the
flattering one would make the speedup an argument rather than a measurement.
`T1`/`T3` are defined once in `tools/unified_timing.py` and applied identically
to both targets.

**Table 2. Endurance run**

A separate, earlier build — same device, Vivado **2020.2**, UART console — is
the longest continuous run on record. It is reported separately rather than
merged into Table 1, because it is a different toolchain and a different board
bring-up.

| Parameter | Result |
| --- | ---: |
| Inferences | 22 batches × 100 = **2,200** |
| Passed / failed | **2,200 / 0** |
| Overruns | **0** |
| Mean obs→action | 45.392 µs |
| Min / max | 45.334 / 46.101 µs |
| Output checksum | `0xBAC07F44`, identical on all 2,200 |

**Table 3. Resources and format**

| Parameter | Value |
| --- | --- |
| Device | `XC7Z010CLG400` on a MINI_7010 board, PL at 100 MHz |
| Logic | 10,919 LUT, 7,594 FF, 52 RAMB36, 72 DSP48E1 — 90% of the part's DSPs |
| Timing closure | WNS +0.400 ns, WHS +0.026 ns |
| Compute | 8×8 INT16 MAC array with INT48 accumulate, 64 DSP for GEMM and 8 for NORM |
| Numeric format | INT16 weights, Q8.8 activations and bias |
| Program | 13 operator descriptors: 7 GEMM, 5 ELU, 1 CONCAT |
| On-chip storage | vector cache 512×128-bit; weight cache 8 banks of 1280×128-bit |
| Boot path | JTAG only; results returned through a JTAG mailbox |

**Table 4. Capacity limits enforced by the exporter**

PL does not range-check dimensions, so the toolchain must.

| Limit | Constraint |
| --- | --- |
| Weight cache, per bank | `sum(ceil(K/8) × ceil(N/8)) ≤ 1280` |
| Vector layout | must fit 512 words |
| Program length | 1 … 32 descriptors |
| During an automatic sequence | the host must not touch the caches or instruction RAM |

---

## 2. Architecture

<img src="assets/architecture.svg" alt="Figure 2. Four stages: training, quantization, reference and verification, and two deployment backends sharing one arithmetic contract." width="100%">

The network topology is **not** hardcoded in RTL. The PS writes a program of
operator descriptors into a 32×128-bit instruction RAM, and a hardware sequencer
executes the whole network from a single register write. Deploying a different
policy is a re-export of cache images and a descriptor program, not an RTL
change.

> **Tested with two policies.** A release audit exported a second checkpoint
> with different weights and different per-layer scales; it runs correctly
> through the same RTL from nothing but a re-export. Getting there required
> fixing the end-to-end testbench, which had hardcoded one policy's descriptor
> program instead of reading the exported one — so it had never actually tested
> programmability. See [`docs/RELEASE_CANDIDATE.md`](docs/RELEASE_CANDIDATE.md) §4.

---

## 3. Verification

A neural controller that looks correct in PyTorch but differs numerically on the
target has not been validated — it has been spot-checked. RoboAccel runs the
same policy through several independently written arithmetic paths and requires
them to agree exactly.

<img src="assets/verification_chain.svg" alt="Figure 3. A NumPy integer reference arbitrates four independent implementations; two are verified on silicon." width="100%">

**Table 5. Implementations required to agree**

| Implementation | Result | Where |
| --- | --- | --- |
| NumPy integer reference | *the arbiter — defines correctness* | host |
| PyTorch fake-quant | 9/9 tensors exact, 4,000 samples | host |
| FPGA descriptor program | same 13 instructions, same 7 weight fracs | host + board |
| STM32 scalar C | **0 mismatches / 24 golden vectors** | **silicon** |
| STM32 SMLALD SIMD | **0 mismatches / 24 golden vectors** | **silicon** |

The FPGA exporter shares no code with the quantization library and still derives
the identical instruction program and the identical per-layer scales. Agreement
between them is evidence, not construction.

On the board, the endurance run of Table 2 completed **2,200 / 2,200 inferences
with 0 failures and 0 overruns**, every one returning the identical checksum
`0xBAC07F44`.

### The raw output behind the opening figure

Unedited, from the three logs:

```text
# RTL golden, make -C fpga/tb all
543 790 74 635 478 -796

# Board, Vivado 2020.2, UART console
Accelerator v1.2, mode=interrupt
Model/program loaded and instruction readback PASS: 132 vector words, 8 x 620 weight words, 13 instructions
Full cache readback PASS
Policy automatic-sequence polling self-test PASS
Policy automatic-sequence interrupt self-test PASS
Actions Q8.8: 543 790 74 635 478 -796

# Board, MINI_7010 + Vivado/Vitis 2026.1, JTAG mailbox
MAILBOX_ERRORS=0
MAILBOX_ACTIONS=543 790 74 635 478 -796
MAILBOX_STATUS=PASS
```

Sources: [`fpga/docs/09_hardware_test_20260821.md`](fpga/docs/09_hardware_test_20260821.md)
and [`fpga/docs/11_mini7010_vivado2026_port.md`](fpga/docs/11_mini7010_vivado2026_port.md).

---

## 4. Quantization results

<img src="assets/precision_cliff.svg" alt="Figure 4. Per-segment control success by precision: FP32, W16A16 and W8A16 all score 1.000; W8A8 and W4A8 collapse." width="100%">

Seven command segments, evaluated closed-loop. **The cliff is at 8-bit
activations, not 8-bit weights** — W8A16 is indistinguishable from FP32, W8A8
collapses.

**Table 6. Control success by precision**

| Precision | Success (mean) | Wheel support | Model bytes |
| --- | ---: | ---: | ---: |
| FP32 | **1.000** | **1.000** | — |
| **W16A16** *(deployed)* | **1.000** | **1.000** | 78,412 |
| W8A16 | **1.000** | **1.000** | 40,012 |
| W8A8 | 0.041 <sub>(0 – 0.164)</sub> | 0.936 <sub>(0.915 – 0.950)</sub> | 39,631 |
| W4A8 | 0.000 | 0.394 <sub>(0.239 – 0.480)</sub> | 20,431 |

*One training seed; the QAT-versus-PTQ comparison behind it is three-seed.*

> **Deployment baseline is W16A16.** W8A16 shows no measured control
> degradation, but the exporter is INT16/Q8.8 by construction and cannot emit
> INT8 weights, so shipping W8A16 would need a bit-width parameter in the export
> path first.

<details>
<summary><b>Why quantization-aware training fails at W8A8</b> — a real optimizer pathology, and the intervention that showed it is not the cause</summary>

<br>

<img src="assets/perturbation_response.svg" alt="Figure 5. Policy KL against weight perturbation magnitude for float, W16A16, W8A16 and W8A8." width="100%">

**The pathology.** Fake-quantization makes the policy a **discontinuous**
function of its weights: a step below one quantization LSB changes nothing, a
step across an LSB boundary snaps that weight by a whole LSB. PPO regulates its
learning rate by measured policy KL and reads that discontinuity as divergence.
One 1e-5 weight step — the algorithm's own floor — produces a policy KL of
**6.73e-02** under W8A8 against **4.06e-07** in float, 6.3× the controller's
threshold *at zero effective learning rate*.

| | W8A8 QAT | FP32 control |
| --- | ---: | ---: |
| Learning rate pinned at its 1e-5 floor | **2000 / 2000 iterations** | 10 / 2000 |
| Median policy KL | 0.0650 | 0.0054 |
| Reward at convergence | 28.64 | 29.30 |

The reward curve stays healthy the whole time. Only the learning-rate trace
shows it.

**The falsification.** Re-running W8A8 with the learning rate held fixed at the
FP32 control's median — 0% of iterations at the floor, verified in the optimizer
— **did not restore turning, and made everything else worse**:

| arm | turn_L | turn_R | stop | height | **mean success** |
| --- | ---: | ---: | ---: | ---: | ---: |
| PTQ W8A8 | 0.758 | 0.523 | 0.984 | 0.539 | 0.401 |
| QAT W8A8 | 0.000 | 0.000 | 0.938 | 1.000 | 0.501 |
| **QAT W8A8, fixed LR** | 0.000 | 0.000 | 0.023 | 0.000 | **0.206** |
| **QAT W8A16** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Restoring the step size without any behaviour-preservation term let the policy
drift *further* from the reference — the learning-rate floor had been an
accidental brake. W8A8 QAT also scores 0.143 in *float* execution: it requires
the quantizer it was trained with. W8A16 scores 1.000 in both. One learned to
depend on the quantizer; the other learned to tolerate it.

**A mechanism that explains a phenomenon is not the mechanism that produces it.**
The learning-rate collapse is real, reproducible and predictable, and it is a
co-symptom rather than the cause. Only the intervention separated them.

Full record: [`docs/qat_failure/`](docs/qat_failure/) ·
[`docs/QAT_RESULTS.md`](docs/QAT_RESULTS.md)

</details>

---

## 5. Getting started

Everything downstream of a trained checkpoint runs here, with no board and no
GPU. Start by proving the arithmetic:

```bash
git clone https://github.com/Functionhx/RoboAccel.git && cd RoboAccel
source training/scripts/env_solid.sh

# The Python reference vs the actual RTL and the actual Cortex-M7 kernel.
python quantization/scripts/verify_against_rtl.py     # -> VERIFY_AGAINST_RTL: PASS

# Five RTL testbenches that need no exported model. No board required.
make -C fpga/tb primitives gemm concat sequencer top
```

`verify_against_rtl.py` parses `fpga/rtl/*.sv` and `stm32/src/ra_kernel.c` and
checks the Python reference against both — the ELU ROM entry by entry and the
requantizer shift by shift.

The three remaining testbench targets — `policy`, `stream`, `v3` — replay a real
exported model against the Python golden, so they need step 3 below first. Cache
images are derived from a checkpoint and are deliberately not committed.

<details>
<summary><b>Full deployment workflow</b> — calibration, export, both backends</summary>

<br>

```bash
# 1. bit-exactness of a specific checkpoint
python quantization/scripts/bitexact_check.py --checkpoint <ckpt> --n 4000 \
       --weight-bits 16 --act-bits 16

# 2. calibrate activation scales from on-policy rollouts   [needs the RL env]
python training/scripts/dump_calib_obs.py --checkpoint <ckpt> --out calib.npz \
       --steps 400 --headless --num_envs 256
python quantization/scripts/calibrate_act_fracs.py --checkpoint <ckpt> \
       --obs calib.npz --act-bits 8 --out fracs.json

# 3. export to ONNX, then to a descriptor program + cache images
python quantization/scripts/export_to_onnx.py <ckpt> --out policy.onnx
python fpga/tools/export_policy.py --model policy.onnx --out fpga/generated \
       --samples 1000 --seed 7

# ...which unlocks the three model-dependent testbenches
make -C fpga/tb policy stream v3

# 4. STM32 codegen + host equivalence               (no board required)
python quantization/scripts/gen_h7_branched.py policy.onnx --name mymodel \
       --out stm32/models/mymodel --golden 24 --golden-npz calib.npz
bash quantization/scripts/hosttest/build_and_run.sh stm32/models/mymodel

# 5. flash + benchmark                                  (board required)
cd stm32 && bash scripts/flash.sh mymodel && python scripts/read_results.py mymodel
```

Step 2 is the only one that needs the RL environment. It is an **external
dependency and is not redistributable** — the upstream repository carries no
licence, so this repo ships only the adapter that imports it. Set
`ROBOACCEL_RL_ENV` and `ROBOACCEL_PYTHON` to your own checkout. See
[`NOTICE.md`](NOTICE.md).

</details>

<details>
<summary><b>Tested configuration</b> — what has actually been run</summary>

<br>

| | |
| --- | --- |
| FPGA board | MINI_7010 — Zynq-7000 `XC7Z010CLG400`, PL 100 MHz |
| FPGA toolchain | AMD Vivado / Vitis 2026.1, Linux, batch mode |
| MCU | STM32H723VGT6 @ 480 MHz, weights and activations in DTCM |
| MCU toolchain | `arm-none-eabi-gcc`, CMake + Ninja |
| RTL simulation | Icarus Verilog, 8 testbench targets — 5 standalone, 3 replay an exported model |

</details>

---

## 6. Repository map

| Directory | What lives there | Go here for |
| --- | --- | --- |
| `quantization/roboaccel_quant/` | Fixed-point reference (the arbiter), torch fake-quant with STE, quantized policy module | the arithmetic contract |
| `quantization/scripts/` | Calibration, bit-exactness checks, diagnostics, ONNX export, host MCU equivalence | **verification**, reproducing a result |
| `fpga/rtl/` `fpga/tb/` | Accelerator RTL and 8 Icarus testbenches | **RTL**, the sequencer, the MAC array |
| `fpga/tools/` | ONNX → descriptor program + cache images | adding a new policy to the FPGA |
| `stm32/src/` `stm32/scripts/` | Scalar and SMLALD INT16 kernels, DTCM placement, DWT benchmarking, flashing | **Cortex-M7 kernels** |
| `training/scripts/` | Adapter that imports the external RL environment; QAT drivers | training and QAT experiments |
| `docs/qat_failure/` | The complete W8A8 research record | **the QAT research** |
| `docs/results/` | Raw JSON behind every figure and table | **checking our numbers** |

---

## 7. Documentation

This repository is deliberately strict about what counts as a result. Every
headline claim is traced and graded in [`docs/EVIDENCE.md`](docs/EVIDENCE.md) —
measured, reproduced, estimated, qualitative, or hypothesis — and retractions
are listed there too. Figures carrying measurements are **generated** from
`docs/results/*.json` by [`assets/make_figures.py`](assets/make_figures.py), so
a figure cannot drift from the experiment it illustrates.

| Document | Contents |
| --- | --- |
| [`docs/EVIDENCE.md`](docs/EVIDENCE.md) | Every claim traced to a source and graded, plus retractions |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Canonical artifacts, the interface between each pair of components, known fragile boundaries |
| [`docs/RELEASE_CANDIDATE.md`](docs/RELEASE_CANDIDATE.md) | Full chain re-run from a clean clone, what could not be run, and the one open defect it found |
| [`docs/QAT_RESULTS.md`](docs/QAT_RESULTS.md) | Full quantization results with evidence grading |
| [`docs/qat_failure/`](docs/qat_failure/) | The W8A8 research record: mechanism, hypothesis ledger, experiment log |
| [`docs/SOLID_POLICY_DEPLOYMENT_MAPPING.md`](docs/SOLID_POLICY_DEPLOYMENT_MAPPING.md) | Layer-by-layer policy → accelerator mapping |
| [`docs/knowledge_transfer/`](docs/knowledge_transfer/) | 13 documents: debugging history, design decisions, failed hypotheses, operations playbook |

<details>
<summary><b>Research record</b> — hypotheses we tested and falsified</summary>

<br>

Kept on purpose. A hypothesis that survived only because nobody tried to kill it
is not a finding, and the falsifications here were more useful than the
confirmations.

- **"QAT recovers 86–101%"** — held only on a task where the robot rested on its
  chassis instead of balancing. A corrected simulation reduced it to
  straight-line motion only.
- **"QAT beats FP32"** — true on numerical MSE (0.1334 vs 0.2898). The matched
  control reached 0.0657. The gain was extra training, not quantization
  awareness. **MSE is not evidence of control quality.**
- **"The latent is the quantization bottleneck"** — layer ablation put it at
  13%; leave-one-out then refuted any single-layer explanation.
- **"Saturation causes the collapse"** — measured activation clipping is 0.0000%.
- **"The requantizer's rounding bias causes the turning failure"** — retraining
  against ideal round-half-away-from-zero left turning at exactly 0.000.
- **"The yaw reward is starved, so rebalance it"** — quadrupling it scaled the
  reward contribution exactly 4.00× and moved yaw RMSE by 0.8%. The optimizer
  was pinned at its learning-rate floor the whole time.
- **"The PL has no element-wise ADD"** — false on ISA v3: `OP_AFFINE` with
  a=1, shift=0 *is* ADD. Still true through the current exporter, which is a
  toolchain limit, not a hardware one.

Sources: [`docs/knowledge_transfer/06_failed_hypotheses.md`](docs/knowledge_transfer/06_failed_hypotheses.md)
· [`docs/qat_failure/goal4_hypotheses.md`](docs/qat_failure/goal4_hypotheses.md)

</details>

Knowledge-transfer documents are written in Chinese, with English technical
terminology, identifiers, paths and register names preserved verbatim.

---

## 8. Licence and attribution

**MIT.** The accelerator derives from `rl_on_fpga` (MIT, © 2026 DreamChaser);
that copyright is preserved in [`LICENSE`](LICENSE).

The reinforcement-learning environment used to train the policies is **not
included and not redistributable** — the upstream repository carries no licence.
This repository contains only the adapter that imports it by `PYTHONPATH`.
Everything downstream of a trained checkpoint is fully self-contained. See
[`NOTICE.md`](NOTICE.md).
