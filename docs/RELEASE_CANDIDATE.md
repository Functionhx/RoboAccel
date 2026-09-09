# Release Candidate Audit

Goal 3 §15. Nothing here is asserted from memory: every row was produced by
running the command shown, on a **fresh `git clone` of the published
repository**, on 2026-09-09 at commit `8d1d1b9`.

The point of this audit is not to demonstrate that the chain works. It is to
find out where it does not. It found one open defect, recorded in §4.

---

## 1. What ran, and what it produced

Clean clone, no board, no GPU, no RL environment.

| # | Step | Command | Result |
|---|---|---|---|
| 0 | Clone | `git clone … && cd RoboAccel` | 145 files, 3.3 MB |
| 1 | Arithmetic contract | `quantization/scripts/verify_against_rtl.py` | **`VERIFY_AGAINST_RTL: PASS`** — ELU ROM entry by entry, requantizer shift by shift, against `fpga/rtl/*.sv` and `stm32/src/ra_kernel.c` |
| 2 | ONNX export | `quantization/scripts/export_to_onnx.py <ckpt>` | max abs diff **2.384e-06** → MATCH |
| 3 | Bit-exactness | `quantization/scripts/bitexact_check.py --n 4000 --weight-bits 16 --act-bits 16` | **9/9 tensors 100.0000% exact, 0.0000 max LSB** → `BIT_EXACT: PASS` |
| 4 | Four-way cross-validation | `quantization/scripts/verify_export.py --onnx … --n 500` | torch ↔ numpy ↔ exporter all **100.0000% exact** → `CROSS_VALIDATION: PASS (all bit-identical)` |
| 5 | FPGA export | `fpga/tools/export_policy.py --model … --out fpga/generated` | **13 descriptors**, weight fracs **[11,12,14,13,14,14,14]**, 620/1280 weight words per bank, 132/512 vector words |
| 6 | RTL regression | `make -C fpga/tb primitives gemm concat sequencer top policy` | **6/6 PASS**, end-to-end sequence **1,799 cycles** |
| 7 | STM32 codegen | `quantization/scripts/gen_h7_branched.py … --golden 24` | **38,400 MAC**, 78,418 B padded weights (encoder 24,384 MAC, actor 14,016 MAC) |
| 8 | STM32 host equivalence | `quantization/scripts/hosttest/build_and_run.sh` | ELU ROM **0/256** entries differ from the RTL; **24 vectors, 0 SIMD + 0 scalar mismatches** → `H7_HOST_CHECK: PASS` |

Steps 1–8 independently reproduce, from source, the claims the README makes
about 38,400 MAC, 13 descriptors, 7 weight fractional bits, 1,799 PL cycles,
9/9 bit-exact tensors, and 0/24 golden-vector mismatches.

## 2. What could not be run here, and why

| Step | Blocker | Consequence |
|---|---|---|
| Training | The RL environment carries **no upstream licence** and is not vendored (`NOTICE.md`) | Reproducing a checkpoint requires obtaining it yourself and setting `ROBOACCEL_RL_ENV` |
| Activation calibration | Same — needs on-policy rollouts | The A8 calibration path cannot run from a clean clone alone |
| FPGA physical benchmark | Needs the MINI_7010 board, Vivado 2026.1 and a JTAG connection | The 46.75–46.79 µs obs→action figure is quoted from a prior board run, not re-measured here |
| STM32 physical benchmark | Needs an STM32H723 and the ARM toolchain (`ROBOACCEL_STM32_ENV`) | The 259.09 µs figure is quoted from a prior on-silicon run, not re-measured here |
| Vivado synthesis | No `DISPLAY`, batch only; needs the 2026.1 install | Utilization figures are quoted from prior reports |

Everything in §1 is reproducible with Python, Icarus Verilog, and a C compiler.
Everything in this table needs hardware or an unredistributable dependency, and
is labelled as such wherever it is quoted.

## 3. Defects this audit found and fixed

All three meant the published repository **did not run at all** for anyone but
the author. They were found by cloning the published repo rather than testing
the working tree.

1. **Every script imported `hwq`.** The package had been renamed
   `roboaccel_quant` during assembly; no import was updated. The first Quick
   Start command failed with `ModuleNotFoundError`. Fixed in 16 files.
2. **Ten files hardcoded `/home/as/...`**, including `env_solid.sh` itself,
   which exported a `QAT_ROOT` pointing outside the repository. Repo paths are
   now derived from the script's own location; the two genuinely external
   dependencies became named variables that fail loudly.
3. **Three referenced files were never vendored** — `gen_h7_branched.py`
   (named in the Quick Start), `fixed_policy.py` and `analyze_policy.py`
   (needed by `verify_export.py`). `verify_export.py` also still pointed at
   `<repo>/tools`, which is `fpga/tools` after the restructure.

Also: `make -C fpga/tb all` **cannot** pass on a clean clone. Three of the eight
targets replay an exported model whose cache images are derived from a
checkpoint and are correctly not committed. The Quick Start now names the five
that do pass, and points at step 3 for the rest.

## 4. Defect found, diagnosed, and fixed — in the testbench, not the RTL

**Status: resolved.** This section previously concluded the opposite. That
conclusion was wrong, and how it was wrong is worth keeping.

### What was observed

Exporting a second checkpoint (`SOLID_FP32_V2`) produced a descriptor program
structurally identical to the deployed one — same 13 opcodes, bases, dims and
cache layout — differing only in three `shift` fields, which are the per-layer
weight fractional bits:

```
deployed        shifts [11, 12, 14, 13, 14, 14, 14]   -> PASS
SOLID_FP32_V2   shifts [13, 14, 14, 14, 14, 14, 14]   -> FAIL, all 6 actions
```

All three software implementations agreed bit-identically over 500 samples, so
the divergence was attributed to the RTL.

### The wrong step

"A testbench regression" was listed as ruled out, on the evidence that the
original model passes. **That test was insufficient.** The original model
passes for a reason that does not generalise, and only a test that varies the
program could have shown it.

### How it was actually found

Dumping the vector cache after the sequence gives every layer's output, since
each descriptor writes to its own address. Comparing them against the software
reference put the first divergence at **descriptor 0**, the very first GEMM.
Sweeping the assumed shift over the software model then reproduced the RTL
output exactly:

```
assumed shift   exact matches / 128
           11                  128   <-- MATCH
           13                    0   <-- what the descriptor requested
```

The RTL applied shift **11**, bit-exactly on all 128 lanes, when the descriptor
asked for **13**. Eleven is the *deployed* model's shift for that layer.

### Root cause

`fpga/tb/tb_policy_e2e.sv` **hardcoded all 13 descriptors**, including one
policy's shifts, and never read the exported `instruction_program.hex`. It
loaded the exported *weights and inputs* and ran them against a *fixed*
program. Any model whose weight fractional bits differ from the deployed one's
therefore failed, and the deployed one passed because its shifts happened to
match the hardcoded constants.

### Fix and verification

The testbench now reads the exported program and writes it through the same
host interface the PS uses:

```systemverilog
$readmemh("../generated/instruction_program.hex", program_words);
for (group = 0; group < 13; group = group + 1)
    instruction_write(group[4:0], program_words[group]);
```

| Model | Before | After |
|---|---|---|
| `fudan_policy` (deployed) | PASS, 1,799 cycles | **PASS, 1,799 cycles** |
| `SOLID_FP32_V2` | FAIL, 6/6 actions | **PASS, 1,799 cycles** |

**No RTL was changed.** The accelerator applied exactly the shift it was given.

### What this means for the claims

The programmability claim is **strengthened, not weakened**: a second policy,
with different weights and different per-layer scales, now runs correctly
through the same RTL from nothing but a re-export. That is the architecture
working as designed, and it is now actually tested.

It also removes a blind spot. The testbench could only ever validate one model,
so "the RTL executes the exported program" had never been checked — the
regression would have passed forever while silently ignoring the program under
test.

## 5. Verdict

**Ready to tag, with the scope stated.**

The deployed policy is verified end to end, reproducibly, from a clean clone.
A second policy with different weights and different per-layer scales also
passes the full RTL regression after the testbench was corrected, which is the
first real evidence for the central architectural claim.

Still outside what this audit can cover, and labelled as such wherever quoted:
the two physical benchmarks and the Vivado utilization figures need hardware,
and training needs an environment that cannot be redistributed.
