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

## 4. Open defect — RTL disagrees with software on a second checkpoint

**Status: open. Reproducible. Not fixed, and deliberately not worked around.**

### What was observed

Exporting a **second** checkpoint (`SOLID_FP32_V2`, sha256 `ab20c190…`) with
the same topology produces a descriptor program that is **structurally
identical** to the deployed one — same 13 opcodes, same source, destination,
aux and weight bases, same `dim_k`/`dim_n`, same cache layout (620 weight words
per bank, 132 vector words). The only differences are three `shift` fields,
which are simply the per-layer weight fractional bits:

```
deployed        shifts [11, 12, 14, 13, 14, 14, 14]   -> tb_policy_e2e PASS
SOLID_FP32_V2   shifts [13, 14, 14, 14, 14, 14, 14]   -> tb_policy_e2e FAIL
```

All six actions differ, e.g. `FAIL action 0 got 1733 expected -623`. The
sequence still completes in exactly 1,799 cycles, and no `$readmemh` file is
missing.

### Where the disagreement is

**In the RTL, not in the exporter or the reference.** On the same checkpoint,
all three software implementations are bit-identical over 500 samples:

```
A torch  vs B numpy       100.0000 %   0.0000 max LSB
B numpy  vs C exporter    100.0000 %   0.0000 max LSB
A torch  vs C exporter    100.0000 %   0.0000 max LSB
CROSS_VALIDATION: PASS (all bit-identical)
```

### What has been ruled out

| Candidate | Test | Result |
|---|---|---|
| A stale or broken vendored exporter | Export the **original** model with the **published** exporter and run the same TB | **PASS**, 1,799 cycles, identical selftest vector — the toolchain is sound |
| A testbench regression (a `cmd_subop` dangling-port warning is present) | Same test | **PASS** — the TB is not the problem |
| Large shift values | Re-export with `MAX_WEIGHT_FRAC` capped to 13, then 12 | **Still fails** at both — not the shift magnitude |
| Missing or truncated cache images | Checked all 24 bank files plus `vector_cache.hex` load | All present, no `$readmemh` error |
| Capacity overrun | 620/1280 weight words, 132/512 vector words, 13/32 instructions | Identical to the passing model |
| Activation saturation | Compared per-layer ranges | The **original** model saturates Q8.8 exactly (`encoder.2` reaches ±128.000) and passes; `SOLID_FP32_V2` peaks at 30.96 and fails — the opposite of the saturation hypothesis |

### Minimal reproducer

```bash
python quantization/scripts/export_to_onnx.py <SOLID_FP32_V2.pt> --out policy.onnx
python fpga/tools/export_policy.py --model policy.onnx --out fpga/generated \
       --samples 1000 --seed 7
make -C fpga/tb policy        # FAIL, 6 action mismatches
```

### Consequence for the project's claims

The README previously stated that deploying a different policy means
re-exporting cache images and a descriptor program rather than rewriting RTL.
**That is the design intent and it is what the architecture is built for, but
this audit shows it is not yet demonstrated for a second set of weights.** The
claim has been softened accordingly, and this defect is now the top item in
`docs/ARCHITECTURE.md` §7.

No RTL was changed. The evidence localizes the disagreement but does not yet
identify the mechanism, and changing hardware on an unidentified mechanism
would be guessing.

## 5. Verdict

**Not ready to tag a release.**

The deployed policy is fully verified end to end, reproducibly, from a clean
clone — that part is solid and is what the README claims. But the central
architectural promise, that the accelerator is programmed rather than
hardwired, is contradicted by §4 for the one case that tests it.

Tagging `v0.1.0` should wait until either the §4 mechanism is identified, or
the claim is restated to describe exactly the configuration that is verified.
