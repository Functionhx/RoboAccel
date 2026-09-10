# From checkpoint to deployment

[← Back to RoboAccel](../README.md)

Start with the [board-free quickstart](../README.md#quickstart). It uses the
included `stm32/models/solid_v2` model and needs only NumPy and a host C compiler.
This guide covers exporting your own checkpoint, preparing additional RTL
fixtures, and the dependencies needed for physical deployment.

## Choose a workflow

| Workflow | What you need |
| :--- | :--- |
| Arithmetic and host C checks | Python 3.10+, NumPy, GCC; included model |
| Five standalone RTL tests | Icarus Verilog (`iverilog`, `vvp`), Make |
| Checkpoint verification and export | Compatible checkpoint, PyTorch, NumPy, ONNX, ONNX Runtime |
| STM32 model generation and host equivalence | Exported ONNX model, the Python packages above, GCC |
| Training and activation calibration | External RL environment, its compatible Python/PyTorch/Isaac Gym installation, GPU |
| FPGA board deployment | MINI_7010, JTAG connection, Vivado/Vitis; recorded runs use 2026.1 |
| STM32 board deployment | STM32H723, SWD connection, ST HAL/CMSIS sources, ARM GCC, CMake, Ninja, STM32CubeProgrammer |

## Export a checkpoint

Run from the repository root in a Python 3.10–3.12 virtual environment. These
instructions use PyTorch 2.4.0, the export version recorded in the
[reproduction notes](REPRODUCE_QAT.md). If your quickstart environment uses a
newer Python, create a separate environment for export.

The following steps need a compatible RoboAccel policy checkpoint; checkpoints
and ONNX files are not committed. The included generated C model is sufficient
for the quickstart but does not replace a checkpoint for export.

```bash
python -m pip install "torch==2.4.0" "numpy<2" onnx onnxruntime
source training/scripts/env_solid.sh
```

If `env_solid.sh` reports that `ROBOACCEL_RL_ENV` is unset, you can continue with
verification and export. That variable is needed for training and rollouts.

**1. Check the quantized policy against the integer reference.** Replace the
example path with your checkpoint.

```bash
RA_CHECKPOINT=/path/to/checkpoint.pt
python quantization/scripts/bitexact_check.py \
  --checkpoint "$RA_CHECKPOINT" --n 4000 --weight-bits 16 --act-bits 16
```

The recorded W16A16 baseline produces `BIT_EXACT: PASS`, with 9/9 tensors exact.

**2. Export ONNX, then the FPGA program and cache images.**

```bash
python quantization/scripts/export_to_onnx.py "$RA_CHECKPOINT" --out policy.onnx
python fpga/tools/export_policy.py \
  --model policy.onnx --out fpga/generated --samples 1000 --seed 7
make -C fpga/tb policy
```

For the demonstrated topology, the FPGA export produces 13 descriptors. The
`policy` testbench loads this program, weights, and golden inputs from
`fpga/generated/`. See the [deployment mapping](SOLID_POLICY_DEPLOYMENT_MAPPING.md)
for the supported topology.

**3. Generate a C model and run host equivalence.**

```bash
python quantization/scripts/gen_h7_branched.py policy.onnx \
  --name mymodel --out stm32/models/mymodel --golden 24
bash quantization/scripts/hosttest/build_and_run.sh stm32/models/mymodel
```

The generator makes random golden observations by default. For observations
from your own rollouts, add `--golden-npz calib.npz`. Host equivalence executes
both kernel paths with a portable C replacement for the ARM SMLALD intrinsic;
it does not measure Cortex-M7 performance.

## Additional RTL fixtures

The targets in [`fpga/tb/Makefile`](../fpga/tb/Makefile) have different prerequisites.
Exporting one policy does **not** prepare them all.

| Target | Required input |
| :--- | :--- |
| `primitives gemm concat sequencer top` | None beyond the committed sources |
| `policy` | Policy program and cache images in `fpga/generated/` |
| `v3` | `fpga/generated_v3_in.hex` and `fpga/generated_v3_exp.hex`; the vector generator referenced by the testbench is not included in this repository |
| `stream` | Separate Go2 and G1 exports in `fpga/generated_go2/` and `fpga/generated_g1/`, matching the dimensions and action addresses in the Makefile |

The `v3` and `stream` fixtures are not committed. Those targets need additional
inputs beyond the checkpoint workflow above, so `make -C fpga/tb all` is not a
clean-clone quickstart.

See the [verification guide](../fpga/docs/07_verification_guide.md) and
[instruction protocol](../fpga/docs/10_instruction_sequence_protocol.md) for
the hardware interfaces exercised by these tests.

## Training and activation calibration

The RL environment and robot assets are external and are not distributed in
this repository. Obtain your own compatible checkout and use its Python
environment. See [NOTICE.md](../NOTICE.md) and
[QAT reproduction](REPRODUCE_QAT.md).

```bash
export ROBOACCEL_RL_ENV=/path/to/wheel_legged_gym_checkout
export ROBOACCEL_PYTHON=/path/to/isaac-gym-env/bin/python
source training/scripts/env_solid.sh

"$ROBOACCEL_PYTHON" training/scripts/dump_calib_obs.py \
  --checkpoint "$RA_CHECKPOINT" --out calib.npz \
  --steps 400 --headless --num_envs 256

"$ROBOACCEL_PYTHON" quantization/scripts/calibrate_act_fracs.py \
  --checkpoint "$RA_CHECKPOINT" --obs calib.npz --act-bits 8 --out fracs.json
```

This A8 calibration is for precision experiments. The current FPGA and STM32
deployment export uses INT16 weights and Q8.8 activations; producing `fracs.json`
does not change that deployment format.

## Board deployment

**FPGA.** Start with the recorded [MINI_7010 + Vivado/Vitis 2026.1 bring-up](../fpga/docs/11_mini7010_vivado2026_port.md)
and [PS driver design](../fpga/docs/05_ps_driver_design.md). The validated path
boots through JTAG and returns results through a mailbox. Historical hardware
notes contain paths from the original development workspace; adapt those paths
to your local toolchain and checkout before following the board commands.

**STM32.** Install the ARM toolchain and STM32CubeProgrammer, and provide the
external ST HAL/CMSIS tree expected by [`stm32/CMakeLists.txt`](../stm32/CMakeLists.txt).
That file currently sets `REF` to the original developer's absolute path;
update it for your checkout before building. Set `ROBOACCEL_STM32_ENV` to a
script that adds the toolchain to `PATH` if needed. With your board connected
and the generated `mymodel` directory in place:

```bash
if [ -n "${ROBOACCEL_STM32_ENV:-}" ]; then source "$ROBOACCEL_STM32_ENV"; fi
bash stm32/scripts/flash.sh mymodel
python stm32/scripts/read_results.py mymodel
```

The flash command builds and programs the target. Timing measured on a host
does not substitute for the on-board DWT cycle measurements.

## Export limits

The FPGA does not range-check model dimensions at runtime. The exporter must
fit the model within these limits:

| Resource | Constraint |
| :--- | :--- |
| Weight cache, per bank | `sum(ceil(K/8) × ceil(N/8)) ≤ 1280` |
| Vector cache | Layout must fit 512 words |
| Instruction RAM | 1–32 descriptors |
| Host access | Do not modify caches or instruction RAM during an automatic sequence |

The [release audit](RELEASE_CANDIDATE.md) records what has been reproduced from
a clean clone and which results still require external hardware or a training
environment.
