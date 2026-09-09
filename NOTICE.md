# Attribution and external dependencies

## Included, MIT licensed
The RoboAccel accelerator RTL, testbenches, exporter and PS software derive
from `rl_on_fpga` (MIT, © 2026 DreamChaser). That licence and copyright are
preserved in `LICENSE`.

## NOT included — external dependency, deliberately
The reinforcement-learning environment, PPO implementation and robot assets
used to train the policies are **not** part of this repository and are **not
redistributable**: the upstream repository carries **no licence**.

This repository contains only the *adapter* (`training/scripts/train_policy.py`),
which imports that environment by `PYTHONPATH` and never vendors it. To
reproduce training you must obtain the environment yourself and export
`ROBOACCEL_RL_ENV` (its checkout) and `ROBOACCEL_PYTHON` (an interpreter with
torch and Isaac Gym) before sourcing `training/scripts/env_solid.sh`.

Everything downstream of a trained checkpoint — quantization, verification,
export, FPGA and MCU deployment — is fully contained here and needs no
unlicensed code.

## Hardware
STM32 HAL sources are not vendored; `stm32/CMakeLists.txt` references an
external ST HAL tree via `REF=`. The ARM toolchain is likewise external — set
`ROBOACCEL_STM32_ENV` to a script that puts `arm-none-eabi-*` and
STM32CubeProgrammer on `PATH`.
