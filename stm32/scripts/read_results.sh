#!/usr/bin/env bash
# Read the benchmark mailbox out of DTCM over SWD and print it.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model="${1:-go2}"
# The ARM toolchain is external to this repository. Set ROBOACCEL_STM32_ENV to
# a script that puts arm-none-eabi-* and STM32CubeProgrammer on PATH.
if [ -n "${ROBOACCEL_STM32_ENV:-}" ]; then source "$ROBOACCEL_STM32_ENV"; fi
command -v arm-none-eabi-gcc >/dev/null || {
    echo "arm-none-eabi-gcc not on PATH; set ROBOACCEL_STM32_ENV" >&2; exit 1; }
elf="$here/build/$model/h7_bench"
addr=$(arm-none-eabi-nm "$elf" | awk '$3=="g_bench"{print "0x"$1}')
[ -n "$addr" ] || { echo "g_bench symbol not found" >&2; exit 1; }
nact=$(grep -oP '#define RA_ACT_FEATURES\s+\K[0-9]+' "$here/models/$model/ra_model.h")
words=$((28 + nact))
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -r32 "$addr" "$words" 2>&1 \
  | grep -vE "^libusb" | sed -n '/0x/p' | tail -30
