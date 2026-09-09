#!/usr/bin/env bash
# Recover an H7 whose resident firmware gates the debug clock before we can
# attach. Bridge the two BOOT0 pads (next to the USB-A shell), tap MCU RESET,
# then run this. The ROM bootloader never sleeps, so SWD -- or USB DFU -- is
# reachable; we mass-erase to evict the offending image.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# The ARM toolchain is external to this repository. Set ROBOACCEL_STM32_ENV to
# a script that puts arm-none-eabi-* and STM32CubeProgrammer on PATH.
if [ -n "${ROBOACCEL_STM32_ENV:-}" ]; then source "$ROBOACCEL_STM32_ENV"; fi
command -v arm-none-eabi-gcc >/dev/null || {
    echo "arm-none-eabi-gcc not on PATH; set ROBOACCEL_STM32_ENV" >&2; exit 1; }
model="${1:-go2}"
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

port=""
for p in "port=SWD mode=HOTPLUG freq=1800" "port=SWD mode=UR freq=1800" "port=USB1"; do
    echo "--- trying $p"
    if timeout 40 STM32_Programmer_CLI -c $p 2>&1 | strip | grep -q "Device ID"; then
        port="$p"; echo "*** reachable via: $p"; break
    fi
done
[ -z "$port" ] && { echo "FAIL: no interface responded. Is BOOT0 bridged and RESET tapped?"; exit 1; }

echo "--- mass erase"
timeout 90 STM32_Programmer_CLI -c $port -e all 2>&1 | strip | grep -Ei "eras|error" | tail -4

echo "--- flashing $model"
timeout 120 STM32_Programmer_CLI -c $port -w "$here/build/$model/h7_bench.hex" -v 2>&1 \
    | strip | grep -Ei "download|verif|error" | tail -6
echo
echo "NOW: remove the BOOT0 bridge and tap MCU RESET."
echo "Then: scripts/read_results.py   (firmware is the hardened build; it spins, never sleeps)"
