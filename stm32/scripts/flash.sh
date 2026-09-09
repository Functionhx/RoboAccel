#!/usr/bin/env bash
# Build and flash one model variant, then leave the target running.
#
# A plain write fails intermittently with "failed to download Sector[0]" when
# the previous image is already running. A mass erase clears it, but erasing
# unconditionally costs seconds on every flash and needlessly rewrites the whole
# part, so erase is a recovery step here, not the default: write, and only if
# that fails, erase and write again.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model="${1:-go2}"
source /home/as/vllm/fpga/env/stm32.sh

hex="$here/build/$model/h7_bench.hex"
CONNECT=(-c port=SWD mode=HOTPLUG freq=1800)
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

cmake -S "$here" -B "$here/build/$model" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake -DRA_MODEL="$model" >/dev/null
# `|| true`: with `set -o pipefail`, an up-to-date build prints nothing the
# filter matches, grep exits 1, and the script would abort before flashing --
# silently, because the same filter swallows the reason.
cmake --build "$here/build/$model" 2>&1 \
  | grep -E "Memory region|FLASH:|DTCMRAM:|text|^ *[0-9]+" | tail -10 || true
[ -f "$hex" ] || { echo "BUILD_FAILED $model: no $hex"; exit 1; }

# Returns 0 only on a verified download; callers treat anything else as failure.
try_write() {
    local out
    out=$(timeout 400 STM32_Programmer_CLI "${CONNECT[@]}" -w "$hex" -v "$@" 2>&1 | strip)
    echo "$out" | grep -E "Download verified successfully|Error" | tail -3
    echo "$out" | grep -q "Download verified successfully"
}

if try_write -rst; then
    echo "FLASH_OK $model (no erase needed)"
    exit 0
fi

echo "--- write failed; falling back to mass erase and retry"
timeout 200 STM32_Programmer_CLI "${CONNECT[@]}" -e all 2>&1 | strip \
  | grep -Ei "Mass erase|Error" | tail -2

if try_write -rst; then
    echo "FLASH_OK $model (recovered via erase)"
    exit 0
fi

echo "FLASH_FAILED $model -- see scripts/swd_catch.sh if the part is locked out"
exit 1
