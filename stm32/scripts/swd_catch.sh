#!/usr/bin/env bash
# Continuous SWD catch-and-recover loop.
#
# The resident firmware reaches __WFI() a few milliseconds after every reset
# and gates the D1 debug clock, so the only way in is to attach inside that
# window. This hammers HOTPLUG attaches; the instant one lands it mass-erases
# in the SAME invocation (a second process would arrive far too late) and then
# flashes the hardened image, which spins instead of sleeping.
#
# Bounded on purpose: a hard deadline and a pidfile, so this cannot become an
# orphan burning a core.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /home/as/vllm/fpga/env/stm32.sh

model="${1:-go2}"
deadline_s="${2:-1500}"                 # 25 min hard stop
hex="$here/build/$model/h7_bench.hex"
log="$here/build/swd_catch.log"
pidfile="$here/build/swd_catch.pid"
echo $$ > "$pidfile"
: > "$log"
strip() { sed 's/\x1b\[[0-9;]*m//g'; }

[ -f "$hex" ] || { echo "missing $hex -- build first"; exit 1; }

start=$(date +%s)
attempt=0
# High frequencies first: a faster attach sequence fits more easily inside the
# post-reset window. The slow ones are there for signal-integrity margin.
freqs=(4000 4000 1800 4000 950 1800 4000 480)

while :; do
    now=$(date +%s); elapsed=$((now - start))
    if [ "$elapsed" -ge "$deadline_s" ]; then
        echo "DEADLINE: $attempt attempts in ${elapsed}s, never caught the window." | tee -a "$log"
        rm -f "$pidfile"; exit 2
    fi
    f=${freqs[$((attempt % ${#freqs[@]}))]}
    attempt=$((attempt + 1))

    # Mass-erasing 1 MB loses the race: the core reaches __WFI() and gates the
    # debug clock mid-erase. Redirecting the boot address is ~100 ms of flash
    # work, and it is permanent -- the part then always comes up in the ROM
    # bootloader, which never sleeps, so every later step is unhurried.
    out=$(timeout 12 STM32_Programmer_CLI -c port=SWD mode=HOTPLUG freq=$f \
              -ob BOOT_CM7_ADD0=0x1FF0 -ob displ 2>&1 | strip)

    if echo "$out" | grep -q "Device ID"; then
        {
          echo "=== CAUGHT on attempt $attempt (freq=$f, ${elapsed}s) ==="
          echo "$out"
        } | tee -a "$log"
        if echo "$out" | grep -qiE "Option Bytes successfully programmed|programmed successfully"; then
            echo "BOOT REDIRECTED: part now comes up in the ROM bootloader." | tee -a "$log"
            rm -f "$pidfile"; exit 0
        fi
        echo "connected, option-byte write did not confirm -- continuing" | tee -a "$log"
    fi
    printf '%s attempt=%d freq=%d elapsed=%ds\n' "$(date +%H:%M:%S)" "$attempt" "$f" "$elapsed" >> "$log"
    sleep 0.2
done
