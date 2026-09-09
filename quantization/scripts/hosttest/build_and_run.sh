#!/usr/bin/env bash
# Compile the generated H7 model on the host and check it against the golden
# vectors. SMLALD is ARM-only inline asm, so it is swapped for a portable C
# equivalent with identical semantics: two signed 16x16 products accumulated
# into 64 bits. Everything else is compiled unmodified.
set -euo pipefail
MODEL_DIR=$1
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
HERE0=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE0/../../.." && pwd)
SRC=${ROBOACCEL_H7_SRC:-$REPO/stm32/src}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "int16_t ra_elu_probe(int16_t x);" > "$WORK/ra_probe.h"
cp "$SRC/ra_kernel.h" "$MODEL_DIR"/ra_model.c "$MODEL_DIR"/ra_model.h \
   "$MODEL_DIR"/ra_golden.h "$HERE/main.c" "$WORK/"
python3 - "$SRC/ra_kernel.c" "$WORK/ra_kernel.c" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
start = s.index("static inline int64_t ra_smlald")
end = s.index("}", s.index("return u.q;")) + 1
s = s[:start] + '''static inline int64_t ra_smlald(uint32_t a, uint32_t b, int64_t acc)
{
    /* Host stand-in for the SMLALD instruction: acc += a.lo*b.lo + a.hi*b.hi,
       signed 16x16 products, 64-bit accumulate. */
    int32_t alo = (int16_t)(a & 0xFFFFu), ahi = (int16_t)(a >> 16);
    int32_t blo = (int16_t)(b & 0xFFFFu), bhi = (int16_t)(b >> 16);
    return acc + (int64_t)alo * blo + (int64_t)ahi * bhi;
}''' + s[end:]
# Expose the static-inline ELU so the host test can probe the ROM directly.
s += "\nint16_t ra_elu_probe(int16_t x) { return ra_elu(x); }\n"
open(sys.argv[2], "w").write(s)
PY
gcc -O2 -std=c11 -Wall -I"$WORK" -o "$WORK/test" \
    "$WORK/main.c" "$WORK/ra_kernel.c" "$WORK/ra_model.c" -lm
"$WORK/test"
