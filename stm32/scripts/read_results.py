#!/usr/bin/env python3
"""Read the H7 benchmark mailbox over SWD and print it."""
import re, subprocess, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
model = sys.argv[1] if len(sys.argv) > 1 else "go2"
elf = HERE / "build" / model / "h7_bench"
hdr = (HERE / "models" / model / "ra_model.h").read_text()
nact = int(re.search(r"#define RA_ACT_FEATURES\s+(\d+)", hdr).group(1))
# key_idle_level, key_presses, led_level are appended after actions, so older
# firmware simply reads short and these stay absent rather than misaligning.
NKEY = 3

nm = subprocess.run(["arm-none-eabi-nm", str(elf)], capture_output=True, text=True).stdout
addr = next(int(l.split()[0], 16) for l in nm.splitlines() if l.split()[-1] == "g_bench")
words = 30 + nact + NKEY

# -r32 takes a byte count, not a word count.
out = subprocess.run(["STM32_Programmer_CLI", "-c", "port=SWD", "mode=HOTPLUG",
                      "-r32", hex(addr), str(words * 4)],
                     capture_output=True, text=True).stdout
out = re.sub(r"\x1b\[[0-9;]*m", "", out)      # CubeProgrammer colourises even when piped
vals = []
for line in out.splitlines():
    m = re.match(r"\s*0x[0-9A-Fa-f]{8}\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)", line)
    if m:
        vals += [int(x, 16) for x in m.group(1).split()]
if len(vals) < words - NKEY:
    print("short read:", len(vals), "of", words); print(out[-1500:]); sys.exit(1)

f = ["magic", "cpu_hz", "runs", "weight_bytes", "macs", "obs_features",
     "act_features", "verify_scalar_ok", "verify_simd_ok"]
d = dict(zip(f, vals[:9]))
# A failed flash leaves the previous run's RAM in place. Without this guard the
# stale contents print as if they were a measurement.
if d["magic"] != 0x48375241:
    print(f"BAD MAGIC 0x{d['magic']:08X} (expected 0x48375241) -- firmware for "
          f"'{model}' is not running; refusing to report stale RAM.")
    sys.exit(2)
i = 9
for name in ("scalar_flash", "simd_flash", "simd_dtcm", "e2e_simd_dtcm"):
    d[name] = vals[i:i+5]; i += 5
d["dtcm_used"] = vals[i]; i += 1
d["actions"] = [(v - (1 << 32) if v >> 31 else v) for v in vals[i:i+nact]]
i += nact
if len(vals) >= i + NKEY:
    d["key_idle_level"], d["key_presses"], d["led_level"] = vals[i:i+NKEY]

hz = d["cpu_hz"] or 480000000
print(f"model={model}  magic=0x{d['magic']:08X}  cpu={hz/1e6:.0f} MHz  runs={d['runs']}")
print(f"weights={d['weight_bytes']:,} B  MACs={d['macs']:,}  obs={d['obs_features']}  act={d['act_features']}")
print(f"verify: scalar={'PASS' if d['verify_scalar_ok'] else 'FAIL'}  simd={'PASS' if d['verify_simd_ok'] else 'FAIL'}")
print(f"dtcm weight copy = {d['dtcm_used']:,} B")
print(f"{'variant':>16} {'mean_cyc':>10} {'p50':>9} {'p99':>9} {'min':>9} {'max':>9} {'mean_us':>9} {'p99_us':>8}")
for name in ("scalar_flash", "simd_flash", "simd_dtcm", "e2e_simd_dtcm"):
    a = d[name]
    if a[0] == 0: continue
    print(f"{name:>16} {a[0]:>10,} {a[1]:>9,} {a[2]:>9,} {a[3]:>9,} {a[4]:>9,} "
          f"{a[0]/hz*1e6:>9.2f} {a[2]/hz*1e6:>8.2f}")
print("actions:", d["actions"])
if "key_idle_level" in d:
    print(f"key PA15: idle level={d['key_idle_level']}  presses={d['key_presses']}  "
          f"LED level={d['led_level']}/255  "
          f"(idle high -> button pulls LOW when pressed; idle low -> pulls HIGH)")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / f"{model}.json").write_text(json.dumps(d, indent=2))
