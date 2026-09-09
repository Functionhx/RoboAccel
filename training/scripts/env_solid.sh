# RoboAccel environment.
#
# Everything inside this repository is located relative to this script, so a
# clone works from any path. The two things that cannot be vendored are named
# as variables you must set:
#
#   ROBOACCEL_RL_ENV   the RL training environment's checkout (Isaac Gym +
#                      the wheel-legged task). It carries no upstream licence
#                      and is deliberately NOT included here -- see NOTICE.md.
#   ROBOACCEL_PYTHON   a Python with torch + Isaac Gym for that checkout.
#
# Only the training component needs them. Quantization, FPGA export and the
# STM32 backend run without either.
#
#   source training/scripts/env_solid.sh

_ra_src="${BASH_SOURCE[0]:-$0}"
export ROBOACCEL_ROOT="$(cd "$(dirname "$_ra_src")/../.." && pwd)"
unset _ra_src

export PYTHONPATH="$ROBOACCEL_ROOT/quantization${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

if [ -n "${ROBOACCEL_RL_ENV:-}" ]; then
    export PYTHONPATH="$ROBOACCEL_RL_ENV:$PYTHONPATH"
else
    echo "env_solid.sh: ROBOACCEL_RL_ENV is not set." >&2
    echo "  Training needs it; quantization, FPGA and STM32 do not." >&2
    echo "  Point it at your Isaac Gym wheel-legged checkout, e.g." >&2
    echo "    export ROBOACCEL_RL_ENV=/path/to/wheel_legged_gym_checkout" >&2
fi

if [ -n "${ROBOACCEL_PYTHON:-}" ]; then
    export ROBOACCEL_PYTHON
    # Isaac Gym ships its shared objects beside the interpreter.
    _ra_lib="$(dirname "$(dirname "$ROBOACCEL_PYTHON")")/lib"
    [ -d "$_ra_lib" ] && export LD_LIBRARY_PATH="$_ra_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _ra_lib
fi

echo "ROBOACCEL_ROOT=$ROBOACCEL_ROOT"
