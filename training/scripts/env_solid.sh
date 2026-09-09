# The frozen Codex "robust-training" infrastructure (goal.md §2).
# Same licensing rule as env_isaac.sh: imported by path, never vendored.
export SOLID_WT=/home/as/vllm/rm-cortex-references/wheel-legged/fudan_rl_wheel_leg/.worktrees/robust-training
export SOLID_PLANE=$SOLID_WT/plane
export ISAAC_PY=/home/as/vllm/rm-cortex-references/wheel-legged/envs/fudan-py38/bin/python
export LD_LIBRARY_PATH=/home/as/vllm/rm-cortex-references/wheel-legged/envs/fudan-py38/lib:${LD_LIBRARY_PATH:-}
export QAT_ROOT=/home/as/vllm/fpga/projects/rl_accel/qat
export PYTHONPATH=$SOLID_PLANE:$QAT_ROOT
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR=$SOLID_WT/.cache/torch_extensions
export CUDA_CACHE_PATH=$SOLID_WT/.cache/cuda
export OMP_NUM_THREADS=1
