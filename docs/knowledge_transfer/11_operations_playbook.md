# 11 · 操作手册（operations playbook）

**精确命令 + 浪费过时间的陷阱。** 陷阱部分比命令更值钱。

---

## 一、环境（绝对不要新建 venv）

见 `actuatex/docs/LOCAL_ENVIRONMENTS.md`。本机已有全部依赖。

```bash
source /home/as/vllm/fpga/projects/rl_accel/qat/scripts/env_solid.sh
```

| 变量 | 指向 | 用途 |
|---|---|---|
| `$ISAAC_PY` | `envs/fudan-py38` (py3.8, torch 2.1.1 CUDA, Isaac Gym Preview 4) | 训练、闭环评估 |
| `$MJ_PY` | `envs/fudan-mujoco-py39` (py3.9, torch 2.4.0, onnx) | 导出、分析、验证 |
| `$SOLID_WT` | `.worktrees/robust-training` | Codex 冻结后的基础设施 |
| `$QAT_ROOT` | `projects/rl_accel/qat` | 本项目 |

**licensing**：上游仓库无 license，**按路径 import，从不 vendoring**。

### 两个环境的分工不能混

* `onnx` 只在 `$MJ_PY` 里 → ONNX 导出必须用它。
* Isaac Gym 只在 `$ISAAC_PY` 里，且 **`import isaacgym` 必须在 `import torch` 之前**，
  否则报 `PyTorch was imported before isaacgym modules`。

---

## 二、主流水线（全部按环境变量参数化，无需改文件）

```bash
export TASK=locomotion_v2
CK=$QAT_ROOT/artifacts/v2/frozen/SOLID_FP32_V2.pt

# 1. 标定观测（必须 on-policy）
$ISAAC_PY scripts/dump_calib_obs.py --checkpoint $CK \
    --out artifacts/v2/calib_obs.npz --steps 400 --headless --num_envs 256
$ISAAC_PY scripts/calibrate_act_fracs.py --checkpoint $CK \
    --obs artifacts/v2/calib_obs.npz --act-bits 8 --out artifacts/v2/act_fracs_a8.json

# 2. PTQ ladder
bash scripts/run_v2_ladder.sh

# 3. QAT + 强制对照（对照先跑）
CKPT=$CK FRACS=artifacts/v2/act_fracs_a8.json TASK=$TASK TAG=_v2 ITERS=2000 \
    bash scripts/run_qat_and_control.sh

# 4. golden vectors（每个 segment 一个进程）
CKPT=$CK OUT=$PWD/artifacts/v2/regimes ENVS=64 bash scripts/dump_regimes.sh

# 5. 导出 + 芯片
PYTHONPATH=$PWD $MJ_PY scripts/export_to_onnx.py $CK --out artifacts/v2/deploy/m.onnx
$MJ_PY ../tools/export_policy.py --model artifacts/v2/deploy/m.onnx \
    --out artifacts/v2/deploy/fpga --samples 1000 --seed 7
```

**标定必须来自 on-policy rollout。** 用高斯噪声标定会把 encoder 输入放到
真实范围之外，浪费掉一半 INT8 网格——然后看起来像量化悬崖，其实是标定 bug。

---

## 三、陷阱（每一条都真实浪费过时间）

### 3.1 `nohup cmd > dir/log` 且 `dir` 不存在

shell 在 exec 脚本**之前**打开重定向。脚本内部的 `mkdir -p` 来不及。
作业瞬间死亡，**wrapper 仍然报 exit 0**。

→ 先建目录；启动后**验证进程存在**，不要相信退出码：
```bash
ps -eo pid,cmd | grep "[t]rain_policy" | grep -oE "\-\-run \S+"
```

### 3.2 永远不要编辑正在运行的 shell 脚本

bash 按**字节偏移**增量读取。插入几行会让偏移错位，
执行会从某条命令中间恢复。本项目因此**静默跳过了一整个评估臂**。

### 3.3 `until ! pgrep -f foo.sh` 会匹配到自己

watcher 的命令行里包含 `foo.sh`，于是永远等自己退出。两次死锁。

→ 改为 grep 日志里的完成标记，或匹配解释器路径：
```bash
ps -eo cmd | grep -c "[f]udan-py38.*train_policy.py --run"
```

### 3.4 参数按位置传，不要打包进分隔字符串

`IFS='|' read -r run seed a b c` 定长读取会把 `--act-fracs` 和它的路径
粘成一个 token，argparse 一小时后才报 `unrecognized arguments`。

### 3.5 GPU 并发上限是 2

实测（4096 envs）：

| 并发 | 合计吞吐 |
|---|---|
| 1 | ~69 iters/min |
| 2 | **~68 iters/min** |
| 3 | ~28 iters/min（颠簸） |
| 4 | `CUDA error: an illegal memory access` |

**"用满 GPU" ≠ "同时跑到崩"。** 保持 2 个常驻，用 slot scheduler 排队。

### 3.6 Isaac Gym 每个进程只能建一个 simulator

释放后再建第二个会 **segfault**。这就是 Codex 的
`run_evaluation_experiments.py` 每个 scenario 起一个子进程的原因。
`dump_regimes.sh` 同理：`--segment` 选一个，外层 shell 循环。

### 3.7 两个评估器不能混用

| checkpoint | 评估器 | 症状（用错时） |
|---|---|---|
| `robust_v1` | `evaluate_robustness.py` | — |
| `locomotion_v2` | `scripts/evaluate_locomotion.py` | `'RobustLeggedRobot' object has no attribute '_reward_body_contact'` |

`evaluate_robustness.py` 里硬编码了 `make_env("robust_v1")`。
`evaluate_locomotion.py` 在 worktree 顶层 `scripts/`（不在 package 内），
入口是 `evaluate()` 不是 `main()`，且**必须显式传 `--seed`**
（base parser 默认 `None`，`torch.manual_seed(None)` 会抛异常）。

### 3.8 `evaluate_locomotion.py` 需要 checkpoint 同目录下有 `config.json`

它读 `checkpoint_path.parent / "config.json"` 并取 `saved["task"]`。
`train_policy.py` 现在会写，但**冻结 checkpoint 时要一起复制**，
且文件名必须就叫 `config.json`。

---

## 四、H7 板子

```bash
cd /home/as/vllm/fpga/projects/h7_bench
cp <qat>/artifacts/v2/deploy/h7/ra_{model.c,model.h,golden.h} models/solid_v2/
source /home/as/vllm/fpga/env/stm32.sh
bash scripts/flash.sh solid_v2          # 先直写，失败才 mass erase
python3 scripts/read_results.py solid_v2
```

* 结果通过 **JTAG mailbox** 回读（板上 CH340 未接本机），
  用 `arm-none-eabi-nm` 解析 `g_bench` 地址后 `-r32` 读。
* `magic` 必须是 `0x48375241`（"AR7H"）。不对就是**旧固件在跑**，
  脚本会拒绝报告 stale RAM。
* 换 model 只改 `models/<name>/`，`main.c` 不动，
  所以 **LED / PA15 按键行为会保留**。

---

## 五、换基线时的检查清单

1. `sha256sum -c artifacts/v2/frozen/infra_manifest_final.sha256` — 基础设施变了吗？
2. 变了 → **重跑一个已知评估结果**并比对，不要假设无害。
3. 拓扑变了吗？`scripts/handoff_audit.py` 的 `derived` / `deployment` 字段。
4. **重新标定 activation fracs** —— 不同策略的激活范围差 2–4 倍。
5. 跑完整算术门（[08](08_verification_chain.md)）。
6. 才开始跑控制实验。
