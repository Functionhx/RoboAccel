# 02 · 代码阅读路径（code reading curriculum）

按依赖顺序走，每一站给出**读什么、为什么、以及读完应该能回答的问题**。

---

## 阶段 0 · 先跑一次验证链（30 分钟）

不要先读代码。先确认环境是好的：

```bash
source qat/scripts/env_solid.sh
$ISAAC_PY qat/scripts/verify_against_rtl.py            # 期望 PASS
bash qat/scripts/hosttest/build_and_run.sh qat/artifacts/v2/deploy/h7
```

**能回答**：验证链的三个层次分别在验证什么？

---

## 阶段 1 · 定点算术核心（最重要）

### 1.1 `qat/hwq/fixed_ref.py`（约 260 行，**全读**）

这是**仲裁者**。读的顺序：

| 函数 | 关注点 |
|---|---|
| `quantize` | 为什么是 `2.0 ** frac` 不是 `1 << frac` |
| `round_shift` | 为什么负数减 half —— 这是**故意复现硬件 bug** |
| `choose_weight_frac` | 如何选每层权重 frac，`MAX_WEIGHT_FRAC` 的作用 |
| `elu_q88` | 256 项 ROM + 3-bit 插值 + 饱和 |
| `LayerRef.__call__` | 一层完整的定点前向 |
| `FudanFixedPolicy.__init__` | **CONCAT 约束的断言在这里** |
| `check_residual_concat` | residual actor 的第二个同类约束 |

**能回答**：给定 `f_w`、`f_in`、`f_out`，一次 GEMM 的 shift 是多少？
为什么 bias 在移位**之后**加？

### 1.2 `qat/hwq/torch_hw.py`（约 330 行）

torch 侧的镜像。关注三个 `autograd.Function`：
`_QuantizeAct`、`_HwGemm`、`_HwElu` —— 前向必须与 1.1 完全一致，
反向是 **STE + clip-aware mask**。

然后读 `QuantConfig`：`act_fracs`、`only_layers`、`IDEAL_ROUNDING` 各自的用途。

**能回答**：为什么用 float64 模拟整数是精确的？（提示：1.34e11 vs 2^53）

### 1.3 `qat/hwq/quant_policy.py`

`QuantActorCriticSequence` 是 upstream 策略的 **drop-in 替换**。
必须保留的三件事：

* 参数名 `encoder.0/.2/.4`、`actor.0/.2/.4/.6`（否则 checkpoint 加载失败）
* `update_distribution` 里的 `self.latent.detach()`
* critic 保持 **FP32**（goal.md §12）

`build_from_checkpoint` 处理 Codex 的 `std → log_std` 改名。

---

## 阶段 2 · 验证脚本

| 脚本 | 验证什么 |
|---|---|
| `scripts/verify_against_rtl.py` | 解析 `.sv` 和 `ra_kernel.c` 源码，比对 ELU ROM 与舍入规则 |
| `scripts/bitexact_check.py` | torch ↔ numpy，9 个张量逐个比 |
| `scripts/hosttest/build_and_run.sh` | 主机上编译真实 H7 C 代码跑 golden vector |

**读 `bitexact_check.py` 时特别注意** `cmp(..., lsb=...)`：
每个张量必须用**自己那层的 LSB** 归一化。用固定 `1 << ACT_FRAC`
曾产生假 FAIL，见 [05](05_debugging_history.md) §1.1。

---

## 阶段 3 · 量化诊断（研究工具）

| 脚本 | 回答什么问题 |
|---|---|
| `calibrate_act_fracs.py` | 每层该用什么 frac？（含 CONCAT 取紧） |
| `quant_diagnostics.py` | 哪些张量饱和？误差有多大？权重浪费了几位？ |
| `layer_sensitivity.py` | 哪一层贡献了误差？**同时做单层和 leave-one-out** |
| `rounding_ablation.py` | 硬件舍入 vs 理想舍入差多少？ |

`layer_sensitivity.py` 值得精读：它**故意**做两种归因，并在两者
排序不一致时打印 `DISAGREE`。这个设计直接推翻了一个假设。

---

## 阶段 4 · 训练与评估

### 4.1 `scripts/train_policy.py`

关注 `install_quant_policy`：它把 upstream 的 policy class 换成量化版。
**必须是 class 不是函数**（runner 在实例化前读 `is_sequence`），
且**两个名字都要 patch**（`ActorCriticSequence` 和 `ActorCriticRobust`）。
构造后有 `isinstance` 断言。

### 4.2 `scripts/eval_quant_locomotion.py`

不实现评估逻辑，而是**重绑定 Codex 评估器的模块全局变量**。
注意它用 `ast` 从上游源码**解析** argparse 参数表——
这样上游加参数不会让 wrapper 失效。

**能回答**：为什么不自己写一个 evaluator？（goal.md §6 + 对照有效性）

---

## 阶段 5 · 导出与硬件

```
.pt ──export_to_onnx.py──► ONNX
                            ├──tools/export_policy.py──► generated/*.hex + policy_map.json
                            └──gen_h7_branched.py──────► ra_model.c / ra_golden.h
```

读 `policy_map.json` 的 `descriptors` 字段：13 条，
`GEMM/ELU/GEMM/ELU/GEMM/CONCAT/GEMM/ELU/GEMM/ELU/GEMM/ELU/GEMM`，
CONCAT 的 `shift` 是 **0**。

**能回答**：为什么 CONCAT 的 shift 必须是 0？（见 [03](03_math_to_code.md) §5）

---

## 阶段 6 · 上游 RL 基础设施（只读，不改）

`$SOLID_WT/plane/wheel_legged_gym/`：

* `envs/base/robust_robot.py` — 观测组装、reset、delay buffer
* `envs/wheel_legged/locomotion_config.py` — `LocomotionV2Cfg`，奖励权重在这里
* `rsl_rl/modules/actor_critic_robust.py` — 策略类
* `rsl_rl/runners/on_policy_runner.py:70` — **policy class 查表的那一行**

**这个 worktree 是只读参考，永远按路径 import，不 vendoring。**
