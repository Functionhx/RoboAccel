# 01 · 前置知识地图（prerequisite knowledge map）

**不要按顺序全部学完再开始。** 按需查阅，标 ★ 的是真正卡住理解的。

---

## 神经网络与推理

| 主题 | 需要到什么程度 | 在本项目哪里出现 |
|---|---|---|
| MLP 前向 | ★ 能手算 `y = Wx + b` 的形状 | 整个策略 |
| **GEMM** | ★ 知道 `K·N` 是 MAC 数，`ceil(K/8)·ceil(N/8)` 是缓存字数 | `hwq/fixed_ref.py:LayerRef` |
| 激活函数 / **ELU** | ★ `ELU(x) = x` 或 `e^x − 1` | 硬件用 256 项 ROM 近似 |
| 张量形状 | ★ 能追 25 → 125 → 3 → 28 → 6 | [00](00_project_map.md) |
| 残差网络 | 知道"两条路相加"即可 | Codex 最终策略 |
| 归一化 | 了解即可（本策略没用 BatchNorm） | PL 有 NORM 算子但未使用 |

## 强化学习

| 主题 | 需要到什么程度 | 备注 |
|---|---|---|
| 策略网络 / 价值网络 | ★ 知道 actor 和 critic 的分工 | **critic 永不导出** |
| PPO | 会读超参即可，**不需要会推导** | 本项目不改 PPO |
| 闭环评估 | ★ 理解"动作影响下一个观测" | QAT 必须在闭环里做 |
| 观测历史编码 | ★ 5 帧堆叠 → encoder | 不是 RNN |
| 域随机化 | 了解 | 上游负责 |

**不需要**：PPO 的数学推导、GAE 的证明、其他 RL 算法。
goal.md 明确把 RL 基础设施划为**冻结的**，本项目只负责量化与部署。

## 数值与定点

| 主题 | 需要到什么程度 |
|---|---|
| 二进制补码 | ★ 知道算术右移对负数是 floor |
| **定点表示** | ★ `x ≈ q·2^(-f)`，能算量程和 LSB |
| 舍入模式 | ★ round-half-away-from-zero vs floor 的区别 |
| 灾难性抵消 | ★ 知道它会毁掉级数求和 → ELU 建表 bug |
| 浮点精度 | 知道 float64 有 53 bit 尾数即可 |

**这一块是本项目最核心的前置知识**，[03](03_math_to_code.md) 有完整推导。

## 嵌入式与 FPGA

| 主题 | 需要到什么程度 | 备注 |
|---|---|---|
| AXI4-Lite | 知道是"寄存器读写"总线即可 | 本项目**没有 DMA** |
| Zynq PS/PL 分工 | ★ 知道谁做什么 | [00](00_project_map.md) |
| DSP48E1 | 知道是硬件乘加单元即可 | 64 个组成 8×8 阵列 |
| Cortex-M7 / DTCM | 了解"零等待 SRAM" | 权重放这里提速 |
| SMILD / SMLALD | 知道是 SIMD 乘加即可 | 不需要会写汇编 |
| JTAG / SWD | 知道能读写目标内存 | mailbox 靠它回读 |

**不需要**：写 RTL、跑综合、理解时序收敛。
RTL 是**只读参考**，`verify_against_rtl.py` 会替你解析它。

## 工具链

| 工具 | 需要到什么程度 |
|---|---|
| PyTorch autograd | ★ 会写 `autograd.Function`（STE 需要） |
| numpy int64 | ★ 知道整数运算不会隐式转 float |
| ONNX | 知道是计算图交换格式即可 |
| Isaac Gym | ★ **必须在 import torch 之前 import** |
| bash / nohup | ★ 见 [11](11_operations_playbook.md) 的陷阱 |
| git | 基本操作 |

---

## 按角色的最小集合

**只想改量化算法** → 定点表示、舍入、GEMM、STE、`hwq/` 三个文件。

**只想跑实验并解读** → [09](09_experiment_interpretation.md) +
[11](11_operations_playbook.md)，算术细节可以先跳过。

**想部署到新硬件** → [03](03_math_to_code.md) 全部 +
[08](08_verification_chain.md) + `tools/export_policy.py`。

**想换一个新策略上板** → [00](00_project_map.md) 的拓扑约束 +
[04](04_design_decisions.md) D7（residual 融合）+ 容量上限
（1280 weight words / 512 vector words / 32 指令）。
