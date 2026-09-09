# 10 · 术语表（glossary）

中文解释 + **规范英文术语**（你在代码、论文、硬件手册里会遇到的那个词）。

---

## 量化（quantization）

| 中文 | English | 在本项目的含义 |
|---|---|---|
| 定点数 | **fixed-point arithmetic** | 用整数 `q` 表示实数 `x ≈ q·2^(-f)` |
| 小数位数 | **fractional bits** (`frac`, `f`) | `LSB = 2^(-f)`。本项目**逐层**不同，可为负 |
| 最低有效位 | **LSB** (least significant bit) | 量化步长。误差常以 LSB 为单位报告 |
| 激活量化 | **activation quantization** | 层间张量的量化。本项目的瓶颈 |
| 权重量化 | **weight quantization** | 权重的量化。INT8 下**基本免费** |
| 再量化 | **requantization** | INT48 累加 → INT16 激活网格的右移+舍入 |
| 训练后量化 | **PTQ** (post-training quantization) | 训好再量化 |
| 量化感知训练 | **QAT** (quantization-aware training) | 训练时就在闭环里模拟整数算术 |
| 伪量化 | **fake quantization** | 前向量化、反向 STE，用 float 模拟整数 |
| 直通估计器 | **STE** (straight-through estimator) | `round()` 反向当恒等函数 |
| 标定 | **calibration** | 从真实 rollout 统计激活范围，选 frac |
| 饱和 | **saturation / clipping** | 超出量程被截断。**本项目标定后为 0** |
| 黄金向量 | **golden vector** | 固定输入+期望输出，跨实现比对用 |
| 位精确 | **bit-exact / bit-identical** | 两个实现输出的整数完全相同 |

## 硬件（hardware）

| 中文 | English | 含义 |
|---|---|---|
| 可编程逻辑 | **PL** (programmable logic) | Zynq 的 FPGA 部分，跑 GEMM/ELU/CONCAT |
| 处理系统 | **PS** (processing system) | Zynq 的 ARM 核，写指令、推数据 |
| 算子描述符 | **operator descriptor** | 一条指令，含 opcode / base 地址 / 维度 / shift |
| 指令序列 | **instruction program** | 32 条上限，本策略用 13 条 |
| 权重缓存 | **weight cache** | 8 banks × 1280 × 128-bit |
| 向量缓存 | **vector cache** | 512 × 128-bit，存激活 |
| 乘累加 | **MAC** (multiply-accumulate) | 本策略 38,400 次 |
| 累加器 | **accumulator** | INT48，防止 GEMM 溢出 |
| 紧耦合内存 | **DTCM** | H7 上的零等待 SRAM，权重放这里 |
| SIMD 乘累加 | **SMLALD** | ARM 指令，一次两个 16×16 乘加到 64 位 |
| 周期计数器 | **DWT CYCCNT** | H7 上的精确计时 |

## 强化学习（reinforcement learning）

| 中文 | English | 含义 |
|---|---|---|
| 观测 | **observation** | 25 维传感器向量 |
| 观测历史 | **observation history** | 5 帧 × 25 = 125 维 |
| 编码器 | **encoder** | 125 → 128 → 64 → 3 的 MLP |
| 潜变量 | **latent** | 3 维，**就是速度估计** |
| 执行器 | **actor** | 28 → 128 → 64 → 32 → 6 的 MLP |
| 评论家 | **critic** | 只在训练时存在，**永不导出** |
| 闭环控制 | **closed-loop control** | 动作影响下一个观测 |
| 域随机化 | **domain randomization** | 训练时随机化物理参数 |
| 残差策略 | **residual policy** | `action = base(x) + residual(x)` |

## 本项目特有

| 术语 | 含义 |
|---|---|
| `SOLID_FP32` / `SOLID_FP32_V2` | 冻结的 FP32 基线，所有实验从它分叉 |
| `SOLID_FP32_PLUS` | §7 强制对照：同样步数、**不量化** |
| `robust_v1` | 旧任务，**存在 chassis-support exploit**，绝对数字不可信 |
| `locomotion_v2` | Codex 修正后的任务，当前基线 |
| `support_fraction` | 靠**轮子**支撑的时间比例。1.000 = 真的在平衡 |
| `success` | 逐段通过/不通过 |
| `W8A8` | 8-bit 权重 + 8-bit 激活 |
| `ACT_FRAC` | 默认激活 frac = 8（Q8.8） |
| `MAX_WEIGHT_FRAC` | 权重 frac 上限 14，**软件常量非硬件限制** |
| `IDEAL_ROUNDING` | 切换到精确舍入的开关，**默认关**（不匹配 RTL） |
| chassis-support exploit | 机器人趴在底盘上而非靠轮子平衡的退化解 |

## 论文用英文表述（precise English formulations）

* *At the deployed 16-bit precision, quantization is lossless and QAT is
  unnecessary. At 8-bit activations, QAT restores wheel support and
  straight-line locomotion but never restores turning.*
* *Improving the float baseline makes QAT more necessary, not less.*
* *The W8A8 collapse is distributed across the actor's hidden layers with
  strong error interaction; no per-layer fix addresses it.*
* *The requantizer's rounding bias is differential across the wheels rather
  than systematically signed; the differential, not the mean, is the
  meaningful quantity.*
