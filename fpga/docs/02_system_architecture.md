# 02 整体系统架构

## 1. 设计边界

系统面向 Xilinx Zynq-7000 `XC7Z010CLG400-2`。PL 以 100 MHz 运行，实现通用定点算子、片上 Cache、指令 RAM 和自动调度器；Cortex-A9 PS 裸机程序负责模型装载，并在初始化阶段把网络拓扑、算子顺序和张量地址编码为指令程序。

网络并未固化在 bitstream 中。兼容现有算子的网络通常只需重新运行导出器并重新编译 PS；只有出现新算子、现有精度无法覆盖或 Cache 容量不足时才需要修改 PL。

## 2. 数据通路

```text
                         AXI4-Lite
Cortex-A9 PS  ------------------------------------+
  program load / cache host window / start / IRQ |
                                                    v
                                  +---------------------------+
                                  | rl_accel_axil_top         |
                                  | registers + host windows  |
                                  +-------------+-------------+
                                                |
                                  +-------------v-------------+
                                  | instruction sequencer     |
                                  | 32 x 128-bit program RAM  |
                                  +-------------+-------------+
                                                |
                                  +-------------v-------------+
                                  | rl_accel_core             |
                                  | command latch + arbiter   |
                                  +--+-----+------+--------+---+
                                     |     |      |        |
                                   GEMM   ELU    NORM   CONCAT
                                     |     |      |        |
                          +----------+-----+------+--------+--+
                          | Vector Cache 512 x 128 bit        |
                          | Weight Cache 8 x 1280 x 128 bit   |
                          +-----------------------------------+
```

核心一次只执行一条算子命令，但 sequencer 会在上一条完成后自动取出下一条并启动核心。PL 没有 DMA 或 DDR master；PS 只在初始化阶段装载最多 32 条描述符，推理阶段发一次序列启动并等待整网完成。

## 3. 数值格式

| 数据 | 格式 | 说明 |
|---|---|---|
| 输入、activation、bias、action | signed Q8.8 INT16 | 实数除以 256 |
| weight | signed INT16、每层独立 fractional bits | 当前七层为 11/12/14/13/14/14/14 |
| GEMM accumulator | signed INT48 | 乘加后按描述符 shift 舍入、饱和 |
| NORM reciprocal | E5M11 | 仅旧网络或其他兼容网络使用 |

每个 128-bit Cache word 含 8 个 INT16 lane，lane 0 位于 `[15:0]`。Weight Cache 的 8 个 bank 同地址并行读出一个 8×8 tile。

固定 Q2.14 无法覆盖当前模型第一层约 ±10 的权重范围，因此导出器按每个 GEMM 的最大绝对权重选择最高不溢出的 fractional bits，并把该值写入 `output_shift`。activation 和 bias 仍统一为 Q8.8。

## 4. PL 算子

### GEMM

8×8 INT16 MAC 阵列同时处理 8 个输入和 8 个输出。64 个乘法映射到 DSP48E1，结果以 INT48 累加；最后加入 Q8.8 bias，做带符号舍入移位并饱和到 INT16。K/N 可以不是 8 的倍数，末 tile 的无效 lane 由控制逻辑屏蔽。

经典脉动阵列通常让 activation 横向传播、weight 纵向传播并由PE逐拍局部累加。当前 `rl_mac_array_8x8` 的物理实现是64路并行乘法与分级归约树，跨K tile的INT48累加位于GEMM engine；实际模块连接和流水级以 [03 RTL模块参考](03_rtl_module_reference.md) 为准。

大矩阵通过空间分块和时间复用映射到8×8阵列：A、B被划分为适合片上阵列的tile；计算一个输出tile `C(i,j)` 时，同一物理阵列依次处理相同 `p` 索引的 `A(i,p)` 与 `B(p,j)`，并跨多个K tile累加部分和。因此网络维度可以超过阵列物理尺寸，并不需要为每个tile实例化一套阵列。

### ELU

每拍并行处理 8 个 Q8.8 数。非负输入直通；负区间使用 `[-8,0]` 的查表和线性插值，`x<=-8` 饱和到 `-1.0`。

### NORM

八路逐特征执行 `(x-mean)*reciprocal`。当前 `fudan_policy.onnx` 不使用 NORM，但保留该算子用于兼容其他策略。

### CONCAT

把两个 Vector Cache 中的连续 INT16 向量拼接到一个目的区域。字段复用为：`src_base=input0`、`aux0_base=input1`、`dst_base=output`、`dim_k=count0`、`dim_n=count1`。引擎按元素重打包，因此支持当前 25+3 这类非 8 对齐拼接。

## 5. 当前网络映射

根目录 `fudan_policy.onnx` 的输入输出为：

- `obs[batch,25]`；
- `obs_history[batch,125]`；
- `actions[batch,6]`。

```text
encoder: obs_history(125) -> GEMM 128 -> ELU
                          -> GEMM  64 -> ELU
                          -> GEMM   3

actor:   CONCAT(obs 25, latent 3) -> 28
         -> GEMM 128 -> ELU
         -> GEMM  64 -> ELU
         -> GEMM  32 -> ELU
         -> GEMM   6
```

指令 RAM 保存 13 条命令：7×GEMM、5×ELU、1×CONCAT。总 MAC 数为：

```text
125*128 + 128*64 + 64*3 + 28*128 + 128*64 + 64*32 + 32*6 = 38,400
```

权威地址、维度和移位见 `generated/policy_map.json`。当前布局使用：

- Vector Cache：132/512 word；
- Weight Cache：每 bank 620/1280 word；
- observation base 0，history base 4，action base 130。

## 6. PS 工作流

启动阶段：

1. 初始化 AXI 基地址并检查版本 `0x00010002`；
2. 配置 GIC ID 61 的 level-high 完成中断；
3. 装载 132 个有效 Vector image word 和 8×620 个有效 Weight word；
4. 把 13 条描述符编码成 128-bit 指令，写入指令 RAM并逐条读回；
5. 可选地读回 Cache 并与编译进 ELF 的模型数据比较；
6. 运行固定输入 golden self-test。

每次推理：

1. 把 25 个 observation 和 125 个 history 量化为 Q8.8；
2. 通过 host window 写入 Vector Cache；
3. 写一次 `SEQUENCE_START`，以中断或轮询等待整张网络完成；
4. 从 action base 130 读出 6 个 Q8.8 结果；
5. 高级 API 除以 256 返回 float。

## 7. 容量与保护边界

所有常驻 GEMM 必须满足：

```text
sum(ceil(K_i/8) * ceil(N_i/8)) <= 1280
```

所有同时存活的输入、bias 和中间张量布局必须落在 512 个 Vector word 内。寄存器维度虽为 16-bit，PL 只检查零维度，并不会证明地址范围；导出器和 PS 调用者必须拒绝越界描述符。

## 8. 当前实现结果

2026-08-21 完整系统实现：

| 指标 | 结果 |
|---|---:|
| 频率 | 100 MHz |
| WNS / WHS | +0.400 ns / +0.026 ns |
| LUT / FF | 10,919 / 7,594 |
| RAMB36 / DSP48E1 | 52 / 72 |
| PL 自动序列完整策略 | 1,799 cycle = 17.99 us |
| 周期测试端到端平均 | 45.392 us |

72 个 DSP 中 64 个属于 GEMM，8 个属于仍保留的 NORM。详细实机证据见 [09 实机验证报告](09_hardware_test_20260821.md)。
