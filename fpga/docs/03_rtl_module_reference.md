# 03 RTL 模块端口与功能参考

## 1. 通用约定

RTL 位于 `rtl/`，主要使用 SystemVerilog。完整系统通过纯 Verilog 包装层 `rl_accel_axil_wrapper.v` 引入 Vivado 2020.2 Block Design。

通用约定如下：

- 时钟域：所有模块均工作在同一个 100 MHz `clk`/`s_axi_aclk` 时钟域；
- 复位：`reset_n` 和 `s_axi_aresetn` 均为低有效同步使用的复位输入；
- 数据：除明确说明外，16-bit 数值均为二进制补码有符号数；
- 向量 word：128 bit，8 lane，lane `k` 位于 `[k*16 +: 16]`；
- 权重总线：1024 bit，bank/输出 lane `j` 的 128-bit word 位于 `[j*128 +: 128]`；
- 累加总线：8×48 bit，输出 `j` 位于 `[j*48 +: 48]`；
- Cache 地址以 128-bit word 为单位，不是字节地址；
- `start` 仅在模块空闲时采样；`done` 为一个时钟周期脉冲；
- `i_valid`/`o_valid` 表示固定延迟流水线中的数据有效，不提供 backpressure。

模块层次如下：

```text
rl_accel_axil_wrapper
└─ rl_accel_axil_top
   ├─ rl_instruction_sequencer
   └─ rl_accel_core
      ├─ rl_vector_cache
      ├─ rl_weight_cache
      ├─ rl_gemm_engine
      │  └─ rl_mac_array_8x8
      ├─ rl_elu_engine
      │  └─ rl_elu_array8
      ├─ rl_norm_engine
      │  └─ rl_norm_array8
      └─ rl_concat_engine

独立算术原语：rl_round_shift_sat16
公共宏定义：rl_accel_params.vh
```

## 2. `rl_accel_params.vh`

该头文件集中定义 RTL 的基本宽度、Cache 深度和 opcode。

| 宏 | 值 | 含义 |
|---|---:|---|
| `RL_DATA_W` | 16 | activation/weight 数据宽度 |
| `RL_ACC_W` | 48 | GEMM 累加器宽度 |
| `RL_LANES` | 8 | 向量 lane 数和 MAC 行/列数 |
| `RL_VEC_W` | 128 | Vector/Weight 单 word 宽度 |
| `RL_WEIGHT_BUS_W` | 1024 | 8 bank 并行读总线宽度 |
| `RL_ACT_FRAC` | 8 | activation/bias Q8.8 小数位 |
| `RL_RAW_FRAC` | 6 | 兼容 NORM 的默认 raw 小数位；当前模型不使用 NORM |
| `RL_WEIGHT_FRAC` | 14 | 默认权重小数位；当前模型由描述符逐 GEMM 指定 shift |
| `RL_VECTOR_DEPTH` | 512 | Vector Cache word 数 |
| `RL_VECTOR_ADDR_W` | 9 | Vector Cache 地址宽度 |
| `RL_WEIGHT_DEPTH` | 1280 | 每个 Weight bank 的 word 数 |
| `RL_WEIGHT_ADDR_W` | 11 | Weight Cache 地址总线宽度 |
| `RL_INSTRUCTION_DEPTH` | 32 | 可常驻的 128-bit 指令数 |
| `RL_INSTRUCTION_ADDR_W` | 5 | 指令 RAM 地址宽度 |
| `RL_OP_GEMM` | 1 | GEMM opcode |
| `RL_OP_ELU` | 2 | ELU opcode |
| `RL_OP_NORM` | 3 | NORM opcode |
| `RL_OP_CONCAT` | 4 | CONCAT opcode |

修改这些宏会影响多个接口宽度和存储结构，不能只改单个模块。特别是物理 Weight Cache 深度为 1280，虽然地址总线可编码到 2047，软件仍必须限制有效地址。

## 3. `rl_round_shift_sat16`

### 3.1 功能

组合逻辑原语，对宽累加值做对称舍入、算术右移、加 bias 和 INT16 饱和。当前 GEMM engine 为提高频率内置了等价的分级流水逻辑，因此本模块主要作为可复用原语和单元测试对象。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `ACC_W` | 48 | 输入累加值宽度 |

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `i_value` | input | `ACC_W` | 有符号待量化累加值 |
| `i_bias` | input | 16 | 有符号 INT16 bias |
| `i_shift` | input | 6 | 算术右移位数 |
| `o_value` | output reg | 16 | 舍入、加 bias、饱和后的有符号结果 |

当 `i_shift>0` 时，正数先加 `2^(shift-1)`，负数先减同一数值，再进行算术右移，因此使用远离零的对称舍入。结果加符号扩展 bias 后限制在 `-32768..32767`。

## 4. `rl_mac_array_8x8`

完整的Cache读取、并行点积、跨K分块累加、重新量化和写回关系见 [GEMM引擎结构图](gemm_engine_architecture.svg)。

### 4.1 功能

8×8 INT16 并行点积阵列。每个有效输入周期接收 8 个 activation 和 64 个 weight，计算 8 个独立的 8 项行和。乘法器显式映射到 DSP48E1，归约树通过 `use_dsp="no"` 约束映射到 LUT/CARRY。

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 时钟 |
| `reset_n` | input | 1 | 低有效复位 |
| `i_valid` | input | 1 | 输入 activation/weight 有效 |
| `i_activations` | input | 128 | 8×INT16 activation |
| `i_weights` | input | 1024 | 8 bank×8 lane×INT16 weight |
| `o_valid` | output | 1 | `o_row_sums` 有效 |
| `o_row_sums` | output | 384 | 8×有符号 INT48 行和 |

权重 `i_weights[j*128 + k*16 +: 16]` 与 activation `i_activations[k*16 +: 16]` 相乘，输出 `j` 为所有 `k=0..7` 乘积之和。

流水级依次为：

1. 64 个 16×16 乘法；
2. 每行 4 个两项和；
3. 每行 2 个四项和；
4. 每行 1 个八项和。

`o_valid` 相对被接受的 `i_valid` 具有固定 4 拍标记延迟。流水线可逐拍接收数据，无 ready/backpressure。

## 5. `rl_norm_array8`

### 5.1 功能

八路标准化算术流水线。每 lane 先计算 17-bit `data-mean`，再乘以 E5M11 reciprocal 的 11-bit 正 mantissa，并根据 exponent 做对称舍入和算术移位，最后饱和为 INT16。

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 时钟 |
| `reset_n` | input | 1 | 低有效复位 |
| `i_valid` | input | 1 | 三组输入有效 |
| `i_data` | input | 128 | 8×原始观测 INT16 |
| `i_mean` | input | 128 | 8×同格式 mean INT16 |
| `i_inv_std` | input | 128 | 8×E5M11 reciprocal |
| `i_shift` | input | 6 | 从 exponent 中扣除的额外左移尺度 |
| `o_valid` | output | 1 | 输出有效 |
| `o_data` | output | 128 | 8×饱和 INT16 标准化结果 |

每个 reciprocal word 的 lane 内，`[15:11]` 是 exponent，`[10:0]` 是 mantissa。有效右移量为 `max(exponent-i_shift, 0)`。当前模型使用 `i_shift=0`。模块使用 8 个 DSP 乘法器，固定有效标记延迟为 4 拍。

## 6. `rl_elu_array8`

### 6.1 功能

八路 Q8.8 ELU，`alpha=1`。正数直接旁路；负数以绝对值生成 ROM 索引和插值余数。

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 时钟 |
| `reset_n` | input | 1 | 低有效复位 |
| `i_valid` | input | 1 | 输入有效 |
| `i_data` | input | 128 | 8×Q8.8 输入 |
| `o_valid` | output | 1 | 输出有效 |
| `o_data` | output | 128 | 8×Q8.8 ELU 输出 |

负数区间的 ROM 节点为 `round((exp(-index/32)-1)×256)`。输入幅值的 `[10:3]` 选择 1/32 间隔节点，`[2:0]` 在相邻节点间做 8 级线性插值，因此保留 Q8.8 输入的 1/256 步进。输入小于等于 -8.0 时输出固定为 -1.0，即 `16'sd-256`。固定有效标记延迟为 3 拍。

## 7. `rl_vector_cache`

### 7.1 功能

512×128-bit 真双口 Vector Cache。host 端口服务 AXI4-Lite 数据窗口，engine 端口服务算子。读操作为同步读，发出 enable 后下一拍产生 valid 和数据。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `DEPTH` | 512 | word 数 |
| `ADDR_W` | 9 | 地址宽度 |

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 公共时钟 |
| `host_wr_en` | input | 1 | host 写使能 |
| `host_wr_addr` | input | 9 | host 写 word 地址 |
| `host_wr_data` | input | 128 | host 写数据 |
| `host_rd_en` | input | 1 | host 读使能 |
| `host_rd_addr` | input | 9 | host 读 word 地址 |
| `host_rd_valid` | output | 1 | host 读数据有效 |
| `host_rd_data` | output | 128 | host 读数据 |
| `eng_en` | input | 1 | engine 端口操作使能 |
| `eng_we` | input | 1 | engine 写使能；0 表示读 |
| `eng_addr` | input | 9 | engine word 地址 |
| `eng_wr_data` | input | 128 | engine 写数据 |
| `eng_rd_valid` | output | 1 | engine 读数据有效 |
| `eng_rd_data` | output | 128 | engine 读数据 |

同地址双口冲突的最终 BRAM read-during-write 行为不作为软件可用特性。系统通过“busy 期间不发 host Cache 命令”的协议避免冲突。

## 8. `rl_weight_cache`

### 8.1 功能

8 bank Weight Cache。host 每次访问一个 128-bit bank word；engine 每次在同一地址读取全部 8 bank，输出 1024 bit。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `DEPTH` | 1280 | 每 bank word 数 |
| `ADDR_W` | 11 | 地址总线宽度 |

端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 公共时钟 |
| `host_wr_en` | input | 1 | host 写使能 |
| `host_bank` | input | 3 | host bank 选择 0..7 |
| `host_addr` | input | 11 | host word 地址 0..1279 |
| `host_wr_data` | input | 128 | host 写数据 |
| `host_rd_en` | input | 1 | host 读使能 |
| `host_rd_valid` | output | 1 | host 读有效 |
| `host_rd_data` | output | 128 | 所选 bank 读数据 |
| `eng_rd_en` | input | 1 | engine 并行读使能 |
| `eng_rd_addr` | input | 11 | engine 公共 word 地址 |
| `eng_rd_valid` | output | 1 | 1024-bit engine 数据有效 |
| `eng_rd_data` | output | 1024 | 8 bank 并行数据 |

每 bank 由 `mem_main[0:1023]` 和 `mem_tail[0:255]` 组成。读延迟为 1 拍。host 和 engine 的读取共用地址选择，若同拍都请求则 host 地址优先；集成协议要求 host 等待核心空闲。

## 9. `rl_gemm_engine`

### 9.1 功能

执行一个 `N×K` 全连接矩阵乘向量。引擎按 8 个输出一组，先读取对应 bias word，再遍历所有输入 group 和 weight word，将 MAC 行和累加到 8 个 INT48 累加器，最后完成量化并写回。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `VEC_ADDR_W` | 9 | Vector Cache 地址宽度 |
| `WT_ADDR_W` | 11 | Weight Cache 地址宽度 |

控制端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk` | input | 1 | 时钟 |
| `reset_n` | input | 1 | 低有效复位 |
| `start` | input | 1 | 空闲时启动命令 |
| `src_base` | input | 9 | 输入向量首 word |
| `dst_base` | input | 9 | 输出向量首 word |
| `bias_base` | input | 9 | bias 首 word |
| `weight_base` | input | 11 | 每 bank 权重首 word |
| `dim_k` | input | 16 | 输入元素数 |
| `dim_n` | input | 16 | 输出元素数 |
| `output_shift` | input | 6 | INT48 累加结果右移位数 |
| `busy` | output | 1 | 引擎正在执行 |
| `done` | output | 1 | 单拍完成脉冲 |

Vector Cache 接口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `vec_en` | output | 1 | 读/写请求 |
| `vec_we` | output | 1 | 写使能 |
| `vec_addr` | output | 9 | word 地址 |
| `vec_wr_data` | output | 128 | 写回输出 word |
| `vec_rd_valid` | input | 1 | 读数据有效 |
| `vec_rd_data` | input | 128 | 输入或 bias word |

Weight Cache 接口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `wt_rd_en` | output | 1 | 并行读请求 |
| `wt_rd_addr` | output | 11 | 公共 word 地址 |
| `wt_rd_valid` | input | 1 | 权重数据有效 |
| `wt_rd_data` | input | 1024 | 8 bank 权重 word |

关键状态包括 `BIAS_REQ/WAIT`、`STREAM`、`QUANT_ADJ`、`QUANT_SHIFT`、`QUANT_BIAS`、`QUANT_SAT` 和 `WRITE`。量化被拆为四级以满足 100 MHz 时序。末输出组中超出 `dim_n` 的 lane 被清零。

## 10. `rl_elu_engine`

### 10.1 功能

顺序处理 `element_count` 个 Vector Cache 元素，并允许源、目的地址相同。

控制端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk`, `reset_n` | input | 1 | 时钟、低有效复位 |
| `start` | input | 1 | 启动 |
| `src_base` | input | 9 | 输入首 word |
| `dst_base` | input | 9 | 输出首 word |
| `element_count` | input | 16 | 元素数 |
| `busy` | output | 1 | 执行中 |
| `done` | output | 1 | 单拍完成脉冲 |

Vector Cache 端口与 GEMM 的 `vec_*` 端口同名、同宽。状态机依次执行 `READ`、`WAIT`、`EXEC`、`RESULT`、`WRITE`。组数为 `ceil(element_count/8)`，末组无效 lane 写零。

## 11. `rl_norm_engine`

### 11.1 功能

顺序读取 source、mean、reciprocal 三个向量区域，对 `element_count` 个元素做标准化。

控制端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk`, `reset_n` | input | 1 | 时钟、低有效复位 |
| `start` | input | 1 | 启动 |
| `src_base` | input | 9 | 输入首 word |
| `dst_base` | input | 9 | 输出首 word |
| `mean_base` | input | 9 | mean 首 word |
| `inv_std_base` | input | 9 | E5M11 reciprocal 首 word |
| `element_count` | input | 16 | 元素数 |
| `output_shift` | input | 6 | 额外尺度移位 |
| `busy` | output | 1 | 执行中 |
| `done` | output | 1 | 单拍完成脉冲 |

Vector Cache 端口与 GEMM 的 `vec_*` 端口同名、同宽。状态机包括 `SRC_REQ/WAIT`、`MEAN_REQ/WAIT`、`INV_REQ/WAIT`、`EXEC`、`RESULT`、`WRITE`。末组无效 lane 写零。

## 12. `rl_concat_engine`

`rl_concat_engine` 把两个 Vector Cache 连续区域按 INT16 元素拼接到目的区域。它逐元素读取并维护一个 128-bit 输出打包寄存器，所以第一输入长度不要求是 8 的倍数；完成或凑满 8 lane 时写回一个 word。

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk`, `reset_n` | input | 1 | 时钟、低有效复位 |
| `start` | input | 1 | 启动 |
| `src0_base` | input | 9 | 第一输入首 word |
| `src1_base` | input | 9 | 第二输入首 word |
| `dst_base` | input | 9 | 输出首 word |
| `count0` | input | 16 | 第一输入元素数 |
| `count1` | input | 16 | 第二输入元素数 |
| `busy`, `done` | output | 1 | 执行中、单拍完成 |

Vector Cache 接口与其他向量引擎相同。状态机按全局输出元素索引选择 source，经历读请求/等待、lane 提取、打包和写回。当前策略用该模块实现 `CONCAT(obs[25], latent[3])`。

## 13. `rl_instruction_sequencer`

### 13.1 功能

`rl_instruction_sequencer` 集成 32×128-bit distributed RAM 和四状态控制器。PS 在初始化阶段通过 host 端口写入并读回网络程序；推理时只给出指令数并产生一次 `start`。序列器依次执行 `FETCH`、`DISPATCH`、`WAIT`，等待当前核心命令完成后再取下一条，整网结束或任一命令出错时只产生一个 `done`。

host 和序列控制端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk`, `reset_n` | input | 1 | 时钟、低有效复位 |
| `host_wr_en` | input | 1 | 指令写使能 |
| `host_addr` | input | 5 | 指令地址 0..31 |
| `host_wr_data` | input | 128 | 待写描述符 |
| `host_rd_en` | input | 1 | 指令读使能 |
| `host_rd_valid` | output | 1 | 读回有效 |
| `host_rd_data` | output | 128 | 读回描述符 |
| `start` | input | 1 | 启动整条程序 |
| `instruction_count` | input | 6 | 有效指令数 1..32 |
| `busy` | output | 1 | 整条程序执行中 |
| `done` | output | 1 | 程序结束单拍脉冲 |
| `error` | output | 1 | 程序长度、描述符或核心执行错误 |
| `current_index` | output | 5 | 当前指令索引 |

核心命令输出与 `rl_accel_core` 的 `cmd_*` 输入同名、同宽，另接收 `core_busy/core_done/core_error`。指令位域为：

```text
word0: opcode[2:0], src[11:3], dst[20:12], aux0[29:21]
word1: aux1[8:0], weight[19:9], shift[25:20]
word2: dim_k[15:0], dim_n[31:16]
word3: reserved = 0
```

序列器检查长度、opcode 和必要的非零维度，但不证明 Cache 地址范围。运行期间 host 不应修改指令 RAM。当前 13 条网络程序相对纯算子执行仅增加 27 cycle，即 0.27 us。

## 14. `rl_accel_core`

### 14.1 功能

核心模块集成两个 Cache 和四个算子引擎，锁存一条命令，校验 opcode/维度，并把 engine 端口复用到当前活动算子。host Cache 端口直接连接两个 Cache 的独立端口。

命令端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `clk`, `reset_n` | input | 1 | 时钟、低有效复位 |
| `cmd_start` | input | 1 | 空闲时提交命令 |
| `cmd_opcode` | input | 3 | 1=GEMM，2=ELU，3=NORM，4=CONCAT |
| `cmd_src_base` | input | 9 | 源 Vector word |
| `cmd_dst_base` | input | 9 | 目的 Vector word |
| `cmd_aux0_base` | input | 9 | bias、mean 或 CONCAT 第二输入 |
| `cmd_aux1_base` | input | 9 | reciprocal |
| `cmd_weight_base` | input | 11 | Weight word |
| `cmd_dim_k` | input | 16 | GEMM K 或 CONCAT 第一输入元素数 |
| `cmd_dim_n` | input | 16 | 输出/元素数或 CONCAT 第二输入元素数 |
| `cmd_shift` | input | 6 | 输出移位 |
| `busy` | output | 1 | 核心忙 |
| `done` | output | 1 | 单拍完成或拒绝脉冲 |
| `error` | output | 1 | 当前命令错误，保持到下一条命令 |

host Vector Cache 端口：

| 端口 | 方向 | 宽度 |
|---|---|---:|
| `host_vec_wr_en` | input | 1 |
| `host_vec_addr` | input | 9 |
| `host_vec_wr_data` | input | 128 |
| `host_vec_rd_en` | input | 1 |
| `host_vec_rd_valid` | output | 1 |
| `host_vec_rd_data` | output | 128 |

host Weight Cache 端口：

| 端口 | 方向 | 宽度 |
|---|---|---:|
| `host_wt_wr_en` | input | 1 |
| `host_wt_bank` | input | 3 |
| `host_wt_addr` | input | 11 |
| `host_wt_wr_data` | input | 128 |
| `host_wt_rd_en` | input | 1 |
| `host_wt_rd_valid` | output | 1 |
| `host_wt_rd_data` | output | 128 |

GEMM 要求 `dim_k!=0` 且 `dim_n!=0`；CONCAT 要求两路计数之和非零；ELU/NORM 要求 `dim_n!=0`。未知 opcode 或非法零维度会直接给出 `done=1,error=1`。核心不进行完整的地址范围证明，描述符生成器和驱动调用方必须保证 `base + ceil(dim/8)` 以及权重二维展开不越界。

## 15. `rl_accel_axil_top`

### 15.1 功能

32-bit AXI4-Lite 从设备，提供单命令调试寄存器、指令程序控制、状态、中断以及三个 128-bit 数据窗口。接口版本固定为 `0x00010002`。顶层在手动单命令和自动序列之间仲裁核心命令；自动序列运行期间，全局 `busy` 持续有效，整网完成后只锁存一次 `done` 并产生一次 IRQ。

参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `AXI_ADDR_W` | 12 | 4 KiB 从设备地址宽度 |
| `AXI_DATA_W` | 32 | AXI 数据宽度 |

AXI 写通道端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `s_axi_awaddr` | input | 12 | 写地址 |
| `s_axi_awvalid` | input | 1 | 写地址有效 |
| `s_axi_awready` | output | 1 | 写地址接受 |
| `s_axi_wdata` | input | 32 | 写数据 |
| `s_axi_wstrb` | input | 4 | 字节写使能 |
| `s_axi_wvalid` | input | 1 | 写数据有效 |
| `s_axi_wready` | output | 1 | 写数据接受 |
| `s_axi_bresp` | output | 2 | 写响应 |
| `s_axi_bvalid` | output | 1 | 写响应有效 |
| `s_axi_bready` | input | 1 | master 接受响应 |

AXI 读通道端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `s_axi_araddr` | input | 12 | 读地址 |
| `s_axi_arvalid` | input | 1 | 读地址有效 |
| `s_axi_arready` | output | 1 | 读地址接受 |
| `s_axi_rdata` | output | 32 | 读数据 |
| `s_axi_rresp` | output | 2 | 读响应 |
| `s_axi_rvalid` | output | 1 | 读数据有效 |
| `s_axi_rready` | input | 1 | master 接受数据 |

公共端口：

| 端口 | 方向 | 宽度 | 含义 |
|---|---|---:|---|
| `s_axi_aclk` | input | 1 | AXI/核心公共时钟 |
| `s_axi_aresetn` | input | 1 | 低有效复位 |
| `irq` | output | 1 | `irq_enable && done_sticky`，高电平有效 |

AW 和 W 可独立到达，模块分别锁存后执行一次写事务，并按 `WSTRB` 合并字节。核心或序列器的单拍 done 被锁存为 STATUS.done；软件写 CONTROL.clear_done 后清除。Vector、Weight 和 Instruction 窗口的 128-bit 读数据均先进入 readback latch，再由四个 32-bit 寄存器读取。完整偏移和位定义见 [04 寄存器映射](04_register_map.md)。

顶层在核心忙时忽略 Vector/Weight Cache command，软件应先检查 busy。由于 BRAM 为同步读，Cache read command 后驱动会插入两个 STATUS 读取作为总线/核心延迟，再读取 readback 寄存器。

## 16. `rl_accel_axil_wrapper`

Vivado 2020.2 Block Design 的 Module Reference 对 SystemVerilog 顶层支持受限，因此提供同名宽度固定的纯 Verilog包装层。它不改变协议和时序，只把完整 AXI4-Lite 端口与 `irq` 一一连接到 `rl_accel_axil_top`。

在 Block Design 中应实例化该 wrapper，而独立 RTL 仿真可以直接实例化 `rl_accel_axil_top` 或 `rl_accel_core`。

## 17. 集成注意事项

- 所有算子共享一个 engine Cache 端口，`rl_accel_core` 保证任一时刻只选中一个引擎；
- host Cache 物理端口虽独立，AXI 协议仍禁止 busy 时发 Cache command；
- 手动 start、序列 start、Cache/Instruction command 和 clear_done 均是写寄存器产生的事件，不应假定对应控制位持续保持；
- done sticky 必须在提交下一条手动命令或自动序列前清除，官方驱动会自动处理；
- irq 为电平信号，应在 ISR 中读状态并清 done，否则 GIC 会持续看到中断有效；
- 修改流水线级数时必须同步调整 valid 管线并重跑原语、引擎和端到端回归；
- 修改 Cache 深度或 lane 数时，需同步修改 RTL 参数、寄存器掩码、导出工具、PS 数据数组和验证向量。
