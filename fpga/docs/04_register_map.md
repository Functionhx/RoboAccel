# 04 PL 控制接口

顶层 `rl_accel_axil_top` 提供一个 32-bit AXI4-Lite 从接口、一个电平型中断输出和 100 MHz 单时钟域。所有地址均为相对于 IP 基地址的字节偏移。

自动程序的描述符语义、装载/读回时序、sequencer与core握手、错误和并发访问规则集中定义在 [10 自动指令序列协议](10_instruction_sequence_protocol.md)。

## 命令寄存器

| 偏移 | 名称 | 位域 | 说明 |
|---:|---|---|---|
| `0x000` | CONTROL | bit0 START；bit1 CLEAR_DONE | START 在空闲时产生单周期启动脉冲；CLEAR_DONE 清除完成状态 |
| `0x004` | STATUS | bit0 BUSY；bit1 DONE；bit2 ERROR | DONE 为粘滞位；非法 opcode 或零维度置 ERROR |
| `0x008` | OPCODE | `[2:0]` | 1=GEMM，2=ELU，3=NORM，4=CONCAT |
| `0x00C` | SRC_BASE | `[8:0]` | 输入向量 Cache 的 128-bit word 地址 |
| `0x010` | DST_BASE | `[8:0]` | 输出向量 Cache word 地址 |
| `0x014` | AUX0_BASE | `[8:0]` | GEMM bias、NORM mean 或 CONCAT 第二输入地址 |
| `0x018` | AUX1_BASE | `[8:0]` | NORM reciprocal 地址 |
| `0x01C` | WEIGHT_BASE | `[10:0]` | GEMM 每个权重 bank 的起始 word 地址 |
| `0x020` | DIM_K | `[15:0]` | GEMM 输入维度；CONCAT 第一输入元素数 |
| `0x024` | DIM_N | `[15:0]` | GEMM 输出数、ELU/NORM 元素数或 CONCAT 第二输入元素数 |
| `0x028` | OUTPUT_SHIFT | `[5:0]` | GEMM 的 INT48 重量定点移位；NORM 的公共附加缩放 |
| `0x02C` | IRQ_ENABLE | bit0 | 使能 `irq = DONE` |
| `0x0FC` | VERSION | `[31:0]` | 当前为 `0x00010002` |

START 时描述符会锁存，软件可在 BUSY 期间准备下一条描述符，但不能再次启动。软件负责保证地址范围有效且源/目标区域不会发生未经规划的覆盖。

CONCAT 的字段语义为：

| 字段 | 含义 |
|---|---|
| SRC_BASE | 第一输入首 word |
| AUX0_BASE | 第二输入首 word |
| DST_BASE | 输出首 word |
| DIM_K | 第一输入 INT16 元素数 |
| DIM_N | 第二输入 INT16 元素数 |

CONCAT 会重新打包非 8 对齐边界；当前网络使用 25+3。两路计数之和必须非零，允许其中一路为空。

## Vector Cache 窗口

Vector Cache 共 512 个 128-bit word，每个 word 按 lane0 到 lane7 存放 8 个 little-endian INT16。

| 偏移 | 说明 |
|---:|---|
| `0x040` | word 地址 `[8:0]` |
| `0x044..0x050` | 128-bit 写暂存区，低 32 bit 在 `0x044` |
| `0x054` | bit0 提交写；bit1 发起读。BUSY 时请求被忽略 |
| `0x058..0x064` | 128-bit 读回锁存区，低 32 bit 在 `0x058` |

读命令后需等待至少两个 PL 时钟再读取锁存区。AXI 软件通常用一次状态读取或短轮询隔开即可。

## Weight Cache 窗口

权重 Cache 有 8 个 bank，每个 bank 为 1280×128 bit。8 个 bank 同地址并行读出 1024 bit，向 8×8 MAC 阵列提供 64 个 INT16 权重。

| 偏移 | 说明 |
|---:|---|
| `0x080` | bank=`[18:16]`，word=`[10:0]` |
| `0x084..0x090` | 128-bit 写暂存区 |
| `0x094` | bit0 提交写；bit1 发起读。BUSY 时请求被忽略 |
| `0x098..0x0A4` | 128-bit 读回锁存区 |

对应的 C 常量和描述符结构在 `sw/rl_accel_regs.h`。

## 指令 RAM 与自动序列

指令 RAM 为 32×128 bit。与 Cache 窗口相同，软件先写四个暂存寄存器，再提交写命令；推理进行时的指令窗口请求会被忽略。

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x0B0` | INSTR_ADDR | 指令地址 `[4:0]` |
| `0x0B4..0x0C0` | INSTR_STAGE0..3 | 128-bit 写暂存区，低 32 bit 在 `0x0B4` |
| `0x0C4` | INSTR_COMMAND | bit0 提交写；bit1 发起读 |
| `0x0C8..0x0D4` | INSTR_READ0..3 | 128-bit 读回锁存区 |
| `0x0D8` | SEQUENCE_LENGTH | `[5:0]`，有效范围 1..32 |
| `0x0DC` | SEQUENCE_CONTROL | bit0 启动完整序列 |
| `0x0E0` | SEQUENCE_STATUS | bit0 busy；bit1 error；`[12:8]` 当前指令索引 |

128-bit 指令格式：

| 32-bit word | 位域 |
|---|---|
| word0 | opcode `[2:0]`，src `[11:3]`，dst `[20:12]`，aux0 `[29:21]` |
| word1 | aux1 `[8:0]`，weight `[19:9]`，shift `[25:20]` |
| word2 | dim_k `[15:0]`，dim_n `[31:16]` |
| word3 | 保留，必须写零 |

写 `SEQUENCE_CONTROL.START` 后，顶层 `STATUS.BUSY` 在整条序列期间保持为 1。任何非法长度、opcode、必要零维度或核心错误都会中止剩余指令，并在整网结束时只产生一次 DONE/IRQ。原有单算子寄存器仍保留用于调试和兼容，但正常推理使用自动序列接口。
