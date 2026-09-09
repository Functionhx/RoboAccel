# 10 自动指令序列协议

本文定义软件装载和启动PL自动指令序列时必须遵守的寄存器、描述符、握手、状态和并发访问规则。目标读者包括PS裸机驱动开发者、模型导出器开发者、RTL验证人员，以及希望绕过现有驱动直接控制加速器的上位软件开发者。

本文以PL接口版本 `0x00010002` 为准。AXI4-Lite基地址当前为 `0x43C00000`；表内地址均为相对于基地址的字节偏移。寄存器常量的权威源码是 `sw/rl_accel_regs.h`，RTL行为的权威源码是 `rtl/rl_accel_axil_top.sv` 和 `rtl/rl_instruction_sequencer.sv`。

128-bit描述符的word划分、精确位域、端序和各opcode字段语义见 [自动序列机指令帧格式图](instruction_sequence_protocol.svg)。

## 1. 协议定位

自动序列接口把一张网络表示为一组线性排列的128-bit算子描述符。PS在初始化阶段装载程序、权重、bias和其他常量；每次推理只写输入、发出一次序列启动，然后等待整网完成。

```text
PS / 模型导出器
  │
  │ 32-bit AXI4-Lite
  ▼
Instruction host window ──> 32×128-bit Instruction RAM
                                  │
                         start + instruction_count
                                  ▼
                       rl_instruction_sequencer
                    FETCH → DISPATCH → WAIT → ...
                                  │ 一次一条cmd_*
                                  ▼
                           rl_accel_core
                   GEMM / ELU / NORM / CONCAT
```

本协议是静态线性程序协议，不提供跳转、循环、条件分支、异常处理程序或算子并发。网络中的数据依赖由Vector Cache地址建立，而不是由指令中的显式依赖字段建立。

## 2. 容量与地址单位

| 项目 | 当前值 | 协议含义 |
|---|---:|---|
| Instruction RAM | 32×128-bit | 最多保存32条描述符 |
| 程序长度 | 1..32 | `SEQUENCE_LENGTH` 的合法范围 |
| Vector地址 | 9 bit | 单位为128-bit word；合法索引0..511 |
| Weight地址 | 11 bit | 每个bank内以128-bit word计；物理深度0..1279 |
| Weight bank | 8 | GEMM同地址并行读8个bank |
| 一个Vector word | 128 bit | 8个little-endian signed INT16 lane |

描述符字段能编码的数值范围不等于物理存储深度。例如11-bit Weight地址可以编码到2047，但当前物理bank深度只有1280。PL只检查opcode和必要的非零维度，不完整检查Cache末地址，因此导出器和驱动必须在启动前证明所有访问都落在物理容量内。

## 3. 128-bit描述符格式

一条描述符由4个little-endian 32-bit word组成。`data[0]`写入最低32 bit，`data[3]`写入最高32 bit。

```text
bit 127                                                        bit 0
┌─────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ word3           │ word2                        │ word1                        │ word0                        │
│ reserved = 0    │ dim_n[31:16], dim_k[15:0]   │ shift/weight/aux1            │ aux0/dst/src/opcode           │
└─────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### 3.1 位域定义

| 128-bit绝对位 | 所属word位 | 名称 | 位宽 | 含义 |
|---:|---:|---|---:|---|
| `[2:0]` | word0 `[2:0]` | `opcode` | 3 | 算子编号 |
| `[11:3]` | word0 `[11:3]` | `src_base` | 9 | 第一输入Vector首word |
| `[20:12]` | word0 `[20:12]` | `dst_base` | 9 | 输出Vector首word |
| `[29:21]` | word0 `[29:21]` | `aux0_base` | 9 | bias、mean或第二输入 |
| `[31:30]` | word0 `[31:30]` | reserved | 2 | 写零 |
| `[40:32]` | word1 `[8:0]` | `aux1_base` | 9 | NORM reciprocal首word |
| `[51:41]` | word1 `[19:9]` | `weight_base` | 11 | GEMM每个Weight bank首word |
| `[57:52]` | word1 `[25:20]` | `shift` | 6 | GEMM/NORM缩放移位 |
| `[63:58]` | word1 `[31:26]` | reserved | 6 | 写零 |
| `[79:64]` | word2 `[15:0]` | `dim_k` | 16 | GEMM K或CONCAT第一路元素数 |
| `[95:80]` | word2 `[31:16]` | `dim_n` | 16 | 输出/元素数或CONCAT第二路元素数 |
| `[127:96]` | word3 `[31:0]` | reserved | 32 | 必须写零 |

当前RTL不拒绝非零reserved位，但兼容实现必须写零，避免未来版本把这些位定义为新字段。

### 3.2 C语言打包公式

```c
data[0] = (opcode & 0x7u)
        | ((src_base  & 0x1ffu) << 3)
        | ((dst_base  & 0x1ffu) << 12)
        | ((aux0_base & 0x1ffu) << 21);

data[1] = (aux1_base & 0x1ffu)
        | ((weight_base & 0x7ffu) << 9)
        | ((output_shift & 0x3fu) << 20);

data[2] = (dim_k & 0xffffu)
        | ((dim_n & 0xffffu) << 16);

data[3] = 0u;
```

现有驱动函数 `rl_accel_pack_instruction()` 已实现这套编码，应用程序不应重复维护另一份位移常量。

## 4. Opcode与字段语义

| opcode | 名称 | `src_base` | `dst_base` | `aux0_base` | `aux1_base` | `weight_base` | `dim_k` | `dim_n` | `shift` |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | GEMM | activation | output | bias | 未使用 | 权重首word | K | N | 权重小数位/输出移位 |
| 2 | ELU | input | output | 未使用 | 未使用 | 未使用 | 未使用 | 元素数 | 未使用 |
| 3 | NORM | input | output | mean | reciprocal | 未使用 | 未使用 | 元素数 | 公共附加缩放 |
| 4 | CONCAT | input0 | output | input1 | 未使用 | 未使用 | input0元素数 | input1元素数 | 未使用 |

必要维度规则：

- GEMM：`dim_k != 0 && dim_n != 0`；
- ELU：`dim_n != 0`；
- NORM：`dim_n != 0`；
- CONCAT：`dim_k != 0 || dim_n != 0`；
- 其他opcode非法。

网络边由地址相等关系表达。例如某GEMM的 `dst_base=20`，后一条ELU的 `src_base=20`，表示ELU消费前一条GEMM的结果。允许算子按自身语义进行原址操作；当前ELU程序通常使用 `src_base == dst_base`。

## 5. Instruction RAM host window

### 5.1 寄存器

| 偏移 | 名称 | 访问 | 含义 |
|---:|---|---|---|
| `0x0B0` | `INSTR_ADDR` | R/W | 指令索引 `[4:0]` |
| `0x0B4` | `INSTR_STAGE0` | R/W | 待写描述符word0 |
| `0x0B8` | `INSTR_STAGE1` | R/W | 待写描述符word1 |
| `0x0BC` | `INSTR_STAGE2` | R/W | 待写描述符word2 |
| `0x0C0` | `INSTR_STAGE3` | R/W | 待写描述符word3 |
| `0x0C4` | `INSTR_COMMAND` | W | bit0=COMMIT，bit1=READ |
| `0x0C8` | `INSTR_READ0` | R | 读回word0 |
| `0x0CC` | `INSTR_READ1` | R | 读回word1 |
| `0x0D0` | `INSTR_READ2` | R | 读回word2 |
| `0x0D4` | `INSTR_READ3` | R | 读回word3 |

AXI4-Lite数据宽度为32 bit，因此128-bit描述符必须经4个stage寄存器组装。Instruction RAM不是直接线性映射到PS地址空间的普通内存。

### 5.2 写入一条指令

软件必须按以下顺序执行：

```text
确认 STATUS.BUSY == 0
write INSTR_ADDR   = index
write INSTR_STAGE0 = data[0]
write INSTR_STAGE1 = data[1]
write INSTR_STAGE2 = data[2]
write INSTR_STAGE3 = data[3]
write INSTR_COMMAND = 0x1     // COMMIT
```

`COMMIT`在PL中生成一个单时钟写脉冲，将 `{stage3,stage2,stage1,stage0}` 写入 `instruction_mem[index]`。若加速器忙，顶层仍会正常返回AXI写响应，但不会产生Instruction RAM写脉冲，因此驱动必须在操作前检查BUSY，不能把AXI `OKAY` 当作RAM已经更新的证明。

### 5.3 读回一条指令

```text
确认 STATUS.BUSY == 0
write INSTR_ADDR = index
write INSTR_COMMAND = 0x2     // READ
等待Instruction RAM同步读延迟
read INSTR_READ0..3
```

当前裸机驱动在提交READ后读取两次 `STATUS`，再读取4个readback寄存器，以覆盖PL同步RAM和读回锁存延迟。若自行实现驱动，应采用等价延迟或轮询机制，不能在READ写操作后立即假设readback已经更新。

### 5.4 程序整体装载

推荐流程：

```text
for index = 0 .. count-1:
    编码descriptor[index]
    写入Instruction RAM[index]
    可选：立即读回并逐32-bit比较

write SEQUENCE_LENGTH = count
```

只有所有指令写入和可选读回验证成功后，才更新 `SEQUENCE_LENGTH`。若装载中途失败，RAM可能部分更新；软件必须重新完整装载，不应继续执行。旧程序中 `index >= count` 的内容无需清零，因为序列器不会取出这些位置。

## 6. 自动序列控制寄存器

| 偏移 | 名称 | 位域 | 含义 |
|---:|---|---|---|
| `0x000` | `CONTROL` | bit1 `CLEAR_DONE` | 清除顶层粘滞DONE和ERROR |
| `0x004` | `STATUS` | bit0 BUSY；bit1 DONE；bit2 ERROR | 整个加速器统一状态 |
| `0x02C` | `IRQ_ENABLE` | bit0 | 允许 `irq = DONE` |
| `0x0D8` | `SEQUENCE_LENGTH` | `[5:0]` | 有效指令数1..32 |
| `0x0DC` | `SEQUENCE_CONTROL` | bit0 START | 产生一次整网启动脉冲 |
| `0x0E0` | `SEQUENCE_STATUS` | bit0 busy；bit1 error；`[12:8]` index | 序列器局部状态 |
| `0x0FC` | `VERSION` | `[31:0]` | 当前 `0x00010002` |

### 6.1 启动顺序

```text
确认 VERSION == 0x00010002
确认 STATUS.BUSY == 0
确认程序、权重、bias、输入均已装载
write CONTROL = CLEAR_DONE
write SEQUENCE_CONTROL = START
等待IRQ或轮询STATUS.DONE
检查STATUS.ERROR
```

写START时如果 `STATUS.BUSY=1`，顶层会忽略启动请求。START不是排队命令，不会在当前工作完成后自动补执行。

`SEQUENCE_LENGTH`在序列器接受START时被锁存，当前程序执行期间修改该寄存器不会改变已经锁存的指令数，但协议禁止在BUSY期间修改程序控制寄存器。

### 6.2 完成与中断

序列器自身的 `done` 是一个PL时钟周期脉冲；AXI顶层将其转换为 `STATUS.DONE` 粘滞位。若 `IRQ_ENABLE.bit0=1`，中断输出为：

```text
irq = irq_enable && done_sticky
```

因此IRQ为level-high，不是单周期脉冲。ISR或轮询完成处理必须最终写 `CONTROL.CLEAR_DONE`，否则IRQ会持续保持高电平。整条网络无论成功还是因错误中止，都只产生一次顶层DONE/IRQ。

`SEQUENCE_STATUS`不包含done位；软件判断整网完成应读取统一的 `STATUS.DONE` 或等待IRQ。

## 7. 序列器与核心握手

内部状态机有4个状态：

```text
S_IDLE
  └─ START且长度合法 → S_FETCH

S_FETCH
  └─ current_descriptor = instruction_mem[current_index]
     → S_DISPATCH

S_DISPATCH
  ├─ 描述符非法 → ERROR + DONE，返回S_IDLE
  └─ core_busy == 0 → cmd_start单拍，进入S_WAIT

S_WAIT
  └─ 等待core_done
       ├─ core_error → ERROR + DONE，返回S_IDLE
       ├─ 最后一条 → DONE，返回S_IDLE
       └─ 否则index++，回到S_FETCH
```

`cmd_start`只保持一个周期。`rl_accel_core`在该周期锁存opcode、地址、维度和shift，并启动相应算子。序列器在收到 `core_done` 前不会取下一条指令，因此算子严格顺序执行，任意时刻最多一条命令在途。

AXI顶层还保留单算子调试寄存器路径。自动序列执行时，`sequence_busy`选择序列器产生的 `cmd_*` 字段；正常整网推理不得同时使用手动START。

## 8. 错误语义

| 错误条件 | 发现位置 | 行为 |
|---|---|---|
| `SEQUENCE_LENGTH == 0` | sequencer | 不执行指令，ERROR+DONE |
| `SEQUENCE_LENGTH > 32` | sequencer | 不执行指令，ERROR+DONE |
| 未知opcode | sequencer/core | 当前序列中止，ERROR+DONE |
| 必要维度为0 | sequencer/core | 当前序列中止，ERROR+DONE |
| core返回错误 | sequencer | 不再执行后续指令，ERROR+DONE |
| Cache地址越界 | 不完整检查 | 行为不受协议保证，必须由软件预防 |
| BUSY期间host COMMIT/READ | AXI顶层 | 请求被忽略，但AXI事务仍可返回OKAY |
| BUSY期间再次START | AXI顶层 | 请求被忽略，不排队 |

`SEQUENCE_STATUS.error`反映序列器本次/最近一次启动的错误状态；`STATUS.ERROR`是由顶层在完成时锁存的统一错误位。开始新序列时两者会按各自逻辑清除，软件仍应在每次启动前显式清除旧的DONE/ERROR。

## 9. 并发访问与一致性规则

为避免host与活动算子争用存储器，必须遵守：

1. 只在 `STATUS.BUSY==0` 时读写Vector Cache、Weight Cache和Instruction RAM；
2. 启动前完成全部静态参数、程序和输入写入；
3. BUSY期间不要修改Instruction RAM、程序长度或描述符；
4. BUSY期间不要使用手动单算子START；
5. 等到统一DONE/IRQ后再读取输出；
6. 装载程序后推荐读回验证，特别是在首次启动、模型更新和板级验证中；
7. 自动序列期间不得假设PS可以观察或修改GEMM内部累加器、算子状态机或Cache engine端口。

Instruction RAM内容没有协议规定的上电默认值，复位后必须先装载合法程序并设置长度，再允许启动。

## 10. PS驱动API映射

现有裸机驱动已经封装协议细节：

| API | 协议动作 |
|---|---|
| `rl_accel_pack_instruction()` | 校验结构体并编码4个32-bit word |
| `rl_accel_instruction_write()` | 通过stage+COMMIT写一条指令 |
| `rl_accel_instruction_read()` | 发起同步读并返回4个word |
| `rl_accel_load_program()` | 批量装载、可选读回、最后设置长度 |
| `rl_accel_execute_program()` | 清DONE、写一次START、等待整网完成 |

推荐应用代码：

```c
rl_accel_t accel;

rl_accel_init(&accel, XPAR_RL_ACCEL_0_BASEADDR, use_interrupt);

/* 初始化阶段，只执行一次。 */
rl_accel_load_program(&accel, commands, command_count, 1);

/* 每次推理：先写输入，再启动整网。 */
write_inputs_to_vector_cache(&accel);
if (rl_accel_execute_program(&accel, RL_ACCEL_TIMEOUT) != XST_SUCCESS) {
    handle_accelerator_error();
}
read_actions_from_vector_cache(&accel);
```

真实API的参数、返回值和中断用法见 [06 PS裸机API参考](06_api_reference.md)。

## 11. 当前模型的13条程序

以下内容来自 `generated/policy_map.json`，用于说明描述符如何表达当前网络。地址单位均为128-bit word。

| index | opcode | src | dst | aux0 | weight | K | N | shift | 网络含义 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | GEMM | 4 | 20 | 36 | 0 | 125 | 128 | 11 | history编码器第一层 |
| 1 | ELU | 20 | 20 | 0 | 0 | 0 | 128 | 0 | 原址激活 |
| 2 | GEMM | 20 | 52 | 60 | 256 | 128 | 64 | 12 | history编码器第二层 |
| 3 | ELU | 52 | 52 | 0 | 0 | 0 | 64 | 0 | 原址激活 |
| 4 | GEMM | 52 | 68 | 69 | 384 | 64 | 3 | 14 | 生成latent[3] |
| 5 | CONCAT | 0 | 70 | 68 | 0 | 25 | 3 | 0 | `concat(obs, latent)` |
| 6 | GEMM | 70 | 74 | 90 | 392 | 28 | 128 | 13 | actor第一层 |
| 7 | ELU | 74 | 74 | 0 | 0 | 0 | 128 | 0 | 原址激活 |
| 8 | GEMM | 74 | 106 | 114 | 456 | 128 | 64 | 14 | actor第二层 |
| 9 | ELU | 106 | 106 | 0 | 0 | 0 | 64 | 0 | 原址激活 |
| 10 | GEMM | 106 | 122 | 126 | 584 | 64 | 32 | 14 | actor第三层 |
| 11 | ELU | 122 | 122 | 0 | 0 | 0 | 32 | 0 | 原址激活 |
| 12 | GEMM | 122 | 130 | 131 | 616 | 32 | 6 | 14 | 输出actions[6] |

这里最重要的拓扑关系是前一条指令的 `dst` 成为后一条指令的 `src`。CONCAT通过 `src=0` 和 `aux0=68`读取两路输入，并把28个元素重新打包到 `dst=70`。

## 12. 可编程范围与限制

不修改bitstream即可改变：

- 支持算子的排列和层数；
- GEMM K/N维度；
- Vector/Weight地址；
- 每层重新量化shift；
- CONCAT两路长度；
- 原址或非原址Buffer规划。

仍然固定在RTL中的内容：

- opcode集合只有GEMM、ELU、NORM和CONCAT；
- 线性程序最多32条；
- 没有跳转、循环和分支；
- 一次只执行一条算子；
- 8×8 INT16 MAC、INT48累加和Q8.8激活格式；
- 当前Cache容量和bank结构；
- 当前运行时为batch=1；
- PL没有DDR master或DMA。

## 13. 最低验证要求

修改本协议、描述符格式或sequencer后，至少执行：

```powershell
make -C tb all
```

并重点确认：

1. Instruction RAM逐条写入和读回完全一致；
2. 非法长度、opcode和零维度能产生ERROR+DONE；
3. 当前13条程序完整执行且输出逐点匹配Python golden；
4. 序列执行期间host窗口请求不会破坏Cache或程序；
5. 整网只产生一次DONE/IRQ；
6. PS驱动装载时启用读回验证；
7. 实机至少完成一次中断和一次轮询执行；
8. 接口格式改变时同步更新VERSION、`sw/rl_accel_regs.h`、PS驱动、RTL testbench和所有协议文档。

当前自动策略序列的权威验证结果为1,799个100 MHz周期，即17.99 us；相对纯算子周期合计只增加27 cycle。性能数字应追溯到当前RTL testbench和实机原始日志，不应仅引用本文。

## 14. 关联资料

- [02 整体系统架构](02_system_architecture.md)
- [03 RTL模块端口与功能参考](03_rtl_module_reference.md)
- [04 AXI4-Lite寄存器映射](04_register_map.md)
- [05 PS裸机驱动库设计](05_ps_driver_design.md)
- [06 PS裸机API参考](06_api_reference.md)
- `rtl/rl_instruction_sequencer.sv`
- `rtl/rl_accel_axil_top.sv`
- `rtl/rl_accel_core.sv`
- `sw/rl_accel_regs.h`
- `generated/policy_map.json`
