# 06 PS 裸机 API 参考

本文档描述 `03_vitis/src` 中面向应用开发者的公开 C API。接口运行于 Cortex-A9 standalone BSP，通过 AXI4-Lite 控制 PL 加速器，不依赖操作系统、堆内存或 DMA。

```c
#include "rl_accel.h"   /* 底层设备、Cache、指令和单算子 API */
#include "rl_policy.h"  /* 当前 fudan_policy 策略运行时 API */
```

底层实现见 [`rl_accel.c`](../03_vitis/src/rl_accel.c)，当前策略封装见 [`rl_policy.c`](../03_vitis/src/rl_policy.c)，寄存器协议见 [`rl_accel_regs.h`](../sw/rl_accel_regs.h)。

## 1. 使用约定

### 1.1 初始化和调用顺序

典型调用顺序如下：

```text
rl_accel_init
      |
      +-- 可选：配置 GIC，并连接 rl_accel_isr
      |
      +-- rl_policy_load                 只需初始化时调用一次
              |
              +-- 可选：rl_policy_verify_model
              +-- 可选：rl_policy_selftest
              |
              +-- rl_policy_infer_fixed  重复推理
              `-- rl_policy_infer        重复推理
```

驱动不是线程安全或可重入的。一个 `rl_accel_t` 实例同一时刻只能有一个调用者，不能在推理执行期间从另一个上下文访问 Cache、改写指令 RAM 或提交单算子命令。

### 1.2 返回值

返回类型为 `int` 的函数使用 `xstatus.h` 中的状态码。实际实现可能返回：

| 返回值 | 含义 |
|---|---|
| `XST_SUCCESS` | 操作成功 |
| `XST_INVALID_PARAM` | 空指针、地址/维度越界、非法 opcode，或部分 Cache API 在硬件忙时被调用 |
| `XST_DEVICE_BUSY` | `rl_accel_execute()` 或 `rl_accel_execute_program()` 启动时设备已经处于 `BUSY` |
| `XST_FAILURE` | 版本不匹配、读回比较失败、硬件报告 `ERROR`、等待超时，或下层调用失败 |

`XST_FAILURE` 不区分硬件错误和软件等待超时。需要诊断时，应在返回后调用 `rl_accel_status()` 并读取序列状态寄存器。

### 1.3 超时参数

`timeout_iterations` 是 PS 忙等循环次数，不是微秒数：

- 传入 `0` 时使用 `RL_ACCEL_TIMEOUT`，当前为 `10,000,000` 次；
- 轮询模式下，每次循环包含一次 AXI4-Lite STATUS 读取；
- 中断模式下，每次循环只检查 `instance->irq_seen`；
- 实际持续时间受编译优化、CPU频率和AXI延迟影响，不能直接当作时间单位。

### 1.4 128-bit word 与地址单位

Vector、Weight 和 Instruction host window 都以一个128-bit word为访问单位。C数组按低位在前排列：

```text
data[0] = bits [ 31:  0]
data[1] = bits [ 63: 32]
data[2] = bits [ 95: 64]
data[3] = bits [127: 96]
```

对INT16向量而言：

```text
data[0][15:0]  = lane 0
data[0][31:16] = lane 1
...
data[3][31:16] = lane 7
```

所有 `word_addr` 和描述符中的 `*_base` 都是word地址，不是字节地址。

## 2. 公共常量和类型

### 2.1 设备和模型常量

```c
#define RL_ACCEL_VERSION          0x00010002u
#define RL_ACCEL_TIMEOUT          10000000u
#define RL_VECTOR_WORDS           512u
#define RL_WEIGHT_BANKS           8u
#define RL_WEIGHT_WORDS           1280u
#define RL_INSTRUCTION_WORDS      32u

#define RL_POLICY_OBS_COUNT       25u
#define RL_POLICY_HISTORY_COUNT   125u
#define RL_POLICY_ACTION_COUNT    6u
```

当前硬件基地址为 `0x43C00000`，通常使用 BSP 宏 `XPAR_RL_ACCEL_0_BASEADDR`；完成中断为 GIC ID 61，通常使用 `XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR`。

### 2.2 opcode

| 常量 | 值 | 算子 |
|---|---:|---|
| `RL_OP_GEMM` | 1 | INT16 GEMM、bias、量化和饱和 |
| `RL_OP_ELU` | 2 | 八路 Q8.8 ELU |
| `RL_OP_NORM` | 3 | 八路逐特征标准化 |
| `RL_OP_CONCAT` | 4 | 两个INT16向量拼接 |

### 2.3 `rl_accel_cmd_t`

```c
typedef struct {
    u32 opcode;
    u32 src_base;
    u32 dst_base;
    u32 aux0_base;
    u32 aux1_base;
    u32 weight_base;
    u32 dim_k;
    u32 dim_n;
    u32 output_shift;
} rl_accel_cmd_t;
```

该结构表示一条算子描述符。各字段随opcode复用：

| 字段 | GEMM | ELU | NORM | CONCAT |
|---|---|---|---|---|
| `src_base` | activation | 输入 | 输入 | input0 |
| `dst_base` | 输出 | 输出 | 输出 | 拼接输出 |
| `aux0_base` | bias | 未使用 | mean | input1 |
| `aux1_base` | 未使用 | 未使用 | reciprocal | 未使用 |
| `weight_base` | Weight tile首地址 | 未使用 | 未使用 | 未使用 |
| `dim_k` | 输入维度K | 未使用 | 未使用 | input0元素数 |
| `dim_n` | 输出维度N | 元素数 | 元素数 | input1元素数 |
| `output_shift` | 乘加结果量化移位 | 未使用 | 归一化移位 | 未使用 |

地址字段只检查首地址是否落在Cache内。驱动不会证明 `base + length`、GEMM tile区间或多层常驻布局不越界，这些条件必须由模型导出器或调用者保证。

### 2.4 `rl_accel_t`

```c
typedef struct {
    UINTPTR base_addr;
    int use_interrupt;
    volatile u32 irq_seen;
    volatile u32 irq_status;
} rl_accel_t;
```

| 字段 | 作用 |
|---|---|
| `base_addr` | AXI4-Lite外设基地址 |
| `use_interrupt` | 非零表示等待函数使用中断标记，否则轮询STATUS |
| `irq_seen` | ISR置位的完成标志，由驱动在每次启动前清零 |
| `irq_status` | ISR捕获的硬件STATUS快照 |

应用应把该实例放在执行期间始终有效的存储区。中断模式下通常使用全局或静态实例，不能把即将离开作用域的局部变量作为ISR回调参数。

### 2.5 `rl_policy_t`

```c
typedef struct {
    rl_accel_t *accel;
    int loaded;
} rl_policy_t;
```

`accel`指向底层设备实例；`loaded`由 `rl_policy_load()` 成功后置为1。应用不应手工修改这两个字段。

## 3. 设备初始化、状态和中断

### 3.1 `rl_accel_init`

```c
int rl_accel_init(rl_accel_t *instance,
                  UINTPTR base_addr,
                  int use_interrupt);
```

初始化一个设备实例，配置PL中断使能，清除遗留完成状态，并校验硬件接口版本。

参数：

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输出 | 调用者分配的设备对象，不得为 `NULL` |
| `base_addr` | 输入 | AXI4-Lite基地址，当前通常为 `XPAR_RL_ACCEL_0_BASEADDR` |
| `use_interrupt` | 输入 | `0`选择STATUS轮询；非零选择中断等待 |

返回：

- `XST_SUCCESS`：读到的VERSION等于 `RL_ACCEL_VERSION`；
- `XST_INVALID_PARAM`：`instance == NULL`；
- `XST_FAILURE`：VERSION不匹配，通常意味着bitstream、XSA/BSP或驱动头文件版本不一致。

实现过程：

1. 保存基地址和等待模式；
2. 清零软件IRQ标志；
3. 写 `IRQ_ENABLE`；
4. 写 `CONTROL.CLEAR_DONE`；
5. 读取VERSION并与 `0x00010002` 比较。

该函数不初始化GIC。若 `use_interrupt != 0`，调用者必须在首次执行前连接并启用 `rl_accel_isr()`。

### 3.2 `rl_accel_set_interrupt_mode`

```c
void rl_accel_set_interrupt_mode(rl_accel_t *instance, int enabled);
```

在轮询等待和中断等待之间切换。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入/输出 | 已初始化设备；为 `NULL` 时函数直接返回 |
| `enabled` | 输入 | `0`关闭PL中断并使用轮询；非零开启PL中断并使用ISR标志 |

函数会清除软件IRQ状态、清除硬件DONE并更新 `IRQ_ENABLE`。应只在设备空闲时调用；它不会配置、连接或启用GIC控制器。

### 3.3 `rl_accel_version`

```c
u32 rl_accel_version(const rl_accel_t *instance);
```

返回PL的32-bit VERSION寄存器。当前预期值为 `0x00010002`，可按高16位主版本、低16位次版本显示为1.2。

`instance`必须非空且 `base_addr` 有效；函数本身不做指针检查，也不提供错误码。

### 3.4 `rl_accel_status`

```c
u32 rl_accel_status(const rl_accel_t *instance);
```

返回STATUS寄存器原值：

| 位 | 宏 | 含义 |
|---:|---|---|
| 0 | `RL_STATUS_BUSY` | Core或自动序列正在执行 |
| 1 | `RL_STATUS_DONE` | 最近一次单算子或整网执行完成 |
| 2 | `RL_STATUS_ERROR` | 描述符或执行过程报告错误 |

`instance`必须非空且已初始化。读取STATUS不会清除DONE。

### 3.5 `rl_accel_isr`

```c
void rl_accel_isr(void *callback_ref);
```

PL完成中断的服务函数，应连接到 `XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR`。

| 参数 | 方向 | 说明 |
|---|---|---|
| `callback_ref` | 输入 | 指向已初始化且持续有效的 `rl_accel_t` |

ISR读取STATUS；若发现DONE或ERROR，则把状态保存到 `irq_status`、置位 `irq_seen`，最后清除硬件DONE以释放level-high中断。它不执行推理后处理，也不负责GIC初始化。

当前平台的GIC连接要点：

```c
XScuGic_Connect(&gic, XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR,
                (Xil_InterruptHandler)rl_accel_isr, &accel);
XScuGic_SetPriorityTriggerType(&gic,
                XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR, 0xa0u, 0x01u);
XScuGic_Enable(&gic, XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR);
```

完整异常注册过程见 [`main.c`](../03_vitis/src/main.c)。

## 4. Vector Cache API

Vector Cache共512个128-bit word，即4096个INT16元素。host访问只允许在 `STATUS.BUSY == 0` 时进行。

### 4.1 `rl_accel_vector_write`

```c
int rl_accel_vector_write(rl_accel_t *instance,
                          u32 word_addr,
                          const u32 data[4]);
```

把一个128-bit word写入Vector Cache。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入 | 已初始化设备 |
| `word_addr` | 输入 | word地址，合法范围 `0..511` |
| `data` | 输入 | 四个32-bit分片，`data[0]`为最低32位 |

返回：

- `XST_SUCCESS`：写命令已提交；
- `XST_INVALID_PARAM`：空指针、地址越界或设备忙。

实现上先写地址和四个staging寄存器，再写 `VEC_COMMAND.COMMIT`。函数返回时写事务已通过AXI4-Lite提交，不额外进行读回比较。

### 4.2 `rl_accel_vector_read`

```c
int rl_accel_vector_read(rl_accel_t *instance,
                         u32 word_addr,
                         u32 data[4]);
```

读取一个Vector Cache word。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入 | 已初始化设备 |
| `word_addr` | 输入 | word地址，合法范围 `0..511` |
| `data` | 输出 | 接收四个32-bit分片的数组 |

返回 `XST_SUCCESS`，或在空指针、地址越界、设备忙时返回 `XST_INVALID_PARAM`。

实现上先写地址和 `VEC_COMMAND.READ`，然后进行两次STATUS读取以覆盖PL同步RAM读延迟，最后读取四个readback寄存器。该API使用固定接口延迟，不是带超时的轮询函数。

## 5. Weight Cache API

Weight Cache由8个bank组成，每bank有1280个128-bit word。GEMM一次从8个bank的相同地址并行取得1024 bit；host API每次只访问一个bank中的一个word。

### 5.1 `rl_accel_weight_write`

```c
int rl_accel_weight_write(rl_accel_t *instance,
                          u32 bank,
                          u32 word_addr,
                          const u32 data[4]);
```

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入 | 已初始化设备 |
| `bank` | 输入 | bank编号，合法范围 `0..7` |
| `word_addr` | 输入 | bank内word地址，合法范围 `0..1279` |
| `data` | 输入 | 要写入的128-bit数据，低32位在前 |

返回 `XST_SUCCESS`；空指针、bank/地址越界或设备忙时返回 `XST_INVALID_PARAM`。

函数把bank和word地址编码到 `WEIGHT_ADDR`，写四个staging寄存器，再提交 `WEIGHT_COMMAND.COMMIT`。它不自动递增地址，也不读回验证。

### 5.2 `rl_accel_weight_read`

```c
int rl_accel_weight_read(rl_accel_t *instance,
                         u32 bank,
                         u32 word_addr,
                         u32 data[4]);
```

参数范围与 `rl_accel_weight_write()` 相同，`data`为输出缓冲区。成功返回 `XST_SUCCESS`；空指针、越界或设备忙返回 `XST_INVALID_PARAM`。

函数提交host read，使用两次STATUS读取覆盖同步RAM延迟，再读取四个Weight readback寄存器。推理期间禁止调用，因为host read和GEMM engine read共享Weight Cache内部读取通路。

## 6. 指令 RAM 和程序 API

Instruction RAM最多保存32条128-bit描述符。初始化完成后，程序在PL中常驻；每次推理只需要执行一次 `rl_accel_execute_program()`。

### 6.1 `rl_accel_instruction_write`

```c
int rl_accel_instruction_write(rl_accel_t *instance,
                               u32 instruction_addr,
                               const u32 data[4]);
```

把一条已经打包的128-bit指令写入Instruction RAM。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入 | 已初始化设备 |
| `instruction_addr` | 输入 | 指令索引，合法范围 `0..31` |
| `data` | 输入 | 四个32-bit指令word |

成功返回 `XST_SUCCESS`；空指针、索引越界或设备忙返回 `XST_INVALID_PARAM`。一般应用应优先使用 `rl_accel_load_program()`，避免手工编码错误。

### 6.2 `rl_accel_instruction_read`

```c
int rl_accel_instruction_read(rl_accel_t *instance,
                              u32 instruction_addr,
                              u32 data[4]);
```

读取一条已打包指令。参数范围与write接口相同，`data`为输出数组。成功返回 `XST_SUCCESS`；空指针、越界或设备忙返回 `XST_INVALID_PARAM`。

与Cache读取类似，函数提交READ后使用两次STATUS读取覆盖PL RAM延迟，再读取四个readback寄存器。

### 6.3 `rl_accel_pack_instruction`

```c
int rl_accel_pack_instruction(const rl_accel_cmd_t *command,
                              u32 data[4]);
```

校验一个结构化描述符并编码为PL要求的四个32-bit word。

| 参数 | 方向 | 说明 |
|---|---|---|
| `command` | 输入 | 待校验和编码的描述符 |
| `data` | 输出 | 编码结果，可直接传给instruction write |

返回：

- `XST_SUCCESS`：描述符合法并完成编码；
- `XST_INVALID_PARAM`：空指针、opcode/基地址/shift越界，或算子维度不合法。

编码格式：

```text
data[0]: opcode[2:0], src[11:3], dst[20:12], aux0[29:21]
data[1]: aux1[8:0], weight[19:9], shift[25:20]
data[2]: dim_k[15:0], dim_n[31:16]
data[3]: 0
```

合法性规则：GEMM要求K/N非零；ELU和NORM要求N非零；CONCAT要求两路元素数之和非零。该函数只检查字段能否编码，并不检查完整Cache区间是否重叠或越界。

### 6.4 `rl_accel_load_program`

```c
int rl_accel_load_program(rl_accel_t *instance,
                          const rl_accel_cmd_t *commands,
                          u32 count,
                          int verify_readback);
```

把一组结构化描述符装入Instruction RAM，并设置自动序列长度。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入 | 已初始化且空闲的设备 |
| `commands` | 输入 | 至少包含 `count` 个描述符的数组 |
| `count` | 输入 | 指令条数，合法范围 `1..32` |
| `verify_readback` | 输入 | 非零时每写一条立即读回并逐32-bit比较 |

返回：

- `XST_SUCCESS`：全部指令写入成功，并已更新 `SEQUENCE_LENGTH`；
- `XST_INVALID_PARAM`：顶层参数非法或设备忙；
- `XST_FAILURE`：某条描述符打包、写入、读回或比较失败。

实现过程：

1. 对每个 `commands[index]` 调用 `rl_accel_pack_instruction()`；
2. 写入Instruction RAM的对应索引；
3. 若要求验证，则立即读回四个word比较；
4. 全部成功后写 `SEQUENCE_LENGTH=count`。

失败时Instruction RAM可能已经被部分改写，但序列长度只在全部成功后更新。解决问题后应重新调用本函数完整装载程序。旧程序中超出 `count` 的指令不会被清零，但也不会执行。

### 6.5 `rl_accel_execute_program`

```c
int rl_accel_execute_program(rl_accel_t *instance,
                             u32 timeout_iterations);
```

启动已经装入PL的完整指令序列，并同步等待整网完成。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入/输出 | 已初始化设备；软件IRQ字段会被更新 |
| `timeout_iterations` | 输入 | 忙等次数；`0`使用 `RL_ACCEL_TIMEOUT` |

返回：

- `XST_SUCCESS`：整条序列完成且STATUS没有ERROR；
- `XST_INVALID_PARAM`：`instance == NULL`；
- `XST_DEVICE_BUSY`：启动时硬件仍忙；
- `XST_FAILURE`：等待超时或硬件报告ERROR。

实现过程：

1. 检查 `STATUS.BUSY`；
2. 清除上一轮DONE和软件IRQ标记；
3. 只写一次 `SEQUENCE_CONTROL.START`；
4. PL sequencer自动逐条提交指令；
5. 根据 `use_interrupt` 等待ISR标记或轮询STATUS.DONE；
6. 检查ERROR并在轮询模式下清除DONE。

调用前必须已经成功调用 `rl_accel_load_program()`。函数不会重新写描述符，也不会检查当前 `SEQUENCE_LENGTH` 是否与调用者预期一致。

## 7. 单算子调试 API

### 7.1 `rl_accel_execute`

```c
int rl_accel_execute(rl_accel_t *instance,
                     const rl_accel_cmd_t *command,
                     u32 timeout_iterations);
```

通过保留的单命令寄存器路径执行一条算子描述符。该接口主要用于模块调试、Cache检查和新算子程序开发；正常整网推理应使用Instruction RAM。

| 参数 | 方向 | 说明 |
|---|---|---|
| `instance` | 输入/输出 | 已初始化设备 |
| `command` | 输入 | 一条合法的GEMM、ELU、NORM或CONCAT命令 |
| `timeout_iterations` | 输入 | 忙等次数；`0`使用默认值 |

返回 `XST_SUCCESS`、`XST_INVALID_PARAM`、`XST_DEVICE_BUSY` 或 `XST_FAILURE`，含义见第1.2节。

函数先校验描述符，把所有字段写入单命令寄存器，然后写 `CONTROL.START`。之后使用与整网执行相同的中断/轮询等待函数。它不会把该命令写入Instruction RAM，也不会改变已装载的自动序列程序。

单条GEMM调试示例：

```c
rl_accel_cmd_t cmd = {
    .opcode       = RL_OP_GEMM,
    .src_base     = 0,
    .dst_base     = 20,
    .aux0_base    = 36,   /* bias */
    .aux1_base    = 0,
    .weight_base  = 0,
    .dim_k        = 125,
    .dim_n        = 128,
    .output_shift = 11
};

status = rl_accel_execute(&accel, &cmd, 0);
```

调用者必须事先把activation、bias和weight放到描述符引用的Cache区域。

## 8. 当前策略运行时 API

本节API固定对应当前 `fudan_policy.onnx` 导出产物。网络地址、权重和13条描述符来自生成的 `rl_policy_data.{c,h}`；若重新导出模型，必须重新编译PS程序。

### 8.1 `rl_policy_load`

```c
int rl_policy_load(rl_policy_t *policy,
                   rl_accel_t *accel);
```

把当前模型的Vector image、Weight image和指令程序装入PL。通常在启动时调用一次。

| 参数 | 方向 | 说明 |
|---|---|---|
| `policy` | 输出 | 调用者分配的策略对象 |
| `accel` | 输入 | 已通过 `rl_accel_init()` 初始化且空闲的设备 |

返回：

- `XST_SUCCESS`：模型和13条指令装载完成；
- `XST_INVALID_PARAM`：任一指针为空；
- `XST_FAILURE`：任一底层Cache/程序操作失败。

实现过程：

1. 把 `policy->loaded` 清零；
2. 写入132个有效Vector word，包括bias、常量和初始张量区域；
3. 逐bank写入 `8 × 620` 个Weight word；
4. 调用 `rl_accel_load_program(..., 13, 1)`；
5. 逐条读回验证全部13条指令；
6. 成功后设置 `policy->loaded=1`。

该函数验证指令程序，但为了控制启动时间，不会自动读回全部Vector和Weight Cache。需要全量验证时再调用 `rl_policy_verify_model()`。失败后Cache可能部分更新，`loaded`保持0，应重新完整装载。

### 8.2 `rl_policy_verify_model`

```c
int rl_policy_verify_model(rl_policy_t *policy);
```

把PL内的模型Cache内容与编译进PS程序的生成数组逐word比较。

| 参数 | 方向 | 说明 |
|---|---|---|
| `policy` | 输入 | 已成功装载且设备空闲的策略对象 |

返回：

- `XST_SUCCESS`：132个Vector word和 `8 × 620` 个Weight word全部一致；
- `XST_INVALID_PARAM`：对象为空、未装载或设备指针为空；
- `XST_FAILURE`：Cache读取失败或任一32-bit分片不一致。

重要限制：推理会覆盖输入、隐藏层和输出所在的Vector Cache区域。因此该函数应在 `rl_policy_load()` 之后、第一次推理之前调用。推理后调用通常会因动态Vector数据变化而失败；Weight Cache本身仍应保持不变。该函数不重复验证Instruction RAM，因为 `rl_policy_load()` 已启用逐条指令读回。

### 8.3 `rl_policy_infer_fixed`

```c
int rl_policy_infer_fixed(
    rl_policy_t *policy,
    const s16 observation_q[RL_POLICY_OBS_COUNT],
    const s16 history_q[RL_POLICY_HISTORY_COUNT],
    s16 actions_q[RL_POLICY_ACTION_COUNT]);
```

执行一次完整的定点策略推理。

| 参数 | 方向 | 格式与长度 |
|---|---|---|
| `policy` | 输入/输出 | 已成功装载的策略对象 |
| `observation_q` | 输入 | 25个signed Q8.8 INT16 |
| `history_q` | 输入 | 125个signed Q8.8 INT16 |
| `actions_q` | 输出 | 6个signed Q8.8 INT16 |

返回：

- `XST_SUCCESS`：输入写入、整网执行和输出读取全部成功；
- `XST_INVALID_PARAM`：对象未装载或任一数组指针为空；
- `XST_FAILURE`：任一底层Cache访问或整网执行失败。

实现过程：

1. 把25个observation打包为4个128-bit word，写入Vector base 0；
2. 把125个history打包为16个word，写入Vector base 4；
3. 每个末尾不足8元素的word自动将无效lane补零；
4. 调用一次 `rl_accel_execute_program()`，PL自动执行13条指令；
5. 从Vector base 130读取一个word；
6. 解包前6个lane到 `actions_q`。

该函数不会重新装载权重或指令。输入和输出使用原始Q8.8整数，例如 `256` 表示实数 `1.0`，`-128` 表示 `-0.5`。

### 8.4 `rl_policy_quantize_inputs`

```c
void rl_policy_quantize_inputs(
    const float observation[RL_POLICY_OBS_COUNT],
    const float history[RL_POLICY_HISTORY_COUNT],
    s16 observation_q[RL_POLICY_OBS_COUNT],
    s16 history_q[RL_POLICY_HISTORY_COUNT]);
```

把两路FP32输入量化为Q8.8，但不访问硬件，也不执行推理。

| 参数 | 方向 | 说明 |
|---|---|---|
| `observation` | 输入 | 25个有限FP32值 |
| `history` | 输入 | 125个有限FP32值 |
| `observation_q` | 输出 | 25个Q8.8 INT16 |
| `history_q` | 输出 | 125个Q8.8 INT16 |

量化规则：

```text
q = saturate_int16(round_away_from_zero(value × 256))
```

Q8.8可表示范围为 `[-128.0, 127.99609375]`。该函数返回 `void` 且不检查空指针、NaN或Infinity；调用者必须提供有效数组和有限输入。

### 8.5 `rl_policy_infer`

```c
int rl_policy_infer(
    rl_policy_t *policy,
    const float observation[RL_POLICY_OBS_COUNT],
    const float history[RL_POLICY_HISTORY_COUNT],
    float actions[RL_POLICY_ACTION_COUNT]);
```

提供FP32形式的便捷推理接口。它并不会让PL执行FP32计算；内部仍是INT16/Q8.8推理。

| 参数 | 方向 | 说明 |
|---|---|---|
| `policy` | 输入/输出 | 已成功装载的策略对象 |
| `observation` | 输入 | 25个有限FP32值 |
| `history` | 输入 | 125个有限FP32值 |
| `actions` | 输出 | 6个FP32动作值 |

实现过程：

1. 调用 `rl_policy_quantize_inputs()`；
2. 调用 `rl_policy_infer_fixed()`；
3. 将每个输出按 `actions_q[i] / 256.0f` 反量化。

成功返回 `XST_SUCCESS`；底层推理失败返回 `XST_FAILURE`。当前实现会在检查策略状态之前量化输入，因此所有指针都必须非空；不要依赖该函数处理空指针。

### 8.6 `rl_policy_selftest`

```c
int rl_policy_selftest(rl_policy_t *policy);
```

使用导出器生成并编译进程序的固定双输入，执行一次整网推理，并要求6个Q8.8输出逐bit匹配golden：

```text
[543, 790, 74, 635, 478, -796]
```

返回 `XST_SUCCESS` 表示推理成功且全部输出一致；推理失败或任一输出不同返回 `XST_FAILURE`。调用者应先完成 `rl_policy_load()`。当前实现没有在入口单独检查空指针或 `loaded`，而是依赖内部 `rl_policy_infer_fixed()` 返回失败，因此仍必须传入有效对象。

Self-test验证当前bitstream、Cache image、指令程序和定点数据通路能对一个已知输入产生精确结果，但不替代随机输入精度验证或实际强化学习环境回报测试。

## 9. 最小轮询示例

```c
#include "xparameters.h"
#include "xstatus.h"
#include "rl_accel.h"
#include "rl_policy.h"

static rl_accel_t accel;
static rl_policy_t policy;

int run_once(void)
{
    float obs[RL_POLICY_OBS_COUNT] = {0.0f};
    float history[RL_POLICY_HISTORY_COUNT] = {0.0f};
    float actions[RL_POLICY_ACTION_COUNT];
    int status;

    status = rl_accel_init(&accel, XPAR_RL_ACCEL_0_BASEADDR, 0);
    if (status != XST_SUCCESS)
        return status;

    status = rl_policy_load(&policy, &accel);
    if (status != XST_SUCCESS)
        return status;

    /* 可选，且应放在第一次推理之前。 */
    status = rl_policy_verify_model(&policy);
    if (status != XST_SUCCESS)
        return status;

    status = rl_policy_selftest(&policy);
    if (status != XST_SUCCESS)
        return status;

    return rl_policy_infer(&policy, obs, history, actions);
}
```

若持续推理，初始化、模型装载和可选验证只执行一次，之后重复调用 `rl_policy_infer()` 或 `rl_policy_infer_fixed()`：

```c
for (;;) {
    acquire_observation(obs, history);
    if (rl_policy_infer(&policy, obs, history, actions) != XST_SUCCESS)
        handle_accelerator_error(rl_accel_status(&accel));
    else
        apply_actions(actions);
}
```

## 10. 中断模式集成检查表

使用中断等待时必须同时满足：

1. `rl_accel_t` 在ISR有效期间始终存在；
2. GIC完成 `CfgInitialize`；
3. CPU异常入口注册为 `XScuGic_InterruptHandler`；
4. GIC ID 61连接到 `rl_accel_isr`，回调参数为同一个设备实例；
5. 中断配置为当前设计要求的level-high；
6. GIC中断和CPU异常均已使能；
7. 调用 `rl_accel_set_interrupt_mode(&accel, 1)` 开启PL中断；
8. ISR和前台代码不并发修改 `irq_seen/irq_status` 之外的设备状态。

如果只把 `use_interrupt` 设为1而没有正确配置GIC，执行函数会一直等待到 `timeout_iterations` 耗尽并返回 `XST_FAILURE`。

## 11. API能力边界

- API只支持裸机同步调用，没有异步提交、队列、DMA或多线程保护；
- 自动程序长度限制为1..32条；
- Vector/Weight地址范围检查不等于完整张量边界检查；
- 执行期间禁止host访问Cache或改写Instruction RAM；
- 当前策略API固定为25+125输入和6输出，兼容其他网络时应重新导出并生成新的策略封装；
- `rl_accel_execute()` 保留用于单算子调试，正式推理优先使用常驻程序；
- 任何API返回失败后，在重新启动前应检查STATUS、清理DONE，并确认模型/程序没有被部分更新。
