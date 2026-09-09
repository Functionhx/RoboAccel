# 05 PS 裸机驱动库设计

## 1. 分层

PS 软件运行在 Cortex-A9 standalone BSP 上，不依赖操作系统。

| 层 | 文件 | 责任 |
|---|---|---|
| 寄存器协议 | `sw/rl_accel_regs.h` | 偏移、状态位、opcode |
| 设备驱动 | `03_vitis/src/rl_accel.{c,h}` | MMIO、Cache/指令窗口、程序装载、序列启动、中断/轮询 |
| 策略运行时 | `03_vitis/src/rl_policy.{c,h}` | 模型/程序装载、双输入量化、单次整网启动、结果读取 |
| 生成数据 | `03_vitis/src/rl_policy_data.{c,h}` | Cache image、描述符、self-test；禁止手改 |
| 示例程序 | `03_vitis/src/main.c` | GIC、全量读回、自检、基准和周期日志 |

PL 基地址为 `0x43C00000`，完成中断为 GIC ID 61、level-high。驱动初始化要求 VERSION 等于 `0x00010002`。

## 2. 设备驱动

`rl_accel_init` 保存基地址、清理 DONE/ERROR 并校验版本。`rl_accel_vector_write/read` 和 `rl_accel_weight_write/read` 把一个 128-bit word 拆为四个 32-bit AXI4-Lite 数据寄存器。

正常整网执行由三组 API 完成：

- `rl_accel_pack_instruction`：校验并把 `rl_accel_cmd_t` 编码为四个 32-bit word；
- `rl_accel_load_program`：装载 1..32 条指令、可选逐条读回，并设置序列长度；
- `rl_accel_execute_program`：清状态、写一次序列启动并等待整网 DONE。

`rl_accel_execute_program` 的顺序为：

1. 确认 `BUSY=0` 并清除上一次 DONE；
2. 若启用中断，清理软件 IRQ 标记；
3. 写一次 `SEQUENCE_CONTROL.START`；
4. 轮询软件 IRQ 标记或 STATUS.DONE；
5. 检查 STATUS.ERROR 并清除 DONE。

GEMM 要求 K/N 均非零；CONCAT 要求两路计数之和非零；ELU/NORM 只要求 N 非零。驱动仍不证明 Cache 区间，模型导出器和上层应用负责边界安全。

中断服务 `rl_accel_isr` 只读取状态、记录 `irq_seen/irq_status` 并清除 DONE，避免在 ISR 中做耗时工作。应用可用 `rl_accel_set_interrupt_mode` 动态切换中断与轮询。

## 3. 当前策略运行时

当前模型输入为 `observation[25]` 和 `history[125]`，输出为 `actions[6]`。两路 float 输入都按 Q8.8、远离零舍入并饱和到 INT16。

`rl_policy_load` 写入 132 个有效 Vector word、8 个 bank 各 620 个有效 Weight word，并把 13 条指令写入 PL 指令 RAM后逐条读回。指令序列在后续推理中保持不变。

`rl_policy_infer_fixed`：

1. 将 25 个 observation 安全打包为 4 个 word；
2. 将 125 个 history 安全打包为 16 个 word；
3. 调用一次 `rl_accel_execute_program`，由 PL 自动执行 13 条指令；
4. 从 word 130 读取 6 个 action。

末 word 的无效 lane 写零，软件不会越界读取 C 输入数组。命令序列是 7×GEMM、5×ELU、1×CONCAT；网络拓扑存在于生成的指令程序中，而不是固化在 RTL 状态机内。

## 4. 生成器与数据同步

`tools/export_policy.py` 支持两个二维输入、Gemm、Elu 和 feature-axis 的二输入 Concat。它执行：

- ONNX shape inference 与合法性检查；
- 动态 Vector Cache 分配；
- 8×8 Weight tile 排布；
- 每个 GEMM 独立选择权重 fractional bits；
- Python 浮点/定点推理；
- 生成 Cache hex、`policy_map.json`、self-test 和 PS C 数组。
- 校验最多 32 条指令及各位域范围，并生成 `instruction_program.hex` 和 manifest 中的打包指令。

重新导出后，`generated/` 与 `03_vitis/src/rl_policy_data.{c,h}` 必须来自同一次运行。不要只替换 ONNX 而沿用旧描述符。

## 5. 对外集成建议

普通应用只需要：

1. 创建并初始化 `rl_accel_t`；
2. 可选配置 GIC，并把 ISR 连接到 `rl_accel_isr`；
3. 调用一次 `rl_policy_load`；
4. 重复调用 `rl_policy_infer`。

希望运行其他兼容网络的开发者可使用 Cache API 和 `rl_accel_load_program/rl_accel_execute_program`。`rl_accel_execute` 单算子接口仍保留用于调试。驱动没有线程安全或并发队列；多个调用者必须由上层串行化。
