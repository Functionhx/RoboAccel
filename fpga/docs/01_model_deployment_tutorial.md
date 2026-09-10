# 01 新ONNX模型导出、网络映射与部署教程

本文给出从一个新的ONNX策略模型开始，到生成定点权重和PL指令、完成RTL行为级验证、构建PS裸机程序、JTAG下载以及实机正确性测试的完整流程。

本教程严格按照当前仓库能力编写。它首先说明“模型是否能直接部署”，再分别给出只更换模型数据的快速路径，以及需要修改RTL的硬件扩展路径。

## 1. 全流程概览

```text
新的 ONNX
    |
    +-- 结构与算子兼容性检查
    |
    +-- tools/export_policy.py
    |      +-- 定点量化
    |      +-- Vector/Weight Cache 布局
    |      +-- 128-bit 指令描述符
    |      +-- Python 定点 golden
    |      `-- PS C 数组
    |
    +-- 检查 generated/policy_map.json
    |
    +-- 适配模型相关的 PS API / TB / UART 协议常量
    |
    +-- make -C tb all
    |
    +-- 构建通用、周期和一致性三个裸机程序
    |
    +-- JTAG 下载 bitstream + ELF
    |
    `-- Cache/指令读回 + self-test + 周期测试 + FP32 一致性测试
```

如果新模型只使用现有算子并且容量没有超限，网络拓扑会被编码进Instruction RAM，不需要重新生成bitstream。只有增加新算子、改变寄存器接口、修改数值格式或扩大Cache时，才需要修改RTL并重新走Vivado流程。

## 2. 先判断模型属于哪条部署路径

### 2.1 路径A：描述符兼容模型

同时满足以下条件时，可以复用当前PL和bitstream：

- 恰好两个运行时输入；
- 每个输入都是二维 `[batch, features]`；
- 实际推理批量为1；
- 图中只包含当前导出器支持的 `Gemm`、`Elu` 和二输入 `Concat`；
- 只有一个图输出；
- 程序不超过32条指令；
- Vector布局不超过512个word；
- 每个Weight bank不超过1280个word；
- 当前公共PS封装要求输出动作数量不超过8；
- 输入、activation、bias和输出可接受Q8.8精度。

这条路径只需要重新导出模型、适配模型相关软件常量、运行仿真并重新编译PS应用。

### 2.2 路径B：需要扩展软件或RTL

出现以下任一情况时，不能直接套用快速路径：

- 输入不是两路，或者不是 `[batch, features]`；
- 出现 `MatMul`、`Add`、`Relu`、`Tanh`、卷积、循环网络、注意力、Softmax等算子；
- ONNX图使用NORM相关节点；PL虽然保留NORM算子，但当前导出器尚未把ONNX归一化子图转换为NORM描述符；
- `Concat`不在feature轴上，或者有超过两个输入；
- 输出超过8个元素；
- Cache或Instruction RAM容量不足；
- 需要不同于Q8.8/INT16/INT48的数值格式；
- 需要batch大于1、DMA、DDR master或并发命令。

其中，输入/输出数量和UART协议可以只扩展软件；新算子、数值格式和硬件容量通常需要修改PL。

## 3. 环境准备

本仓库当前验证环境为：

| 项目 | 版本/配置 |
|---|---|
| 目标器件 | `XC7Z010CLG400-2` |
| Vivado/Vitis | 2020.2 |
| Python | 支持当前 `onnx`、`numpy` 依赖的版本 |
| RTL仿真 | Icarus Verilog，支持SystemVerilog 2012 |
| PS运行环境 | Cortex-A9 standalone裸机 |
| 开发板启动 | JTAG |
| UART | COM17，115200 8N1 |

从仓库根目录安装导出和一致性测试依赖：

```powershell
python -m pip install -r .\tools\requirements.txt
python -m pip install -r .\03_vitis_consistency_test\requirements.txt
```

确认工具可用：

```powershell
python --version
iverilog -V
& 'D:\XILINX\Vivado\2020.2\bin\vivado.bat' -version
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' -eval 'puts [version]'
```

`03_vitis_periodic_test/build.ps1` 和 `03_vitis_consistency_test/build.ps1` 当前包含本机Python路径。如果在另一台电脑复现，应先把脚本中的 `$python` 和 `$pythonSite` 改为该机器实际使用的解释器和site-packages路径。

## 4. ONNX模型接口要求

### 4.1 输入和输出

当前导出器要求：

```text
input 0: [batch, feature_count_0]
input 1: [batch, feature_count_1]
output : [batch, action_count] 或能推导出固定feature数量的单输出
```

第一维可以在ONNX中是动态batch，但当前运行时每次只提交一个样本。第二维必须是大于0的静态整数。

使用 `--c-out` 生成公共PS数组时，两路输入必须按顺序命名为：

```text
obs
obs_history
```

为了直接复用一致性测试脚本，建议把唯一输出命名为：

```text
actions
```

如果输入名称不同，纯HEX导出仍可能成功，但生成公共PS C数组时会报错。可以在导出ONNX时保留这些名称，也可以扩展 `write_c_images()` 和策略API。

### 4.2 当前支持的ONNX节点

#### Gemm

要求：

- `transA=0`；
- `alpha=1`；
- `beta=1`；
- weight和bias必须是initializer；
- weight必须是二维矩阵；
- bias长度必须等于输出维度；
- `transB=0/1`均可，导出器会整理成PL使用的 `[N,K]` 形式。

建议确保训练框架最终导出为单个 `Gemm`，不要导出为 `MatMul + Add`，因为当前转换器不会自动融合这两个节点。

#### Elu

只支持：

```text
alpha = 1
```

#### Concat

只支持：

- 恰好两个输入；
- `axis=1` 或 `axis=-1`；
- 两个输入都已在前面的图节点中获得确定feature数量。

### 4.3 模型结构快速检查

先对新模型运行ONNX checker并打印接口和节点：

```powershell
@'
import onnx
from onnx import shape_inference

path = "new_policy.onnx"
model = shape_inference.infer_shapes(onnx.load(path))
onnx.checker.check_model(model)

print("inputs:")
for value in model.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param
            for d in value.type.tensor_type.shape.dim]
    print("  ", value.name, dims)

print("outputs:")
for value in model.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param
            for d in value.type.tensor_type.shape.dim]
    print("  ", value.name, dims)

print("nodes:")
for index, node in enumerate(model.graph.node):
    print(f"  {index:02d}: {node.op_type:8s} {node.name}")
'@ | python -
```

如果节点列表出现当前不支持的算子，应先决定是在训练/导出侧做算子融合与替换，还是进入RTL扩展路径。不要在未理解数值等价性的情况下直接删除节点。

## 5. 放置新的权威模型

当前多个构建脚本默认读取根目录：

```text
fudan_policy.onnx
```

最少修改的做法是：先提交或备份当前模型，再用新模型替换这个权威文件。结构图也应同步更新为 `fudan_net.png`。

如果希望保留新的文件名，则至少需要同步修改：

- `03_vitis_periodic_test/build.ps1`；
- `03_vitis_consistency_test/build.ps1`；
- `03_vitis_consistency_test/host/validate_consistency.py` 的默认模型；
- `03_vitis_consistency_test/host/verify_log.py` 的默认模型；
- 根目录README、AGENTS和模型相关文档。

无论采用哪种文件名，后续所有生成物、PS程序和验证日志必须来自同一个ONNX文件。建议在验证日志中记录ONNX的SHA-256。

## 6. 导出权重、Cache布局和指令程序

从根目录执行：

```powershell
python .\tools\export_policy.py `
  --model .\fudan_policy.onnx `
  --out .\generated `
  --samples 1000 `
  --seed 7 `
  --c-out .\03_vitis\src
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--model` | 输入ONNX路径 |
| `--out` | HEX、manifest和self-test输出目录 |
| `--samples` | 随机FP32/定点对比样本数 |
| `--seed` | 随机验证种子，固定后可复现 |
| `--c-out` | 生成 `rl_policy_data.c/h` 的目录 |

导出器执行以下工作：

1. ONNX shape inference和结构合法性检查；
2. 按图顺序把Gemm、Elu、Concat转换为描述符；
3. 为输入、中间张量、bias和输出分配Vector Cache；
4. 每个GEMM独立选择不溢出INT16的最高权重小数位；
5. 将权重排列成8 bank、8×8 tile布局；
6. 生成128-bit指令程序；
7. 运行Python FP32模型和定点模型的随机对比；
8. 生成一组确定性self-test输入和Q8.8 golden输出；
9. 可选生成PS可直接编译的C数组。

### 6.1 生成文件说明

| 文件 | 用途 |
|---|---|
| `generated/policy_map.json` | 网络、张量地址、容量、描述符、量化参数和误差指标的权威manifest |
| `generated/vector_cache.hex` | 512×128-bit Vector Cache初始化镜像 |
| `generated/weight_bank0..7.hex` | 8个完整Weight bank镜像 |
| `generated/weight_bank*_main.hex` | 每bank地址0..1023，供RTL BRAM main区域仿真初始化 |
| `generated/weight_bank*_tail.hex` | 每bank地址1024..1279，供tail区域初始化 |
| `generated/instruction_program.hex` | 32×128-bit Instruction RAM镜像 |
| `generated/selftest_input0.hex` | 第0路确定性输入 |
| `generated/selftest_input1.hex` | 第1路确定性输入 |
| `generated/selftest_actions.hex` | Q8.8 self-test输出；当前软件路径要求输出不超过8个 |
| `03_vitis/src/rl_policy_data.c/h` | PS模型、权重、描述符和self-test数组 |

这些文件是同一次导出的整体，不要手工只替换其中某一个大数组。

## 7. 检查导出结果和容量

导出成功只代表脚本没有发现结构和容量错误，还必须检查manifest：

```powershell
$map = Get-Content .\generated\policy_map.json -Raw | ConvertFrom-Json

$map.inputs
$map.output
$map.vector_words_used
$map.weight_words_per_bank
$map.instruction_count
$map.accuracy_metrics
$map.descriptors | Format-Table opcode,src_base,dst_base,aux0_base,weight_base,dim_k,dim_n,shift,node
```

必须满足：

```text
vector_words_used       <= 512
weight_words_per_bank   <= 1280
instruction_count       <= 32
1 <= output.count       <= 8    # 当前公共PS/周期/UART封装限制
```

Weight容量也可以在导出前估算：

```text
sum(ceil(K_i/8) * ceil(N_i/8)) <= 1280
```

当前Vector分配器只对ELU做原地复用；GEMM输出、bias和Concat输出会继续向后分配。因此应以manifest的 `vector_words_used` 为准，而不是只估算某一时刻的最大activation。

### 7.1 检查描述符是否表达了预期网络

描述符顺序就是PL执行的网络顺序。例如：

```text
GEMM -> ELU -> GEMM -> CONCAT -> GEMM
```

逐条检查：

- `opcode` 是否对应ONNX节点；
- `src_base` 是否指向上一层输出；
- GEMM的 `aux0_base` 是否指向bias；
- `dim_k/dim_n` 是否等于输入/输出feature数；
- `weight_base` 是否按层单调推进且不重叠；
- Concat的两路长度和地址是否正确；
- 最后一条描述符的 `dst_base` 是否等于manifest输出地址。

这一步就是“搭建网络”：PS初始化时把这些描述符写入Instruction RAM，推理时PL根据它们逐层选择算子和Cache地址。网络不需要重新硬编码进RTL状态机。

### 7.2 解释随机量化指标

manifest中的MAE、RMSE和argmax一致率来自标准正态随机输入，只适合作为快速量化回归。它不能代替真实环境观测分布测试。

如果新模型的真实输入有不同量纲、限幅或归一化，应额外使用实际数据集运行FP32 ONNX与定点模型对比。是否接受某个误差数值，应由控制任务的稳定性、动作尺度和回报指标决定，而不是沿用当前模型阈值。

## 8. 适配模型相关的软件和测试代码

导出器负责数据镜像和描述符，但当前仓库仍有少量模型相关的接口常量和测试协议。新模型维度发生变化时必须同步更新。

### 8.1 公共策略API

检查生成的：

```text
03_vitis/src/rl_policy_data.h
```

把其中的新输入/输出数量同步到：

```text
03_vitis/src/rl_policy.h
```

需要一致的宏为：

```c
RL_POLICY_OBS_COUNT
RL_POLICY_HISTORY_COUNT
RL_POLICY_ACTION_COUNT
```

`rl_policy.c` 已根据生成的base、word数量和描述符工作。只要仍是 `obs + obs_history -> actions`、输出不超过8个，通常不需要改变其控制流程。

如果输出超过8个，必须扩展 `rl_policy_infer_fixed()`，读取 `ceil(action_count/8)` 个Vector word，并同步扩展self-test、周期程序和UART响应协议；当前代码只读取一个128-bit action word。

### 8.2 端到端RTL testbench

`tb_policy_e2e.sv` 当前是模型相关测试，仍硬编码了：

- 两路self-test输入word数量；
- 输入Vector base；
- 指令条数；
- 每条描述符字段；
- action base和action数量。

根据 `generated/policy_map.json` 更新这些值。每条描述符必须与manifest完全一致。更新后，testbench继续从新生成的Vector/Weight HEX和self-test文件装载数据。

不要只让TB“能跑完”；它必须从manifest输出base读取数据，并与新 `selftest_actions.hex` 逐lane比较。

### 8.3 周期性能程序

`03_vitis_periodic_test/tools/generate_model.py` 会从ONNX动态生成输入数量、输出数量、Cache镜像、描述符和四组golden测试向量。它仍要求输入名为 `obs` 和 `obs_history`。

如果权威模型仍叫 `fudan_policy.onnx`，运行 `build.ps1` 会自动重新生成 `periodic_model.c/h`。如果使用其他文件名，应修改 `03_vitis_periodic_test/build.ps1` 的 `--model` 参数。

### 8.4 UART一致性程序

输入或输出数量变化时，需要同步修改：

- `03_vitis_consistency_test/host/rl_uart_protocol.py` 中的 `OBS_COUNT`、`HISTORY_COUNT` 和 `ACTION_COUNT`；
- 该文件中响应解包的action切片和 `inference_us` 索引；
- `03_vitis_consistency_test/src/main.c` 的启动日志维度；
- `validate_consistency.py` 和 `verify_log.py` 中的ONNX输入/输出名称；
- 协议单元测试中的期望packet长度。

板端C协议长度会通过 `rl_policy.h` 宏自动变化，但Python端使用独立常量，二者必须完全一致，否则会出现 `bad_length`、响应解析失败或串口超时。

### 8.5 文档和权威模型信息

同步更新：

- 根目录 `README.md`；
- 根目录 `AGENTS.md` 和 `CLAUDE.md`；
- `docs/README.md`；
- 系统架构、API和验证文档中的模型结构、容量、指令数和精度数据；
- 根目录网络结构图。

旧模型日志和性能数据只能保留为历史证据，不能继续作为新模型结论。

## 9. 行为级仿真

先运行导出器本身的Python定点golden，再运行RTL回归：

```powershell
make -C tb all
```

最低通过要求：

```text
PASS tb_primitives
PASS tb_gemm_engine
PASS tb_concat_engine
PASS tb_instruction_sequencer
PASS tb_axil_top
PASS tb_policy_e2e
```

`tb_policy_e2e`输出的cycle数会随网络变化，不应继续要求当前模型的1799 cycle。应记录新值，并按100 MHz换算纯PL时间：

```text
PL time in microseconds = cycle_count / 100
```

例如2000 cycle对应20.00 µs。

如果只换描述符兼容模型，单元TB验证算子实现没有退化，policy TB验证新的Cache布局、指令顺序和定点golden。如果policy TB仍使用旧的硬编码描述符，即使其他TB全部PASS，也不能证明新网络正确。

## 10. 构建PS裸机程序

### 10.1 兼容模型：XSA没有变化

只替换模型数据时，不需要重新生成Vivado工程，也不需要重建BSP。直接构建通用程序：

```powershell
Set-Location D:\rl_on_fpga\03_vitis
& .\build_software.ps1
```

预期产物：

```text
03_vitis/build/rl_policy_app.elf
03_vitis/boot/BOOT.BIN
```

构建周期程序：

```powershell
Set-Location D:\rl_on_fpga\03_vitis_periodic_test
& .\build.ps1
```

构建UART一致性程序：

```powershell
Set-Location D:\rl_on_fpga\03_vitis_consistency_test
& .\build.ps1
```

注意：一致性工程的 `build.ps1` 会再次运行公共导出器并覆盖 `generated/` 和 `03_vitis/src/rl_policy_data.c/h`。必须确保它引用的ONNX与前面验证的是同一个文件。

### 10.2 BSP不存在或XSA已经变化

如果是首次构建，或者硬件XSA发生变化，先重建三个standalone platform/BSP：

```powershell
Set-Location D:\rl_on_fpga\03_vitis
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' .\create_workspace.tcl
& .\build_software.ps1

Set-Location D:\rl_on_fpga\03_vitis_periodic_test
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' .\create_platform.tcl
& .\build.ps1

Set-Location D:\rl_on_fpga\03_vitis_consistency_test
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' .\create_platform.tcl
& .\build.ps1
```

不要让旧BSP继续引用旧XSA。尤其是基地址、中断号或bitstream变化后，三个工程都必须重建。

## 11. 需要修改PL时的完整硬件路径

如果新模型需要新算子、更多Cache、不同精度或接口字段，应按从下到上的顺序修改：

1. 在 `rtl/` 添加或修改底层算子；
2. 为新算子增加独立单元testbench；
3. 在 `rl_accel_params.vh` 分配opcode或调整参数；
4. 在 `rl_accel_core.sv` 增加启动译码、Cache仲裁和done选择；
5. 在 `rl_instruction_sequencer.sv` 增加描述符合法性规则；
6. 在 `rl_accel_axil_top.sv` 和 `sw/rl_accel_regs.h` 同步接口；
7. 更新 `tools/export_policy.py` 的ONNX解析、量化、布局、golden和C代码生成；
8. 更新两个PS驱动、三个应用和全部相关文档；
9. 提升 `RL_ACCEL_VERSION`，使旧软件不能误连新硬件；
10. 运行全部RTL回归；
11. 重新综合、布局布线并生成bitstream/XSA；
12. 重建三个BSP和ELF；
13. 做Cache/指令全量读回和实机验证。

生成完整系统：

```powershell
Set-Location D:\rl_on_fpga\02_vivado
& 'D:\XILINX\Vivado\2020.2\bin\vivado.bat' `
  -mode batch -source .\create_system.tcl -nojournal
```

必须检查：

- `02_vivado/system_reports/timing_summary.rpt` 中setup/hold满足约束；
- `02_vivado/system_reports/utilization.rpt` 中资源未超过XC7Z010容量；
- `02_vivado/output/rl_system.bit` 和 `rl_system.xsa` 来自本次构建。

综合成功不等于功能正确；Vivado实现必须建立在行为级测试已通过的基础上。

## 12. JTAG上板部署

### 12.1 通用程序

```powershell
Set-Location D:\rl_on_fpga
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' '.\03_vitis\run_jtag.tcl'
```

串口应看到：

- 正确的接口版本；
- 模型和指令装载成功；
- 指令读回PASS；
- 可选的完整Cache读回PASS；
- polling和interrupt self-test PASS；
- 新模型的Q8.8输出；
- `READY`。

### 12.2 周期性能测试

```powershell
& D:\rl_on_fpga\03_vitis_periodic_test\run_and_capture.ps1 `
  -Port COM17 -Seconds 15
```

验收时至少检查：

```text
failed=0
overrun=0
每批 attempted=100
每批 checksum 在同一测试向量调度下稳定
```

性能数字应说明是否包含输入MMIO、启动、等待和输出读回，不要把纯PL cycle与PS端到端时间混写。

### 12.3 FP32随机输入一致性测试

先运行协议单元测试：

```powershell
python -m unittest .\03_vitis_consistency_test\host\test_protocol.py
```

再运行COM17闭环：

```powershell
& D:\rl_on_fpga\03_vitis_consistency_test\run_test.ps1 `
  -Port COM17 -Count 100 -Interval 1.0
```

测试脚本每次生成两路FP32输入，同时：

1. 发送给PS并在FPGA上执行Q8.8推理；
2. 用ONNX `ReferenceEvaluator`执行原始FP32模型；
3. 比较MAE、RMSE、最大误差和argmax；
4. 保存逐样本JSONL和summary JSON。

输入维度、输出维度或名称变化后，必须先完成第8.4节的协议适配，否则这个测试不能作为新模型证据。

## 13. 验证结论应如何形成

一个新模型至少应留下四类证据：

| 层级 | 必要证据 |
|---|---|
| 导出 | `policy_map.json`、ONNX哈希、容量和随机量化指标 |
| RTL | 六个TB日志，特别是新描述符驱动的policy E2E PASS |
| 实现 | 时序和资源报告；仅改兼容模型时可注明复用已验证bitstream |
| 实机 | Cache/指令读回、self-test、周期日志、FP32一致性JSONL |

建议在新模型报告中明确记录：

- ONNX文件SHA-256；
- Git提交或源码快照；
- exporter命令、样本数和seed；
- 输入分布及限幅；
- 每层权重fractional bits；
- Vector/Weight/Instruction使用率；
- 纯PL cycle和端到端延迟；
- MAE、RMSE、最大误差、相对误差和argmax一致率；
- 开发板、bitstream、ELF、工具版本和串口原始日志路径。

## 14. 常见错误

| 报错/现象 | 常见原因 | 处理方式 |
|---|---|---|
| `expected two policy inputs` | ONNX不是两路运行时输入 | 在模型侧封装两路输入，或扩展导出器和PS API |
| `expected [batch, features]` | 输入rank不是2或feature维动态 | 导出固定feature维的二维输入 |
| `unsupported ONNX operator` | 出现MatMul/Add/Relu等节点 | 在导出侧融合/替换，或实现新算子 |
| `public PS API requires inputs named obs and obs_history` | `--c-out`要求固定输入名 | 重命名ONNX输入或扩展C生成器/API |
| `vector cache exhausted` | 静态布局超过512 word | 减小网络、增加复用，或扩大Cache并重新实现PL |
| `weight words/bank; limit is 1280` | 权重tile总量超限 | 减小网络，或扩大Weight Cache并重新实现 |
| `instructions; limit is 32` | 算子节点过多 | 融合算子、增大Instruction RAM或增加循环型指令 |
| Python指标正确但policy TB失败 | TB仍硬编码旧描述符或地址 | 按新manifest更新 `tb_policy_e2e.sv` |
| `accelerator version mismatch` | bitstream与驱动VERSION不同 | 下载正确bitstream/XSA/ELF或同步接口版本 |
| 模型装载失败 | 设备忙、生成数组不同步或Cache写入异常 | 冷复位，确认同一次导出，并启用读回定位 |
| UART `bad_length`/超时 | Python与C协议维度不一致 | 同步输入/输出计数、packet格式和协议测试 |
| 一致性脚本输出数量错误 | host仍硬编码旧output名称或数量 | 更新协议常量和ONNX输出名 |
| 第二次JTAG下载异常 | 复位序列不完整或GIC残留 | 保留仓库 `run_jtag.tcl` 的完整复位序列 |

## 15. 发布前检查清单

- [ ] ONNX checker通过；
- [ ] 输入名、输入数量、shape、输出名符合约定；
- [ ] 图中没有未支持算子；
- [ ] exporter成功且生成物来自同一次运行；
- [ ] `policy_map.json` 的容量、地址和描述符已人工复核；
- [ ] 公共PS API计数已同步；
- [ ] policy E2E TB已按新manifest更新；
- [ ] 六个RTL testbench全部PASS；
- [ ] 三个PS工程均成功编译；
- [ ] 如XSA变化，三个BSP均已重建；
- [ ] JTAG冷启动和重复下载均成功；
- [ ] Cache、Weight和Instruction读回通过；
- [ ] 新self-test逐bit通过；
- [ ] 周期测试无失败和overrun；
- [ ] FP32一致性测试保存非空JSONL和summary；
- [ ] 文档中的性能和精度数字指向新模型证据；
- [ ] 旧模型结论已明确标记为历史数据。

完成以上步骤后，才可以把新ONNX视为已经在当前平台上完成部署，而不仅仅是“导出脚本运行成功”。
