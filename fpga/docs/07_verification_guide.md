# 07 测试与验证指南

验证遵循“模型导出 → RTL 行为仿真 → Vivado 实现 → PS 编译 → JTAG/UART 实机”的顺序。综合通过不能替代功能仿真，UART 结论必须对应非空原始日志。

## 1. 模型导出和数值检查

```powershell
Set-Location D:\rl_on_fpga
python .\tools\export_policy.py --model .\fudan_policy.onnx --out .\generated --samples 1000 --seed 7 --c-out .\03_vitis\src
```

导出器校验两个输入、Gemm/Elu/Concat 拓扑、Cache 容量和定点路径，并生成：

- `generated/policy_map.json`；
- Vector/Weight Cache hex；
- 32×128-bit `instruction_program.hex` 和指令编码 manifest；
- 两路 self-test 输入和 action golden；
- `03_vitis/src/rl_policy_data.{c,h}`。

当前 1,000 组标准正态随机双输入对比指标为 MAE 0.006934、RMSE 0.034239、argmax 一致率 99.8%。该统计只用于量化回归，不等同于任务环境中的最终策略回报。

## 2. RTL 行为级仿真

```powershell
make -C tb all
```

通过时必须看到：

```text
PASS tb_primitives
PASS tb_gemm_engine
PASS tb_concat_engine
PASS tb_instruction_sequencer
PASS tb_axil_top
Policy automatic sequence cycles: 1799
PASS tb_policy_e2e
```

| testbench | 覆盖 |
|---|---|
| `tb_primitives.sv` | 舍入/饱和、MAC、ELU、NORM |
| `tb_gemm_engine.sv` | 非整 tile K/N、bias、权重和输出 |
| `tb_concat_engine.sv` | 非 8 对齐的 25+3 拼接和重打包 |
| `tb_instruction_sequencer.sv` | 指令读回、连续发射、非法长度/opcode 中止 |
| `tb_axil_top.sv` | AXI 握手、版本、Cache/指令窗口和一次启动多算子 |
| `tb_policy_e2e.sv` | 一次启动自动执行 13 条命令并匹配 action golden |

1,799 cycle 是从序列启动到整网完成的 PL 延迟，包含指令取出/分派但不含 PS 的输入/输出 AXI4-Lite 软件开销。相对原 1,772 个纯算子周期，调度器只增加 27 cycle。

## 3. Vivado 完整实现

```powershell
Set-Location D:\rl_on_fpga\02_vivado
& 'D:\XILINX\Vivado\2020.2\bin\vivado.bat' -mode batch -source .\create_system.tcl -nojournal
```

输出：

- `02_vivado/output/rl_system.bit`；
- `02_vivado/output/rl_system.xsa`；
- `02_vivado/system_reports/timing_summary.rpt`；
- `02_vivado/system_reports/utilization.rpt`。

2026-08-21 当前结果：

| 项目 | 结果 |
|---|---:|
| WNS | +0.400 ns |
| WHS | +0.026 ns |
| LUT | 10,919 |
| FF | 7,594 |
| RAMB36 | 52 |
| DSP48E1 | 72 |

必须确认报告含 `All user specified timing constraints are met`，且 route failed nets 为 0。

## 4. PS/BSP 构建

XSA 改变后不能复用旧 BSP。重新生成平台：

```powershell
Set-Location D:\rl_on_fpga\03_vitis
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' .\create_workspace.tcl
& .\build_software.ps1
```

周期工程：

```powershell
Set-Location D:\rl_on_fpga\03_vitis_periodic_test
& 'D:\XILINX\Vitis\2020.2\bin\xsct.bat' .\create_platform.tcl
& .\build.ps1
```

产物分别为 `03_vitis/build/rl_policy_app.elf`、`03_vitis/boot/BOOT.BIN` 和 `03_vitis_periodic_test/build/periodic_test.elf`。

## 5. JTAG 与 COM17 实机验证

前提：开发板设为 JTAG 启动，UART0 为 COM17、115200 8N1。

自动周期测试：

```powershell
Set-Location D:\rl_on_fpga
& .\03_vitis_periodic_test\run_and_capture.ps1 -Port COM17 -Seconds 30 -LogFile .\03_vitis_periodic_test\logs\board_sequence_v1_2_20260821_final.log
```

脚本会复位、下载 bit/ELF、采集 UART，并自动要求：

- 恰好一个 `PERIODIC_TEST_READY`；
- 至少三批；
- 每批 attempted=passed=100、failed=0；
- 非首批 period 在 990,000--1,010,000 us；
- `overrun=0`。

通用中断程序自动下载并捕获：

```powershell
Set-Location D:\rl_on_fpga
& .\03_vitis\run_and_capture.ps1 -Port COM17 -Seconds 15
```

UART 应依次出现版本 1.2、Cache/13 条指令装载读回、polling self-test PASS、interrupt self-test PASS、正确 actions 和 READY。

通用 `run_jtag.tcl` 必须保留 `rst -srst`，否则重复下载可能残留 GIC active/pending 状态并造成第一次中断测试间歇失败。

## 6. 当前实机基线

当前周期日志 `03_vitis_periodic_test/logs/board_sequence_v1_2_20260821_final.log` 包含 22 批、2,200 次推理，全部正确。加权平均 45.392 us，单次范围 45.334--46.101 us，checksum 恒为 `0xBAC07F44`。

通用程序日志 `03_vitis/logs/board_sequence_v1_2_20260821_final.log` 证明 GIC 和轮询路径均通过；100 次推理约 4.572 ms。详细摘录与口径见 [09 实机验证报告](09_hardware_test_20260821.md)。

### 随机输入 UART 一致性测试

```powershell
& .\03_vitis_consistency_test\run_test.ps1 -Port COM17 -Count 100 -Interval 1.0
python .\03_vitis_consistency_test\host\verify_log.py `
  .\03_vitis_consistency_test\logs\board_consistency_20260821_100samples.jsonl
```

该测试以 CRC32 和序号保护 150 个 FP32 输入及 6 个 Q8.8 输出，上位机直接运行原始 ONNX FP32 模型并保存完整 JSONL。当前 100 个实机样本的独立复核为 PASS：MAE 0.005849、RMSE 0.007603、整体相对 MAE 0.2381%、归一化 RMSE 0.2267%、最大绝对误差 0.032914、argmax 99%。具体误差定义、分母口径和近似平局说明见 [08 一致性验证](08_consistency_validation.md)。

## 7. 修改后的最低回归

| 修改类型 | 最低要求 |
|---|---|
| ONNX/量化/布局 | 重导出、Python golden、`tb_policy_e2e` |
| GEMM/ELU/NORM/CONCAT | 对应单元 TB + AXI/策略 TB |
| Cache 深度或 AXI 协议 | RTL 全回归 + Vivado + PS 全量读回 |
| bitstream/XSA | 时序/资源报告 + 三个 BSP 重建 |
| PS 指令程序/API | 指令读回 + golden self-test + 周期实机测试 |
| JTAG 脚本 | 冷复位、版本、READY、异常状态 |
| UART 一致性 demo | 协议单元测试 + ARM 构建 + 连续序号实机日志 + `verify_log.py` |
