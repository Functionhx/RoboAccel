# 09 2026-08-21 新网络实机验证报告

## 1. 对象与环境

| 项目 | 值 |
|---|---|
| FPGA | XC7Z010CLG400-2 |
| 工具链 | Vivado/Vitis 2020.2 |
| PL 时钟 | 100 MHz |
| 启动 | JTAG |
| UART | UART0，COM17，115200 8N1 |
| 模型 | `fudan_policy.onnx` |
| PL 接口版本 | `0x00010002` |

模型输入为 `obs[25]` 和 `obs_history[125]`，输出为 6 个 action。初始化时 PS 装载 7×GEMM、5×ELU 和 1×CONCAT 共 13 条指令；推理时 PL 自动执行整张网络。

## 2. 行为级和实现结果

`make -C tb all` 的六个自检全部 PASS，完整自动序列为 1,799 个 PL 周期（17.99 us），输出 golden 为：

```text
543 790 74 635 478 -796
```

完整 Zynq 系统路由后：

| 指标 | 结果 |
|---|---:|
| WNS | +0.400 ns |
| WHS | +0.026 ns |
| LUT | 10,919 |
| FF | 7,594 |
| RAMB36 | 52 |
| DSP48E1 | 72 |

报告源为 `02_vivado/system_reports/timing_summary.rpt` 和 `utilization.rpt`。

## 3. 周期测试

原始日志：`03_vitis_periodic_test/logs/board_sequence_v1_2_20260821_final.log`。

启动检查：

```text
CHECK,host_model_crc,pass,weight=0xB2B813DA
CHECK,pl_cache_and_program_readback,pass,commands=13
PERIODIC_TEST_READY
```

日志含 22 批、每批 100 次推理：

| 统计量 | 结果 |
|---|---:|
| 总推理数 | 2,200 |
| passed / failed | 2,200 / 0 |
| overrun | 0 |
| 加权平均延迟 | 45.392 us |
| 单次最小 / 最大 | 45.334 / 46.101 us |
| 非首批 period | 均通过 990,000--1,010,000 us 判据 |
| checksum | `0xBAC07F44`，全部一致 |

延迟包含两路输入 MMIO 写入、一次序列启动、轮询等待和输出读回；不含一次性 Cache/指令装载。PL 自动序列为 17.99 us，因此平均剩余 PS/MMIO 调度为约 27.402 us。

旧版逐算子 PS 调度的同模型基线为 75.081 us。新版本节省 29.689 us，端到端速度提升为 1.654×；PL 本身仅增加 0.27 us，净收益来自删除 13 次描述符提交和逐算子等待。

## 4. 通用驱动、中断和轮询

原始日志：

- `03_vitis/logs/board_sequence_v1_2_20260821_final.log`。

最终冷启动日志满足：

```text
Accelerator v1.2, mode=interrupt
Model/program loaded and instruction readback PASS: 132 vector words, 8 x 620 weight words, 13 instructions
Full cache readback PASS
Policy automatic-sequence polling self-test PASS
Policy automatic-sequence interrupt self-test PASS
Actions Q8.8: 543 790 74 635 478 -796
READY
```

100-run 中断 benchmark 为 4,572 us，即约 45.72 us/inference。整网只产生一次完成中断，不再为每个算子进入/退出 GIC。

## 5. 复位问题与修复

验证中发现，仅使用 `rst -system` 重复下载时，上一应用遗留的 GIC active/pending 状态可能使第一次中断 self-test 失败，而相同模型的轮询路径仍正确。

`03_vitis/run_jtag.tcl` 已改为：

```text
SRST -> system reset -> FPGA program -> ps7_init/post_config
     -> Cortex-A9 register reset -> ELF download
```

加入板级 `rst -srst` 后连续两次自动冷启动均通过。该结果说明失败来自 PS/GIC 复位完整性，而非 PL 数值计算或模型 Cache。

## 6. 结论

当前新网络已经获得以下相互独立的证据：

1. Python 定点 golden 与完整 RTL 行为仿真一致；
2. 100 MHz 完整系统满足 setup/hold 时序；
3. 实机模型 CRC 和所有已装载 Cache 数据读回一致；
4. 2,200 次轮询推理全部正确且周期稳定；
5. 冷启动后的中断与轮询自动序列 self-test 均通过。

因此，当前 bitstream、PS 软件、导出模型和指令程序可视为同一已验证版本。SD 卡物理启动仍未在本报告中验证。

## 7. 证据 SHA-256

| 文件 | SHA-256 |
|---|---|
| `fudan_policy.onnx` | `C2859573CBCE1FAEF78D4333A2AD461389DE9760B416C0896459C8D6EF4D263E` |
| `02_vivado/output/rl_system.bit` | `E444F43C3F2AD76DA9CFCA1188D3B91DE41AFE9A3DC482EA05C8CC03FE44AB07` |
| `02_vivado/output/rl_system.xsa` | `4C7B16A9203C5170EE521C4B364760715CC8DD9477FF23B8EEC8912C33ADF59A` |
| `03_vitis/build/rl_policy_app.elf` | `C5438B0B4B986E44A09B66E8908CF135221A26A280F874100422EACCFE08E08E` |
| `03_vitis_periodic_test/build/periodic_test.elf` | `8C7814B1337DB8C2E847BCE59C0FE288C8B42E432D28F086D5DE0760242D2BC8` |
| `03_vitis/logs/board_sequence_v1_2_20260821_final.log` | `EB572F98249DA7277B5A7A52A7FBA56E891B4793A3037B5E6389AED3008697B2` |
| `03_vitis_periodic_test/logs/board_sequence_v1_2_20260821_final.log` | `61939E5D71A7C0DE1809771F7C8CAD4D5E5F8FDA6BFD6B6D86C57CC90378AC13` |
