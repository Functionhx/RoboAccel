# MINI_7010 + Vivado/Vitis 2026.1 实机复现

本文记录将 XC7Z010 RL INT16 加速器迁移到 MINI_7010 开发板和 Linux
Vivado/Vitis 2026.1 的可复现流程。所有板端步骤均为 JTAG 临时下载，不写入
QSPI 或 SD 卡。

## 1. 环境与硬件

| 项目 | 值 |
|---|---|
| 开发板 | MINI_7010 |
| FPGA | XC7Z010-CLG400，按保守的 `-1` speed grade 构建 |
| PS 时钟 | 33.333333 MHz 输入，Cortex-A9 实际 666.666687 MHz |
| PL 时钟 | FCLK0 100 MHz |
| DDR | 512 MiB、x16 DDR3L，533.333 MHz |
| UART | PS UART1，MIO 48..49，115200 8N1 |
| 加速器地址 | `0x43C00000` |
| 工具链 | Vivado/Vitis 2026.1 |
| Python | `uv`、CPython 3.13 |

项目 Python 依赖由根目录 `pyproject.toml` 和 `uv.lock` 管理。Vitis 平台生成
脚本必须由 `vitis -s` 运行，因为 `vitis` 模块只存在于 AMD 自带的 Python
环境；这不参与项目 Python 依赖管理。

## 2. Python 与 RTL 回归

```bash
source /home/as/vllm/fpga/env/amd-fpga.sh
uv sync --no-install-project
uv run --no-sync python tools/export_policy.py \
  --model fudan_policy.onnx --out generated \
  --samples 1000 --seed 7 --c-out 03_vitis/src
make -C tb all
```

2026-09-04 结果：六个 testbench 全部通过，完整策略序列为 1799 cycle。
模型 SHA-256 为：

```text
c2859573cbce1faef78d4333a2ad461389de9760b416c0896459c8d6ef4d263e
```

## 3. Vivado 实现

纯加速器 OOC：

```bash
vivado -mode batch -nojournal -nolog -notrace \
  -source 02_vivado/build_ooc_mini7010.tcl
```

完整 PS + AXI 系统：

```bash
vivado -mode batch -nojournal -nolog -notrace \
  -source 02_vivado/create_system_mini7010.tcl
```

100 MHz、`xc7z010clg400-1` 的结果：

| 项目 | OOC | 完整系统 |
|---|---:|---:|
| WNS | +0.037 ns | +0.104 ns |
| WHS | +0.008 ns | +0.019 ns |
| LUT | 10,601 | 11,351 |
| FF | 7,720 | 8,088 |
| RAMB36 | 52 | 52 |
| DSP48E1 | 72 | 72 |

完整系统约使用 64.5% LUT、23.0% FF、86.7% BRAM36 和 90% DSP48E1。

## 4. Vitis 2026.1 BSP 与 ELF

```bash
mkdir -p /home/as/fpga-build/rl-vitis-cli
cd /home/as/fpga-build/rl-vitis-cli
vitis -s /home/as/vllm/fpga/projects/rl_accel/03_vitis/create_platform_mini7010.py

cd /home/as/vllm/fpga/projects/rl_accel
./03_vitis/build_software_mini7010.sh
```

Vitis 2026.1 使用 Python API 代替已移除的 XSCT 工程 API。生成的 ELF 位于：

```text
03_vitis/mini7010_build/rl_policy_app_mini7010.elf
```

链接后脚本会强制检查 `_vector_table`、`main` 和 ELF 入口。当前入口为
`0x01000000`。

## 5. JTAG 实机运行

终端 A：

```bash
source /home/as/vllm/fpga/env/amd-fpga.sh
hw_server -s tcp::3121
```

终端 B：

```bash
source /home/as/vllm/fpga/env/amd-fpga.sh
cd /home/as/vllm/fpga/projects/rl_accel
xsdb 03_vitis/jtag_smoke_mini7010.tcl
xsdb 03_vitis/run_jtag_mini7010.tcl
sleep 3
./03_vitis/read_mailbox_mini7010.sh
```

JTAG smoke test验证：

- DDR `0x01000000` 写读；
- PL 接口版本 `0x00010002`；
- Vector Cache word 511 的 128-bit 写入和读回。

应用 mailbox 验证：

```text
MAILBOX_ERRORS=0
MAILBOX_ACTIONS=543 790 74 635 478 -796
MAILBOX_STATUS=PASS
```

连续五批观测的 100 次推理耗时为 4675--4679 us，即
46.75--46.79 us/inference。PL 自动序列本身仍为 1799 cycle，即 17.99 us；
剩余时间主要是 PS 通过 AXI4-Lite 写入 150 个输入、轮询和读回输出。

## 6. 2026.1 兼容性说明

1. 2026.1 SDT BSP 不再把旧 `xtime_l.h` 安装进公共 include，应用改为优先包含
   `xiltimer.h`。
2. 自动生成的 `COUNTS_PER_SECOND` 展开为未加括号的表达式
   `XPAR_CPU_CORE_CLOCK_FREQ_HZ/2`。除法时必须写成
   `(u64)(COUNTS_PER_SECOND)`，否则结果会少算四倍。
3. 新 BSP 没有生成旧式 `XPAR_FABRIC_RL_ACCEL_0_IRQ_INTR` 宏，因此当前应用
   自动采用 polling。PL 中断线路仍在 XSA 中，后续需要按 SDT 中断描述适配驱动。
4. MINI_7010 板载 UART 是 UART1；原作者硬件是 UART0，不能复用其 PS preset。
5. 当前板载 CH340 USB-UART 尚未连接到开发主机，因此使用 JTAG mailbox 获取
   action、错误数和计时结果。
