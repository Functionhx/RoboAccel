# 08 UART 推理一致性 Demo 与实机结果

## 1. 目的

`03_vitis_consistency_test/` 用于验证任意上位机输入在原始 FP32 ONNX 和 FPGA INT16 实现之间的数值一致性。它独立于周期测速 demo，不复用固定 golden 输入。

每轮测试的闭环为：

```text
上位机生成 FP32 obs[25] + obs_history[125]
        ├─ ONNX ReferenceEvaluator → FP32 actions[6]
        └─ UART/CRC32 → PS Q8.8 量化 → PL 自动序列 → Q8.8 actions[6]
                                                        ↓ UART/CRC32
                         反量化、逐分量误差、argmax 和日志汇总
```

默认输入为固定种子的 clipped standard normal：`N(0,1)` 后裁剪至 `[-4,4]`。该分布与导出器随机验证口径一致，并严格落在 signed Q8.8 可表示范围内。比较基准使用未量化的原始 FP32 输入，因此误差包含输入、权重、激活和逐层舍入等完整定点化影响。

## 2. 软件与协议

- PS 裸机入口：`03_vitis_consistency_test/src/main.c`；
- UART 协议：`src/uart_protocol.{c,h}`；
- 上位机：`host/validate_consistency.py`；
- 独立复核：`host/verify_log.py`；
- 模型执行后端：ONNX 自带的 `onnx.reference.ReferenceEvaluator`，直接加载根目录 `fudan_policy.onnx`。

启动日志为 ASCII；出现 `CONSISTENCY_TEST_READY` 后切换成固定长度小端二进制包。请求和响应均包含 4-byte magic、协议版本、类型、payload 长度、32-bit 序号和 IEEE CRC-32。请求携带 150 个 FP32；响应携带状态、6 个 Q8.8 INT16 和本次 FPGA 端到端推理微秒数。

该设计可检测丢包、错包、截断、旧响应和 bit error，避免把串口通信错误误判成模型误差。

## 3. 运行

```powershell
Set-Location D:\rl_on_fpga
& .\03_vitis_consistency_test\run_test.ps1 -Port COM17 -Count 100 -Interval 1.0
```

脚本会依次构建 ELF、冷复位、下载 bitstream/ELF、等待 READY，并按 1 秒周期测试。每个 JSONL 样本保存全部输入、量化输入、两套输出、逐分量误差和计时，能够离线复算。

复核已有实机日志：

```powershell
python .\03_vitis_consistency_test\host\verify_log.py `
  .\03_vitis_consistency_test\logs\board_consistency_20260821_100samples.jsonl
```

## 4. 2026-08-21 COM17 实机结果

随机种子为 `20260821`，共 100 个独立输入、600 个动作输出分量：

| 指标 | 结果 |
|---|---:|
| 完成请求 | 100/100 |
| 序号范围/唯一数 | 1..100 / 100 |
| FPGA 与 Python 定点参考逐 bit 一致 | 100/100 |
| MAE | 0.0058493 |
| RMSE | 0.0076035 |
| 最大绝对误差 | 0.0329137 |
| FP32 输出平均绝对值 | 2.4570410 |
| 整体相对 MAE | 0.2381% |
| FP32 输出 RMS | 3.3542079 |
| 归一化 RMSE / 全局相对 L2 | 0.2267% |
| argmax 一致 | 99/100 |
| FPGA 推理计时 | 45 us（整数微秒分辨率） |

唯一 argmax 不一致发生在序号 5。FP32 第一、第二大输出仅相差 `0.0011059`，小于一个 Q8.8 LSB `0.00390625`，因此应解释为输出近似平局在量化后的次序翻转；该样本最大数值误差为 `0.0073615`，通信状态和 CRC 均正常。

### 4.1 相对误差口径

整体相对 MAE 使用所有 600 个输出的 MAE 除以 FP32 输出绝对值的总体均值：

\[
\mathrm{Relative\ MAE}
=\frac{\operatorname{mean}(|y_{FPGA}-y_{FP32}|)}
{\operatorname{mean}(|y_{FP32}|)}
=\frac{0.0058493}{2.4570410}
=0.2381\%.
\]

归一化 RMSE 使用误差 RMSE 除以 FP32 输出的均方根；它与把全部输出视为一个向量时的全局相对 L2 误差等价：

\[
\mathrm{NRMSE}
=\frac{\sqrt{\operatorname{mean}((y_{FPGA}-y_{FP32})^2)}}
{\sqrt{\operatorname{mean}(y_{FP32}^2)}}
=\frac{0.0076035}{3.3542079}
=0.2267\%.
\]

因此，当前 100 组实机样本可概括为：FPGA INT16 相对于原始 FP32 模型的整体数值误差约为 **0.23%**。

逐分量百分比误差 `|error/reference|` 在 reference 接近零时会被任意放大，不能直接对全部分量求普通 MAPE。仅统计 `|y_FP32|>=0.1` 的 581/600 个分量时，平均相对误差为 `0.5187%`，中位数为 `0.2396%`，95% 分位为 `2.0657%`。论文或报告的主结果应优先引用整体相对 MAE 和归一化 RMSE，同时说明该阈值化逐分量统计仅作为补充。

权威原始日志为 `03_vitis_consistency_test/logs/board_consistency_20260821_100samples.jsonl`，汇总为同目录的 `_summary.json`。SHA-256：

| 文件 | SHA-256 |
|---|---|
| `consistency_test.elf` | `C0F7DA31952C13CA2F593D11504A7274B221D09E8CBB1A703D3F067925B691EF` |
| 原始 JSONL | `F494D729E4D67564EDEAAEB047B1E766F69E7CFB2543853F1650BB1EDEA875DF` |
| summary JSON | `B70848B5E720AC79BB1718032C8E06622EF3318AAE9E10F482A93EAC72A0B124` |

独立复核还会从原始 ONNX 重新构建 Python 定点参考，并要求每个样本的 6 个 INT16 输出与 FPGA 逐 bit 相同。当前 100/100 均满足，因此观测到的 FP32 误差来自设计定义的量化近似，而不是 UART 或 RTL 实现偏差。

这些结果证明当前通信链、PS 量化/驱动和 PL 推理能对随机输入稳定工作。100 个样本适合 demo 和持续集成冒烟验证；若用于论文最终统计，建议把 `-Count` 提高到至少 1,000，并预先固定分布、种子、误差阈值和近似平局判据。
