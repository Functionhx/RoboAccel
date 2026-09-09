# ActuateX 训练端与 FPGA 推理端边界

更新日期：2026-09-04

## 结论

轮腿强化学习的训练、感知状态估计、策略评估和模型导出归
[ActuateX](https://github.com/Functionhx/actuatex) 管理；本仓库只负责把已冻结的策略模型
量化并在 Zynq-7010 上执行。RM-Cortex 不再作为轮腿策略训练来源。

训练端合同 `actuatex.wheel-legged-history-policy/v1` 的 SHA-256 为：

```text
220e901d9785f2568cfeea917c1f3423cb6b0512b271970eb21d3c675eb12e5f
```

合同固定以下接口：

```text
obs_history(125) -> 128 -> ELU -> 64 -> ELU -> latent(3)
obs(25) + latent(3) -> 128 -> ELU -> 64 -> ELU -> 32 -> ELU -> actions(6)
```

- 5 帧历史按最旧到最新排列；
- 25 维当前观测和 6 维动作的顺序由训练端合同定义；
- policy 为 100 Hz，actuator/controller 为 500 Hz；
- 线性层共 38,400 个权重，与本加速器当前 datapath 匹配；
- 模型交付必须包含合同 JSON、ONNX SHA-256、FP32/INT16 golden、calibration set 和导出版本。

## 当前模型身份

根目录 `fudan_policy.onnx` 的 SHA-256 是：

```text
c2859573cbce1faef78d4333a2ad461389de9760b416c0896459c8d6ef4d263e
```

这个文件名是历史兼容名称。该哈希与当前固定的 Fudan/XYEGA 六个参考 ONNX 均不相同，
因此不能把它描述为复旦仓库原始权重；它只是采用了 Fudan-compatible 网络拓扑的当前
FPGA 权威测试模型。重命名会影响既有脚本和证据，现阶段保留文件名并在文档中澄清身份。

固定的外部参考哈希如下，仅用于来源核对，不进入本仓库：

| 参考策略 | SHA-256 |
|---|---|
| XYEGA stable | `37084bdc04f719297cfaf87611fe665a1c680446615450879970599fd7f4f3ea` |
| XYEGA spin | `e5dd968f2ed859651648b44684107fa53e95318d3824b6c1692030f1fcb65a7e` |
| XYEGA upstairs | `bfd2a6f1a31c66f237bfbaa65e9809631ba4ea26cc40d6f3f7c8d0d004ebcb64` |
| XYEGA jump | `54c9972506a38e94a6b5de30e79ad689a7b9fd92e72640874eee1bdb4c36dc78` |
| Fudan plane/upstairs | `2e0907d67d314e5fb12358231089b951033aa6b7d76b60aa8c6e7a330d3549f5` |
| Fudan jump | `8215af3333cbecfb684a7e24e696e8f2cd65194f005953f242762eee3b72f5d6` |

Fudan 和 XYEGA 仓库在固定 revision 上未发现仓库级软件许可证。本仓库不复制或再发布其
权重；后续 FPGA 模型由 ActuateX 自主训练并带完整 provenance 交付。

## 跨仓库验收门槛

每次替换模型按以下顺序验收：

1. 训练端校验合同摘要、输入顺序、缩放、frame order 和静态 shape；
2. ONNX Runtime 与 eager PyTorch 对同一输入逐值比较；
3. `tools/export_policy.py` 生成定点 image、mapping 和 golden；
4. Python fixed-point reference 与 RTL policy testbench bit-exact；
5. Zynq JTAG self-test 与随机输入一致性测试；
6. 达到上述门槛后，模型才能进入 shadow/HIL；FPGA 输出不得直接绕过 CPU/MCU safety。

训练端仓库地图与迁移记录见 ActuateX 的
`docs/WHEEL_LEGGED_REPOSITORY_MAP.zh-CN.md`。

