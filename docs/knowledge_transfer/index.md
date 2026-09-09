# 知识迁移文档 — RoboAccel 强化学习策略量化项目

**目的**：让你在没有当前 Claude session 的情况下，独立拥有并继续这个项目。

写于 2026-09-09。文档顺序按 **重建难度** 排列，而不是按教科书的讲解顺序。
代码结构和通用概念可以通过读仓库恢复；但**某个决策背后的推理**、
**三个被推翻的假设**、以及**四个产生了"看起来合理但错误"结果的 bug**，
一旦这个 session 结束就完全无法恢复。

## 建议阅读顺序

| # | 文档 | 为什么重要 |
|---|---|---|
| [05](05_debugging_history.md) | **调试历史** | 四个 bug 产生过自信但错误的结果。可迁移的部分是"识别信号" |
| [04](04_design_decisions.md) | **设计决策** | 为什么算术要复现硬件的 *bug*，为什么评估器是借用而非自建 |
| [06](06_failed_hypotheses.md) | **失败假设档案** | 三个符合证据、但被实验推翻的机理解释 |
| [09](09_experiment_interpretation.md) | **实验解读指南** | 哪些数字是结论性的，哪些是假象，哪些指标会骗人 |
| [08](08_verification_chain.md) | **验证链** | 五个实现如何做到 bit-identical，以及如何复现 |
| [11](11_operations_playbook.md) | **操作手册** | 精确命令、环境、以及浪费过数小时的陷阱 |
| [00](00_project_map.md) | 项目地图 | 端到端数据流，逐文件对应 |
| [03](03_math_to_code.md) | 数学 → 代码 → 硬件 | Q8.8、requantization、ELU ROM、CONCAT 约束 |
| [02](02_code_reading_path.md) | 代码阅读路径 | 有序的实现走查 |
| [07](07_insights.md) | 研究洞见 | 超出这个机器人之外仍然成立的结论 |
| [01](01_prerequisite_map.md) | 前置知识 | 按需补充的背景 |
| [10](10_glossary.md) | 术语表 | 中英对照 |
| [12](12_learning_curriculum.md) | 学习路径 | 带检查点的能力养成路线 |

## 证据标记约定（evidence convention）

每条论断都会标记：

* **[已验证]** — 可从本仓库的文件、日志或实测复现，并给出路径。
* **[重建]** — 从留存的 artifact 推断得出，推断过程会写明。
* **[假设]** — 与证据一致，但**未经实验检验**。
* **[无法恢复]** — `Unable to reconstruct from available evidence.`

没有标记的默认视为 **[已验证]**。

## 一段话版本（the one-paragraph version）

一个轮腿机器人（wheel-legged robot）由一个小 MLP 策略控制：25 维观测
（observation）、5 帧历史编码器（history encoder）、4 层 actor、6 维动作
（action）、38,400 次 MAC。这个策略必须以**定点整数算术**（fixed-point
integer arithmetic）运行在 Zynq-7000 FPGA 加速器（RoboAccel，约 18 µs）
或 STM32H723（约 259 µs）上。项目要回答的问题是：**在训练时就把这套整数
算术放进闭环里模拟**（QAT，quantization-aware training），是否比训练完再
量化（PTQ，post-training quantization）保留更多控制性能？

在修正后的仿真上，答案是：**在实际部署的 16-bit 精度下量化本身是无损的，
QAT 没有必要；在 8-bit 激活下，QAT 能恢复平衡与直线行驶，但从未恢复转向。**

英文表述（for papers）：*At the deployed 16-bit precision, quantization is
lossless and QAT is unnecessary. At 8-bit activations, QAT restores wheel
support and straight-line locomotion but never restores turning.*

这句话能支持什么、不能支持什么，见 [09](09_experiment_interpretation.md)。
