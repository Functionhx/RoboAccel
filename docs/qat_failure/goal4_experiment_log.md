# Goal 4 · 实验日志

按时间顺序。每条记录：目的、配置、结果、以及**这条实验排除了什么**。
所有训练臂共享：`locomotion_v2`、seed 1、4096 envs、2000 iters、
从 `SOLID_FP32_V2.pt`（sha256 `ab20c190…3ec90`）热启动。
差异只在"变量"一列。

---

## L1 · 定位实验（§3 A/B/C）

* **目的**：区分"训练失败"与"执行精度失败"。
* **变量**：同一个 W8A8 checkpoint，三条执行路径——
  A 浮点、B fake-quant、C 导出定点。evaluator / seed=11 / 128 envs 全同。
* **结果**：A 在**七个 segment 全部 0.000**；转向在 A 和 B 中**都是** 0.000。
* **排除**：goal4.md §2 第 4、5、6、8 项作为转向失败主因。
* **产出**：`docs/goal4/01_localization.md`，`qat/artifacts/v2/m_L_*`

## L2 · 理想舍入消融

* **目的**：检验 RTL 的 floor 舍入偏置是否导致转向失败。
* **变量**：`--ideal-rounding`（round-half-away-from-zero）。
* **结果**：转向仍 **0.000**。
* **排除**：舍入偏置。
* **产出**：`qat/artifacts/v2/m_L_qat_ideal`

## L3 · 逐层敏感度

* **结果**：`enc2`（latent 层）只占总量化误差 **13.2%**。
* **排除**：latent 瓶颈假设。
* **产出**：`qat/artifacts/v2/layer_sensitivity_w8a8.json`

## L4 · 偏航奖励重加权（×2 / ×4）

* **目的**：检验"奖励失衡饿死偏航"。
* **变量**：`--reward-ang-vel 2.0` / `4.0`（带构造后读回断言）。
* **结果**：奖励贡献精确按 **2.00× / 4.00×** 缩放，
  `yaw_rmse_rad_s` 0.1546 → **0.1533 / 0.1540**（无变化）。
* **排除**：目标 / 奖励失衡作为主因。
* **备注**：两次运行在 1991 / 1977 步被外部信号终止（会话进程组清理，
  非训练故障）。最后保存的 checkpoint 是 `model_1750.pt`；
  `metrics.jsonl` 完整到终止点，上述结论取自末 50 步均值，不受影响。
  **操作教训**：长训练必须用 `setsid` 启动。
* **产出**：`qat/checkpoints/qat_v2_yaw{2,4}/`

## L5 · 前一次无效实验的隔离

* 第一版偏航实验中 `--reward-ang-vel` 是**静默空操作**
  （`make_env` 在覆盖之前调用，`LeggedRobot` 在构造时就把系数乘上了 dt）。
  三个不同奖励权重产生了 **bit-identical** 的权重（差 0.000e+00）。
* 已修复（覆盖前置 + 从**构造后的 env** 读回断言），
  旧结果隔离在 `qat/artifacts/v2/INVALID_yaw_experiment/`。

## L6 · KL 放大测量（离线，无仿真）

* **目的**：检验"量化噪声击穿 PPO 自适应学习率控制器"。
* **方法**：对同一组参数施加一次 Adam 步长（`|Δw| ≈ lr` 逐元素），
  在量化开 / 关两条前向路径上测同一个策略 KL。
* **结果**（lr = 1e-5，控制器阈值 0.01）：
  float 4.06e-07 · W16A16 1.44e-04 · W8A16 8.78e-04 ·
  **W8A8 6.73e-02** · W4A8 1.94e-02。
* **判定**：**W8A8 是精度阶梯上唯一噪声地板高过阈值的配置。**
* **产出**：`qat/scripts/kl_amplification.py`，
  `qat/artifacts/v2/kl_amplification_{qat,fp32}.json`

## L7 · 训练轨迹审计

* **结果**：W8A8 学习率 **2000/2000 次迭代在 1e-5 下限**；
  FP32 对照 10/2000。W8A8 的 KL 在 **100%** 的迭代超过阈值，对照 0.1%。
* **附带发现**：编码器由独立的 `extra_optimizer` 驱动，
  学习率固定 1e-3，**不受该控制器管辖**；
  `Encoder/policy_kl` W8A8 是对照的 **100 倍**。

## L8 · `desired_kl = 0.08`（已中止）

* **目的**：把阈值抬到量化噪声地板之上。
* **结果**：23 步后 KL = 0.147，落在 `[desired_kl/2, desired_kl*2]`
  = [0.04, 0.16] 的**死区**内——既不降也不升，学习率仍停在下限。
* **处置**：**中止**（约 2 分钟，无损失），改用直接固定学习率的 L9。
  死区本身是可预期的空结果，不值得 2.6 小时 GPU。

## L9 · 固定学习率 W8A8 —— **进行中**

* **目的**：给 W8A8 与 FP32 对照**相同的步长**。
* **变量**：`--schedule fixed --learning-rate 2.563e-4`
  （= FP32 对照 2000 步学习率的中位数），带优化器读回断言。
* **可证伪性**：若 `yaw_rmse` 仍停在 0.15，则 H6 为真但**不是充分原因**。
* **run**：`qat_v2_fixlr`

## L10 · W8A16 正对照 —— **进行中**

* **目的**：机制预测 W8A16 的噪声地板（8.78e-04）低于阈值，
  因此学习率**不会**被钉住，QAT 正常。
* **变量**：`--quant W8A16`，其余与 `qat_w8a8_v2_long` 相同。
* **run**：`qat_v2_w8a16`

---

## 结果表

见 [`goal4_results_table.md`](goal4_results_table.md)（由
[`collect_results.py`](collect_results.py) 从原始 `metrics.jsonl` 生成，
无手工转录），机器可读版本 `goal4_results.json`。
