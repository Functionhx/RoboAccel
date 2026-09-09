# Goal 4 · 假设账本（Hypothesis Ledger）

对应 `goal4.md` §17。每条假设记录：陈述、预测、检验方式、判定、证据文件。
判定只有四种：**已证实 / 已证伪 / 部分支持 / 未检验**。
"看起来合理"不是判定。

---

## H0 —— 中心工作假设

> **陈述**：保留一个可靠的参考策略，只学习低精度执行所需要的修正
> （*Preserve a reliable reference policy and learn only the correction required
> by low-precision execution.*）

**当前状态**：**未检验（但前提已被强化）**。

H0 的前提是"现有 QAT 会破坏参考策略"。这一前提已由 [E1](#e1) 直接证实
（arm A 七个 segment 全部 0.000）。因此 H0 从"听起来合理"升级为
"针对一个已测量到的破坏机制的对策"。H0 本身仍待 §4 的
frozen-base residual-QAT 实验检验。

---

## 已证伪的假设

### H1 —— 激活分辨率不足（goal4.md §2 第 4 项） · **已证伪（作为转向失败主因）**

* **陈述**：INT8 激活的分辨率不足以表达转向所需的控制量。
* **预测**：若成立，去掉量化后（浮点执行同一组参数）转向应当恢复。
* **结果**：arm A（QAT 参数、浮点执行）`turn_left = turn_right = 0.000`。
* **判定**：**证伪**。分辨率不是主因；转向在**没有任何量化**时同样失败。
* **证据**：`docs/goal4/01_localization.md`，`qat/artifacts/v2/m_L_qat`

### H2 —— 激活裁剪 / scale 分配（§2 第 5 项） · **已证伪（同上）**

* **预测**：同 H1，浮点执行应当恢复转向。
* **结果**：同 H1。另有独立证据：W8A8 的实测裁剪比例为 **0.0000%**。
* **判定**：**证伪**。
* **证据**：`qat/artifacts/v2/diag_w16a16.json`，同上定位实验。

### H3 —— 共享激活网格约束（§2 第 6 项） · **已证伪（同上）**

* **判定**：**证伪**，理由同 H1。

### H4 —— fake-quant 与导出定点的数值不一致（§2 第 8 项） · **已证伪**

* **预测**：若成立，失败应当出现在 B（fake-quant）与 C（定点）之间。
* **结果**：A 与 B **都**失败，且 B↔C 在 4000 个样本上 bit-exact。
* **判定**：**证伪**。
* **证据**：`qat/golden_vectors/`，`scripts/bitexact_check.py`

### H5 —— 目标函数 / 奖励失衡（§2 第 2 项） · **已证伪（作为主因）**

* **陈述**：`tracking_lin_vel = 2.0` 压过 `tracking_ang_vel = 1.0`，
  QAT 把 INT8 容量花在奖励占优的轴上，饿死了偏航。
* **预测**：把 `tracking_ang_vel` 提高到 2.0 / 4.0，转向应当改善。
* **结果**（同 seed、同 task、同 init、同 2000 步）：

  | 运行 | ang_vel 奖励贡献 | `yaw_rmse_rad_s` |
  |---|---|---|
  | FP32 对照 | 0.00986 | **0.0584** |
  | W8A8 ×1 | 0.00913 | 0.1546 |
  | W8A8 ×2 | 0.01826（**2.00×**） | 0.1533 |
  | W8A8 ×4 | 0.03646（**4.00×**） | 0.1540 |

  奖励项确实按 1/2/4 精确缩放（证明覆盖真正生效），
  **而 `yaw_rmse` 变化 0.8%，即没有变化。**
* **判定**：**证伪**。奖励重加权对 W8A8 完全无效。
* **机制解释**：三次运行的学习率都被钉在 1e-5 下限上 **100% 的迭代**
  （见 H6）。**当优化器无法迈步时，改变目标函数不可能有任何效果。**
* **证据**：`qat/checkpoints/qat_v2_yaw{2,4}/metrics.jsonl`，本文件 §E3

### H6-alt —— 舍入偏置导致转向失败 · **已证伪**

* **预测**：把 RTL 的 floor 语义换成理想的 round-half-away-from-zero，转向应当恢复。
* **结果**：理想舍入下转向仍为 **0.000**。
* **判定**：**证伪**。
* **证据**：`qat/artifacts/v2/m_L_qat_ideal`

### H7-alt —— latent 瓶颈 · **已证伪**

* **陈述**：3 维 latent 在 INT8 下坍塌。
* **结果**：`enc2`（产生 latent 的层）对总误差的贡献仅 **13.2%**。
* **判定**：**证伪**。
* **证据**：`qat/artifacts/v2/layer_sensitivity_w8a8.json`

---

## 已证实的假设

### E1 —— QAT 学到的策略**依赖**量化器，而不是**容忍**量化器 · **已证实**

* **陈述**：*The QAT parameters are not a quantization-tolerant policy; they are
  a policy that requires the quantizer. Fake-quant has become part of the
  learned function.*
* **证据**：同一 checkpoint、同一 evaluator、同一 seed：

  | segment | FP32 对照 | PTQ | **A 浮点** | **B fake-quant** |
  |---|---|---|---|---|
  | forward | 1.000 | 0.000 | **0.000** | **1.000** |
  | height | 1.000 | 0.539 | **0.000** | **1.000** |
  | stop | 1.000 | 0.984 | **0.000** | 0.938 |
  | turn_left | 1.000 | 0.758 | 0.000 | 0.000 |
  | turn_right | 1.000 | 0.523 | 0.000 | 0.000 |

  A 在 B 完美通过的 segment 上也是 0.000。
* **文件**：`docs/goal4/01_localization.md`

### H6 —— 量化噪声击穿 PPO 的自适应学习率信任域 · **已证实**

这是 Goal 4 的核心机制发现，详见 [`goal4_final_analysis.md`](goal4_final_analysis.md)。

* **陈述**：rsl_rl 用策略 KL 调节学习率；fake-quant 使策略输出对权重
  **不连续**，测得的 KL 由"有多少权重跨过量化边界"支配，而这几乎不随
  学习率变化。控制器唯一的调节手段（降低学习率）**无法降低它所调节的量**。
* **预测 1**：W8A8 运行的学习率会被钉在 1e-5 下限。
  **实测：2000/2000 次迭代（100%）在下限**，FP32 对照仅 10/2000（0.5%）。
* **预测 2**：离线测量单步扰动引起的 KL，W8A8 应超过控制器阈值 0.01，
  而 W16A16 / W8A16 应低于它。
  **实测（`scripts/kl_amplification.py`，QAT checkpoint，lr = 1e-5）**：

  | 精度 | 单步策略 KL | vs 浮点 | 控制器 |
  |---|---|---|---|
  | float | 4.06e-07 | 1× | 自由 |
  | W16A16 | 1.44e-04 | 354× | 自由 |
  | W8A16 | 8.78e-04 | 2 162× | 自由 |
  | **W8A8** | **6.73e-02** | **165 737×** | **钉在下限** |

* **预测 3**：把 PPO 每次迭代的 20 个 minibatch 步累计进去
  （`num_learning_epochs=5 × num_mini_batches=4`），
  等效步长 ≈ 2e-4，离线表在 lr=2.56e-4 处给出 W8A8 KL = **0.125**。
  **实测训练中位 KL = 0.065，实测 kl08 arm 早期 KL = 0.147。** 量级吻合。
* **判定**：**已证实**（三个独立预测均命中）。
* **证据**：`qat/artifacts/v2/kl_amplification_{qat,fp32}.json`，
  `qat/checkpoints/*/metrics.jsonl`，
  `fudan_rl_wheel_leg/.../rsl_rl/algorithms/ppo.py:208-215`

---

## 待检验的假设

### H8 —— 解除学习率钳制即可恢复 W8A8 · **进行中**

* **陈述**：如果 H6 是主因，那么给 W8A8 与 FP32 对照**相同的步长**，
  转向应当恢复。
* **实验**：`qat_v2_fixlr` —— `--schedule fixed --learning-rate 2.563e-4`
  （FP32 对照 2000 步的学习率中位数），其余与 `qat_w8a8_v2_long` 完全一致。
* **正对照**：`qat_v2_w8a16` —— 机制预测其学习率**不会**被钉住。
* **可证伪性**：若 `qat_v2_fixlr` 的 `yaw_rmse` 仍停在 0.15，
  则 H6 是真实的但**不是**转向失败的充分原因，责任回到 H0 / 残差方案。

### H9 —— 冻结 base + 残差 QAT（goal4.md §4） · **已设计，未执行**

anchor = W8A8 PTQ 策略（冻结），残差 `alpha*tanh(r_phi(x))`，
1,126 个可训练参数对 38,825 个冻结参数，末层零初始化使 `t=0` 时
逐位等于 anchor。设计与硬件映射见
[`goal4_experiment_designs.md`](goal4_experiment_designs.md) §1。

### H10 —— FP32 教师行为保持（goal4.md §7） · **已设计，未执行**

`L = L_PPO + beta * KL(teacher || student)`，教师为冻结的 `SOLID_FP32_V2`，
KL 在 **student 自己的** on-policy 状态上计算。设计见同文件 §2。

### 两者共同的先决条件（来自 Goal 5）

**必须带 `--schedule fixed --learning-rate 2.563e-4`。**
否则残差分支会撞上同一个学习率下限，实验将无法区分
"方案无效"与"优化器根本没动"。这是 Goal 5 的机制测量反馈给
Goal 4 实验设计的直接结果。
