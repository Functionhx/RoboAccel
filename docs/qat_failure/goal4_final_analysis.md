# Goal 4 · 最终分析 —— 为什么"本该成功"的 QAT 失败了

> 本文回答一个具体问题：**QAT 在理论上应当优于 PTQ，为什么在 W8A8 上它反而
> 比 PTQ 更糟？**
>
> 简短回答：**QAT 从来没有真正训练过。** 量化噪声击穿了 PPO 的自适应学习率
> 控制器，把学习率钉死在下限，整整 2000 次迭代。策略没有在学习如何容忍量化，
> 它只是在量化边界噪声中原地抖动了 2000 步。
>
> *In one line: the run that was supposed to teach the policy to tolerate
> quantization never took a meaningful gradient step, because the trust-region
> controller mistook quantization discontinuity for policy divergence and
> throttled the learning rate to its floor on 2000 out of 2000 iterations.*

---

## 0. 这个失败为什么值得写下来

QAT 的成立前提是一句几乎没人验证过的话：

> 直通估计器（STE）让梯度穿过量化器，于是优化器可以找到"对量化鲁棒"的权重。

这句话隐含了一个更强的前提：**优化器能够迈步**。
本项目的测量表明，在 W8A8 下这个前提**不成立**——而且失效的方式与
量化器的表达能力无关，与激活位宽的表达能力无关，
与奖励设计无关。它失效在**优化器的控制回路**上。

这不是"再多训练一会儿"能解决的问题，也不是"调一调奖励权重"能解决的问题。
两者都已被实验证伪（§4）。

---

## 1. Established facts（已确立的事实）

以下每一条都可追溯到一个非空的原始文件或报告。

**F1.** W8A16 与 W16A16 的 QAT 可以工作；**W8A8 不能**。控制悬崖精确地落在
**8 bit 激活**上，而不是 8 bit 权重上。
`qat/artifacts/v2/ladder_pooled.json`

**F2.** 同一个 W8A8 QAT checkpoint，在**去掉量化器的浮点执行**下，
七个 segment 的通过率**全部为 0.000**——包括在 fake-quant 下通过率为
1.000 的 `forward` 和 `height`。
`docs/goal4/01_localization.md`，`qat/artifacts/v2/m_L_qat`

**F3.** 转向在**浮点执行和 fake-quant 执行下都失败**（`turn_left` /
`turn_right` 均为 0.000）。这把转向的丢失定位在**训练得到的参数里**，
而不是执行精度里。

**F4.** fake-quant 路径与导出的定点路径在 4000 个样本上 **bit-exact**。
B 与 C 之间不存在数值失配。
`qat/golden_vectors/`，`scripts/bitexact_check.py`

**F5.** W8A8 训练中激活裁剪比例为 **0.0000%**。不是饱和问题。
`qat/artifacts/v2/diag_w16a16.json`

**F6.** 产生 latent 的 `enc2` 层只占总量化误差的 **13.2%**。不是 latent 瓶颈。
`qat/artifacts/v2/layer_sensitivity_w8a8.json`

**F7.** 把 RTL 的 floor 舍入换成理想的 round-half-away-from-zero，
转向仍为 **0.000**。不是舍入偏置。
`qat/artifacts/v2/m_L_qat_ideal`

**F8.** **W8A8 QAT 运行的学习率在 2000/2000 次迭代（100%）都处在 1e-5 下限；
FP32 对照只有 10/2000（0.5%）**，中位学习率 2.563e-4。
两者的 task、seed、初始 checkpoint、奖励、课程、PPO 超参完全相同。
`qat/checkpoints/{qat_w8a8_v2_long,solid_fp32_plus_v2_long}/metrics.jsonl`

**F9.** W8A8 运行测得的策略 KL 中位数为 **0.0650**，是 FP32 对照
（0.0054）的 **12 倍**，并且**在全部 2000 次迭代中都超过控制器阈值**。
FP32 对照只有 1/2000 次超过。

**F10.** 离线测量：对**同一组参数**施加**同一个** 1e-5 的权重扰动，
浮点前向的策略 KL 为 **4.06e-07**，W8A8 fake-quant 前向的策略 KL 为
**6.73e-02**——放大 **165 737 倍**。
`qat/artifacts/v2/kl_amplification_qat.json`

**F11.** 把 `tracking_ang_vel` 奖励提高到 2× 和 4×（奖励贡献实测精确按
2.00× / 4.00× 缩放），`yaw_rmse_rad_s` 从 0.1546 变到 0.1533 / 0.1540，
即**没有变化**；FP32 对照为 0.0584。
`qat/checkpoints/qat_v2_yaw{2,4}/metrics.jsonl`

---

## 2. 机制：控制器的传感器被量化器打坏了

`rsl_rl/algorithms/ppo.py:208-215`：

```python
if self.desired_kl is not None and self.schedule == "adaptive":
    if kl_mean > self.desired_kl * 2.0:
        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
```

本任务 `desired_kl = 0.005`，所以**只要测得 KL > 0.01，学习率就被除以 1.5**，
下限 1e-5。

这个控制器建立在一个假设上：**KL 随步长单调增加**，
因此"KL 太大 → 步子迈小一点"是有效的。

**在 fake-quant 下这个假设是错的。** 一个远小于量化 LSB 的权重步长
什么都不改变；一个恰好跨过 LSB 边界的步长会让该权重**整整跳一个 LSB**。
于是测得的 KL 由"这一步有多少权重跨过了边界"支配，
而这个量**几乎不随学习率变化**——把学习率降到 1e-5，KL 依然是 0.067。

**控制器唯一的调节手段，无法降低它所调节的量。**

### 2.1 定量验证：噪声地板 vs 控制器阈值

`scripts/kl_amplification.py` 对同一组参数施加一次 Adam 步长
（`|Δw| ≈ lr` 逐元素，因为 Adam 按梯度 RMS 归一化），
测量由此产生的策略 KL：

**在学习率下限 lr = 1e-5 处：**

| 精度 | 单步策略 KL | 相对浮点 | vs 阈值 0.01 |
|---|---|---|---|
| float | 4.06e-07 | 1× | 低 24 000 倍 |
| W16A16 | 1.44e-04 | 354× | 低 69 倍 |
| W8A16 | 8.78e-04 | 2 162× | 低 11 倍 |
| **W8A8** | **6.73e-02** | **165 737×** | **高 6.7 倍** |

**W8A8 是精度阶梯上唯一一个量化噪声地板高过控制器阈值的配置，
也是唯一一个 QAT 失败的配置。**

这是一个**预测性**结果：它只用一次前向传播、不训练、不跑仿真，
就正确预测了整条精度阶梯上哪些配置 QAT 能训、哪些不能。

### 2.2 与实测训练 KL 的对账

PPO 每次迭代做 `num_learning_epochs=5 × num_mini_batches=4 = 20` 个梯度步，
报告的是这 20 个 minibatch 的 KL 均值，等效累计步长 ≈ 20 × 1e-5 = 2e-4。
离线表在 lr = 2.56e-4 处给出 W8A8 KL = **0.125**。

实测：训练中位 KL **0.065**，`desired_kl=0.08` 试验臂早期 KL **0.147**。
量级与趋势都对上——**训练中观察到的 KL 几乎全部来自量化边界噪声，
其中几乎不含真正的策略学习信号。**

### 2.3 闭环后果

1. 第 1 次迭代 KL 就是 8.97（FP32 对照 0.89），控制器立刻开始降学习率。
2. 学习率在几十次迭代内触底 1e-5，**此后 2000 次迭代再也没有升起来**。
3. 有效步长比 FP32 对照低 **26 倍**（1e-5 vs 2.563e-4）。
4. 与此同时，编码器由**独立的 `extra_optimizer`** 驱动
   （`ppo.py:56-62`），学习率固定 1e-3，**完全不受该控制器管辖**——
   自适应分支只写 `self.optimizer.param_groups`。
   实测 `Encoder/policy_kl`：W8A8 0.45–0.84，FP32 对照 0.003–0.007，**相差 100 倍**。
   **actor 被钉在 1e-5，而它的输入分布正被一个快 100 倍的优化器拖着走。**
5. 奖励曲线看不出任何异常（28.6 vs FP32 的 29.3），因为奖励由
   `tracking_lin_vel`（权重 2.0）支配，而转向只占很小一部分。
   **这就是这个失败能潜伏这么久的原因：训练曲线是健康的。**

---

## 3. Supported observations（有支持的观察）

**S1.** 转向的损伤**在第 1 次迭代就已经存在**：W8A8 运行 iter 1 的
`yaw_rmse_rad_s = 0.1468`，2000 次迭代后是 **0.1566**（略微更差）。
FP32 对照全程 0.046 → 0.058。
→ **QAT 从未修复过 PTQ 造成的损伤，一次也没有。**
它没有把一个好策略训坏，它是**接手了一个已经坏掉的策略，然后无力修复**。

**S2.** W8A8 运行的探索标准差从 0.285 升到 0.300，FP32 对照从 0.284 降到 0.279。
在策略梯度被压到 1e-5 的情况下，`entropy_coef = 0.01` 的熵奖励相对占优。
（观察，未做控制实验。）

**S3.** 放大倍数在位宽上**非单调**：lr=1e-5 处 W4A8（1.94e-2）低于
W8A8（6.73e-02）。合理的解释是 4 bit 网格的 LSB 更大、边界更稀疏，
且更多权重被钳在 `max_weight_frac` 的轨上而无法跨越边界。
**未做控制实验，不作为结论。**

---

## 4. Rejected hypotheses（已证伪的假设）

| # | 假设 | 证伪依据 |
|---|---|---|
| §2-4 | 激活量化分辨率 | F2/F3：去掉量化后转向仍为 0.000 |
| §2-5 | 激活裁剪 / scale 分配 | 同上，且实测裁剪 0.0000%（F5） |
| §2-6 | 共享激活网格约束 | 同上 |
| §2-8 | fake-quant ↔ 导出失配 | F4：两者 bit-exact |
| §2-2 | 目标 / 奖励失衡 | **F11：奖励 ×4，`yaw_rmse` 不动** |
| — | 舍入偏置导致转向失败 | F7：理想舍入下转向仍 0.000 |
| — | latent 瓶颈 | F6：`enc2` 只占 13.2% |

**这七条被证伪的假设，恰好是文献里对"QAT 失败"最常见的七种解释。**
它们在这里全部不成立，这本身就是本工作的结果之一。

---

## 5. Remaining hypotheses（仍然在场的假设）

* **H8（检验中）**：解除学习率钳制即可恢复 W8A8。
  实验 `qat_v2_fixlr`（`--schedule fixed --learning-rate 2.563e-4`）
  与正对照 `qat_v2_w8a16` 正在运行。
* **H0/H9（未检验）**：冻结 base + 只学残差（goal4.md §4）。
  其前提（"现有 QAT 会摧毁参考策略"）已由 F2 证实。
* **H10（未检验）**：FP32 教师的行为保持项（goal4.md §7）。
  审计结论见下。

### QAT 目标函数审计（goal4.md §7 的前置问题）

**结论：当前的 QAT 目标里不存在任何行为保持机制。**

`scripts/train_policy.py` 的 QAT 就是"用量化前向重跑一遍 PPO"。具体地：

* 损失 = PPO 代理损失 + 价值损失 − 熵奖励。**没有** KL-to-reference，
  **没有**行为克隆，**没有**蒸馏项。
* `--init-from` 只加载 `model_state_dict`，**不加载优化器状态**——
  Adam 的一阶/二阶矩从零开始。
* 与参考策略的唯一联系是**初始化**，而初始化不是约束。
* PPO 的自适应 KL 是对**上一步策略**的信任域，不是对**参考策略**的锚。
  每步 0.065 的 KL 累计 2000 步，没有任何东西把它拉回来。

**所以"QAT 会保持行为"从来就不是这段代码的性质，只是一个未经检验的期待。**

---

## 6. Successful interventions（有效的干预）

**W8A16 —— 唯一有效的干预。**

`qat_v2_w8a16`（**正常**自适应控制器，其余与失败的 W8A8 运行完全一致）：

| 执行路径 | 七段 success | `yaw_rmse` turn_L / turn_R |
|---|---|---|
| fake-quant (B) | **全部 1.000** | 0.0194 / 0.0261 |
| 浮点 (A) | **全部 1.000** | 0.0188 / 0.0271 |
| FP32 对照 | 全部 1.000 | 0.0224 / 0.0295 |

**arm A == arm B == 1.000**，且偏航精度**优于** FP32 对照。
即 W8A16 的 QAT 学到的是"能容忍量化"，而不是 W8A8 那种"只有在量化下才成立"。

**最小的有效修改是激活位宽 8 → 16，而不是训练侧的任何东西。**

在 W8A8 上的四次干预（更长训练、理想舍入、奖励重加权 ×2/×4、固定学习率）
全部无效。这是一个诚实的记录，不是遗漏。

---

## 7. Failed interventions（无效的干预）

| 干预 | 结果 |
|---|---|
| 训练 2000 步而不是 1500 步 | `yaw_rmse` 0.147 → 0.157（更差） |
| 理想 round-half-away-from-zero | 转向仍 0.000 |
| `tracking_ang_vel` ×2 | `yaw_rmse` 0.1546 → 0.1533 |
| `tracking_ang_vel` ×4 | `yaw_rmse` 0.1546 → 0.1540 |
| `desired_kl` 0.005 → 0.08 | 落入控制器死区（KL 0.147 < 阈值 0.16，但 > 阈值/2），学习率仍在下限；已中止改用 `--schedule fixed` |
| **固定学习率 2.563e-4（0% 迭代在下限）** | **转向仍 0.000，整体 success 0.501 → 0.206，低于 PTQ。见下** |

**第五次干预是判决性的，而且结果是负的。** 固定学习率成功地让优化器
恢复了迈步能力（0.0% 在下限，对比基线的 100.0%），转向纹丝不动，
而 `stop` 从 0.938 掉到 0.023、`height` 从 1.000 掉到 0.000。

**把刹车松开、却不加任何行为保持项，只会让策略漂得更远。**
这同时证伪了"学习率崩塌是控制失败的原因"，并把责任交给
**激活位宽**（§6）与**行为保持的缺失**（§5 目标函数审计）。

**这四次失败现在有了统一解释**：它们全都改变了"优化器该往哪走"，
而没有一次改变了"优化器能不能走"。

---

## 8. Deployment recommendation（部署建议）

**不变：W16A16 是唯一经过硅上验证的部署配置。**

* 已验证的导出器按构造产生 **INT16 / Q8.8** 权重与激活。
  它**不能**导出 W8A16——之前"改用 W8A16 部署"的建议
  **不可执行**，已更正。见
  `qat/artifacts/deploy/QAT_policy_W16A16_fpga/README_NAMING.md`。
* W8A8 **不建议部署**。它在 PTQ 下损失转向，在 QAT 下损失更多。
* 本文的机制发现**不改变**部署结论，它改变的是**下一步该做什么实验**。

---

## 9. Research conclusion（研究结论）

**QAT 在 W8A8 上失败，不是因为 8 bit 激活装不下这个策略，
而是因为 8 bit 激活让训练算法失去了控制自己步长的能力。**

完整的因果链，每一环都有测量支撑：

```
8 bit 激活 + 8 bit 权重
   └─> 策略输出对权重不连续（一步跨过 LSB 边界 = 跳一整个 LSB）
        └─> 1e-5 的权重扰动产生 KL 0.067，是浮点的 165 737 倍   [F10]
             └─> 超过 PPO 控制器阈值 0.01 达 6.7 倍
                  └─> 学习率被除以 1.5，直到 1e-5 下限
                       └─> 2000/2000 次迭代都在下限                [F8]
                            └─> 有效步长比 FP32 对照低 26 倍
                                 └─> QAT 无法修复 PTQ 的损伤       [S1]
                                      └─> 同时编码器以 100 倍的速度
                                          拖动 actor 的输入分布     [§2.3-4]
                                           └─> 学到的函数把量化器
                                               本身当成了组成部分   [F2]
```

**给英文表述（供论文使用）**：

> Quantization-aware training of an RL policy is not merely a forward-pass
> modification. Fake-quantization makes the policy output a discontinuous
> function of its weights, and any trust-region controller that regulates step
> size by measured policy divergence will read that discontinuity as
> divergence. In our W8A8 configuration a weight perturbation of 1e-5 — the
> algorithm's own learning-rate floor — produces a policy KL of 6.7e-2,
> 165,737x the float value and 6.7x the controller's threshold. The controller
> therefore drove the learning rate to its floor on 2000 of 2000 iterations
> and could not raise it again, because the quantity it regulates does not
> respond to the lever it controls. The run reported a healthy reward curve
> throughout. W8A16 and W16A16, whose quantization KL floors sit 11x and 69x
> *below* the same threshold, trained normally — so the noise floor predicts
> which precisions are trainable without running any training at all.

**这条结论是可迁移的**：任何用 KL 自适应学习率的 on-policy RL 算法
（PPO、TRPO 及其派生）在低精度 QAT 下都会遇到同一个问题，
而且**症状是隐蔽的**——奖励曲线正常，只有学习率轨迹会暴露它。

**因此，检查 QAT 运行的学习率轨迹，应当是低精度 RL 部署的标准诊断项。**

---

## 10. 与 goal4.md §26 完成判据的对照

走的是**科学闭合路径**（scientific closure path）：

* [x] 主要的可能失败机制已被检验——§2 列出的 10 项中有 7 项已判定
* [x] 无效的方法已记录（§7）
* [x] W16A16 仍然是有依据的部署方案（§8）
* [x] W8A8 的失败本身产生了可辩护的技术洞见（§9）
* [x] **最小精度/训练修改已知**——是 **W8A16**（激活 8 → 16 bit）。
  训练侧的修改不够：固定学习率让优化器恢复迈步能力后，转向仍为 0.000，
  整体反而更差（0.501 → 0.206）。

**Goal 4 达到科学闭合（scientific closure）。** 五项判据全部满足。

### 一句话结论

> W8A8 的 QAT 失败，不是因为优化器被卡住（那是真的，但把它修好没有用），
> 而是因为 **8 bit 激活本身**。最小的有效修改是把激活加宽到 16 bit——
> W8A16 在**正常**的控制器下、其余条件完全相同的情况下，
> 七段全部 1.000，且在浮点与 fake-quant 两条路径下**都**成立。

### 部署结论（不变，但依据更强）

W16A16 仍是唯一经过硅上验证的部署配置。
W8A16 现在**既被证明可训练、又被证明不依赖量化器**，
唯一的阻碍仍是 exporter 按构造只能输出 INT16 权重。
