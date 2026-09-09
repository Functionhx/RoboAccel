# 09 · 实验解读指南（experiment interpretation guide）

**这份文档回答的问题是：哪些数字能拿去下结论，哪些不能。**

---

## 一、会骗人的指标

### 1.1 `peak_roll_rad` 是最大值，不能用来算 recovery

`peak_roll_rad` 的定义是**整个 condition 内最差的那一条 episode 的峰值**，
即 768 条 episode 上的 max，被单个离群点主导。

多种子实验直接暴露了它：

| 指标 | seed 1 | seed 2 | seed 3 | 跨种子跨度 |
|---|---|---|---|---|
| yaw RMSE recovery | 89.5% | 89.6% | 93.0% | **0.1 点** |
| `peak_roll_rad` recovery (nominal) | 19.3% | 39.1% | 60.8% | **41 点** |
| `peak_roll_rad` recovery (push) | 32.8% | 108.2% | — | **75 点** |

**在其他每个指标上一致到 0.1 点的三个种子，在这个指标上差 75 点。**

用稳定的同伴指标 `mean_episode_peak_roll_rad`（逐 episode 峰值的均值）：

| scenario | max 统计量说 | mean 统计量说 |
|---|---|---|
| nominal | 19.3% | **97.0%** |
| delay:2 | 94.4% | **100.5%** |
| mass:2 | **−20.0%** | **96.2%** |

> **一个被撤回的结论**：早期版本报告过"QAT 在 mass:+2 上比 PTQ 差 20%"。
> 那是 768 条 episode 里的**一条**倒霉 episode，不是方法的性质。
> 用 mean 统计量，姿态恢复 96–108%，和跟踪一样好。

**规则**：max 统计量可以作为**最坏情况**报告，**永远不要**用它算 recovery 分数。

### 1.2 `robust_v1` 上的 "zero falls" 是失效判据的漏洞

在 `robust_v1` 上所有 13 个 scenario 都是 0.000 fall rate、20 s 存活。
**那不是鲁棒性**——旧的 fall 判据不把 chassis 接触算作跌倒。

Codex 的 `physical_probe.json`：*"zero leg actions settle onto the chassis;
most cases show 91–100% chassis contact while the old failure logic records
no falls."* 机器人是**趴在底盘上**，不是靠轮子平衡。

**独立佐证**（来自量化侧，非 Codex）：在 `robust_v1` 的 122,880 个样本上，
golden vector 的 `large_attitude` regime（要求倾角 > 0.35 rad ≈ 20°）
**一个都找不到**，倾角通道峰值 0.346 / 0.233。趴着的机器人不会倾斜。
换到 `locomotion_v2` 后，同一个分类器找到 **9,807** 个。**[已验证]**

**规则**：任何"零失败"的结果，先去看**失败是怎么定义的**。

### 1.3 nominal 段的小差异是混沌发散，不是量化代价

FP32 vs W16A16 在 nominal 上差 ~9–11%，但在所有扰动 scenario 上一致到 <1%。
原因：nominal 没有外部扰动把轨迹拉回同一条，2000 步下微小算术差异会
混沌发散。**扰动段的一致性才是无损的证据。**

---

## 二、必须成对读的数字

### 2.1 任何 QAT 数字都必须配 §7 对照组

`robust_v1` 上的教训：

| arm | iters | nominal fwd RMSE |
|---|---|---|
| SOLID_FP32 | 5000 | 0.2898 |
| **QAT W8A8** | 5500 | **0.1334** |
| **SOLID_FP32_PLUS（对照）** | 5500 | **0.0657** |

只看前两行会得出"QAT 打败了 FP32"。加上对照组就清楚了：
**那个增益是多训练的 500 步，不是量化感知训练。**

**规则**：QAT 与 PTQ 必须在**相同训练步数**上比较。
`robust_v1` 早期的对比里 PTQ 用的是 5000 步 checkpoint、QAT 是 5500 步，
那个 gap 混入了训练长度，是无效的。

### 2.2 `success` 必须配 `support_fraction`

`success` 是通过/不通过，`support_fraction` 说明**是不是还站在轮子上**。
W8A8 的 `support_fraction` 降到 0.915–0.950 —— 机器人正在失去轮子支撑，
而不只是跟踪变差。W4A8 降到 **0.239**，即大部分时间趴在地上。

---

## 三、当前结论的强度分级

### [已验证] 可以直接引用

* **cliff 恰好在 8-bit 激活**。FP32 / W16A16 / W8A16 在全部 7 个 segment 上
  `success` 与 `support_fraction` 都是 1.000；W8A8 掉到 0.000–0.164。
  **INT8 权重是免费的，INT8 激活不是。**
* **五个实现 bit-identical**：torch fake-quant、numpy 整数参考、
  FPGA descriptor program、H7 scalar C、H7 SMLALD，后两个在**实体芯片**上验证。
* **RoboAccel 17.99 µs vs H7 259.09 µs = 14.4×**，两者都是实测。
* **更好的 FP32 策略更难量化**：同一套 W8A8，5000 步 checkpoint 退化 1.7×，
  5500 步退化 **8.4×**，且绝对值更差。在 `locomotion_v2` 上独立复现（12×）。
  英文：*Improving the float baseline makes QAT more necessary, not less.*

### [已验证] 但只在特定条件下成立

* **QAT 在 W8A8 上恢复轮子支撑 95–100%**（三种子，跨度 <0.02）。
* **QAT 恢复直线运动 39–84%**，但**种子方差很大**（`reverse` 跨 0.12–0.88）。
  单种子数字会在任一方向误导。
* **QAT 从不恢复转向**：三个种子、两个方向全部 0.000 [0.00, 0.00]，
  而 PTQ 保留 0.47–0.54。**QAT 在转向上比不做还差。**

### [假设] 尚未检验

* QAT 用 yaw 精度换 forward 精度，因为奖励里
  `tracking_lin_vel = 2.0` 是 `tracking_ang_vel = 1.0` 的两倍。
  判决性实验正在跑（`--reward-ang-vel 2.0 / 4.0`）。

### [已撤回]

* ~~"QAT 恢复 86–101%"~~ — 只在机器人趴着的任务上成立。
* ~~"latent 是瓶颈"~~ — `enc2` 占 13.2%，见 [06](06_failed_hypotheses.md)。
* ~~"舍入偏置导致转向失败"~~ — 修正舍入后转向仍为 0.000。
* ~~"requantizer 偏置是系统性负偏"~~ — 均值会抵消，**差速**才是有意义的量。

---

## 四、一个贯穿全局的警告

**`robust_v1` 上的所有绝对数字都不是关于 locomotion 的陈述。**
相对结论（量化敏感性）仍然成立，因为所有 arm 用同一个任务、同一个评估器、
同一批 seed 和 horizon 打分——**那是关于算术的论断，不是关于机器人的。**

`QAT_RESULTS.md` 顶部有一个 validity notice 记录这一点。
