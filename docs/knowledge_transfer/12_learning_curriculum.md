# 12 · 学习路径（learning curriculum）

目标是**技术所有权移交**，不是读完文档。每一阶段有**可验证的检查点**——
能做到那件事，才算通过。

---

## 阶段 1 · 确认环境可用（半天）

**做**
```bash
source qat/scripts/env_solid.sh
$ISAAC_PY qat/scripts/verify_against_rtl.py
bash qat/scripts/hosttest/build_and_run.sh qat/artifacts/v2/deploy/h7
```

**检查点**
- [ ] 两个命令都 PASS
- [ ] 能说出这两个命令**分别在验证什么**（源码级 vs 编译执行级）
- [ ] 知道 `$ISAAC_PY` 和 `$MJ_PY` 为什么不能互换

---

## 阶段 2 · 理解定点算术（1–2 天）

**读** [03](03_math_to_code.md) → `hwq/fixed_ref.py` 全文。

**做**：手算一次。取 `f_w = 13`、`f_in = 3`、`f_out = 2`，
构造一个小 GEMM，用纸笔算出 `round_shift` 的结果，再用 numpy 验证。

**检查点**
- [ ] 能解释为什么 `quantize` 用 `2.0 ** frac` 而不是 `1 << frac`
- [ ] 能解释 `round_shift` 对负数的行为，以及**为什么故意不修**
- [ ] 能说出 CONCAT 约束是什么、为什么存在
- [ ] 能算出一个给定 MLP 的 weight words 和 MAC 数

---

## 阶段 3 · 跑通一次完整验证链（1 天）

**做**：换一个 checkpoint，跑 [08](08_verification_chain.md) 的完整回归。

**检查点**
- [ ] 9 个张量在 W16A16 和 W8A8 下都 100% exact
- [ ] FPGA exporter 推出的 weight fracs 与 `choose_weight_frac` 一致
- [ ] 板子上 `verify: scalar=PASS simd=PASS`
- [ ] **能解释为什么 numpy 整数参考是仲裁者**

---

## 阶段 4 · 复现一个实验并解读（2–3 天）

**做**：跑 PTQ ladder（`run_v2_ladder.sh`），用
`summarize_locomotion.py` 出表。

**检查点**
- [ ] 能指出 cliff 在哪里，并解释**为什么是激活而不是权重**
- [ ] 看到一个 `success = 0.000` 时，知道要去看 `support_fraction`
- [ ] 知道**不能**用 `peak_roll_rad` 算 recovery，以及为什么
- [ ] 能说出为什么 nominal 段的 9% 差异不是量化代价

---

## 阶段 5 · 理解已被推翻的假设（1 天，**最重要**）

**读** [06](06_failed_hypotheses.md) 全文。

**检查点**
- [ ] 能复述三个假设**各自的支持证据**（它们都不是瞎猜）
- [ ] 能说出**每个假设是被什么实验推翻的**
- [ ] 能解释为什么"更多观察"推翻不了错误机理
- [ ] 拿到一个新现象时，能立刻提出一个**能证伪自己解释**的实验

> 这一阶段没有代码产出，但它是最难替代的部分。
> 前四个阶段的知识可以重新学，这一阶段的不能。

---

## 阶段 6 · 独立诊断一个故障（2 天）

**做**：故意制造故障，练习定位。建议三个：

1. 把 `act_fracs` 里某一层的 frac 改错 → 观察 saturation 和控制指标
2. 把 `install_quant_policy` 改回返回函数 → 看它怎么崩
3. 用 `evaluate_robustness.py` 去评估一个 `locomotion_v2` checkpoint

**检查点**
- [ ] 三个故障的报错都能在 30 分钟内定位到根因
- [ ] 能说出**每个故障如果不报错会怎样**（这才是危险的情况）

---

## 阶段 7 · 判断一个新策略能否上板（1 天）

**做**：拿一个任意 MLP 策略，回答它能否映射到 RoboAccel。

**检查点**
- [ ] 能算出 program length、weight words、vector words、MAC
- [ ] 能识别出**当前 exporter 不生成的算子**（Conv1D、attention）。
      注意 element-wise ADD **不在此列**：ISA v3 的 `OP_AFFINE`
      在 a=1、shift=0 时就是 ADD，只是 exporter 目前不生成它。
      **区分"硬件不支持"和"工具链没接"是这一课的重点。**
- [ ] 对于 residual actor，能说出融合方案和它引入的新约束
- [ ] 知道 A4/A5 的 causal Conv1D **为什么不能上板**

---

## 阶段 8 · 继续研究（持续）

当前唯一未决的科学问题（见 [06](06_failed_hypotheses.md) 假设 3 的后继）：

> QAT 是否在用 yaw 精度换 forward 精度？

判决实验：`--reward-ang-vel 2.0 / 4.0` 重训 QAT。

**下一步优先级**
1. 在 RTL 里修 requantizer 的负数舍入路径（独立成立：`reverse` 0.57→0.99）
2. 导出 residual actor（融合已验证，grid 断言已就位）
3. §33 QAT-from-scratch（可选）

**维护规则**（goal2.md §18）：新 bug → 更新 05；假设被推翻 → 更新 06；
架构决策 → 更新 04；新实验 → 更新 09；工作流变化 → 更新 11。

---

## 通过标准（definition of done）

能在**没有 Claude** 的情况下做到：

- [ ] 讲清端到端架构
- [ ] 定位任一子系统在代码里的位置
- [ ] 解释定点与量化流水线的数学
- [ ] 复现完整验证链
- [ ] 独立运行并**正确解读**主要实验
- [ ] 说出重要的失败路径及原因
- [ ] 区分**已验证事实**与**假设**
- [ ] 独立诊断常见故障
- [ ] 判断一个新策略能否映射到 RoboAccel
