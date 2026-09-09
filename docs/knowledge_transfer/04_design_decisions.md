# 04 · 设计决策记录（design decision log）

每条记录：**决策 / 备选方案 / 为什么这样选 / 代价 / 什么情况下应该重新考虑**。

---

## D1 · 复现硬件的 bug，而不是修正它

**决策** `hwq/fixed_ref.py:round_shift` 与 `hwq/torch_hw.py:hw_round_shift`
**故意复现** `rl_round_shift_sat16.sv` 的非理想舍入：先加半个 LSB
（away from zero）再算术右移，导致负数 99.6% 概率低一个 LSB。

**备选** 用数学上正确的 round-half-away-from-zero。

**为什么** 整个项目的价值建立在**软件预测硬件**之上。
一个"更正确"的软件参考会让 golden vector 在板子上失配，
于是每次不一致都要先排查"是硬件还是我的参考"——验证链就废了。
**匹配硬件优先于正确（matching the hardware beats being correct）。**

**代价** 引入了一个真实的偏置。在 16-bit 下可忽略（<1 LSB），
在 8-bit 下不可忽略（两轮 ~4 rad/s 差速）。

**何时重新考虑** 已经重新考虑过。`IDEAL_ROUNDING` 开关（默认关）允许
把两者都测出来。结论：修正舍入让 `reverse` 0.570→0.992、yaw RMSE 降约 20%，
**值得在下一版 RTL 里修**，但它不是转向失败的原因。见
[06](06_failed_hypotheses.md) 假设 3。

---

## D2 · 借用 Codex 的评估器，而不是自建

**决策** `scripts/eval_quant_robustness.py` 和
`scripts/eval_quant_locomotion.py` 都不实现评估逻辑，而是
**重绑定（rebind）Codex 评估器里的一个模块全局变量**：

```python
ev.ActorCriticRobust = injected
ev.ActorCriticSequence = injected
```

**为什么可行** 评估器在**调用时**才从模块全局查表拿 policy class，
且随后只调用 `model.act_inference(obs, history) -> (actions, latent)`，
正好是 `QuantActorCriticSequence` 的签名。Codex 的文件**从不被修改**。

**为什么这样选** goal.md §6 明确禁止另起一套 benchmark。
更实际的理由：如果 FP32 和量化臂用不同的代码打分，
两者之差里就混进了"评估器差异"，整个对照实验失去意义。

**代价** 依赖上游文件的内部结构。Codex 在开发期间给
`evaluate_locomotion.py` 加了 `--replay_output`，
硬编码的参数表立刻失效（`AttributeError` 出现在 `evaluate()` 深处）。
**修复**：改为用 `ast` 从上游源码里**解析** `get_args([...])`，
所以上游加参数会被自动接住。

> 注意 `ast.unparse` 是 Python 3.9+，而 Isaac 环境是 3.8。
> 第一版解析器因此静默返回 `None`。现在手工遍历 `Dict` 节点。

---

## D3 · 用 SHA256 manifest 固定基线，而不是 git commit

**决策** `artifacts/v2/frozen/infra_manifest_final.sha256` 记录 145 个文件的哈希。

**为什么** Codex 的整套改动**全部是未提交的工作区状态**
（uncommitted working-tree state）。`git rev-parse HEAD` 永远返回
`8204e85`，无法区分 25 小时里的任何一个中间状态。

**回报** 这个 manifest 三次抓到了实质变化：
1. 08:09 新增 `locomotion_v1` 注册（纯增量，无影响）
2. 09:40 `robust_robot.py` 改动，**恰好落在 seed-3 的对照臂和量化臂之间**
3. `shared/training/` 重构一次改了 15 个文件

每次都通过**重跑一个已知结果**来判定影响，两次都得到 **0.0000% bit-identical**。
没有 manifest 就不会知道要去检查。

---

## D4 · 逐层激活 frac，而不是固定 Q8.8

**决策** `QuantConfig.act_fracs` 允许每层有自己的 fractional bits，
并且**允许为负**（`quantize` 用 `2.0 ** frac` 而不是 `1 << frac`）。

**为什么必须** 不同层的动态范围差 7 倍。固定 Q8.8 在 INT8 下量程是 ±0.5，
而实际激活能到 22——**未标定的 INT8 run 测的只是饱和，不是量化**。
在 `robust_v1` 上 `enc0` 峰值 130.7，需要 frac = −1（LSB = 2.0），
`1 << -1` 会直接抛异常。

**代价** 引入 CONCAT 约束：PL 的 CONCAT 不能 requantize，所以
`frac(obs)` 必须等于 `frac(latent)`。`fixed_ref.py` 里有断言。
Codex 的 residual actor 引入了**第二个**同类约束，已加
`check_residual_concat`。

---

## D5 · §7 对照组先跑，QAT 后跑

**决策** `run_qat_and_control.sh` 先训 `SOLID_FP32_PLUS`，再训 QAT。

**为什么** 如果先看到 QAT 的漂亮数字，再去跑对照，会有强烈的动机
把对照的异常解释掉。先跑对照就没有这个诱惑。

**它救了什么** 在 `robust_v1` 上，QAT 看起来**打败了 FP32**
（0.1334 vs 0.2898）。对照组——同样 500 步、不量化——达到 **0.0657**，
在每个指标上都比 QAT 好。**那个"增益"完全是多训练的 500 步。**
没有对照就会发表一个错误结论。

---

## D6 · 部署基线用 Codex 的 checkpoint，不自己重训

**决策** `SOLID_FP32_V2` = Codex 的 `locomotion_v2` seed-1 checkpoint。

**为什么** goal.md §0/§1 要求"识别并冻结 Codex 的基线"，不是"复现"。
Codex 自己的 `nominal_three_seed_summary.json` 已经三个种子验证过。
重训只会引入一个无法与他们的证据对齐的新策略。

**代价** 依赖上游的选择。若 Codex 之后 promote 了别的 checkpoint，
需要重跑——但流水线已按 `TASK` / `CKPT` 环境变量参数化，
重跑是**GPU 小时**而不是返工。见 `RERUN_PLAN.md`。

---

## D7 · residual actor 用 CONCAT 融合，而不是新增 ADD 算子

**决策** Codex 最终 promote 的控制器是 `action = base(x) + residual(x)`，
两条并行 actor 分支。PL 有 GEMM / ELU / NORM / CONCAT，**没有 element-wise ADD**。

**方案** 两条分支都以一个输出 6 维的 GEMM 结尾（输入分别是 32 和 64），
它们的和恰好等于**在拼接后的倒数第二层激活上做一次 GEMM**：

```
W_b @ a + c_b  +  W_r @ b + c_r  ==  [W_b | W_r] @ [a ; b] + (c_b + c_r)
```

在真实权重上 512 个输入验证：最大差 **4.8e-07**（float32 舍入）。**[已验证]**

**为什么这样选** goal.md §5 明确说"优先扩展现有 datapath 表示，
不要另建不兼容的推理框架"。缺失的 ADD 变成硬件已有的 CONCAT。

**代价** 资源从 620 涨到 836 weight words、38,400 → 52,352 MAC，
程序长度 13 → 18（上限 32）。全部装得下。
并引入 D4 的第二个 grid 约束。

---

## D8 · 两个并发训练，不是三个

**决策** GPU 调度维持 2 个 4096-env 作业常驻。

**为什么** 实测：3 个并发合计 ~28 iters/min，单个独占 ~69 iters/min。
**卡在颠簸而不是并行。** 降到 2 个后合计 68/min，2.4 倍吞吐。
第 4 个作业直接触发 `CUDA error: an illegal memory access`。

**教训** "用满 GPU" 与 "同时跑到崩" 是相反的两件事。
