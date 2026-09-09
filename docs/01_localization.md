# Goal 4 · §3 定位实验（localization）— 转向能力在哪里消失

**结论先行**：转向不是被"执行精度"杀死的，是在**训练中被参数本身丢掉的**；
而 QAT 学到的参数**依赖量化器才能工作**。

---

## 实验设计

同一个 QAT checkpoint（`checkpoints/qat_w8a8_v2_long/model_2000.pt`，
2000 步，从 `SOLID_FP32_V2` 出发），三条执行路径，
**相同 evaluator、相同 seed=11、相同初始条件、相同 128 envs**：

| arm | 执行方式 | 命令 |
|---|---|---|
| **A** | 浮点执行（无 fake-quant） | `eval_quant_locomotion.py --fp32` |
| **B** | fake-quant（QAT 训练时用的数值路径） | `--weight-bits 8 --act-bits 8 --act-fracs ...` |
| **C** | 导出定点（bit-exact 整数参考） | 同上 `+ --integer` |

## 结果 **[已验证]**

`success`（逐段通过率）：

| segment | FP32 对照 | PTQ | **A 浮点** | **B fake-quant** |
|---|---|---|---|---|
| forward | 1.000 | 0.000 | **0.000** | **1.000** |
| reverse | 1.000 | 0.000 | **0.000** | 0.570 |
| turn_left | 1.000 | 0.758 | 0.000 | 0.000 |
| turn_right | 1.000 | 0.523 | 0.000 | 0.000 |
| stop | 1.000 | 0.984 | **0.000** | 0.938 |
| height | 1.000 | 0.539 | **0.000** | **1.000** |
| combined | 1.000 | 0.000 | 0.000 | 0.000 |

`yaw_rmse_rad_s`：

| segment | FP32 对照 | PTQ | A 浮点 | B fake-quant |
|---|---|---|---|---|
| turn_left | 0.0224 | 0.1465 | 0.3952 | 0.2371 |
| turn_right | 0.0295 | 0.1444 | 0.1391 | 0.2766 |

## 两个独立的发现

### 发现 1 · QAT 参数**离开量化器就不工作**

A 在**所有七个 segment 上都是 0.000**，包括那些 B 完美通过的
（forward 1.000、height 1.000、stop 0.938）。

**即：fake-quant 运算已经成为学到的函数的一部分。**
QAT 没有学出"一个能容忍量化的策略"，而是学出了
"一个只有在量化下才成立的策略"。

英文表述：*The QAT parameters are not a quantization-tolerant policy; they are
a policy that requires the quantizer. Fake-quant has become part of the learned
function.*

排除：这不是 `--fp32` 路径本身的问题——同一条路径在 FP32 对照上给出全部 1.000。

### 发现 2 · 转向在 **A 和 B 中都失败**

其余能力（forward / height / stop）在 A 中失败、在 B 中恢复，
说明它们依赖量化器交互。

**但转向在两者中都是 0.000。** 这把责任定位到**参数**，
而不是执行精度。

因此可以**排除** goal4.md §2 的第 4、5、6、8 项作为转向失败的**主因**：

* ~~4. activation quantization resolution~~ —— 若是分辨率问题，A（无量化）应当能转向
* ~~5. activation clipping / scale allocation~~ —— 同上
* ~~6. shared activation-grid constraints~~ —— 同上
* ~~8. fake-quant/export numerical mismatch~~ —— A 与 B 都失败，不是 B↔C 的差异

**剩下的候选**：1. policy drift during QAT，2. objective/reward imbalance，
3. loss of behavior preservation，7. quantization-sensitive internal features。

## 这对 goal4 中心假设的意义

goal4.md 的中心工作想法是：

> Preserve a reliable reference policy and learn only the correction required by
> low-precision execution.

本实验**强烈支持这个方向**，理由不是"它听起来合理"，而是：
**当前的 QAT 已经把参考策略彻底破坏了**——A 全 0.000 就是证据。
一个冻结 base + 只学残差的方案，在构造上不可能出现这种破坏。

这仍然是**假设**，需要 §4 的 frozen-base residual-QAT 实验来检验。

## 待补

C（导出定点）仍在运行。预期与 B 接近（两者已在 4000 样本上 bit-exact），
若 C 与 B 出现显著差异，那将是一个**新的正确性问题**而非研究结论。
