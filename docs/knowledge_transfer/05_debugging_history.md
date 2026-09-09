# 05 · 调试历史（debugging history）

goal2.md §16 把调试历史列为**最高优先级**，理由是：代码结构以后能重读，
但"当时为什么会怀疑这里"无法重建。

本项目最危险的 bug **不会报错**。它们产生完全合理的数字。
下面每一条都记录三件事：**症状 / 真正原因 / 当时是什么线索让人起疑**。
第三条才是可迁移的部分。

---

## 一、致命的一类：产生"看起来对"的结果

### 1.1 固定 ACT_FRAC 假设（三次重复出现）

**症状**
* `quant_diagnostics.py` 在给了 `--act-fracs` 之后报告 **80–96% saturation**。
* `bitexact_check.py` 在 W8A8 下报 `FAIL`，中间张量差到 16129 LSB。

**真正原因**
两个脚本都写在**逐层激活标定（per-layer activation calibration）出现之前**，
内部硬编码了模块级常量：

```python
LSB = 1.0 / (1 << ACT_FRAC)          # ACT_FRAC = 8，固定 Q8.8
a_qmax = QMAX[act_bits] * LSB
r = np.stack([tr[short] for tr in traces]) / (1 << ACT_FRAC)
```

一旦每层有自己的 frac（本项目 INT8 下 frac ∈ [1,4]），这些归一化就错了
2^(8−frac) 倍，即 **16 到 128 倍**。

**起疑的线索（可迁移）**
* saturation 那次：**两个不同的策略给出了完全相同的 saturation 数字**。
  不同的网络不可能碰巧饱和率一致 → 这个数字不依赖输入 → 它是常量算出来的。
* bitexact 那次：**`action` 100% 精确，而中间层全错**。
  这在物理上不可能——`enc0` 的误差必然传播到 `action`。
  所以错的是**比较方式**，不是算术。

**修复** `hwq/torch_hw.py` 的 `FRAC_KEY` 映射 + 每个 tap 用自己层的 frac；
`scripts/bitexact_check.py` 的 `cmp(..., lsb=...)` 参数。修复后：
所有张量 0.0000% 差异，saturation 0.0000%。**[已验证]**

**教训**：一个"太整齐"的数字（两处相同、恰好 100%）比一个难看的数字更可疑。

---

### 1.2 policy class 替换成了函数而不是类

**症状**
QAT 训练启动即崩：`'function' object has no attribute 'is_sequence'`。

**真正原因**
`install_quant_policy()` 把 `opr.ActorCriticSequence` 替换成一个闭包。
但 `on_policy_runner.py:70-72` 在**实例化之前**读类属性：

```python
actor_critic_class = {...}[self.cfg["policy_class_name"]]
if actor_critic_class.is_sequence:            # 类属性，闭包没有
    num_critic_obs += self.policy_cfg["latent_dim"]
    self.policy_cfg["num_encoder_obs"] = ...
```

**为什么这次是走运**
它**崩了**。如果 runner 写的是 `getattr(cls, "is_sequence", False)`，
函数会被接受，那个分支会被静默跳过，encoder 输入宽度就会错——
训练照常收敛，报出一个毫无意义的 QAT 数字。

**修复** 用 `class factory(QuantActorCriticSequence)` 子类替代闭包，
并在构造后加断言：

```python
if not isinstance(runner.alg.actor_critic, QuantActorCriticSequence):
    raise SystemExit("quantization requested but the runner built ...")
```

成功时打印 `quantized policy confirmed: factory W8A8`。**[已验证]**

---

### 1.3 只 patch 了 `ActorCriticSequence`，没 patch `ActorCriticRobust`

`robust_v1` 的 runner 配置是 `policy_class_name = "ActorCriticRobust"`。
只替换前者的话，查表命中的是**未被替换的** `ActorCriticRobust` →
训练出一个**完全没有量化**的策略，却被标记成 QAT 结果。

同样没有报错，只有 1.2 的断言能抓到。**[已验证]**

---

### 1.4 hardcoded `--task wheel_legged`

`train_policy.py` 向 Isaac Gym 传的 argv 里写死了旧任务名，
而 `make_env(name=known.task)` 用的是新任务名。结果：
**QAT 和它的对照组会在与 checkpoint 不同的环境里训练**——
正是 goal.md §1 设置实验树要避免的因果归因错误。**[已验证]**

---

## 二、会浪费时间但不会污染结论的一类

| 现象 | 原因 | 教训 |
|---|---|---|
| `nohup cmd > dir/log` 静默失败，wrapper 仍报 exit 0 | shell 在 exec 脚本**之前**打开重定向，而 `mkdir -p` 在脚本内部 | 先建目录；启动后**验证进程存在**，不要相信退出码 |
| 评估臂被静默跳过 | **在 bash 执行脚本的过程中编辑了该脚本**。bash 按字节偏移增量读取，插入行后偏移错位，从命令中间恢复执行 | 永远不要编辑正在运行的 shell 脚本 |
| 两个 watcher 互相死锁 | `until ! pgrep -f foo.sh` **匹配到自己的命令行** | 改为 grep 日志里的完成标记，或匹配解释器路径 |
| seed-3 QAT 启动一小时后才发现没跑 | `IFS='|' read -r run seed a b c` 定长读取，把 `--act-fracs` 和路径粘成一个 token | 参数按位置传递，不要打包进分隔字符串 |
| 第四个训练 CUDA illegal memory access | 4 个 4096-env 作业超出 16 GB | "用满 GPU" ≠ "同时跑到崩"；见 [11](11_operations_playbook.md) 的并发结论 |

---

## 三、真实的硬件 bug（不是我们的代码）

### 3.1 STM32H7 的 ELU ROM 有 9 项是错的

**症状** H7 输出与 numpy 参考在少数样本上不一致。

**原因** 固件用交替级数直接算 `exp(-x)`：x 接近 8 时单项达到 ~400，
而和只有 ~3e-4 → **灾难性抵消（catastrophic cancellation）**，
表项 247..255 变成负数。

**修复** 改为 `1/exp(+x)`：

```c
double x = ((double)i) / 32.0;
double e = 1.0, term = 1.0;
for (int n = 1; n < 24; ++n) { term *= x / (double)n; e += term; }
ra_table[i] = (int16_t)ra_round_away((1.0 / e - 1.0) * 256.0);
```

`scripts/verify_against_rtl.py` 现在会**拒绝旧写法**（回归护栏已验证会在旧代码上失败）。**[已验证]**

### 3.2 `SysTick_Handler` 未实现 → CPU 卡死

板上绿灯不亮的真正原因。WS2812 更新会调用 `HAL_GetTick()`，
而未实现的 handler 落进 `Default_Handler` 的 `b .` 死循环。
benchmark 本身不调用 tick，所以 mailbox 仍然有效——
**这正是为什么"结果正确"没能掩盖 CPU 已经死了**。**[已验证]**

> 注：我对绿灯问题的**第一个解释是错的**（归因于 20% duty 心跳）。
> 记录在此是因为"第一个合理解释往往是错的"本身就是可迁移的。

### 3.3 requantizer 的非理想舍入（**故意保留**）

`rl_round_shift_sat16.sv` 先加半个 LSB（away from zero）再算术右移。
负数因此有 99.6% 的概率比真正的 round-half-away-from-zero **低一个 LSB**。

这**不是要修的 bug**，而是**必须复现的行为**——见
[04](04_design_decisions.md) 的"匹配硬件优先于正确"。
它的实际代价见 [09](09_experiment_interpretation.md) §20。

---

## 四、如果你要继续这个项目

新增 bug 时按同样三段式追加到本文件。特别记录**当时是什么让你起疑**——
那部分才是别人（和未来的你）真正用得上的。
