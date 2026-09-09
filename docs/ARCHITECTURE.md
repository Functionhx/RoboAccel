# 架构与接口契约 · Architecture and Interface Contract

本文回答一个问题：**四个组件之间到底靠什么连接。**

RoboAccel 的四个部分（`training/` `quantization/` `fpga/` `stm32/`）
是独立实现的。它们之间**不共享代码**——共享的是**数值定义**和
**少数几个显式产物**。这既是本项目验证策略的基础
（两个独立实现的一致性才算证据），也是最容易出问题的地方。

---

## 1. 规范的策略表示（canonical policy representation）

| 阶段 | 表示 | 定义位置 |
|---|---|---|
| 训练 | float32 `state_dict` | 外部 RL 环境的 `ActorCriticSequence` 布局 |
| 量化边界 | `QuantActorCriticSequence` | `quantization/roboaccel_quant/quant_policy.py` |
| **仲裁者** | **`FudanFixedPolicy`（NumPy int64）** | **`quantization/roboaccel_quant/fixed_ref.py`** |
| 交换格式 | ONNX | `quantization/scripts/export_to_onnx.py` |
| FPGA | 描述符程序 + cache 镜像 | `fpga/tools/export_policy.py` |
| STM32 | 生成的 C 数组 + kernel | `quantization/scripts/gen_h7_branched.py` |

**`fixed_ref.py` 是唯一的真值定义。** 其余四条路径都是它的独立复现，
一致即证据。任何关于"正确"的争议都以它为准。

## 2. 每一步产出什么、消费什么

```
训练            产出  model_N.pt              (float32 state_dict + critic)
量化            消费  model_N.pt + calib .npz
                产出  policy.onnx             (交换格式)
                     fracs.json               (逐层激活小数位)
                     golden vectors           (24 组，跨实现的判据)
FPGA 导出       消费  policy.onnx
                产出  generated/*.hex         (cache 镜像)
                     generated/policy_map.json  ← 地址的**权威来源**
                     rl_policy_data.c
STM32 codegen   消费  policy.onnx + golden .npz
                产出  stm32/models/<name>/{ra_model.c,ra_model.h,ra_golden.h}
```

**FPGA 与 STM32 共享的表示是 `policy.onnx` 加同一份算术定义**，
不是彼此的中间产物。两条后端路径**不互相依赖**——这是刻意的，
否则它们的一致就不构成独立证据。

## 3. 数值契约

这是全部四个组件唯一必须逐位一致的东西：

| 项 | 值 | 定义位置 |
|---|---|---|
| 激活 / bias | INT16，Q8.8（`ACT_FRAC = 8`） | `fixed_ref.py` |
| 权重 | INT16，**逐 GEMM** 小数位，`MAX_WEIGHT_FRAC = 14` | `fixed_ref.py` |
| 累加 | INT48 | `fpga/rtl/rl_mac_array_8x8.sv` |
| 重量化 | 先加半 LSB，再**算术右移（向下取整）**，饱和到 INT16 | `fpga/rtl/rl_round_shift_sat16.sv` |
| ELU | 256 项 ROM，`[-8,0)` 上 1/32 网格 + 3 bit 插值 | `fpga/rtl/rl_elu_array8.sv` |

> **重量化的取整语义是硬件事实，不是缺陷。**
> 负数有 99.61% 的概率低一个 LSB。Python 参考**刻意复现**这一点。
> 把它"修好"会让参考与硅片不一致——已在
> `docs/knowledge_transfer/` 中作为一条设计决策记录。

`quantization/scripts/verify_against_rtl.py` **解析 RTL 与 C 源码本身**
来检查这张表，而不是相信文档。

## 4. 一条命令判断一个策略能否部署

```bash
python quantization/scripts/verify_against_rtl.py     # 算术契约
make -C fpga/tb all                                   # 8 个 testbench
bash quantization/scripts/hosttest/build_and_run.sh stm32/models/<name>
```

前两条不需要 checkpoint，也不需要板子；第三条需要先跑 codegen。
**目前没有单一的 `deployable?` 入口**——这是一个已知的缺口，
见 §7 findings。

## 5. 容量约束（exporter 与 PS 必须自己检查）

PL **不做**维度范围检查：

* 每个权重 bank：`sum(ceil(K_i/8) * ceil(N_i/8)) <= 1280`
* vector 布局必须放得下 512 个 128-bit word
* 程序长度 1..32 条描述符
* 自动序列执行期间，host **不得**触碰 cache 或指令 RAM

当前模型：7 GEMM / 5 ELU / 1 CONCAT，13 条描述符，38,400 MAC，620/1280 word。

## 6. 版本与地址的权威来源

| 想知道 | 读哪里 | **不要**读 |
|---|---|---|
| cache 地址 | `generated/policy_map.json` | 任何散文文档 |
| 寄存器布局 | `fpga/sw/rl_accel_regs.h` + `fpga/rtl/rl_accel_params.vh` | 文档中的表格 |
| 接口版本 | PL `VERSION` 寄存器（当前 `0x00010002`） | README |
| 逐层小数位 | `policy_map.json` 与 `fracs.json` | 代码里的常量 |

寄存器或协议一旦改动，必须**同时**落到：
`rl_accel_regs.h`、`rl_accel_params.vh`、AXI 顶层、两个 PS driver、
testbench、文档，以及 `VERSION` 寄存器。

---

## 7. Findings —— 已知的脆弱边界

按严重程度排列。这些是**当前**状态，不是历史。

### F0 · RTL 与三个软件实现在第二个 checkpoint 上不一致 —— **高**

导出 `SOLID_FP32_V2` 得到的描述符程序与已部署模型**结构完全相同**
（同样 13 条指令、同样的 base/dim/cache 布局），三个软件实现之间
**500 样本 bit-identical**，但 `tb_policy_e2e` 六个 action 全部不符。

已排除：exporter 陈旧（用发布版 exporter 导出原模型 → PASS）、
testbench 回归（同上）、shift 过大（上限压到 13、12 仍失败）、
cache 镜像缺失、容量越界、激活饱和
（**原模型才是撞到 Q8.8 上限的那个，而它通过**）。

**这条直接影响本项目的核心架构主张**——"换策略只需重新导出，不需要改 RTL"。
详见 `RELEASE_CANDIDATE.md` §4。未改 RTL：证据定位了分歧，
但尚未定位机理，在机理不明时改硬件是猜测。

### F1 · 没有单一的"可部署吗"入口 —— 中

判断一个策略能否部署需要人工跑三条命令并读三份输出。
应当有一个 `scripts/check_deployable.py`，
输入 checkpoint 或 ONNX，输出一个布尔值加一份清单：
容量约束、算术契约、golden vector、描述符长度。

### F2 · `training/` 依赖外部环境，且没有版本锁 —— 中

`training/scripts/train_policy.py` 通过 `PYTHONPATH` 导入外部 RL 环境，
但没有任何机制断言那个 checkout 的版本。上游改了策略类的键名
（`log_std` → `std`）就曾经悄悄破坏加载。
适配器现在处理了这一种情况，但**下一次改名不会被自动发现**。

### F3 · 交换格式经过 ONNX，损失了量化元数据 —— 中

`policy.onnx` 只携带 float 权重与拓扑。逐层激活小数位走的是
**另一条路**（`fracs.json`），两者靠文件名约定而不是显式接口关联。
一次错配会产生一个能跑但数值错误的模型。
应当把 fracs 嵌入 ONNX metadata，或者定义一个显式的
`policy_bundle.json` 同时引用两者。

### F4 · `.vvp` 构建产物曾被提交 —— 低（已修）

`fpga/tb/*.vvp` 是 Icarus 的编译输出，不应进入版本库。已加入
`.gitignore` 并删除。

### F5 · 容量约束只在 exporter 里检查 —— 低

PL 不检查，PS driver 也不检查。一个手工构造的描述符程序可以
越界写 cache 而不报错。这在当前工作流下不会发生
（描述符只由 exporter 生成），但它是一个**隐式**的安全边界，
不是显式的。

### F6 · 时间边界的定义分散 —— 低

T1（纯推理）/ T2（调用）/ T3（obs→action）三个边界的定义
写在分析脚本的 docstring 里，而不是一个共享常量。
两个后端各自实现了同一套定义。目前一致，但没有机制保证它继续一致。

---

## 8. 显式接口优于隐式文件系统约定

上面 F1–F6 有一个共同点：**它们都是"约定"而不是"接口"**。
本项目在数值上极其严格（逐位比对、独立实现互相仲裁），
但在**产物之间的连接**上仍然依赖文件名和目录布局。

这份文档的目的就是把这些约定写下来，
让下一步可以把它们逐条变成显式接口。
