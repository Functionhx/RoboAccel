# 08 · 验证链（verification chain）

**核心主张**：同一组权重在**五个独立实现**上给出 bit-identical 的结果，
其中两个在**实体芯片**上验证。

这条链是整个项目可信度的根基。如果它断了，所有控制结论都失去意义——
所以它在每次基线变更后都要重跑。

---

## 五个实现

| # | 实现 | 位置 | 角色 |
|---|---|---|---|
| 1 | torch fake-quant | `hwq/torch_hw.py` | 训练时可微，走 STE |
| 2 | numpy 整数参考 | `hwq/fixed_ref.py` | **仲裁者（arbiter）**，纯 int64 |
| 3 | FPGA descriptor program | `tools/export_policy.py` | PL 上真正执行的指令流 |
| 4 | H7 scalar C | `h7_bench/models/*/ra_model.c` | MCU 标量实现 |
| 5 | H7 SMLALD SIMD | 同上 + `ra_kernel.c` | MCU 向量实现 |

**为什么 #2 是仲裁者**：它是唯一没有浮点、没有编译器优化、
没有硬件依赖的实现。#1 用 float64 在 GPU 上**模拟**整数
（最大部分和 1.34e11 ≪ 2^53，所以是精确的），#4/#5 有编译器，
#3 有综合工具。出分歧时以 #2 为准。

---

## 逐条验证及命令

### 链路 A：torch ↔ numpy 整数参考

```bash
source scripts/env_solid.sh
$ISAAC_PY scripts/bitexact_check.py \
  --checkpoint artifacts/v2/frozen/SOLID_FP32_V2.pt \
  --n 4000 --weight-bits 16 --act-bits 16
# W8A8 需要标定：
$ISAAC_PY scripts/bitexact_check.py --checkpoint <ckpt> \
  --n 4000 --weight-bits 8 --act-bits 8 --act-fracs artifacts/v2/act_fracs_a8.json
```

期望：9 个张量（`latent`、`action`、`enc0..2`、`act0..3`）全部
`100.0000%` exact，`BIT_EXACT: PASS`。**[已验证]**

> 历史坑：这个脚本曾用固定 `1 << ACT_FRAC` 反归一化中间 trace，
> 在标定后的 W8A8 下把差异放大 16–128 倍，报出假 FAIL。
> 见 [05](05_debugging_history.md) §1.1。

### 链路 B：numpy ↔ RTL / H7 kernel（源码级）

```bash
$ISAAC_PY scripts/verify_against_rtl.py
```

它做三件事：
1. 解析 `rtl/rl_elu_array8.sv` 的 case 语句（全部 256 项）与常量；
2. 解析 `h7_bench/src/ra_kernel.c` 的建表写法，**拒绝有灾难性抵消的旧版本**；
3. 断言 `round_shift` 与 RTL 公式在 shift 0..15 上一致。

期望 `VERIFY_AGAINST_RTL: PASS`，以及 `ELU ROM entries differing: 0/256`。

### 链路 C：numpy ↔ H7 C（主机上编译执行）

```bash
bash scripts/hosttest/build_and_run.sh artifacts/v2/deploy/h7
```

在主机上编译真实的 `ra_model.c`（SMLALD 内联汇编换成语义等价的可移植 C），
用 golden vector 对比。期望 `vectors 24  simd mismatches 0  scalar mismatches 0`。

### 链路 D：numpy ↔ FPGA exporter（独立实现）

```bash
$MJ_PY tools/export_policy.py --model <onnx> --out <dir> --samples 1000 --seed 7
```

exporter **只读 ONNX**，与 `hwq/` 没有共享代码。它独立推导出：
13 条指令、7 GEMM / 5 ELU / 1 CONCAT、CONCAT 在 slot 5 且 shift 0、
620/1280 weight words、以及 weight fracs `[12,13,14,13,14,14,14]`——
与 `hwq/fixed_ref.py:choose_weight_frac` **完全一致**。**[已验证]**

这是最强的一条：**两个独立写成的实现在 7 个层的定标上逐一吻合。**

### 链路 E：实体芯片

```bash
cd /home/as/vllm/fpga/projects/h7_bench
source /home/as/vllm/fpga/env/stm32.sh
bash scripts/flash.sh solid_v2
python3 scripts/read_results.py solid_v2
```

期望：`verify: scalar=PASS  simd=PASS`，且回读的 `weights` / `MACs`
与主机生成完全一致。结果通过 **JTAG mailbox** 读回（板上 CH340 未接本机）。

最近一次：124,363 cycles = **259.09 µs**，200 次运行，DWT CYCCNT。**[已验证]**

---

## 完整回归（换基线后必跑）

```bash
# 1. 算术层
$ISAAC_PY scripts/verify_against_rtl.py
$ISAAC_PY scripts/bitexact_check.py --checkpoint <new> --n 4000 --weight-bits 16 --act-bits 16
$ISAAC_PY scripts/bitexact_check.py --checkpoint <new> --n 4000 --weight-bits 8 --act-bits 8 --act-fracs <fracs>
# 2. 导出层
PYTHONPATH=$PWD $MJ_PY scripts/export_to_onnx.py <ckpt> --out <onnx>
$MJ_PY ../tools/export_policy.py --model <onnx> --out <dir> --samples 1000 --seed 7
PYTHONPATH=$PWD $MJ_PY scripts/gen_h7_branched.py <onnx> --name <n> --out <h7dir> \
    --golden 24 --golden-npz <regime_obs.npz>
bash scripts/hosttest/build_and_run.sh <h7dir>
# 3. 芯片层
（见链路 E）
```

**任何一条不过，控制结论都不作数**——这是 `run_full_v2.sh` 把算术门
放在最前面的原因。

---

## 一条容易漏的：基础设施是否变了

Codex 的改动是**未提交的工作区状态**，git SHA 无法识别。
换基线或怀疑上游变动时：

```bash
cd "$SOLID_WT" && sha256sum -c \
  /home/as/vllm/fpga/projects/rl_accel/qat/artifacts/v2/frozen/infra_manifest_final.sha256
```

若有文件 FAILED，**不要假设它无害**——用一个已知结果重跑评估并比对。
本项目做过两次，两次都是 **0.0000% bit-identical**，
但那是**测出来的**，不是猜的。
