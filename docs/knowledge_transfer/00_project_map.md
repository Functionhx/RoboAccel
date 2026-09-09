# 00 · 项目地图（project map）

读代码之前需要的**最小正确心智模型**。

---

## 项目要解决什么问题

一个**轮腿机器人**（wheel-legged robot，复旦 RM2026 平衡步兵）由一个小的
神经网络策略控制。这个策略在仿真里用 PPO 训练出来，是 float32 的。
但它最终要跑在**嵌入式硬件**上：

* **RoboAccel** — Zynq-7000（XC7Z010）上的 INT16 加速器，约 **18 µs**
* **STM32H723** — 480 MHz MCU，约 **259 µs**

两者都只做**定点整数算术**（fixed-point integer arithmetic）。
于是核心问题是：

> 训练时就把部署用的整数算术放进闭环里模拟（**QAT**），
> 是否比训练完再量化（**PTQ**）保留更多**控制性能**？

注意是**控制性能**，不是数值 MSE。goal.md 明确要求：
*"Do not declare victory solely because numerical MSE is lower."*

---

## RoboAccel 是什么

**不是**把网络结构写死在 RTL 状态机里的加速器。

PS（ARM 核）把一段**算子描述符程序**（operator descriptors）写进 PL 内部
一个 32×128-bit 的 instruction RAM，然后写一次
`SEQUENCE_CONTROL.START`，硬件 sequencer 就执行完整个网络。

**换模型 = 重新导出 cache 镜像和描述符程序，不需要改 RTL。**

* AXI4-Lite slave @ `0x43C00000`，接口版本 `0x00010002`
* 算子：`GEMM`（8×8 INT16 MAC 阵列，64 个 DSP48E1，INT48 累加）、
  `ELU`、`NORM`、`CONCAT`
* 片上存储：Vector Cache 512×128-bit，Weight Cache 8 banks × 1280×128-bit
* **没有 DMA、没有 DDR master** —— PS 通过 AXI4-Lite 推入所有输入、拉出所有输出

> 这个"没有 ADD 算子"的事实后来变得关键：Codex 最终 promote 的策略是
> `base(x) + residual(x)`，需要用 CONCAT 融合绕过。见 [04](04_design_decisions.md) D7。

---

## 策略网络长什么样

```
obs[25] ──┐
          │  history[125] = 5 帧 × 25
          ↓
      encoder: 125 → 128 → 64 → 3          (2× ELU)
          ↓
      latent[3]  ← 这就是速度估计（velocity estimate）
          ↓
   CONCAT(obs[25], latent[3]) → 28         ← 不能 requantize！
          ↓
      actor:   28 → 128 → 64 → 32 → 6      (3× ELU)
          ↓
      action[6]
```

* 7 个 GEMM、5 个 ELU、1 个 CONCAT → **program length 13**（上限 32）
* **38,400 MAC**、620/1280 weight words、132/512 vector words
* critic（256/128/64）**只在训练时存在**，永远不导出

关键细节：`update_distribution` 里是 `self.latent.detach()` ——
encoder 的梯度不通过 actor 回传，它由一个独立的速度估计 MSE 辅助损失训练。
`hwq/quant_policy.py` 必须精确复现这一点。

---

## 端到端数据流（每一步对应真实文件）

```
Isaac Gym 仿真
  └─ observation[25]  ──────────────  plane/wheel_legged_gym/envs/base/robust_robot.py
       ↓
  历史缓冲 append_history → history[125]
       ↓
  ┌─ 训练期（只在训练时存在）─────────────────────────────────┐
  │  ActorCriticSequence / ActorCriticRobust                   │
  │    rsl_rl/modules/actor_critic_robust.py                   │
  │  PPO 更新  rsl_rl/algorithms/ppo.py                        │
  │  QAT 时被替换为 QuantActorCriticSequence                    │
  │    qat/hwq/quant_policy.py                                 │
  └────────────────────────────────────────────────────────────┘
       ↓  model_N.pt (float32 权重 + critic + optimizer)
       ↓
  ┌─ 导出期（只在导出时存在）─────────────────────────────────┐
  │  .pt → ONNX          qat/scripts/export_to_onnx.py         │
  │  ONNX → 定点         tools/export_policy.py                │
  │     产出 generated/*.hex、policy_map.json、rl_policy_data.c │
  │  ONNX → H7 C         qat/scripts/gen_h7_branched.py        │
  └────────────────────────────────────────────────────────────┘
       ↓
  ┌─ 运行期 ─────────────────────────────────────────────────┐
  │  FPGA:  PS 写 instruction RAM → SEQUENCE_CONTROL.START     │
  │         PL sequencer 执行 13 条描述符                       │
  │         03_vitis/src/*.c 驱动，AXI4-Lite 推 obs / 拉 action │
  │  H7:    ra_model.c + ra_kernel.c，SMLALD + DTCM             │
  │         h7_bench/src/main.c，结果走 JTAG mailbox            │
  └────────────────────────────────────────────────────────────┘
       ↓
  action[6] → 关节指令 → 机器人 → 下一个 observation
```

---

## 什么跑在哪里

| 位置 | 内容 | 文件 |
|---|---|---|
| **Host GPU** | Isaac Gym 仿真、PPO 训练、QAT、闭环评估 | `$SOLID_WT/plane/`、`qat/scripts/train_policy.py` |
| **Host CPU** | 量化分析、导出、bit-exact 验证、golden vector | `qat/hwq/`、`qat/scripts/` |
| **Zynq PS** | 写 instruction RAM、推 obs、拉 action、轮询 | `03_vitis/src/` |
| **Zynq PL** | GEMM / ELU / CONCAT 序列执行 | `rtl/` |
| **STM32H7** | 同一个网络的标量 + SMLALD 实现 | `h7_bench/src/`、`models/` |

**只在训练时存在**：critic、optimizer state、PPO rollout buffer、domain randomization。
**只在导出时存在**：ONNX、weight fracs 选择、hex 镜像、descriptor program。
**运行时存在**：weight cache、vector cache、13 条指令、ELU ROM。

---

## 仿真 / 评估 / 导出 / 验证 / FPGA / MCU 如何连起来

关键在于**同一组权重必须在五个实现上给出相同的整数结果**：

```
        torch fake-quant  ─┐
        (qat/hwq/torch_hw.py)│
                            ├─→ 必须 bit-identical ←─ numpy 整数参考（仲裁者）
        FPGA descriptors  ─┤                          (qat/hwq/fixed_ref.py)
        (tools/export_policy.py)
                            │
        H7 scalar C       ─┤
        H7 SMLALD SIMD    ─┘
        (h7_bench/)
```

验证方法与命令见 [08](08_verification_chain.md)。
**这条链断了，所有控制结论都不作数。**

---

## 一个必须先知道的历史事实

本项目的实验做过两轮：

1. **`robust_v1`（已废弃作为绝对结论）** —— 该任务存在
   **chassis-support exploit**：机器人趴在底盘上而不是靠轮子平衡，
   而旧的失效判据不把 chassis 接触算作跌倒。
   所有绝对控制数字都不是关于 locomotion 的陈述。
2. **`locomotion_v2`（当前基线）** —— Codex 修正后的任务：
   初始高度 z=0.18 m、显式的 non-wheel contact / tilt / clearance 失效判据。
   `support_fraction` 和逐段 `success` 是通过/不通过指标。

细节见 [09](09_experiment_interpretation.md)。
