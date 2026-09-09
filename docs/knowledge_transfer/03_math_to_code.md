# 03 · 数学 → 代码 → 硬件（math to code to hardware）

每一节的结构：**数学定义 → 本仓库的实现 → 硬件上的样子 → 容易踩的坑**。

---

## 1. 定点表示（fixed-point representation）

### 数学
一个 `Qm.f` 定点数用整数 `q` 表示实数 `x`：

```
x ≈ q · 2^(-f)          f = fractional bits，LSB = 2^(-f)
```

INT16 时 `q ∈ [-32768, 32767]`，量程 `±32768·2^(-f)`。

### 代码 `hwq/fixed_ref.py`

```python
def quantize(v, frac, bits=16):
    return np.clip(round_away(np.asarray(v, np.float64) * (2.0 ** frac)),
                   QMIN[bits], QMAX[bits]).astype(np.int64)
```

**注意是 `2.0 ** frac` 而不是 `1 << frac`。** 标定出的 frac **可以为负**：
在 `robust_v1` 的 INT8 下 `enc0` 峰值 130.7，需要 frac = −1（LSB = 2.0），
`1 << -1` 会直接抛 `ValueError`。这是真实踩过的坑。

### 硬件
激活默认 **Q8.8**（`ACT_FRAC = 8`，LSB = 1/256，量程 ±128）。
权重每个 GEMM 有自己的 frac，上限 `MAX_WEIGHT_FRAC = 14`。

---

## 2. 再量化（requantization）—— 本项目最重要的一段

### 数学
GEMM 的累加结果在 INT48。要写回 INT16 激活网格，需要右移：

```
shift = f_w + f_in − f_out
out   = round(acc · 2^(−shift))
```

### 硬件真正做的事 `rtl/rl_round_shift_sat16.sv`

```
先加半个 LSB（away from zero），再算术右移（floor）
```

对负数，"加了 half 之后再 floor" 会**过度修正**：

```
v = -8, shift = 1:  (-8 - 1) >> 1 = floor(-4.5) = -5
真正的 round-half-away-from-zero:              = -4
```

**99.6% 的负数会低一个 LSB。**

### 代码 `hwq/fixed_ref.py:round_shift`

```python
def round_shift(v, shift):
    if shift == 0: return v
    if shift < 0:  return v << (-shift)
    half = np.int64(1) << (shift - 1)
    return (v + np.where(v < 0, -half, half)) >> shift
```

torch 侧对应 `hwq/torch_hw.py:hw_round_shift`。

### 为什么不修
见 [04](04_design_decisions.md) D1：**匹配硬件优先于正确**。
一个"更正确"的参考会让 golden vector 在板子上失配。

`IDEAL_ROUNDING = True` 可以切换到精确舍入，用于测量这个偏置的代价——
但它**不匹配当前 RTL**，默认关闭。

### 实测代价 **[已验证]**

| | W16A16 | W8A8 |
|---|---|---|
| 与理想舍入不同的动作比例 | 86.5% | **91.8%** |
| 均值偏置 | −0.052 LSB | +0.133 LSB |
| **左右轮差速** | — | **+4.05 rad/s** |

**重点是差速，不是均值。** 各动作偏置符号不同会在均值里抵消，
但两个轮子之间的差值 ≈ 4 rad/s，相当于一个**持续的偏航力矩**。

---

## 3. GEMM

### 数学
`y = W·x + b`，`W ∈ ℝ^(N×K)`，MAC 数 = `K·N`。

### 定点形式
```
acc[n] = Σ_k Wq[n,k] · xq[k]        # INT48 累加
y_q[n] = round_shift(acc[n], f_w + f_in − f_out) + b_q[n]
```

bias 直接量化到**输出网格** `f_out`，所以在移位**之后**相加。

### 硬件
8×8 的 INT16 MAC 阵列，64 个 DSP48E1。一组 ROWS 个输出累加
`ceil(K/COLS)` 拍后 requantize 写回。weight cache 占用：

```
words = ceil(K/8) · ceil(N/8)      # 每个 128-bit word 放 8 个 INT16
```

本策略 7 层合计 **620 words**（上限 1280）。

---

## 4. ELU

### 数学
```
ELU(x) = x            if x ≥ 0
       = e^x − 1      if x < 0        （α = 1）
```

### 硬件 `rtl/rl_elu_array8.sv`
* x ≥ 0 直接透传
* x < 0 查 **256 项 ROM**，覆盖 `[-8, 0)`，步长 1/32
* ROM 之间用 **3-bit 线性插值**
* x < −8 饱和到 −1.0

### 代码 `hwq/fixed_ref.py:elu_q88`
逐位复现上述过程（查表 + 插值 + 饱和）。

### 坑：建表方式
H7 固件曾用**交替级数**直接算 `exp(-x)`。x 接近 8 时单项达 ~400 而和只有
~3e-4 → **灾难性抵消**，表项 247..255 变成负数。
正确写法是 `1/exp(+x)`。见 [05](05_debugging_history.md) §3.1。

`scripts/verify_against_rtl.py` 现在会**拒绝旧写法**。

---

## 5. CONCAT 与它的约束

### 数学
纯粹的数据搬移：`z = [a ; b]`。

### 硬件限制
**PL 的 CONCAT 不能 requantize**（描述符里 `shift = 0`）。
所以拼接的两个操作数**必须已经在同一个网格上**：

```
frac(obs) == frac(latent)
```

### 代码 `hwq/fixed_ref.py:FudanFixedPolicy.__init__`

```python
assert self.latent_frac == self.obs_frac, (
    "concat needs obs and latent on one grid; PL CONCAT cannot requantize")
```

`scripts/calibrate_act_fracs.py` 因此取两者中**更紧的那个**：

```python
shared = min(fracs["obs"], fracs["enc2"])
fracs["obs"] = fracs["enc2"] = shared
```

### 第二个同类约束（residual actor）
Codex 最终 promote 的策略是 `base(x) + residual(x)`，而 PL **没有 ADD**。
解法：两条分支的末层 GEMM 融合成一个在拼接激活上的 GEMM

```
W_b @ a + c_b  +  W_r @ b + c_r  ==  [W_b | W_r] @ [a ; b] + (c_b + c_r)
```

真实权重上验证：最大差 **4.8e-07**。
于是产生新约束 `frac(base_penult) == frac(residual_penult)`，
由 `FudanFixedPolicy.check_residual_concat` 断言。

---

## 6. 直通估计器（straight-through estimator, STE）

### 问题
`round()` 的导数几乎处处为 0，梯度无法回传。

### 数学
前向用量化值，反向**假装它是恒等函数**：

```
forward:   y = quantize(x)
backward:  ∂y/∂x = 1        （在未饱和区间内）
```

### 代码 `hwq/torch_hw.py:_QuantizeAct`
用 `torch.autograd.Function`，并带 **clip-aware mask**：
饱和的输出和被裁剪的权重处梯度置 0，否则会把网络往量程外推。

### 为什么 float64 是精确的
GPU 上用 float64 模拟整数：本网络最大部分和 **1.34e11 ≪ 2^53**，
所以 float64 能精确表示每一个中间整数。这是 torch 路径能与 numpy
整数参考 bit-identical 的原因。

---

## 7. 逐层激活 frac 的必要性

各层动态范围差 7 倍。固定 Q8.8 在 INT8 下量程只有 ±0.5，
而实际激活到 22 —— **未标定的 INT8 run 测的是饱和，不是量化**。

`locomotion_v2` 标定结果：

| 层 | 峰值 | frac | LSB | ±1σ 内的台阶数 |
|---|---|---|---|---|
| enc0 | 12.81 | 2 | 0.250 | — |
| enc1 | 16.60 | 2 | 0.250 | — |
| enc2 (latent) | 3.56 | 3 | 0.125 | ~6 |
| act3 (action) | 7.63 | 3 | 0.125 | — |

标定后**裁剪率 0.0000%** —— 崩溃是**分辨率**问题，不是饱和问题。
