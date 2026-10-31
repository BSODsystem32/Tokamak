# Tokamak (`.tkm`) Language Specification

This document defines the complete syntax, typing, branching semantics, and built-in operator library for the `.tkm` neural network description language.

---

## 1. Top-Level Statements

A valid `.tkm` file consists of top-level key-value headers followed by an optional `flow:` block.

### 1.1 Input Declaration (`input:`)
Specifies the input tensor shape in `HxWxC` format (or `1xSeqxDim` for sequences).
```yaml
input: 28x28x1       # 2D Image: Height=28, Width=28, Channels=1
input: 1x256x512     # Sequence: Batch=1, Seq=256, HiddenDim=512
```

### 1.2 PRNG Engine Selection (`prng:`)
Sets the default Pseudorandom Number Generator for all probabilistic operators (`Drop`, `Rand*`).
* `wang_xorshift32` (Default): Thomas Wang 32-bit Hash + Xorshift32. Zero multiplication, high avalanche.
* `pcg32`: PCG-XSH-RR 64-bit state LCG. Gold-standard statistical uniformity.
* `philox`: Counter-based 7-round bijective cipher. Stateless and deterministic across CUDA threads.
* `xorshift32`: Raw 3-op shift register. Fastest execution.

```yaml
prng: pcg32
```

---

## 2. Structural Syntax & Flow Semantics

### 2.1 Sequential Pipeline (`->`)
Chains operators where the output of operator $A$ directly feeds into operator $B$.
```yaml
conv1: Conv(c=32, k=3, s=1, p=1) -> BN -> SiLU -> MixPool(2, ratio=0.85)
```

### 2.2 Parallel Branching & Merge (`{ ... } -> MergeOp`)
Encloses parallel sub-chains that share the identical input tensor before the opening `{`.
* If a branch specifies `Identity`, the input tensor passes through without memory copies.
* The closing `} -> MergeOp` combines the branch outputs using either `Add` (elementwise summation), `Mul` (elementwise gating), or `GQA` (attention mechanism).

```yaml
# Residual Block
res3: {
  branch: Conv(c=64, k=3, s=1, p=1) -> BN(0.35) -> SiLU
  skip  : Identity
} -> Add

# Multi-Branch Gated FFN
ffn: {
  gate: FC(1024) -> SiLU
  up  : FC(1024)
} -> Mul -> FC(512)
```

### 2.3 Loop Macro Expansion (`block_name x N: { ... }`)
Automatically unrolls repeated sub-networks $N$ times. Each iteration generates independent SSA node IDs, independent trainable weight tensors, and isolated memory slots.
```yaml
layer x 12: {
  attn: {
    branch: RMSNorm -> { q: FC(512), k: FC(128), v: FC(128) } -> GQA(q=8, kv=2, d=64) -> FC(512)
    skip  : Identity
  } -> Add
  ffn: {
    branch: RMSNorm -> { gate: FC(1024) -> SiLU, up: FC(1024) } -> Mul -> FC(512)
    skip  : Identity
  } -> Add
}
```

---

## 3. Built-In Operator Reference (All 42 Operators)

Plasma operators and `MixPool` use CUDA `half2` kernels and require an even channel count. Graphs that violate this constraint are rejected during optimization with a shape error.

| Category | Operator | Arguments & Defaults | Description |
| :--- | :--- | :--- | :--- |
| **Compute Anchors** | `Conv` | `c` (out channels), `k=3`, `s=1`, `p=0` | 2D Spatial Convolution (Direct Conv1 for $C_{in}=1$; Im2Col+GEMM for general) |
| | `FC` | `c` / positional (out features) | Fully Connected Linear Layer (Tensor Core GEMM) |
| | `GQA` | `q=8`, `kv=2`, `d=64` | Hardware WMMA Tensor Core Grouped-Query FlashAttention-2 |
| **Spatial Transform** | `MixPool` | `k=2`, `ratio=0.85` | Mixed Anti-Aliased 2x2 Pooling: `ratio*Max + (1-ratio)*Avg` |
| | `AvgPool` | `k=2` | Standard 2x2 Average Pooling |
| | `Flatten` | None | Flattens $[H, W, C]$ to $[1, 1, H \cdot W \cdot C]$ |
| **2D Augmentation** | `RandShift` | `fval=2` (max shift pixels) | Random spatial translation $(dx, dy)$ |
| *(Fused, 0 SMEM)* | `RandRotate` | `fval=10.0` (max angle deg) | Random rotation around image center $(cx, cy)$ |
| | `RandScale` | `fval=0.10` (delta scale) | Random zoom-in/out scaling |
| | `RandFlipX` | `fval=0.5` (probability) | Random horizontal reflection |
| | `RandFlipY` | `fval=0.5` (probability) | Random vertical reflection |
| | `RandElastic` | `fval=1.6` (alpha), `r=4` (sigma) | Continuous 2-octave harmonic elastic displacement field |
| | `RandNoise` | `fval=0.03` (std dev) | In-register Gaussian noise injection |
| | `RandCutout` | `fval=4.0` (box size) | Random zero-masking rectangular patch |
| | `RandBrightness` | `fval=0.15` (delta) | Random additive brightness shift |
| | `RandContrast` | `fval=0.20` (delta) | Random multiplicative contrast adjustment |
| **Normalization** | `RMSNorm` | `eps=1e-5` | In-Register RMSNorm: $\frac{x}{\text{RMS}(x)} \cdot \gamma$ |
| | `LN` | `eps=1e-5` | In-Register LayerNorm: $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$ |
| | `BN` | `eps=1e-2` | Fast Channel-wise Group Normalization |
| | `L2Norm` | `eps=1e-5` | Parameter-free unit hypersphere projection: $x / \|x\|_2$ |
| | `QKNorm` | `eps=1e-5` | Sub-Warp per-head RMSNorm executed before RoPE |
| **Pointwise Fluid** | `SiLU` | None | Sigmoid Linear Unit: $x \cdot \sigma(x)$ |
| | `ReLU` | None | Rectified Linear Unit: $\max(0, x)$ |
| | `ReLU2` | None | Squared ReLU: $\max(0, x)^2$ (Primer / Nemotron) |
| | `LeakyReLU` | `alpha=0.01` | Leaky ReLU with custom slope on negative axis |
| | `ELU` | `alpha=1.0` | Exponential Linear Unit: $\alpha(e^x - 1)$ for $x < 0$ |
| | `GELU` | None | Fast Tanh Gaussian Error Linear Unit |
| | `Mish` | None | Mish Self-Regularized Activation: $x \cdot \tanh(\ln(1 + e^x))$ |
| | `HardSwish` | None | Piecewise Linear Hard Swish: $x \cdot \text{clip}(x + 3, 0, 6) / 6$ |
| | `HardSigmoid` | None | Piecewise Linear Hard Sigmoid: $\text{clip}(x + 3, 0, 6) / 6$ |
| | `Tanh` | None | Standard Hyperbolic Tangent |
| | `Sigmoid` | None | Standard Logistic Sigmoid |
| | `SwishBeta` | `beta=1.5` | Parametric Swish: $x \cdot \sigma(\beta x)$ |
| | `Sin` | `freq=1.0` | Periodic Sine Activation for SIREN / PINN |
| | `CosGate` | None | Cosine Modulation: $x \cdot \cos(x)$ |
| | `SoftCap` | `cap=30.0` | Logit/Activation Soft-Capping: $\text{cap} \cdot \tanh(x / \text{cap})$ |
| | `Clamp` | `limit=6.0` | Symmetric value bounding: $\text{clip}(x, -L, L)$ |
| | `Scale` | `fval` (multiplier) | Constant scalar multiply |
| | `LayerScale`| `fval=1e-4` (init scale) | Learnable per-channel diagonal scaling vector |
| | `DyT` | `alpha=0.5` | Dynamic Tanh Modulation: $\tanh(\alpha x) \cdot \gamma$ |
| | `PReLU` | `init=0.25` | Learnable per-channel parametric negative slope |
| | `xATLU` | `init=0.5` | Learnable gated interpolation between Linear and SiLU |
| | `Drop` | `p=0.1` | In-Register Dropout with bitmask storage |
| | `RoPE` | None | Rotary Position Embedding |
| | `SoftmaxCE` | None | Fused Softmax Cross-Entropy Loss Sink |
```
