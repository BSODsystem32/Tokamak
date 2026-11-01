<div align="center">
  <img src="assets/logo.png" width="220" alt="Tokamak Logo" />
  <h1>⚡ Tokamak (tkm)</h1>
  <p><b>Bare-Metal Markup-to-CUDA Graph JIT Compiler & Deep Learning Engine</b></p>
  <p><i>Zero Python. Zero PyTorch. Zero LLVM. Pure C & NVRTC.</i></p>
</div>

---

## 💡 What is Tokamak?

**Tokamak (`tkm`)** is an ultra-lightweight, high-performance deep learning JIT compiler designed for researchers and hackers who want to build custom neural network architectures like puzzle blocks—**without being forced into the heavy Python / PyTorch / Triton ecosystem.**

You write your model architecture in a clean `.tkm` markup DSL. Tokamak parses the computational DAG, automatically infers all tensor shapes, applies a two-pass peephole fusion optimizer (horizontal GEMM merge + vertical register confinement), and compiles the entire forward, backward, and optimizer pipeline into **native CUDA C++ via in-memory NVRTC**—finally launching as a **single, zero-CPU-overhead `cudaGraphExec_t`**.

---

## 🚀 Key Highlights

* **Pure Markup DSL (`.tkm`)**: Express complex multi-branch architectures (`{ branch, skip } -> Add`), loop macro expansion (`layer x 12: { ... }`), and custom exotic activations in seconds.
* **Auto Shape Inference**: Never manually compute `Flatten` offsets, Conv padding, or FC input dimensions again.
* **Two-Pass Peephole Fuser**:
  * **Horizontal Fusion**: Merges parallel sibling projections into single GEMMs (e.g., 2-way SwiGLU / GeGLU $\to$ single `2x GEMM`; 3-way GQA QKV $\to$ single `3x GEMM`).
  * **Vertical Plasma Confinement**: Fuses Pointwise + Normalization + Residual-Add into single-launch NVRTC kernels, eliminating up to **89% of intermediate VRAM traffic**.
  * **Algebraic Scale & Identity Folding**: Dead-code elimination for no-op scales and dropouts.
* **First-Principles Numerical Finite-Difference Calculus Verification (`tkm numcheck`)**:
  Validates analytical CUDA gradients directly against numerical secant limits $\lim_{\epsilon \to 0} \frac{\mathcal{L}(x+\epsilon) - \mathcal{L}(x-\epsilon)}{2\epsilon}$ in hardware registers.
* **Zero-Memory Register Micro-Replay**: Backpropagation reconstructs intermediate activations in hardware registers rather than saving multi-megabyte activation maps to VRAM.
* **100% Deadlock-Free Sub-Warp Reduction**: Guaranteed convergence across arbitrary channel counts without intra-warp divergence.
* **Hardware Tensor Core (WMMA 16x16x16) GQA FlashAttention-2**: Hardware `mma_sync` acceleration for both Forward and Backward attention matrices with bank-conflict-free shared memory.
* **Multi-Scheme PRNG Engine**: Built-in support for `pcg32`, `philox4x32_7`, `wang_xorshift32`, and `xorshift32`—selectable globally or per-operator.
* **Math-Defined Custom Kernels (`src/tkm_registry.c`)**: Add supported pointwise research operators with scalar forward and derivative formulas; the engine emits the CUDA kernel, so no CUDA code or new enum is required. Reductions, standalone Softmax, and loss sinks remain runtime-specialized.

---

## 🧩 The `.tkm` Markup Language

### Example 1: 4-Layer Transformer with GQA, QK-Norm, RoPE, and Asymmetric FFN
```yaml
input: 1x256x512
prng : pcg32

flow:
  layer x 4: {
    # 1. Pre-RMSNorm + GQA FlashAttention + Post-Attn Soft-Capping + Residual
    attn: {
      branch: RMSNorm -> {
        q: FC(512) -> QKNorm -> RoPE
        k: FC(128) -> QKNorm -> RoPE
        v: FC(128)
      } -> GQA(q=8, kv=2, d=64) -> FC(512) -> SoftCap(50.0) -> LayerScale -> Drop(0.1)
      skip: Identity
    } -> Add

    # 2. Pre-RMSNorm + Asymmetric Twin-Branch SwiGLU FFN + DyT Modulation + Residual
    ffn: {
      branch: RMSNorm -> {
        gate: FC(1024) -> SoftCap(30.0) -> CosGate -> SiLU
        up  : FC(1024) -> ReLU2
      } -> Mul -> FC(512) -> DyT(0.5) -> Drop(0.1)
      skip: Identity
    } -> Add
  }

  head: RMSNorm -> FC(10) -> SoftmaxCE
```

### Example 2: ResNet-CNN with Mixed Anti-Aliased Pooling & Live GPU Augmentation
```yaml
input: 28x28x1
prng : philox

flow:
  # Fused 2D Coordinate & Pixel Augmentation (0 Shared Memory, auto-bypassed during eval!)
  aug: RandShift(2) -> RandRotate(10.0) -> RandScale(0.10) -> RandElastic(1.6, r=4) -> RandBrightness(0.1) -> RandContrast(0.1) -> RandNoise(0.03)

  c1: Conv(c=32, k=3, p=1) -> BN -> Mish -> MixPool(2, ratio=0.85)
  c2: Conv(c=64, k=3, p=1) -> BN -> HardSwish -> MixPool(2, ratio=0.85)

  res: {
    branch: Conv(c=64, k=3, p=1) -> BN(0.35) -> SiLU
    skip  : Identity
  } -> Add

  head: Flatten -> {
    gate: FC(512) -> SiLU
    up  : FC(512)
  } -> Mul -> LN -> PReLU(0.2) -> Drop(0.2) -> FC(10) -> SoftmaxCE
```

---

## 🛠️ Project Structure

```text
tkm/
├── include/
│   └── tkm.h             # Core IR, Unified ABI Signature table & Public C APIs
├── examples/
│   └── mnist_idx.c       # External C ABI caller using raw MNIST IDX files
├── docs/
│   └── C_API.md          # Host-side C ABI and MNIST IDX walkthrough
├── src/
│   ├── tkm_registry.c    # [Contributor Hub] Operator Specs & Fusion Rules
│   ├── tkm_core.c        # DSL Parser, Loop Macro Expander, Shape Inference & Arena Planner
│   ├── tkm_codegen.c     # Table-Driven Bi-Directional NVRTC CUDA C++ Generator
│   ├── tkm_gpu.cu        # WMMA FlashAttention, In-Memory NVRTC, CUDA Graph Engine
│   └── main.c            # Standalone CLI Entry (tkm.exe / tkm)
├── models/
│   ├── resnet_mnist.tkm  # ResNet-CNN markup
│   └── exotic_llama.tkm  # 4-Layer GQA + SwiGLU LLaMA markup
├── build.bat             # Windows MSVC Build Script (CLI, DLL, and MNIST ABI example)
├── Makefile              # Linux GCC Build Script (Outputs tkm & libtkm.so)
└── README.md
```

---

## ⚡ Building & Running

### Windows (MSVC + CUDA 11.6 ~ 13.x)
Open `x64 Native Tools Command Prompt for VS` and run:
```cmd
build.bat
```
This builds **`tkm.exe`** (CLI executable with an optional embedded icon), **`tkm.dll` / `tkm.lib`** (shared library and import library), **`tkm_static.lib`** (static SDK library), and C ABI examples linked against both library types.

### Linux (GCC + CUDA 11.6 ~ 13.x)
```bash
make -j
```
This builds the CLI, `libtkm.so`, and `libtkm.a`. Build the dynamic and static MNIST IDX callers with `make mnist_idx` and `make mnist_idx_static`, respectively.

---

## 🖥️ Command-Line Usage

### 1. Verify Operators via Fundamental Theorem of Calculus (`numcheck`)
Mathematically audits analytical CUDA gradients against numerical secant limits in GPU registers:
```bash
tkm.exe numcheck
```
```text
  [Calc-Check] SiLU           | Analytical:  +0.0052 | Numerical:  +0.0051 | RelErr: 7.72e-05 | PASS
  [Calc-Check] ReLU2          | Analytical:  +0.0009 | Numerical:  +0.0008 | RelErr: 4.63e-05 | PASS
  [Calc-Check] GELU           | Analytical:  +0.0048 | Numerical:  +0.0050 | RelErr: 1.58e-04 | PASS
  [Calc-Check] SoftCap        | Analytical:  +0.0086 | Numerical:  +0.0086 | RelErr: 7.63e-06 | PASS
  [Calc-Check] DyT            | Analytical:  +0.0068 | Numerical:  +0.0069 | RelErr: 1.39e-04 | PASS
  [Calc-Check] xATLU          | Analytical:  +0.0052 | Numerical:  +0.0051 | RelErr: 7.72e-05 | PASS
  >>> ALL OPERATORS MATHEMATICALLY VERIFIED AGAINST FINITE-DIFFERENCE CALCULUS! <<<
```

### 2. Multi-Architecture Hardware Benchmark Leaderboard (`bench`)
Runs built-in benchmark suite across 4 major architectures (ResNet, ConvNeXt, SIREN, LLaMA):
```bash
tkm.exe bench
```
```text
=========================================================================================
 Architecture                       | Batch | FusedK | Arena MB (Saved) | ms/step | Samples/s
------------------------------------+-------+--------+------------------+---------+----------
 ResNet-CNN + 7-Op Aug + SwiGLU     |   512 |     12 |  94.94 MB (48.3%) |   6.024 |   84995.5
 ConvNeXt-Vision + DyT + xATLU      |   256 |     12 |  95.83 MB (62.5%) |   6.896 |   37120.8
 SIREN-PINN Periodic Field Net      |  1024 |     14 |  12.31 MB (72.2%) |   0.992 | 1032665.8
 Exotic-LLaMA (4L GQA+QKNorm+RoPE)  |    32 |     50 | 520.47 MB (30.8%) |  38.136 |     839.1
=========================================================================================
```

### 3. End-to-End Training & Checkpointing Working Demo (`demo`)
Trains a custom hybrid CNN+Twin-Branch network on synthetic pattern data, converges loss, evaluates test accuracy, and verifies weight save/load:
```bash
tkm.exe demo
```

### 4. Inspect Custom Architecture & Export CUDA Source (`inspect`)
```bash
tkm.exe inspect models/exotic_llama.tkm --batch 32 --emit generated_llama.cu
```

---

## 🔌 Public C / C++ Library API

You can embed Tokamak into any host application using the exported API. The caller only includes `tkm.h` and passes ordinary host pointers; it does not need CUDA headers or CUDA types. `examples/mnist_idx.c` is a complete MNIST IDX caller that trains `models/resnet_mnist.tkm` and evaluates on the IDX test set.

Release downloads provide independent choices: the executable/runtime bundle (`tkm-windows-x64.zip` or `tkm-linux-x64.tar.gz`) for users who want the CLI and model files, and a static SDK bundle (`tkm-sdk-windows-x64.zip` or `tkm-sdk-linux-x64.tar.gz`) for embedding Tokamak into a host application. The SDK includes `include/tkm.h`, the platform static library, the minimal CUDA runtime/NVRTC/cuBLAS libraries, and the link libraries needed by the host compiler; it does not require the full CUDA Toolkit to build or run the host application. An NVIDIA driver is still required at runtime. See [docs/C_API.md](docs/C_API.md) for link commands and deployment details.

```c
#include "tkm.h"

int main() {
    TkmRuntime *rt = tkm_load_file("models/resnet_mnist.tkm", 64);
    if (!rt) return 1;
    float host_images[64 * 28 * 28] = {0};
    uint8_t host_labels[64] = {0};
    float loss = tkm_runtime_step(rt, host_images, host_labels, 1e-3f);
    (void)loss;
    tkm_save_weights(rt, "checkpoint.bin");
    tkm_runtime_free(rt);
}
```

Build the example caller with `build.bat` on Windows or `make mnist_idx` on Linux. Keep `tkm.dll`/`libtkm.so`, the model file, and CUDA runtime dependencies available when launching the caller. The IDX example accepts the standard uncompressed IDX image and label files; decompress the usual `.gz` downloads before running it:

```text
mnist_idx train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte [batch=64] [epochs=1]
```

With batch size 1024 and 30 epochs, one run reported train loss `0.1206` and test accuracy `9846/10000 (98.46%)`; this is an observed result, not a guarantee. Training uses complete fixed-size batches, so the example reports and skips any leftover training samples. See [docs/C_API.md](docs/C_API.md) for the full ABI walkthrough and command example.

`tkm_runtime_eval` accepts any positive sample count and excludes padded rows in a final partial batch. Plasma operators and `MixPool` use CUDA `half2` kernels and therefore require even channel counts.

---

## 🤝 How to Contribute an Operator

Supported pointwise `CAT_FLUID` activations and gating functions can be added by editing `src/tkm_registry.c` with scalar math formulas; the engine generates the CUDA kernel and derivative application. No CUDA code or new enum is needed. This path does not apply to reductions, standalone Softmax, or loss sinks, which need dedicated GPU runtime handling; see [docs/CONTRIBUTING_OP.md](docs/CONTRIBUTING_OP.md).

Simply add one entry to `TKM_OP_REGISTRY[]`:

```c
/* Example: Adding Softplus from math formulas (no CUDA source) */
{
    OP_USER_FLUID,
    "Softplus",                    // Keyword used in .tkm markup
    CAT_FLUID,
    RES_NONE,
    0.0f,
    NULL, NULL,                    // No handwritten CUDA
    "log(1 + exp(x))",             // Forward scalar formula
    "1 / (1 + exp(-x))"            // Derivative with respect to x
}
```
Expressions support `x`, `fval`, `pi`, arithmetic, parentheses, and the functions `exp`, `log`/`ln`, `sqrt`, `sin`, `cos`, `tanh`, `abs`, `max`, `min`, and `pow`. The derivative expression is multiplied by the incoming gradient automatically. Advanced operators that need parameters, reductions, or special memory behavior still require dedicated engine support; existing low-level CUDA snippets remain available for those cases. See [docs/CONTRIBUTING_OP.md](docs/CONTRIBUTING_OP.md) for the full authoring contract.

Research-derived example: `Snake(1.7)` is registered as a formula-only pointwise operator, so it uses generated CUDA code and remains eligible for fluid fusion. Its formula is $x+\sin^2(\alpha x)/\alpha$; `alpha` must be positive. The implementation credits [BigVGAN (Lee et al., 2023)](https://arxiv.org/abs/2206.04658).
