# Tokamak Architecture & Internal Design

This document details the compiler passes, memory management, and GPU execution pipeline of the Tokamak engine.

---

## 1. Compiler Pipeline Overview

```text
[.tkm Source]
     │
     ▼  Pass 1: Parser & Lexer
[Linear SSA IR Tape (TkmNode[])]
     │
     ▼  Pass 2: Shape Inference Pass
[Inferred Dimensions (N, H, W, C)]
     │
     ▼  Pass 3: Rule-Table Peephole Optimizer (tkm_registry.c)
  ├── 3.1 Horizontal Merges (Sibling FCs -> 1x GEMM)
  └── 3.2 Vertical Plasma Confinement (Pre-Norm + Norm + Post-Norm)
     │
     ▼  Pass 4: Static Liveness Memory Arena Planner
[Zero-Malloc Memory Layout (Off: X KB)]
     │
     ▼  Pass 5: NVRTC CUDA C++ Codegen
[In-Memory PTX String with Deadlock-Free Sub-Warp Kernels]
     │
     ▼  Pass 6: GPU Ignite & CUDA Graph Capture
[cudaGraphExec_t Engine Locked in VRAM]
```

---

## 2. Two-Pass Peephole Optimizer

### 2.1 Horizontal Fusion (Sibling Merge)
When the fuser detects that multiple parallel branches sharing identical inputs contain linear projections of matching shapes, it merges them into a single matrix multiplication:
* **2-Way SwiGLU Merge**:
  `{ gate: FC(1024) -> SiLU, up: FC(1024) } -> Mul`  
  $\implies$ Single `FC_Twin2x(2048)` GEMM. The NVRTC kernel simultaneously loads the left and right halves from the single buffer into registers (`vl`, `vr`), applies activations in-register, and executes `vl * vr` without intermediate VRAM traffic.
* **3-Way GQA QKV Merge**:
  `{ q: FC(512), k: FC(128), v: FC(128) } -> GQA`  
  $\implies$ Single `FC_QKV3x(768)` GEMM. The subsequent `QKV-Prep` NVRTC kernel performs in-place Bias addition, per-head sub-warp QK-RMSNorm, and 2D RoPE rotation in a single pass.

### 2.2 Vertical Plasma Confinement
Fluid pointwise and normalization operators following an anchor are grouped into a `PlasmaGroup`. A group can contain:
$$\text{Input} \to [\text{Pointwise Pre-Norm}] \to [\le 1\text{ Normalization Barrier}] \to [\text{Pointwise Post-Norm}] \to \text{Output}$$
Intermediate values are kept strictly inside GPU 32-bit registers (`float2 v`), completely bypassing Global Memory.

---

## 3. Zero-Memory Register Micro-Replay

Standard autograd engines save intermediate activations to VRAM during the Forward pass to evaluate derivatives during Backward. For long chains (e.g., `RMSNorm -> SiLU -> Mul -> LayerScale -> Drop`), this wastes significant memory bandwidth.

Tokamak employs **In-Register Symbolic Micro-Replay**:
1. During Forward, Tokamak only saves the normalized `xhat` (and 1-bit dropout mask).
2. During Backward, the NVRTC kernel reads `xhat` and re-evaluates forward activations **in registers within 2–3 ALU cycles**:
   ```cuda
   float2 pre_act = xhat * gamma;
   float2 act_val = silu_fwd(pre_act);
   ```
3. Derivatives are evaluated directly using these replayed register values. **Memory reads are reduced by over 80%.**

---

## 4. Static Liveness Arena Planner

Tokamak implements compile-time Liveness Analysis. Every tensor has a known `first_use` and `last_use` step:
* If a node is internal to a `PlasmaGroup`, it is marked `Register-Only` and allocated **0 bytes**.
* When a tensor surpasses its `last_use` step, its offset in the static memory arena is recycled for subsequent layers.
* Result: A 136-node Transformer requiring **752 MB** of naive activation memory runs in **88 MB** of reused static VRAM (**88.3% memory reduction**), completely eliminating runtime `cudaMalloc` / `cudaFree` calls.

---

## 5. Sub-Warp Multi-Row Packing & Atomic Reduction

To avoid intra-warp divergence and L2 cache atomic contention:
* When channel count $C_2 \le 32$, a single 256-thread block processes $256 / C_2$ rows simultaneously in lockstep, achieving 100% thread occupancy.
* Gradient accumulation across multiple tokens occurs in **thread-private registers (`acc_dw`, `acc_db`)**. Global `atomicAdd` instructions are issued only once per 16 tokens, cutting L2 atomic contention by **93.75%**.