# Contributor Guide: Adding Custom Operators to Tokamak

Tokamak is built on a **Single-File Registration Architecture**. To contribute a new operator, activation function, or fusion rule, **you only ever edit `src/tkm_registry.c`**. You do not need to modify the parser, memory planner, or GPU execution runtime.

---

## Step 1: Open `include/tkm.h`
Add your new operator enum to `TkmOpType`:
```c
typedef enum {
    ...
    OP_MY_ACTIVATION,   // <-- Add your enum here
    OP_COUNT
} TkmOpType;
```

---

## Step 2: Open `src/tkm_registry.c`
Add a single entry to the `TKM_OP_REGISTRY[]` table.

### Case A: Adding a Pure Pointwise Activation
Example: Adding **Swish-Beta** ($x \cdot \sigma(\beta x)$):
```c
{
    OP_SWISH_BETA,
    "SwishBeta",        // Keyword used in .tkm markup
    CAT_FLUID,          // Fusable into registers
    RES_NONE,           // Requires no extra parameters or buffers
    1.5f,               // Default hyperparameter fval (Beta)
    /* Forward CUDA */
    "float sx = 1.0f / (1.0f + __expf(-fval * v.x)), sy = 1.0f / (1.0f + __expf(-fval * v.y));\n"
    "v = make_float2(v.x * sx, v.y * sy);",
    /* Backward CUDA (Variable `pre` is automatically replayed in registers!) */
    "float sx = 1.0f / (1.0f + __expf(-fval * pre.x)), sy = 1.0f / (1.0f + __expf(-fval * pre.y));\n"
    "dy.x *= sx * (1.0f + fval * pre.x * (1.0f - sx));\n"
    "dy.y *= sy * (1.0f + fval * pre.y * (1.0f - sy));"
}
```

### Case B: Adding a Parameterized Layer with Learnable Weights
Example: Adding **Learnable PReLU** ($x > 0 \ ? \ x : w_c \cdot x$, with per-channel learnable $w_c$ vector):
```c
{
    OP_PRELU,
    "PReLU",
    CAT_FLUID,
    RES_PARAM_WEIGHT,   // Engine automatically allocates & binds w_vec and acc_dw!
    0.25f,              // Initial alpha value
    /* Forward CUDA (w_vec[tid] is automatically available in scope) */
    "float2 w = w_vec[tid];\n"
    "v.x = (v.x > 0.0f) ? v.x : (v.x * w.x * fval);\n"
    "v.y = (v.y > 0.0f) ? v.y : (v.y * w.y * fval);",
    /* Backward CUDA (Accumulates directly into local register acc_dw!) */
    "float2 w = w_vec[tid];\n"
    "if (pre.x <= 0.0f) acc_dw.x += dy.x * pre.x * fval;\n"
    "if (pre.y <= 0.0f) acc_dw.y += dy.y * pre.y * fval;\n"
    "dy.x *= (pre.x > 0.0f) ? 1.0f : (w.x * fval);\n"
    "dy.y *= (pre.y > 0.0f) ? 1.0f : (w.y * fval);"
}
```

---

## Step 3: Recompile and Verify Against Calculus First Principles
Rebuild Tokamak:
```cmd
build.bat
```
Run the automated finite-difference calculus verification suite:
```cmd
tkm.exe numcheck
```
If your analytical CUDA backward snippet mathematically matches the 2-point secant slope limit $\frac{\mathcal{L}(x+\epsilon) - \mathcal{L}(x-\epsilon)}{2\epsilon}$, it will output `PASS` with relative error $< 10^{-4}$.

You can now submit your PR with confidence!
```