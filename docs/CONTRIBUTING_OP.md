# Contributor Guide: Adding Custom Operators to Tokamak

Tokamak uses a central operator registry, but the single-file path applies to
operators that fit the existing execution model. A new pointwise `CAT_FLUID`
operator can be registered in `src/tkm_registry.c` using scalar math formulas;
the engine generates the CUDA expressions and applies the derivative to the
incoming gradient.
Structural operators, reductions, normalizations, and terminal losses may need
new enum values and dedicated code generation or runtime kernels; the existing
Softmax and loss operators are examples of those runtime-specialized cases.

---

## Step 1: Add an enum only for runtime-specialized operators
For an operator that needs dedicated execution code, add its enum to `TkmOpType`
in `include/tkm.h`. Math-defined pointwise `CAT_FLUID` operators use the shared
`OP_USER_FLUID` value and do not need this step.
```c
typedef enum {
    ...
    OP_MY_ACTIVATION,   // <-- Add your enum here
    OP_COUNT
} TkmOpType;
```

---

## Step 2: Open `src/tkm_registry.c`
Register the operator in `TKM_OP_REGISTRY[]`. For a math-defined pointwise
operator, this is the only implementation file that needs to change.

### Case A: Adding a Pure Pointwise Activation Without CUDA
Example: Adding **Softplus**, with a scalar forward formula and its derivative:
```c
{
    OP_USER_FLUID, "Softplus", CAT_FLUID, RES_NONE, 0.0f,
    NULL, NULL,
    "log(1 + exp(x))",
    "1 / (1 + exp(-x))"
}
```

The expression variable `x` represents one scalar channel value. Supported
functions are `exp`, `log`/`ln`, `sqrt`, `sin`, `cos`, `tanh`, `abs`, `max`,
`min`, and `pow`; arithmetic, parentheses, `fval`, and `pi` are also available.
Write the scalar derivative `d output / d x`; the generated kernel multiplies
it by the back-propagated gradient. Formula registration avoids CUDA syntax but
still requires the contributor to provide the mathematically correct derivative.
For research-derived operators, credit the source paper in a short comment beside
the registry entry and in the operator documentation. For example, Snake as used
in BigVGAN (Sang-gil Lee et al., “[BigVGAN: A Universal Neural Vocoder with Large-Scale Training](https://arxiv.org/abs/2206.04658),” ICLR 2023)
has $f_\alpha(x)=x+\sin^2(\alpha x)/\alpha$ and
$f'_\alpha(x)=1+\sin(2\alpha x)$; use a positive `alpha` (`fval`).

### Case B: Advanced CUDA-Backed Parameters
Parameterized kernels with learnable weights can still use the lower-level
CUDA snippet interface. For example, learnable PReLU:
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
If the analytical backward derivative matches the 2-point secant slope limit $\frac{\mathcal{L}(x+\epsilon) - \mathcal{L}(x-\epsilon)}{2\epsilon}$, it will output `PASS` with relative error $< 10^{-4}$.

You can now submit your PR with confidence!
```