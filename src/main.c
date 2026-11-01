// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * src/main.c -- Tokamak CLI Driver with Numerical Finite-Difference Calculus GradCheck
 */
#include "tkm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double tkm_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double tkm_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

static TkmGraph g_shared_graph;

static char *load_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { perror("fread"); exit(1); }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static int is_numcheck_testable(TkmOpType op) {
    switch (op) {
        case OP_SILU: case OP_RELU: case OP_RELU2: case OP_LEAKY_RELU:
        case OP_ELU:  case OP_GELU: case OP_MISH:  case OP_HARDSWISH:
        case OP_HARDSIGMOID: case OP_TANH: case OP_SIGMOID: case OP_SWISH_BETA:
        case OP_SIN:  case OP_COSGATE: case OP_SOFTCAP: case OP_CLAMP:
        case OP_SCALE: case OP_LAYERSCALE: case OP_DYT: case OP_PRELU: case OP_XATLU:
            return 1;
        default:
            return 0;
    }
}

static int is_numcheck_testable_spec(const TkmOpSpec *spec) {
    if (!spec) return 0;
    if (spec->op == OP_USER_FLUID) return spec->fwd_expr && spec->bwd_expr;
    return is_numcheck_testable(spec->op);
}

static int run_fused_user_fluid_check(void) {
    static const char dsl[] =
        "input: 1x1x8\n"
        "flow:\n"
        "  trunk: FC(8) -> Softplus -> FC(8) -> Tanh\n"
        "  gate: {\n"
        "    branch: FC(8) -> Softplus -> Scale(0.7)\n"
        "    skip: FC(8) -> GELU\n"
        "  } -> Mul -> Softplus -> FC(4) -> SoftmaxCE\n";
    const int batch = 4;
    float input[32] = {
        0.4f, -0.2f, 0.8f, -0.5f, 0.3f, 0.7f, -0.1f, 0.6f,
        0.2f, 0.5f, -0.4f, 0.1f, 0.8f, -0.3f, 0.6f, -0.7f,
        -0.2f, 0.3f, 0.4f, 0.6f, -0.8f, 0.2f, 0.1f, 0.5f,
        0.7f, -0.5f, 0.2f, 0.4f, 0.3f, -0.1f, 0.8f, -0.4f
    };
    const uint8_t labels[4] = { 1, 0, 3, 2 };
    TkmGraph *g = &g_shared_graph;
    tkm_parse_dsl(g, dsl, batch);
    tkm_optimize_and_bind(g);

    int twin_fused = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->nodes[i].dead && g->nodes[i].op == OP_FC_TWIN_FUSED) twin_fused = 1;
    }
    if (!twin_fused) {
        fprintf(stderr, "  [Fused-Graph-Check] Custom pointwise op blocked twin-FC fusion.\n");
        return 1;
    }

    tkm_plan_memory(g);
    char *cuda_src = tkm_codegen_cuda_string(g);
    TkmRuntime *rt = tkm_runtime_ignite(g, cuda_src);
    free(cuda_src);
    if (!rt) return 1;

    int errors = 0;
    float loss = tkm_runtime_step(rt, input, labels, 0.0f);
    const float eps = 5e-2f;
    for (int i = 0; i < 8; i++) {
        float plus[32], minus[32];
        memcpy(plus, input, sizeof(input));
        memcpy(minus, input, sizeof(input));
        plus[i] += eps;
        minus[i] -= eps;
        float loss_plus = tkm_runtime_step(rt, plus, labels, 0.0f);
        float loss_minus = tkm_runtime_step(rt, minus, labels, 0.0f);
        tkm_runtime_step(rt, input, labels, 0.0f);
        float analytical = tkm_get_grad_sample(rt, 0, i);
        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        if (!isfinite(analytical) || !isfinite(numerical) ||
            fabsf(analytical - numerical) > 1e-2f) {
            printf("  [Fused-Graph-Check] grad[%d] analytical %+0.5f numerical %+0.5f FAIL\n",
                   i, analytical, numerical);
            errors++;
        }
    }

    if (errors == 0 && isfinite(loss)) {
        printf("  [Fused-Graph-Check] Twin-FC + custom Softplus | Loss: %.5f | Input gradients: PASS\n", loss);
    } else if (!isfinite(loss)) {
        fprintf(stderr, "  [Fused-Graph-Check] Non-finite training loss.\n");
        errors++;
    }
    tkm_runtime_free(rt);
    return errors;
}

/* Fallback mathematical scalar evaluation for headless CI runners without physical GPUs */
static float eval_scalar_fwd(TkmOpType op, float x, float w, float fval) {
    switch (op) {
        case OP_SILU:        return x / (1.0f + expf(-x));
        case OP_RELU:        return fmaxf(x, 0.0f);
        case OP_RELU2:       { float r = fmaxf(x, 0.0f); return r * r; }
        case OP_LEAKY_RELU:  return (x > 0.0f) ? x : (x * fval);
        case OP_ELU:         return (x > 0.0f) ? x : (fval * (expf(x) - 1.0f));
        case OP_GELU:        { float k = 0.79788456f * (x + 0.044715f * x * x * x); return 0.5f * x * (1.0f + tanhf(k)); }
        case OP_MISH:        { float sp = (x > 20.f) ? x : logf(1.f + expf(x)); return x * tanhf(sp); }
        case OP_HARDSWISH:   return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
        case OP_HARDSIGMOID: return fminf(fmaxf((x + 3.0f) / 6.0f, 0.0f), 1.0f);
        case OP_TANH:        return tanhf(x);
        case OP_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case OP_SWISH_BETA:  return x / (1.0f + expf(-fval * x));
        case OP_SIN:         return sinf(fval * x);
        case OP_COSGATE:     return x * cosf(x);
        case OP_SOFTCAP:     return fval * tanhf(x / fval);
        case OP_CLAMP:       return fminf(fmaxf(x, -fval), fval);
        case OP_SCALE:       return x * fval;
        case OP_LAYERSCALE:  return x * w;
        case OP_DYT:         return tanhf(fval * x) * w;
        case OP_PRELU:       return (x > 0.0f) ? x : (x * w * fval);
        case OP_XATLU:       { float s = 1.0f / (1.0f + expf(-x)); return x * (w * s + (1.0f - w)); }
        default:             return x;
    }
}

static float eval_scalar_bwd(TkmOpType op, float x, float w, float fval, float dy) {
    switch (op) {
        case OP_SILU:        { float s = 1.0f / (1.0f + expf(-x)); return dy * s * (1.0f + x * (1.0f - s)); }
        case OP_RELU:        return dy * (x > 0.0f ? 1.0f : 0.0f);
        case OP_RELU2:       return dy * (x > 0.0f ? 2.0f * x : 0.0f);
        case OP_LEAKY_RELU:  return dy * (x > 0.0f ? 1.0f : fval);
        case OP_ELU:         return dy * (x > 0.0f ? 1.0f : (fval * expf(x)));
        case OP_GELU:        { float k = 0.79788456f * (x + 0.044715f * x * x * x), t = tanhf(k);
                               return dy * (0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * 0.79788456f * (1.0f + 0.134145f * x * x)); }
        case OP_MISH:        { float sp = (x > 20.f) ? x : logf(1.f + expf(x)), t = tanhf(sp), s = 1.f / (1.f + expf(-x));
                               return dy * (t + x * (1.f - t * t) * s); }
        case OP_HARDSWISH:   return dy * ((x <= -3.0f) ? 0.0f : ((x >= 3.0f) ? 1.0f : ((2.0f * x + 3.0f) / 6.0f)));
        case OP_HARDSIGMOID: return dy * ((x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f);
        case OP_TANH:        { float t = tanhf(x); return dy * (1.0f - t * t); }
        case OP_SIGMOID:     { float s = 1.0f / (1.0f + expf(-x)); return dy * s * (1.0f - s); }
        case OP_SWISH_BETA:  { float s = 1.0f / (1.0f + expf(-fval * x)); return dy * s * (1.0f + fval * x * (1.0f - s)); }
        case OP_SIN:         return dy * fval * cosf(fval * x);
        case OP_COSGATE:     return dy * (cosf(x) - x * sinf(x));
        case OP_SOFTCAP:     { float t = tanhf(x / fval); return dy * (1.0f - t * t); }
        case OP_CLAMP:       return dy * ((x > -fval && x < fval) ? 1.0f : 0.0f);
        case OP_SCALE:       return dy * fval;
        case OP_LAYERSCALE:  return dy * w;
        case OP_DYT:         { float t = tanhf(fval * x); return dy * w * fval * (1.0f - t * t); }
        case OP_PRELU:       return dy * ((x > 0.0f) ? 1.0f : (w * fval));
        case OP_XATLU:       { float s = 1.0f / (1.0f + expf(-x)); return dy * ((1.0f - w) + w * s * (1.0f + x * (1.0f - s))); }
        default:             return dy;
    }
}

static int run_numerical_calculus_check(void) {
    printf("=========================================================================================\n");
    printf("     TOKAMAK FINITE-DIFFERENCE NUMERICAL GRADIENT VERIFICATION (FIRST PRINCIPLES)        \n");
    printf("=========================================================================================\n");

    int dev_cnt = 0;
    cudaGetDeviceCount(&dev_cnt);

    const float eps = 1e-3f;
    const float tol = 5e-3f;
    int failed_ops = 0;

    /* Mode A: Headless / No-GPU Fallback for GitHub Free CI Runner */
    if (dev_cnt <= 0) {
        printf("  [Hardware Notice] No physical CUDA device detected. Running Host Calculus Audit...\n");
        const float test_pts[4] = { -1.6f, -0.4f, 0.5f, 1.4f };
        for (int spec_id = 0; spec_id < tkm_get_spec_count(); spec_id++) {
            const TkmOpSpec *spec = tkm_find_spec_by_index(spec_id);
            if (!is_numcheck_testable_spec(spec)) continue;

            float fval = (spec->default_fval != 0.0f) ? spec->default_fval : 1.2f;
            float max_err = 0.0f;
            for (int p = 0; p < 4; p++) {
                float x = test_pts[p], w = 0.8f, dy = 1.3f;
                float ana_dx, num_dx;
                if (spec->op == OP_USER_FLUID) {
                    float deriv, y_plus, y_minus;
                    if (!tkm_eval_math_expr(spec->bwd_expr, x, fval, &deriv) ||
                        !tkm_eval_math_expr(spec->fwd_expr, x + eps, fval, &y_plus) ||
                        !tkm_eval_math_expr(spec->fwd_expr, x - eps, fval, &y_minus)) {
                        max_err = INFINITY;
                        break;
                    }
                    ana_dx = dy * deriv;
                    num_dx = dy * (y_plus - y_minus) / (2.0f * eps);
                } else {
                    ana_dx = eval_scalar_bwd(spec->op, x, w, fval, dy);
                    num_dx = dy * (eval_scalar_fwd(spec->op, x + eps, w, fval) -
                                   eval_scalar_fwd(spec->op, x - eps, w, fval)) / (2.0f * eps);
                }
                float err = fabsf(ana_dx - num_dx) / fmaxf(1.0f, fabsf(num_dx));
                if (err > max_err) max_err = err;
            }
            int pass = (max_err < tol);
            printf("  [Host-Check] %-14s | Max RelErr: %.2e | %s\n", spec->dsl_name, max_err, pass ? "PASS" : "FAIL!");
            if (!pass) failed_ops++;
        }
        printf("=========================================================================================\n");
        return failed_ops;
    }

    /* Mode B: Full NVRTC GPU Kernel Calculus Audit */
    const int B = 2;
    float   test_x[64];
    uint8_t test_y[2] = { 3, 7 };
    for (int i = 0; i < 64; i++) {
        test_x[i] = sinf((float)i * 0.4f) * 1.5f + 0.2f;
        if (fabsf(test_x[i]) < 0.1f) test_x[i] = 0.5f;
    }

    for (int spec_id = 0; spec_id < tkm_get_spec_count(); spec_id++) {
        const TkmOpSpec *spec = tkm_find_spec_by_index(spec_id);
        if (!is_numcheck_testable_spec(spec)) continue;

        char mini_dsl[256];
        if (strcmp(spec->dsl_name, "Snake") == 0) {
            snprintf(mini_dsl, sizeof(mini_dsl),
                     "input: 1x1x32\ntest: Snake(1.7) -> SoftmaxCE");
        } else {
            snprintf(mini_dsl, sizeof(mini_dsl),
                     "input: 1x1x32\ntest: %s -> SoftmaxCE", spec->dsl_name);
        }

        TkmRuntime *rt = tkm_load_string(mini_dsl, B);
        if (!rt) {
            fprintf(stderr, "  [FAIL] NVRTC could not compile test harness for op '%s'\n", spec->dsl_name);
            failed_ops++;
            continue;
        }

        tkm_runtime_step(rt, test_x, test_y, 0.0f);
        float ana_dx = tkm_get_grad_sample(rt, 0, 8);

        test_x[8] += eps;
        tkm_runtime_step(rt, test_x, test_y, 0.0f);
        float loss_p = tkm_eval_current_loss(rt);

        test_x[8] -= 2.0f * eps;
        tkm_runtime_step(rt, test_x, test_y, 0.0f);
        float loss_m = tkm_eval_current_loss(rt);

        test_x[8] += eps;

        float num_dx  = (loss_p - loss_m) / (2.0f * eps);
        float rel_err = fabsf(ana_dx - num_dx) / fmaxf(1.0f, fmaxf(fabsf(ana_dx), fabsf(num_dx)));

        int pass = (rel_err < tol);
        printf("  [GPU-Check]  %-14s | Analytical: %+8.4f | Numerical: %+8.4f | RelErr: %.2e | %s\n",
               spec->dsl_name, ana_dx, num_dx, rel_err, pass ? "PASS" : "FAIL!");

        if (!pass) failed_ops++;
        tkm_runtime_free(rt);
    }

    failed_ops += run_fused_user_fluid_check();

    printf("=========================================================================================\n");
    if (failed_ops == 0) {
        printf("  >>> ALL OPERATORS MATHEMATICALLY VERIFIED AGAINST FINITE-DIFFERENCE CALCULUS! <<<\n");
    } else {
        printf("  >>> DETECTED %d OPERATORS WITH MATHEMATICAL/GRADIENT FORMULA ERRORS! <<<\n", failed_ops);
    }
    return failed_ops;
}

typedef struct {
    const char *name;
    int         batch;
    int         steps;
    const char *dsl;
} BenchCase;

static const BenchCase BENCH_SUITE[] = {
    {
        "ResNet-CNN + 7-Op Aug + SwiGLU", 512, 200,
        "input: 28x28x1\n"
        "prng : philox\n"
        "flow:\n"
        "  aug: RandShift(2) -> RandRotate(10) -> RandScale(0.1) -> RandElastic(1.6, r=4) -> RandBrightness(0.1) -> RandContrast(0.1) -> RandNoise(0.03)\n"
        "  c1 : Conv(c=32, k=3, p=1) -> BN -> Mish -> MixPool(2, ratio=0.85)\n"
        "  c2 : Conv(c=64, k=3, p=1) -> BN -> HardSwish -> MixPool(2, ratio=0.85)\n"
        "  res: { b: Conv(c=64, k=3, p=1) -> BN -> SiLU, s: Identity } -> Add\n"
        "  hd : Flatten -> { g: FC(512) -> SiLU, u: FC(512) } -> Mul -> LN -> PReLU(0.2) -> Drop(0.2) -> FC(10) -> SoftmaxCE\n"
    },
    {
        "ConvNeXt-Vision + DyT + xATLU", 256, 200,
        "input: 28x28x1\n"
        "prng : pcg32\n"
        "flow:\n"
        "  stem: Conv(c=64, k=3, p=1) -> LN -> GELU -> MixPool(2)\n"
        "  stage x 3: {\n"
        "    blk: {\n"
        "      b: Conv(c=64, k=3, p=1) -> LN -> SwishBeta(1.5) -> LayerScale(1e-3) -> DyT(0.5) -> Drop(0.1)\n"
        "      s: Identity\n"
        "    } -> Add\n"
        "  }\n"
        "  head: MixPool(2) -> Flatten -> FC(512) -> L2Norm -> xATLU(0.5) -> FC(10) -> SoftmaxCE\n"
    },
    {
        "SIREN-PINN Periodic Field Net", 1024, 300,
        "input: 1x1x128\n"
        "flow:\n"
        "  in_proj: FC(512) -> Sin(3.0)\n"
        "  harmonics x 4: {\n"
        "    res: {\n"
        "      b: FC(512) -> LN -> Sin(1.5) -> CosGate -> Clamp(6.0) -> ELU -> LeakyReLU(0.02)\n"
        "      s: Identity\n"
        "    } -> Add\n"
        "  }\n"
        "  out: FC(512) -> L2Norm -> HardSigmoid -> Tanh -> FC(10) -> SoftmaxCE\n"
    },
    {
        "Exotic-LLaMA (4L GQA+QKNorm+RoPE)", 32, 100,
        "input: 1x256x512\n"
        "prng : pcg32\n"
        "flow:\n"
        "  layer x 4: {\n"
        "    attn: {\n"
        "      b: RMSNorm -> { q: FC(512)->QKNorm->RoPE, k: FC(128)->QKNorm->RoPE, v: FC(128) } -> GQA(q=8, kv=2, d=64) -> FC(512) -> SoftCap(50) -> LayerScale -> Drop(0.1)\n"
        "      s: Identity\n"
        "    } -> Add\n"
        "    ffn: {\n"
        "      b: RMSNorm -> { gate: FC(1024)->SoftCap(30)->CosGate->Mish, up: FC(1024)->ReLU2 } -> Mul -> FC(512) -> DyT(0.5) -> Drop(0.1)\n"
        "      s: Identity\n"
        "    } -> Add\n"
        "  }\n"
        "  head: RMSNorm -> FC(10) -> SoftmaxCE\n"
    }
};

static void run_benchmark_suite(void) {
    int n_cases = (int)(sizeof(BENCH_SUITE) / sizeof(BENCH_SUITE[0]));
    double latencies[8], throughputs[8], saved_pct[8], arena_mbs[8];
    int groups_cnt[8];

    printf("=========================================================================================\n");
    printf("                    TOKAMAK (tkm) MULTI-ARCHITECTURE BENCHMARK SUITE                     \n");
    printf("=========================================================================================\n");

    for (int i = 0; i < n_cases; i++) {
        const BenchCase *bc = &BENCH_SUITE[i];
        printf("\n>>> [%d/%d] Compiling & Igniting: %s (Batch=%d, Steps=%d)\n",
               i + 1, n_cases, bc->name, bc->batch, bc->steps);

        tkm_parse_dsl(&g_shared_graph, bc->dsl, bc->batch);
        tkm_optimize_and_bind(&g_shared_graph);
        tkm_plan_memory(&g_shared_graph);

        double naive_mb = g_shared_graph.naive_arena_bytes / (1024.0 * 1024.0);
        double peak_mb  = g_shared_graph.total_arena_bytes / (1024.0 * 1024.0);
        arena_mbs[i]    = peak_mb;
        saved_pct[i]    = 100.0 * (1.0 - peak_mb / (naive_mb > 0 ? naive_mb : 1.0));
        groups_cnt[i]   = g_shared_graph.num_groups;

        char *cuda_src = tkm_codegen_cuda_string(&g_shared_graph);
        TkmRuntime *rt = tkm_runtime_ignite(&g_shared_graph, cuda_src);

        tkm_runtime_step(rt, NULL, NULL, 1e-3f);

        double t0 = tkm_time_ms();
        for (int s = 0; s < bc->steps; s++) {
            tkm_runtime_step(rt, NULL, NULL, 1e-3f);
        }
        tkm_runtime_free(rt);
        free(cuda_src);

        double ms = tkm_time_ms() - t0;
        latencies[i]   = ms / bc->steps;
        throughputs[i] = (double)(bc->steps * bc->batch) / (ms / 1000.0);
    }

    printf("\n=========================================================================================\n");
    printf(" %-34s | Batch | FusedK | Arena MB (Saved) | ms/step | Samples/s\n", "Architecture");
    printf("------------------------------------+-------+--------+------------------+---------+----------\n");
    for (int i = 0; i < n_cases; i++) {
        printf(" %-34s | %5d | %6d | %6.2f MB (%4.1f%%) | %7.3f | %9.1f\n",
               BENCH_SUITE[i].name, BENCH_SUITE[i].batch, groups_cnt[i] * 2,
               arena_mbs[i], saved_pct[i], latencies[i], throughputs[i]);
    }
    printf("=========================================================================================\n");
}

static void run_working_demo(void) {
    const char *demo_dsl =
        "input: 28x28x1\n"
        "prng : pcg32\n"
        "flow:\n"
        "  aug  : RandShift(1) -> RandNoise(0.02)\n"
        "  conv1: Conv(c=32, k=3, p=1) -> LN -> Mish -> MixPool(2, ratio=0.85)\n"
        "  res2 : {\n"
        "    b: Conv(c=32, k=3, p=1) -> LN -> HardSwish -> DyT(0.5)\n"
        "    s: Identity\n"
        "  } -> Add -> MixPool(2, ratio=0.85)\n"
        "  head : Flatten -> {\n"
        "    g: FC(256) -> SiLU\n"
        "    u: FC(256) -> PReLU(0.25)\n"
        "  } -> Mul -> LN -> xATLU(0.5) -> Drop(0.1) -> FC(10) -> SoftmaxCE\n";

    const int B = 128;
    const int NUM_BATCHES = 8;
    const int TOTAL_SAMPLES = B * NUM_BATCHES;
    const int IMG_SZ = 28 * 28;

    printf("========================================================================\n");
    printf("   TOKAMAK WORKING EXAMPLE: End-to-End Pattern Classification Training  \n");
    printf("========================================================================\n");

    float   *h_x = (float*)malloc((size_t)TOTAL_SAMPLES * IMG_SZ * sizeof(float));
    uint8_t *h_y = (uint8_t*)malloc((size_t)TOTAL_SAMPLES * sizeof(uint8_t));

    srand(42);
    for (int n = 0; n < TOTAL_SAMPLES; n++) {
        int cls = n % 10;
        h_y[n] = (uint8_t)cls;
        float fx = (float)(cls % 5 + 1) * 0.35f;
        float fy = (float)(cls / 5 + 1) * 0.45f;
        float ph = (float)cls * 0.6f;
        for (int py = 0; py < 28; py++) {
            for (int px = 0; px < 28; px++) {
                float clean = 0.5f + 0.4f * sinf(px * fx + ph) * cosf(py * fy - ph);
                float noise = ((float)(rand() % 1000) / 1000.0f - 0.5f) * 0.15f;
                h_x[n * IMG_SZ + py * 28 + px] = fminf(fmaxf(clean + noise, 0.0f), 1.0f);
            }
        }
    }

    tkm_parse_dsl(&g_shared_graph, demo_dsl, B);
    tkm_optimize_and_bind(&g_shared_graph);
    tkm_plan_memory(&g_shared_graph);
    tkm_dump_summary(&g_shared_graph, stdout);

    char *cuda_src = tkm_codegen_cuda_string(&g_shared_graph);
    TkmRuntime *rt = tkm_runtime_ignite(&g_shared_graph, cuda_src);

    printf("\nEpoch |   LR    | Train Loss | Eval Accuracy (Aug & Drop Auto-Bypassed)\n");
    printf("------+---------+------------+-----------------------------------------\n");

    double t0 = tkm_time_ms();
    for (int epoch = 1; epoch <= 15; epoch++) {
        float lr = (epoch <= 3) ? (0.015f * epoch / 3.0f) : (0.015f * 0.5f * (1.0f + cosf(3.14159f * (epoch - 3) / 12.0f)));
        float epoch_loss = 0.0f;

        for (int b = 0; b < NUM_BATCHES; b++) {
            const float   *bx = h_x + (size_t)b * B * IMG_SZ;
            const uint8_t *by = h_y + (size_t)b * B;
            epoch_loss += tkm_runtime_step(rt, bx, by, lr);
        }
        epoch_loss /= (float)NUM_BATCHES;

        if (epoch == 1 || epoch % 3 == 0 || epoch == 15) {
            int corr = tkm_runtime_eval(rt, h_x, h_y, TOTAL_SAMPLES);
            printf("  %2d  | %.5f |   %.4f   | %4d / %4d (%6.2f%%)\n",
                   epoch, lr, epoch_loss, corr, TOTAL_SAMPLES, 100.0f * corr / TOTAL_SAMPLES);
        }
    }
    double ms = tkm_time_ms() - t0;
    printf("------+---------+------------+-----------------------------------------\n");
    printf("Training & Eval completed in %.2f ms!\n", ms);

    tkm_save_weights(rt, "demo_checkpoint.bin");
    tkm_load_weights(rt, "demo_checkpoint.bin");
    int final_corr = tkm_runtime_eval(rt, h_x, h_y, TOTAL_SAMPLES);
    printf("Verified Reloaded Checkpoint Eval Accuracy: %.2f%%\n", 100.0f * final_corr / TOTAL_SAMPLES);

    tkm_runtime_free(rt);
    free(cuda_src);
    free(h_x);
    free(h_y);
}

static void print_usage(const char *prog) {
    printf("Tokamak (tkm) -- Markup-to-CUDA Graph JIT Engine (Windows/Linux)\n\n");
    printf("Usage:\n");
    printf("  %s verify-registry\n", prog);
    printf("  %s numcheck\n", prog);
    printf("  %s bench\n", prog);
    printf("  %s demo\n", prog);
    printf("  %s inspect <model.tkm> [--batch N] [--emit out.cu]\n", prog);
    printf("  %s run     <model.tkm> [--batch N] [--steps S] [--lr LR] [--save w.bin] [--load w.bin]\n\n", prog);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "verify-registry")) {
        int errs = tkm_verify_registry();
        if (errs) {
            printf("[Registry Check] FAILED with %d validation errors.\n", errs);
            return 1;
        }
        printf("[Registry Check] PASSED: All %d operator specifications and fusion rules are valid.\n",
               tkm_get_spec_count());
        return 0;
    }

    if (argc >= 2 && !strcmp(argv[1], "numcheck")) {
        return run_numerical_calculus_check();
    }

    if (argc >= 2 && (!strcmp(argv[1], "bench") || !strcmp(argv[1], "bench-all"))) {
        run_benchmark_suite();
        return 0;
    }

    if (argc >= 2 && !strcmp(argv[1], "demo")) {
        run_working_demo();
        return 0;
    }

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode     = argv[1];
    const char *tkm_path = argv[2];
    const char *emit_cu  = NULL;
    const char *save_bin = NULL;
    const char *load_bin = NULL;
    int         batch_sz = 64;
    int         steps    = 100;
    float       lr       = 1e-3f;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch_sz = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit_cu = argv[++i];
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) save_bin = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) load_bin = argv[++i];
    }

    char *dsl_text = load_text_file(tkm_path);

    printf("=== [Tokamak JIT] Compiling '%s' (Batch=%d) ===\n", tkm_path, batch_sz);
    tkm_parse_dsl(&g_shared_graph, dsl_text, batch_sz);
    tkm_optimize_and_bind(&g_shared_graph);
    tkm_plan_memory(&g_shared_graph);
    tkm_dump_summary(&g_shared_graph, stdout);

    char *cuda_src = tkm_codegen_cuda_string(&g_shared_graph);
    if (emit_cu) {
        FILE *f = fopen(emit_cu, "w");
        if (f) {
            fputs(cuda_src, f);
            fclose(f);
            printf("  [Codegen] Emitted NVRTC CUDA C++ source to '%s'\n", emit_cu);
        }
    }

    if (!strcmp(mode, "run")) {
        TkmRuntime *rt = tkm_runtime_ignite(&g_shared_graph, cuda_src);
        if (load_bin) tkm_load_weights(rt, load_bin);

        tkm_runtime_step(rt, NULL, NULL, lr);

        printf("  [Execution] Running %d Fwd+Bwd+SGD steps via CUDA Graph...\n", steps);
        double t0 = tkm_time_ms();

        for (int s = 0; s < steps; s++) {
            tkm_runtime_step(rt, NULL, NULL, lr);
        }

        if (save_bin) tkm_save_weights(rt, save_bin);
        tkm_runtime_free(rt);

        double ms = tkm_time_ms() - t0;
        printf("  [Done] %d steps completed in %.2f ms (%.3f ms/step, %.1f samples/sec)\n",
               steps, ms, ms / steps, (double)(steps * batch_sz) / (ms / 1000.0));
    }

    free(cuda_src);
    free(dsl_text);
    return 0;
}