// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * src/tkm_registry.c -- Complete Contributor Operator & Fusion Rule Registry
 *   Operator Specs + 4 Structural/Algebraic Peephole Fusion Rules
 */
#include "tkm.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>

const char *tkm_rng_name(TkmRngType r) {
    switch (r) {
        case RNG_XORSHIFT32:      return "xorshift32";
        case RNG_WANG_XORSHIFT32: return "wang_xorshift32";
        case RNG_PCG32:           return "pcg32";
        case RNG_PHILOX:          return "philox";
        default:                  return "wang_xorshift32";
    }
}

const char *tkm_rng_struct_name(TkmRngType r) {
    switch (r) {
        case RNG_XORSHIFT32:      return "RngXorshift32";
        case RNG_WANG_XORSHIFT32: return "RngWangXorshift32";
        case RNG_PCG32:           return "RngPcg32";
        case RNG_PHILOX:          return "RngPhilox";
        default:                  return "RngWangXorshift32";
    }
}

TkmRngType tkm_parse_rng_name(const char *s) {
    if (!strcmp(s, "xorshift32") || !strcmp(s, "xorshift")) return RNG_XORSHIFT32;
    if (!strcmp(s, "wang_xorshift32") || !strcmp(s, "wang")) return RNG_WANG_XORSHIFT32;
    if (!strcmp(s, "pcg32") || !strcmp(s, "pcg"))           return RNG_PCG32;
    if (!strcmp(s, "philox") || !strcmp(s, "philox4x32"))   return RNG_PHILOX;
    return RNG_WANG_XORSHIFT32;
}

static const TkmOpSpec TKM_OP_REGISTRY[] = {
    /* --- Graph Boundary Meta Nodes --- */
    { OP_NONE,          "None",       CAT_META,    RES_NONE, 0.0f,  NULL, NULL },
    { OP_INPUT,         "Input",      CAT_META,    RES_NONE, 0.0f,  NULL, NULL },

    /* --- Anchors & Spatial --- */
    { OP_CONV,          "Conv",       CAT_ANCHOR,  RES_NONE, 0.0f,  NULL, NULL },
    { OP_FC,            "FC",         CAT_ANCHOR,  RES_NONE, 0.0f,  NULL, NULL },
    { OP_FC_TWIN_FUSED, "FC_Twin2x",  CAT_ANCHOR,  RES_NONE, 0.0f,  NULL, NULL },
    { OP_FC_QKV_FUSED,  "FC_QKV3x",   CAT_ANCHOR,  RES_NONE, 0.0f,  NULL, NULL },
    { OP_GQA_ATTN,      "GQA",        CAT_ANCHOR,  RES_NONE, 0.0f,  NULL, NULL },
    { OP_MIXPOOL,       "MixPool",    CAT_SPATIAL, RES_NONE, 0.85f, NULL, NULL },
    { OP_AVGPOOL,       "AvgPool",    CAT_SPATIAL, RES_NONE, 0.0f,  NULL, NULL },
    { OP_FLATTEN,       "Flatten",    CAT_SPATIAL, RES_NONE, 0.0f,  NULL, NULL },

    /* --- Resolution-Independent 2D Coordinate & Pixel Bricks (CAT_COORD, 0 Shared Memory!) --- */
    {
        OP_RAND_SHIFT, "RandShift", CAT_COORD, RES_RNG_SEED, 2.0f,
        "int span = 2 * (int)fval + 1;\n"
        "float dx = (float)((int)(rng_u32(&img_rng) % span) - (int)fval);\n"
        "float dy = (float)((int)(rng_u32(&img_rng) % span) - (int)fval);\n"
        "fx -= dx; fy -= dy;",
        NULL
    },
    {
        OP_RAND_ROTATE, "RandRotate", CAT_COORD, RES_RNG_SEED, 10.0f,
        "float ang = (rng_u32(&img_rng) & 1u) ? ((rng_uniform(&img_rng) * 2.0f - 1.0f) * (fval * 0.01745329f)) : 0.0f;\n"
        "float sinA, cosA; __sincosf(ang, &sinA, &cosA);\n"
        "float rx = fx - cx, ry = fy - cy;\n"
        "fx =  cosA * rx + sinA * ry + cx;\n"
        "fy = -sinA * rx + cosA * ry + cy;",
        NULL
    },
    {
        OP_RAND_SCALE, "RandScale", CAT_COORD, RES_RNG_SEED, 0.10f,
        "float sc = (rng_u32(&img_rng) & 1u) ? (1.0f / (1.0f + (rng_uniform(&img_rng) * 2.0f - 1.0f) * fval)) : 1.0f;\n"
        "fx = (fx - cx) * sc + cx;\n"
        "fy = (fy - cy) * sc + cy;",
        NULL
    },
    {
        OP_RAND_FLIPX, "RandFlipX", CAT_COORD, RES_RNG_SEED, 0.5f,
        "if (rng_uniform(&img_rng) < fval) fx = (2.0f * cx) - fx;",
        NULL
    },
    {
        OP_RAND_FLIPY, "RandFlipY", CAT_COORD, RES_RNG_SEED, 0.5f,
        "if (rng_uniform(&img_rng) < fval) fy = (2.0f * cy) - fy;",
        NULL
    },
    {
        OP_RAND_ELASTIC, "RandElastic", CAT_COORD, RES_RNG_SEED, 1.6f,
        "if (rng_u32(&img_rng) & 1u) {\n"
        "    float freq1 = 3.14159265f / fmaxf(fval2 * 2.0f, 1.0f);\n"
        "    float freq2 = freq1 * 1.73205f;\n"
        "    float p0 = rng_uniform(&img_rng) * 6.283185f, p1 = rng_uniform(&img_rng) * 6.283185f;\n"
        "    float p2 = rng_uniform(&img_rng) * 6.283185f, p3 = rng_uniform(&img_rng) * 6.283185f;\n"
        "    float p4 = rng_uniform(&img_rng) * 6.283185f, p5 = rng_uniform(&img_rng) * 6.283185f;\n"
        "    float ex = 0.7f * __sinf(fx * freq1 + p0) * __cosf(fy * freq1 + p1) + 0.3f * __sinf(fx * freq2 + p4);\n"
        "    float ey = 0.7f * __cosf(fx * freq1 + p2) * __sinf(fy * freq1 + p3) + 0.3f * __cosf(fy * freq2 + p5);\n"
        "    fx += fval * ex; fy += fval * ey;\n"
        "}",
        NULL
    },
    {
        OP_RAND_NOISE, "RandNoise", CAT_COORD, RES_RNG_SEED, 0.03f,
        "if (rng_u32(&img_rng) & 1u) {\n"
        "    float2 n2 = rng_normal2(&pix_rng);\n"
        "    val = fminf(fmaxf(fmaf(n2.x, fval, val), 0.0f), 1.0f);\n"
        "}",
        NULL
    },
    {
        OP_RAND_CUTOUT, "RandCutout", CAT_COORD, RES_RNG_SEED, 4.0f,
        "int box_r = (int)(fval * 0.5f);\n"
        "int cut_x = (int)(rng_u32(&img_rng) % W), cut_y = (int)(rng_u32(&img_rng) % H);\n"
        "if (abs(px - cut_x) <= box_r && abs(py - cut_y) <= box_r) val = 0.0f;",
        NULL
    },
    {
        OP_RAND_BRIGHTNESS, "RandBrightness", CAT_COORD, RES_RNG_SEED, 0.15f,
        "float db = (rng_uniform(&img_rng) * 2.0f - 1.0f) * fval;\n"
        "val = fminf(fmaxf(val + db, 0.0f), 1.0f);",
        NULL
    },
    {
        OP_RAND_CONTRAST, "RandContrast", CAT_COORD, RES_RNG_SEED, 0.20f,
        "float cf = 1.0f + (rng_uniform(&img_rng) * 2.0f - 1.0f) * fval;\n"
        "val = fminf(fmaxf((val - 0.5f) * cf + 0.5f, 0.0f), 1.0f);",
        NULL
    },

    /* --- Normalization Barriers --- */
    {
        OP_BN, "BN", CAT_NORM,
        RES_PARAM_WEIGHT | RES_PARAM_BETA | RES_SAVE_ISTD | RES_SAVE_XHAT, 1e-2f,
        "float m = row_reduce_sum(v.x + v.y) * inv_C;\n"
        "float vr = fmaxf(0.f, row_reduce_sum(v.x*v.x + v.y*v.y) * inv_C - m * m);\n"
        "float istd = rsqrtf(vr + fval);\n"
        "if (tid == 0) save_istd[row] = istd;\n"
        "float2 xh = make_float2((v.x - m) * istd, (v.y - m) * istd);\n"
        "save_xhat[idx2] = __float22half2_rn(xh);\n"
        "float2 gm = w_vec[tid], bt = beta_vec[tid];\n"
        "v = make_float2(fmaf(gm.x, xh.x, bt.x), fmaf(gm.y, xh.y, bt.y));",
        "float2 xh = __half22float2(__ldg(&save_xhat[idx2]));\n"
        "float2 gm = w_vec[tid];\n"
        "acc_dw.x += dy.x * xh.x; acc_dw.y += dy.y * xh.y;\n"
        "acc_dbeta.x += dy.x;     acc_dbeta.y += dy.y;\n"
        "float2 dxh = make_float2(dy.x * gm.x, dy.y * gm.y);\n"
        "float m1 = row_reduce_sum(dxh.x + dxh.y) * inv_C;\n"
        "float m2 = row_reduce_sum(dxh.x*xh.x + dxh.y*xh.y) * inv_C;\n"
        "float istd = save_istd[row];\n"
        "dy = make_float2(istd * (dxh.x - m1 - xh.x * m2), istd * (dxh.y - m1 - xh.y * m2));"
    },
    {
        OP_LN, "LN", CAT_NORM,
        RES_PARAM_WEIGHT | RES_PARAM_BETA | RES_SAVE_ISTD | RES_SAVE_XHAT, 1e-5f,
        "float m = row_reduce_sum(v.x + v.y) * inv_C;\n"
        "float vr = fmaxf(0.f, row_reduce_sum(v.x*v.x + v.y*v.y) * inv_C - m * m);\n"
        "float istd = rsqrtf(vr + fval);\n"
        "if (tid == 0) save_istd[row] = istd;\n"
        "float2 xh = make_float2((v.x - m) * istd, (v.y - m) * istd);\n"
        "save_xhat[idx2] = __float22half2_rn(xh);\n"
        "float2 gm = w_vec[tid], bt = beta_vec[tid];\n"
        "v = make_float2(fmaf(gm.x, xh.x, bt.x), fmaf(gm.y, xh.y, bt.y));",
        "float2 xh = __half22float2(__ldg(&save_xhat[idx2]));\n"
        "float2 gm = w_vec[tid];\n"
        "acc_dw.x += dy.x * xh.x; acc_dw.y += dy.y * xh.y;\n"
        "acc_dbeta.x += dy.x;     acc_dbeta.y += dy.y;\n"
        "float2 dxh = make_float2(dy.x * gm.x, dy.y * gm.y);\n"
        "float m1 = row_reduce_sum(dxh.x + dxh.y) * inv_C;\n"
        "float m2 = row_reduce_sum(dxh.x*xh.x + dxh.y*xh.y) * inv_C;\n"
        "float istd = save_istd[row];\n"
        "dy = make_float2(istd * (dxh.x - m1 - xh.x * m2), istd * (dxh.y - m1 - xh.y * m2));"
    },
    {
        OP_RMSNORM, "RMSNorm", CAT_NORM,
        RES_PARAM_WEIGHT | RES_SAVE_ISTD | RES_SAVE_XHAT, 1e-5f,
        "float sq = row_reduce_sum(v.x * v.x + v.y * v.y);\n"
        "float istd = rsqrtf(sq * inv_C + fval);\n"
        "if (tid == 0) save_istd[row] = istd;\n"
        "float2 xh = make_float2(v.x * istd, v.y * istd);\n"
        "save_xhat[idx2] = __float22half2_rn(xh);\n"
        "float2 gm = w_vec[tid];\n"
        "v = make_float2(xh.x * gm.x, xh.y * gm.y);",
        "float2 xh = __half22float2(__ldg(&save_xhat[idx2]));\n"
        "float2 gm = w_vec[tid];\n"
        "acc_dw.x += dy.x * xh.x; acc_dw.y += dy.y * xh.y;\n"
        "float2 dxh = make_float2(dy.x * gm.x, dy.y * gm.y);\n"
        "float m2 = row_reduce_sum(dxh.x * xh.x + dxh.y * xh.y) * inv_C;\n"
        "float istd = save_istd[row];\n"
        "dy = make_float2(istd * (dxh.x - xh.x * m2), istd * (dxh.y - xh.y * m2));"
    },
    {
        OP_L2NORM, "L2Norm", CAT_NORM,
        RES_SAVE_ISTD | RES_SAVE_XHAT, 1e-5f,
        "float sq = row_reduce_sum(v.x * v.x + v.y * v.y);\n"
        "float istd = rsqrtf(sq * inv_C + fval);\n"
        "if (tid == 0) save_istd[row] = istd;\n"
        "float2 xh = make_float2(v.x * istd, v.y * istd);\n"
        "save_xhat[idx2] = __float22half2_rn(xh);\n"
        "v = xh;",
        "float2 xh = __half22float2(__ldg(&save_xhat[idx2]));\n"
        "float m2 = row_reduce_sum(dy.x * xh.x + dy.y * xh.y) * inv_C;\n"
        "float istd = save_istd[row];\n"
        "dy = make_float2(istd * (dy.x - xh.x * m2), istd * (dy.y - xh.y * m2));"
    },
    {
        OP_QKNORM, "QKNorm", CAT_FLUID,
        RES_PARAM_BIAS | RES_PARAM_WEIGHT | RES_SAVE_ISTD | RES_SAVE_XHAT, 1e-5f,
        NULL, NULL
    },

    /* --- Fluid Pointwise & Learnable/Stateful Operators --- */
    {
        OP_BIAS, "Bias", CAT_FLUID, RES_PARAM_BIAS, 0.0f,
        "float2 b = b_vec[tid]; v.x += b.x; v.y += b.y;",
        "acc_db.x += dy.x; acc_db.y += dy.y;"
    },
    {
        OP_SILU, "SiLU", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = v.x / (1.0f + __expf(-v.x)); v.y = v.y / (1.0f + __expf(-v.y));",
        "float sx = 1.0f / (1.0f + __expf(-pre.x)), sy = 1.0f / (1.0f + __expf(-pre.y));\n"
        "dy.x *= sx * (1.0f + pre.x * (1.0f - sx));\n"
        "dy.y *= sy * (1.0f + pre.y * (1.0f - sy));"
    },
    {
        OP_RELU, "ReLU", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = fmaxf(v.x, 0.0f); v.y = fmaxf(v.y, 0.0f);",
        "dy.x *= (pre.x > 0.0f ? 1.0f : 0.0f); dy.y *= (pre.y > 0.0f ? 1.0f : 0.0f);"
    },
    {
        OP_RELU2, "ReLU2", CAT_FLUID, RES_NONE, 0.0f,
        "float rx = fmaxf(v.x, 0.0f), ry = fmaxf(v.y, 0.0f); v = make_float2(rx*rx, ry*ry);",
        "dy.x *= (pre.x > 0.0f ? 2.0f * pre.x : 0.0f); dy.y *= (pre.y > 0.0f ? 2.0f * pre.y : 0.0f);"
    },
    {
        OP_LEAKY_RELU, "LeakyReLU", CAT_FLUID, RES_NONE, 0.01f,
        "v.x = (v.x > 0.0f) ? v.x : (v.x * fval);\n"
        "v.y = (v.y > 0.0f) ? v.y : (v.y * fval);",
        "dy.x *= (pre.x > 0.0f ? 1.0f : fval);\n"
        "dy.y *= (pre.y > 0.0f ? 1.0f : fval);"
    },
    {
        OP_ELU, "ELU", CAT_FLUID, RES_NONE, 1.0f,
        "v.x = (v.x > 0.0f) ? v.x : (fval * (__expf(v.x) - 1.0f));\n"
        "v.y = (v.y > 0.0f) ? v.y : (fval * (__expf(v.y) - 1.0f));",
        "dy.x *= (pre.x > 0.0f ? 1.0f : (fval * __expf(pre.x)));\n"
        "dy.y *= (pre.y > 0.0f ? 1.0f : (fval * __expf(pre.y)));"
    },
    {
        OP_GELU, "GELU", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = 0.5f * v.x * (1.0f + tanhf(0.79788456f * (v.x + 0.044715f * v.x * v.x * v.x)));\n"
        "v.y = 0.5f * v.y * (1.0f + tanhf(0.79788456f * (v.y + 0.044715f * v.y * v.y * v.y)));",
        "float kx = 0.79788456f * (pre.x + 0.044715f * pre.x * pre.x * pre.x), tx = tanhf(kx);\n"
        "float ky = 0.79788456f * (pre.y + 0.044715f * pre.y * pre.y * pre.y), ty = tanhf(ky);\n"
        "dy.x *= 0.5f * (1.0f + tx) + 0.5f * pre.x * (1.0f - tx * tx) * 0.79788456f * (1.0f + 0.134145f * pre.x * pre.x);\n"
        "dy.y *= 0.5f * (1.0f + ty) + 0.5f * pre.y * (1.0f - ty * ty) * 0.79788456f * (1.0f + 0.134145f * pre.y * pre.y);"
    },
    {
        OP_MISH, "Mish", CAT_FLUID, RES_NONE, 0.0f,
        "float spx = (v.x > 20.f) ? v.x : __logf(1.f + __expf(v.x));\n"
        "float spy = (v.y > 20.f) ? v.y : __logf(1.f + __expf(v.y));\n"
        "v.x = v.x * tanhf(spx); v.y = v.y * tanhf(spy);",
        "float spx = (pre.x > 20.f) ? pre.x : __logf(1.f + __expf(pre.x)), tx = tanhf(spx), sigx = 1.f / (1.f + __expf(-pre.x));\n"
        "float spy = (pre.y > 20.f) ? pre.y : __logf(1.f + __expf(pre.y)), ty = tanhf(spy), sigy = 1.f / (1.f + __expf(-pre.y));\n"
        "dy.x *= tx + pre.x * (1.f - tx * tx) * sigx;\n"
        "dy.y *= ty + pre.y * (1.f - ty * ty) * sigy;"
    },
    {
        OP_HARDSWISH, "HardSwish", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = v.x * fminf(fmaxf(v.x + 3.0f, 0.0f), 6.0f) * 0.16666667f;\n"
        "v.y = v.y * fminf(fmaxf(v.y + 3.0f, 0.0f), 6.0f) * 0.16666667f;",
        "dy.x *= (pre.x <= -3.0f) ? 0.0f : ((pre.x >= 3.0f) ? 1.0f : ((2.0f * pre.x + 3.0f) * 0.16666667f));\n"
        "dy.y *= (pre.y <= -3.0f) ? 0.0f : ((pre.y >= 3.0f) ? 1.0f : ((2.0f * pre.y + 3.0f) * 0.16666667f));"
    },
    {
        OP_HARDSIGMOID, "HardSigmoid", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = fminf(fmaxf((v.x + 3.0f) * 0.16666667f, 0.0f), 1.0f);\n"
        "v.y = fminf(fmaxf((v.y + 3.0f) * 0.16666667f, 0.0f), 1.0f);",
        "dy.x *= (pre.x > -3.0f && pre.x < 3.0f) ? 0.16666667f : 0.0f;\n"
        "dy.y *= (pre.y > -3.0f && pre.y < 3.0f) ? 0.16666667f : 0.0f;"
    },
    {
        OP_TANH, "Tanh", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = tanhf(v.x); v.y = tanhf(v.y);",
        "float tx = tanhf(pre.x), ty = tanhf(pre.y);\n"
        "dy.x *= (1.0f - tx * tx); dy.y *= (1.0f - ty * ty);"
    },
    {
        OP_SIGMOID, "Sigmoid", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = 1.0f / (1.0f + __expf(-v.x)); v.y = 1.0f / (1.0f + __expf(-v.y));",
        "float sx = 1.0f / (1.0f + __expf(-pre.x)), sy = 1.0f / (1.0f + __expf(-pre.y));\n"
        "dy.x *= sx * (1.0f - sx); dy.y *= sy * (1.0f - sy);"
    },
    {
        OP_SWISH_BETA, "SwishBeta", CAT_FLUID, RES_NONE, 1.5f,
        "float sx = 1.0f / (1.0f + __expf(-fval * v.x)), sy = 1.0f / (1.0f + __expf(-fval * v.y));\n"
        "v = make_float2(v.x * sx, v.y * sy);",
        "float sx = 1.0f / (1.0f + __expf(-fval * pre.x)), sy = 1.0f / (1.0f + __expf(-fval * pre.y));\n"
        "dy.x *= sx * (1.0f + fval * pre.x * (1.0f - sx));\n"
        "dy.y *= sy * (1.0f + fval * pre.y * (1.0f - sy));"
    },
    {
        OP_SIN, "Sin", CAT_FLUID, RES_NONE, 1.0f,
        "v.x = __sinf(fval * v.x); v.y = __sinf(fval * v.y);",
        "dy.x *= fval * __cosf(fval * pre.x); dy.y *= fval * __cosf(fval * pre.y);"
    },
    {
        OP_COSGATE, "CosGate", CAT_FLUID, RES_NONE, 0.0f,
        "v.x = v.x * __cosf(v.x); v.y = v.y * __cosf(v.y);",
        "float sx, cx, sy, cy; __sincosf(pre.x, &sx, &cx); __sincosf(pre.y, &sy, &cy);\n"
        "dy.x *= (cx - pre.x * sx); dy.y *= (cy - pre.y * sy);"
    },
    {
        OP_SOFTCAP, "SoftCap", CAT_FLUID, RES_NONE, 30.0f,
        "v.x = fval * tanhf(v.x / fval); v.y = fval * tanhf(v.y / fval);",
        "float tx = tanhf(pre.x / fval), ty = tanhf(pre.y / fval);\n"
        "dy.x *= (1.0f - tx * tx); dy.y *= (1.0f - ty * ty);"
    },
    {
        OP_CLAMP, "Clamp", CAT_FLUID, RES_NONE, 6.0f,
        "v.x = fminf(fmaxf(v.x, -fval), fval); v.y = fminf(fmaxf(v.y, -fval), fval);",
        "dy.x *= (pre.x > -fval && pre.x < fval) ? 1.0f : 0.0f;\n"
        "dy.y *= (pre.y > -fval && pre.y < fval) ? 1.0f : 0.0f;"
    },
    {
        OP_SCALE, "Scale", CAT_FLUID, RES_NONE, 1.0f,
        "v.x *= fval; v.y *= fval;",
        "dy.x *= fval; dy.y *= fval;"
    },
    {
        OP_LAYERSCALE, "LayerScale", CAT_FLUID, RES_PARAM_WEIGHT, 1e-4f,
        "float2 w = w_vec[tid]; v.x *= w.x; v.y *= w.y;",
        "float2 w = w_vec[tid];\n"
        "acc_dw.x += dy.x * pre.x; acc_dw.y += dy.y * pre.y;\n"
        "dy.x *= w.x; dy.y *= w.y;"
    },
    {
        OP_DYT, "DyT", CAT_FLUID, RES_PARAM_WEIGHT, 0.5f,
        "float2 w = w_vec[tid];\n"
        "v.x = tanhf(fval * v.x) * w.x; v.y = tanhf(fval * v.y) * w.y;",
        "float2 w = w_vec[tid];\n"
        "float tx = tanhf(fval * pre.x), ty = tanhf(fval * pre.y);\n"
        "acc_dw.x += dy.x * tx; acc_dw.y += dy.y * ty;\n"
        "dy.x *= w.x * fval * (1.0f - tx * tx);\n"
        "dy.y *= w.y * fval * (1.0f - ty * ty);"
    },
    {
        OP_PRELU, "PReLU", CAT_FLUID, RES_PARAM_WEIGHT, 0.25f,
        "float2 w = w_vec[tid];\n"
        "v.x = (v.x > 0.0f) ? v.x : (v.x * w.x * fval);\n"
        "v.y = (v.y > 0.0f) ? v.y : (v.y * w.y * fval);",
        "float2 w = w_vec[tid];\n"
        "if (pre.x <= 0.0f) acc_dw.x += dy.x * pre.x * fval;\n"
        "if (pre.y <= 0.0f) acc_dw.y += dy.y * pre.y * fval;\n"
        "dy.x *= (pre.x > 0.0f) ? 1.0f : (w.x * fval);\n"
        "dy.y *= (pre.y > 0.0f) ? 1.0f : (w.y * fval);"
    },
    {
        OP_XATLU, "xATLU", CAT_FLUID, RES_PARAM_WEIGHT, 0.5f,
        "float2 w = w_vec[tid];\n"
        "float sx = 1.0f / (1.0f + __expf(-v.x)), sy = 1.0f / (1.0f + __expf(-v.y));\n"
        "v.x = v.x * (w.x * sx + (1.0f - w.x));\n"
        "v.y = v.y * (w.y * sy + (1.0f - w.y));",
        "float2 w = w_vec[tid];\n"
        "float sx = 1.0f / (1.0f + __expf(-pre.x)), sy = 1.0f / (1.0f + __expf(-pre.y));\n"
        "acc_dw.x += dy.x * pre.x * (sx - 1.0f);\n"
        "acc_dw.y += dy.y * pre.y * (sy - 1.0f);\n"
        "dy.x *= (1.0f - w.x) + w.x * sx * (1.0f + pre.x * (1.0f - sx));\n"
        "dy.y *= (1.0f - w.y) + w.y * sy * (1.0f + pre.y * (1.0f - sy));"
    },
    { OP_ROPE, "RoPE", CAT_FLUID, RES_NONE, 0.0f, NULL, NULL },
    {
        OP_DROP, "Drop", CAT_FLUID, RES_SAVE_MASK | RES_RNG_SEED, 0.1f,
        "float kp = 1.0f - fval, ikp = 1.0f / kp;\n"
        "uint8_t k0 = rng_uniform(&rng) < kp, k1 = rng_uniform(&rng) < kp;\n"
        "save_mask[idx2] = k0 | (k1 << 8);\n"
        "v.x = k0 ? v.x * ikp : 0.0f; v.y = k1 ? v.y * ikp : 0.0f;",
        "uint16_t m = save_mask[idx2]; float ikp = 1.0f / (1.0f - fval);\n"
        "dy.x *= (m & 0xFF) ? ikp : 0.0f; dy.y *= (m >> 8) ? ikp : 0.0f;"
    },
    {
        OP_ADD, "Add", CAT_FLUID, RES_BRANCH_IN, 0.0f,
        "float2 sk = __half22float2(__ldg(&branch_in[idx2])); v.x += sk.x; v.y += sk.y;",
        "dbranch_out[idx2] = __float22half2_rn(dy);"
    },
    { OP_MUL,            "Mul",         CAT_FLUID,  RES_NONE, 0.0f, NULL, NULL },
    { OP_SOFTMAX,         "Softmax",     CAT_ANCHOR, RES_NONE, 0.0f, NULL, NULL },
    { OP_SOFTMAX_CE,      "SoftmaxCE",   CAT_ANCHOR, RES_NONE, 0.0f, NULL, NULL },
    { OP_SOFTMAX_FOCAL,   "SoftmaxFocal",CAT_ANCHOR, RES_NONE, 2.0f, NULL, NULL },
    { OP_SOFTMAX_MSE,     "SoftmaxMSE",  CAT_ANCHOR, RES_NONE, 0.0f, NULL, NULL },
    {
        OP_USER_FLUID, "Softplus", CAT_FLUID, RES_NONE, 0.0f, NULL, NULL,
        "log(1.0 + exp(x))",
        "1.0 / (1.0 + exp(-x))"
    },
    /* Snake activation as used in BigVGAN (Lee et al., ICLR 2023; arXiv:2206.04658). */
    {
        OP_USER_FLUID, "Snake", CAT_FLUID, RES_NONE, 1.0f, NULL, NULL,
        "x + sin(fval * x) * sin(fval * x) / fval",
        "1.0 + sin(2.0 * fval * x)"
    }
};

#define NUM_OP_SPECS (sizeof(TKM_OP_REGISTRY) / sizeof(TKM_OP_REGISTRY[0]))

const TkmOpSpec *tkm_find_spec_by_op(TkmOpType op) {
    for (size_t i = 0; i < NUM_OP_SPECS; i++) {
        if (TKM_OP_REGISTRY[i].op == op) return &TKM_OP_REGISTRY[i];
    }
    return NULL;
}

const TkmOpSpec *tkm_find_spec_by_index(int index) {
    if (index < 0 || (size_t)index >= NUM_OP_SPECS) return NULL;
    return &TKM_OP_REGISTRY[index];
}

const TkmOpSpec *tkm_find_spec_by_node(const TkmNode *node) {
    if (!node) return NULL;
    if (node->op == OP_USER_FLUID) return tkm_find_spec_by_index(node->spec_id);
    return tkm_find_spec_by_op(node->op);
}

int tkm_get_spec_index(const TkmOpSpec *spec) {
    if (!spec) return -1;
    for (size_t i = 0; i < NUM_OP_SPECS; i++) {
        if (spec == &TKM_OP_REGISTRY[i]) return (int)i;
    }
    return -1;
}

int tkm_get_spec_count(void) {
    return (int)NUM_OP_SPECS;
}

typedef struct {
    const char *p;
    float x;
    float fval;
    int failed;
} MathExprParser;

static void math_expr_skip_ws(MathExprParser *parser) {
    while (isspace((unsigned char)*parser->p)) parser->p++;
}

static float math_expr_parse_sum(MathExprParser *parser);

static float math_expr_parse_primary(MathExprParser *parser) {
    math_expr_skip_ws(parser);
    if (*parser->p == '(') {
        parser->p++;
        float value = math_expr_parse_sum(parser);
        math_expr_skip_ws(parser);
        if (*parser->p != ')') parser->failed = 1;
        else parser->p++;
        return value;
    }

    if (isdigit((unsigned char)*parser->p) || *parser->p == '.') {
        char *end = NULL;
        float value = strtof(parser->p, &end);
        if (end == parser->p) parser->failed = 1;
        else parser->p = end;
        return value;
    }

    if (isalpha((unsigned char)*parser->p) || *parser->p == '_') {
        char name[32];
        int len = 0;
        while ((isalnum((unsigned char)*parser->p) || *parser->p == '_') && len < (int)sizeof(name) - 1) {
            name[len++] = *parser->p++;
        }
        name[len] = '\0';
        math_expr_skip_ws(parser);
        if (*parser->p != '(') {
            if (!strcmp(name, "x")) return parser->x;
            if (!strcmp(name, "fval")) return parser->fval;
            if (!strcmp(name, "pi")) return 3.14159265358979323846f;
            parser->failed = 1;
            return 0.0f;
        }

        parser->p++;
        float a = math_expr_parse_sum(parser);
        math_expr_skip_ws(parser);
        int has_b = (*parser->p == ',');
        float b = 0.0f;
        if (has_b) {
            parser->p++;
            b = math_expr_parse_sum(parser);
        }
        math_expr_skip_ws(parser);
        if (*parser->p != ')') parser->failed = 1;
        else parser->p++;

        if (!strcmp(name, "exp") && !has_b) return expf(a);
        if ((!strcmp(name, "log") || !strcmp(name, "ln")) && !has_b) return logf(a);
        if (!strcmp(name, "sqrt") && !has_b) return sqrtf(a);
        if (!strcmp(name, "sin") && !has_b) return sinf(a);
        if (!strcmp(name, "cos") && !has_b) return cosf(a);
        if (!strcmp(name, "tanh") && !has_b) return tanhf(a);
        if (!strcmp(name, "abs") && !has_b) return fabsf(a);
        if (!strcmp(name, "max") && has_b) return fmaxf(a, b);
        if (!strcmp(name, "min") && has_b) return fminf(a, b);
        if (!strcmp(name, "pow") && has_b) return powf(a, b);
        parser->failed = 1;
        return 0.0f;
    }

    parser->failed = 1;
    return 0.0f;
}

static float math_expr_parse_unary(MathExprParser *parser) {
    math_expr_skip_ws(parser);
    if (*parser->p == '+') {
        parser->p++;
        return math_expr_parse_unary(parser);
    }
    if (*parser->p == '-') {
        parser->p++;
        return -math_expr_parse_unary(parser);
    }
    return math_expr_parse_primary(parser);
}

static float math_expr_parse_product(MathExprParser *parser) {
    float value = math_expr_parse_unary(parser);
    while (!parser->failed) {
        math_expr_skip_ws(parser);
        char op = *parser->p;
        if (op != '*' && op != '/') break;
        parser->p++;
        float rhs = math_expr_parse_unary(parser);
        value = op == '*' ? value * rhs : value / rhs;
    }
    return value;
}

static float math_expr_parse_sum(MathExprParser *parser) {
    float value = math_expr_parse_product(parser);
    while (!parser->failed) {
        math_expr_skip_ws(parser);
        char op = *parser->p;
        if (op != '+' && op != '-') break;
        parser->p++;
        float rhs = math_expr_parse_product(parser);
        value = op == '+' ? value + rhs : value - rhs;
    }
    return value;
}

int tkm_eval_math_expr(const char *expr, float x, float fval, float *out_value) {
    if (!expr || !out_value) return 0;
    MathExprParser parser = { expr, x, fval, 0 };
    float value = math_expr_parse_sum(&parser);
    math_expr_skip_ws(&parser);
    if (parser.failed || *parser.p) return 0;
    *out_value = value;
    return 1;
}

const TkmOpSpec *tkm_find_spec_by_name(const char *dsl_name) {
    for (size_t i = 0; i < NUM_OP_SPECS; i++) {
        if (!strcmp(TKM_OP_REGISTRY[i].dsl_name, dsl_name)) return &TKM_OP_REGISTRY[i];
    }
    return NULL;
}

/* =========================================================================
   PART 2: Contributor Peephole Fusion Rule Table (4 Rules!)
   ========================================================================= */
static int trace_fc_chain(const TkmGraph *g, int end_id, int *out_fc_id, int *chain_nodes, int *len) {
    int stack[TKM_MAX_CHAIN], cnt = 0, cur = end_id;
    while (cur >= 0 && cnt < TKM_MAX_CHAIN) {
        const TkmNode *n = &g->nodes[cur];
        if (n->op == OP_FC) {
            *out_fc_id = cur;
            *len = cnt;
            for (int i = 0; i < cnt; i++) chain_nodes[i] = stack[cnt - 1 - i];
            return 1;
        }
        if (n->cat != CAT_FLUID && n->op != OP_QKNORM) return 0;
        stack[cnt++] = cur;
        cur = n->in0;
    }
    return 0;
}

static void kill_chain_exclude(TkmGraph *g, int end_id, int stop_id) {
    int cur = end_id;
    while (cur >= 0 && cur != stop_id) {
        g->nodes[cur].dead = 1;
        cur = g->nodes[cur].in0;
    }
}

/* --- Rule 1: Generalized 2-Way Twin-Branch Horizontal GEMM + Dual-Register Fusion --- */
static int match_twin_branch_rule(const TkmGraph *g, int nid) {
    const TkmNode *n = &g->nodes[nid];
    if (n->op != OP_MUL || n->in0 < 0 || n->in1 < 0) return 0;
    int l_fc = -1, r_fc = -1, l_c[TKM_MAX_CHAIN], r_c[TKM_MAX_CHAIN], ll = 0, rl = 0;
    if (trace_fc_chain(g, n->in0, &l_fc, l_c, &ll) &&
        trace_fc_chain(g, n->in1, &r_fc, r_c, &rl)) {
        return (g->nodes[l_fc].in0 == g->nodes[r_fc].in0 &&
                g->nodes[l_fc].c_out == g->nodes[r_fc].c_out);
    }
    return 0;
}

static void apply_twin_branch_rule(TkmGraph *g, int nid) {
    TkmNode *n = &g->nodes[nid];
    int l_fc = -1, r_fc = -1;
    trace_fc_chain(g, n->in0, &l_fc, n->left_nodes,  &n->left_len);
    trace_fc_chain(g, n->in1, &r_fc, n->right_nodes, &n->right_len);

    TkmNode *fc0 = &g->nodes[l_fc];
    int src_in = fc0->in0, half_c = fc0->c_out;
    printf("  [Rule Hit: Horizontal Twin-Branch Merge] #%d + #%d -> 1x GEMM(%d) + Dual-Track Kernel\n",
           l_fc, r_fc, half_c * 2);

    kill_chain_exclude(g, n->in0, src_in);
    kill_chain_exclude(g, n->in1, src_in);
    fc0->dead = 0;
    fc0->op = OP_FC_TWIN_FUSED;
    fc0->cat = CAT_ANCHOR;
    fc0->c_out = half_c * 2;
    snprintf(fc0->tag, sizeof(fc0->tag), "twin_gemm2x_%d", fc0->id);

    n->in0 = fc0->id; n->in1 = -1; n->c_out = half_c;
}

/* --- Rule 2: 3-Way GQA QKV + Optional QKNorm + RoPE Horizontal Merge --- */
static int match_qkv_3way_rule(const TkmGraph *g, int nid) {
    const TkmNode *n = &g->nodes[nid];
    if (n->op != OP_GQA_ATTN || n->in0 < 0 || n->in1 < 0 || n->in2 < 0) return 0;
    int qf = -1, kf = -1, vf = -1, qc[TKM_MAX_CHAIN], kc[TKM_MAX_CHAIN], vc[TKM_MAX_CHAIN], ql, kl, vl;
    if (trace_fc_chain(g, n->in0, &qf, qc, &ql) &&
        trace_fc_chain(g, n->in1, &kf, kc, &kl) &&
        trace_fc_chain(g, n->in2, &vf, vc, &vl)) {
        return (g->nodes[qf].in0 == g->nodes[kf].in0 && g->nodes[kf].in0 == g->nodes[vf].in0);
    }
    return 0;
}

static void apply_qkv_3way_rule(TkmGraph *g, int nid) {
    TkmNode *n = &g->nodes[nid];
    int qf = -1, kf = -1, vf = -1, qc[TKM_MAX_CHAIN], kc[TKM_MAX_CHAIN], vc[TKM_MAX_CHAIN], ql, kl, vl;
    trace_fc_chain(g, n->in0, &qf, qc, &ql);
    trace_fc_chain(g, n->in1, &kf, kc, &kl);
    trace_fc_chain(g, n->in2, &vf, vc, &vl);

    int has_qkn = 0, has_rp = 0;
    for (int m = 0; m < ql; m++) {
        if (g->nodes[qc[m]].op == OP_QKNORM) has_qkn = 1;
        if (g->nodes[qc[m]].op == OP_ROPE)   has_rp = 1;
    }

    TkmNode *fq = &g->nodes[qf], *fk = &g->nodes[kf], *fv = &g->nodes[vf];
    int tot = fq->c_out + fk->c_out + fv->c_out;
    printf("  [Rule Hit: Horizontal 3-Way GQA QKV%s%s Merge] Q(%d)+K(%d)+V(%d) -> 1x GEMM(%d) + Fused QKV-Prep Kernel\n",
           has_qkn ? "+QKNorm" : "", has_rp ? "+RoPE" : "", fq->c_out, fk->c_out, fv->c_out, tot);

    int src_in = fq->in0;
    int prep_nid = qc[0];
    kill_chain_exclude(g, n->in0, src_in);
    kill_chain_exclude(g, n->in1, src_in);
    kill_chain_exclude(g, n->in2, src_in);

    fq->dead = 0; fq->op = OP_FC_QKV_FUSED; fq->cat = CAT_ANCHOR;
    fq->q_c = fq->c_out; fq->k_c = fk->c_out; fq->v_c = fv->c_out;
    fq->c_out = tot;

    TkmNode *prep = &g->nodes[prep_nid];
    prep->dead = 0;
    prep->op = OP_QKNORM;
    prep->cat = CAT_FLUID;
    prep->in0 = fq->id; prep->in1 = -1; prep->in2 = -1;
    prep->q_c = fq->q_c; prep->k_c = fq->k_c; prep->v_c = fq->v_c;
    prep->c_out = tot;
    prep->q_heads = n->q_heads; prep->kv_heads = n->kv_heads; prep->head_dim = n->head_dim;
    prep->has_qknorm = has_qkn; prep->has_rope = has_rp;
    snprintf(prep->tag, sizeof(prep->tag), "qkv_prep_%d", prep->id);
    snprintf(fq->tag, sizeof(fq->tag), "qkv3x_gemm_%d", fq->id);

    n->in0 = prep->id; n->in1 = -1; n->in2 = -1;
}

/* --- Rule 3: Algebraic Consecutive Scale Folding: Scale(a) -> Scale(b) ==> Scale(a*b) --- */
static int match_scale_fold_rule(const TkmGraph *g, int nid) {
    const TkmNode *n = &g->nodes[nid];
    if (n->op != OP_SCALE || n->in0 < 0) return 0;
    const TkmNode *prev = &g->nodes[n->in0];
    return (!prev->dead && prev->op == OP_SCALE);
}

static void apply_scale_fold_rule(TkmGraph *g, int nid) {
    TkmNode *n = &g->nodes[nid];
    TkmNode *prev = &g->nodes[n->in0];
    float combined = prev->fval * n->fval;
    printf("  [Rule Hit: Algebraic Scale Fold] #%d(Scale %.3f) * #%d(Scale %.3f) -> Scale(%.4f)\n",
           prev->id, prev->fval, n->id, n->fval, combined);
    prev->dead = 1;
    n->in0  = prev->in0;
    n->fval = combined;
}

/* --- Rule 4: Dead No-Op Elimination: Drop(0.0) or Scale(1.0) ==> Bypassed! --- */
static int match_noop_elim_rule(const TkmGraph *g, int nid) {
    const TkmNode *n = &g->nodes[nid];
    if (n->op == OP_DROP  && fabsf(n->fval) < 1e-6f) return 1;
    if (n->op == OP_SCALE && fabsf(n->fval - 1.0f) < 1e-6f) return 1;
    return 0;
}

static void apply_noop_elim_rule(TkmGraph *g, int nid) {
    TkmNode *n = &g->nodes[nid];
    printf("  [Rule Hit: Dead No-Op Elimination] Bypassed identity operator #%d(%s)\n", n->id, n->tag);
    int upstream = n->in0;
    n->dead = 1;
    for (int k = nid + 1; k < g->num_nodes; k++) {
        if (g->nodes[k].in0 == nid) g->nodes[k].in0 = upstream;
        if (g->nodes[k].in1 == nid) g->nodes[k].in1 = upstream;
        if (g->nodes[k].in2 == nid) g->nodes[k].in2 = upstream;
    }
}

static const TkmFusionRule TKM_FUSION_RULES[] = {
    { "Horizontal_Twin_Branch_Merge", match_twin_branch_rule, apply_twin_branch_rule },
    { "Horizontal_3Way_QKV_Merge",    match_qkv_3way_rule,    apply_qkv_3way_rule    },
    { "Algebraic_Scale_Fold",         match_scale_fold_rule,  apply_scale_fold_rule  },
    { "Dead_NoOp_Elimination",        match_noop_elim_rule,   apply_noop_elim_rule   }
};

const TkmFusionRule *tkm_get_fusion_rules(int *out_count) {
    *out_count = (int)(sizeof(TKM_FUSION_RULES) / sizeof(TKM_FUSION_RULES[0]));
    return TKM_FUSION_RULES;
}