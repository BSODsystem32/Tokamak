// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * include/tkm.h -- Tokamak (tkm) Complete JIT Engine & C API Header
 *   1. Loop Macro Expansion (`block x N: { ... }`)
 *   2. Multi-Scheme PRNGs (Xorshift32, Wang+Xorshift32, PCG32, Philox4x32-7)
 *   3. Fused 2D Coordinate+Pixel Augmentation, Twin-Branch GLU, & 3-Way QKV+QKNorm+RoPE
 *   4. First-Principles Finite-Difference Numerical Derivative Verification (`tkm numcheck`)
 *   5. Exportable C API (`tkm.dll` / `libtkm.so`)
 */
#ifndef TKM_H
#define TKM_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(TKM_BUILD_DLL)
#define TKM_API __declspec(dllexport)
#elif defined(_WIN32) && defined(TKM_USE_DLL)
#define TKM_API __declspec(dllimport)
#else
#define TKM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TKM_MAX_NODES   1024
#define TKM_MAX_GROUPS  256
#define TKM_MAX_BUFFERS 512
#define TKM_MAX_CHAIN   8
#define TKM_MAX_KARGS   28
#define TKM_MAX_TENSORS 512

typedef enum {
    RNG_DEFAULT = 0,
    RNG_XORSHIFT32,
    RNG_WANG_XORSHIFT32,
    RNG_PCG32,
    RNG_PHILOX
} TkmRngType;

typedef enum {
    RES_NONE         = 0,
    RES_PARAM_BIAS   = (1 << 0),
    RES_PARAM_WEIGHT = (1 << 1),
    RES_PARAM_BETA   = (1 << 2),
    RES_SAVE_ISTD    = (1 << 3),
    RES_SAVE_XHAT    = (1 << 4),
    RES_SAVE_MASK    = (1 << 5),
    RES_RNG_SEED     = (1 << 6),
    RES_BRANCH_IN    = (1 << 7)
} TkmResourceFlags;

typedef enum {
    CAT_META = 0,
    CAT_ANCHOR,
    CAT_SPATIAL,
    CAT_COORD,
    CAT_NORM,
    CAT_FLUID
} TkmOpCat;

typedef enum {
    OP_NONE = 0,
    OP_INPUT,
    /* Anchors */
    OP_CONV, OP_FC, OP_FC_TWIN_FUSED, OP_FC_QKV_FUSED, OP_GQA_ATTN,
    /* Spatial */
    OP_MIXPOOL, OP_AVGPOOL, OP_FLATTEN,
    /* Fusable 2D Coordinate & Pixel Augmentation Bricks */
    OP_RAND_SHIFT, OP_RAND_ROTATE, OP_RAND_SCALE, OP_RAND_FLIPX, OP_RAND_FLIPY,
    OP_RAND_ELASTIC, OP_RAND_NOISE, OP_RAND_CUTOUT, OP_RAND_BRIGHTNESS, OP_RAND_CONTRAST,
    /* Norms */
    OP_BN, OP_LN, OP_RMSNORM, OP_L2NORM, OP_QKNORM,
    /* Fluid Activations & Modulations */
    OP_BIAS, OP_SILU, OP_RELU, OP_RELU2, OP_LEAKY_RELU, OP_ELU,
    OP_GELU, OP_MISH, OP_HARDSWISH, OP_HARDSIGMOID, OP_TANH, OP_SIGMOID,
    OP_SWISH_BETA, OP_SIN, OP_COSGATE, OP_SOFTCAP, OP_CLAMP,
    OP_SCALE, OP_LAYERSCALE, OP_DYT, OP_PRELU, OP_XATLU,
    OP_ROPE, OP_DROP, OP_ADD, OP_MUL,
    OP_SOFTMAX, OP_SOFTMAX_CE, OP_SOFTMAX_FOCAL, OP_SOFTMAX_MSE,
    OP_USER_FLUID,
    OP_COUNT
} TkmOpType;

typedef struct {
    TkmOpType        op;
    const char      *dsl_name;
    TkmOpCat         cat;
    uint32_t         res_flags;
    float            default_fval;
    const char      *fwd_cuda;
    const char      *bwd_cuda;
    const char      *fwd_expr;
    const char      *bwd_expr;
} TkmOpSpec;

typedef enum {
    KARG_IN_MAIN = 0, KARG_OUT_MAIN, KARG_DOUT_MAIN, KARG_DIN_MAIN,
    KARG_BRANCH_IN, KARG_DBRANCH_OUT,
    KARG_BIAS, KARG_DBIAS, KARG_WEIGHT_VEC, KARG_DWEIGHT_VEC, KARG_BETA, KARG_DBETA,
    KARG_SAVE_ISTD, KARG_SAVE_XHAT, KARG_SAVE_MASK, KARG_STEP_SEED
} TkmArgKind;

typedef struct {
    TkmArgKind kind;
    int        node_id;
    char       c_type[32];
    char       c_name[32];
} TkmKernelArg;

typedef struct {
    char         func_name[64];
    TkmKernelArg args[TKM_MAX_KARGS];
    int          num_args;
    int          grid_x, block_x;
} TkmKernelSig;

typedef struct {
    int        id;
    int        spec_id;
    char       tag[32];
    TkmOpType  op;
    TkmOpCat   cat;
    int        dead;

    int        in0, in1, in2;
    int        users;
    int        plasma_grp;

    int        c_out, k, s, p;
    int        q_c, k_c, v_c;
    int        q_heads, kv_heads, head_dim;
    int        has_qknorm, has_rope, causal;
    int        qkv_prep_nid;
    float      fval;
    float      fval2;
    float      rope_theta, rope_scale, rope_low_freq_factor, rope_high_freq_factor;
    float      rope_original_context_length;
    TkmRngType rng_type;

    int        left_nodes[TKM_MAX_CHAIN];
    int        left_len;
    int        right_nodes[TKM_MAX_CHAIN];
    int        right_len;

    int        H, W, C;
    int        out_buf_id;
    size_t     out_offset, out_bytes;
} TkmNode;

typedef struct {
    int          id;
    int          anchor_id;
    int          node_ids[16];
    int          count;
    int          has_norm;
    int          is_twin_epilogue;
    int          is_coord_group;
    int          is_qkv_prep;
    int          out_node_id;
    int          H, W, C;
    TkmKernelSig fwd_sig;
    TkmKernelSig bwd_sig;
} PlasmaGroup;

typedef struct TkmGraph {
    TkmNode      nodes[TKM_MAX_NODES];
    int          num_nodes;
    PlasmaGroup  groups[TKM_MAX_GROUPS];
    int          num_groups;
    int          batch_size;
    TkmRngType   default_rng;
    size_t       total_arena_bytes;
    size_t       naive_arena_bytes;
} TkmGraph;

typedef struct {
    const char *name;
    int  (*match)(const TkmGraph *g, int node_id);
    void (*apply)(TkmGraph *g, int node_id);
} TkmFusionRule;

const TkmOpSpec     *tkm_find_spec_by_op(TkmOpType op);
const TkmOpSpec     *tkm_find_spec_by_name(const char *dsl_name);
const TkmOpSpec     *tkm_find_spec_by_index(int index);
const TkmOpSpec     *tkm_find_spec_by_node(const TkmNode *node);
int                  tkm_get_spec_index(const TkmOpSpec *spec);
int                  tkm_get_spec_count(void);
int                  tkm_eval_math_expr(const char *expr, float x, float fval, float *out_value);
const TkmFusionRule *tkm_get_fusion_rules(int *out_count);
const char          *tkm_rng_name(TkmRngType r);
const char          *tkm_rng_struct_name(TkmRngType r);
TkmRngType           tkm_parse_rng_name(const char *s);

TKM_API int   tkm_verify_registry(void);
TKM_API void  tkm_parse_dsl(TkmGraph *g, const char *dsl_text, int batch_size);
TKM_API void  tkm_infer_shapes(TkmGraph *g);
TKM_API void  tkm_optimize_and_bind(TkmGraph *g);
TKM_API void  tkm_plan_memory(TkmGraph *g);
TKM_API void  tkm_dump_summary(const TkmGraph *g, FILE *out);
TKM_API char *tkm_codegen_cuda_string(const TkmGraph *g);

/* =========================================================================
   5. Public High-Level Engine C API (For External C / C++ / Python Callers)
   ========================================================================= */
typedef struct TkmRuntime TkmRuntime;

TKM_API TkmRuntime *tkm_runtime_ignite(const TkmGraph *g, const char *cuda_src);
TKM_API TkmRuntime *tkm_load_file(const char *tkm_path, int batch_size);
TKM_API TkmRuntime *tkm_load_string(const char *dsl_text, int batch_size);
TKM_API float       tkm_runtime_step(TkmRuntime *rt, const float *h_input, const uint8_t *h_labels, float lr);
TKM_API int         tkm_runtime_eval(TkmRuntime *rt, const float *h_input, const uint8_t *h_labels, int count);
TKM_API size_t      tkm_runtime_forward(TkmRuntime *rt, const float *h_input, float *h_output, size_t output_capacity);
TKM_API int         tkm_save_weights(TkmRuntime *rt, const char *bin_path);
TKM_API int         tkm_load_weights(TkmRuntime *rt, const char *bin_path);
TKM_API void        tkm_runtime_free(TkmRuntime *rt);

/* First-Principles Numerical Finite-Difference GradCheck API */
TKM_API float       tkm_get_grad_sample(TkmRuntime *rt, int node_id, int elem_idx);
TKM_API void        tkm_perturb_tensor(TkmRuntime *rt, int node_id, int elem_idx, float delta);
TKM_API float       tkm_eval_current_loss(TkmRuntime *rt);

#ifdef __cplusplus
}
#endif

#endif /* TKM_H */