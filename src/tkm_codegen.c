// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * src/tkm_codegen.c -- 100% Table-Driven NVRTC Codegen (Zero Operator Special-Cases!)
 */
#include "tkm.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define TKM_ROWS_PER_CTA 16

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} StrBuf;

static void sb_init(StrBuf *sb) {
    sb->cap = 16384;
    sb->len = 0;
    sb->buf = (char*)malloc(sb->cap);
    sb->buf[0] = '\0';
}

static void sb_printf(StrBuf *sb, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    while (sb->len + (size_t)need + 1 > sb->cap) {
        sb->cap *= 2;
        sb->buf = (char*)realloc(sb->buf, sb->cap);
    }
    va_start(ap, fmt);
    vsnprintf(sb->buf + sb->len, (size_t)need + 1, fmt, ap);
    va_end(ap);
    sb->len += (size_t)need;
}

static void emit_sig(StrBuf *sb, const TkmKernelSig *sig) {
    sb_printf(sb, "extern \"C\" __global__ __launch_bounds__(%d, 4)\nvoid %s(\n",
              sig->block_x, sig->func_name);
    for (int i = 0; i < sig->num_args; i++) {
        sb_printf(sb, "    %-30s %s%s\n",
                  sig->args[i].c_type, sig->args[i].c_name, (i + 1 < sig->num_args) ? "," : "");
    }
    sb_printf(sb, ") {\n");
}

static int is_pixel_aug_op(TkmOpType op) {
    return (op == OP_RAND_NOISE || op == OP_RAND_CUTOUT ||
            op == OP_RAND_BRIGHTNESS || op == OP_RAND_CONTRAST);
}

static void emit_coord_plasma_group(const TkmGraph *g, const PlasmaGroup *pg, StrBuf *sb) {
    int H = pg->H, W = pg->W, C = pg->C, HW = H * W, HWC = HW * C;

    emit_sig(sb, &pg->fwd_sig);
    sb_printf(sb, "    int img_id = blockIdx.x;\n");
    sb_printf(sb, "    const int H = %d, W = %d, C = %d;\n", H, W, C);
    sb_printf(sb, "    const float cx = %.4ff, cy = %.4ff;\n", (W - 1) * 0.5f, (H - 1) * 0.5f);
    sb_printf(sb, "    const __half* src_img = in_img  + (size_t)img_id * %d;\n", HWC);
    sb_printf(sb, "    __half*       dst_img = out_img + (size_t)img_id * %d;\n", HWC);
    sb_printf(sb, "    uint32_t base_seed = *d_seed;\n");
    sb_printf(sb, "    for (int pix_id = threadIdx.x; pix_id < %d; pix_id += blockDim.x) {\n", HW);
    sb_printf(sb, "        int px = pix_id %% W, py = pix_id / W;\n");
    sb_printf(sb, "        float fx = (float)px, fy = (float)py;\n");

    for (int i = 0; i < pg->count; i++) {
        const TkmNode *n = &g->nodes[pg->node_ids[i]];
        if (is_pixel_aug_op(n->op)) continue;
        const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
        const char *rng_s = tkm_rng_struct_name(n->rng_type);
        sb_printf(sb, "        { // [Coord Brick #%d: %s]\n", n->id, spec->dsl_name);
        sb_printf(sb, "          const float fval = %.6ff, fval2 = %.6ff; (void)fval; (void)fval2;\n", n->fval, n->fval2);
        sb_printf(sb, "          %s img_rng; rng_init(&img_rng, base_seed ^ %du, (uint32_t)img_id);\n", rng_s, n->id * 19937 + 1);
        sb_printf(sb, "          %s\n        }\n", spec->fwd_cuda);
    }

    sb_printf(sb, "        for (int ch = 0; ch < C; ch++) {\n");
    sb_printf(sb, "            float val = bilinear_sample_nhwc_h(src_img, fx, fy, ch, H, W, C);\n");
    for (int i = 0; i < pg->count; i++) {
        const TkmNode *n = &g->nodes[pg->node_ids[i]];
        if (!is_pixel_aug_op(n->op)) continue;
        const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
        const char *rng_s = tkm_rng_struct_name(n->rng_type);
        sb_printf(sb, "            { // [Pixel Brick #%d: %s]\n", n->id, spec->dsl_name);
        sb_printf(sb, "              const float fval = %.6ff; (void)fval;\n", n->fval);
        sb_printf(sb, "              %s img_rng; rng_init(&img_rng, base_seed ^ %du, (uint32_t)img_id);\n", rng_s, n->id * 19937 + 1);
        sb_printf(sb, "              %s pix_rng; rng_init(&pix_rng, base_seed ^ %du, (uint32_t)((img_id * %d + pix_id) * C + ch));\n",
                  rng_s, n->id * 31337 + 7, HW);
        sb_printf(sb, "              (void)img_rng; (void)pix_rng;\n");
        sb_printf(sb, "              %s\n            }\n", spec->fwd_cuda);
    }
    sb_printf(sb, "            dst_img[pix_id * C + ch] = __float2half(val);\n");
    sb_printf(sb, "        }\n    }\n}\n\n");

    emit_sig(sb, &pg->bwd_sig);
    sb_printf(sb, "    int img_id = blockIdx.x;\n");
    sb_printf(sb, "    for (int idx = threadIdx.x; idx < %d; idx += blockDim.x) {\n", HWC);
    sb_printf(sb, "        din_img[(size_t)img_id * %d + idx] = dout_img[(size_t)img_id * %d + idx];\n    }\n}\n\n", HWC, HWC);
}

static void emit_qkv_prep_plasma_group(const TkmGraph *g, const PlasmaGroup *pg, StrBuf *sb) {
    const TkmNode *n = &g->nodes[pg->node_ids[0]];
    int rows = g->batch_size * pg->H * pg->W;
    int seq_len = pg->H * pg->W;
    int C2 = pg->C / 2;
    int qk_c2 = (n->q_c + n->k_c) / 2;
    int hd = (n->head_dim > 0) ? n->head_dim : 64;
    int hd2 = hd / 2;
    int shfl_span = (hd2 <= 32) ? (hd2 / 2) : 16;
    int aligned_c2 = ((C2 + 255) / 256) * 256;

    emit_sig(sb, &pg->fwd_sig);
    sb_printf(sb, "    int base_row = blockIdx.x * %d;\n", TKM_ROWS_PER_CTA);
    sb_printf(sb, "    for (int tid = threadIdx.x; tid < %d; tid += 256) {\n", aligned_c2);
    sb_printf(sb, "        int active_c = (tid < %d);\n", C2);
    sb_printf(sb, "        float2 b = active_c ? b_vec_%d[tid] : make_float2(0.f, 0.f);\n", n->id);
    if (n->has_qknorm) {
        sb_printf(sb, "        float2 gm = (active_c && tid < %d) ? w_vec_%d[tid] : make_float2(1.f, 1.f);\n", qk_c2, n->id);
    }
    sb_printf(sb, "        for (int r = 0; r < %d; r++) {\n", TKM_ROWS_PER_CTA);
    sb_printf(sb, "            int row = base_row + r;\n");
    sb_printf(sb, "            int active = active_c && (row < %d);\n", rows);
    sb_printf(sb, "            int pos = row %% %d;\n", seq_len > 0 ? seq_len : 1);
    sb_printf(sb, "            int idx2 = row * %d + tid;\n", C2);
    sb_printf(sb, "            float2 v = active ? __half22float2(__ldg(&in_0[idx2])) : make_float2(0.f, 0.f);\n");
    sb_printf(sb, "            v.x += b.x; v.y += b.y;\n");
    if (n->has_qknorm) {
        sb_printf(sb, "            float sq = subwarp_reduce_sum(v.x * v.x + v.y * v.y, %d);\n", shfl_span);
        sb_printf(sb, "            float istd = rsqrtf(sq * %.8ff + 1e-5f);\n", 1.0f / (float)hd);
        sb_printf(sb, "            if (active && tid < %d) {\n", qk_c2);
        sb_printf(sb, "                float2 xh = make_float2(v.x * istd, v.y * istd);\n");
        sb_printf(sb, "                save_xhat_%d[idx2] = __float22half2_rn(xh);\n", n->id);
        sb_printf(sb, "                if ((tid & %d) == 0) save_istd_%d[row] = istd;\n", hd2 - 1, n->id);
        sb_printf(sb, "                v = make_float2(xh.x * gm.x, xh.y * gm.y);\n            }\n");
    }
    if (n->has_rope) {
        sb_printf(sb, "            if (active && tid < %d) {\n", qk_c2);
        sb_printf(sb, "                int lane = tid & %d;\n", hd2 - 1);
        sb_printf(sb, "                float freq = (float)pos * __expf(-0.1440425f * (float)lane);\n");
        sb_printf(sb, "                float sf, cf; __sincosf(freq, &sf, &cf);\n");
        sb_printf(sb, "                v = make_float2(v.x * cf - v.y * sf, v.x * sf + v.y * cf);\n            }\n");
    }
    sb_printf(sb, "            if (active) out_0[idx2] = __float22half2_rn(v);\n        }\n    }\n}\n\n");

    emit_sig(sb, &pg->bwd_sig);
    sb_printf(sb, "    int base_row = blockIdx.x * %d; (void)replay_in_0; (void)b_vec_%d;\n", TKM_ROWS_PER_CTA, n->id);
    sb_printf(sb, "    for (int tid = threadIdx.x; tid < %d; tid += 256) {\n", aligned_c2);
    sb_printf(sb, "        int active_c = (tid < %d);\n", C2);
    sb_printf(sb, "        float2 acc_dw = make_float2(0.f, 0.f), acc_db = make_float2(0.f, 0.f);\n");
    if (n->has_qknorm) {
        sb_printf(sb, "        float2 gm = (active_c && tid < %d) ? w_vec_%d[tid] : make_float2(1.f, 1.f);\n", qk_c2, n->id);
    }
    sb_printf(sb, "        for (int r = 0; r < %d; r++) {\n", TKM_ROWS_PER_CTA);
    sb_printf(sb, "            int row = base_row + r;\n");
    sb_printf(sb, "            int active = active_c && (row < %d);\n", rows);
    sb_printf(sb, "            int pos = row %% %d;\n", seq_len > 0 ? seq_len : 1);
    sb_printf(sb, "            int idx2 = row * %d + tid;\n", C2);
    sb_printf(sb, "            float2 dy = active ? __half22float2(__ldg(&dout_0[idx2])) : make_float2(0.f, 0.f);\n");
    if (n->has_rope) {
        sb_printf(sb, "            if (active && tid < %d) {\n", qk_c2);
        sb_printf(sb, "                int lane = tid & %d;\n", hd2 - 1);
        sb_printf(sb, "                float freq = (float)pos * __expf(-0.1440425f * (float)lane);\n");
        sb_printf(sb, "                float sf, cf; __sincosf(freq, &sf, &cf);\n");
        sb_printf(sb, "                dy = make_float2(dy.x * cf + dy.y * sf, -dy.x * sf + dy.y * cf);\n            }\n");
    }
    if (n->has_qknorm) {
        sb_printf(sb, "            float2 xh = (active && tid < %d) ? __half22float2(__ldg(&save_xhat_%d[idx2])) : make_float2(0.f, 0.f);\n", qk_c2, n->id);
        sb_printf(sb, "            if (active && tid < %d) { acc_dw.x += dy.x * xh.x; acc_dw.y += dy.y * xh.y; }\n");
        sb_printf(sb, "            float2 dxh = make_float2(dy.x * gm.x, dy.y * gm.y);\n");
        sb_printf(sb, "            float m2 = subwarp_reduce_sum(dxh.x * xh.x + dxh.y * xh.y, %d) * %.8ff;\n", shfl_span, 1.0f / (float)hd);
        sb_printf(sb, "            if (active && tid < %d) {\n", qk_c2);
        sb_printf(sb, "                float istd = save_istd_%d[row];\n", n->id);
        sb_printf(sb, "                dy = make_float2(istd * (dxh.x - xh.x * m2), istd * (dxh.y - xh.y * m2));\n            }\n");
    }
    sb_printf(sb, "            if (active) {\n"
                  "                acc_db.x += dy.x; acc_db.y += dy.y;\n"
                  "                float2 prev = __half22float2(din_0[idx2]);\n"
                  "                din_0[idx2] = __floats2half2_rn(prev.x + dy.x, prev.y + dy.y);\n            }\n        }\n");
    if (n->has_qknorm) {
        sb_printf(sb, "        if (active_c && tid < %d) {\n"
                      "            atomicAdd(&dw_vec_%d[tid*2+0], acc_dw.x); atomicAdd(&dw_vec_%d[tid*2+1], acc_dw.y);\n        }\n",
                  qk_c2, n->id, n->id);
    }
    sb_printf(sb, "        if (active_c) {\n"
                  "            atomicAdd(&db_vec_%d[tid*2+0], acc_db.x); atomicAdd(&db_vec_%d[tid*2+1], acc_db.y);\n        }\n    }\n}\n\n",
              n->id, n->id);
}

static void emit_node_fwd(StrBuf *sb, const TkmNode *n, const char *var_name) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec || !spec->fwd_cuda) return;
    uint32_t f = spec->res_flags;

    sb_printf(sb, "            { // [Op #%d: %s]\n", n->id, spec->dsl_name);
    sb_printf(sb, "              float2 v = %s; const float fval = %.6ff; (void)fval;\n", var_name, n->fval);
    if (f & RES_PARAM_BIAS)   sb_printf(sb, "              const float2* b_vec = b_vec_%d;\n", n->id);
    if (f & RES_PARAM_WEIGHT) sb_printf(sb, "              const float2* w_vec = w_vec_%d;\n", n->id);
    if (f & RES_PARAM_BETA)   sb_printf(sb, "              const float2* beta_vec = beta_vec_%d;\n", n->id);
    if (f & RES_SAVE_ISTD)    sb_printf(sb, "              float* save_istd = save_istd_%d;\n", n->id);
    if (f & RES_SAVE_XHAT)    sb_printf(sb, "              __half2* save_xhat = save_xhat_%d;\n", n->id);
    if (f & RES_SAVE_MASK)    sb_printf(sb, "              uint16_t* save_mask = save_mask_%d;\n", n->id);
    if (f & RES_RNG_SEED) {
        const char *rng_s = tkm_rng_struct_name(n->rng_type);
        sb_printf(sb, "              %s rng; rng_init(&rng, (*d_seed_%d) ^ %du, (uint32_t)idx2);\n",
                  rng_s, n->id, n->id * 19937 + 1);
    }
    if (f & RES_BRANCH_IN)    sb_printf(sb, "              const __half2* branch_in = branch_in_%d;\n", n->in1);

    if (n->cat == CAT_NORM) {
        if (n->op == OP_RMSNORM || n->op == OP_L2NORM) {
            sb_printf(sb, "              float sq = row_reduce_sum(v.x * v.x + v.y * v.y);\n"
                          "              float istd = rsqrtf(sq * inv_C + fval);\n"
                          "              float2 xh = make_float2(v.x * istd, v.y * istd);\n"
                          "              if (active) {\n"
                          "                  if (tid == 0) save_istd[row] = istd;\n"
                          "                  save_xhat[idx2] = __float22half2_rn(xh);\n");
            if (f & RES_PARAM_WEIGHT)
                sb_printf(sb, "                  float2 gm = w_vec[tid]; v = make_float2(xh.x * gm.x, xh.y * gm.y);\n");
            else
                sb_printf(sb, "                  v = xh;\n");
            sb_printf(sb, "              }\n");
        } else {
            sb_printf(sb, "              float m = row_reduce_sum(v.x + v.y) * inv_C;\n"
                          "              float vr = fmaxf(0.f, row_reduce_sum(v.x*v.x + v.y*v.y) * inv_C - m * m);\n"
                          "              float istd = rsqrtf(vr + fval);\n"
                          "              float2 xh = make_float2((v.x - m) * istd, (v.y - m) * istd);\n"
                          "              if (active) {\n"
                          "                  if (tid == 0) save_istd[row] = istd;\n"
                          "                  save_xhat[idx2] = __float22half2_rn(xh);\n"
                          "                  float2 gm = w_vec[tid], bt = beta_vec[tid];\n"
                          "                  v = make_float2(fmaf(gm.x, xh.x, bt.x), fmaf(gm.y, xh.y, bt.y));\n"
                          "              }\n");
        }
    } else {
        sb_printf(sb, "              if (active) {\n                %s\n              }\n", spec->fwd_cuda);
    }
    sb_printf(sb, "              %s = v;\n            }\n", var_name);
}

static void emit_node_bwd_acc_decl(StrBuf *sb, const TkmNode *n) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec) return;
    uint32_t f = spec->res_flags;
    if (f & RES_PARAM_WEIGHT) sb_printf(sb, "        float2 acc_dw_%d = make_float2(0.f, 0.f);\n", n->id);
    if (f & RES_PARAM_BETA)   sb_printf(sb, "        float2 acc_dbeta_%d = make_float2(0.f, 0.f);\n", n->id);
    if (f & RES_PARAM_BIAS)   sb_printf(sb, "        float2 acc_db_%d = make_float2(0.f, 0.f);\n", n->id);
}

static void emit_node_bwd_acc_flush(StrBuf *sb, const TkmNode *n, int subwarp_mode, int C2) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec) return;
    uint32_t f = spec->res_flags;
    if (!(f & (RES_PARAM_WEIGHT | RES_PARAM_BETA | RES_PARAM_BIAS))) return;

    if (subwarp_mode) {
        int rows_per_blk = 256 / C2;
        if (f & RES_PARAM_WEIGHT) {
            sb_printf(sb, "        { __shared__ float2 s_dw[256]; s_dw[threadIdx.x] = acc_dw_%d; __syncthreads();\n"
                          "          if (threadIdx.x < %d) {\n"
                          "              float2 s = make_float2(0.f, 0.f);\n"
                          "              for (int r = 0; r < %d; r++) { s.x += s_dw[r*%d+threadIdx.x].x; s.y += s_dw[r*%d+threadIdx.x].y; }\n"
                          "              atomicAdd(&dw_vec_%d[threadIdx.x*2+0], s.x); atomicAdd(&dw_vec_%d[threadIdx.x*2+1], s.y);\n"
                          "          } __syncthreads(); }\n", n->id, C2, rows_per_blk, C2, C2, n->id, n->id);
        }
        if (f & RES_PARAM_BETA) {
            sb_printf(sb, "        { __shared__ float2 s_db[256]; s_db[threadIdx.x] = acc_dbeta_%d; __syncthreads();\n"
                          "          if (threadIdx.x < %d) {\n"
                          "              float2 s = make_float2(0.f, 0.f);\n"
                          "              for (int r = 0; r < %d; r++) { s.x += s_db[r*%d+threadIdx.x].x; s.y += s_db[r*%d+threadIdx.x].y; }\n"
                          "              atomicAdd(&dbeta_vec_%d[threadIdx.x*2+0], s.x); atomicAdd(&dbeta_vec_%d[threadIdx.x*2+1], s.y);\n"
                          "          } __syncthreads(); }\n", n->id, C2, rows_per_blk, C2, C2, n->id, n->id);
        }
        if (f & RES_PARAM_BIAS) {
            sb_printf(sb, "        { __shared__ float2 s_b[256]; s_b[threadIdx.x] = acc_db_%d; __syncthreads();\n"
                          "          if (threadIdx.x < %d) {\n"
                          "              float2 s = make_float2(0.f, 0.f);\n"
                          "              for (int r = 0; r < %d; r++) { s.x += s_b[r*%d+threadIdx.x].x; s.y += s_b[r*%d+threadIdx.x].y; }\n"
                          "              atomicAdd(&db_vec_%d[threadIdx.x*2+0], s.x); atomicAdd(&db_vec_%d[threadIdx.x*2+1], s.y);\n"
                          "          } __syncthreads(); }\n", n->id, C2, rows_per_blk, C2, C2, n->id, n->id);
        }
    } else {
        sb_printf(sb, "        if (tid < %d) {\n", C2);
        if (f & RES_PARAM_WEIGHT)
            sb_printf(sb, "            atomicAdd(&dw_vec_%d[tid*2+0], acc_dw_%d.x); atomicAdd(&dw_vec_%d[tid*2+1], acc_dw_%d.y);\n", n->id, n->id, n->id, n->id);
        if (f & RES_PARAM_BETA)
            sb_printf(sb, "            atomicAdd(&dbeta_vec_%d[tid*2+0], acc_dbeta_%d.x); atomicAdd(&dbeta_vec_%d[tid*2+1], acc_dbeta_%d.y);\n", n->id, n->id, n->id, n->id);
        if (f & RES_PARAM_BIAS)
            sb_printf(sb, "            atomicAdd(&db_vec_%d[tid*2+0], acc_db_%d.x); atomicAdd(&db_vec_%d[tid*2+1], acc_db_%d.y);\n", n->id, n->id, n->id, n->id);
        sb_printf(sb, "        }\n");
    }
}

static void emit_node_bwd_step(StrBuf *sb, const TkmNode *n, const char *grad_var, const char *pre_var) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec || !spec->bwd_cuda) return;
    uint32_t f = spec->res_flags;

    sb_printf(sb, "            { // [Bwd Op #%d: %s]\n", n->id, spec->dsl_name);
    sb_printf(sb, "              float2 dy = %s, pre = %s; const float fval = %.6ff; (void)pre; (void)fval;\n", grad_var, pre_var, n->fval);
    if (f & RES_PARAM_BIAS)   sb_printf(sb, "              float2 acc_db = acc_db_%d;\n", n->id);
    if (f & RES_PARAM_WEIGHT) sb_printf(sb, "              const float2* w_vec = w_vec_%d; float2 acc_dw = acc_dw_%d;\n", n->id, n->id);
    if (f & RES_PARAM_BETA)   sb_printf(sb, "              const float2* beta_vec = beta_vec_%d; float2 acc_dbeta = acc_dbeta_%d; (void)beta_vec;\n", n->id, n->id);
    if (f & RES_SAVE_ISTD)    sb_printf(sb, "              const float* save_istd = save_istd_%d;\n", n->id);
    if (f & RES_SAVE_XHAT)    sb_printf(sb, "              const __half2* save_xhat = save_xhat_%d;\n", n->id);
    if (f & RES_SAVE_MASK)    sb_printf(sb, "              const uint16_t* save_mask = save_mask_%d;\n", n->id);
    if (f & RES_BRANCH_IN)    sb_printf(sb, "              __half2* dbranch_out = dbranch_out_%d;\n", n->in1);

    if (n->cat == CAT_NORM) {
        sb_printf(sb, "              float2 xh = active ? __half22float2(__ldg(&save_xhat[idx2])) : make_float2(0.f, 0.f);\n");
        if (f & RES_PARAM_WEIGHT) {
            sb_printf(sb, "              float2 gm = active ? w_vec[tid] : make_float2(0.f, 0.f);\n"
                          "              if (active) { acc_dw.x += dy.x * xh.x; acc_dw.y += dy.y * xh.y; }\n");
        } else {
            sb_printf(sb, "              float2 gm = make_float2(1.f, 1.f);\n");
        }
        if (f & RES_PARAM_BETA) {
            sb_printf(sb, "              if (active) { acc_dbeta.x += dy.x; acc_dbeta.y += dy.y; }\n");
        }
        sb_printf(sb, "              float2 dxh = make_float2(dy.x * gm.x, dy.y * gm.y);\n");
        if (n->op == OP_RMSNORM || n->op == OP_L2NORM) {
            sb_printf(sb, "              float m2 = row_reduce_sum(dxh.x * xh.x + dxh.y * xh.y) * inv_C;\n"
                          "              if (active) {\n"
                          "                  float istd = save_istd[row];\n"
                          "                  dy = make_float2(istd * (dxh.x - xh.x * m2), istd * (dxh.y - xh.y * m2));\n"
                          "              }\n");
        } else {
            sb_printf(sb, "              float m1 = row_reduce_sum(dxh.x + dxh.y) * inv_C;\n"
                          "              float m2 = row_reduce_sum(dxh.x * xh.x + dxh.y * xh.y) * inv_C;\n"
                          "              if (active) {\n"
                          "                  float istd = save_istd[row];\n"
                          "                  dy = make_float2(istd * (dxh.x - m1 - xh.x * m2), istd * (dxh.y - m1 - xh.y * m2));\n"
                          "              }\n");
        }
    } else if (n->op == OP_ADD) {
        sb_printf(sb, "              if (active) {\n"
                      "                  float2 p_br = __half22float2(dbranch_out[idx2]);\n"
                      "                  dbranch_out[idx2] = __floats2half2_rn(p_br.x + dy.x, p_br.y + dy.y);\n              }\n");
    } else {
        sb_printf(sb, "              if (active) {\n                %s\n              }\n", spec->bwd_cuda);
    }

    if (f & RES_PARAM_BIAS)   sb_printf(sb, "              acc_db_%d = acc_db;\n", n->id);
    if (f & RES_PARAM_WEIGHT) sb_printf(sb, "              acc_dw_%d = acc_dw;\n", n->id);
    if (f & RES_PARAM_BETA)   sb_printf(sb, "              acc_dbeta_%d = acc_dbeta;\n", n->id);
    sb_printf(sb, "              %s = dy;\n            }\n", grad_var);
}

static void emit_replay_step(StrBuf *sb, const TkmNode *n, const char *vn) {
    const TkmOpSpec *spec = tkm_find_spec_by_op(n->op);
    if (!spec) return;
    if (spec->res_flags & RES_SAVE_XHAT) {
        sb_printf(sb, "            if (active) { float2 xh = __half22float2(__ldg(&save_xhat_%d[idx2]));\n", n->id);
        if (spec->res_flags & RES_PARAM_BETA) {
            sb_printf(sb, "              float2 gm = w_vec_%d[tid], bt = beta_vec_%d[tid];\n"
                          "              %s = make_float2(fmaf(gm.x, xh.x, bt.x), fmaf(gm.y, xh.y, bt.y)); }\n",
                      n->id, n->id, vn);
        } else if (spec->res_flags & RES_PARAM_WEIGHT) {
            sb_printf(sb, "              float2 gm = w_vec_%d[tid];\n"
                          "              %s = make_float2(gm.x * xh.x, gm.y * xh.y); }\n", n->id, vn);
        } else {
            sb_printf(sb, "              %s = xh; }\n", vn);
        }
    } else if (!(spec->res_flags & (RES_SAVE_MASK | RES_BRANCH_IN))) {
        emit_node_fwd(sb, n, vn);
    }
}

static void emit_plasma_group(const TkmGraph *g, const PlasmaGroup *pg, StrBuf *sb) {
    if (pg->is_coord_group) { emit_coord_plasma_group(g, pg, sb); return; }
    if (pg->is_qkv_prep)    { emit_qkv_prep_plasma_group(g, pg, sb); return; }

    int rows = g->batch_size * pg->H * pg->W;
    int C = pg->C, C2 = C / 2;
    int subwarp_mode = (C2 > 0 && C2 <= 32 && (256 % C2 == 0));
    int shfl_span = subwarp_mode ? (C2 / 2) : 16;
    int aligned_c2 = ((C2 + 255) / 256) * 256;

    emit_sig(sb, &pg->fwd_sig);
    sb_printf(sb, "    const float inv_C = %.8ff; (void)inv_C;\n", 1.0f / (float)C);
    sb_printf(sb, "    #define row_reduce_sum(val) subwarp_reduce_sum((val), %d)\n", shfl_span);
    if (subwarp_mode) {
        int rows_per_blk = 256 / C2;
        sb_printf(sb, "    int row = blockIdx.x * %d + (threadIdx.x / %d);\n", rows_per_blk, C2);
        sb_printf(sb, "    int tid = threadIdx.x %% %d;\n", C2);
        sb_printf(sb, "    int active = (row < %d);\n    {\n        {\n", rows);
    } else {
        sb_printf(sb, "    int base_row = blockIdx.x * %d;\n", TKM_ROWS_PER_CTA);
        sb_printf(sb, "    for (int tid = threadIdx.x; tid < %d; tid += 256) {\n", aligned_c2);
        sb_printf(sb, "        for (int r = 0; r < %d; r++) {\n", TKM_ROWS_PER_CTA);
        sb_printf(sb, "            int row = base_row + r;\n");
        sb_printf(sb, "            int active = (row < %d) && (tid < %d);\n", rows, C2);
    }
    sb_printf(sb, "            int idx2 = row * %d + tid;\n", C2);

    if (pg->is_twin_epilogue) {
        const TkmNode *mul_n = &g->nodes[pg->node_ids[0]];
        sb_printf(sb, "            float2 vl = active ? __half22float2(__ldg(&in_0[row * %d + tid])) : make_float2(0.f, 0.f);\n", C);
        sb_printf(sb, "            float2 vr = active ? __half22float2(__ldg(&in_0[row * %d + %d + tid])) : make_float2(0.f, 0.f);\n", C, C2);
        for (int i = 0; i < mul_n->left_len; i++)  emit_node_fwd(sb, &g->nodes[mul_n->left_nodes[i]],  "vl");
        for (int i = 0; i < mul_n->right_len; i++) emit_node_fwd(sb, &g->nodes[mul_n->right_nodes[i]], "vr");
        sb_printf(sb, "            float2 cur_v = make_float2(vl.x * vr.x, vl.y * vr.y);\n");
        for (int i = 1; i < pg->count; i++) emit_node_fwd(sb, &g->nodes[pg->node_ids[i]], "cur_v");
    } else {
        sb_printf(sb, "            float2 cur_v = active ? __half22float2(__ldg(&in_0[idx2])) : make_float2(0.f, 0.f);\n");
        for (int i = 0; i < pg->count; i++) emit_node_fwd(sb, &g->nodes[pg->node_ids[i]], "cur_v");
    }
    sb_printf(sb, "            if (active) out_0[idx2] = __float22half2_rn(cur_v);\n        }\n    }\n    #undef row_reduce_sum\n}\n\n");

    emit_sig(sb, &pg->bwd_sig);
    sb_printf(sb, "    const float inv_C = %.8ff; (void)inv_C; (void)replay_in_0;\n", 1.0f / (float)C);
    sb_printf(sb, "    #define row_reduce_sum(val) subwarp_reduce_sum((val), %d)\n", shfl_span);
    for (int i = 0; i < pg->bwd_sig.num_args; i++) {
        if (pg->bwd_sig.args[i].kind == KARG_BIAS) sb_printf(sb, "    (void)%s;\n", pg->bwd_sig.args[i].c_name);
    }

    if (subwarp_mode) {
        int rows_per_blk = 256 / C2;
        sb_printf(sb, "    int row = blockIdx.x * %d + (threadIdx.x / %d);\n", rows_per_blk, C2);
        sb_printf(sb, "    int tid = threadIdx.x %% %d;\n", C2);
        sb_printf(sb, "    int active = (row < %d);\n    {\n", rows);
    } else {
        sb_printf(sb, "    int base_row = blockIdx.x * %d;\n", TKM_ROWS_PER_CTA);
        sb_printf(sb, "    for (int tid = threadIdx.x; tid < %d; tid += 256) {\n", aligned_c2);
    }

    if (pg->is_twin_epilogue) {
        const TkmNode *mul_n = &g->nodes[pg->node_ids[0]];
        for (int i = 0; i < mul_n->left_len; i++)  emit_node_bwd_acc_decl(sb, &g->nodes[mul_n->left_nodes[i]]);
        for (int i = 0; i < mul_n->right_len; i++) emit_node_bwd_acc_decl(sb, &g->nodes[mul_n->right_nodes[i]]);
    }
    for (int i = 0; i < pg->count; i++) emit_node_bwd_acc_decl(sb, &g->nodes[pg->node_ids[i]]);

    if (subwarp_mode) {
        sb_printf(sb, "        {\n");
    } else {
        sb_printf(sb, "        for (int r = 0; r < %d; r++) {\n", TKM_ROWS_PER_CTA);
        sb_printf(sb, "            int row = base_row + r;\n");
        sb_printf(sb, "            int active = (row < %d) && (tid < %d);\n", rows, C2);
    }
    sb_printf(sb, "            int idx2 = row * %d + tid;\n", C2);
    sb_printf(sb, "            float2 cur_dy = active ? __half22float2(__ldg(&dout_0[idx2])) : make_float2(0.f, 0.f);\n");

    if (pg->is_twin_epilogue) {
        const TkmNode *mul_n = &g->nodes[pg->node_ids[0]];
        sb_printf(sb, "            float2 vl_0 = active ? __half22float2(__ldg(&replay_in_0[row * %d + tid])) : make_float2(0.f, 0.f);\n", C);
        for (int i = 0; i < mul_n->left_len; i++) {
            sb_printf(sb, "            float2 vl_%d = vl_%d;\n", i + 1, i);
            char vn[16]; snprintf(vn, sizeof(vn), "vl_%d", i + 1);
            emit_replay_step(sb, &g->nodes[mul_n->left_nodes[i]], vn);
        }
        sb_printf(sb, "            float2 vr_0 = active ? __half22float2(__ldg(&replay_in_0[row * %d + %d + tid])) : make_float2(0.f, 0.f);\n", C, C2);
        for (int i = 0; i < mul_n->right_len; i++) {
            sb_printf(sb, "            float2 vr_%d = vr_%d;\n", i + 1, i);
            char vn[16]; snprintf(vn, sizeof(vn), "vr_%d", i + 1);
            emit_replay_step(sb, &g->nodes[mul_n->right_nodes[i]], vn);
        }
        sb_printf(sb, "            float2 vp_0 = make_float2(vl_%d.x * vr_%d.x, vl_%d.y * vr_%d.y); (void)vp_0;\n",
                  mul_n->left_len, mul_n->right_len, mul_n->left_len, mul_n->right_len);
        for (int i = 1; i < pg->count; i++) {
            sb_printf(sb, "            float2 vp_%d = vp_%d; (void)vp_%d;\n", i, i - 1, i);
            char vn[16]; snprintf(vn, sizeof(vn), "vp_%d", i);
            emit_replay_step(sb, &g->nodes[pg->node_ids[i]], vn);
        }
        for (int i = pg->count - 1; i >= 1; i--) {
            char pv[16]; snprintf(pv, sizeof(pv), "vp_%d", i - 1);
            emit_node_bwd_step(sb, &g->nodes[pg->node_ids[i]], "cur_dy", pv);
        }
        sb_printf(sb, "            float2 dvl = make_float2(cur_dy.x * vr_%d.x, cur_dy.y * vr_%d.y);\n", mul_n->right_len, mul_n->right_len);
        sb_printf(sb, "            float2 dvr = make_float2(cur_dy.x * vl_%d.x, cur_dy.y * vl_%d.y);\n", mul_n->left_len, mul_n->left_len);
        for (int i = mul_n->left_len - 1; i >= 0; i--) {
            char pv[16]; snprintf(pv, sizeof(pv), "vl_%d", i);
            emit_node_bwd_step(sb, &g->nodes[mul_n->left_nodes[i]], "dvl", pv);
        }
        for (int i = mul_n->right_len - 1; i >= 0; i--) {
            char pv[16]; snprintf(pv, sizeof(pv), "vr_%d", i);
            emit_node_bwd_step(sb, &g->nodes[mul_n->right_nodes[i]], "dvr", pv);
        }
        sb_printf(sb, "            if (active) {\n"
                      "                din_0[row * %d + tid]       = __float22half2_rn(dvl);\n"
                      "                din_0[row * %d + %d + tid] = __float22half2_rn(dvr);\n"
                      "            }\n", C, C, C2);
    } else {
        sb_printf(sb, "            float2 v_0 = active ? __half22float2(__ldg(&replay_in_0[idx2])) : make_float2(0.f, 0.f); (void)v_0;\n");
        for (int i = 0; i < pg->count; i++) {
            sb_printf(sb, "            float2 v_%d = v_%d; (void)v_%d;\n", i + 1, i, i + 1);
            char vn[16]; snprintf(vn, sizeof(vn), "v_%d", i + 1);
            emit_replay_step(sb, &g->nodes[pg->node_ids[i]], vn);
        }
        for (int i = pg->count - 1; i >= 0; i--) {
            char pv[16]; snprintf(pv, sizeof(pv), "v_%d", i);
            emit_node_bwd_step(sb, &g->nodes[pg->node_ids[i]], "cur_dy", pv);
        }
        sb_printf(sb, "            if (active) {\n"
                      "                float2 prev_d = __half22float2(din_0[idx2]);\n"
                      "                din_0[idx2] = __floats2half2_rn(prev_d.x + cur_dy.x, prev_d.y + cur_dy.y);\n"
                      "            }\n");
    }
    sb_printf(sb, "        }\n");

    if (pg->is_twin_epilogue) {
        const TkmNode *mul_n = &g->nodes[pg->node_ids[0]];
        for (int i = 0; i < mul_n->left_len; i++)  emit_node_bwd_acc_flush(sb, &g->nodes[mul_n->left_nodes[i]],  subwarp_mode, C2);
        for (int i = 0; i < mul_n->right_len; i++) emit_node_bwd_acc_flush(sb, &g->nodes[mul_n->right_nodes[i]], subwarp_mode, C2);
    }
    for (int i = 0; i < pg->count; i++) emit_node_bwd_acc_flush(sb, &g->nodes[pg->node_ids[i]], subwarp_mode, C2);

    sb_printf(sb, "    }\n    #undef row_reduce_sum\n}\n\n");
}

char *tkm_codegen_cuda_string(const TkmGraph *g) {
    StrBuf sb;
    sb_init(&sb);
    sb_printf(&sb, "// --- Auto-Generated by Tokamak (tkm) Table-Driven Codegen ---\n\n");
    for (int i = 0; i < g->num_groups; i++) {
        emit_plasma_group(g, &g->groups[i], &sb);
    }
    return sb.buf;
}