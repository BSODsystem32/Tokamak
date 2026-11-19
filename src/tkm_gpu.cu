// SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// Copyright (c) 2026 BSODsystem32

/*
 * src/tkm_gpu.cu -- Complete GPU Execution Backend for Tokamak (tkm)
 *   1. Zero-Header In-Memory NVRTC Compilation (Multi-PRNG + NHWC Bilinear Prelude)
 *   2. Tensor Core (WMMA 16x16x16, 22KB SMEM) GQA FlashAttention-2 Forward & Backward
 *   3. First-Principles Numerical Finite-Difference GradCheck Probe Utilities
 *   4. Full-Step CUDA Graph Capture + Weight Save/Load & External Eval API
 */
#include "tkm.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <nvrtc.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_s; cuGetErrorString(_r, &_s); \
        fprintf(stderr, "[CU Driver Error] %s:%d: %s\n", __FILE__, __LINE__, _s); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[cuBLAS Error] %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        exit(1); \
    } \
} while(0)

#define NVRTC_CHECK(call) do { \
    nvrtcResult _r = (call); \
    if (_r != NVRTC_SUCCESS) { \
        fprintf(stderr, "[NVRTC Error] %s:%d: %s\n", __FILE__, __LINE__, nvrtcGetErrorString(_r)); \
        exit(1); \
    } \
} while(0)

__global__ void k_f32_to_f16(const float * __restrict__ src, __half * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

struct TkmSchedule {
    float peak_lr;
    float min_lr_ratio;
    uint32_t total_steps;
    uint32_t warmup_steps;
};

__global__ void k_resident_gather(const __half * __restrict__ images,
                                  const uint8_t * __restrict__ labels,
                                  const uint32_t * __restrict__ order,
                                  __half * __restrict__ batch_images,
                                  uint8_t * __restrict__ batch_labels,
                                  const uint32_t * __restrict__ batch_idx,
                                  int sample_elems, int batch_size) {
    int sample = blockIdx.x;
    if (sample >= batch_size) return;
    uint32_t b = *batch_idx;
    uint32_t source = order[(size_t)b * (size_t)batch_size + (size_t)sample];
    if (threadIdx.x == 0)
        batch_labels[sample] = labels[source];

    const __half *src = images + (size_t)source * sample_elems;
    __half *dst = batch_images + (size_t)sample * sample_elems;
    if (sample_elems % 8 == 0) {
        const uint4 *src4 = (const uint4*)src;
        uint4 *dst4 = (uint4*)dst;
        int vec_elems = sample_elems / 8;
        for (int i = threadIdx.x; i < vec_elems; i += 256)
            dst4[i] = src4[i];
    } else if (sample_elems % 2 == 0) {
        const half2 *src2 = (const half2*)src;
        half2 *dst2 = (half2*)dst;
        int vec_elems = sample_elems / 2;
        for (int i = threadIdx.x; i < vec_elems; i += 256)
            dst2[i] = src2[i];
    } else {
        for (int i = threadIdx.x; i < sample_elems; i += 256)
            dst[i] = src[i];
    }
}

__global__ void k_resident_advance_lr(float * __restrict__ d_lr,
                                      uint32_t * __restrict__ batch_idx,
                                      uint32_t * __restrict__ global_step,
                                      const TkmSchedule * __restrict__ sched) {
    if (threadIdx.x != 0) return;
    uint32_t step = *global_step;
    float warmup = (sched->warmup_steps > 0 && step < sched->warmup_steps)
        ? (float)(step + 1u) / (float)sched->warmup_steps : 1.0f;
    float progress = sched->total_steps > 1
        ? fminf((float)step / (float)(sched->total_steps - 1u), 1.0f) : 1.0f;
    float cosine = 0.5f * (1.0f + cosf(3.14159265358979323846f * progress));
    *d_lr = sched->peak_lr * warmup *
        (sched->min_lr_ratio + (1.0f - sched->min_lr_ratio) * cosine);
    *batch_idx = *batch_idx + 1u;
    *global_step = step + 1u;
}

__global__ void k_perturb_f16(__half * __restrict__ ptr, int idx, float delta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = __half2float(ptr[idx]);
        ptr[idx] = __float2half(v + delta);
    }
}

__global__ __launch_bounds__(256, 4)
void k_mixpool2x2_fwd_h2(const __half2 * __restrict__ in, __half2 * __restrict__ out,
                         uint16_t * __restrict__ mask, int N, int H_out, int W_out, int C2, float ratio) {
    int total = N * H_out * W_out * C2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c2 = idx % C2, rem = idx / C2;
    int ox = rem % W_out, tmp = rem / W_out;
    int oy = tmp % H_out, n = tmp / H_out;
    int H_in = H_out * 2, W_in = W_out * 2, ix = ox * 2, iy = oy * 2;

    const __half2 *base = in + ((size_t)n * H_in * W_in) * C2 + c2;
    float2 v0 = __half22float2(__ldg(&base[(iy * W_in + ix) * C2]));
    float2 v1 = __half22float2(__ldg(&base[(iy * W_in + ix + 1) * C2]));
    float2 v2 = __half22float2(__ldg(&base[((iy + 1) * W_in + ix) * C2]));
    float2 v3 = __half22float2(__ldg(&base[((iy + 1) * W_in + ix + 1) * C2]));

    float bx = v0.x; uint8_t mx = 0;
    if (v1.x > bx) { bx = v1.x; mx = 1; }
    if (v2.x > bx) { bx = v2.x; mx = 2; }
    if (v3.x > bx) { bx = v3.x; mx = 3; }

    float by = v0.y; uint8_t my = 0;
    if (v1.y > by) { by = v1.y; my = 1; }
    if (v2.y > by) { by = v2.y; my = 2; }
    if (v3.y > by) { by = v3.y; my = 3; }

    float aw = (1.0f - ratio) * 0.25f;
    float2 res = make_float2(ratio * bx + aw * (v0.x + v1.x + v2.x + v3.x),
                             ratio * by + aw * (v0.y + v1.y + v2.y + v3.y));
    out[idx]  = __float22half2_rn(res);
    if (mask) mask[idx] = (uint16_t)mx | ((uint16_t)my << 8);
}

__global__ __launch_bounds__(256, 4)
void k_mixpool2x2_bwd_h2(const __half2 * __restrict__ dout, const uint16_t * __restrict__ mask,
                         __half2 * __restrict__ din, int N, int H_out, int W_out, int C2, float ratio) {
    int total = N * H_out * W_out * C2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c2 = idx % C2, rem = idx / C2;
    int ox = rem % W_out, tmp = rem / W_out;
    int oy = tmp % H_out, n = tmp / H_out;
    int H_in = H_out * 2, W_in = W_out * 2, ix = ox * 2, iy = oy * 2;

    float2 g = __half22float2(__ldg(&dout[idx]));
    uint16_t m = mask ? __ldg(&mask[idx]) : 0;
    uint8_t mx = m & 0xFF, my = m >> 8;

    float aw = (1.0f - ratio) * 0.25f;
    float pw = ratio + aw;
    __half2 *base = din + ((size_t)n * H_in * W_in) * C2 + c2;
    int o0 = (iy * W_in + ix) * C2, o1 = (iy * W_in + ix + 1) * C2;
    int o2 = ((iy + 1) * W_in + ix) * C2, o3 = ((iy + 1) * W_in + ix + 1) * C2;
    float2 p0 = __half22float2(base[o0]), p1 = __half22float2(base[o1]);
    float2 p2 = __half22float2(base[o2]), p3 = __half22float2(base[o3]);
    base[o0] = __floats2half2_rn(p0.x + g.x * (mx == 0 ? pw : aw), p0.y + g.y * (my == 0 ? pw : aw));
    base[o1] = __floats2half2_rn(p1.x + g.x * (mx == 1 ? pw : aw), p1.y + g.y * (my == 1 ? pw : aw));
    base[o2] = __floats2half2_rn(p2.x + g.x * (mx == 2 ? pw : aw), p2.y + g.y * (my == 2 ? pw : aw));
    base[o3] = __floats2half2_rn(p3.x + g.x * (mx == 3 ? pw : aw), p3.y + g.y * (my == 3 ? pw : aw));
}

static __device__ __forceinline__ int conv_weight_index(int k, int ci, int co, int in_c, int out_c) {
    return (k * in_c + ci) * out_c + co;
}

__global__ __launch_bounds__(256, 4)
void k_conv_direct_small_cin_fwd_h(const __half * __restrict__ in, const __half * __restrict__ W,
                                   __half * __restrict__ out, int N, int H, int W_dim,
                                   int in_c, int out_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W_dim * out_c;
    if (idx >= total) return;
    int co = idx % out_c, rem = idx / out_c;
    int x = rem % W_dim, t = rem / W_dim, y = t % H, n = t / H;
    if (in_c != 1) return;
    float sum = 0.0f;
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        int iy = y + ky;
        if (iy < 0 || iy >= H) continue;
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            int ix = x + kx;
            if (ix < 0 || ix >= W_dim) continue;
            int k = (ky + 1) * 3 + (kx + 1);
            float xv = __half2float(__ldg(&in[((size_t)n * H * W_dim + iy * W_dim + ix)]));
            float w = __half2float(__ldg(&W[conv_weight_index(k, 0, co, 1, out_c)]));
            sum = fmaf(xv, w, sum);
        }
    }
    out[((size_t)n * H * W_dim + y * W_dim + x) * out_c + co] = __float2half(sum);
}

__global__ __launch_bounds__(256, 4)
void k_conv_direct_small_cin_dX_h(const __half * __restrict__ dout, const __half * __restrict__ W,
                                  __half * __restrict__ din, int N, int H, int W_dim,
                                  int in_c, int out_c) {
    int total = N * H * W_dim * in_c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ci = idx % in_c, q = idx / in_c, x = q % W_dim, t = q / W_dim, y = t % H, n = t / H;
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ky++) {
        int oy = y - ky;
        if (oy < 0 || oy >= H) continue;
        for (int kx = -1; kx <= 1; kx++) {
            int ox = x - kx;
            if (ox < 0 || ox >= W_dim) continue;
            int k = (ky + 1) * 3 + (kx + 1);
            const __half *dy = dout + ((size_t)n * H * W_dim + oy * W_dim + ox) * out_c;
            for (int co = 0; co < out_c; co++)
                sum = fmaf(__half2float(__ldg(&dy[co])),
                           __half2float(__ldg(&W[conv_weight_index(k, ci, co, in_c, out_c)])), sum);
        }
    }
    din[idx] = __float2half(__half2float(din[idx]) + sum);
}

__global__ __launch_bounds__(256, 4)
void k_conv_direct_small_cin_dW_h(const __half * __restrict__ in, const __half * __restrict__ dout,
                                  float * __restrict__ dW, int N, int H, int W_dim,
                                  int in_c, int out_c) {
    int tiles_x = (W_dim + 15) / 16;
    int tiles_y = (H + 15) / 16;
    int spatial = (int)blockIdx.x;
    int n = spatial / (tiles_y * tiles_x);
    if (n >= N) return;
    int rem = spatial % (tiles_y * tiles_x);
    int by = rem / tiles_x, bx = rem % tiles_x;
    int tid = threadIdx.x, c_local = tid & 31, sub = tid >> 5;
    int co = (int)blockIdx.y * 32 + c_local;
    __shared__ float s_in[18][18][4];
    __shared__ float s_dw[8][36][33];
    for (int i = tid; i < 18 * 18 * in_c; i += blockDim.x) {
        int ci = i % in_c, q = i / in_c;
        int sy = q / 18, sx = q % 18;
        int gy = by * 16 + sy - 1, gx = bx * 16 + sx - 1;
        s_in[sy][sx][ci] = (gy >= 0 && gy < H && gx >= 0 && gx < W_dim)
            ? __half2float(__ldg(&in[((size_t)n * H * W_dim + gy * W_dim + gx) * in_c + ci]))
            : 0.0f;
    }
    __syncthreads();
    if (co >= out_c) return;
    float dw[9][4] = {};
    for (int p = sub; p < 256; p += 8) {
        int ly = p / 16, lx = p % 16;
        int y = by * 16 + ly, x = bx * 16 + lx;
        if (y >= H || x >= W_dim) continue;
        float dy = __half2float(__ldg(&dout[((size_t)n * H * W_dim + y * W_dim + x) * out_c + co]));
        #pragma unroll
        for (int k = 0; k < 9; k++) {
            int ky = k / 3 - 1, kx = k % 3 - 1;
            for (int ci = 0; ci < in_c; ci++)
                dw[k][ci] = fmaf(s_in[ly + ky + 1][lx + kx + 1][ci], dy, dw[k][ci]);
        }
    }
    for (int k = 0; k < 9; k++)
        for (int ci = 0; ci < in_c; ci++)
            s_dw[sub][k * 4 + ci][c_local] = dw[k][ci];
    __syncthreads();
    if (sub == 0) {
        for (int k = 0; k < 9; k++)
            for (int ci = 0; ci < in_c; ci++) {
                float sum = 0.0f;
                for (int q = 0; q < 8; q++)
                    sum += s_dw[q][k * 4 + ci][c_local];
                atomicAdd(&dW[((size_t)k * in_c + ci) * out_c + co], sum);
            }
    }
}

static __device__ __forceinline__ void conv_decode_row(int row, int H, int W, int *n, int *y, int *x) {
    *x = row % W;
    int q = row / W;
    *y = q % H;
    *n = q / H;
}

template<int BN>
__global__ void k_conv3x3_wmma_fwd_h(const __half * __restrict__ in, const __half * __restrict__ W,
                                     __half * __restrict__ out, int rows, int H, int W_dim,
                                     int in_c, int out_c) {
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row0 = (int)blockIdx.x * 64;
    int co0 = (int)blockIdx.y * BN;
    __shared__ int s_n[64], s_y[64], s_x[64];
    __shared__ __half s_a[64][24];
    __shared__ __half s_b[2][16][24];
    __shared__ float s_out[64][32];
    for (int j = threadIdx.x; j < 64; j += blockDim.x)
        conv_decode_row(row0 + j, H, W_dim, &s_n[j], &s_y[j], &s_x[j]);
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    for (int ch = 0; ch < (BN / 16); ch++) wmma::fill_fragment(acc[ch], 0.0f);
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            int k = (ky + 1) * 3 + (kx + 1);
            for (int ci0 = 0; ci0 < in_c; ci0 += 16) {
                for (int j = threadIdx.x; j < 64 * 16; j += blockDim.x) {
                    int r = j / 16, c = j % 16;
                    int gy = s_y[r] + ky, gx = s_x[r] + kx;
                    s_a[r][c] = (row0 + r < rows && gy >= 0 && gy < H && gx >= 0 && gx < W_dim)
                        ? __ldg(&in[((size_t)s_n[r] * H * W_dim + gy * W_dim + gx) * in_c + ci0 + c])
                        : __float2half(0.0f);
                }
                for (int ch = 0; ch < (BN / 16); ch++) {
                    if (warp == ch) {
                        int cbase = co0 + ch * 16;
                        for (int j = lane; j < 256; j += 32) {
                            int r = j / 16, c = j % 16;
                            s_b[ch][r][c] = (cbase + c < out_c)
                                ? __ldg(&W[((k * in_c + ci0 + r) * out_c) + cbase + c])
                                : __float2half(0.0f);
                        }
                    }
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
                int rr = warp * 16;
                wmma::load_matrix_sync(fa, &s_a[rr][0], 24);
                for (int ch = 0; ch < (BN / 16); ch++) {
                    wmma::load_matrix_sync(fb, &s_b[ch][0][0], 24);
                    wmma::mma_sync(acc[ch], fa, fb, acc[ch]);
                }
                __syncthreads();
            }
        }
    int rr = warp * 16;
    for (int ch = 0; ch < (BN / 16); ch++)
        wmma::store_matrix_sync(&s_out[rr][ch * 16], acc[ch], 32, wmma::mem_row_major);
    __syncthreads();
    for (int j = threadIdx.x; j < 64 * BN; j += blockDim.x) {
        int r = j / BN, c = j % BN;
        if (row0 + r < rows && co0 + c < out_c) {
            out[((size_t)s_n[r] * H * W_dim + s_y[r] * W_dim + s_x[r]) * out_c + co0 + c] =
                __float2half(s_out[r][c]);
        }
    }
}

__global__ void k_conv3x3_wmma_dX_h(const __half * __restrict__ dout, const __half * __restrict__ W,
                                    __half * __restrict__ din, int rows, int H, int W_dim,
                                    int in_c, int out_c) {
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int row0 = (int)blockIdx.x * 64;
    int ci0 = (int)blockIdx.y * 32;
    int BN = (in_c - ci0 >= 32) ? 32 : (in_c - ci0);
    __shared__ int s_n[64], s_y[64], s_x[64];
    __shared__ __half s_a[64][24];
    __shared__ __half s_b[2][16][24];
    __shared__ float s_din[64][32];
    for (int j = threadIdx.x; j < 64; j += blockDim.x)
        conv_decode_row(row0 + j, H, W_dim, &s_n[j], &s_y[j], &s_x[j]);
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    for (int ch = 0; ch < (BN + 15) / 16; ch++) wmma::fill_fragment(acc[ch], 0.0f);
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            int k = (ky + 1) * 3 + (kx + 1);
            for (int co0 = 0; co0 < out_c; co0 += 16) {
                for (int j = threadIdx.x; j < 64 * 16; j += blockDim.x) {
                    int r = j / 16, c = j % 16;
                    int oy = s_y[r] - ky, ox = s_x[r] - kx;
                    s_a[r][c] = (row0 + r < rows && oy >= 0 && oy < H && ox >= 0 && ox < W_dim)
                        ? dout[((size_t)s_n[r] * H * W_dim + oy * W_dim + ox) * out_c + co0 + c]
                        : __float2half(0.0f);
                }
                int ch_count = (BN + 15) / 16;
                for (int ch = 0; ch < ch_count; ch++) {
                    if (warp == ch) {
                        int cb = ci0 + ch * 16;
                        for (int j = lane; j < 256; j += 32) {
                            int r = j / 16, c = j % 16;
                            s_b[ch][c][r] = (cb + r < in_c)
                                ? __ldg(&W[((k * in_c + cb + r) * out_c) + co0 + c])
                                : __float2half(0.0f);
                        }
                    }
                }
                __syncthreads();
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                int rr = warp * 16;
                wmma::load_matrix_sync(fa, &s_a[rr][0], 24);
                for (int ch = 0; ch < ch_count; ch++) {
                    wmma::load_matrix_sync(fb, &s_b[ch][0][0], 24);
                    wmma::mma_sync(acc[ch], fa, fb, acc[ch]);
                }
                __syncthreads();
            }
        }
    int rr = warp * 16;
    int ch_count = (BN + 15) / 16;
    for (int ch = 0; ch < ch_count; ch++)
        wmma::store_matrix_sync(&s_din[rr][ch * 16], acc[ch], 32, wmma::mem_row_major);
    __syncthreads();
    for (int j = threadIdx.x; j < 64 * BN; j += blockDim.x) {
        int r = j / BN, c = j % BN;
        if (row0 + r < rows && ci0 + c < in_c) {
            size_t idx = ((size_t)s_n[r] * H * W_dim + s_y[r] * W_dim + s_x[r]) * in_c + ci0 + c;
            din[idx] = __float2half(__half2float(din[idx]) + s_din[r][c]);
        }
    }
}

__global__ void k_conv3x3_wmma_dW_h(const __half * __restrict__ in, const __half * __restrict__ dout,
                                    float * __restrict__ dW, int rows, int H, int W_dim,
                                    int in_c, int out_c, int split_k) {
    int tile = (int)blockIdx.x;
    int ci0 = (tile % (in_c / 16)) * 16, co0 = (tile / (in_c / 16)) * 16;
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int k = (int)blockIdx.y, split = (int)blockIdx.z;
    int ky = k / 3 - 1, kx = k % 3 - 1;
    int start = (rows * split) / split_k, end = (rows * (split + 1)) / split_k;
    __shared__ int s_n[64], s_y[64], s_x[64];
    __shared__ __half s_a[4][16][24], s_b[4][16][24];
    __shared__ float s_dw[4][16][16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int row = start; row < end; row += 64) {
        if (threadIdx.x < 64)
            conv_decode_row(row + threadIdx.x, H, W_dim,
                            &s_n[threadIdx.x], &s_y[threadIdx.x], &s_x[threadIdx.x]);
        __syncthreads();
        for (int j = lane; j < 256; j += 32) {
            int r = j / 16, c = j % 16;
            int q = warp * 16 + r;
            int iy = s_y[q] + ky, ix = s_x[q] + kx;
            s_a[warp][r][c] = (row + q < end && row + q < rows && iy >= 0 && iy < H && ix >= 0 && ix < W_dim)
                ? __ldg(&in[((size_t)s_n[q] * H * W_dim + iy * W_dim + ix) * in_c + ci0 + c])
                : __float2half(0.0f);
            s_b[warp][r][c] = (row + q < end && row + q < rows)
                ? __ldg(&dout[((size_t)s_n[q] * H * W_dim + s_y[q] * W_dim + s_x[q]) * out_c + co0 + c])
                : __float2half(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, &s_a[warp][0][0], 24);
        wmma::load_matrix_sync(fb, &s_b[warp][0][0], 24);
        wmma::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(&s_dw[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncthreads();
    if (threadIdx.x < 32) {
        for (int j = lane; j < 256; j += 32) {
            int r = j / 16, c = j % 16;
            float sum = s_dw[0][r][c] + s_dw[1][r][c] + s_dw[2][r][c] + s_dw[3][r][c];
            if (ci0 + r < in_c && co0 + c < out_c)
                atomicAdd(&dW[((size_t)k * in_c + ci0 + r) * out_c + co0 + c], sum);
        }
    }
}

static int conv_uses_direct(int in_c, int k, int s, int p) {
    return k == 3 && s == 1 && p == 1 && in_c >= 1 && in_c <= 4;
}

static int conv_uses_wmma(int in_c, int out_c, int k, int s, int p) {
    return k == 3 && s == 1 && p == 1 && (in_c % 16 == 0) && (out_c % 16 == 0);
}

#define CONV_MODE_1X1 1
#define CONV_MODE_DIRECT_3X3 2
#define CONV_MODE_WMMA_3X3 3
#define CONV_MODE_IM2COL 4

static int conv_select_mode(const TkmNode *n, const TkmNode *in0) {
    if (n->k == 1 && n->s == 1 && n->p == 0) return CONV_MODE_1X1;
    if (n->k == 3 && n->s == 1 && n->p == 1 && in0->C == 1)
        return CONV_MODE_DIRECT_3X3;
    return CONV_MODE_IM2COL;
}

__global__ __launch_bounds__(256, 4)
void k_im2col_3x3_h(const __half * __restrict__ in, __half * __restrict__ col, int N, int H, int W, int C) {
    int total = N * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C, rem = idx / C, x = rem % W, tmp = rem / W, y = tmp % H, n = tmp / H;
    int row = (n * H + y) * W + x, stride = 9 * C;
    const __half *src_n = in + (size_t)n * H * W * C + c;
    __half *dst_r = col + (size_t)row * stride + c;
    int k = 0;
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        int yy = y + ky;
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            int xx = x + kx;
            dst_r[k * C] = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? __ldg(&src_n[(yy * W + xx) * C]) : __float2half(0.0f);
            k++;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_col2im_3x3_h(const __half * __restrict__ col, __half * __restrict__ din, int N, int H, int W, int C) {
    int total = N * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C, rem = idx / C, x = rem % W, tmp = rem / W, y = tmp % H, n = tmp / H;
    int stride = 9 * C;
    float sum = __half2float(din[idx]);
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        int oy = y - ky;
        if (oy >= 0 && oy < H) {
            #pragma unroll
            for (int kx = -1; kx <= 1; kx++) {
                int ox = x - kx;
                if (ox >= 0 && ox < W) {
                    int k = (ky + 1) * 3 + (kx + 1);
                    int orow = (n * H + oy) * W + ox;
                    sum += __half2float(__ldg(&col[(size_t)orow * stride + k * C + c]));
                }
            }
        }
    }
    din[idx] = __float2half(sum);
}

__global__ __launch_bounds__(256, 4)
void k_im2col_nhwc_h(const __half * __restrict__ in, __half * __restrict__ col,
                     int N, int H_in, int W_in, int H_out, int W_out,
                     int C, int k, int s, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (C % 2 == 0) {
        int C2 = C / 2;
        int total = N * H_out * W_out * C2;
        if (idx >= total) return;
        int c2 = idx % C2, rem = idx / C2;
        int ox = rem % W_out, tmp = rem / W_out;
        int oy = tmp % H_out, n = tmp / H_out;
        int row = (n * H_out + oy) * W_out + ox;
        int Kdim = k * k * C;
        half2 *dst = (half2*)(col + (size_t)row * Kdim);
        for (int ky = 0; ky < k; ky++) {
            int iy = oy * s - p + ky;
            for (int kx = 0; kx < k; kx++) {
                int ix = ox * s - p + kx;
                dst[(ky * k + kx) * C2 + c2] =
                    (iy >= 0 && iy < H_in && ix >= 0 && ix < W_in)
                    ? ((const half2*)in)[((size_t)n * H_in * W_in + iy * W_in + ix) * C2 + c2]
                    : __float2half2_rn(0.0f);
            }
        }
        return;
    }
    int total_pairs = N * H_out * W_out * C;
    if (idx >= total_pairs) return;
    int c = idx % C, rem = idx / C;
    int ox = rem % W_out, tmp = rem / W_out;
    int oy = tmp % H_out, n = tmp / H_out;
    int row = (n * H_out + oy) * W_out + ox;
    int Kdim = k * k * C;
    __half *dst = col + (size_t)row * Kdim + c;
    for (int ky = 0; ky < k; ky++) {
        int iy = oy * s - p + ky;
        for (int kx = 0; kx < k; kx++) {
            int ix = ox * s - p + kx;
            *dst = (iy >= 0 && iy < H_in && ix >= 0 && ix < W_in)
                ? __ldg(&in[((size_t)n * H_in * W_in + iy * W_in + ix) * C + c])
                : __float2half(0.0f);
            dst += C;
        }
    }
}

__global__ __launch_bounds__(256, 4)
void k_col2im_nhwc_h(const __half * __restrict__ col, __half * __restrict__ din,
                     int N, int H_in, int W_in, int H_out, int W_out,
                     int C, int k, int s, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (C % 2 == 0) {
        int C2 = C / 2;
        int total = N * H_in * W_in * C2;
        if (idx >= total) return;
        int c2 = idx % C2, rem = idx / C2;
        int ix = rem % W_in, tmp = rem / W_in;
        int iy = tmp % H_in, n = tmp / H_in;
        int Kdim = k * k * C;
        half2 *dp = (half2*)din;
        half2 *dpp = dp + ((size_t)n * H_in * W_in + iy * W_in + ix) * C2 + c2;
        float2 sum = __half22float2(*dpp);
        for (int ky = 0; ky < k; ky++) {
            int oy_num = iy + p - ky;
            if (oy_num < 0) continue;
            for (int kx = 0; kx < k; kx++) {
                int ox_num = ix + p - kx;
                if (ox_num >= 0 && (oy_num % s) == 0 && (ox_num % s) == 0) {
                    int oy = oy_num / s, ox = ox_num / s;
                    if (oy < H_out && ox < W_out) {
                        int row = (n * H_out + oy) * W_out + ox;
                        int kidx = ((ky * k + kx) * C2 + c2);
                        float2 v = __half22float2(((const half2*)col)[(size_t)row * (Kdim / 2) + kidx]);
                        sum.x += v.x;
                        sum.y += v.y;
                    }
                }
            }
        }
        *dpp = __floats2half2_rn(sum.x, sum.y);
        return;
    }
    int total = N * H_in * W_in * C;
    if (idx >= total) return;
    int c = idx % C, rem = idx / C;
    int ix = rem % W_in, tmp = rem / W_in;
    int iy = tmp % H_in, n = tmp / H_in;
    int Kdim = k * k * C;
    float sum = __half2float(din[idx]);
    for (int ky = 0; ky < k; ky++) {
        int oy_num = iy + p - ky;
        if (oy_num >= 0) {
            for (int kx = 0; kx < k; kx++) {
                int ox_num = ix + p - kx;
                if (ox_num >= 0 && (oy_num % s) == 0 && (ox_num % s) == 0) {
                    int oy = oy_num / s, ox = ox_num / s;
                    if (oy < H_out && ox < W_out) {
                        int row = (n * H_out + oy) * W_out + ox;
                        int kidx = (ky * k + kx) * C + c;
                        sum += __half2float(__ldg(&col[(size_t)row * Kdim + kidx]));
                    }
                }
            }
        }
    }
    din[idx] = __float2half(sum);
}

#define SMEM_PAD_16 24

template<int HEAD_DIM>
__global__ __launch_bounds__(64, 4)
void k_gqa_flash_attn_tc_fwd(const __half * __restrict__ qkv, __half * __restrict__ out,
                             float * __restrict__ save_lse,
                             int B, int Seq, int Q_H, int KV_H, float scale, int causal) {
    constexpr int HEAD_PAD = HEAD_DIM + 8;
    int q_h     = blockIdx.x % Q_H;
    int b       = blockIdx.x / Q_H;
    int q_blk   = blockIdx.y;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int tid     = threadIdx.x;

    int group_g = Q_H / KV_H;
    int kv_h    = q_h / group_g;
    int q_base  = q_blk * 32 + warp_id * 16;

    int total_h    = Q_H + 2 * KV_H;
    int qkv_stride = total_h * HEAD_DIM;
    int out_stride = Q_H * HEAD_DIM;

    __shared__ __half s_q[32][HEAD_PAD];
    __shared__ __half s_k[16][HEAD_PAD];
    __shared__ __half s_v[16][HEAD_PAD];
    __shared__ float  s_scores[2][16][SMEM_PAD_16];
    __shared__ __half s_p[2][16][SMEM_PAD_16];
    __shared__ float  s_out[2][16][HEAD_PAD];

    const __half *b_qkv = qkv + (size_t)b * Seq * qkv_stride;
    __half       *b_out = out + (size_t)b * Seq * out_stride;

    for (int i = tid * 2; i < 32 * HEAD_DIM; i += 128) {
        int r = i / HEAD_DIM, c = i % HEAD_DIM;
        int g_r = q_blk * 32 + r;
        if (g_r < Seq) {
            *reinterpret_cast<__half2*>(&s_q[r][c]) =
                *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + q_h * HEAD_DIM + c]);
        } else {
            *reinterpret_cast<__half2*>(&s_q[r][c]) = __float2half2_rn(0.f);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_q[HEAD_DIM / 16];
    #pragma unroll
    for (int ks = 0; ks < HEAD_DIM / 16; ks++) {
        wmma::load_matrix_sync(frag_q[ks], &s_q[warp_id * 16][ks * 16], HEAD_PAD);
    }

    #pragma unroll
    for (int r = 0; r < 16; r++) {
        for (int c = lane * 2; c < HEAD_DIM; c += 64) {
            s_out[warp_id][r][c]     = 0.0f;
            s_out[warp_id][r][c + 1] = 0.0f;
        }
    }
    float m_row = -1e30f, l_row = 0.0f;

    for (int kv_base = 0; kv_base < Seq; kv_base += 16) {
        for (int i = tid * 2; i < 16 * HEAD_DIM; i += 128) {
            int r = i / HEAD_DIM, c = i % HEAD_DIM;
            int g_r = kv_base + r;
            if (g_r < Seq) {
                *reinterpret_cast<__half2*>(&s_k[r][c]) =
                    *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + (Q_H + kv_h) * HEAD_DIM + c]);
                *reinterpret_cast<__half2*>(&s_v[r][c]) =
                    *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + (Q_H + KV_H + kv_h) * HEAD_DIM + c]);
            } else {
                *reinterpret_cast<__half2*>(&s_k[r][c]) = __float2half2_rn(0.f);
                *reinterpret_cast<__half2*>(&s_v[r][c]) = __float2half2_rn(0.f);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s;
        wmma::fill_fragment(frag_s, 0.0f);
        #pragma unroll
        for (int ks = 0; ks < HEAD_DIM / 16; ks++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> frag_k;
            wmma::load_matrix_sync(frag_k, &s_k[0][ks * 16], HEAD_PAD);
            wmma::mma_sync(frag_s, frag_q[ks], frag_k, frag_s);
        }
        wmma::store_matrix_sync(&s_scores[warp_id][0][0], frag_s, SMEM_PAD_16, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16) {
            float row_max = m_row;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                int key_pos = kv_base + c;
                float v = s_scores[warp_id][lane][c] * scale;
                if (key_pos >= Seq || (causal && key_pos > q_base + lane)) v = -1e30f;
                s_scores[warp_id][lane][c] = v;
                if (v > row_max) row_max = v;
            }
            float alpha = __expf(m_row - row_max);
            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                float p = __expf(s_scores[warp_id][lane][c] - row_max);
                s_p[warp_id][lane][c] = __float2half(p);
                row_sum += p;
            }
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                s_out[warp_id][lane][d] *= alpha;
            }
            l_row = l_row * alpha + row_sum;
            m_row = row_max;
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_p;
        wmma::load_matrix_sync(frag_p, &s_p[warp_id][0][0], SMEM_PAD_16);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM / 16; d++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_o;
            wmma::load_matrix_sync(frag_o, &s_out[warp_id][0][d * 16], HEAD_PAD, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_v;
            wmma::load_matrix_sync(frag_v, &s_v[0][d * 16], HEAD_PAD);
            wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
            wmma::store_matrix_sync(&s_out[warp_id][0][d * 16], frag_o, HEAD_PAD, wmma::mem_row_major);
        }
        __syncthreads();
    }

    if (lane < 16 && (q_base + lane) < Seq) {
        float inv_l = 1.0f / (l_row + 1e-8f);
        if (save_lse) save_lse[(b * Q_H + q_h) * Seq + q_base + lane] = m_row + __logf(l_row + 1e-8f);
        __half *dst_row = b_out + (q_base + lane) * out_stride + q_h * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            *reinterpret_cast<__half2*>(&dst_row[d]) =
                __floats2half2_rn(s_out[warp_id][lane][d] * inv_l, s_out[warp_id][lane][d + 1] * inv_l);
        }
    }
}

template<int HEAD_DIM>
__global__ __launch_bounds__(64, 4)
void k_gqa_flash_attn_tc_bwd(const __half * __restrict__ qkv, const __half * __restrict__ out,
                             const __half * __restrict__ dout, const float * __restrict__ save_lse,
                             __half * __restrict__ dqkv,
                             int B, int Seq, int Q_H, int KV_H, float scale, int causal) {
    constexpr int HEAD_PAD = HEAD_DIM + 8;
    int q_h     = blockIdx.x % Q_H;
    int b       = blockIdx.x / Q_H;
    int q_blk   = blockIdx.y;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int tid     = threadIdx.x;

    int group_g = Q_H / KV_H;
    int kv_h    = q_h / group_g;
    int q_base  = q_blk * 32 + warp_id * 16;

    int total_h    = Q_H + 2 * KV_H;
    int qkv_stride = total_h * HEAD_DIM;
    int out_stride = Q_H * HEAD_DIM;

    __shared__ __half s_q[32][HEAD_PAD];
    __shared__ __half s_do[32][HEAD_PAD];
    __shared__ __half s_k[16][HEAD_PAD];
    __shared__ __half s_v[16][HEAD_PAD];
    __shared__ float  s_s[2][16][SMEM_PAD_16];
    __shared__ float  s_dp[2][16][SMEM_PAD_16];
    __shared__ __half s_ds[2][16][SMEM_PAD_16];
    __shared__ __half s_dq_h[2][16][HEAD_PAD];

    const __half *b_qkv  = qkv  + (size_t)b * Seq * qkv_stride;
    const __half *b_out  = out  + (size_t)b * Seq * out_stride;
    const __half *b_dout = dout + (size_t)b * Seq * out_stride;
    __half       *b_dqkv = dqkv + (size_t)b * Seq * qkv_stride;

    for (int i = tid * 2; i < 32 * HEAD_DIM; i += 128) {
        int r = i / HEAD_DIM, c = i % HEAD_DIM;
        int g_r = q_blk * 32 + r;
        if (g_r < Seq) {
            *reinterpret_cast<__half2*>(&s_q[r][c]) =
                *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + q_h * HEAD_DIM + c]);
            *reinterpret_cast<__half2*>(&s_do[r][c]) =
                *reinterpret_cast<const __half2*>(&b_dout[g_r * out_stride + q_h * HEAD_DIM + c]);
        } else {
            *reinterpret_cast<__half2*>(&s_q[r][c])  = __float2half2_rn(0.f);
            *reinterpret_cast<__half2*>(&s_do[r][c]) = __float2half2_rn(0.f);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_q[HEAD_DIM / 16];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_do[HEAD_DIM / 16];
    #pragma unroll
    for (int ks = 0; ks < HEAD_DIM / 16; ks++) {
        wmma::load_matrix_sync(frag_q[ks],  &s_q[warp_id * 16][ks * 16],  HEAD_PAD);
        wmma::load_matrix_sync(frag_do[ks], &s_do[warp_id * 16][ks * 16], HEAD_PAD);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_dq[HEAD_DIM / 16];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; d++) wmma::fill_fragment(frag_dq[d], 0.0f);

    float lse_row = 0.0f, D_row = 0.0f;
    if (lane < 16 && (q_base + lane) < Seq) {
        lse_row = save_lse[(b * Q_H + q_h) * Seq + q_base + lane];
        const __half *o_ptr  = b_out  + (q_base + lane) * out_stride + q_h * HEAD_DIM;
        const __half *do_ptr = b_dout + (q_base + lane) * out_stride + q_h * HEAD_DIM;
        float sum_d = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) sum_d += __half2float(o_ptr[d]) * __half2float(do_ptr[d]);
        D_row = sum_d;
    }

    for (int kv_base = 0; kv_base < Seq; kv_base += 16) {
        for (int i = tid * 2; i < 16 * HEAD_DIM; i += 128) {
            int r = i / HEAD_DIM, c = i % HEAD_DIM;
            int g_r = kv_base + r;
            if (g_r < Seq) {
                *reinterpret_cast<__half2*>(&s_k[r][c]) =
                    *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + (Q_H + kv_h) * HEAD_DIM + c]);
                *reinterpret_cast<__half2*>(&s_v[r][c]) =
                    *reinterpret_cast<const __half2*>(&b_qkv[g_r * qkv_stride + (Q_H + KV_H + kv_h) * HEAD_DIM + c]);
            } else {
                *reinterpret_cast<__half2*>(&s_k[r][c]) = __float2half2_rn(0.f);
                *reinterpret_cast<__half2*>(&s_v[r][c]) = __float2half2_rn(0.f);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_s, frag_dp;
        wmma::fill_fragment(frag_s, 0.0f);
        wmma::fill_fragment(frag_dp, 0.0f);
        #pragma unroll
        for (int ks = 0; ks < HEAD_DIM / 16; ks++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> frag_kt, frag_vt;
            wmma::load_matrix_sync(frag_kt, &s_k[0][ks * 16], HEAD_PAD);
            wmma::load_matrix_sync(frag_vt, &s_v[0][ks * 16], HEAD_PAD);
            wmma::mma_sync(frag_s,  frag_q[ks],  frag_kt, frag_s);
            wmma::mma_sync(frag_dp, frag_do[ks], frag_vt, frag_dp);
        }
        wmma::store_matrix_sync(&s_s[warp_id][0][0],  frag_s,  SMEM_PAD_16, wmma::mem_row_major);
        wmma::store_matrix_sync(&s_dp[warp_id][0][0], frag_dp, SMEM_PAD_16, wmma::mem_row_major);
        __syncwarp();

        if (lane < 16) {
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                int key_pos = kv_base + c;
                int masked = key_pos >= Seq || (causal && key_pos > q_base + lane);
                float p_ij  = masked ? 0.0f : __expf(s_s[warp_id][lane][c] * scale - lse_row);
                float ds_ij = masked ? 0.0f : p_ij * (s_dp[warp_id][lane][c] - D_row) * scale;
                s_ds[warp_id][lane][c] = __float2half(ds_ij);
            }
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_ds;
        wmma::load_matrix_sync(frag_ds, &s_ds[warp_id][0][0], SMEM_PAD_16);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM / 16; d++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_k_row;
            wmma::load_matrix_sync(frag_k_row, &s_k[0][d * 16], HEAD_PAD);
            wmma::mma_sync(frag_dq[d], frag_ds, frag_k_row, frag_dq[d]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; d++) {
        wmma::store_matrix_sync(reinterpret_cast<float*>(&s_s[warp_id][0][0]), frag_dq[d], 16, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                s_dq_h[warp_id][lane][d * 16 + c] = __float2half(s_s[warp_id][lane][c]);
            }
        }
        __syncwarp();
    }

    if (lane < 16 && (q_base + lane) < Seq) {
        int r_idx = q_base + lane;
        __half *dq_row = b_dqkv + r_idx * qkv_stride + q_h * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            *reinterpret_cast<__half2*>(&dq_row[d]) = *reinterpret_cast<const __half2*>(&s_dq_h[warp_id][lane][d]);
        }
        if ((q_h % group_g) == 0) {
            __half *dk_row = b_dqkv + r_idx * qkv_stride + (Q_H + kv_h) * HEAD_DIM;
            __half *dv_row = b_dqkv + r_idx * qkv_stride + (Q_H + KV_H + kv_h) * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                *reinterpret_cast<__half2*>(&dk_row[d]) = *reinterpret_cast<__half2*>(&dq_row[d]);
                *reinterpret_cast<__half2*>(&dv_row[d]) = *reinterpret_cast<const __half2*>(&s_do[warp_id * 16 + lane][d]);
            }
        }
    }
}

__global__ void k_softmax_fwd_h(const __half * __restrict__ logits, __half * __restrict__ probs,
                                int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float mx = -1e30f;
    for (int c = 0; c < cols; c++) mx = fmaxf(mx, __half2float(logits[r * cols + c]));
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) sum += expf(__half2float(logits[r * cols + c]) - mx);
    for (int c = 0; c < cols; c++) {
        float p = expf(__half2float(logits[r * cols + c]) - mx) / sum;
        probs[r * cols + c] = __float2half(p);
    }
}

__global__ void k_softmax_bwd_h(const __half * __restrict__ probs, const __half * __restrict__ dprobs,
                                __half * __restrict__ dlogits, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float dot = 0.0f;
    for (int c = 0; c < cols; c++) {
        int idx = r * cols + c;
        dot += __half2float(probs[idx]) * __half2float(dprobs[idx]);
    }
    for (int c = 0; c < cols; c++) {
        int idx = r * cols + c;
        float p = __half2float(probs[idx]);
        float dx = p * (__half2float(dprobs[idx]) - dot);
        float prev = __half2float(dlogits[idx]);
        dlogits[idx] = __float2half(prev + dx);
    }
}

__global__ void k_softmax_loss_fwd_bwd_h(const __half * __restrict__ logits, const uint8_t * __restrict__ labels,
                                         __half * __restrict__ dlogits, float * __restrict__ loss_acc,
                                         int * __restrict__ corr_acc, int rows, int cols, int is_eval,
                                         int valid_rows, int loss_kind, float gamma) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float mx = -1e30f;
    int pred = 0;
    for (int c = 0; c < cols; c++) {
        float v = __half2float(logits[r * cols + c]);
        if (v > mx) { mx = v; pred = c; }
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) sum += expf(__half2float(logits[r * cols + c]) - mx);
    int lbl = labels ? labels[r] : 0;
    if (corr_acc && (!is_eval || r < valid_rows) && pred == lbl) atomicAdd(corr_acc, 1);
    if (is_eval) return;

    float inv_rows = 1.0f / (float)rows;
    float target_prob = expf(__half2float(logits[r * cols + lbl]) - mx) / sum;
    target_prob = fmaxf(target_prob, 1e-7f);
    float focal_weight = powf(1.0f - target_prob, gamma);
    float focal_grad_scale = loss_kind == 1
        ? focal_weight - gamma * target_prob * powf(1.0f - target_prob, gamma - 1.0f) * logf(target_prob)
        : 1.0f;
    float brier_dot = 0.0f;
    if (loss_kind == 2) {
        for (int c = 0; c < cols; c++) {
            float prob = expf(__half2float(logits[r * cols + c]) - mx) / sum;
            brier_dot += prob * (prob - (c == lbl ? 1.0f : 0.0f));
        }
    }
    for (int c = 0; c < cols; c++) {
        float prob = expf(__half2float(logits[r * cols + c]) - mx) / sum;
        float target = c == lbl ? 1.0f : 0.0f;
        float grad;
        if (loss_kind == 0) {
            if (c == lbl && loss_acc) atomicAdd(loss_acc, -logf(target_prob) * inv_rows);
            grad = prob - target;
        } else if (loss_kind == 1) {
            if (c == lbl && loss_acc) atomicAdd(loss_acc, -focal_weight * logf(target_prob) * inv_rows);
            grad = focal_grad_scale * (prob - target);
        } else {
            float diff = prob - target;
            if (loss_acc) atomicAdd(loss_acc, (diff * diff / (float)cols) * inv_rows);
            grad = (2.0f / (float)cols) * prob * (diff - brier_dot);
        }
        grad *= inv_rows;
        dlogits[r * cols + c] = __float2half(grad);
    }
}

struct SgdTensorList {
    int    count;
    float  *master[TKM_MAX_TENSORS];
    __half *fp16_w[TKM_MAX_TENSORS];
    float  *vel[TKM_MAX_TENSORS];
    const float *grad[TKM_MAX_TENSORS];
    int    offset[TKM_MAX_TENSORS + 1];
    int    size[TKM_MAX_TENSORS];
};
__constant__ SgdTensorList c_tkm_sgd;

__global__ __launch_bounds__(256, 4)
void k_tkm_multi_tensor_sgd(const float * __restrict__ d_lr, float momentum, float wd, uint32_t * __restrict__ d_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nt = c_tkm_sgd.count;
    float lr = *d_lr;
    if (idx < c_tkm_sgd.offset[nt]) {
        int t = 0;
        for (int k = 1; k < nt; k++) {
            if (idx >= c_tkm_sgd.offset[k]) t = k;
        }
        int local = idx - c_tkm_sgd.offset[t];
        if (local < c_tkm_sgd.size[t]) {
            float p = c_tkm_sgd.master[t][local];
            float g = c_tkm_sgd.grad[t][local];
            if (c_tkm_sgd.fp16_w[t] != NULL) g = fmaf(wd, p, g);
            float v = fmaf(momentum, c_tkm_sgd.vel[t][local], g);
            p = fmaf(-lr, fmaf(momentum, v, g), p);
            c_tkm_sgd.vel[t][local]    = v;
            c_tkm_sgd.master[t][local] = p;
            if (c_tkm_sgd.fp16_w[t] != NULL) {
                c_tkm_sgd.fp16_w[t][local] = __float2half(p);
            }
        }
    }
    if (idx == 0 && d_seed) *d_seed += 0x9E3779B9u;
}

static const char *NVRTC_STANDALONE_PRELUDE =
"typedef unsigned char  uint8_t;\n"
"typedef unsigned short uint16_t;\n"
"typedef unsigned int   uint32_t;\n"
"struct __align__(2) __half  { uint16_t x; };\n"
"struct __align__(4) __half2 { uint16_t x, y; };\n"
"__device__ __forceinline__ __half  __ldg(const __half  *p) { return *p; }\n"
"__device__ __forceinline__ __half2 __ldg(const __half2 *p) { return *p; }\n"
"__device__ __forceinline__ float __half2float(__half h) {\n"
"    float f; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h.x)); return f;\n"
"}\n"
"__device__ __forceinline__ __half __float2half(float f) {\n"
"    __half h; asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(h.x) : \"f\"(f)); return h;\n"
"}\n"
"__device__ __forceinline__ float2 __half22float2(__half2 h) {\n"
"    __half h0 = {h.x}, h1 = {h.y};\n"
"    return make_float2(__half2float(h0), __half2float(h1));\n"
"}\n"
"__device__ __forceinline__ __half2 __float22half2_rn(float2 f) {\n"
"    __half h0 = __float2half(f.x), h1 = __float2half(f.y);\n"
"    __half2 r = {h0.x, h1.x}; return r;\n"
"}\n"
"__device__ __forceinline__ __half2 __floats2half2_rn(float fx, float fy) {\n"
"    return __float22half2_rn(make_float2(fx, fy));\n"
"}\n"
"__device__ __forceinline__ float subwarp_reduce_sum(float v, int span) {\n"
"    for (int m = span; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);\n"
"    return v;\n"
"}\n"
"struct RngXorshift32 { uint32_t s; };\n"
"__device__ __forceinline__ void rng_init(RngXorshift32 *r, uint32_t seed, uint32_t idx) {\n"
"    uint32_t x = seed ^ (idx * 0x9E3779B9u + 0x7F4A7C15u); r->s = x ? x : 1u;\n"
"}\n"
"__device__ __forceinline__ uint32_t rng_u32(RngXorshift32 *r) {\n"
"    uint32_t x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return (r->s = x);\n"
"}\n"
"struct RngWangXorshift32 { uint32_t s; };\n"
"__device__ __forceinline__ void rng_init(RngWangXorshift32 *r, uint32_t seed, uint32_t idx) {\n"
"    uint32_t k = seed ^ idx; k = (~k) + (k << 18); k ^= (k >> 31);\n"
"    k = (k + (k << 2)) + (k << 4); k ^= (k >> 11); k += (k << 6); k ^= (k >> 22);\n"
"    r->s = k ? k : 1u;\n"
"}\n"
"__device__ __forceinline__ uint32_t rng_u32(RngWangXorshift32 *r) {\n"
"    uint32_t x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return (r->s = x);\n"
"}\n"
"struct RngPcg32 { unsigned long long state; unsigned long long inc; };\n"
"__device__ __forceinline__ uint32_t rng_u32(RngPcg32 *r) {\n"
"    unsigned long long old = r->state;\n"
"    r->state = old * 6364136223846793005ULL + r->inc;\n"
"    uint32_t xsh = (uint32_t)(((old >> 18u) ^ old) >> 27u);\n"
"    uint32_t rot = (uint32_t)(old >> 59u);\n"
"    return (xsh >> rot) | (xsh << ((32u - rot) & 31u));\n"
"}\n"
"__device__ __forceinline__ void rng_init(RngPcg32 *r, uint32_t seed, uint32_t idx) {\n"
"    r->state = 0ULL; r->inc = ((unsigned long long)idx << 1u) | 1ULL;\n"
"    rng_u32(r); r->state += (unsigned long long)seed ^ 0x853C49E6748FEA9BULL; rng_u32(r);\n"
"}\n"
"struct RngPhilox { uint32_t ctr, key, sub; };\n"
"__device__ __forceinline__ void rng_init(RngPhilox *r, uint32_t seed, uint32_t idx) {\n"
"    r->ctr = idx; r->key = seed; r->sub = 0u;\n"
"}\n"
"__device__ __forceinline__ uint32_t rng_u32(RngPhilox *r) {\n"
"    uint32_t c0 = r->ctr, c1 = r->sub++, k0 = r->key;\n"
"    #pragma unroll\n"
"    for (int i = 0; i < 7; i++) {\n"
"        uint32_t hi = __umulhi(0xD2511F53u, c0), lo = 0xD2511F53u * c0;\n"
"        c0 = hi ^ c1 ^ k0; c1 = lo; k0 += 0x9E3779B9u;\n"
"    }\n"
"    return c0;\n"
"}\n"
"template <typename RngT> __device__ __forceinline__ float rng_uniform(RngT *r) {\n"
"    return (float)(rng_u32(r) >> 8) * (1.0f / 16777216.0f);\n"
"}\n"
"template <typename RngT> __device__ __forceinline__ float2 rng_normal2(RngT *r) {\n"
"    float u1 = rng_uniform(r) + 1e-7f, u2 = rng_uniform(r), s, c;\n"
"    __sincosf(6.2831853f * u2, &s, &c);\n"
"    float mag = sqrtf(-2.0f * __logf(u1));\n"
"    return make_float2(mag * c, mag * s);\n"
"}\n"
"__device__ __forceinline__ float bilinear_sample_nhwc_h(const __half *src, float fx, float fy, int ch, int H, int W, int C) {\n"
"    int x0 = (int)floorf(fx), y0 = (int)floorf(fy), x1 = x0 + 1, y1 = y0 + 1;\n"
"    float ax = fx - x0, ay = fy - y0, v = 0.0f, w = 0.0f;\n"
"    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) { float wt = (1.f-ax)*(1.f-ay); v += wt * __half2float(src[(y0*W+x0)*C+ch]); w += wt; }\n"
"    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) { float wt = ax*(1.f-ay);       v += wt * __half2float(src[(y0*W+x1)*C+ch]); w += wt; }\n"
"    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) { float wt = (1.f-ax)*ay;       v += wt * __half2float(src[(y1*W+x0)*C+ch]); w += wt; }\n"
"    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) { float wt = ax*ay;             v += wt * __half2float(src[(y1*W+x1)*C+ch]); w += wt; }\n"
"    return w > 0.0f ? (v / w) : 0.0f;\n"
"}\n"
"__device__ __forceinline__ int tkm_reflect_idx(int i, int n) {\n"
"    if (i < 0) i = -i;\n"
"    if (i >= n) i = 2 * n - 2 - i;\n"
"    return i < 0 ? 0 : (i >= n ? n - 1 : i);\n"
"}\n"
"__device__ __forceinline__ float bilinear_sample_nhwc_reflect_h(const __half *src, float fx, float fy, int ch, int H, int W, int C) {\n"
"    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);\n"
"    float ax = fx - (float)x0, ay = fy - (float)y0;\n"
"    int rx0 = tkm_reflect_idx(x0, W), rx1 = tkm_reflect_idx(x0 + 1, W);\n"
"    int ry0 = tkm_reflect_idx(y0, H), ry1 = tkm_reflect_idx(y0 + 1, H);\n"
"    float v00 = __half2float(src[(ry0 * W + rx0) * C + ch]);\n"
"    float v10 = __half2float(src[(ry0 * W + rx1) * C + ch]);\n"
"    float v01 = __half2float(src[(ry1 * W + rx0) * C + ch]);\n"
"    float v11 = __half2float(src[(ry1 * W + rx1) * C + ch]);\n"
"    float top = fmaf(ax, v10 - v00, v00);\n"
"    float bot = fmaf(ax, v11 - v01, v01);\n"
"    return fmaf(ay, bot - top, top);\n"
"}\n\n";

struct NodeGpuState {
    int       in_dim, out_dim;
    int       conv_mode;
    float    *d_W_master, *d_vW, *d_dW;
    __half   *d_W_h;
    float    *d_bias,  *d_vbias,  *d_dbias;
    float    *d_w_vec, *d_vw_vec, *d_dw_vec;
    float    *d_beta,  *d_vbeta,  *d_dbeta;
    float    *d_save_istd;
    __half   *d_save_xhat;
    uint16_t *d_save_mask;
};

struct TkmDataset {
    TkmRuntime *rt;
    const float *h_images;
    const uint8_t *h_labels;
    size_t num_samples;
    size_t sample_elems;
    int resident;

    __half *d_images_h;
    uint8_t *d_labels_u8;
    uint32_t *d_order;
    uint32_t *d_batch_idx;
    uint32_t *d_global_step;
    TkmSchedule *d_sched;
    uint32_t *h_pin_order;

    float *h_pin_in[2];
    uint8_t *h_pin_lbl[2];
    float *h_pin_lr[2];
    cudaEvent_t slot_done[2];
    cudaGraph_t stream_graph[2];
    cudaGraphExec_t stream_exec[2];
    cudaGraph_t resident_graph;
    cudaGraphExec_t resident_exec;
    int graphs_captured;
};

struct TkmRuntime {
    TkmGraph        graph;
    int             training_enabled;
    cudaStream_t    stream;
    cublasHandle_t  cublas;
    CUmodule        nvrtc_mod;
    CUfunction      fwd_funcs[TKM_MAX_GROUPS];
    CUfunction      bwd_funcs[TKM_MAX_GROUPS];

    uint8_t        *d_arena;
    uint8_t        *d_grad_arena;
    float          *d_input_f32;
    uint8_t        *d_labels;
    __half         *d_col_ws;
    void           *d_cublas_ws;
    float          *d_one, *d_zero;
    float          *d_lr;
    float          *d_loss_buf;
    int            *d_corr_buf;
    uint32_t       *d_seed;
    float          *h_pin_loss;
    float          *h_pin_lr;
    int             total_sgd_elems;
    SgdTensorList   h_sgd;

    cudaGraph_t     step_graph;
    cudaGraphExec_t step_exec;
    int             graph_captured;
    float           last_lr;
    int             eval_start_node;

    NodeGpuState    nstate[TKM_MAX_NODES];
};

static void alloc_and_register_param(SgdTensorList *sgd, float **d_m, __half **d_h,
                                     float **d_v, float **d_g, int n, float init_scale, float const_val) {
    CUDA_CHECK(cudaMalloc(d_m, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_v, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_g, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(*d_v, 0, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(*d_g, 0, (size_t)n * sizeof(float)));

    float *h = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        if (init_scale > 0.0f) {
            float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            h[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * init_scale;
        } else {
            h[i] = const_val;
        }
    }
    CUDA_CHECK(cudaMemcpy(*d_m, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);

    if (d_h) {
        CUDA_CHECK(cudaMalloc(d_h, (size_t)n * sizeof(__half)));
        k_f32_to_f16<<<(n + 255) / 256, 256>>>(*d_m, *d_h, n);
    }

    int idx = sgd->count++;
    sgd->master[idx] = *d_m;
    sgd->fp16_w[idx] = d_h ? *d_h : NULL;
    sgd->vel[idx]    = *d_v;
    sgd->grad[idx]   = *d_g;
    sgd->size[idx]   = n;
    sgd->offset[idx + 1] = sgd->offset[idx] + n;
}

static void alloc_inference_param(float **d_f, __half **d_h, int n,
                                  float init_scale, float const_val) {
    float *h = (float*)malloc((size_t)n * sizeof(float));
    if (!h) {
        fprintf(stderr, "[TKM Error] Host allocation failed for %d inference parameters.\n", n);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        if (init_scale > 0.0f) {
            float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
            h[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2) * init_scale;
        } else {
            h[i] = const_val;
        }
    }
    if (d_f) {
        CUDA_CHECK(cudaMalloc(d_f, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*d_f, h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (d_h) {
        __half *h16 = (__half*)malloc((size_t)n * sizeof(__half));
        if (!h16) {
            free(h);
            fprintf(stderr, "[TKM Error] Host allocation failed for %d half parameters.\n", n);
            exit(1);
        }
        for (int i = 0; i < n; i++) h16[i] = __float2half(h[i]);
        CUDA_CHECK(cudaMalloc(d_h, (size_t)n * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(*d_h, h16, (size_t)n * sizeof(__half), cudaMemcpyHostToDevice));
        free(h16);
    }
    free(h);
}

static void init_node_resources(TkmRuntime *rt, SgdTensorList *sgd, int nid) {
    const TkmNode *n = &rt->graph.nodes[nid];
    NodeGpuState *st = &rt->nstate[nid];
    int rows = rt->graph.batch_size * n->H * n->W;
    int C = n->C;
    if (C <= 0) return;
    if (n->op == OP_CONV) {
        st->conv_mode = conv_select_mode(n, &rt->graph.nodes[n->in0]);
    }

    if (n->op == OP_FC || n->op == OP_FC_TWIN_FUSED || n->op == OP_FC_QKV_FUSED) {
        int in_c = rt->graph.nodes[n->in0].C;
        st->in_dim = in_c; st->out_dim = C;
        if (rt->training_enabled) {
            alloc_and_register_param(sgd, &st->d_W_master, &st->d_W_h, &st->d_vW, &st->d_dW,
                                     in_c * C, sqrtf(2.0f / (float)in_c), 0.0f);
        } else {
            alloc_inference_param(NULL, &st->d_W_h, in_c * C,
                                  sqrtf(2.0f / (float)in_c), 0.0f);
        }
    } else if (n->op == OP_CONV) {
        int in_c = rt->graph.nodes[n->in0].C;
        int kdim = n->k * n->k * in_c;
        st->in_dim = kdim; st->out_dim = C;
        if (rt->training_enabled) {
            alloc_and_register_param(sgd, &st->d_W_master, &st->d_W_h, &st->d_vW, &st->d_dW,
                                     kdim * C, sqrtf(2.0f / (float)kdim), 0.0f);
        } else {
            alloc_inference_param(NULL, &st->d_W_h, kdim * C,
                                  sqrtf(2.0f / (float)kdim), 0.0f);
        }
    } else if (n->op == OP_MIXPOOL) {
        CUDA_CHECK(cudaMalloc(&st->d_save_mask, (size_t)rows * (C / 2) * sizeof(uint16_t)));
    } else if (n->op == OP_GQA_ATTN) {
        CUDA_CHECK(cudaMalloc(&st->d_save_istd, (size_t)rows * n->q_heads * sizeof(float)));
    }

    const TkmOpSpec *spec = tkm_find_spec_by_node(n);
    if (!spec) return;
    uint32_t f = spec->res_flags;
    if (f & RES_PARAM_BIAS) {
        if (rt->training_enabled)
            alloc_and_register_param(sgd, &st->d_bias, NULL, &st->d_vbias, &st->d_dbias, C, 0.0f, 0.0f);
        else
            alloc_inference_param(&st->d_bias, NULL, C, 0.0f, 0.0f);
    }
    if (f & RES_PARAM_WEIGHT) {
        if (rt->training_enabled)
            alloc_and_register_param(sgd, &st->d_w_vec, NULL, &st->d_vw_vec, &st->d_dw_vec, C, 0.0f, 1.0f);
        else
            alloc_inference_param(&st->d_w_vec, NULL, C, 0.0f, 1.0f);
    }
    if (f & RES_PARAM_BETA) {
        if (rt->training_enabled)
            alloc_and_register_param(sgd, &st->d_beta, NULL, &st->d_vbeta, &st->d_dbeta, C, 0.0f, 0.0f);
        else
            alloc_inference_param(&st->d_beta, NULL, C, 0.0f, 0.0f);
    }
    if (f & RES_SAVE_ISTD)
        CUDA_CHECK(cudaMalloc(&st->d_save_istd, (size_t)rows * sizeof(float)));
    if (f & RES_SAVE_XHAT)
        CUDA_CHECK(cudaMalloc(&st->d_save_xhat, (size_t)rows * C * sizeof(__half)));
    if (f & RES_SAVE_MASK)
        CUDA_CHECK(cudaMalloc(&st->d_save_mask, (size_t)rows * (C / 2) * sizeof(uint16_t)));
}

static void benchmark_aligned_conv_backend(TkmRuntime *rt, int nid) {
    const TkmGraph *g = &rt->graph;
    const TkmNode *n = &g->nodes[nid];
    const TkmNode *in0 = &g->nodes[n->in0];
    NodeGpuState *st = &rt->nstate[nid];
    if (st->conv_mode != CONV_MODE_WMMA_3X3) return;

    void *in_ptr = rt->d_arena + in0->out_offset;
    void *out_ptr = rt->d_arena + n->out_offset;
    int rows = g->batch_size * n->H * n->W;
    int bn = (n->C % 32 == 0) ? 32 : 16;
    dim3 wmma_grid((rows + 63) / 64, (n->C + bn - 1) / bn);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int warm = 0; warm < 2; warm++) {
        if (bn == 32)
            k_conv3x3_wmma_fwd_h<32><<<wmma_grid, 128, 0, rt->stream>>>(
                (const __half*)in_ptr, st->d_W_h, (__half*)out_ptr,
                rows, in0->H, in0->W, in0->C, n->C);
        else
            k_conv3x3_wmma_fwd_h<16><<<wmma_grid, 128, 0, rt->stream>>>(
                (const __half*)in_ptr, st->d_W_h, (__half*)out_ptr,
                rows, in0->H, in0->W, in0->C, n->C);
    }
    cudaStreamSynchronize(rt->stream);
    CUDA_CHECK(cudaEventRecord(start, rt->stream));
    for (int rep = 0; rep < 3; rep++) {
        if (bn == 32)
            k_conv3x3_wmma_fwd_h<32><<<wmma_grid, 128, 0, rt->stream>>>(
                (const __half*)in_ptr, st->d_W_h, (__half*)out_ptr,
                rows, in0->H, in0->W, in0->C, n->C);
        else
            k_conv3x3_wmma_fwd_h<16><<<wmma_grid, 128, 0, rt->stream>>>(
                (const __half*)in_ptr, st->d_W_h, (__half*)out_ptr,
                rows, in0->H, in0->W, in0->C, n->C);
    }
    CUDA_CHECK(cudaEventRecord(stop, rt->stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float wmma_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&wmma_ms, start, stop));
    wmma_ms /= 3.0f;

    int elems = g->batch_size * in0->H * in0->W * in0->C;
    for (int warm = 0; warm < 2; warm++) {
        k_im2col_nhwc_h<<<(elems + 255) / 256, 256, 0, rt->stream>>>(
            (const __half*)in_ptr, rt->d_col_ws, g->batch_size,
            in0->H, in0->W, in0->H, in0->W, in0->C, 3, 1, 1);
        CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n->C, rows, st->in_dim, rt->d_one,
            st->d_W_h, CUDA_R_16F, n->C,
            rt->d_col_ws, CUDA_R_16F, st->in_dim,
            rt->d_zero, out_ptr, CUDA_R_16F, n->C,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaStreamSynchronize(rt->stream);
    CUDA_CHECK(cudaEventRecord(start, rt->stream));
    for (int rep = 0; rep < 3; rep++) {
        k_im2col_nhwc_h<<<(elems + 255) / 256, 256, 0, rt->stream>>>(
            (const __half*)in_ptr, rt->d_col_ws, g->batch_size,
            in0->H, in0->W, in0->H, in0->W, in0->C, 3, 1, 1);
        CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n->C, rows, st->in_dim, rt->d_one,
            st->d_W_h, CUDA_R_16F, n->C,
            rt->d_col_ws, CUDA_R_16F, st->in_dim,
            rt->d_zero, out_ptr, CUDA_R_16F, n->C,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop, rt->stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
    cublas_ms /= 3.0f;
    if (cublas_ms < wmma_ms) st->conv_mode = CONV_MODE_IM2COL;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

static TkmRuntime *tkm_runtime_ignite_mode(const TkmGraph *g, const char *cuda_src, int training_enabled) {
    cudaFree(0);
    TkmRuntime *rt = (TkmRuntime*)calloc(1, sizeof(TkmRuntime));
    rt->graph = *g;
    rt->training_enabled = training_enabled;
    rt->last_lr = -1.0f;

    CUDA_CHECK(cudaStreamCreateWithFlags(&rt->stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasCreate(&rt->cublas));
    CUBLAS_CHECK(cublasSetStream(rt->cublas, rt->stream));
    CUBLAS_CHECK(cublasSetPointerMode(rt->cublas, CUBLAS_POINTER_MODE_DEVICE));
    CUDA_CHECK(cudaMalloc(&rt->d_cublas_ws, 8 * 1024 * 1024));
    CUBLAS_CHECK(cublasSetWorkspace(rt->cublas, rt->d_cublas_ws, 8 * 1024 * 1024));

    size_t full_len = strlen(NVRTC_STANDALONE_PRELUDE) + strlen(cuda_src) + 64;
    char *full_src = (char*)malloc(full_len);
    snprintf(full_src, full_len, "%s\n%s", NVRTC_STANDALONE_PRELUDE, cuda_src);

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, full_src, "tkm_jit_kernels.cu", 0, NULL, NULL));

    int major = 8, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);
    const char *opts[] = { arch_opt, "--use_fast_math", "-std=c++14" };

    if (nvrtcCompileProgram(prog, 3, opts) != NVRTC_SUCCESS) {
        size_t lsz; nvrtcGetProgramLogSize(prog, &lsz);
        char *log = (char*)malloc(lsz); nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "=== NVRTC Compile Log ===\n%s\n", log);
        exit(1);
    }
    size_t ptx_sz; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_sz));
    char *ptx = (char*)malloc(ptx_sz); NVRTC_CHECK(nvrtcGetPTX(prog, ptx));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    free(full_src);

    CU_CHECK(cuModuleLoadDataEx(&rt->nvrtc_mod, ptx, 0, 0, 0));
    free(ptx);

    for (int i = 0; i < g->num_groups; i++) {
        CU_CHECK(cuModuleGetFunction(&rt->fwd_funcs[i], rt->nvrtc_mod, g->groups[i].fwd_sig.func_name));
        CU_CHECK(cuModuleGetFunction(&rt->bwd_funcs[i], rt->nvrtc_mod, g->groups[i].bwd_sig.func_name));
    }

    CUDA_CHECK(cudaMalloc(&rt->d_arena, g->total_arena_bytes));
    if (training_enabled)
        CUDA_CHECK(cudaMalloc(&rt->d_grad_arena, g->total_arena_bytes));
    CUDA_CHECK(cudaMemset(rt->d_arena,      0, g->total_arena_bytes));
    if (training_enabled)
        CUDA_CHECK(cudaMemset(rt->d_grad_arena, 0, g->total_arena_bytes));

    size_t in_elems = (size_t)g->batch_size * g->nodes[0].H * g->nodes[0].W * g->nodes[0].C;
    CUDA_CHECK(cudaMalloc(&rt->d_input_f32, in_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rt->d_labels,    (size_t)g->batch_size * g->nodes[0].H * g->nodes[0].W * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(rt->d_labels, 0,  (size_t)g->batch_size * g->nodes[0].H * g->nodes[0].W * sizeof(uint8_t)));

    size_t max_col = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->nodes[i].dead && g->nodes[i].op == OP_CONV) {
            TkmNode *n = &rt->graph.nodes[i];
            TkmNode *in0 = &rt->graph.nodes[n->in0];
            int mode = conv_select_mode(n, in0);
            rt->nstate[i].conv_mode = mode;
            if (mode == CONV_MODE_1X1) continue;
            size_t csz = (size_t)g->batch_size * n->H * n->W * n->k * n->k * in0->C;
            if (csz > max_col) max_col = csz;
        }
    }
    if (max_col > 0) CUDA_CHECK(cudaMalloc(&rt->d_col_ws, max_col * sizeof(__half)));

    float one = 1.0f, zero = 0.0f, init_lr = 1e-3f;
    uint32_t init_seed = 0x13579BDFu;
    CUDA_CHECK(cudaMalloc(&rt->d_one,  sizeof(float))); CUDA_CHECK(cudaMemcpy(rt->d_one,  &one,     sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rt->d_zero, sizeof(float))); CUDA_CHECK(cudaMemcpy(rt->d_zero, &zero,    sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rt->d_lr,   sizeof(float))); CUDA_CHECK(cudaMemcpy(rt->d_lr,   &init_lr, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&rt->d_loss_buf, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rt->d_corr_buf, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rt->d_seed,     sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&rt->h_pin_loss, sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&rt->h_pin_lr, sizeof(float)));
    *rt->h_pin_loss = 0.0f;
    *rt->h_pin_lr = init_lr;
    CUDA_CHECK(cudaMemcpy(rt->d_seed, &init_seed, sizeof(uint32_t), cudaMemcpyHostToDevice));

    memset(&rt->h_sgd, 0, sizeof(SgdTensorList));
    for (int i = 0; i < g->num_nodes; i++) {
        if (rt->graph.nodes[i].C == 0 && rt->graph.nodes[i].in0 >= 0) {
            int p_id = rt->graph.nodes[i].in0;
            rt->graph.nodes[i].H = rt->graph.nodes[p_id].H;
            rt->graph.nodes[i].W = rt->graph.nodes[p_id].W;
            rt->graph.nodes[i].C = (rt->graph.nodes[p_id].c_out > 0) ? rt->graph.nodes[p_id].c_out : rt->graph.nodes[p_id].C;
        }
        init_node_resources(rt, &rt->h_sgd, i);
    }
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->nodes[i].dead && g->nodes[i].op == OP_CONV)
            benchmark_aligned_conv_backend(rt, i);
    }
    int has_im2col = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (!g->nodes[i].dead && g->nodes[i].op == OP_CONV &&
            (rt->nstate[i].conv_mode == CONV_MODE_IM2COL ||
             rt->nstate[i].conv_mode == CONV_MODE_DIRECT_3X3)) {
            has_im2col = 1;
            break;
        }
    }
    if (!has_im2col && rt->d_col_ws) {
        cudaFree(rt->d_col_ws);
        rt->d_col_ws = NULL;
    }
    rt->total_sgd_elems = rt->h_sgd.offset[rt->h_sgd.count];
    CUDA_CHECK(cudaMemcpyToSymbol(c_tkm_sgd, &rt->h_sgd, sizeof(SgdTensorList)));

    printf("  [TKM GPU Engine] Ignited %d NVRTC Kernels | %d Learnable Tensors (%d params) | %.2f MB Arena\n",
           g->num_groups * 2, rt->h_sgd.count, rt->total_sgd_elems, g->total_arena_bytes / (1024.0 * 1024.0));
    return rt;
}

TkmRuntime *tkm_runtime_ignite(const TkmGraph *g, const char *cuda_src) {
    return tkm_runtime_ignite_mode(g, cuda_src, 1);
}

TkmRuntime *tkm_runtime_ignite_inference(const TkmGraph *g, const char *cuda_src) {
    return tkm_runtime_ignite_mode(g, cuda_src, 0);
}

static void *resolve_arg_ptr(TkmRuntime *rt, const TkmKernelArg *a) {
    int nid = a->node_id;
    const TkmNode *n = (nid >= 0) ? &rt->graph.nodes[nid] : NULL;
    switch (a->kind) {
        case KARG_IN_MAIN:
        case KARG_OUT_MAIN:
        case KARG_BRANCH_IN:   return rt->d_arena      + (n ? n->out_offset : 0);
        case KARG_DOUT_MAIN:
        case KARG_DIN_MAIN:
        case KARG_DBRANCH_OUT: return rt->d_grad_arena + (n ? n->out_offset : 0);
        case KARG_BIAS:        return rt->nstate[nid].d_bias;
        case KARG_DBIAS:       return rt->nstate[nid].d_dbias;
        case KARG_WEIGHT_VEC:  return rt->nstate[nid].d_w_vec;
        case KARG_DWEIGHT_VEC: return rt->nstate[nid].d_dw_vec;
        case KARG_BETA:        return rt->nstate[nid].d_beta;
        case KARG_DBETA:       return rt->nstate[nid].d_dbeta;
        case KARG_SAVE_ISTD:   return rt->nstate[nid].d_save_istd;
        case KARG_SAVE_XHAT:   return rt->nstate[nid].d_save_xhat;
        case KARG_SAVE_MASK:   return rt->nstate[nid].d_save_mask;
        case KARG_STEP_SEED:   return rt->d_seed;
        default:               return NULL;
    }
}

static void launch_plasma_sig(TkmRuntime *rt, CUfunction fn, const TkmKernelSig *sig) {
    void *ptrs[TKM_MAX_KARGS];
    void *kargs[TKM_MAX_KARGS];
    for (int i = 0; i < sig->num_args; i++) {
        ptrs[i]  = resolve_arg_ptr(rt, &sig->args[i]);
        kargs[i] = (void*)&ptrs[i];
    }
    CU_CHECK(cuLaunchKernel(fn, sig->grid_x, 1, 1, sig->block_x, 1, 1, 0, (CUstream)rt->stream, kargs, NULL));
}

static void execute_forward_pass(TkmRuntime *rt, int is_eval, int valid_examples) {
    const TkmGraph *g = &rt->graph;
    int B = g->batch_size;
    int executed_grp[TKM_MAX_GROUPS] = {0};

    for (int i = 1; i < g->num_nodes; i++) {
        const TkmNode *n = &g->nodes[i];
        if (n->dead) continue;
        if (rt->eval_start_node >= 0 && i <= rt->eval_start_node) continue;

        if (n->plasma_grp >= 0) {
            int gid = n->plasma_grp;
            if (!executed_grp[gid]) {
                executed_grp[gid] = 1;
                if (rt->eval_start_node >= 0 &&
                    g->groups[gid].anchor_id <= rt->eval_start_node)
                    continue;
                if (is_eval && g->groups[gid].is_coord_group) {
                    const TkmNode *in0 = &g->nodes[g->groups[gid].anchor_id];
                    const TkmNode *out = &g->nodes[g->groups[gid].out_node_id];
                    CUDA_CHECK(cudaMemcpyAsync(rt->d_arena + out->out_offset,
                                               rt->d_arena + in0->out_offset,
                                               out->out_bytes, cudaMemcpyDeviceToDevice, rt->stream));
                } else {
                    launch_plasma_sig(rt, rt->fwd_funcs[gid], &g->groups[gid].fwd_sig);
                }
            }
            continue;
        }

        void *in_ptr  = rt->d_arena + g->nodes[n->in0].out_offset;
        void *out_ptr = rt->d_arena + n->out_offset;
        int rows = B * n->H * n->W;

        switch (n->op) {
            case OP_FC:
            case OP_FC_TWIN_FUSED:
            case OP_FC_QKV_FUSED: {
                int K = rt->nstate[i].in_dim, N_out = rt->nstate[i].out_dim;
                CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N_out, rows, K, rt->d_one,
                    rt->nstate[i].d_W_h, CUDA_R_16F, N_out,
                    in_ptr, CUDA_R_16F, K,
                    rt->d_zero, out_ptr, CUDA_R_16F, N_out,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                break;
            }
            case OP_CONV: {
                const TkmNode *in0 = &g->nodes[n->in0];
                int conv_mode = rt->nstate[i].conv_mode;
                if (conv_mode == CONV_MODE_1X1) {
                    CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        n->C, rows, in0->C, rt->d_one,
                        rt->nstate[i].d_W_h, CUDA_R_16F, n->C,
                        in_ptr, CUDA_R_16F, in0->C,
                        rt->d_zero, out_ptr, CUDA_R_16F, n->C,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                } else if (conv_mode == CONV_MODE_DIRECT_3X3) {
                    int total = B * in0->H * in0->W * n->C;
                    k_conv_direct_small_cin_fwd_h<<<(total + 255) / 256, 256, 0, rt->stream>>>(
                        (const __half*)in_ptr, rt->nstate[i].d_W_h, (__half*)out_ptr,
                        B, in0->H, in0->W, in0->C, n->C);
                } else if (conv_mode == CONV_MODE_WMMA_3X3) {
                    int bn = (n->C % 32 == 0) ? 32 : 16;
                    dim3 grid((rows + 63) / 64, (n->C + bn - 1) / bn);
                    if (bn == 32)
                        k_conv3x3_wmma_fwd_h<32><<<grid, 128, 0, rt->stream>>>(
                            (const __half*)in_ptr, rt->nstate[i].d_W_h, (__half*)out_ptr,
                            rows, in0->H, in0->W, in0->C, n->C);
                    else
                        k_conv3x3_wmma_fwd_h<16><<<grid, 128, 0, rt->stream>>>(
                            (const __half*)in_ptr, rt->nstate[i].d_W_h, (__half*)out_ptr,
                            rows, in0->H, in0->W, in0->C, n->C);
                } else {
                    int elems = B * n->H * n->W * in0->C;
                    k_im2col_nhwc_h<<<(elems + 255) / 256, 256, 0, rt->stream>>>(
                        (const __half*)in_ptr, rt->d_col_ws, B, in0->H, in0->W,
                        n->H, n->W, in0->C, n->k, n->s, n->p);
                    int K = rt->nstate[i].in_dim, N_out = rt->nstate[i].out_dim;
                    CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N_out, rows, K, rt->d_one,
                        rt->nstate[i].d_W_h, CUDA_R_16F, N_out,
                        rt->d_col_ws, CUDA_R_16F, K,
                        rt->d_zero, out_ptr, CUDA_R_16F, N_out,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                }
                break;
            }
            case OP_MIXPOOL: {
                int total = B * n->H * n->W * (n->C / 2);
                k_mixpool2x2_fwd_h2<<<(total + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half2*)in_ptr, (__half2*)out_ptr, rt->nstate[i].d_save_mask,
                    B, n->H, n->W, n->C / 2, n->fval);
                break;
            }
            case OP_AVGPOOL: {
                int total = B * n->H * n->W * (n->C / 2);
                k_mixpool2x2_fwd_h2<<<(total + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half2*)in_ptr, (__half2*)out_ptr, NULL,
                    B, n->H, n->W, n->C / 2, 0.0f);
                break;
            }
            case OP_GQA_ATTN: {
                int Seq = n->H * n->W;
                dim3 grid(B * n->q_heads, (Seq + 31) / 32);
                if (n->head_dim == 64) {
                    k_gqa_flash_attn_tc_fwd<64><<<grid, 64, 0, rt->stream>>>(
                        (const __half*)in_ptr, (__half*)out_ptr, rt->nstate[i].d_save_istd,
                        B, Seq, n->q_heads, n->kv_heads, 1.0f / sqrtf((float)n->head_dim), n->causal);
                } else {
                    k_gqa_flash_attn_tc_fwd<128><<<grid, 64, 0, rt->stream>>>(
                        (const __half*)in_ptr, (__half*)out_ptr, rt->nstate[i].d_save_istd,
                        B, Seq, n->q_heads, n->kv_heads, 1.0f / sqrtf((float)n->head_dim), n->causal);
                }
                break;
            }
            case OP_FLATTEN:
                if (out_ptr != in_ptr)
                    CUDA_CHECK(cudaMemcpyAsync(out_ptr, in_ptr, n->out_bytes, cudaMemcpyDeviceToDevice, rt->stream));
                break;
            case OP_SOFTMAX: {
                k_softmax_fwd_h<<<(rows + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half*)in_ptr, (__half*)out_ptr, rows, n->C);
                break;
            }
            case OP_SOFTMAX_CE:
            case OP_SOFTMAX_FOCAL:
            case OP_SOFTMAX_MSE: {
                void *dout_ptr = rt->d_grad_arena + g->nodes[n->in0].out_offset;
                int loss_kind = n->op == OP_SOFTMAX_FOCAL ? 1 : (n->op == OP_SOFTMAX_MSE ? 2 : 0);
                k_softmax_loss_fwd_bwd_h<<<(rows + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half*)in_ptr, rt->d_labels, (__half*)dout_ptr,
                    rt->d_loss_buf, rt->d_corr_buf, rows, n->C, is_eval,
                    is_eval ? (valid_examples * rows + B - 1) / B : rows, loss_kind, n->fval);
                break;
            }
            default: break;
        }
    }
}

static void record_full_step_stream(TkmRuntime *rt, int clear_loss) {
    const TkmGraph *g = &rt->graph;
    int B = g->batch_size;

    if (clear_loss)
        CUDA_CHECK(cudaMemsetAsync(rt->d_loss_buf, 0, sizeof(float), rt->stream));
    CUDA_CHECK(cudaMemsetAsync(rt->d_grad_arena, 0, g->total_arena_bytes, rt->stream));
    for (int i = 0; i < rt->h_sgd.count; i++) {
        if (rt->h_sgd.fp16_w[i] == NULL) {
            CUDA_CHECK(cudaMemsetAsync((void*)rt->h_sgd.grad[i], 0, (size_t)rt->h_sgd.size[i] * sizeof(float), rt->stream));
        }
    }

    execute_forward_pass(rt, 0, B);

    int executed_grp[TKM_MAX_GROUPS] = {0};
    for (int i = g->num_nodes - 1; i >= 1; i--) {
        const TkmNode *n = &g->nodes[i];
        if (n->dead || n->op == OP_SOFTMAX_CE || n->op == OP_SOFTMAX_FOCAL ||
            n->op == OP_SOFTMAX_MSE) continue;

        if (n->plasma_grp >= 0) {
            int gid = n->plasma_grp;
            if (!executed_grp[gid]) {
                executed_grp[gid] = 1;
                const PlasmaGroup *pg = &g->groups[gid];
                if (!(pg->is_coord_group && pg->anchor_id == 0))
                    launch_plasma_sig(rt, rt->bwd_funcs[gid], &pg->bwd_sig);
            }
            continue;
        }

        void *in_act  = rt->d_arena      + g->nodes[n->in0].out_offset;
        void *out_act = rt->d_arena      + n->out_offset;
        void *dout    = rt->d_grad_arena + n->out_offset;
        void *din     = rt->d_grad_arena + g->nodes[n->in0].out_offset;
        int rows = B * n->H * n->W;

        switch (n->op) {
            case OP_FC:
            case OP_FC_TWIN_FUSED:
            case OP_FC_QKV_FUSED: {
                int K = rt->nstate[i].in_dim, N_out = rt->nstate[i].out_dim;
                CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    N_out, K, rows, rt->d_one,
                    dout, CUDA_R_16F, N_out, in_act, CUDA_R_16F, K,
                    rt->d_zero, rt->nstate[i].d_dW, CUDA_R_32F, N_out,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    K, rows, N_out, rt->d_one,
                    rt->nstate[i].d_W_h, CUDA_R_16F, N_out, dout, CUDA_R_16F, N_out,
                    rt->d_one, din, CUDA_R_16F, K,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                break;
            }
            case OP_CONV: {
                const TkmNode *in0 = &g->nodes[n->in0];
                int conv_mode = rt->nstate[i].conv_mode;
                int needs_input_grad = !(in0->op == OP_INPUT || in0->cat == CAT_COORD);
                if (conv_mode == CONV_MODE_1X1) {
                    CUDA_CHECK(cudaMemsetAsync(rt->nstate[i].d_dW, 0,
                                               (size_t)in0->C * n->C * sizeof(float), rt->stream));
                    CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        n->C, in0->C, rows, rt->d_one,
                        dout, CUDA_R_16F, n->C, in_act, CUDA_R_16F, in0->C,
                        rt->d_zero, rt->nstate[i].d_dW, CUDA_R_32F, n->C,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    if (needs_input_grad) {
                        CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            in0->C, rows, n->C, rt->d_one,
                            rt->nstate[i].d_W_h, CUDA_R_16F, n->C, dout, CUDA_R_16F, n->C,
                            rt->d_one, din, CUDA_R_16F, in0->C,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    }
                } else if (conv_mode == CONV_MODE_DIRECT_3X3) {
                    int elems = B * n->H * n->W * in0->C;
                    k_im2col_nhwc_h<<<(elems + 255) / 256, 256, 0, rt->stream>>>(
                        (const __half*)in_act, rt->d_col_ws, B, in0->H, in0->W,
                        n->H, n->W, in0->C, n->k, n->s, n->p);
                    int K = rt->nstate[i].in_dim, N_out = rt->nstate[i].out_dim;
                    CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        N_out, K, rows, rt->d_one,
                        dout, CUDA_R_16F, N_out, rt->d_col_ws, CUDA_R_16F, K,
                        rt->d_zero, rt->nstate[i].d_dW, CUDA_R_32F, N_out,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    if (needs_input_grad) {
                        CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            K, rows, N_out, rt->d_one,
                            rt->nstate[i].d_W_h, CUDA_R_16F, N_out, dout, CUDA_R_16F, N_out,
                            rt->d_zero, rt->d_col_ws, CUDA_R_16F, K,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        int din_elems = B * in0->H * in0->W * in0->C;
                        k_col2im_nhwc_h<<<(din_elems + 255) / 256, 256, 0, rt->stream>>>(
                            rt->d_col_ws, (__half*)din, B, in0->H, in0->W,
                            n->H, n->W, in0->C, n->k, n->s, n->p);
                    }
                } else if (conv_mode == CONV_MODE_WMMA_3X3) {
                    CUDA_CHECK(cudaMemsetAsync(rt->nstate[i].d_dW, 0,
                                               (size_t)9 * in0->C * n->C * sizeof(float), rt->stream));
                    int split_k = (rows + 255) / 256;
                    if (split_k < 1) split_k = 1;
                    if (split_k > 32) split_k = 32;
                    dim3 grid((in0->C / 16) * (n->C / 16), 9, split_k);
                    k_conv3x3_wmma_dW_h<<<grid, 128, 0, rt->stream>>>(
                        (const __half*)in_act, (const __half*)dout, rt->nstate[i].d_dW,
                        rows, in0->H, in0->W, in0->C, n->C, split_k);
                    if (needs_input_grad) {
                        dim3 dx_grid((rows + 63) / 64, (in0->C + 15) / 16);
                        k_conv3x3_wmma_dX_h<<<dx_grid, 128, 0, rt->stream>>>(
                            (const __half*)dout, rt->nstate[i].d_W_h, (__half*)din,
                            rows, in0->H, in0->W, in0->C, n->C);
                    }
                } else {
                    int elems = B * n->H * n->W * in0->C;
                    k_im2col_nhwc_h<<<(elems + 255) / 256, 256, 0, rt->stream>>>(
                        (const __half*)in_act, rt->d_col_ws, B, in0->H, in0->W,
                        n->H, n->W, in0->C, n->k, n->s, n->p);
                    int K = rt->nstate[i].in_dim, N_out = rt->nstate[i].out_dim;
                    CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                        N_out, K, rows, rt->d_one,
                        dout, CUDA_R_16F, N_out, rt->d_col_ws, CUDA_R_16F, K,
                        rt->d_zero, rt->nstate[i].d_dW, CUDA_R_32F, N_out,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                    if (needs_input_grad) {
                        CUBLAS_CHECK(cublasGemmEx(rt->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            K, rows, N_out, rt->d_one,
                            rt->nstate[i].d_W_h, CUDA_R_16F, N_out, dout, CUDA_R_16F, N_out,
                            rt->d_zero, rt->d_col_ws, CUDA_R_16F, K,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        int din_elems = B * in0->H * in0->W * in0->C;
                        k_col2im_nhwc_h<<<(din_elems + 255) / 256, 256, 0, rt->stream>>>(
                            rt->d_col_ws, (__half*)din, B, in0->H, in0->W,
                            n->H, n->W, in0->C, n->k, n->s, n->p);
                    }
                }
                break;
            }
            case OP_MIXPOOL: {
                int total = B * n->H * n->W * (n->C / 2);
                k_mixpool2x2_bwd_h2<<<(total + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half2*)dout, rt->nstate[i].d_save_mask, (__half2*)din,
                    B, n->H, n->W, n->C / 2, n->fval);
                break;
            }
            case OP_AVGPOOL: {
                int total = B * n->H * n->W * (n->C / 2);
                k_mixpool2x2_bwd_h2<<<(total + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half2*)dout, NULL, (__half2*)din,
                    B, n->H, n->W, n->C / 2, 0.0f);
                break;
            }
            case OP_GQA_ATTN: {
                int Seq = n->H * n->W;
                dim3 grid(B * n->q_heads, (Seq + 31) / 32);
                if (n->head_dim == 64) {
                    k_gqa_flash_attn_tc_bwd<64><<<grid, 64, 0, rt->stream>>>(
                        (const __half*)in_act, (const __half*)out_act, (const __half*)dout,
                        rt->nstate[i].d_save_istd, (__half*)din,
                        B, Seq, n->q_heads, n->kv_heads, 1.0f / sqrtf((float)n->head_dim), n->causal);
                } else {
                    k_gqa_flash_attn_tc_bwd<128><<<grid, 64, 0, rt->stream>>>(
                        (const __half*)in_act, (const __half*)out_act, (const __half*)dout,
                        rt->nstate[i].d_save_istd, (__half*)din,
                        B, Seq, n->q_heads, n->kv_heads, 1.0f / sqrtf((float)n->head_dim), n->causal);
                }
                break;
            }
            case OP_FLATTEN:
                if (din != dout)
                    CUDA_CHECK(cudaMemcpyAsync(din, dout, g->nodes[n->in0].out_bytes, cudaMemcpyDeviceToDevice, rt->stream));
                break;
            case OP_SOFTMAX:
                k_softmax_bwd_h<<<(rows + 255) / 256, 256, 0, rt->stream>>>(
                    (const __half*)out_act, (const __half*)dout, (__half*)din, rows, n->C);
                break;
            default: break;
        }
    }

    if (rt->total_sgd_elems > 0) {
        k_tkm_multi_tensor_sgd<<<(rt->total_sgd_elems + 255) / 256, 256, 0, rt->stream>>>(
            rt->d_lr, 0.9f, 1e-4f, rt->d_seed);
    }
}

TkmRuntime *tkm_load_string(const char *dsl_text, int batch_size) {
    if (!dsl_text || batch_size <= 0) return NULL;
    return tkm_load_buffer(dsl_text, strlen(dsl_text), batch_size);
}

TkmRuntime *tkm_load_buffer(const char *dsl_data, size_t dsl_size, int batch_size) {
    if (!dsl_data || dsl_size == 0 || batch_size <= 0 || dsl_size > SIZE_MAX - 1) return NULL;
    TkmGraph *g = (TkmGraph*)calloc(1, sizeof(TkmGraph));
    if (!g) return NULL;
    char *dsl_text = (char*)malloc(dsl_size + 1);
    if (!dsl_text) {
        free(g);
        return NULL;
    }
    memcpy(dsl_text, dsl_data, dsl_size);
    dsl_text[dsl_size] = '\0';
    tkm_parse_dsl(g, dsl_text, batch_size);
    tkm_optimize_and_bind(g);
    tkm_plan_memory(g);
    char *cuda_src = tkm_codegen_cuda_string(g);
    TkmRuntime *rt = tkm_runtime_ignite(g, cuda_src);
    free(cuda_src);
    free(dsl_text);
    free(g);
    return rt;
}

TkmRuntime *tkm_load_file(const char *tkm_path, int batch_size) {
    if (!tkm_path || batch_size <= 0) return NULL;
    FILE *f = fopen(tkm_path, "rb");
    if (!f) { perror(tkm_path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *dsl = (char*)malloc((size_t)sz);
    if (!dsl) {
        fclose(f);
        return NULL;
    }
    if (fread(dsl, 1, (size_t)sz, f) != (size_t)sz) {
        free(dsl);
        fclose(f);
        return NULL;
    }
    fclose(f);

    TkmRuntime *rt = tkm_load_buffer(dsl, (size_t)sz, batch_size);
    free(dsl);
    return rt;
}

float tkm_runtime_step(TkmRuntime *rt, const float *h_input, const uint8_t *h_labels, float lr) {
    rt->eval_start_node = -1;
    int B = rt->graph.batch_size;
    int in_elems = B * rt->graph.nodes[0].H * rt->graph.nodes[0].W * rt->graph.nodes[0].C;

    if (lr != rt->last_lr) {
        rt->last_lr = lr;
        *rt->h_pin_lr = lr;
        CUDA_CHECK(cudaMemcpyAsync(rt->d_lr, rt->h_pin_lr, sizeof(float), cudaMemcpyHostToDevice, rt->stream));
    }
    if (h_input) {
        CUDA_CHECK(cudaMemcpyAsync(rt->d_input_f32, h_input, (size_t)in_elems * sizeof(float), cudaMemcpyHostToDevice, rt->stream));
        k_f32_to_f16<<<(in_elems + 255) / 256, 256, 0, rt->stream>>>(
            rt->d_input_f32, (__half*)(rt->d_arena + rt->graph.nodes[0].out_offset), in_elems);
    }
    if (h_labels) {
        CUDA_CHECK(cudaMemcpyAsync(rt->d_labels, h_labels, (size_t)B * sizeof(uint8_t), cudaMemcpyHostToDevice, rt->stream));
    }

    if (!rt->graph_captured) {
        CUDA_CHECK(cudaStreamSynchronize(rt->stream));
        CUDA_CHECK(cudaStreamBeginCapture(rt->stream, cudaStreamCaptureModeGlobal));
        record_full_step_stream(rt, 1);
        CUDA_CHECK(cudaStreamEndCapture(rt->stream, &rt->step_graph));
        CUDA_CHECK(cudaGraphInstantiate(&rt->step_exec, rt->step_graph, NULL, NULL, 0));
        rt->graph_captured = 1;
        printf("  [CUDA Graph] Full Fwd+Bwd+SGD step captured into single cudaGraphExec_t!\n");
    }

    CUDA_CHECK(cudaGraphLaunch(rt->step_exec, rt->stream));

    float h_loss = 0.0f;
    if (h_input) {
        CUDA_CHECK(cudaMemcpyAsync(rt->h_pin_loss, rt->d_loss_buf, sizeof(float), cudaMemcpyDeviceToHost, rt->stream));
        CUDA_CHECK(cudaStreamSynchronize(rt->stream));
        h_loss = *rt->h_pin_loss;
    }
    return h_loss;
}

size_t tkm_runtime_forward(TkmRuntime *rt, const float *h_input, float *h_output, size_t output_capacity) {
    if (!rt || !h_input || !h_output) {
        fprintf(stderr, "[TKM API Error] tkm_runtime_forward requires a runtime and non-null input/output buffers.\n");
        return 0;
    }

    const TkmGraph *g = &rt->graph;
    int B = g->batch_size;
    int in_elems = B * g->nodes[0].H * g->nodes[0].W * g->nodes[0].C;
    int output_id = -1;
    for (int i = g->num_nodes - 1; i >= 0; i--) {
        if (!g->nodes[i].dead) {
            output_id = i;
            break;
        }
    }
    if (output_id < 0 || g->nodes[output_id].op == OP_SOFTMAX_CE ||
        g->nodes[output_id].op == OP_SOFTMAX_FOCAL ||
        g->nodes[output_id].op == OP_SOFTMAX_MSE) {
        fprintf(stderr, "[TKM API Error] tkm_runtime_forward requires a graph ending in a tensor output, not a loss sink.\n");
        return 0;
    }

    const TkmNode *output = &g->nodes[output_id];
    size_t output_elems = (size_t)B * output->H * output->W * output->C;
    if (output_elems > 2147483647u) {
        fprintf(stderr, "[TKM API Error] Inference output exceeds the supported element count.\n");
        return 0;
    }
    if (output_capacity < output_elems) {
        fprintf(stderr, "[TKM API Error] Inference output buffer needs %zu floats; capacity is %zu.\n",
                output_elems, output_capacity);
        return 0;
    }
    __half *host_output = (__half*)malloc(output_elems * sizeof(__half));
    if (!host_output) {
        fprintf(stderr, "[TKM Error] Unable to allocate temporary inference output.\n");
        return 0;
    }

    CUDA_CHECK(cudaMemcpyAsync(rt->d_input_f32, h_input, (size_t)in_elems * sizeof(float),
                               cudaMemcpyHostToDevice, rt->stream));
    k_f32_to_f16<<<(in_elems + 255) / 256, 256, 0, rt->stream>>>(
        rt->d_input_f32, (__half*)(rt->d_arena + g->nodes[0].out_offset), in_elems);
    execute_forward_pass(rt, 1, B);
    CUDA_CHECK(cudaMemcpyAsync(host_output, rt->d_arena + output->out_offset,
                               output_elems * sizeof(__half), cudaMemcpyDeviceToHost, rt->stream));
    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    for (size_t i = 0; i < output_elems; i++) h_output[i] = __half2float(host_output[i]);
    free(host_output);
    return output_elems;
}

/* Evaluate single scalar loss of current state on GPU */
float tkm_eval_current_loss(TkmRuntime *rt) {
    CUDA_CHECK(cudaMemsetAsync(rt->d_loss_buf, 0, sizeof(float), rt->stream));
    execute_forward_pass(rt, 0, rt->graph.batch_size);
    float l = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&l, rt->d_loss_buf, sizeof(float), cudaMemcpyDeviceToHost, rt->stream));
    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    rt->eval_start_node = -1;
    return l;
}

/* Perturb element idx of node_id by delta in GPU arena */
void tkm_perturb_tensor(TkmRuntime *rt, int node_id, int elem_idx, float delta) {
    const TkmNode *n = &rt->graph.nodes[node_id];
    __half *ptr = (__half*)(rt->d_arena + n->out_offset);
    k_perturb_f16<<<1, 1, 0, rt->stream>>>(ptr, elem_idx, delta);
    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    rt->eval_start_node = node_id;
}

/* Sample analytical gradient of element idx from grad_arena */
float tkm_get_grad_sample(TkmRuntime *rt, int node_id, int elem_idx) {
    const TkmNode *n = &rt->graph.nodes[node_id];
    __half val_h;
    CUDA_CHECK(cudaMemcpy(&val_h, (__half*)(rt->d_grad_arena + n->out_offset) + elem_idx, sizeof(__half), cudaMemcpyDeviceToHost));
    return __half2float(val_h);
}

int tkm_runtime_eval(TkmRuntime *rt, const float *h_input, const uint8_t *h_labels, int count) {
    if (!rt || !h_input || !h_labels || count <= 0) return 0;

    int B = rt->graph.batch_size;
    if (B <= 0) return 0;
    int in_elems = B * rt->graph.nodes[0].H * rt->graph.nodes[0].W * rt->graph.nodes[0].C;
    int elems_per_example = in_elems / B;
    int total_correct = 0;

    CUDA_CHECK(cudaMemsetAsync(rt->d_corr_buf, 0, sizeof(int), rt->stream));
    for (int offset = 0; offset < count; offset += B) {
        int valid_examples = (count - offset < B) ? count - offset : B;
        size_t valid_elems = (size_t)valid_examples * elems_per_example;
        size_t batch_bytes = (size_t)in_elems * sizeof(float);
        size_t valid_bytes = valid_elems * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(rt->d_input_f32, h_input + (size_t)offset * elems_per_example,
                                   valid_bytes, cudaMemcpyHostToDevice, rt->stream));
        if (valid_bytes < batch_bytes) {
            CUDA_CHECK(cudaMemsetAsync((char*)rt->d_input_f32 + valid_bytes, 0,
                                       batch_bytes - valid_bytes, rt->stream));
        }
        k_f32_to_f16<<<(in_elems + 255) / 256, 256, 0, rt->stream>>>(
            rt->d_input_f32, (__half*)(rt->d_arena + rt->graph.nodes[0].out_offset), in_elems);
        CUDA_CHECK(cudaMemcpyAsync(rt->d_labels, h_labels + offset, (size_t)valid_examples * sizeof(uint8_t), cudaMemcpyHostToDevice, rt->stream));
        if (valid_examples < B) {
            CUDA_CHECK(cudaMemsetAsync(rt->d_labels + valid_examples, 0,
                                       (size_t)(B - valid_examples) * sizeof(uint8_t), rt->stream));
        }
        execute_forward_pass(rt, 1, valid_examples);
    }
    CUDA_CHECK(cudaMemcpyAsync(&total_correct, rt->d_corr_buf, sizeof(int), cudaMemcpyDeviceToHost, rt->stream));
    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    return total_correct;
}

#define TKM_WEIGHT_MAGIC 0x544B4D31u

size_t tkm_runtime_input_elements(const TkmRuntime *rt) {
    if (!rt || rt->graph.batch_size <= 0 || rt->graph.num_nodes <= 0) return 0;
    const TkmNode *input = &rt->graph.nodes[0];
    return (size_t)rt->graph.batch_size * input->H * input->W * input->C;
}

size_t tkm_runtime_output_elements(const TkmRuntime *rt) {
    if (!rt || rt->graph.batch_size <= 0) return 0;
    for (int i = rt->graph.num_nodes - 1; i >= 0; i--) {
        const TkmNode *node = &rt->graph.nodes[i];
        if (!node->dead) {
            if (node->op == OP_SOFTMAX_CE || node->op == OP_SOFTMAX_FOCAL ||
                node->op == OP_SOFTMAX_MSE) return 0;
            return (size_t)rt->graph.batch_size * node->H * node->W * node->C;
        }
    }
    return 0;
}

size_t tkm_runtime_weight_bytes(const TkmRuntime *rt) {
    if (!rt || rt->h_sgd.count < 0 || rt->h_sgd.count > UINT32_MAX) return 0;
    size_t bytes = 2 * sizeof(uint32_t);
    for (int i = 0; i < rt->h_sgd.count; i++) {
        if (rt->h_sgd.size[i] < 0 ||
            bytes > SIZE_MAX - sizeof(uint32_t) ||
            (size_t)rt->h_sgd.size[i] > (SIZE_MAX - bytes - sizeof(uint32_t)) / sizeof(float)) {
            return 0;
        }
        bytes += sizeof(uint32_t) + (size_t)rt->h_sgd.size[i] * sizeof(float);
    }
    return bytes;
}

int tkm_save_weights_buffer(TkmRuntime *rt, void *buffer, size_t capacity, size_t *out_size) {
    if (out_size) *out_size = 0;
    if (!rt) return 0;
    size_t required = tkm_runtime_weight_bytes(rt);
    if (required == 0) return 0;
    if (out_size) *out_size = required;
    if (!buffer || capacity < required) return 0;

    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    uint8_t *dst = (uint8_t*)buffer;
    uint32_t magic = TKM_WEIGHT_MAGIC, nt = (uint32_t)rt->h_sgd.count;
    memcpy(dst, &magic, sizeof(magic)); dst += sizeof(magic);
    memcpy(dst, &nt, sizeof(nt)); dst += sizeof(nt);
    for (int i = 0; i < rt->h_sgd.count; i++) {
        uint32_t n = (uint32_t)rt->h_sgd.size[i];
        memcpy(dst, &n, sizeof(n)); dst += sizeof(n);
        float *h = (float*)malloc((size_t)n * sizeof(float));
        if (n > 0 && !h) return 0;
        CUDA_CHECK(cudaMemcpy(h, rt->h_sgd.master[i], (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
        memcpy(dst, h, (size_t)n * sizeof(float));
        dst += (size_t)n * sizeof(float);
        free(h);
    }
    return 1;
}

int tkm_load_weights_buffer(TkmRuntime *rt, const void *buffer, size_t size) {
    if (!rt || !buffer || size < 2 * sizeof(uint32_t)) return 0;
    const uint8_t *src = (const uint8_t*)buffer;
    uint32_t magic = 0, nt = 0;
    memcpy(&magic, src, sizeof(magic)); src += sizeof(magic);
    memcpy(&nt, src, sizeof(nt)); src += sizeof(nt);
    if (magic != TKM_WEIGHT_MAGIC || nt != (uint32_t)rt->h_sgd.count) return 0;
    size -= 2 * sizeof(uint32_t);
    for (int i = 0; i < rt->h_sgd.count; i++) {
        uint32_t n = 0;
        if (size < sizeof(uint32_t)) return 0;
        memcpy(&n, src, sizeof(n)); src += sizeof(n); size -= sizeof(n);
        if (n != (uint32_t)rt->h_sgd.size[i] ||
            (size_t)n > size / sizeof(float)) return 0;
        float *h = (float*)malloc((size_t)n * sizeof(float));
        if (n > 0 && !h) return 0;
        memcpy(h, src, (size_t)n * sizeof(float));
        src += (size_t)n * sizeof(float);
        size -= (size_t)n * sizeof(float);
        CUDA_CHECK(cudaMemcpy(rt->h_sgd.master[i], h, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
        if (rt->h_sgd.fp16_w[i]) {
            k_f32_to_f16<<<(n + 255) / 256, 256>>>(rt->h_sgd.master[i], rt->h_sgd.fp16_w[i], n);
        }
        free(h);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return 1;
}

int tkm_save_weights(TkmRuntime *rt, const char *bin_path) {
    if (!rt || !bin_path) return 0;
    size_t size = tkm_runtime_weight_bytes(rt);
    if (size == 0) return 0;
    uint8_t *buffer = (uint8_t*)malloc(size);
    if (!buffer || !tkm_save_weights_buffer(rt, buffer, size, NULL)) {
        free(buffer);
        return 0;
    }
    FILE *f = fopen(bin_path, "wb");
    if (!f) { perror(bin_path); free(buffer); return 0; }
    int ok = fwrite(buffer, 1, size, f) == size;
    fclose(f);
    free(buffer);
    if (!ok) return 0;
    printf("  [TKM I/O] Saved %d learnable tensors (%d params) to '%s'\n",
           rt->h_sgd.count, rt->total_sgd_elems, bin_path);
    return 1;
}

int tkm_load_weights(TkmRuntime *rt, const char *bin_path) {
    if (!rt || !bin_path) return 0;
    FILE *f = fopen(bin_path, "rb");
    if (!f) { perror(bin_path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long file_size = ftell(f);
    if (file_size <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }
    size_t size = (size_t)file_size;
    uint8_t *buffer = (uint8_t*)malloc(size);
    if (!buffer || fread(buffer, 1, size, f) != size) {
        free(buffer);
        fclose(f);
        return 0;
    }
    fclose(f);
    int ok = tkm_load_weights_buffer(rt, buffer, size);
    free(buffer);
    if (ok) {
        printf("  [TKM I/O] Loaded %d learnable tensors (%d params) from '%s'\n",
               rt->h_sgd.count, rt->total_sgd_elems, bin_path);
    }
    return ok;
}

int tkm_runtime_train_dataset(TkmRuntime *rt, TkmDatasetNext next_batch,
                              void *user, size_t max_batches, float lr,
                              float *out_average_loss) {
    if (out_average_loss) *out_average_loss = 0.0f;
    if (!rt || !next_batch || rt->graph.batch_size <= 0) return 0;
    size_t input_elems = tkm_runtime_input_elements(rt);
    float *input = (float*)malloc(input_elems * sizeof(float));
    uint8_t *labels = (uint8_t*)malloc((size_t)rt->graph.batch_size * sizeof(uint8_t));
    if (!input || !labels) { free(input); free(labels); return 0; }
    size_t batches = 0;
    double loss_sum = 0.0;
    int result = 1;
    while (max_batches == 0 || batches < max_batches) {
        size_t count = 0;
        int status = next_batch(user, input, labels, (size_t)rt->graph.batch_size, &count);
        if (status == 0) break;
        if (status < 0 || count != (size_t)rt->graph.batch_size) { result = 0; break; }
        loss_sum += tkm_runtime_step(rt, input, labels, lr);
        batches++;
    }
    if (result && batches > 0 && out_average_loss) *out_average_loss = (float)(loss_sum / batches);
    free(input);
    free(labels);
    return result && batches > 0;
}

static float tkm_epoch_lr(float peak_lr, float min_lr_ratio, size_t step,
                          size_t total_steps, size_t warmup_steps) {
    float warmup = (warmup_steps > 0 && step < warmup_steps)
        ? (float)(step + 1) / (float)warmup_steps : 1.0f;
    float progress = total_steps > 1
        ? fminf((float)step / (float)(total_steps - 1), 1.0f) : 1.0f;
    float cosine = 0.5f * (1.0f + cosf(3.14159265358979323846f * progress));
    return peak_lr * warmup * (min_lr_ratio + (1.0f - min_lr_ratio) * cosine);
}

TkmDataset *tkm_dataset_create(TkmRuntime *rt, const float *h_images,
                               const uint8_t *h_labels, size_t num_samples) {
    if (!rt || !rt->training_enabled || !h_images || !h_labels || num_samples == 0) return NULL;
    TkmDataset *ds = (TkmDataset*)calloc(1, sizeof(*ds));
    if (!ds) return NULL;
    ds->rt = rt;
    ds->h_images = h_images;
    ds->h_labels = h_labels;
    ds->num_samples = num_samples;
    ds->sample_elems = (size_t)rt->graph.nodes[0].H * rt->graph.nodes[0].W * rt->graph.nodes[0].C;

    if (num_samples > SIZE_MAX / ds->sample_elems ||
        num_samples * ds->sample_elems > (size_t)INT_MAX) {
        free(ds);
        return NULL;
    }
    CUDA_CHECK(cudaMallocHost(&ds->h_pin_order, num_samples * sizeof(uint32_t)));
    size_t fp16_bytes = num_samples * ds->sample_elems * sizeof(__half);
    if (fp16_bytes <= SIZE_MAX - num_samples) fp16_bytes += num_samples;
    size_t free_vram = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
    ds->resident = fp16_bytes <= free_vram / 2;

    if (ds->resident) {
        __half *h_images_h = (__half*)malloc(num_samples * ds->sample_elems * sizeof(__half));
        if (!h_images_h) { tkm_dataset_free(ds); return NULL; }
        for (size_t i = 0; i < num_samples * ds->sample_elems; i++)
            h_images_h[i] = __float2half(h_images[i]);
        CUDA_CHECK(cudaMalloc(&ds->d_images_h, num_samples * ds->sample_elems * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&ds->d_labels_u8, num_samples * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&ds->d_order, num_samples * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds->d_batch_idx, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds->d_global_step, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&ds->d_sched, sizeof(TkmSchedule)));
        CUDA_CHECK(cudaMemcpy(ds->d_images_h, h_images_h,
                              num_samples * ds->sample_elems * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ds->d_labels_u8, h_labels, num_samples * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        free(h_images_h);

        CUDA_CHECK(cudaStreamSynchronize(rt->stream));
        CUDA_CHECK(cudaStreamBeginCapture(rt->stream, cudaStreamCaptureModeGlobal));
        k_resident_gather<<<rt->graph.batch_size, 256, 0, rt->stream>>>(
            ds->d_images_h, ds->d_labels_u8, ds->d_order,
            (__half*)(rt->d_arena + rt->graph.nodes[0].out_offset), rt->d_labels,
            ds->d_batch_idx, (int)ds->sample_elems, rt->graph.batch_size);
        k_resident_advance_lr<<<1, 1, 0, rt->stream>>>(
            rt->d_lr, ds->d_batch_idx, ds->d_global_step, ds->d_sched);
        record_full_step_stream(rt, 0);
        CUDA_CHECK(cudaStreamEndCapture(rt->stream, &ds->resident_graph));
        CUDA_CHECK(cudaGraphInstantiate(&ds->resident_exec, ds->resident_graph, NULL, NULL, 0));
    } else {
        size_t in_bytes = (size_t)rt->graph.batch_size * ds->sample_elems * sizeof(float);
        for (int slot = 0; slot < 2; slot++) {
            CUDA_CHECK(cudaMallocHost(&ds->h_pin_in[slot], in_bytes));
            CUDA_CHECK(cudaMallocHost(&ds->h_pin_lbl[slot], (size_t)rt->graph.batch_size * sizeof(uint8_t)));
            CUDA_CHECK(cudaMallocHost(&ds->h_pin_lr[slot], sizeof(float)));
            CUDA_CHECK(cudaEventCreateWithFlags(&ds->slot_done[slot], cudaEventDisableTiming));
            *ds->h_pin_lr[slot] = 0.0f;
            CUDA_CHECK(cudaStreamSynchronize(rt->stream));
            CUDA_CHECK(cudaStreamBeginCapture(rt->stream, cudaStreamCaptureModeGlobal));
            CUDA_CHECK(cudaMemcpyAsync(rt->d_input_f32, ds->h_pin_in[slot], in_bytes,
                                       cudaMemcpyHostToDevice, rt->stream));
            CUDA_CHECK(cudaMemcpyAsync(rt->d_labels, ds->h_pin_lbl[slot],
                                       (size_t)rt->graph.batch_size * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, rt->stream));
            CUDA_CHECK(cudaMemcpyAsync(rt->d_lr, ds->h_pin_lr[slot], sizeof(float),
                                       cudaMemcpyHostToDevice, rt->stream));
            k_f32_to_f16<<<((int)rt->graph.batch_size * (int)ds->sample_elems + 255) / 256,
                           256, 0, rt->stream>>>(
                rt->d_input_f32,
                (__half*)(rt->d_arena + rt->graph.nodes[0].out_offset),
                (int)((size_t)rt->graph.batch_size * ds->sample_elems));
            record_full_step_stream(rt, 0);
            CUDA_CHECK(cudaStreamEndCapture(rt->stream, &ds->stream_graph[slot]));
            CUDA_CHECK(cudaGraphInstantiate(&ds->stream_exec[slot], ds->stream_graph[slot], NULL, NULL, 0));
        }
    }
    ds->graphs_captured = 1;
    return ds;
}

void tkm_dataset_free(TkmDataset *ds) {
    if (!ds) return;
    if (ds->resident_exec) cudaGraphExecDestroy(ds->resident_exec);
    if (ds->resident_graph) cudaGraphDestroy(ds->resident_graph);
    for (int slot = 0; slot < 2; slot++) {
        if (ds->stream_exec[slot]) cudaGraphExecDestroy(ds->stream_exec[slot]);
        if (ds->stream_graph[slot]) cudaGraphDestroy(ds->stream_graph[slot]);
        if (ds->slot_done[slot]) cudaEventDestroy(ds->slot_done[slot]);
        if (ds->h_pin_in[slot]) cudaFreeHost(ds->h_pin_in[slot]);
        if (ds->h_pin_lbl[slot]) cudaFreeHost(ds->h_pin_lbl[slot]);
        if (ds->h_pin_lr[slot]) cudaFreeHost(ds->h_pin_lr[slot]);
    }
    if (ds->d_images_h) cudaFree(ds->d_images_h);
    if (ds->d_labels_u8) cudaFree(ds->d_labels_u8);
    if (ds->d_order) cudaFree(ds->d_order);
    if (ds->d_batch_idx) cudaFree(ds->d_batch_idx);
    if (ds->d_global_step) cudaFree(ds->d_global_step);
    if (ds->d_sched) cudaFree(ds->d_sched);
    if (ds->h_pin_order) cudaFreeHost(ds->h_pin_order);
    free(ds);
}

int tkm_dataset_is_resident(const TkmDataset *ds) {
    return ds ? ds->resident : 0;
}

int tkm_runtime_train_epoch(TkmRuntime *rt, TkmDataset *ds,
                            const uint32_t *h_order, float peak_lr,
                            float min_lr_ratio, size_t global_step_start,
                            size_t total_steps, size_t warmup_steps,
                            float *out_average_loss) {
    if (out_average_loss) *out_average_loss = 0.0f;
    if (!rt || !ds || ds->rt != rt || !ds->graphs_captured ||
        total_steps == 0 || peak_lr <= 0.0f || min_lr_ratio < 0.0f ||
        min_lr_ratio > 1.0f) return 0;
    size_t batch_size = (size_t)rt->graph.batch_size;
    size_t full_batches = ds->num_samples / batch_size;
    if (full_batches == 0 || global_step_start > UINT32_MAX ||
        total_steps > UINT32_MAX || warmup_steps > UINT32_MAX) return 0;

    if (h_order) memcpy(ds->h_pin_order, h_order, ds->num_samples * sizeof(uint32_t));
    else for (size_t i = 0; i < ds->num_samples; i++) ds->h_pin_order[i] = (uint32_t)i;

    CUDA_CHECK(cudaMemsetAsync(rt->d_loss_buf, 0, sizeof(float), rt->stream));
    if (ds->resident) {
        TkmSchedule sched = { peak_lr, min_lr_ratio, (uint32_t)total_steps, (uint32_t)warmup_steps };
        CUDA_CHECK(cudaMemcpyAsync(ds->d_order, ds->h_pin_order,
                                   ds->num_samples * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, rt->stream));
        uint32_t zero = 0, start = (uint32_t)global_step_start;
        CUDA_CHECK(cudaMemcpyAsync(ds->d_batch_idx, &zero, sizeof(zero),
                                   cudaMemcpyHostToDevice, rt->stream));
        CUDA_CHECK(cudaMemcpyAsync(ds->d_global_step, &start, sizeof(start),
                                   cudaMemcpyHostToDevice, rt->stream));
        CUDA_CHECK(cudaMemcpyAsync(ds->d_sched, &sched, sizeof(sched),
                                   cudaMemcpyHostToDevice, rt->stream));
        for (size_t b = 0; b < full_batches; b++)
            CUDA_CHECK(cudaGraphLaunch(ds->resident_exec, rt->stream));
    } else {
        size_t sample_bytes = ds->sample_elems * sizeof(float);
        for (size_t b = 0; b < full_batches; b++) {
            int slot = (int)(b & 1u);
            if (b >= 2) CUDA_CHECK(cudaEventSynchronize(ds->slot_done[slot]));
            for (size_t row = 0; row < batch_size; row++) {
                uint32_t source = ds->h_pin_order[b * batch_size + row];
                memcpy(ds->h_pin_in[slot] + row * ds->sample_elems,
                       ds->h_images + (size_t)source * ds->sample_elems, sample_bytes);
                ds->h_pin_lbl[slot][row] = ds->h_labels[source];
            }
            *ds->h_pin_lr[slot] = tkm_epoch_lr(peak_lr, min_lr_ratio,
                                               global_step_start + b,
                                               total_steps, warmup_steps);
            CUDA_CHECK(cudaGraphLaunch(ds->stream_exec[slot], rt->stream));
            CUDA_CHECK(cudaEventRecord(ds->slot_done[slot], rt->stream));
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(rt->h_pin_loss, rt->d_loss_buf, sizeof(float),
                               cudaMemcpyDeviceToHost, rt->stream));
    CUDA_CHECK(cudaStreamSynchronize(rt->stream));
    if (out_average_loss) *out_average_loss = *rt->h_pin_loss / (float)full_batches;
    return 1;
}

int tkm_runtime_eval_dataset(TkmRuntime *rt, TkmDatasetNext next_batch,
                             void *user, size_t max_batches,
                             int *out_correct, size_t *out_samples) {
    if (out_correct) *out_correct = 0;
    if (out_samples) *out_samples = 0;
    if (!rt || !next_batch || rt->graph.batch_size <= 0 ||
        tkm_runtime_input_elements(rt) > (size_t)INT_MAX) return 0;
    size_t input_elems = tkm_runtime_input_elements(rt);
    float *input = (float*)malloc(input_elems * sizeof(float));
    uint8_t *labels = (uint8_t*)malloc((size_t)rt->graph.batch_size * sizeof(uint8_t));
    if (!input || !labels) { free(input); free(labels); return 0; }
    size_t batches = 0, samples = 0;
    int correct = 0, result = 1;
    while (max_batches == 0 || batches < max_batches) {
        size_t count = 0;
        int status = next_batch(user, input, labels, (size_t)rt->graph.batch_size, &count);
        if (status == 0) break;
        if (status < 0 || count == 0 || count > (size_t)rt->graph.batch_size ||
            samples > SIZE_MAX - count || count > (size_t)INT_MAX) { result = 0; break; }
        correct += tkm_runtime_eval(rt, input, labels, (int)count);
        samples += count;
        batches++;
    }
    if (out_correct) *out_correct = correct;
    if (out_samples) *out_samples = samples;
    free(input);
    free(labels);
    return result && batches > 0;
}

void tkm_runtime_free(TkmRuntime *rt) {
    if (!rt) return;
    cudaStreamSynchronize(rt->stream);
    if (rt->step_exec)    cudaGraphExecDestroy(rt->step_exec);
    if (rt->step_graph)   cudaGraphDestroy(rt->step_graph);
    if (rt->d_arena)      cudaFree(rt->d_arena);
    if (rt->d_grad_arena) cudaFree(rt->d_grad_arena);
    if (rt->d_input_f32)  cudaFree(rt->d_input_f32);
    if (rt->d_labels)     cudaFree(rt->d_labels);
    if (rt->d_col_ws)     cudaFree(rt->d_col_ws);
    if (rt->d_cublas_ws)  cudaFree(rt->d_cublas_ws);
    if (rt->h_pin_loss)   cudaFreeHost(rt->h_pin_loss);
    if (rt->h_pin_lr)     cudaFreeHost(rt->h_pin_lr);
    for (int i = 0; i < TKM_MAX_NODES; i++) {
        NodeGpuState *st = &rt->nstate[i];
        if (st->d_W_master) cudaFree(st->d_W_master);
        if (st->d_W_h)      cudaFree(st->d_W_h);
        if (st->d_vW)       cudaFree(st->d_vW);
        if (st->d_dW)       cudaFree(st->d_dW);
        if (st->d_bias)     cudaFree(st->d_bias);
        if (st->d_vbias)    cudaFree(st->d_vbias);
        if (st->d_dbias)    cudaFree(st->d_dbias);
        if (st->d_w_vec)    cudaFree(st->d_w_vec);
        if (st->d_vw_vec)   cudaFree(st->d_vw_vec);
        if (st->d_dw_vec)   cudaFree(st->d_dw_vec);
        if (st->d_beta)     cudaFree(st->d_beta);
        if (st->d_vbeta)    cudaFree(st->d_vbeta);
        if (st->d_dbeta)    cudaFree(st->d_dbeta);
        if (st->d_save_istd) cudaFree(st->d_save_istd);
        if (st->d_save_xhat) cudaFree(st->d_save_xhat);
        if (st->d_save_mask) cudaFree(st->d_save_mask);
    }
    cublasDestroy(rt->cublas);
    cudaStreamDestroy(rt->stream);
    free(rt);
}