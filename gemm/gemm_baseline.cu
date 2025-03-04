#include "gemm_baseline.h"

#include <cassert>

namespace gemm_baseline {

// Old fashioned poorly-optimized (but still tiled) GPU kernel; output stationary.

constexpr unsigned SMEM_M = 128;
constexpr unsigned SMEM_N = 128;
constexpr unsigned SMEM_K = 16;

constexpr unsigned THREAD_M = 8;
constexpr unsigned THREAD_N = 8;
constexpr unsigned CTA_SIZE = (SMEM_M / THREAD_M) * (SMEM_N / THREAD_N);

// CTA cooperates to fill SMEM_M x SMEM_N c output tile at given (cta_m_offset, cta_n_offset) offset.
template <bool a_col_major, bool b_col_major, bool c_col_major>
__device__ void device_task(GPU_Tensors t, unsigned cta_m_offset, unsigned cta_n_offset)
{
    assert(t.a_col_major == a_col_major);
    assert(t.b_col_major == b_col_major);
    assert(t.c_col_major == c_col_major);

    __shared__ float smem_a_mk[SMEM_M][SMEM_K];
    __shared__ float smem_b_kn[SMEM_K][SMEM_N];

    // Per-thread tile to accumulate + offset within block tile.
    const uint32_t thr_m_offset = THREAD_M * (threadIdx.x % (SMEM_M / THREAD_M));
    const uint32_t thr_n_offset = THREAD_N * (threadIdx.x / (SMEM_M / THREAD_M));
    float thread_accum_mn[THREAD_M][THREAD_N];
    for (unsigned m_seq = 0; m_seq < THREAD_M; ++m_seq) {
        for (unsigned n_seq = 0; n_seq < THREAD_N; ++n_seq) {
            thread_accum_mn[m_seq][n_seq] = 0.0f;
        }
    }

    for (unsigned k_seq_offset = 0; k_seq_offset < t.K; k_seq_offset += SMEM_K) {
        // Load tiles (predicated).
        for (unsigned i = threadIdx.x; i < SMEM_M * SMEM_K; i += blockDim.x) {
            unsigned k0 = i % SMEM_K;
            unsigned m0 = i / SMEM_K;
            float val = 0;
            unsigned m = cta_m_offset + m0;
            unsigned k = k_seq_offset + k0;
            if (m < t.M && k < t.K) {
                if (a_col_major) {
                    val = t.a[k * t.M + m];
                }
                else {
                    val = t.a[m * t.K + k];
                }
            }
            smem_a_mk[m0][k0] = val;
        }
        for (unsigned i = threadIdx.x; i < SMEM_K * SMEM_N; i += blockDim.x) {
            unsigned n0 = i % SMEM_N;
            unsigned k0 = i / SMEM_N;
            float val = 0;
            unsigned k = k_seq_offset + k0;
            unsigned n = cta_n_offset + n0;
            if (k < t.K && n < t.N) {
                if (b_col_major) {
                    val = t.b[n * t.K + k];
                }
                else {
                    val = t.b[k * t.N + n];
                }
            }
            smem_b_kn[k0][n0] = val;
        }
        __syncthreads();

        // Accumulate tiles (very inefficient).
        for (unsigned m_seq = 0; m_seq < THREAD_M; ++m_seq) {
            for (unsigned n_seq = 0; n_seq < THREAD_N; ++n_seq) {
                for (unsigned k0 = 0; k0 < SMEM_K; ++k0) {
                    const float val_a = smem_a_mk[thr_m_offset + m_seq][k0];
                    const float val_b = smem_b_kn[k0][thr_n_offset + n_seq];
                    thread_accum_mn[m_seq][n_seq] += val_a * val_b;
                }
            }
        }
        __syncthreads();
    }
    // End k loop

    // Store tile (predicated).
    for (unsigned m_seq = 0; m_seq < THREAD_M; ++m_seq) {
        for (unsigned n_seq = 0; n_seq < THREAD_N; ++n_seq) {
            unsigned m = cta_m_offset + thr_m_offset + m_seq;
            unsigned n = cta_n_offset + thr_n_offset + n_seq;
            if (m < t.M && n < t.N) {
                if (c_col_major) {
                    t.c[n * t.M + m] = thread_accum_mn[m_seq][n_seq];
                }
                else {
                    t.c[m * t.N + n] = thread_accum_mn[m_seq][n_seq];
                }
            }
        }
    }
}

template <bool a_col_major, bool b_col_major, bool c_col_major>
__launch_bounds__(CTA_SIZE)
__global__ void device_kernel(GPU_Tensors t)
{
    const unsigned m_tiles = (t.M + SMEM_M - 1) / SMEM_M;
    const unsigned n_tiles = (t.N + SMEM_N - 1) / SMEM_N;
    for (unsigned i = blockIdx.x; i < m_tiles * n_tiles; i += gridDim.x) {
        const unsigned cta_m_offset = (i / n_tiles) * SMEM_M;
        const unsigned cta_n_offset = (i % n_tiles) * SMEM_N;
        device_task<a_col_major, b_col_major, c_col_major>(t, cta_m_offset, cta_n_offset);
    }
}

template <bool a_col_major, bool b_col_major, bool c_col_major>
void launch(GPU_Tensors t, cudaStream_t stream)
{
    assert(t.a_col_major == a_col_major);
    assert(t.b_col_major == b_col_major);
    assert(t.c_col_major == c_col_major);
    device_kernel<a_col_major, b_col_major, c_col_major> <<<48, CTA_SIZE, 0, stream>>>(t);
}

}  // end namespace

void matmul_baseline(GPU_Tensors t, cudaStream_t stream)
{
    using namespace gemm_baseline;
    if (t.a_col_major) {
        if (t.b_col_major) {
            if (t.c_col_major) {
                launch<1, 1, 1>(t, stream);
            }
            else {
                launch<1, 1, 0>(t, stream);
            }
        }
        else {
            if (t.c_col_major) {
                launch<1, 0, 1>(t, stream);
            }
            else {
                launch<1, 0, 0>(t, stream);
            }
        }
    }
    else {
        if (t.b_col_major) {
            if (t.c_col_major) {
                launch<0, 1, 1>(t, stream);
            }
            else {
                launch<0, 1, 0>(t, stream);
            }
        }
        else {
            if (t.c_col_major) {
                launch<0, 0, 1>(t, stream);
            }
            else {
                launch<0, 0, 0>(t, stream);
            }
        }
    }
}
