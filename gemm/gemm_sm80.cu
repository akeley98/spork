#include "gemm_sm80.h"

#include <cassert>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include <cuda/std/array>

#include "gpu_tensor.h"

namespace {

__device__ __forceinline__ void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES) : "memory");
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

struct WmmaA : cuda::std::array<unsigned, 4>
{
};

struct WmmaB : cuda::std::array<unsigned, 2>
{
};

struct WmmaD : cuda::std::array<unsigned, 4>
{
};

inline __device__ void load_a(WmmaA& rmem, const float* gmem, cuda::std::array<unsigned, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    const float* gmem_thread_baseaddr = &gmem[warp_lane / 4u * row_stride + warp_lane % 4u * col_stride];
    rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
    rmem[1] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride]);
    rmem[2] = __float_as_uint(gmem_thread_baseaddr[4 * col_stride]);
    rmem[3] = __float_as_uint(gmem_thread_baseaddr[8 * row_stride + 4 * col_stride]);
}

inline __device__ void load_b(WmmaB& rmem, const float* gmem, cuda::std::array<unsigned, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
    rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
    rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

inline __device__ void store_d(float* gmem, WmmaD rmem, cuda::std::array<unsigned, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    float* gmem_thread_baseaddr = &gmem[(warp_lane / 4u) * row_stride + (warp_lane % 4u) * 2u * col_stride];
    gmem_thread_baseaddr[0] = __uint_as_float(rmem[0]);
    gmem_thread_baseaddr[col_stride] = __uint_as_float(rmem[1]);
    gmem_thread_baseaddr[8 * row_stride] = __uint_as_float(rmem[2]);
    gmem_thread_baseaddr[8 * row_stride + col_stride] = __uint_as_float(rmem[3]);
}

inline __device__ WmmaD wmma(WmmaA a, WmmaB b, WmmaD c)
{
    WmmaD d;
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\t"
        "{%0,%1,%2,%3},\n\t"
        "{%4,%5,%6,%7},\n\t"
        "{%8,%9},\n\t"
        "{%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    return d;
}

/* TODO: your GPU kernels here... */

template <uint32_t SMEM_I, uint32_t SMEM_J, uint32_t SMEM_K, uint32_t CTA_MODULUS>
struct TiledMultiplier
{
    uint32_t size_i, size_j, size_k;
    float const* a;
    float const* b;
    float* c;

    static constexpr uint32_t WARP_I = 16;
    static constexpr uint32_t WARP_J = 16;
    static constexpr uint32_t WARP_K = 8;

    // One buffer of double buffer.
    struct alignas(float4) Buffers
    {
        float a[SMEM_I][SMEM_K];  // Stores SMEM_I × SMEM_K of a
        float b[SMEM_K][SMEM_J];  // Stores SMEM_K × SMEM_J of b
    };

    __host__ __device__ static constexpr uint32_t smem_size()
    {
        return sizeof(Buffers) * 2u;
    }

    __host__ __device__ static constexpr uint32_t cta_size()
    {
        // If output matrix is cut into (WARP_I, WARP_J) blocks, one warp handles one matrix block.
        static_assert(SMEM_I % WARP_I == 0);
        static_assert(SMEM_J % WARP_J == 0);
        return (SMEM_I / WARP_I) * (SMEM_J / WARP_J) * 32;
    }

    // If output matrix is cut into (SMEM_I, SMEM_J) blocks, one CTA handles one matrix block.

    __host__ __device__ uint32_t i_cta() const
    {
        assert(size_i % SMEM_I == 0);
        return size_i / SMEM_I;
    }

    __host__ __device__ uint32_t j_cta() const
    {
        assert(size_j % SMEM_J == 0);
        return size_j / SMEM_J;
    }

    // Fill shared memory A tile with SMEM_I×SMEM_K block starting at (cta_i_offset, cta_k_offset)
    // Fill shared memory B tile with SMEM_K×SMEM_J block starting at (cta_k_offset, cta_j_offset)
    __device__ void cta_async_load_block(Buffers& buffers, uint32_t cta_i_offset,
                                         uint32_t cta_j_offset, uint32_t cta_k_offset) const
    {
        for (unsigned work_idx = threadIdx.x; work_idx < SMEM_I * SMEM_K / 4u; work_idx += blockDim.x) {
            const unsigned thr_i_offset = work_idx / (SMEM_K / 4u);
            const unsigned thr_k_offset = 4 * (work_idx % (SMEM_K / 4u));
            cp_async4(&buffers.a[thr_i_offset][thr_k_offset],
                      &a[(thr_i_offset + cta_i_offset) * size_k + cta_k_offset + thr_k_offset]);
        }
        for (unsigned work_idx = threadIdx.x; work_idx < SMEM_K * SMEM_J / 4u; work_idx += blockDim.x) {
            const unsigned thr_k_offset = work_idx / (SMEM_J / 4u);
            const unsigned thr_j_offset = 4 * (work_idx % (SMEM_J / 4u));
            cp_async4(&buffers.b[thr_k_offset][thr_j_offset],
                      &b[(thr_k_offset + cta_k_offset) * size_j + cta_j_offset + thr_j_offset]);
        }
    }

    // Static assignment of warps within CTA to per-warp output tiles (WARP_I, WARP_J) within
    // per-CTA output tile (SMEM_I, SMEM_J)
    __device__ uint32_t get_warp_i_idx() const
    {
        const uint32_t warp_index = threadIdx.x / 32u;
        return warp_index / (SMEM_J / WARP_J);
    }

    __device__ uint32_t get_warp_j_idx() const
    {
        const uint32_t warp_index = threadIdx.x / 32u;
        return warp_index % (SMEM_J / WARP_J);
    }

    // Accumulate warp-assigned matrix tile based on values in shared memory buffers.
    __device__ void warp_accum_block(WmmaD& accum0, WmmaD& accum1, const Buffers& buffers) const
    {
        const uint32_t warp_i_idx = get_warp_i_idx();
        const uint32_t warp_j_idx = get_warp_j_idx();

        for (uint32_t warp_k_idx = 0; warp_k_idx < SMEM_K / WARP_K; ++warp_k_idx) {
            WmmaA wmma_a;
            WmmaB wmma_b0, wmma_b1;
            load_a(wmma_a, &buffers.a[warp_i_idx * WARP_I][warp_k_idx * WARP_K], {SMEM_K, 1});
            load_b(wmma_b0, &buffers.b[warp_k_idx * WARP_K][warp_j_idx * WARP_J], {SMEM_J, 1});
            load_b(wmma_b1, &buffers.b[warp_k_idx * WARP_K][warp_j_idx * WARP_J + 8], {SMEM_J, 1});
            accum0 = wmma(wmma_a, wmma_b0, accum0);
            accum1 = wmma(wmma_a, wmma_b1, accum1);
        }
    }

    // CTA cooperates to fill the output matrix block of size (SMEM_I, SMEM_J) starting at (cta_i_offset, cta_j_offset).
    // Requires smem-allocated double buffer.
    __device__ void cta_compute_block(uint32_t cta_i_offset, uint32_t cta_j_offset, Buffers buffers[2]) const
    {
        assert(cta_i_offset % SMEM_I == 0);
        assert(cta_j_offset % SMEM_J == 0);
        assert(size_k % SMEM_K == 0);

        const uint32_t k_blk_dim = size_k / SMEM_K;
        WmmaD accum0{{0, 0, 0, 0}};
        WmmaD accum1{{0, 0, 0, 0}};

        for (uint32_t cta_k_idx = 0; cta_k_idx <= k_blk_dim; ++cta_k_idx) {
            if (cta_k_idx < k_blk_dim) {
                cta_async_load_block(buffers[cta_k_idx % 2u], cta_i_offset, cta_j_offset, cta_k_idx * SMEM_K);
            }
            if (cta_k_idx > 0) {
                // Accumulate smem buffer filled in previous iteration. NB <= in for loop is critical for this!
                warp_accum_block(accum0, accum1, buffers[(cta_k_idx % 2u) ^ 1u]);
            }
            async_memcpy_waitall();
            __syncthreads();
        }

        const uint32_t warp_i_offset = get_warp_i_idx() * WARP_I + cta_i_offset;
        const uint32_t warp_j_offset = get_warp_j_idx() * WARP_J + cta_j_offset;
        store_d(c + warp_i_offset * size_j + warp_j_offset, accum0, {size_j, 1});
        store_d(c + warp_i_offset * size_j + (warp_j_offset + 8), accum1, {size_j, 1});
    }

    __device__ void kernel_main()
    {
        assert(gridDim.x == i_cta() * j_cta());
        assert(blockDim.x == cta_size());

        const uint32_t cta_rows = size_i / SMEM_I;
        const uint32_t cta_cols = size_j / SMEM_J;
        const uint32_t cta_col_remainder = cta_cols % CTA_MODULUS;
        const uint32_t superblock_count = cta_cols / CTA_MODULUS;
        const uint32_t superblock_cta_count = cta_rows * CTA_MODULUS;
        const uint32_t superblock_idx = blockIdx.x / superblock_cta_count;
        const uint32_t cta_index_in_superblock = blockIdx.x % superblock_cta_count;

        uint32_t cta_i_idx, cta_j_idx;

        if (superblock_idx < superblock_count) {
            cta_i_idx = cta_index_in_superblock / CTA_MODULUS;
            cta_j_idx = cta_index_in_superblock % CTA_MODULUS + CTA_MODULUS * superblock_idx;
        }
        else {
            assert(superblock_idx == superblock_count);
            cta_i_idx = cta_index_in_superblock / cta_col_remainder;
            cta_j_idx = cta_index_in_superblock % cta_col_remainder + CTA_MODULUS * superblock_idx;
        }
        assert(cta_i_idx < cta_rows);
        assert(cta_j_idx < cta_cols);

        extern __shared__ char smem[];
        cta_compute_block(cta_i_idx * SMEM_I, cta_j_idx * SMEM_J, reinterpret_cast<Buffers*>(smem));
    }

    void launch(cudaStream_t stream);
};

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size())
tiled_multiplier_kernel(Multiplier multiplier)
{
    multiplier.kernel_main();
}

template <uint32_t SMEM_I, uint32_t SMEM_J, uint32_t SMEM_K, uint32_t CTA_MODULUS>
void TiledMultiplier<SMEM_I, SMEM_J, SMEM_K, CTA_MODULUS>::launch(cudaStream_t stream)
{
    using Multiplier = std::remove_reference_t<decltype(*this)>;
    const dim3 grid{i_cta() * j_cta(), 1, 1};
    const uint32_t block = cta_size();
    const uint32_t smem = smem_size();
    cudaFuncSetAttribute(tiled_multiplier_kernel<Multiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tiled_multiplier_kernel<Multiplier> <<<grid, block, smem, stream>>>(*this);
}

}  // end namespace

void matmul_sm80(GPU_Tensors t, cudaStream_t stream) {
    constexpr uint32_t smem_i = 64;
    constexpr uint32_t smem_j = 64;
    constexpr uint32_t smem_k = 16;
    constexpr uint32_t cta_modulus = 4;

    const uint32_t size_i = t.M;
    const uint32_t size_j = t.N;
    const uint32_t size_k = t.K;

    assert(!t.a_col_major);
    assert(!t.b_col_major);
    assert(!t.c_col_major);

    if (size_i % smem_i == 0 && size_j % smem_j == 0 && size_k % smem_k == 0) {
        TiledMultiplier<smem_i, smem_j, smem_k, cta_modulus> multiplier{size_i, size_j, size_k, t.a, t.b, t.c};
        multiplier.launch(stream);
    }
    else {
        assert(size_i % 16 == 0);
        assert(size_j % 16 == 0);
        assert(size_k % 16 == 0);
        TiledMultiplier<16, 16, 16, cta_modulus> multiplier{size_i, size_j, size_k, t.a, t.b, t.c};
        multiplier.launch(stream);
    }
}
