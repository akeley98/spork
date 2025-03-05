#include "gemm_sm80.h"

#include <cassert>
#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include <cuda/std/array>

#include "gpu_tensor.h"

// Sorry:
//   I = M
//   J = N
// Since we inherited code from 6.S894 lab6
//
// fyi THIS IS ALL WILLIAM BRANDON'S FAULT

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

inline __device__ void wmma(WmmaD& d, WmmaA a, WmmaB b)
{
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n\t"
        "{%0,%1,%2,%3},\n\t"
        "{%4,%5,%6,%7},\n\t"
        "{%8,%9},\n\t"
        "{%10,%11,%12,%13};" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));
}
static constexpr uint32_t MMA_I = 16;
static constexpr uint32_t MMA_J = 8;
static constexpr uint32_t MMA_K = 8;

/* TODO: your GPU kernels here... */

struct TileConfig
{
    uint32_t smem_i, smem_j, smem_k;
    uint32_t warp_i, warp_j;
    uint32_t cta_modulus;
};

template <TileConfig tile_config>
struct TiledMultiplier
{
    uint32_t size_i, size_j, size_k;
    float const* a;
    float const* b;
    float* c;

    static constexpr uint32_t SMEM_I = tile_config.smem_i;
    static constexpr uint32_t SMEM_J = tile_config.smem_j;
    static constexpr uint32_t SMEM_K = tile_config.smem_k;
    static constexpr uint32_t WARP_I = tile_config.warp_i;
    static constexpr uint32_t WARP_J = tile_config.warp_j;
    static constexpr uint32_t CTA_MODULUS = tile_config.cta_modulus;

    // One buffer of ring buffer.
    struct alignas(float4) Buffers
    {
        float a[SMEM_I][SMEM_K];  // Stores SMEM_I × SMEM_K of a
        float b[SMEM_K][SMEM_J];  // Stores SMEM_K × SMEM_J of b
    };

    struct SmemLayout
    {
        Buffers buffers[2];  // TODO ring buffer
    };

    // Accumulator tiles per warp
    struct WarpAccum
    {
        static_assert(WARP_I % MMA_I == 0);
        static_assert(WARP_J % MMA_J == 0);
        WmmaD ij[WARP_I / MMA_I][WARP_J / MMA_J];
    };

    __host__ __device__ static constexpr uint32_t grid_size()
    {
        return 96;
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
    __device__ void warp_accum_block(WarpAccum& accum, const Buffers& buffers) const
    {
        const uint32_t warp_i_idx = get_warp_i_idx();
        const uint32_t warp_j_idx = get_warp_j_idx();

        // B tiles pre-cached in registers for re-use.
        WmmaB mma_b_kj[SMEM_K / MMA_K][WARP_J / MMA_J];
        for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
            for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                const uint32_t j = warp_j_idx*WARP_J + mma_j_idx*MMA_J;
                const uint32_t k = mma_k_idx*MMA_K;
                load_b(mma_b_kj[mma_k_idx][mma_j_idx], &buffers.b[k][j], {SMEM_J, 1});
            }
        }

        for (uint32_t mma_i_idx = 0; mma_i_idx < WARP_I / MMA_I; ++mma_i_idx) {
            // Load A tiles needed.
            WmmaA mma_a_k[SMEM_K / MMA_K];
            const uint32_t i = warp_i_idx*WARP_I + mma_i_idx*MMA_I;
            for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                const uint32_t k = mma_k_idx*MMA_K;
                load_a(mma_a_k[mma_k_idx], &buffers.a[i][k], {SMEM_K, 1});
            }
            for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
                for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                    wmma(accum.ij[mma_i_idx][mma_j_idx], mma_a_k[mma_k_idx], mma_b_kj[mma_k_idx][mma_j_idx]);
                }
            }
        }
    }

    // CTA cooperates to fill the output matrix block of size (SMEM_I, SMEM_J) starting at (cta_i_offset, cta_j_offset).
    // Requires smem-allocated double buffer.
    __device__ void cta_compute_block(uint32_t cta_i_offset, uint32_t cta_j_offset, SmemLayout& smem)
    {
        assert(cta_i_offset % SMEM_I == 0);
        assert(cta_j_offset % SMEM_J == 0);
        assert(size_k % SMEM_K == 0);

        WarpAccum accum{};
        const uint32_t k_blk_dim = size_k / SMEM_K;

        for (uint32_t cta_k_idx = 0; cta_k_idx <= k_blk_dim; ++cta_k_idx) {
            if (cta_k_idx < k_blk_dim) {
                cta_async_load_block(smem.buffers[cta_k_idx % 2u], cta_i_offset, cta_j_offset, cta_k_idx * SMEM_K);
            }
            if (cta_k_idx > 0) {
                // Accumulate smem buffer filled in previous iteration. NB <= in for loop is critical for this!
                warp_accum_block(accum, smem.buffers[(cta_k_idx % 2u) ^ 1u]);
            }
            async_memcpy_waitall();
            __syncthreads();
        }

        const uint32_t warp_i_offset = get_warp_i_idx() * WARP_I + cta_i_offset;
        const uint32_t warp_j_offset = get_warp_j_idx() * WARP_J + cta_j_offset;
        for (uint32_t mma_i_idx = 0; mma_i_idx < WARP_I / MMA_I; ++mma_i_idx) {
            for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
                const auto i = warp_i_offset + MMA_I * mma_i_idx;
                const auto j = warp_j_offset + MMA_J * mma_j_idx;
                store_d(c + i * size_j + j, accum.ij[mma_i_idx][mma_j_idx], {size_j, 1});
            }
        }
    }

    __device__ void kernel_main()
    {
        assert(gridDim.x == grid_size());
        assert(blockDim.x == cta_size());

        const uint32_t cta_rows = size_i / SMEM_I;
        const uint32_t cta_cols = size_j / SMEM_J;
        const uint32_t cta_col_remainder = cta_cols % CTA_MODULUS;
        const uint32_t superblock_count = cta_cols / CTA_MODULUS;
        const uint32_t superblock_cta_count = cta_rows * CTA_MODULUS;

        for (uint32_t task_index = blockIdx.x; task_index < i_cta() * j_cta(); task_index += grid_size()) {
            uint32_t cta_i_idx, cta_j_idx;

            const uint32_t superblock_idx = task_index / superblock_cta_count;
            const uint32_t cta_index_in_superblock = task_index % superblock_cta_count;

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
            cta_compute_block(cta_i_idx * SMEM_I, cta_j_idx * SMEM_J, reinterpret_cast<SmemLayout&>(smem[0]));
        }
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

template <TileConfig tile_config>
void TiledMultiplier<tile_config>::launch(cudaStream_t stream)
{
    using Multiplier = std::remove_reference_t<decltype(*this)>;
    const uint32_t grid = grid_size();
    const uint32_t block = cta_size();
    const uint32_t smem = sizeof(SmemLayout);
    cudaFuncSetAttribute(tiled_multiplier_kernel<Multiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tiled_multiplier_kernel<Multiplier> <<<grid, block, smem, stream>>>(*this);
}

}  // end namespace

constexpr TileConfig tile_config
{
    192, 128, 16,
    48, 64,
    1,
};

void matmul_sm80(GPU_Tensors t, cudaStream_t stream) {
    const uint32_t size_i = t.M;
    const uint32_t size_j = t.N;
    const uint32_t size_k = t.K;

    assert(matmul_sm80_supports(t));
    TiledMultiplier<tile_config> multiplier{size_i, size_j, size_k, t.a, t.b, t.c};
    multiplier.launch(stream);
}

bool matmul_sm80_supports(GPU_Tensors t)
{
    return !t.a_col_major && !t.b_col_major && !t.c_col_major && t.M % tile_config.smem_i == 0 && t.N % tile_config.smem_j == 0 && t.K % tile_config.smem_k == 0;
};
