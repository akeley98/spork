#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace {

template <uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K, uint32_t CTA_MODULUS>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;
    float const* a;
    float const* b;
    float* c;

    static constexpr uint32_t WG_M = 64;
    static constexpr uint32_t WG_N = 64;
    static constexpr uint32_t WG_K = 8;
    static constexpr uint32_t RING_BUFFER_SIZE = 1;

    // One buffer of ring buffer.
    struct Buffers
    {
        float a_tile[SMEM_M * SMEM_K];
        float bT_tile[SMEM_N * SMEM_K];  // Transposed
    };

    struct Shared
    {
        Buffers buffers[RING_BUFFER_SIZE];
        float c_tile[SMEM_M * SMEM_N];
    };

    __host__ __device__ static constexpr uint32_t smem_size()
    {
        return sizeof(Shared);
    }

    __host__ __device__ static constexpr uint32_t consumer_wg_count()
    {
        // If output matrix is cut into (WG_M, WG_N) blocks, one warpgroup handles one matrix block.
        static_assert(SMEM_M % WG_M == 0);
        static_assert(SMEM_N % WG_N == 0);
        return (SMEM_M / WG_M) * (SMEM_N / WG_N);
    }
    __host__ __device__ static constexpr uint32_t cta_size()
    {
        // 1 extra warpgroup for memory.
        return (1 + consumer_wg_count()) * 128;
    }

    // If output matrix is cut into (SMEM_M, SMEM_N) blocks, one CTA handles one matrix block.
    __host__ __device__ uint32_t m_cta() const
    {
        assert(size_m % SMEM_M == 0);
        return size_m / SMEM_M;
    }

    __host__ __device__ uint32_t n_cta() const
    {
        assert(size_n % SMEM_N == 0);
        return size_n / SMEM_N;
    }

    // Per-warpgroup accumulator, holding one (WG_M, WG_N) tile.
    struct WG_Accum
    {
        // TODO use wgmma
        static constexpr uint32_t regcount = WG_M * WG_N / 128u;
        float regs[regcount];
    };

    // Warpgroup-convergent code.
    // Accumulate data from shared memory. Multiply the block matrices
    //   a_tile[wg_m_offset : wg_m_offset + WG_M, wg_k_offset : wg_k_offset + WG_K]
    //   bT_tile[wg_n_offset : wg_n_offset + WG_N, wg_k_offset : wg_k_offset + WG_K]
    // and add to the (WG_M, WG_N) tile held in WG_Accum.
    __device__ void wg_accum_tile(WG_Accum& accum, const Buffers& buffers, uint32_t wg_m_offset, uint32_t wg_n_offset,
                                  uint32_t wg_k_offset, bool zero_output) const
    {
        const uint32_t lane = threadIdx.x % 128u;

        for (uint32_t r = 0; r < accum.regcount; ++r) {
            const uint32_t r_in_wg = r + accum.regcount * lane;
            const uint32_t local_m = r_in_wg / WG_N;
            const uint32_t local_n = r_in_wg % WG_N;

            for (uint32_t local_k = 0; local_k < WG_K; ++local_k) {
                const uint32_t outer_m = wg_m_offset + local_m;
                const uint32_t outer_n = wg_n_offset + local_n;
                const uint32_t outer_k = wg_k_offset + local_k;
                const float a_val = buffers.a_tile[outer_m * SMEM_K + outer_k];
                const float b_val = buffers.bT_tile[outer_n * SMEM_K + outer_k];
                if (zero_output && local_k == 0) {
                    accum.regs[r] = a_val * b_val;
                }
                else {
                    accum.regs[r] = fma(a_val, b_val, accum.regs[r]);
                }
            }
        }
    }

    // Warpgroup-convergent code
    // Write the (WG_M, WG_N) tile to shared.c_tile, at offset (wg_m_offset, wg_n_offset).
    __device__ void wg_accum_to_shared(Shared& shared, const WG_Accum& accum,
                                       uint32_t wg_m_offset, uint32_t wg_n_offset) const
    {
        const uint32_t lane = threadIdx.x % 128u;

        for (uint32_t r = 0; r < accum.regcount; ++r) {
            const uint32_t r_in_wg = r + accum.regcount * lane;
            const uint32_t local_m = r_in_wg / WG_N;
            const uint32_t local_n = r_in_wg % WG_N;
            const uint32_t outer_m = wg_m_offset + local_m;
            const uint32_t outer_n = wg_n_offset + local_n;
            shared.c_tile[outer_m * SMEM_N + outer_n] = accum.regs[r];
        }
    }

    // Fill shared memory A tile with SMEM_M×SMEM_K block starting at (cta_m_offset, cta_k_offset)
    // Fill shared memory B tile with SMEM_K×SMEM_N block starting at (cta_k_offset, cta_n_offset)
    __device__ void warp_async_load_block(Buffers& buffers, uint32_t cta_m_offset,
                                          uint32_t cta_n_offset, uint32_t cta_k_offset) const
    {
        // TODO TMA so it's really async.
        const uint32_t lane = threadIdx.x % 32u;
        for (uint32_t local_m = 0; local_m < SMEM_M; local_m++) {
            for (uint32_t local_k = lane; local_k < SMEM_K; local_k += 32) {
                const uint32_t global_m = local_m + cta_m_offset;
                const uint32_t global_k = local_k + cta_k_offset;
                buffers.a_tile[local_m * SMEM_K + local_k] = a[global_m * size_k + global_k];
            }
        }
        for (uint32_t local_n = 0; local_n < SMEM_N; local_n++) {
            for (uint32_t local_k = lane; local_k < SMEM_K; local_k += 32) {
                const uint32_t global_n = local_n + cta_n_offset;
                const uint32_t global_k = local_k + cta_k_offset;
                const float b_val = b[global_k * size_n + global_n];

                buffers.bT_tile[local_n * SMEM_K + local_k] = b_val;
            }
        }
    }

    // Static assignment of warpgroups within CTA to per-warpgroup output tiles (WG_M, WG_N) within
    // per-CTA output tile (SMEM_M, SMEM_N), plus one extra warpgroup for memory transactions.
    __device__ bool is_memory_wg() const
    {
        return (threadIdx.x / 128u) == consumer_wg_count();
    }

    __device__ uint32_t get_wg_m_idx() const
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        assert(wg_index < consumer_wg_count());
        return wg_index / (SMEM_N / WG_N);
    }

    __device__ uint32_t get_wg_n_idx() const
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        return wg_index % (SMEM_N / WG_N);
    }

    // CTA cooperates to fill the output matrix block of size (SMEM_M, SMEM_N) starting at (cta_m_offset, cta_n_offset).
    // Requires smem-allocated ring buffer.
    __device__ void cta_compute_block(uint32_t cta_m_offset, uint32_t cta_n_offset, Shared& shared) const
    {
        assert(cta_m_offset % SMEM_M == 0);
        assert(cta_n_offset % SMEM_N == 0);
        assert(size_k % SMEM_K == 0);
        const uint32_t k_blk_dim = size_k / SMEM_K;

        WG_Accum accum;

        for (uint32_t cta_k_idx = 0; cta_k_idx < k_blk_dim; ++cta_k_idx) {
            if (is_memory_wg()) {
                if (threadIdx.x % 128u < 32u) {
                    const auto cta_k_offset = cta_k_idx * SMEM_K;
                    warp_async_load_block(shared.buffers[0], cta_m_offset, cta_n_offset, cta_k_offset);
                }
            }
            __syncthreads();  // TODO
            if (!is_memory_wg()) {
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
                for (uint32_t wg_k_idx = 0; wg_k_idx < SMEM_K / WG_K; ++wg_k_idx) {
                    const uint32_t wg_k_offset = wg_k_idx * WG_K;
                    const bool zero_accum = cta_k_idx == 0 && wg_k_offset == 0;
                    wg_accum_tile(accum, shared.buffers[0], wg_m_offset, wg_n_offset, wg_k_offset, zero_accum);
                }
            }
            __syncthreads();  // TODO
        }

        if (!is_memory_wg()) {
            const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
            const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
            wg_accum_to_shared(shared, accum, wg_m_offset, wg_n_offset);
        }

        __syncthreads();  // TODO

        for (uint32_t m = 0; m < SMEM_M; m++) {
            for (uint32_t n = threadIdx.x; n < SMEM_N; n += blockDim.x) {
                const uint32_t global_m = m + cta_m_offset;
                const uint32_t global_n = n + cta_n_offset;
                c[global_m * size_n + global_n] = shared.c_tile[m * SMEM_N + n];
            }
        }
    }

    __device__ void kernel_main()
    {
        assert(gridDim.x == m_cta() * n_cta());
        assert(blockDim.x == cta_size());

        const uint32_t cta_rows = size_m / SMEM_M;
        const uint32_t cta_cols = size_n / SMEM_N;
        const uint32_t cta_col_remainder = cta_cols % CTA_MODULUS;
        const uint32_t superblock_count = cta_cols / CTA_MODULUS;
        const uint32_t superblock_cta_count = cta_rows * CTA_MODULUS;
        const uint32_t superblock_idx = blockIdx.x / superblock_cta_count;
        const uint32_t cta_index_mn_superblock = blockIdx.x % superblock_cta_count;

        uint32_t cta_m_idx, cta_n_idx;

        if (superblock_idx < superblock_count) {
            cta_m_idx = cta_index_mn_superblock / CTA_MODULUS;
            cta_n_idx = cta_index_mn_superblock % CTA_MODULUS + CTA_MODULUS * superblock_idx;
        }
        else {
            assert(superblock_idx == superblock_count);
            cta_m_idx = cta_index_mn_superblock / cta_col_remainder;
            cta_n_idx = cta_index_mn_superblock % cta_col_remainder + CTA_MODULUS * superblock_idx;
        }
        assert(cta_m_idx < cta_rows);
        assert(cta_n_idx < cta_cols);

        extern __shared__ char smem[];
        cta_compute_block(cta_m_idx * SMEM_M, cta_n_idx * SMEM_N, reinterpret_cast<Shared&>(*smem));
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

template <uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K, uint32_t CTA_MODULUS>
void TiledMultiplier<SMEM_M, SMEM_N, SMEM_K, CTA_MODULUS>::launch(cudaStream_t stream)
{
    using Multiplier = std::remove_reference_t<decltype(*this)>;
    const dim3 grid{m_cta() * n_cta(), 1, 1};
    const uint32_t block = cta_size();
    const uint32_t smem = smem_size();
    cudaFuncSetAttribute(tiled_multiplier_kernel<Multiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tiled_multiplier_kernel<Multiplier> <<<grid, block, smem, stream>>>(*this);
}

}  // end namespace

void matmul_sm90(GPU_Tensors t, cudaStream_t stream)
{
    constexpr uint32_t smem_m = 128;
    constexpr uint32_t smem_n = 128;
    constexpr uint32_t smem_k = 8;
    constexpr uint32_t cta_modulus = 4;

    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    if (size_m % smem_m == 0 && size_n % smem_n == 0 && size_k % smem_k == 0) {
        TiledMultiplier<smem_m, smem_n, smem_k, cta_modulus> multiplier{size_m, size_n, size_k, t.a, t.b, t.c};
        multiplier.launch(stream);
    }
    else {
        assert(0);
    }
}
