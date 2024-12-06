#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda.h>

#define DEVICE_INLINE __device__ __forceinline__

#define DEVICE_ASSERT(x) assert(x)
// #define DEVICE_ASSERT(x)

namespace {

using mbarrier_t = long long;

DEVICE_INLINE uint32_t smem_ptr_u32(const void* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size())
tiled_multiplier_kernel(uint32_t size_m, uint32_t size_n, uint32_t size_k,
                        const float* a, const float* bT, float* c,
                        __grid_constant__ const CUtensorMap tensorMap_a,
                        __grid_constant__ const CUtensorMap tensorMap_bT,
                        __grid_constant__ const CUtensorMap tensorMap_c)
{
    Multiplier multiplier{size_m, size_n, size_k, a, bT, c, &tensorMap_a, &tensorMap_bT, &tensorMap_c};
    multiplier.kernel_main();
}

template <uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K, uint32_t K_MAX_TILES, uint32_t CTA_MODULUS>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;

    // TODO remove
    float const* a;
    float const* bT;
    float* c;

    const CUtensorMap* tensorMap_a;
    const CUtensorMap* tensorMap_bT;  // Transposed; column major
    const CUtensorMap* tensorMap_c;

    static constexpr uint32_t WG_M = 64;
    static constexpr uint32_t WG_N = 64;
    static constexpr uint32_t WG_K = 8;
    static constexpr uint32_t RING_BUFFER_SIZE = 3;
    static constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

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
        mbarrier_t tile_fill_mbar[RING_BUFFER_SIZE];
        mbarrier_t tile_read_mbar[RING_BUFFER_SIZE];
    };

    __host__ DEVICE_INLINE static constexpr uint32_t smem_size()
    {
        return sizeof(Shared);
    }

    __host__ DEVICE_INLINE static constexpr uint32_t consumer_wg_count()
    {
        // If output matrix is cut into (WG_M, WG_N) blocks, one warpgroup handles one matrix block.
        static_assert(SMEM_M % WG_M == 0);
        static_assert(SMEM_N % WG_N == 0);
        return (SMEM_M / WG_M) * (SMEM_N / WG_N);
    }
    __host__ DEVICE_INLINE static constexpr uint32_t cta_size()
    {
        // 1 extra warpgroup for memory.
        return (1 + consumer_wg_count()) * 128;
    }

    // If output matrix is cut into (SMEM_M, SMEM_N) blocks, one CTA k-group handles one matrix block.
    // One CTA k-group cooperates to fill the block using split-k strategy (reduce into output memory),
    // except if k_cta is 0.
    static __host__ DEVICE_INLINE uint32_t m_cta(uint32_t size_m)
    {
        DEVICE_ASSERT(size_m % SMEM_M == 0);
        return size_m / SMEM_M;
    }

    static __host__ DEVICE_INLINE uint32_t n_cta(uint32_t size_n)
    {
        DEVICE_ASSERT(size_n % SMEM_N == 0);
        return size_n / SMEM_N;
    }

    static __host__ DEVICE_INLINE uint32_t k_cta(uint32_t size_k)  // size of CTA k-group
    {
        constexpr uint32_t k_divisor = SMEM_K * K_MAX_TILES;
        return (size_k + k_divisor - 1) / k_divisor;
    }

    static __host__ DEVICE_INLINE bool using_cp_reduce(uint32_t size_k)
    {
#ifndef __CUDA_ARCH__
        assert(k_cta(size_k) != 0);
#endif
        return k_cta(size_k) != 1;
    }

    static DEVICE_INLINE void mbar_wait(mbarrier_t& mbar, uint32_t parity)
    {
        asm volatile(
                "{.reg.pred P1; BEFORE_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni WAIT_DONE; bra.uni BEFORE_WAIT; WAIT_DONE: }"
        :
        : "r"(smem_ptr_u32(&mbar)), "r"(parity));
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
    DEVICE_INLINE void wg_accum_tile(WG_Accum& d, const Buffers& buffers, uint32_t wg_m_offset, uint32_t wg_n_offset,
                                     uint32_t wg_k_offset, bool zero_output) const
    {
        const uint32_t lane = threadIdx.x % 128u;
        for (uint32_t local_k = 0; local_k < WG_K; ++local_k) {
            for (uint32_t r = 0; r < d.regcount; ++r) {
                const uint32_t r_in_wg = r + d.regcount * lane;
                const uint32_t local_m = r_in_wg / WG_N;
                const uint32_t local_n = r_in_wg % WG_N;
                const uint32_t outer_m = wg_m_offset + local_m;
                const uint32_t outer_n = wg_n_offset + local_n;
                const uint32_t outer_k = wg_k_offset + local_k;
                const float a_val = buffers.a_tile[outer_m * SMEM_K + outer_k];
                const float b_val = buffers.bT_tile[outer_n * SMEM_K + outer_k];
                if (zero_output && local_k == 0) {
                    d.regs[r] = a_val * b_val;
                }
                else {
                    d.regs[r] = fma(a_val, b_val, d.regs[r]);
                }
            }
        }
    }

    // Warpgroup-convergent code
    // Write the (WG_M, WG_N) tile to shared.c_tile, at offset (wg_m_offset, wg_n_offset).
    DEVICE_INLINE void wg_accum_to_shared(Shared& shared, const WG_Accum& accum,
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
    // Fill shared memory B^T tile with SMEM_N×SMEM_K block starting at (cta_n_offset, cta_k_offset)
    DEVICE_INLINE void warp_async_load_block(Buffers& buffers, mbarrier_t& mbar,
                                             uint32_t cta_m_offset, uint32_t cta_n_offset, uint32_t cta_k_offset) const
    {
        const uint32_t lane = threadIdx.x % 32u;
        if (lane == 0) {  // XXX lane == 0 is "wrong"
            constexpr uint32_t expect_count = (SMEM_M + SMEM_N) * SMEM_K * sizeof(float);
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(smem_ptr_u32(&mbar)), "n"(expect_count));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(smem_ptr_u32(&buffers.a_tile)),
                  "l"(tensorMap_a),
                  "r"(smem_ptr_u32(&mbar)),
                  "r"(cta_k_offset), "r"(cta_m_offset)
                : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4}], [%2];"
                :
                : "r"(smem_ptr_u32(&buffers.bT_tile)),
                  "l"(tensorMap_bT),
                  "r"(smem_ptr_u32(&mbar)),
                  "r"(cta_k_offset), "r"(cta_n_offset)
                : "memory");
        }
    }

    // Static assignment of warpgroups within CTA to per-warpgroup output tiles (WG_M, WG_N) within
    // per-CTA output tile (SMEM_M, SMEM_N), plus one extra warpgroup for memory transactions.
    DEVICE_INLINE bool is_memory_wg() const
    {
        return (threadIdx.x / 128u) == consumer_wg_count();
    }

    DEVICE_INLINE uint32_t get_wg_m_idx() const
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        DEVICE_ASSERT(wg_index < consumer_wg_count());
        return wg_index / (SMEM_N / WG_N);
    }

    DEVICE_INLINE uint32_t get_wg_n_idx() const
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        return wg_index % (SMEM_N / WG_N);
    }

    DEVICE_INLINE void cta_first_time_init(Shared& shared) const
    {
        for (uint32_t i = threadIdx.x; i < RING_BUFFER_SIZE; i += blockDim.x) {
            constexpr unsigned consumer_thread_count = consumer_wg_count() * 128u;
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_ptr_u32(&shared.tile_fill_mbar[i])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(smem_ptr_u32(&shared.tile_read_mbar[i])), "n"(consumer_thread_count));
            asm volatile("fence.proxy.async;");
        }
    }

    // CTA cooperates to fill or add to the output matrix block of size (SMEM_M, SMEM_N) starting at
    // (cta_m_offset, cta_n_offset); we process up to K_MAX_TILES input blocks on the K dimension starting
    // at cta_k_offset.
    // Requires smem-allocated ring buffer.
    DEVICE_INLINE void cta_compute_block(uint32_t cta_m_offset, uint32_t cta_n_offset, uint32_t cta_k_offset,
                                         Shared& shared) const
    {
        DEVICE_ASSERT(cta_m_offset % SMEM_M == 0);
        DEVICE_ASSERT(cta_n_offset % SMEM_N == 0);
        DEVICE_ASSERT(cta_k_offset % (SMEM_K * K_MAX_TILES) == 0);
        DEVICE_ASSERT(size_k % SMEM_K == 0);

        const uint32_t k_iters = min((size_k - cta_k_offset) / SMEM_K, K_MAX_TILES);

        WG_Accum accum;
        bool first_time = true;

        for (uint32_t cta_k_blk_counter = 0; cta_k_blk_counter < k_iters; ++cta_k_blk_counter, cta_k_offset += SMEM_K) {
            const uint32_t ring_idx = cta_k_blk_counter % RING_BUFFER_SIZE;
            const uint32_t ring_usage_parity = (cta_k_blk_counter / RING_BUFFER_SIZE) % 2u;

            if (is_memory_wg()) {
                if (ring_idx != cta_k_blk_counter) {
                    mbar_wait(shared.tile_read_mbar[ring_idx], !ring_usage_parity);
                }
                if (threadIdx.x % 128u < 32u) {
                    warp_async_load_block(shared.buffers[ring_idx], shared.tile_fill_mbar[ring_idx],
                                          cta_m_offset, cta_n_offset, cta_k_offset);
                }
            }

            if (!is_memory_wg()) {
                mbar_wait(shared.tile_fill_mbar[ring_idx], ring_usage_parity);
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
                for (uint32_t wg_k_idx = 0; wg_k_idx < SMEM_K / WG_K; ++wg_k_idx) {
                    const uint32_t wg_k_offset = wg_k_idx * WG_K;
                    wg_accum_tile(accum, shared.buffers[ring_idx], wg_m_offset, wg_n_offset, wg_k_offset, first_time);
                    first_time = false;
                }
                asm volatile(
                    "mbarrier.arrive.shared::cta.b64 _, [%0];"
                    :
                    : "r"(smem_ptr_u32(&shared.tile_read_mbar[ring_idx])));
            }
        }

        if (!is_memory_wg()) {
            const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
            const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
            wg_accum_to_shared(shared, accum, wg_m_offset, wg_n_offset);
        }

        __syncthreads();  // TODO

        if (is_memory_wg()) {
            if (threadIdx.x % 128u == 0u) {
                asm volatile("fence.proxy.async;");
                if (using_cp_reduce(size_k)) {
                    asm volatile(
                    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset), "r"(cta_m_offset),
                      "r"(smem_ptr_u32(&shared.c_tile)));
                }
                else {
                    asm volatile(
                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset), "r"(cta_m_offset),
                      "r"(smem_ptr_u32(&shared.c_tile)));
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    DEVICE_INLINE void kernel_main()
    {
        DEVICE_ASSERT(blockDim.x == cta_size());

        const uint32_t cta_rows = m_cta(size_m);
        const uint32_t cta_cols = n_cta(size_n);
        const uint32_t num_output_block_matrix = cta_rows * cta_cols;
        const uint32_t cta_k_group = k_cta(size_k);
        const uint32_t cta_col_remainder = cta_cols % CTA_MODULUS;
        const uint32_t superblock_count = cta_cols / CTA_MODULUS;
        const uint32_t superblock_cta_count = cta_rows * CTA_MODULUS;
        const uint32_t superblock_idx = (blockIdx.x % num_output_block_matrix) / superblock_cta_count;
        const uint32_t cta_index_in_superblock = (blockIdx.x % num_output_block_matrix) % superblock_cta_count;
        const uint32_t cta_k_idx = blockIdx.x / num_output_block_matrix;
        const uint32_t cta_k_offset = cta_k_idx * (SMEM_K * K_MAX_TILES);

        DEVICE_ASSERT(cta_k_group * num_output_block_matrix == gridDim.x);

        uint32_t cta_m_idx, cta_n_idx;

        if (superblock_idx < superblock_count) {
            cta_m_idx = cta_index_in_superblock / CTA_MODULUS;
            cta_n_idx = cta_index_in_superblock % CTA_MODULUS + CTA_MODULUS * superblock_idx;
        }
        else {
            DEVICE_ASSERT(superblock_idx == superblock_count);
            cta_m_idx = cta_index_in_superblock / cta_col_remainder;
            cta_n_idx = cta_index_in_superblock % cta_col_remainder + CTA_MODULUS * superblock_idx;
        }
        DEVICE_ASSERT(cta_m_idx < cta_rows);
        DEVICE_ASSERT(cta_n_idx < cta_cols);

        extern __shared__ char smem[];
        cta_first_time_init(reinterpret_cast<Shared&>(*smem));
        __syncthreads();
        cta_compute_block(cta_m_idx * SMEM_M, cta_n_idx * SMEM_N, cta_k_offset, reinterpret_cast<Shared&>(*smem));
    }

    static void init_tensorMap(CUtensorMap* tensorMap, const float* globalAddress, uint32_t rows, uint32_t cols,
                               uint32_t smem_rows, uint32_t smem_cols)
    {
        const CUtensorMapDataType tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        const uint32_t tensorRank = 2;
        const cuuint64_t globalDim[2] = {cols, rows};
        const cuuint64_t globalStrides[1] = {4*cols};
        const cuuint32_t boxDim[2] = {smem_cols, smem_rows};
        const cuuint32_t elementStrides[2] = {1, 1};
        const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
        const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
        const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;

        const CUresult result = cuTensorMapEncodeTiled(
                tensorMap,
                tensorDataType,
                tensorRank,
                const_cast<float*>(globalAddress),
                globalDim,
                globalStrides,
                boxDim,
                elementStrides,
                interleave,
                swizzle,
                l2Promotion,
                oobFill);
        if (result != 0) {
            fprintf(stderr, "cuTensorMapEncodeTiled: %i\n", (int)result);
            assert(0);
        }
    }

    static void launch(cudaStream_t stream, uint32_t size_m, uint32_t size_n, uint32_t size_k,
                       const float* a, const float* bT, float* c)
    {
        CUtensorMap tensorMap_a, tensorMap_bT, tensorMap_c;
        init_tensorMap(&tensorMap_a, a, size_m, size_k, SMEM_M, SMEM_K);
        init_tensorMap(&tensorMap_bT, bT, size_n, size_k, SMEM_N, SMEM_K);
        init_tensorMap(&tensorMap_c, c, size_m, size_n, SMEM_M, SMEM_N);

        if (using_cp_reduce(size_k)) {
            cudaMemsetAsync(c, 0, size_m * size_n * sizeof(*c), stream);
        }

        const uint32_t grid = m_cta(size_m) * n_cta(size_n) * k_cta(size_k);
        const uint32_t block = cta_size();
        const uint32_t smem = smem_size();
        cudaFuncSetAttribute(tiled_multiplier_kernel<TiledMultiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        tiled_multiplier_kernel<TiledMultiplier> <<<grid, block, smem, stream>>>(
                size_m, size_n, size_k, a, bT, c, tensorMap_a, tensorMap_bT, tensorMap_c);
    }
};

}  // end namespace

void matmul_sm90(GPU_Tensors t, cudaStream_t stream)
{
    constexpr uint32_t smem_m = 64;
    constexpr uint32_t smem_n = 256;
    constexpr uint32_t smem_k = 16;
    constexpr uint32_t cta_k_max_tiles = 12;
    constexpr uint32_t cta_modulus = 4;

    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    if (size_m % smem_m == 0 && size_n % smem_n == 0 && size_k % smem_k == 0) {
        TiledMultiplier<smem_m, smem_n, smem_k, cta_k_max_tiles, cta_modulus>::launch(
                stream, size_m, size_n, size_k, t.a, t.b, t.c);
    }
    else {
        assert(0);
    }
}
