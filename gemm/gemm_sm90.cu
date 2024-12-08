#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <cuda.h>

#define DEVICE_INLINE __device__ __forceinline__

// #define DEVICE_ASSERT(x) assert(x)
#define DEVICE_ASSERT(x)

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
                        __grid_constant__ const CUtensorMap tensorMap_a,
                        __grid_constant__ const CUtensorMap tensorMap_bT,
                        __grid_constant__ const CUtensorMap tensorMap_c)
{
    Multiplier multiplier{size_m, size_n, size_k, &tensorMap_a, &tensorMap_bT, &tensorMap_c};
    multiplier.kernel_main();
}

template <uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K, uint32_t K_MAX_TILES, uint32_t CTA_MODULUS, uint32_t RING_BUFFER_SIZE>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;

    const CUtensorMap* tensorMap_a;
    const CUtensorMap* tensorMap_bT;  // Transposed; column major
    const CUtensorMap* tensorMap_c;

    static constexpr uint32_t WG_M = 64;
    static constexpr uint32_t WG_N = 128;
    static constexpr uint32_t WG_K = 8;
    static constexpr CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

    // wgmma core matrix dimensions.
    static constexpr uint32_t SMEM_MN_INNER = 8u;
    static constexpr uint32_t SMEM_K_INNER = 16u / sizeof(float);

    static constexpr uint32_t SMEM_M_OUTER = SMEM_M / SMEM_MN_INNER;
    static_assert(SMEM_MN_INNER * SMEM_M_OUTER == SMEM_M);
    static constexpr uint32_t SMEM_N_OUTER = SMEM_N / SMEM_MN_INNER;
    static_assert(SMEM_MN_INNER * SMEM_N_OUTER == SMEM_N);
    static constexpr uint32_t SMEM_K_OUTER = SMEM_K / SMEM_K_INNER;
    static_assert(SMEM_K_INNER * SMEM_K_OUTER == SMEM_K);

    // One buffer of ring buffer.
    struct Buffers
    {
        // Inner (rightmost) 2 dimensions correspond to a wgmma core matrix.
        // Note outer 2 dimensions have MN/K swapped compared to inner 2 dimensions.
        // This is to make TMA (cp.async.bulk.tensor) more efficient.
        // We can treat the right 3 dimensions as a C-order 2D array, which is what 2D TMA supports.
        float a_tile[SMEM_K_OUTER][SMEM_M_OUTER][SMEM_MN_INNER][SMEM_K_INNER];
        float bT_tile[SMEM_K_OUTER][SMEM_N_OUTER][SMEM_MN_INNER][SMEM_K_INNER];

        DEVICE_INLINE const float* a_tile_mk_offset(uint32_t m_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(m_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &a_tile[k_offset / SMEM_K_INNER][m_offset / SMEM_MN_INNER][0][0];
        }

        DEVICE_INLINE const float* bT_tile_nk_offset(uint32_t n_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(n_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &bT_tile[k_offset / SMEM_K_INNER][n_offset / SMEM_MN_INNER][0][0];
        }
    };

    // Strides between core matrices, and configuration for TMA tensormap.
    static constexpr uint32_t core_matrix_mn_stride = sizeof(float) * SMEM_MN_INNER * SMEM_K_INNER;
    static constexpr uint32_t core_matrix_a_k_stride = core_matrix_mn_stride * SMEM_M_OUTER;
    static constexpr uint32_t core_matrix_bT_k_stride = core_matrix_mn_stride * SMEM_N_OUTER;
    static constexpr uint32_t tensorMap_box_m = SMEM_M;
    static constexpr uint32_t tensorMap_box_n = SMEM_N;
    static constexpr uint32_t tensorMap_box_k = SMEM_K_INNER;

    struct Shared
    {
        union {
            Buffers aliased_input_ring_buffer[RING_BUFFER_SIZE];
            float aliased_output_c_tile[SMEM_M * SMEM_N];
        };
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
        static constexpr uint32_t regcount = WG_M * WG_N / 128u;

        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
        float d32, d33, d34, d35, d36, d37, d38, d39;
        float d40, d41, d42, d43, d44, d45, d46, d47;
        float d48, d49, d50, d51, d52, d53, d54, d55;
        float d56, d57, d58, d59, d60, d61, d62, d63;
        static_assert(regcount == 64);
    };

    static DEVICE_INLINE uint64_t matrix_descriptor_encode(uint32_t val)
    {
        uint64_t enc = (val & 0x3FFFF) >> 4;
        DEVICE_ASSERT(val == enc << 4);
        return enc;
    }

    static DEVICE_INLINE uint64_t matrix_descriptor_mn_k_stride(const float* smem_ptr,
                                                                uint32_t mn_stride, uint32_t k_stride)
    {
        // Swizzling not supported.
        static_assert(swizzle == CU_TENSOR_MAP_SWIZZLE_NONE);
        return matrix_descriptor_encode(smem_ptr_u32(smem_ptr))
               | matrix_descriptor_encode(k_stride) << 16u
               | matrix_descriptor_encode(mn_stride) << 32u;
    }

    // Warpgroup-convergent code.
    // Accumulate data from shared memory. Multiply the block matrices
    //   a_tile[wg_m_offset : wg_m_offset + WG_M, wg_k_offset : wg_k_offset + WG_K]
    //   bT_tile[wg_n_offset : wg_n_offset + WG_N, wg_k_offset : wg_k_offset + WG_K]
    // and add to the (WG_M, WG_N) tile held in WG_Accum.
    DEVICE_INLINE void wg_accum_tile(WG_Accum& d, const Buffers& buffers, uint32_t wg_m_offset, uint32_t wg_n_offset,
                                     uint32_t wg_k_offset, bool zero_output) const
    {
        [[maybe_unused]] const uint32_t lane = threadIdx.x % 128u;
        static_assert(WG_K % SMEM_K_INNER == 0);
        auto desc_a = matrix_descriptor_mn_k_stride(buffers.a_tile_mk_offset(wg_m_offset, wg_k_offset),
                                                    core_matrix_mn_stride, core_matrix_a_k_stride);
        auto desc_b = matrix_descriptor_mn_k_stride(buffers.bT_tile_nk_offset(wg_n_offset, wg_k_offset),
                                                    core_matrix_mn_stride, core_matrix_bT_k_stride);
        static_assert(WG_M == 64);
        static_assert(WG_N == 128);
        static_assert(WG_K == 8);
        asm volatile(
        "{\n"
          ".reg .pred p;\n"
          "setp.ne.b32 p, %66, 0;\n"
          "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
          "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
          " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
          " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
          " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
          " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
          " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
          " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
          " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
          " %64,"
          " %65,"
          " p,    1,  1;\n"
        "}\n"
          : "+f"(d.d0), "+f"(d.d1), "+f"(d.d2), "+f"(d.d3),
            "+f"(d.d4), "+f"(d.d5), "+f"(d.d6), "+f"(d.d7),
            "+f"(d.d8), "+f"(d.d9), "+f"(d.d10), "+f"(d.d11),
            "+f"(d.d12), "+f"(d.d13), "+f"(d.d14), "+f"(d.d15),
            "+f"(d.d16), "+f"(d.d17), "+f"(d.d18), "+f"(d.d19),
            "+f"(d.d20), "+f"(d.d21), "+f"(d.d22), "+f"(d.d23),
            "+f"(d.d24), "+f"(d.d25), "+f"(d.d26), "+f"(d.d27),
            "+f"(d.d28), "+f"(d.d29), "+f"(d.d30), "+f"(d.d31),
            "+f"(d.d32), "+f"(d.d33), "+f"(d.d34), "+f"(d.d35),
            "+f"(d.d36), "+f"(d.d37), "+f"(d.d38), "+f"(d.d39),
            "+f"(d.d40), "+f"(d.d41), "+f"(d.d42), "+f"(d.d43),
            "+f"(d.d44), "+f"(d.d45), "+f"(d.d46), "+f"(d.d47),
            "+f"(d.d48), "+f"(d.d49), "+f"(d.d50), "+f"(d.d51),
            "+f"(d.d52), "+f"(d.d53), "+f"(d.d54), "+f"(d.d55),
            "+f"(d.d56), "+f"(d.d57), "+f"(d.d58), "+f"(d.d59),
            "+f"(d.d60), "+f"(d.d61), "+f"(d.d62), "+f"(d.d63)
          :  "l"(desc_a),
             "l"(desc_b),
             "r"(int32_t(!zero_output)));
    }

    // Warpgroup-convergent code
    // Write the (WG_M, WG_N) tile to shared.c_tile, at offset (wg_m_offset, wg_n_offset).
    DEVICE_INLINE void wg_accum_to_shared(Shared& shared, const WG_Accum& d,
                                          uint32_t wg_m_offset, uint32_t wg_n_offset) const
    {
        const uint32_t tid = threadIdx.x % 128u;

        static_assert(WG_Accum::regcount == 64);
        #define X(REG_INDEX) { \
            const uint32_t r = (tid / 32u) * 16u + (tid % 32u) / 4u + ((REG_INDEX % 4u) / 2u) * 8u; \
            const uint32_t c = (tid % 4u) * 2u + (REG_INDEX / 4u) * 8 + (REG_INDEX % 2u); \
            const uint32_t smem_m_offset = wg_m_offset + r; \
            const uint32_t smem_n_offset = wg_n_offset + c; \
            shared.aliased_output_c_tile[smem_m_offset * SMEM_N + smem_n_offset] = d.d##REG_INDEX; }
        X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
        X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
        X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)
        X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)
        X(32) X(33) X(34) X(35) X(36) X(37) X(38) X(39)
        X(40) X(41) X(42) X(43) X(44) X(45) X(46) X(47)
        X(48) X(49) X(50) X(51) X(52) X(53) X(54) X(55)
        X(56) X(57) X(58) X(59) X(60) X(61) X(62) X(63)
        #undef X
    }

    // Fill shared memory A tile with SMEM_M×SMEM_K block starting at (cta_m_offset, cta_k_offset)
    // Fill shared memory B^T tile with SMEM_N×SMEM_K block starting at (cta_n_offset, cta_k_offset)
    // Due to the tiling of shared memory, this is done as SMEM_K / SMEM_K_INNER separate copies.
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

            for (uint32_t smem_k_offset = 0; smem_k_offset < SMEM_K; smem_k_offset += SMEM_K_INNER) {
                static_assert(tensorMap_box_m == SMEM_M_OUTER * SMEM_MN_INNER);
                static_assert(tensorMap_box_n == SMEM_N_OUTER * SMEM_MN_INNER);
                static_assert(tensorMap_box_k == SMEM_K_INNER);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(smem_ptr_u32(buffers.a_tile_mk_offset(0, smem_k_offset))),
                      "l"(tensorMap_a),
                      "r"(smem_ptr_u32(&mbar)),
                      "r"(cta_k_offset + smem_k_offset), "r"(cta_m_offset)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(smem_ptr_u32(buffers.bT_tile_nk_offset(0, smem_k_offset))),
                      "l"(tensorMap_bT),
                      "r"(smem_ptr_u32(&mbar)),
                      "r"(cta_k_offset + smem_k_offset), "r"(cta_n_offset)
                    : "memory");
            }
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
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile("wgmma.commit_group.sync.aligned;");
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
                    warp_async_load_block(shared.aliased_input_ring_buffer[ring_idx], shared.tile_fill_mbar[ring_idx],
                                          cta_m_offset, cta_n_offset, cta_k_offset);
                }
            }

            if (!is_memory_wg()) {
                mbar_wait(shared.tile_fill_mbar[ring_idx], ring_usage_parity);
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
                for (uint32_t wg_k_idx = 0; wg_k_idx < SMEM_K / WG_K; ++wg_k_idx) {
                    const uint32_t wg_k_offset = wg_k_idx * WG_K;
                    wg_accum_tile(accum, shared.aliased_input_ring_buffer[ring_idx],
                                  wg_m_offset, wg_n_offset, wg_k_offset, first_time);
                    first_time = false;
                }
                asm volatile(
                    "mbarrier.arrive.shared::cta.b64 _, [%0];"
                    :
                    : "r"(smem_ptr_u32(&shared.tile_read_mbar[ring_idx])));
            }
        }

        __syncthreads();  // Switch from using aliased shared memory for input tile to output tile.

        if (!is_memory_wg()) {
            // Wait for wgmma and copy registers to shared memory C tile.
            asm volatile("wgmma.commit_group.sync.aligned;");
            asm volatile("wgmma.wait_group.sync.aligned 0;");
            const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
            const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
            wg_accum_to_shared(shared, accum, wg_m_offset, wg_n_offset);
        }

        __syncthreads();

        if (is_memory_wg()) {
            // Copy or reduce shared memory C tile to global memory.
            static_assert(tensorMap_box_m == SMEM_M);
            static_assert(tensorMap_box_n == SMEM_N);
            if (threadIdx.x % 128u == 0u) {
                asm volatile("fence.proxy.async;");
                if (using_cp_reduce(size_k)) {
                    asm volatile(
                    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset), "r"(cta_m_offset),
                      "r"(smem_ptr_u32(&shared.aliased_output_c_tile)));
                }
                else {
                    asm volatile(
                    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset), "r"(cta_m_offset),
                      "r"(smem_ptr_u32(&shared.aliased_output_c_tile)));
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
        init_tensorMap(&tensorMap_a, a, size_m, size_k, tensorMap_box_m, tensorMap_box_k);
        init_tensorMap(&tensorMap_bT, bT, size_n, size_k, tensorMap_box_n, tensorMap_box_k);
        init_tensorMap(&tensorMap_c, c, size_m, size_n, tensorMap_box_m, tensorMap_box_n);

        if (using_cp_reduce(size_k)) {
            cudaMemsetAsync(c, 0, size_m * size_n * sizeof(*c), stream);
        }

        const uint32_t grid = m_cta(size_m) * n_cta(size_n) * k_cta(size_k);
        const uint32_t block = cta_size();
        const uint32_t smem = smem_size();
        assert(smem <= 256u << 10);
        cudaFuncSetAttribute(tiled_multiplier_kernel<TiledMultiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        tiled_multiplier_kernel<TiledMultiplier> <<<grid, block, smem, stream>>>(
                size_m, size_n, size_k, tensorMap_a, tensorMap_bT, tensorMap_c);
    }
};

}  // end namespace

void matmul_sm90(GPU_Tensors t, cudaStream_t stream)
{
    constexpr uint32_t smem_m = 128;
    constexpr uint32_t smem_n = 256;
    constexpr uint32_t smem_k = 48;
    constexpr uint32_t cta_k_max_tiles = 36;
    constexpr uint32_t cta_modulus = 4;
    constexpr uint32_t ring_buffer_size = 3;

    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(!t.c_col_major);

    if (size_m % smem_m == 0 && size_n % smem_n == 0 && size_k % smem_k == 0) {
        TiledMultiplier<smem_m, smem_n, smem_k, cta_k_max_tiles, cta_modulus, ring_buffer_size>::launch(
                stream, size_m, size_n, size_k, t.a, t.b, t.c);
    }
    else {
        assert(0);
    }
}
