#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#include <cuda.h>

#define DEVICE_INLINE __device__ __forceinline__

#define DEVICE_ASSERT(x) assert(x)

namespace gemm_sm90 {

using mbarrier_t = long long;

DEVICE_INLINE uint32_t smem_ptr_u32(const void* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

// cute::elect_one_sync
DEVICE_INLINE uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFF));
    return pred;
}

// cutlass::canonical_warp_idx_sync
DEVICE_INLINE int canonical_warp_idx_sync()
{
    return __shfl_sync(0xffffffff, threadIdx.x / 32u, 0);
}

DEVICE_INLINE void prefetch_tensormap(const CUtensorMap* tensorMap)
{
    asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<uint64_t>(tensorMap)) : "memory");
}

struct TiledMultiplierArgs
{
    uint32_t size_m, size_n, size_k;
    CUtensorMap tensorMap_a, tensorMap_bT, tensorMap_c;
    float* c;
};

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size())
tiled_multiplier_kernel(__grid_constant__ const TiledMultiplierArgs args)
{
    Multiplier multiplier{args.size_m, args.size_n, args.size_k,
                          &args.tensorMap_a, &args.tensorMap_bT, &args.tensorMap_c,
                          args.c};
    multiplier.kernel_main();
}

template <uint32_t CLUSTER_M, uint32_t CLUSTER_N, uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K,
          uint32_t WG_M, uint32_t WG_N, uint32_t WG_K,
          uint32_t K_MAX_TILES, uint32_t CLUSTER_MODULUS, uint32_t RING_BUFFER_SIZE, bool DEDICATED_PRODUCER_WG>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;

    const CUtensorMap* tensorMap_a;
    const CUtensorMap* tensorMap_bT;  // Transposed; column major
    const CUtensorMap* tensorMap_c;
    float* c;

    static constexpr uint32_t CLUSTER_M_NUM_CTA = CLUSTER_M / SMEM_M;
    static constexpr uint32_t CLUSTER_N_NUM_CTA = CLUSTER_N / SMEM_N;
    static constexpr uint32_t CLUSTER_NUM_CTA = CLUSTER_M_NUM_CTA * CLUSTER_N_NUM_CTA;
    static_assert(CLUSTER_M_NUM_CTA * SMEM_M == CLUSTER_M);
    static_assert(CLUSTER_N_NUM_CTA * SMEM_N == CLUSTER_N);

    static_assert(WG_M == 64);
    static_assert(WG_K == 8);
    static constexpr CUtensorMapSwizzle input_swizzle = CU_TENSOR_MAP_SWIZZLE_128B;  // Not trivial to change.

    // wgmma core matrix dimensions; each 8 core matrices in K-dimension is swizzled together.
    static_assert(input_swizzle == CU_TENSOR_MAP_SWIZZLE_128B, "Assumed 8 swizzled core matrices");
    static constexpr uint32_t CORE_MATRIX_K = 16u / sizeof(float);
    static constexpr uint32_t CORE_MATRIX_MN = 8u;
    static constexpr uint32_t SMEM_K_INNER = CORE_MATRIX_K * 8;
    static constexpr uint32_t SMEM_MN_INNER = CORE_MATRIX_MN;

    static constexpr uint32_t SMEM_M_OUTER = SMEM_M / SMEM_MN_INNER;
    static constexpr uint32_t SMEM_N_OUTER = SMEM_N / SMEM_MN_INNER;
    static constexpr uint32_t SMEM_K_OUTER = SMEM_K / SMEM_K_INNER;

    // One buffer of ring buffer.
    struct Buffers
    {
        // Inner (rightmost) dimension corresponds to swizzled core matrices.
        // Right 2 dimensions correspond to CLUSTER_{N,M}_NUM_CTA-many TMA boxes.  [N for a, M for bT]
        // We will distribute the boxes across blocks in the same cluster sharing the same cta_{m,n}_idx_cluster().
        // ergo, the a_tile/bT_tile will be distributed among CLUSTER_{N,M}_NUM_CTA-many CTAs; note m/n swap.
        float a_tile[SMEM_K_OUTER][SMEM_M_OUTER][SMEM_K_INNER * SMEM_MN_INNER];
        float bT_tile[SMEM_K_OUTER][SMEM_N_OUTER][SMEM_K_INNER * SMEM_MN_INNER];

        // Number of CTAs cooperating for each a/bT tile (again note M/N swapped from usual a/bT association).
        static constexpr uint32_t a_tile_cta_count = CLUSTER_N_NUM_CTA;
        static constexpr uint32_t bT_tile_cta_count = CLUSTER_M_NUM_CTA;

        // Tensormap box sizes
        static constexpr uint32_t tensorMap_box_m = SMEM_M_OUTER * SMEM_MN_INNER / CLUSTER_N_NUM_CTA;
        static constexpr uint32_t tensorMap_box_n = SMEM_N_OUTER * SMEM_MN_INNER / CLUSTER_M_NUM_CTA;
        static constexpr uint32_t tensorMap_box_k = SMEM_K_INNER;

        DEVICE_INLINE float* a_tma_box(uint32_t k_offset)
        {
            // TMA distributed among CLUSTER_N_NUM_CTA-many CTAs sharing the same cta_m_idx_cluster()
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &a_tile[k_offset / SMEM_K_INNER][SMEM_M_OUTER / CLUSTER_N_NUM_CTA * cta_n_idx_cluster()][0];
            static_assert(SMEM_M_OUTER % CLUSTER_N_NUM_CTA == 0);
        }

        DEVICE_INLINE float* bT_tma_box(uint32_t k_offset)
        {
            // TMA distributed among CLUSTER_M_NUM_CTA-many CTAs sharing the same cta_n_idx_cluster()
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &bT_tile[k_offset / SMEM_K_INNER][SMEM_N_OUTER / CLUSTER_M_NUM_CTA * cta_m_idx_cluster()][0];
            static_assert(SMEM_N_OUTER % CLUSTER_M_NUM_CTA == 0);
        }

        // M-offset into smem A tile assigned for this CTA to fill; this is already accounted for in a_tma_box()
        // but is needed to compute the coordinates for the TMA.
        static DEVICE_INLINE uint32_t cta_smem_m_offset()
        {
            return cta_n_idx_cluster() * (SMEM_M / CLUSTER_N_NUM_CTA);
        }

        // N-offset into smem bT tile assigned for this CTA to fill, also accounted for in bT_tma_box()
        static DEVICE_INLINE uint32_t cta_smem_n_offset()
        {
            return cta_m_idx_cluster() * (SMEM_N / CLUSTER_M_NUM_CTA);
        }

        DEVICE_INLINE const float* a_mk_core_matrices(uint32_t m_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(m_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &a_tile[k_offset / SMEM_K_INNER][m_offset / SMEM_MN_INNER][k_offset % SMEM_K_INNER];
        }

        DEVICE_INLINE const float* bT_nk_core_matrices(uint32_t n_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(n_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &bT_tile[k_offset / SMEM_K_INNER][n_offset / SMEM_MN_INNER][k_offset % SMEM_K_INNER];
        }
    };

    // Configuration for TMA tensormap.
    static constexpr uint32_t tensorMap_a_box_m = Buffers::tensorMap_box_m;
    static constexpr uint32_t tensorMap_bT_box_n = Buffers::tensorMap_box_n;
    static constexpr uint32_t tensorMap_c_box_m = WG_M;
    static constexpr uint32_t tensorMap_c_box_n = WG_N;
    static constexpr uint32_t tensorMap_box_k = Buffers::tensorMap_box_k;

    __host__ DEVICE_INLINE static constexpr uint32_t smem_size()
    {
        static_assert(sizeof(Shared) <= (227u << 10));
        return sizeof(Shared);
    }

    __host__ DEVICE_INLINE static constexpr uint32_t consumer_wg_count()
    {
        // If output matrix is cut into (WG_M, WG_N) blocks, one warpgroup handles one matrix block.
        static_assert(SMEM_M % WG_M == 0);
        static_assert(SMEM_N % WG_N == 0);
        return (SMEM_M / WG_M) * (SMEM_N / WG_N);
    }

    DEVICE_INLINE static uint32_t consumer_wg_index()
    {
        DEVICE_ASSERT(threadIdx.x < 128u * consumer_wg_count());
        return threadIdx.x / 128u;
    }

    // Static assignment of consumer warpgroups within CTA to per-warpgroup output tiles (WG_M, WG_N) within
    // per-CTA output tile (SMEM_M, SMEM_N); one warpgroup is the producer (load tiles w/ TMA) warpgroup.
    //
    // * if DEDICATED_PRODUCER_WG, one extra warpgroup is the producer
    // * if !DEDICATED_PRODUCER_WG, the 0th consumer is also the producer.
    DEVICE_INLINE static bool is_producer_wg()
    {
        return (threadIdx.x / 128u) == (DEDICATED_PRODUCER_WG ? consumer_wg_count() : 0u);
    }

    DEVICE_INLINE static uint32_t get_wg_m_idx()
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        DEVICE_ASSERT(wg_index < consumer_wg_count());
        return wg_index / (SMEM_N / WG_N);
    }

    DEVICE_INLINE static uint32_t get_wg_n_idx()
    {
        const uint32_t wg_index = threadIdx.x / 128u;
        return wg_index % (SMEM_N / WG_N);
    }

    __host__ DEVICE_INLINE static constexpr uint32_t cta_size()
    {
        // Optional 1 extra warpgroup for producer.
        return (DEDICATED_PRODUCER_WG + consumer_wg_count()) * 128;
    }

    // If output matrix is cut into (CLUSTER_M, CLUSTER_N) blocks, one cluster work item handles one output matrix
    // block, unless k_cluster_items > 1; in that case, k_cluster_items-many cluster items cooperate to fill the block
    // using split-k strategy (reduce into output memory).
    //
    // We further partition the output tile into (SMEM_M, SMEM_N) tiles allocated to CTAs within the cluster.
    struct ClusterWorkItem
    {
        uint32_t m_offset, n_offset, k_offset;
    };

    static __host__ DEVICE_INLINE uint32_t m_cluster_items(uint32_t size_m)
    {
        return (size_m + CLUSTER_M - 1) / CLUSTER_M;
    }

    static __host__ DEVICE_INLINE uint32_t n_cluster_items(uint32_t size_n)
    {
        return (size_n + CLUSTER_N - 1) / CLUSTER_N;
    }

    static __host__ DEVICE_INLINE uint32_t k_cluster_items(uint32_t size_k)
    {
        constexpr uint32_t k_divisor = SMEM_K * K_MAX_TILES;
        return (size_k + k_divisor - 1) / k_divisor;
    }

    static __host__ DEVICE_INLINE uint32_t num_cluster_items(uint32_t size_m, uint32_t size_n, uint32_t size_k)
    {
        return m_cluster_items(size_m) * n_cluster_items(size_n) * k_cluster_items(size_k);
    }

    // Assignment of CTAs in cluster tile into indexed sub-tiles, of size (SMEM_M, SMEM_N).
    static DEVICE_INLINE uint32_t cta_m_idx_cluster()
    {
        if constexpr (CLUSTER_M_NUM_CTA == 1) {
            return 0;
        }
        return (blockIdx.x % CLUSTER_NUM_CTA) / CLUSTER_N_NUM_CTA;
    }
    static DEVICE_INLINE uint32_t cta_n_idx_cluster()
    {
        if constexpr (CLUSTER_N_NUM_CTA == 1) {
            return 0;
        }
        return (blockIdx.x % CLUSTER_NUM_CTA) % CLUSTER_N_NUM_CTA;
    }

    // Get mask of CTAs (in the same cluster) with shared cta_m_idx_cluster(), cta_n_idx_cluster() respectively.
    static DEVICE_INLINE uint16_t cta_shared_m_mask()
    {
        uint16_t mask = uint16_t((1u << CLUSTER_N_NUM_CTA) - 1u) << (cta_m_idx_cluster() * CLUSTER_N_NUM_CTA);
        DEVICE_ASSERT(__popc(mask) == CLUSTER_N_NUM_CTA);
        DEVICE_ASSERT(mask < (1u << CLUSTER_NUM_CTA));
        return mask;
    }
    static DEVICE_INLINE uint16_t cta_shared_n_mask()
    {
        uint16_t base_mask = 0u;
        for (uint32_t i = 0; i < CLUSTER_M_NUM_CTA; ++i) {
            base_mask = base_mask << CLUSTER_N_NUM_CTA | 1u;  // Hopefully handled at compile time
        }
        uint16_t mask = base_mask << cta_n_idx_cluster();
        DEVICE_ASSERT(__popc(mask) == CLUSTER_M_NUM_CTA);
        DEVICE_ASSERT(mask < (1u << CLUSTER_NUM_CTA));
        return mask;
    }

    // Linearized indexing of all ClusterWorkItem
    DEVICE_INLINE ClusterWorkItem cluster_get_item(uint32_t item_idx) const
    {
        DEVICE_ASSERT(item_idx < num_cluster_items(size_m, size_n, size_k));
        const uint32_t cluster_rows = m_cluster_items(size_m);
        const uint32_t cluster_cols = n_cluster_items(size_n);
        const uint32_t num_output_block_matrix = cluster_rows * cluster_cols;
        const uint32_t cluster_k_group = k_cluster_items(size_k);
        const uint32_t cluster_col_remainder = cluster_cols % CLUSTER_MODULUS;
        const uint32_t superblock_count = cluster_cols / CLUSTER_MODULUS;
        const uint32_t superblock_cluster_count = cluster_rows * CLUSTER_MODULUS;
        const uint32_t superblock_idx = (item_idx % num_output_block_matrix) / superblock_cluster_count;
        const uint32_t cluster_idx_in_superblock = (item_idx % num_output_block_matrix) % superblock_cluster_count;
        const uint32_t cluster_k_idx = item_idx / num_output_block_matrix;
        const uint32_t cluster_k_offset = cluster_k_idx * (SMEM_K * K_MAX_TILES);

        uint32_t cluster_m_idx, cluster_n_idx;

        if (superblock_idx < superblock_count) {
            cluster_m_idx = cluster_idx_in_superblock / CLUSTER_MODULUS;
            cluster_n_idx = cluster_idx_in_superblock % CLUSTER_MODULUS + CLUSTER_MODULUS * superblock_idx;
        }
        else {
            DEVICE_ASSERT(superblock_idx == superblock_count);
            cluster_m_idx = cluster_idx_in_superblock / cluster_col_remainder;
            cluster_n_idx = cluster_idx_in_superblock % cluster_col_remainder + CLUSTER_MODULUS * superblock_idx;
        }
        DEVICE_ASSERT(cluster_m_idx < cluster_rows);
        DEVICE_ASSERT(cluster_n_idx < cluster_cols);

        return ClusterWorkItem{cluster_m_idx * CLUSTER_M, cluster_n_idx * CLUSTER_N, cluster_k_offset};
    }

    static __host__ DEVICE_INLINE bool using_cp_reduce(uint32_t size_k)
    {
#ifndef __CUDA_ARCH__
        assert(k_cluster_items(size_k) != 0);
#endif
        return k_cluster_items(size_k) != 1;
    }

    // Arrive on the mbarrier only for this CTA
    static DEVICE_INLINE void mbar_arrive_cta(mbarrier_t& mbar)
    {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_ptr_u32(&mbar)));
    }

    // Arrive on the corresponding mbarrier for all CTAs in the cluster
    static DEVICE_INLINE void mbar_arrive_cluster_broadcast(mbarrier_t& mbar)
    {
        if constexpr (CLUSTER_NUM_CTA == 1) {
            mbar_arrive_cta(mbar);
        }
        else {
            const uint32_t addr_cta = smem_ptr_u32(&mbar);
            for (uint32_t i = 0; i < CLUSTER_NUM_CTA; ++i) {
                uint32_t addr_cluster;
                asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(addr_cluster) : "r"(addr_cta), "r"(i));
                asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(addr_cluster));
            }
        }
    }

    struct Shared
    {
        union {
            Buffers aliased_input_ring_buffer[RING_BUFFER_SIZE];
            float aliased_per_wg_c_tile[consumer_wg_count()][WG_M * WG_N];
        };
        mbarrier_t tile_fill_mbar[RING_BUFFER_SIZE];
        mbarrier_t tile_read_mbar[RING_BUFFER_SIZE];
        mbarrier_t per_consumer_wg_mbar[consumer_wg_count()];
        mbarrier_t cta_all_consumers_mbar;
    };

    // Helper to wait on an mbarrier while tracking the phase.
    struct PhaseBits
    {
        unsigned tile_fill_bits : RING_BUFFER_SIZE = 0;
        unsigned tile_read_bits : RING_BUFFER_SIZE = 0;
        unsigned per_consumer_wg_bit : 1 = 0;
        unsigned all_consumers_bit : 1 = 0;

        DEVICE_INLINE void tile_fill_wait(uint32_t i, Shared& shared)
        {
            DEVICE_ASSERT(i < RING_BUFFER_SIZE);
            mbar_wait(shared.tile_fill_mbar[i], (tile_fill_bits >> i) & 1u);
            tile_fill_bits ^= 1u << i;
        }

        DEVICE_INLINE void tile_read_wait(uint32_t i, Shared& shared)
        {
            DEVICE_ASSERT(i < RING_BUFFER_SIZE);
            mbar_wait(shared.tile_read_mbar[i], (tile_read_bits >> i) & 1u);
            tile_read_bits ^= 1u << i;
        }

        DEVICE_INLINE void per_consumer_wg_wait(bool enable_wait, Shared& shared)
        {
            if (enable_wait) {
                mbar_wait(shared.per_consumer_wg_mbar[consumer_wg_index()], per_consumer_wg_bit);
            }
            per_consumer_wg_bit ^= 1;
        }

        DEVICE_INLINE void cta_all_consumers_wait(Shared& shared)
        {
            mbar_wait(shared.cta_all_consumers_mbar, all_consumers_bit);
            all_consumers_bit ^= 1;
        }

        static DEVICE_INLINE void mbar_wait(mbarrier_t& mbar, uint32_t parity)
        {
            asm volatile(
                    "{.reg.pred P1; BEFORE_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni WAIT_DONE; bra.uni BEFORE_WAIT; WAIT_DONE: }"
            :
            : "r"(smem_ptr_u32(&mbar)), "r"(parity));
        }
    };


    static constexpr uint32_t wgmma_regcount = WG_M * WG_N / 128u;

    struct WG_Accum_m64n64
    {
        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
    };

    struct WG_Accum_m64n96
    {
        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
        float d32, d33, d34, d35, d36, d37, d38, d39;
        float d40, d41, d42, d43, d44, d45, d46, d47;
    };

    struct WG_Accum_m64n128
    {
        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
        float d32, d33, d34, d35, d36, d37, d38, d39;
        float d40, d41, d42, d43, d44, d45, d46, d47;
        float d48, d49, d50, d51, d52, d53, d54, d55;
        float d56, d57, d58, d59, d60, d61, d62, d63;
    };

    // Per-warpgroup accumulator, holding one (WG_M, WG_N) tile.
    using WG_Accum = std::conditional_t<WG_N == 64, WG_Accum_m64n64, std::conditional_t<WG_N == 96, WG_Accum_m64n96, WG_Accum_m64n128>>;
    static_assert(WG_N == 64 || WG_N == 96 || WG_N == 128);
    static_assert(wgmma_regcount == sizeof(WG_Accum) / 4u);

    static DEVICE_INLINE uint64_t matrix_descriptor_encode(uint32_t val)
    {
        DEVICE_ASSERT(val % 16u == 0);
        uint64_t enc = (val & 0x3FFFF) >> 4;
        return enc;
    }

    static DEVICE_INLINE uint64_t matrix_descriptor_mn_k_stride(const float* smem_ptr,
                                                                uint32_t mn_stride, uint32_t k_stride)
    {
        // Swizzling encoding isn't the same on the host API side and the device side lol (swap 1 and 3)
        constexpr unsigned enum_as_uint = static_cast<unsigned>(input_swizzle);
        static_assert(enum_as_uint <= 3);
        const unsigned swizzle_bits = enum_as_uint == 3 ? 1 : enum_as_uint == 1 ? 3 : enum_as_uint;
        return matrix_descriptor_encode(smem_ptr_u32(smem_ptr))
               | matrix_descriptor_encode(k_stride) << 16u
               | matrix_descriptor_encode(mn_stride) << 32u
               | uint64_t(swizzle_bits) << 62;
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
        static_assert(input_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE, "need to set k stride");
        auto desc_a = matrix_descriptor_mn_k_stride(buffers.a_mk_core_matrices(wg_m_offset, wg_k_offset),
                                                    SMEM_MN_INNER * SMEM_K_INNER * sizeof(float), 0);
        auto desc_b = matrix_descriptor_mn_k_stride(buffers.bT_nk_core_matrices(wg_n_offset, wg_k_offset),
                                                    SMEM_MN_INNER * SMEM_K_INNER * sizeof(float), 0);
        static_assert(WG_M == 64);
        static_assert(WG_K == 8);

        if constexpr (WG_N == 64) {
            asm volatile(
            "{  // GMMA \n"
              ".reg .pred p;\n"
              "setp.ne.b32 p, %34, 0;\n"
              "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
              "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
              " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
              " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
              " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},  "
              " %32,"
              " %33,"
              " p,    1,  1;\n"
            "}\n"
              : "+f"(d.d0), "+f"(d.d1), "+f"(d.d2), "+f"(d.d3),
                "+f"(d.d4), "+f"(d.d5), "+f"(d.d6), "+f"(d.d7),
                "+f"(d.d8), "+f"(d.d9), "+f"(d.d10), "+f"(d.d11),
                "+f"(d.d12), "+f"(d.d13), "+f"(d.d14), "+f"(d.d15),
                "+f"(d.d16), "+f"(d.d17), "+f"(d.d18), "+f"(d.d19),
                "+f"(d.d20), "+f"(d.d21), "+f"(d.d22), "+f"(d.d23),
                "+f"(d.d24), "+f"(d.d25), "+f"(d.d26), "+f"(d.d27),
                "+f"(d.d28), "+f"(d.d29), "+f"(d.d30), "+f"(d.d31)
              :  "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(!zero_output)));

        }
        else if constexpr (WG_N == 96) {
            asm volatile(
            "{  // GMMA \n"
              ".reg .pred p;\n"
              "setp.ne.b32 p, %50, 0;\n"
              "wgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32 "
              "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
              " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
              " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
              " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
              " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
              " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47}, "
              " %48,"
              " %49,"
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
                "+f"(d.d44), "+f"(d.d45), "+f"(d.d46), "+f"(d.d47)
              :  "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(!zero_output)));
        }
        else if constexpr (WG_N == 128) {
            asm volatile(
            "{  // GMMA \n"
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
        else {
            static_assert(WG_N != WG_N, "Add code for wgmma of this size");
        }
    }

    template <bool BoundsCheck>
    DEVICE_INLINE void wg_accum_store_to_tile(float* tile, uint32_t row_stride, uint32_t r_guard, uint32_t c_guard,
                                              const WG_Accum& d) const
    {
        const uint32_t tid = threadIdx.x % 128u;

        #define X(REG_INDEX) { \
            const uint32_t r = (tid / 32u) * 16u + (tid % 32u) / 4u + ((REG_INDEX % 4u) / 2u) * 8u; \
            const uint32_t c = (tid % 4u) * 2u + (REG_INDEX / 4u) * 8 + (REG_INDEX % 2u); \
            if (!BoundsCheck || (r < r_guard && c < c_guard)) tile[r * row_stride + c] = d.d##REG_INDEX; }

        X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
        X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
        X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)
        X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)
        if constexpr (WG_N >= 96) {
            X(32) X(33) X(34) X(35) X(36) X(37) X(38) X(39)
            X(40) X(41) X(42) X(43) X(44) X(45) X(46) X(47)
        }
        if constexpr (WG_N >= 128) {
            X(48) X(49) X(50) X(51) X(52) X(53) X(54) X(55)
            X(56) X(57) X(58) X(59) X(60) X(61) X(62) X(63)
        }
        static_assert(WG_N == 64 || WG_N == 96 || WG_N == 128, "Add code for wgmma of this size");
        #undef X
    }

    // Warpgroup-convergent code
    // Write the (WG_M, WG_N) tile to shared.aliased_per_wg_c_tile, at the entry reserved for this warpgroup.
    DEVICE_INLINE void wg_accum_to_shared(Shared& shared, const WG_Accum& d) const
    {
        wg_accum_store_to_tile<false>(shared.aliased_per_wg_c_tile[consumer_wg_index()], WG_N, 0, 0, d);
    }

    template <uint32_t MulticastCtaCount>
    DEVICE_INLINE void tma_maybe_multicast(float* dst, const CUtensorMap* tensorMap, mbarrier_t& mbar,
                                           uint32_t k_coord, uint32_t mn_coord, uint16_t cta_mask) const
    {
        DEVICE_ASSERT(__popc(cta_mask) == MulticastCtaCount);
        constexpr uint64_t cache_hint = 1152921504606846976;  // ??? no idea what this does, copied from cutlass PTX
        if constexpr (MulticastCtaCount == 1) {
            asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :
                    : "r"(smem_ptr_u32(dst)),
                      "l"(tensorMap), "r"(k_coord), "r"(mn_coord),
                      "r"(smem_ptr_u32(&mbar)), "n"(cache_hint)
                    : "memory");
        }
        else {
            asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                    :
                    : "r"(smem_ptr_u32(dst)),
                      "l"(tensorMap), "r"(k_coord), "r"(mn_coord),
                      "r"(smem_ptr_u32(&mbar)), "h"(cta_mask), "n"(cache_hint)
                    : "memory");
        }
    }

    // Fill shared memory A tile with SMEM_M×SMEM_K block starting at (cta_m_offset, cta_k_offset)
    // Fill shared memory B^T tile with SMEM_N×SMEM_K block starting at (cta_n_offset, cta_k_offset)
    //
    // Due to the tiling of shared memory, this is done as SMEM_K / SMEM_K_INNER separate copies.
    //
    // Furthermore, if the cluster size is not 1, we assume all blocks in the cluster call this function, and we
    // distribute+multicast the copies. We signal the mbar of all CTAs of the cluster that received
    // the same a/bT tile.
    DEVICE_INLINE void warp_async_load_block_clustered(
            Buffers& buffers, mbarrier_t& mbar,
            uint32_t cta_m_offset, uint32_t cta_n_offset, uint32_t cta_k_offset) const
    {
        if (elect_one_sync()) {
            constexpr uint32_t Tsz = sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                         "r"(smem_ptr_u32(&mbar)), "n"((SMEM_M + SMEM_N) * SMEM_K * Tsz));

            for (uint32_t smem_k_offset = 0; smem_k_offset < SMEM_K; smem_k_offset += SMEM_K_INNER) {
                static_assert(tensorMap_a_box_m == SMEM_M_OUTER * SMEM_MN_INNER / Buffers::a_tile_cta_count);
                static_assert(tensorMap_bT_box_n == SMEM_N_OUTER * SMEM_MN_INNER / Buffers::bT_tile_cta_count);
                static_assert(tensorMap_box_k == SMEM_K_INNER);

                tma_maybe_multicast<Buffers::a_tile_cta_count>(
                    buffers.a_tma_box(smem_k_offset), tensorMap_a, mbar,
                    cta_k_offset + smem_k_offset, cta_m_offset + Buffers::cta_smem_m_offset(), cta_shared_m_mask());

                tma_maybe_multicast<Buffers::bT_tile_cta_count>(
                    buffers.bT_tma_box(smem_k_offset), tensorMap_bT, mbar,
                    cta_k_offset + smem_k_offset, cta_n_offset + Buffers::cta_smem_n_offset(), cta_shared_n_mask());
            }
        }
    }

    template <uint32_t ARRIVE_THREADS>
    DEVICE_INLINE void init_mbar(mbarrier_t& mbar) const
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_ptr_u32(&mbar)), "n"(ARRIVE_THREADS));
    }

    template <uint32_t ARRIVE_THREADS, uint32_t COUNT>
    DEVICE_INLINE void cta_init_mbar_array(mbarrier_t (&mbar_array) [COUNT]) const
    {
        DEVICE_ASSERT(COUNT < blockDim.x);
        if (threadIdx.x < COUNT) {
            init_mbar<ARRIVE_THREADS>(mbar_array[threadIdx.x]);
        }
    }

    DEVICE_INLINE void cta_first_time_init(Shared& shared) const
    {
        constexpr unsigned cta_consumer_thread_count = consumer_wg_count() * 128u;
        constexpr unsigned cluster_consumer_thread_count = consumer_wg_count() * 128u * CLUSTER_NUM_CTA;

        cta_init_mbar_array<1>(shared.tile_fill_mbar);
        cta_init_mbar_array<cluster_consumer_thread_count>(shared.tile_read_mbar);
        cta_init_mbar_array<128>(shared.per_consumer_wg_mbar);
        if (threadIdx.x == 0) {
            init_mbar<cta_consumer_thread_count>(shared.cta_all_consumers_mbar);
        }
        asm volatile("fence.proxy.async;");

        if (canonical_warp_idx_sync() == 0 && elect_one_sync()) {
            prefetch_tensormap(tensorMap_a);
            prefetch_tensormap(tensorMap_bT);
            prefetch_tensormap(tensorMap_c);
        }
    }

    DEVICE_INLINE void cluster_sync() const
    {
        if constexpr (CLUSTER_NUM_CTA == 1) {
            __syncthreads();
        }
        else {
            asm("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;\n"::);
        }
    }

    // Cluster cooperates to fill or add to the output matrix block of size (CLUSTER_M, CLUSTER_N) starting at
    // (cta_item.m_offset, cta_item.n_offset); we process up to K_MAX_TILES input blocks on the K dimension starting
    // at cta_item.k_offset.
    // Requires smem-allocated ring buffer.
    //
    // We assume that cluster_sync is run between calls to this. We also require a cluster sync before kernel exit
    // if a non-1 cluster size is used, to prevent unexpected dead shared memory.
    template <bool ENABLE_PRODUCER_BRANCH, bool IS_CONSUMER>
    DEVICE_INLINE void cluster_process_item(ClusterWorkItem cluster_item, PhaseBits& phase_bits, Shared& shared) const
    {
        const uint32_t cta_m_offset = cluster_item.m_offset + cta_m_idx_cluster() * SMEM_M;
        const uint32_t cta_n_offset = cluster_item.n_offset + cta_n_idx_cluster() * SMEM_N;
        const uint32_t cta_k_initial_offset = cluster_item.k_offset;

        DEVICE_ASSERT(cta_m_offset % SMEM_M == 0);
        DEVICE_ASSERT(cta_n_offset % SMEM_N == 0);
        DEVICE_ASSERT(cta_k_initial_offset % (SMEM_K * K_MAX_TILES) == 0);
        DEVICE_ASSERT(ENABLE_PRODUCER_BRANCH || !is_producer_wg());
        DEVICE_ASSERT(IS_CONSUMER || is_producer_wg());

        const uint32_t k_num_iters = min((size_k - cta_k_initial_offset + SMEM_K - 1u) / SMEM_K, K_MAX_TILES);

        std::conditional_t<IS_CONSUMER, WG_Accum, char> accum;
        bool zero_accum = true;

        auto producer_on_k_iter = [&] (uint32_t k_iter)
        {
            if constexpr (ENABLE_PRODUCER_BRANCH) {
                const uint32_t ring_idx = k_iter % RING_BUFFER_SIZE;
                if (is_producer_wg() && k_iter < k_num_iters) {
                    const uint32_t tma_k_offset = cta_k_initial_offset + SMEM_K * k_iter;
                    if (k_iter >= RING_BUFFER_SIZE) {
                        // After initial fill of ring buffer, we need to wait for the producers to signal
                        // they are done reading the tile before overwriting it.
                        phase_bits.tile_read_wait(ring_idx, shared);
                    }
                    if (threadIdx.x % 128u < 32u) {
                        warp_async_load_block_clustered(
                                shared.aliased_input_ring_buffer[ring_idx],
                                shared.tile_fill_mbar[ring_idx],
                                cta_m_offset, cta_n_offset, tma_k_offset);
                    }
                }
            }
        };

        // Producer kicks of loads to fill the ring buffer except the last .
        for (uint32_t k_iter = 0; k_iter < k_num_iters && k_iter < RING_BUFFER_SIZE - 1; ++k_iter) {
            producer_on_k_iter(k_iter);
        }

        for (uint32_t k_iter = 0; k_iter < k_num_iters; ++k_iter) {
            const uint32_t ring_idx = k_iter % RING_BUFFER_SIZE;

            if constexpr (IS_CONSUMER) {
                phase_bits.tile_fill_wait(ring_idx, shared);
                asm volatile("wgmma.fence.sync.aligned;  // GMMA");
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;

                for (uint32_t wg_k_idx = 0; wg_k_idx < SMEM_K / WG_K; ++wg_k_idx) {
                    const uint32_t wg_k_offset = wg_k_idx * WG_K;
                    wg_accum_tile(accum, shared.aliased_input_ring_buffer[ring_idx],
                                  wg_m_offset, wg_n_offset, wg_k_offset, zero_accum);
                    zero_accum = false;
                }
                asm volatile("wgmma.commit_group.sync.aligned;  // GMMA");

                // Wait for previous iteration's wgmma to retire, then signal that the tiles read from
                // the previous iteration may be overwritten by the full cluster. We skip this signalling, however,
                // if it's known that the tile will not be filled (due to reaching the iteration count).
                if (k_iter >= 1) {
                    asm volatile("wgmma.wait_group.sync.aligned 1;  // GMMA");
                    if (k_iter + RING_BUFFER_SIZE - 1 < k_num_iters) {
                        mbar_arrive_cluster_broadcast(shared.tile_read_mbar[(k_iter - 1) % RING_BUFFER_SIZE]);
                        static_assert(RING_BUFFER_SIZE >= 2);
                    }
                }
            }

            // Keep pipelining producer (fill tile for RING_BUFFER_SIZE-1 iterations ahead).
            producer_on_k_iter(k_iter + RING_BUFFER_SIZE - 1);
        }

        if constexpr (IS_CONSUMER) {
            // Wait for all wgmma to finish
            asm volatile("wgmma.wait_group.sync.aligned 0;  // GMMA");
            const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
            const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;

            if (using_cp_reduce(size_k)) {
                // After wgmma finished, issue all-to-all consumer warpgroups sync (inside CTA),
                // indicating that SMEM is ready to be repurposed.
                mbar_arrive_cta(shared.cta_all_consumers_mbar);
                phase_bits.cta_all_consumers_wait(shared);

                // Now copy wgmma registers to shared memory C tile.
                wg_accum_to_shared(shared, accum);

                // 0th thread per consumer warpgroup waits for its own warpgroup and
                // issues TMA copy from shared tile to global.
                static_assert(WG_M == tensorMap_c_box_m);
                static_assert(WG_N == tensorMap_c_box_n);

                const bool elected = threadIdx.x % 128u < 32 && elect_one_sync();
                asm volatile("fence.proxy.async;");
                mbar_arrive_cta(shared.per_consumer_wg_mbar[consumer_wg_index()]);
                phase_bits.per_consumer_wg_wait(elected, shared);
                if (elected) {
                    asm volatile(
                    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset + wg_n_offset), "r"(cta_m_offset + wg_m_offset),
                      "r"(smem_ptr_u32(&shared.aliased_per_wg_c_tile[consumer_wg_index()])));
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    // We must wait for the TMA to complete before exiting this function, otherwise the
                    // shared memory might get stomped on before the TMA finishing reading from it.
                }
            }
            else {
                // Store tile directly to output memory, bypassing TMA and shared memory.
                // We need to bounds check as TMA is not being used.
                const uint32_t arg_m_offset = cta_m_offset + wg_m_offset;
                const uint32_t arg_n_offset = cta_n_offset + wg_n_offset;
                wg_accum_store_to_tile<true>(c + arg_m_offset * size_n + arg_n_offset,
                                             size_n, size_m - arg_m_offset, size_n - arg_n_offset, accum);
            }
        }
    }

    template <bool ENABLE_PRODUCER_BRANCH, bool IS_CONSUMER>
    DEVICE_INLINE void cluster_main_loop(PhaseBits& phase_bits, Shared& smem) const
    {
        const uint32_t num_items = num_cluster_items(size_m, size_n, size_k);
        const uint32_t item_stride = gridDim.x / CLUSTER_NUM_CTA;
        for (uint32_t item_idx = blockIdx.x / CLUSTER_NUM_CTA; item_idx < num_items; item_idx += item_stride) {
            cluster_sync();
            cluster_process_item<ENABLE_PRODUCER_BRANCH, IS_CONSUMER>(cluster_get_item(item_idx), phase_bits, smem);
        }
    }

    DEVICE_INLINE void kernel_main()
    {
        DEVICE_ASSERT(blockDim.x == cta_size());

        extern __shared__ char raw_smem[];
        Shared& smem = *reinterpret_cast<Shared*>(raw_smem);
        PhaseBits phase_bits{};

        cta_first_time_init(smem);

        if constexpr (!DEDICATED_PRODUCER_WG) {
            cluster_main_loop<true, true>(phase_bits, smem);
        }
        else if (is_producer_wg()) {
            cluster_main_loop<true, false>(phase_bits, smem);
        }
        else {
            cluster_main_loop<false, true>(phase_bits, smem);
        }

        if constexpr (CLUSTER_NUM_CTA != 1) {
            cluster_sync();  // Required for distributed shared memory safety, I think?
        }
    }

    static void init_tensorMap(CUtensorMap* tensorMap, const float* globalAddress, uint32_t rows, uint32_t cols,
                               uint32_t smem_rows, uint32_t smem_cols, CUtensorMapSwizzle swizzle)
    {
        const CUtensorMapDataType tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        const uint32_t tensorRank = 2;
        const cuuint64_t globalDim[2] = {cols, rows};
        const cuuint64_t globalStrides[1] = {4*cols};
        const cuuint32_t boxDim[2] = {smem_cols, smem_rows};
        const cuuint32_t elementStrides[2] = {1, 1};
        const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
        const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
        const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

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
            fprintf(stderr, "cuTensorMapEncodeTiled: %i {%u, %u}\n", (int)result, smem_cols, smem_rows);
            assert(0);
        }
    }

    static void launch(cudaStream_t stream, uint32_t size_m, uint32_t size_n, uint32_t size_k,
                       const float* a, const float* bT, float* c)
    {
        CUtensorMap tensorMap_a, tensorMap_bT, tensorMap_c;
        init_tensorMap(&tensorMap_a, a, size_m, size_k, tensorMap_a_box_m, tensorMap_box_k, input_swizzle);
        init_tensorMap(&tensorMap_bT, bT, size_n, size_k, tensorMap_bT_box_n, tensorMap_box_k, input_swizzle);
        init_tensorMap(&tensorMap_c, c, size_m, size_n, tensorMap_c_box_m, tensorMap_c_box_n, CU_TENSOR_MAP_SWIZZLE_NONE);

        if (using_cp_reduce(size_k)) {
            cudaMemsetAsync(c, 0, size_m * size_n * sizeof(*c), stream);
        }

        const uint32_t max_cluster = 132 / CLUSTER_NUM_CTA;
        const uint32_t num_cluster_items = m_cluster_items(size_m) * n_cluster_items(size_n) * k_cluster_items(size_k);
        const uint32_t grid = (num_cluster_items < max_cluster ? num_cluster_items : max_cluster) * CLUSTER_NUM_CTA;
        const uint32_t block = cta_size();
        const uint32_t smem = smem_size();

        const auto kernel = tiled_multiplier_kernel<TiledMultiplier>;
        static std::once_flag f;
        std::call_once(f, [&] () {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (1) {
                fprintf(stderr, "GRID:    %u\n", grid);
                fprintf(stderr, "BLOCK:   %u\n", block);
                fprintf(stderr, "SMEM:    %g KiB\n", double(smem) / 1024.0);
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, kernel);
                fprintf(stderr, "numRegs: %i\n", attr.numRegs);
            }
        });

        TiledMultiplierArgs args{size_m, size_n, size_k, tensorMap_a, tensorMap_bT, tensorMap_c, c};

        cudaLaunchConfig_t config = {0};
        config.gridDim = dim3{grid, 1, 1};
        config.blockDim = dim3{block, 1, 1};
        config.dynamicSmemBytes = smem;
        config.stream = stream;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = CLUSTER_NUM_CTA;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        if (CLUSTER_NUM_CTA != 1) {
            // For some reason setting a cluster size of (1, 1, 1) tanks performance even though it should do nothing!
            config.attrs = attribute;
            config.numAttrs = 1;
        }
        cudaLaunchKernelEx(&config, kernel, args);
    }
};

}  // end namespace

void matmul_sm90(GPU_Tensors t, cudaStream_t stream)
{
    using namespace gemm_sm90;

    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(!t.c_col_major);

    assert(uintptr_t(t.a) % 16 == 0);
    assert(uintptr_t(t.b) % 16 == 0);
    assert(uintptr_t(t.c) % 16 == 0);
    assert(t.K % 4 == 0);

    constexpr uint32_t cluster_m = 256;
    constexpr uint32_t cluster_n = 256;
    constexpr uint32_t smem_m = 256;
    constexpr uint32_t smem_n = 128;
    constexpr uint32_t smem_k = 32;
    constexpr uint32_t wg_m = 64;
    constexpr uint32_t wg_n = 128;
    constexpr uint32_t wg_k = 8;
    constexpr uint32_t cta_k_max_tiles = 16384u / smem_k;
    constexpr uint32_t cluster_modulus = 1024u / cluster_m;
    constexpr uint32_t ring_buffer_size = 4;
    constexpr bool dedicated_producer = true;

    using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                       cta_k_max_tiles, cluster_modulus, ring_buffer_size, dedicated_producer>;
    Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
}
