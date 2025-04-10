#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <utility>

#include <cuda.h>

#define DEVICE_INLINE __device__ __forceinline__

#define DEVICE_ASSERT(x) assert(x)

namespace gemm_sm90 {

constexpr uint32_t MAX_CTA = 132;

// Controls use of TMA and its .reduce functionality for writing the OUTPUT to GMEM.
enum class TmaMode
{
    // Never use TMA, hence never reduce to GMEM
    never_reduce,

    // Use TMA conditionally, only if reducing to GMEM
    if_reduce,

    // Use TMA always, and force use of the reduce mode
    always_force_reduce,

    // Use TMA always, conditional use of the reduce mode
    always_cond_reduce,
};

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
    CUtensorMap tensorMap_aT, tensorMap_bN, tensorMap_cN;
    float* cN;
};

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size())
tiled_multiplier_kernel(__grid_constant__ const TiledMultiplierArgs args)
{
    Multiplier multiplier{args.size_m, args.size_n, args.size_k,
                          &args.tensorMap_aT, &args.tensorMap_bN, &args.tensorMap_cN, args.cN};
    multiplier.kernel_main();
}

template <uint32_t CLUSTER_M, uint32_t CLUSTER_N, uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K,
          uint32_t WG_M, uint32_t WG_N, uint32_t WG_K, gemm_sm90_k_mode K_MODE,
          uint32_t CLUSTER_MODULUS, uint32_t RING_BUFFER_SIZE, bool DEDICATED_PRODUCER_WG>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;

    const CUtensorMap* tensorMap_aT;  // Row major
    const CUtensorMap* tensorMap_bN;  // Column major
    const CUtensorMap* tensorMap_cN;  // Column major
    float* cN;

    static constexpr uint32_t split_k_divisor = 16384;  // Maybe should not hard wire this.

    static constexpr uint32_t CLUSTER_M_NUM_CTA = CLUSTER_M / SMEM_M;
    static constexpr uint32_t CLUSTER_N_NUM_CTA = CLUSTER_N / SMEM_N;
    static constexpr uint32_t CLUSTER_NUM_CTA = CLUSTER_M_NUM_CTA * CLUSTER_N_NUM_CTA;
    static_assert(CLUSTER_M_NUM_CTA * SMEM_M == CLUSTER_M);
    static_assert(CLUSTER_N_NUM_CTA * SMEM_N == CLUSTER_N);

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
        // Right 2 dimensions correspond to CLUSTER_{N,M}_NUM_CTA-many TMA boxes.  [N for aT, M for bN]
        // We will distribute the boxes across blocks in the same cluster sharing the same cta_{m,n}_idx_cluster().
        // ergo, the aT_tile/bN_tile will be distributed among CLUSTER_{N,M}_NUM_CTA-many CTAs; note m/n swap.
        float aT_tile[SMEM_K_OUTER][SMEM_M_OUTER][SMEM_K_INNER * SMEM_MN_INNER];
        float bN_tile[SMEM_K_OUTER][SMEM_N_OUTER][SMEM_K_INNER * SMEM_MN_INNER];

        // Number of CTAs cooperating for each aT/bN tile (again note M/N swapped from usual aT/bN association).
        static constexpr uint32_t aT_tile_cta_count = CLUSTER_N_NUM_CTA;
        static constexpr uint32_t bN_tile_cta_count = CLUSTER_M_NUM_CTA;

        // Tensormap box sizes
        static constexpr uint32_t tensorMap_box_m = SMEM_M_OUTER * SMEM_MN_INNER / CLUSTER_N_NUM_CTA;
        static constexpr uint32_t tensorMap_box_n = SMEM_N_OUTER * SMEM_MN_INNER / CLUSTER_M_NUM_CTA;
        static constexpr uint32_t tensorMap_box_k = SMEM_K_INNER;

        DEVICE_INLINE float* aT_tma_box(uint32_t k_offset)
        {
            // TMA distributed among CLUSTER_N_NUM_CTA-many CTAs sharing the same cta_m_idx_cluster()
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &aT_tile[k_offset / SMEM_K_INNER][SMEM_M_OUTER / CLUSTER_N_NUM_CTA * cta_n_idx_cluster()][0];
            static_assert(SMEM_M_OUTER % CLUSTER_N_NUM_CTA == 0);
        }

        DEVICE_INLINE float* bN_tma_box(uint32_t k_offset)
        {
            // TMA distributed among CLUSTER_M_NUM_CTA-many CTAs sharing the same cta_n_idx_cluster()
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &bN_tile[k_offset / SMEM_K_INNER][SMEM_N_OUTER / CLUSTER_M_NUM_CTA * cta_m_idx_cluster()][0];
            static_assert(SMEM_N_OUTER % CLUSTER_M_NUM_CTA == 0);
        }

        // M-offset into smem A tile assigned for this CTA to fill; this is already accounted for in aT_tma_box()
        // but is needed to compute the coordinates for the TMA.
        static DEVICE_INLINE uint32_t cta_smem_m_offset()
        {
            return cta_n_idx_cluster() * (SMEM_M / CLUSTER_N_NUM_CTA);
        }

        // N-offset into smem bN tile assigned for this CTA to fill, also accounted for in bN_tma_box()
        static DEVICE_INLINE uint32_t cta_smem_n_offset()
        {
            return cta_m_idx_cluster() * (SMEM_N / CLUSTER_M_NUM_CTA);
        }

        DEVICE_INLINE const float* aT_mk_core_matrices(uint32_t m_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(m_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &aT_tile[k_offset / SMEM_K_INNER][m_offset / SMEM_MN_INNER][k_offset % SMEM_K_INNER];
        }

        DEVICE_INLINE const float* bN_nk_core_matrices(uint32_t n_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(n_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &bN_tile[k_offset / SMEM_K_INNER][n_offset / SMEM_MN_INNER][k_offset % SMEM_K_INNER];
        }
    };

    // Configuration for TMA tensormap.
    static constexpr uint32_t tensorMap_aT_box_m = Buffers::tensorMap_box_m;
    static constexpr uint32_t tensorMap_bN_box_n = Buffers::tensorMap_box_n;
    static constexpr uint32_t tensorMap_cN_box_m = WG_M;
    static constexpr uint32_t tensorMap_cN_box_n = WG_N;
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

    // A (CLUSTER_M, CLUSTER_N) size tile of the output matrix.
    struct ClusterMN_Tile
    {
        uint32_t m_offset, n_offset;
    };

    // A (CLUSTER_M, CLUSTER_N, SMEM_K) size tile of M x N x K work space.
    // This is the "smallest" unit of work that can be assigned to one cluster.
    struct ClusterMN_SmemK_Tile
    {
        uint32_t m_offset, n_offset, k_offset;
    };

    // A (CLUSTER_M, CLUSTER_N, k_work_size) size portion of the M x N x K work space.
    // Aggregation of ClusterMN_SmemK_Tile that are consecutive in k (thus k_work_size must be divisible by SMEM_K).
    //
    // We further partition the output tile into (SMEM_M, SMEM_N) tiles allocated to CTAs within the cluster.
    struct ClusterWorkItem
    {
        uint32_t m_offset, n_offset, k_offset;
        uint32_t k_work_size;  // Must be divisible by SMEM_K
    };

    static __host__ DEVICE_INLINE uint32_t m_cluster_tiles(uint32_t size_m)
    {
        return (size_m + CLUSTER_M - 1) / CLUSTER_M;
    }

    static __host__ DEVICE_INLINE uint32_t n_cluster_tiles(uint32_t size_n)
    {
        return (size_n + CLUSTER_N - 1) / CLUSTER_N;
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

    static __host__ DEVICE_INLINE uint32_t num_clusterMN_tiles(uint32_t size_m, uint32_t size_n)
    {
        return m_cluster_tiles(size_m) * n_cluster_tiles(size_n);
    }

    // Linearized indexing of cluster MN tiles
    DEVICE_INLINE ClusterMN_Tile get_clusterMN_tile(uint32_t tile_idx) const
    {
        DEVICE_ASSERT(tile_idx < num_clusterMN_tiles(size_m, size_n));

        const uint32_t cluster_rows = m_cluster_tiles(size_m);
        const uint32_t cluster_cols = n_cluster_tiles(size_n);
        const uint32_t cluster_col_remainder = cluster_cols % CLUSTER_MODULUS;
        const uint32_t superblock_count = cluster_cols / CLUSTER_MODULUS;
        const uint32_t superblock_cluster_count = cluster_rows * CLUSTER_MODULUS;
        const uint32_t superblock_idx = tile_idx/ superblock_cluster_count;
        const uint32_t cluster_idx_in_superblock = tile_idx % superblock_cluster_count;

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

        return {cluster_m_idx * CLUSTER_M, cluster_n_idx * CLUSTER_N};
    }

    static __host__ DEVICE_INLINE bool need_zero_init_c([[maybe_unused]] uint32_t size_k)
    {
        switch (K_MODE) {
          case gemm_sm90_k_mode::output_stationary:
            return false;
          case gemm_sm90_k_mode::split_k_inner:
          case gemm_sm90_k_mode::split_k_outer:
            return size_k >= split_k_divisor;
          case gemm_sm90_k_mode::stream_k_early_tma:
          case gemm_sm90_k_mode::stream_k_late_tma:
            return true;
        }
        DEVICE_ASSERT(0);
        return true;
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
        mbarrier_t cluster_mbar;
    };

    // Helper for ring buffering and for waiting on an mbarrier while tracking the phase.
    struct RingState
    {
        // Phase bit tracking
        unsigned tile_fill_bits : RING_BUFFER_SIZE = 0;
        unsigned tile_read_bits : RING_BUFFER_SIZE = 0;
        unsigned per_consumer_wg_bit : 1 = 0;

        unsigned consumer_ring_idx : 5 = 0;
        unsigned producer_ring_idx : 5 = 0;
        unsigned producer_skip_tile_read_mbar : 1 = 1;
        static_assert(RING_BUFFER_SIZE < 32);

        DEVICE_INLINE void tile_fill_wait_ring(Shared& shared)
        {
            const uint32_t i = consumer_ring_idx;
            DEVICE_ASSERT(i < RING_BUFFER_SIZE);
            mbar_wait(shared.tile_fill_mbar[i], (tile_fill_bits >> i) & 1u);
            tile_fill_bits ^= 1u << i;
        }

        DEVICE_INLINE void tile_read_wait_ring(Shared& shared)
        {
            const uint32_t i = producer_ring_idx;
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

        static DEVICE_INLINE void mbar_wait(mbarrier_t& mbar, uint32_t parity)
        {
            asm volatile(
                    "{.reg.pred P1; BEFORE_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni WAIT_DONE; bra.uni BEFORE_WAIT; WAIT_DONE: }"
            :
            : "r"(smem_ptr_u32(&mbar)), "r"(parity));
        }

        DEVICE_INLINE void advance_consumer_ring_idx()
        {
            const unsigned tmp = consumer_ring_idx;
            consumer_ring_idx = tmp == RING_BUFFER_SIZE - 1u ? 0u : tmp + 1u;
        }

        DEVICE_INLINE void advance_producer_ring_idx()
        {
            const unsigned tmp = producer_ring_idx;
            if (tmp == RING_BUFFER_SIZE - 1) {
                producer_skip_tile_read_mbar = 0;
                producer_ring_idx = 0u;
            }
            else {
                producer_ring_idx = tmp + 1u;
            }
        }
    };


    static constexpr uint32_t MMA_M = 64;

    struct WGMMA_D_m64n64
    {
        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
    };

    struct WGMMA_D_m64n96
    {
        // wgmma register tile
        float d0, d1, d2, d3, d4, d5, d6, d7;
        float d8, d9, d10, d11, d12, d13, d14, d15;
        float d16, d17, d18, d19, d20, d21, d22, d23;
        float d24, d25, d26, d27, d28, d29, d30, d31;
        float d32, d33, d34, d35, d36, d37, d38, d39;
        float d40, d41, d42, d43, d44, d45, d46, d47;
    };

    struct WGMMA_D_m64n128
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

    using WGMMA_D = std::conditional_t<WG_N == 64, WGMMA_D_m64n64, std::conditional_t<WG_N == 96, WGMMA_D_m64n96, WGMMA_D_m64n128>>;

    // Per-warpgroup accumulator, holding one (WG_M, WG_N) tile.
    struct WG_Accum
    {
        static_assert(WG_N == 64 || WG_N == 96 || WG_N == 128);

        static_assert(WG_M % MMA_M == 0);
        static constexpr uint32_t num_m_tiles = WG_M / MMA_M;
        WGMMA_D m_tiles[num_m_tiles];
    };

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
    //   aT_tile[wg_m_offset : wg_m_offset + WG_M, wg_k_offset : wg_k_offset + WG_K]
    //   bN_tile[wg_n_offset : wg_n_offset + WG_N, wg_k_offset : wg_k_offset + WG_K]
    // and add to the (WG_M, WG_N) tile held in WG_Accum.
    DEVICE_INLINE void wg_accum_tile(WG_Accum& accum, const Buffers& buffers,
                                     uint32_t wg_m_offset, uint32_t wg_n_offset, uint32_t wg_k_offset,
                                     bool zero_output) const
    {
        static_assert(input_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE, "need to set k stride");
        auto desc_b = matrix_descriptor_mn_k_stride(buffers.bN_nk_core_matrices(wg_n_offset, wg_k_offset),
                                                    SMEM_MN_INNER * SMEM_K_INNER * sizeof(float), 0);
        static_assert(MMA_M == 64);
        static_assert(WG_K == 8);

        #pragma unroll WG_Accum::num_m_tiles
        for (uint32_t i = 0; i < WG_Accum::num_m_tiles; ++i) {
            auto desc_a = matrix_descriptor_mn_k_stride(buffers.aT_mk_core_matrices(wg_m_offset + i * MMA_M,
                                                                                    wg_k_offset),
                                                        SMEM_MN_INNER * SMEM_K_INNER * sizeof(float), 0);
            WGMMA_D& d = accum.m_tiles[i];
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
    }

    // Stored in column major
    template <bool BoundsCheck>
    DEVICE_INLINE void wg_accum_store_to_tile(float* tile, uint32_t c_stride, uint32_t r_guard, uint32_t c_guard,
                                              const WG_Accum& accum) const
    {
        #pragma unroll WG_Accum::num_m_tiles
        for (uint32_t i = 0; i < WG_Accum::num_m_tiles; ++i) {
            const WGMMA_D& d = accum.m_tiles[i];
            const uint32_t tid = threadIdx.x % 128u;
            const uint32_t r_base = (tid / 32u) * 16u + (tid % 32u) / 4u + i * MMA_M;
            const uint32_t c_base = (tid % 4u) * 2u;

            #define X(REG_INDEX) { \
                const uint32_t r = r_base + ((REG_INDEX % 4u) / 2u) * 8u; \
                const uint32_t c = c_base + (REG_INDEX / 4u) * 8 + (REG_INDEX % 2u); \
                if (!BoundsCheck || (r < r_guard && c < c_guard)) tile[c * c_stride + r] = d.d##REG_INDEX; }

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
    }

    // Warpgroup-convergent code
    // Write the (WG_M, WG_N) tile to shared.aliased_per_wg_c_tile, at the entry reserved for this warpgroup.
    DEVICE_INLINE void wg_accum_to_shared(Shared& shared, const WG_Accum& accum) const
    {
        wg_accum_store_to_tile<false>(shared.aliased_per_wg_c_tile[consumer_wg_index()], WG_M, 0, 0, accum);
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

    // Fill shared memory aT tile with SMEM_K×SMEM_M block starting at (cta_k_offset, cta_m_offset)
    // Fill shared memory bN tile with SMEM_K×SMEM_N block starting at (cta_k_offset, cta_n_offset)
    //
    // Due to the tiling of shared memory, this is done as SMEM_K / SMEM_K_INNER separate copies.
    //
    // Furthermore, if the cluster size is not 1, we assume all blocks in the cluster call this function, and we
    // distribute+multicast the copies. We signal the mbar of all CTAs of the cluster that received
    // the same aT/bN tile.
    DEVICE_INLINE void warp_async_load_block_clustered(
            Buffers& buffers, mbarrier_t& mbar,
            uint32_t cta_m_offset, uint32_t cta_n_offset, uint32_t cta_k_offset) const
    {
        if (elect_one_sync()) {
            constexpr uint32_t Tsz = sizeof(float);
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                         "r"(smem_ptr_u32(&mbar)), "n"((SMEM_M + SMEM_N) * SMEM_K * Tsz));

            for (uint32_t smem_k_offset = 0; smem_k_offset < SMEM_K; smem_k_offset += SMEM_K_INNER) {
                static_assert(tensorMap_aT_box_m == SMEM_M_OUTER * SMEM_MN_INNER / Buffers::aT_tile_cta_count);
                static_assert(tensorMap_bN_box_n == SMEM_N_OUTER * SMEM_MN_INNER / Buffers::bN_tile_cta_count);
                static_assert(tensorMap_box_k == SMEM_K_INNER);

                tma_maybe_multicast<Buffers::aT_tile_cta_count>(
                    buffers.aT_tma_box(smem_k_offset), tensorMap_aT, mbar,
                    cta_k_offset + smem_k_offset, cta_m_offset + Buffers::cta_smem_m_offset(), cta_shared_m_mask());

                tma_maybe_multicast<Buffers::bN_tile_cta_count>(
                    buffers.bN_tma_box(smem_k_offset), tensorMap_bN, mbar,
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
            init_mbar<CLUSTER_NUM_CTA * cta_size()>(shared.cluster_mbar);
        }
        asm volatile("fence.proxy.async;");

        if (canonical_warp_idx_sync() == 0 && elect_one_sync()) {
            prefetch_tensormap(tensorMap_aT);
            prefetch_tensormap(tensorMap_bN);
            prefetch_tensormap(tensorMap_cN);
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

    // Cluster cooperates to fill or partially reduce (determined by TMA_MODE) to an MN tile
    // of the output matrix block, with the work specified by cluster_item.
    // Requires smem-allocated ring buffer.
    //
    // Each work item consists of k_tiles iterations of accumulating on the k-axis.
    template <bool ENABLE_PRODUCER_BRANCH, bool IS_CONSUMER, TmaMode TMA_MODE>
    DEVICE_INLINE void cluster_process_item(ClusterWorkItem cluster_item, RingState& state, Shared& shared) const
    {
        DEVICE_ASSERT(ENABLE_PRODUCER_BRANCH || !is_producer_wg());
        DEVICE_ASSERT(IS_CONSUMER || is_producer_wg());

        const uint32_t cta_m_offset = cluster_item.m_offset + cta_m_idx_cluster() * SMEM_M;
        const uint32_t cta_n_offset = cluster_item.n_offset + cta_n_idx_cluster() * SMEM_N;
        DEVICE_ASSERT(cta_m_offset % SMEM_M == 0);
        DEVICE_ASSERT(cta_n_offset % SMEM_N == 0);

        constexpr uint32_t k_pad_iter = DEDICATED_PRODUCER_WG ? 0u : RING_BUFFER_SIZE - 1u;
        const uint32_t k_tiles = cluster_item.k_work_size / SMEM_K;
        const uint32_t k_num_iters = k_tiles + k_pad_iter;
        DEVICE_ASSERT(cluster_item.k_work_size % SMEM_K == 0);

        auto producer_on_k_iter = [&] (uint32_t k_iter)
        {
            if constexpr (ENABLE_PRODUCER_BRANCH) {
                if (is_producer_wg() && k_iter < k_tiles) {
                    const uint32_t tma_k_offset = cluster_item.k_offset + SMEM_K * k_iter;
                    if (!state.producer_skip_tile_read_mbar) {
                        // Special exception to synchronization: on the first time cluster_process_item is called
                        // after CTA startup, for the first RING_BUFFER_SIZE iterations, we don't wait for the
                        // tiles to be read since there is no prior work.
                        state.tile_read_wait_ring(shared);
                    }
                    if (threadIdx.x % 128u < 32u) {
                        warp_async_load_block_clustered(
                                shared.aliased_input_ring_buffer[state.producer_ring_idx],
                                shared.tile_fill_mbar[state.producer_ring_idx],
                                cta_m_offset, cta_n_offset, tma_k_offset);
                    }
                    state.advance_producer_ring_idx();
                }
            }
        };

        // We process the work item in `k_num_iter = k_tiles + k_pad_iter` iterations.
        // Producer skips the last k_pad_iter iterations; consumer skips the first k_pad_iter iterations.

        if constexpr (k_pad_iter != 0) {
            for (uint32_t k_iter = 0; k_iter < k_pad_iter; ++k_iter) {
                producer_on_k_iter(k_iter);
            }
        }

        std::conditional_t<IS_CONSUMER, WG_Accum, char> accum;
        bool zero_accum = true;

        for (uint32_t k_iter = k_pad_iter; k_iter < k_num_iters; ++k_iter) {
            if constexpr (IS_CONSUMER) {
                state.tile_fill_wait_ring(shared);

                asm volatile("wgmma.fence.sync.aligned;  // GMMA");
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;

                for (uint32_t wg_k_idx = 0; wg_k_idx < SMEM_K / WG_K; ++wg_k_idx) {
                    const uint32_t wg_k_offset = wg_k_idx * WG_K;
                    wg_accum_tile(accum, shared.aliased_input_ring_buffer[state.consumer_ring_idx],
                                  wg_m_offset, wg_n_offset, wg_k_offset, zero_accum);
                    zero_accum = false;
                }
                asm volatile("wgmma.commit_group.sync.aligned;  // GMMA");

                if (k_iter == k_pad_iter) {
                    asm("// first wgmma iteration");
                }
                else {
                    asm("// subsequent wgmma iteration");
                }

                // Wait for previous iteration's wgmma to retire, then signal that the tiles read from
                // the previous iteration may be overwritten by the full cluster.
                if (k_iter >= k_pad_iter + 1) {
                    static_assert(RING_BUFFER_SIZE >= 2);
                    asm volatile("wgmma.wait_group.sync.aligned 1;  // GMMA");
                    const unsigned prev_ring_idx = state.consumer_ring_idx == 0
                                                     ? RING_BUFFER_SIZE - 1 : state.consumer_ring_idx - 1u;
                    mbar_arrive_cluster_broadcast(shared.tile_read_mbar[prev_ring_idx]);
                }
                // On last iteration, wait for all wgmma to retire, and signal again.
                if (k_iter == k_num_iters - 1u) {
                    asm volatile("wgmma.wait_group.sync.aligned 0;  // GMMA");
                    mbar_arrive_cluster_broadcast(shared.tile_read_mbar[state.consumer_ring_idx]);
                }
                state.advance_consumer_ring_idx();
            }

            producer_on_k_iter(k_iter);
        }

        // Conditions written so as to allow static removal of TMA code for TmaMode::never_reduce.
        const bool is_partial_k = cluster_item.k_work_size < size_k;  // Not != due to rounding up to SMEM_K
        const bool use_tma = TMA_MODE != TmaMode::never_reduce && (TMA_MODE != TmaMode::if_reduce || is_partial_k);

        if (use_tma) {
            // If we are using TMA to write the output tiles, we need to do a full sync of the cluster before and after
            // using TMA, since we need to re-use shared memory to stage the tile.
            // The sync should have the needed effect as at this point all outstanding producer TMA commands have been
            // waited for by consumers (and no new ones will be issued until after the second sync).
            // NOTE, this defeats the pipelining across cluster work items.

            mbar_arrive_cluster_broadcast(shared.cluster_mbar);
            RingState::mbar_wait(shared.cluster_mbar, 0);

            if constexpr (IS_CONSUMER) {
                // Now copy wgmma registers to shared memory C tile.
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
                wg_accum_to_shared(shared, accum);

                // 0th thread per consumer warpgroup waits for its own warpgroup and
                // issues TMA copy from shared tile to global.
                static_assert(WG_M == tensorMap_cN_box_m);
                static_assert(WG_N == tensorMap_cN_box_n);

                const bool elected = threadIdx.x % 128u < 32 && elect_one_sync();
                asm volatile("fence.proxy.async;");
                mbar_arrive_cta(shared.per_consumer_wg_mbar[consumer_wg_index()]);
                state.per_consumer_wg_wait(elected, shared);
                if (elected) {
                    if (TMA_MODE != TmaMode::always_cond_reduce || is_partial_k) {
                        DEVICE_ASSERT(need_zero_init_c(size_k));
                        asm volatile(
                            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
                            " [%0, {%1, %2}], [%3];"
                            :
                            : "l"(tensorMap_cN),
                              "r"(cta_m_offset + wg_m_offset), "r"(cta_n_offset + wg_n_offset),
                              "r"(smem_ptr_u32(&shared.aliased_per_wg_c_tile[consumer_wg_index()])));
                    }
                    else {
                        DEVICE_ASSERT(TMA_MODE == TmaMode::always_cond_reduce);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                            " [%0, {%1, %2}], [%3];"
                            :
                            : "l"(tensorMap_cN),
                              "r"(cta_m_offset + wg_m_offset), "r"(cta_n_offset + wg_n_offset),
                              "r"(smem_ptr_u32(&shared.aliased_per_wg_c_tile[consumer_wg_index()])));
                    }

                    // We must wait for the TMA to complete before the cta_mbar, otherwise the
                    // shared memory might get stomped on before the TMA finishes reading from it.
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }

            mbar_arrive_cluster_broadcast(shared.cluster_mbar);
            RingState::mbar_wait(shared.cluster_mbar, 1);
        }
        else {
            // Store tile directly to output memory, bypassing TMA and shared memory.
            // Clearly, this requires that the full k-column was accumulated locally.
            DEVICE_ASSERT(cluster_item.k_offset == 0 && cluster_item.k_work_size >= size_k);
            if constexpr (IS_CONSUMER) {
                const uint32_t wg_m_offset = get_wg_m_idx() * WG_M;
                const uint32_t wg_n_offset = get_wg_n_idx() * WG_N;
                const uint32_t arg_m_offset = cta_m_offset + wg_m_offset;
                const uint32_t arg_n_offset = cta_n_offset + wg_n_offset;
                if (arg_m_offset < size_m && arg_n_offset < size_n) {
                    // We need to bounds check as TMA is not being used.
                    wg_accum_store_to_tile<true>(cN + arg_n_offset * size_m + arg_m_offset,
                                                 size_m, size_m - arg_m_offset, size_n - arg_n_offset, accum);
                }
            }
        }
    }

    template <bool ENABLE_PRODUCER_BRANCH, bool IS_CONSUMER>
    DEVICE_INLINE void cluster_main_loop(RingState& state, Shared& shared) const
    {
        DEVICE_ASSERT(gridDim.x % CLUSTER_NUM_CTA == 0);
        DEVICE_ASSERT(gridDim.y == 1);
        DEVICE_ASSERT(gridDim.z == 1);

        const uint32_t num_clusters = gridDim.x / CLUSTER_NUM_CTA;
        const uint32_t cluster_index = blockIdx.x / CLUSTER_NUM_CTA;
        const uint32_t size_k_tiles = (size_k + SMEM_K - 1) / SMEM_K;  // Count of k tiles along k-axis

        if constexpr (K_MODE == gemm_sm90_k_mode::stream_k_early_tma || K_MODE == gemm_sm90_k_mode::stream_k_late_tma) {
            constexpr bool early_tma = K_MODE == gemm_sm90_k_mode::stream_k_early_tma;
            constexpr bool late_tma = K_MODE == gemm_sm90_k_mode::stream_k_late_tma;
            static_assert(early_tma != late_tma);

            // https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
            // Following the above advice to assign full k-columns of MN tiles to clusters if possible,
            // and only using stream-K for the tail of up to (2 * num_clusters - 1) MN tiles
            const uint32_t num_mn_tiles = num_clusterMN_tiles(size_m, size_n);
            const uint32_t tail_mn_tiles = umin(num_mn_tiles, num_mn_tiles % num_clusters + num_clusters);
            DEVICE_ASSERT(tail_mn_tiles < 2 * num_clusters);
            const uint32_t full_mn_tiles = num_mn_tiles - tail_mn_tiles;

            if constexpr (early_tma) {
                // Output stationary over MN tiles; 1 tile per cluster.
                // Don't handle the tail MN tiles.
                for (uint32_t tile_index = cluster_index; tile_index < full_mn_tiles; tile_index += num_clusters) {
                    ClusterMN_Tile tile = get_clusterMN_tile(tile_index);
                    ClusterWorkItem work{};
                    work.m_offset = tile.m_offset;
                    work.n_offset = tile.n_offset;
                    work.k_offset = 0;
                    work.k_work_size = size_k_tiles * SMEM_K;
                    cluster_process_item<ENABLE_PRODUCER_BRANCH, IS_CONSUMER, TmaMode::never_reduce>(work, state,
                                                                                                     shared);
                }
            }

            // MN tiles are subdivided on the K dimension into MNK tiles.
            // Evenly distribute MNK tiles of the tail MN tiles to clusters.
            // First `mnk_mod_clusters` clusters will get `base_mnk_per_cluster + 1` tiles to work on
            // and the remainder will get only `base_mnk_per_cluster` tiles.
            const uint32_t tail_mnk_tiles = size_k_tiles * tail_mn_tiles;
            const uint32_t base_mnk_per_cluster = tail_mnk_tiles / num_clusters;
            const uint32_t mnk_mod_clusters = tail_mnk_tiles % num_clusters;
            auto tail_mnk_index_for_cluster =
                [full_mn_tiles, size_k_tiles, base_mnk_per_cluster, mnk_mod_clusters] (uint32_t _cluster)
            {
                return full_mn_tiles * size_k_tiles         // Skip non-tail MN tiles
                       + base_mnk_per_cluster * _cluster    // Evenly divide tail MNK tiles to clusters
                       + umin(_cluster, mnk_mod_clusters);  // Assign 1 extra tile for first mnk_mod_clusters clusters.
            };
            // This cluster is assigned tail MNK tiles in range [tail_mnk_start, tail_mnk_end)
            const uint32_t tail_mnk_start = tail_mnk_index_for_cluster(cluster_index);
            const uint32_t tail_mnk_end = tail_mnk_index_for_cluster(cluster_index + 1);
            uint32_t mnk_index;
            if (early_tma || full_mn_tiles == 0) {
                // In early TMA mode, we only have to handle the tail (due to the output stationary loop above).
                // This is also the case for late TMA if the tail comprises the entire workload.
                mnk_index = tail_mnk_start;
            }
            else {
                // In late TMA mode, we start with all work assigned.
                // Start with cluster_index-th MN tile, unless the tail comprises the entire workload [above].
                mnk_index = cluster_index * size_k_tiles;
            }

            // Handle MNK tiles
            while (mnk_index < tail_mnk_end) {
                // Figure out the MN-tile (output matrix tile) corresponding to this MNK tile
                // and accumulate on the K-axis in the range SMEM_K * [k_tile_offset, k_tile_offset + work_k_tiles).
                const uint32_t mn_index = mnk_index / size_k_tiles;
                const uint32_t k_tile_offset = mnk_index % size_k_tiles;
                // We accumulate either to the end of the K-axis, or to the boundary between different clusters' work.
                const uint32_t work_k_tiles = umin(size_k_tiles - k_tile_offset, tail_mnk_end - mnk_index);

                ClusterWorkItem work{};
                ClusterMN_Tile tile = get_clusterMN_tile(mn_index);
                work.m_offset = tile.m_offset;
                work.n_offset = tile.n_offset;
                work.k_offset = k_tile_offset * SMEM_K;
                work.k_work_size = work_k_tiles * SMEM_K;
                constexpr auto TMA_MODE = late_tma ? TmaMode::if_reduce : TmaMode::always_force_reduce;
                cluster_process_item<ENABLE_PRODUCER_BRANCH, IS_CONSUMER, TMA_MODE>(work, state, shared);

                if (late_tma && mn_index < full_mn_tiles) {
                    // Seek to the next full size MN tile (we are not handling the tail yet)
                    mnk_index += size_k_tiles * num_clusters;
                    DEVICE_ASSERT(mnk_index % size_k_tiles == 0);

                    if (mnk_index >= full_mn_tiles * size_k_tiles) {
                        // Transition to stream-k tail
                        mnk_index = tail_mnk_start;
                    }
                }
                else {
                    // Stream-K tail; proceed past work just done and to immediate next MN tile.
                    mnk_index += work_k_tiles;
                }
            }

        }
        else if constexpr (K_MODE == gemm_sm90_k_mode::split_k_inner || K_MODE == gemm_sm90_k_mode::split_k_outer) {
            constexpr bool k_inner = K_MODE == gemm_sm90_k_mode::split_k_inner;
            const uint32_t num_mn_tiles = num_clusterMN_tiles(size_m, size_n);
            const uint32_t num_k_work_items = (size_k + split_k_divisor - 1) / split_k_divisor;
            const uint32_t num_mnk_work_items = num_mn_tiles * num_k_work_items;

            for (uint32_t work_index = cluster_index; work_index < num_mnk_work_items; work_index += num_clusters) {
                const uint32_t mn_tile_index = k_inner ? work_index / num_k_work_items : work_index % num_mn_tiles;
                const uint32_t k_work_index = k_inner ? work_index % num_k_work_items : work_index / num_mn_tiles;

                ClusterMN_Tile tile = get_clusterMN_tile(mn_tile_index);
                ClusterWorkItem work{};
                work.m_offset = tile.m_offset;
                work.n_offset = tile.n_offset;
                work.k_offset = split_k_divisor * k_work_index;
                const uint32_t work_k_tiles = umin(split_k_divisor / SMEM_K, size_k_tiles - work.k_offset);
                work.k_work_size = work_k_tiles * SMEM_K;
                cluster_process_item<ENABLE_PRODUCER_BRANCH, IS_CONSUMER, TmaMode::if_reduce>(work, state, shared);
            }
        }
        else {
            static_assert(K_MODE == gemm_sm90_k_mode::output_stationary);

            // Output stationary over MN tiles; 1 tile per cluster.
            const uint32_t num_mn_tiles = num_clusterMN_tiles(size_m, size_n);
            for (uint32_t tile_index = cluster_index; tile_index < num_mn_tiles; tile_index += num_clusters) {
                ClusterMN_Tile tile = get_clusterMN_tile(tile_index);
                ClusterWorkItem work{};
                work.m_offset = tile.m_offset;
                work.n_offset = tile.n_offset;
                work.k_offset = 0;
                work.k_work_size = size_k_tiles * SMEM_K;
                cluster_process_item<ENABLE_PRODUCER_BRANCH, IS_CONSUMER, TmaMode::never_reduce>(work, state, shared);
            }
        }

        if constexpr (CLUSTER_NUM_CTA != 1) {
            // Required for distributed shared memory safety, I think?
            mbar_arrive_cluster_broadcast(shared.cluster_mbar);
            RingState::mbar_wait(shared.cluster_mbar, 0);
        }
    }

    DEVICE_INLINE void kernel_main()
    {
        DEVICE_ASSERT(blockDim.x == cta_size());

        extern __shared__ char raw_smem[];
        Shared& smem = *reinterpret_cast<Shared*>(raw_smem);
        RingState state{};

        cta_first_time_init(smem);
        cluster_sync();

        if constexpr (!DEDICATED_PRODUCER_WG) {
            cluster_main_loop<true, true>(state, smem);
        }
        else if (is_producer_wg()) {
            if constexpr (cta_size() == 384) {
                asm("setmaxnreg.dec.sync.aligned.u32 40;");
            }
            cluster_main_loop<true, false>(state, smem);
        }
        else {
            if constexpr (cta_size() == 384) {
                asm("setmaxnreg.inc.sync.aligned.u32 232;");
            }
            cluster_main_loop<false, true>(state, smem);
        }
    }

    static void init_tensorMap(CUtensorMap* tensorMap, const float* globalAddress,
                               uint32_t gmem_outer, uint32_t gmem_inner,
                               uint32_t smem_outer, uint32_t smem_inner, CUtensorMapSwizzle swizzle)
    {
        const CUtensorMapDataType tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        const uint32_t tensorRank = 2;
        const cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
        const cuuint64_t globalStrides[1] = {4*gmem_inner};
        const cuuint32_t boxDim[2] = {smem_inner, smem_outer};
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
            fprintf(stderr, "cuTensorMapEncodeTiled: %i {%u, %u}\n", (int)result, smem_inner, smem_outer);
            assert(0);
        }
    }

    static void launch(cudaStream_t stream, uint32_t size_m, uint32_t size_n, uint32_t size_k,
                       const float* aT, const float* bN, float* cN)
    {
        CUtensorMap tensorMap_aT, tensorMap_bN, tensorMap_cN;
        init_tensorMap(&tensorMap_aT, aT, size_m, size_k, tensorMap_aT_box_m, tensorMap_box_k, input_swizzle);
        init_tensorMap(&tensorMap_bN, bN, size_n, size_k, tensorMap_bN_box_n, tensorMap_box_k, input_swizzle);
        init_tensorMap(&tensorMap_cN, cN, size_n, size_m, tensorMap_cN_box_n, tensorMap_cN_box_m, CU_TENSOR_MAP_SWIZZLE_NONE);

        if (need_zero_init_c(size_k)) {
            cudaMemsetAsync(cN, 0, size_m * size_n * sizeof(*cN), stream);
        }

        const uint32_t max_cluster = MAX_CTA / CLUSTER_NUM_CTA;
        const uint32_t grid = max_cluster * CLUSTER_NUM_CTA;
        const uint32_t block = cta_size();
        const uint32_t smem = smem_size();

        const auto kernel = tiled_multiplier_kernel<TiledMultiplier>;
        static std::once_flag f;
        std::call_once(f, [&] () {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (1) {
                const char* k_mode_str = "?";
                switch (K_MODE) {
                    case gemm_sm90_k_mode::output_stationary: k_mode_str = "output_stationary"; break;
                    case gemm_sm90_k_mode::split_k_outer: k_mode_str = "split_k_outer"; break;
                    case gemm_sm90_k_mode::split_k_inner: k_mode_str = "split_k_inner"; break;
                    case gemm_sm90_k_mode::stream_k_early_tma: k_mode_str = "stream_k_early_tma"; break;
                    case gemm_sm90_k_mode::stream_k_late_tma: k_mode_str = "stream_k_late_tma"; break;
                }
                fprintf(stderr, "K_MODE:  %i (%s)\n", static_cast<int>(K_MODE), k_mode_str);
                fprintf(stderr, "GRID:    %u\n", grid);
                fprintf(stderr, "BLOCK:   %u\n", block);
                fprintf(stderr, "SMEM:    %g KiB\n", double(smem) / 1024.0);
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, kernel);
                fprintf(stderr, "numRegs: %i\n", attr.numRegs);
                fprintf(stderr, "clusterDim: %u\n", CLUSTER_NUM_CTA);
                fprintf(stderr, "\n");
            }
        });

        TiledMultiplierArgs args{size_m, size_n, size_k, tensorMap_aT, tensorMap_bN, tensorMap_cN, cN};

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

void matmul_impl(GPU_Tensors t, gemm_sm90_k_mode k_mode, cudaStream_t stream)
{
    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(t.c_col_major);

    assert(uintptr_t(t.a) % 16 == 0);
    assert(uintptr_t(t.b) % 16 == 0);
    assert(uintptr_t(t.c) % 16 == 0);
    assert(t.K % 4 == 0);

    constexpr uint32_t cluster_m = 256;
    // constexpr uint32_t cluster_n = 256;
    constexpr uint32_t cluster_n = 128;
    constexpr uint32_t smem_m = 256;
    constexpr uint32_t smem_n = 128;
    constexpr uint32_t smem_k = 32;
    constexpr uint32_t wg_m = 128;
    constexpr uint32_t wg_n = 128;
    constexpr uint32_t wg_k = 8;
    constexpr uint32_t cluster_modulus = 1024u / cluster_m;
    constexpr uint32_t ring_buffer_size = 4;
    constexpr bool dedicated_producer = true;

    switch (k_mode) {
      case gemm_sm90_k_mode::output_stationary:
      {
        using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                           gemm_sm90_k_mode::output_stationary,
                                           cluster_modulus, ring_buffer_size, dedicated_producer>;
        Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
        return;
      }
      default:
      {
        fprintf(stderr, "non-output-stationary disabled\n");
        assert(0);
      }
#if 0
      case gemm_sm90_k_mode::split_k_outer:
      {
        using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                           gemm_sm90_k_mode::split_k_outer,
                                           cluster_modulus, ring_buffer_size, dedicated_producer>;
        Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
        return;
      }
      case gemm_sm90_k_mode::split_k_inner:
      {
        using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                           gemm_sm90_k_mode::split_k_inner,
                                           cluster_modulus, ring_buffer_size, dedicated_producer>;
        Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
        return;
      }
      case gemm_sm90_k_mode::stream_k_early_tma:
      {
        // Due to annoying compiler warning
        fprintf(stderr, "gemm_sm90_k_mode::stream_k_early_tma disabled\n");
        assert(0);
#if 0
        using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                           gemm_sm90_k_mode::stream_k_early_tma,
                                           cluster_modulus, ring_buffer_size, dedicated_producer>;
        Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
#endif
        return;
      }
      case gemm_sm90_k_mode::stream_k_late_tma:
      {
        using Multiplier = TiledMultiplier<cluster_m, cluster_n, smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                                           gemm_sm90_k_mode::stream_k_late_tma,
                                           cluster_modulus, ring_buffer_size, dedicated_producer>;
        Multiplier::launch(stream, size_m, size_n, size_k, t.a, t.b, t.c);
        return;
      }
#endif
    }

    assert(0);
}

}  // end namespace

void matmul_sm90(GPU_Tensors t, gemm_sm90_k_mode k_mode, cudaStream_t stream)
{
    if (!t.a_col_major && t.b_col_major && t.c_col_major) {
        gemm_sm90::matmul_impl(t, k_mode, stream);
    }
    else if (!t.a_col_major && t.b_col_major && !t.c_col_major) {
        std::swap(t.a, t.b);
        std::swap(t.M, t.N);
        t.a_col_major = false;
        t.b_col_major = true;
        t.c_col_major = true;
        gemm_sm90::matmul_impl(t, k_mode, stream);
    }
    else {
        assert(0);
    }
}
