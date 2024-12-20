#include "gemm_sm90.h"

#include <cassert>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#include <cuda.h>

#define DEVICE_INLINE __device__ __forceinline__

// #define DEVICE_ASSERT(x) assert(x)
#define DEVICE_ASSERT(x)

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

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size())
tiled_multiplier_kernel(uint32_t size_m, uint32_t size_n, uint32_t size_k,
                        __grid_constant__ const CUtensorMap tensorMap_a,
                        __grid_constant__ const CUtensorMap tensorMap_bT,
                        __grid_constant__ const CUtensorMap tensorMap_c,
                        float* c)
{
    Multiplier multiplier{size_m, size_n, size_k, &tensorMap_a, &tensorMap_bT, &tensorMap_c, c};
    multiplier.kernel_main();
}

template <uint32_t SMEM_M, uint32_t SMEM_N, uint32_t SMEM_K,
          uint32_t WG_M, uint32_t WG_N, uint32_t WG_K,
          uint32_t K_MAX_TILES, uint32_t CTA_MODULUS, uint32_t RING_BUFFER_SIZE, bool DEDICATED_PRODUCER_WG>
struct TiledMultiplier
{
    uint32_t size_m, size_n, size_k;

    const CUtensorMap* tensorMap_a;
    const CUtensorMap* tensorMap_bT;  // Transposed; column major
    const CUtensorMap* tensorMap_c;
    float* c;

    static_assert(WG_M == 64);
    static_assert(WG_K == 8);
    static constexpr CUtensorMapSwizzle input_swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

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
        // XXX I don't think SMEM_K_OUTER works if SMEM_K_OUTER != 1
        static_assert(SMEM_K_OUTER == 1, "todo investigate");
        float a_tile[SMEM_K_OUTER][SMEM_M_OUTER][SMEM_K_INNER * SMEM_MN_INNER];
        float bT_tile[SMEM_K_OUTER][SMEM_N_OUTER][SMEM_K_INNER * SMEM_MN_INNER];

        static constexpr uint32_t tensorMap_box_m = SMEM_M_OUTER * SMEM_MN_INNER;
        static constexpr uint32_t tensorMap_box_n = SMEM_N_OUTER * SMEM_MN_INNER;
        static constexpr uint32_t tensorMap_box_k = SMEM_K_OUTER * SMEM_K_INNER;

        __device__ float* a_tma_box(uint32_t k_offset)
        {
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &a_tile[k_offset / SMEM_K_INNER][0][0];
        }

        __device__ float* bT_tma_box(uint32_t k_offset)
        {
            DEVICE_ASSERT(k_offset % SMEM_K_INNER == 0);
            return &bT_tile[k_offset / SMEM_K_INNER][0][0];
        }

        __device__ const float* a_mk_core_matrices(uint32_t m_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(m_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &a_tile[k_offset / SMEM_K_INNER][m_offset / SMEM_MN_INNER][k_offset];
            // XXX not sure the last coordinate is correct if SMEM_K_OUTER != 1
        }

        __device__ const float* bT_nk_core_matrices(uint32_t n_offset, uint32_t k_offset) const
        {
            DEVICE_ASSERT(n_offset % SMEM_MN_INNER == 0);
            DEVICE_ASSERT(k_offset % CORE_MATRIX_K == 0);
            return &bT_tile[k_offset / SMEM_K_INNER][n_offset / SMEM_MN_INNER][k_offset];
            // XXX not sure the last coordinate is correct if SMEM_K_OUTER != 1
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

    DEVICE_INLINE uint32_t consumer_wg_index() const
    {
        DEVICE_ASSERT(threadIdx.x < 128u * consumer_wg_count());
        return threadIdx.x / 128u;
    }

    // Static assignment of consumer warpgroups within CTA to per-warpgroup output tiles (WG_M, WG_N) within
    // per-CTA output tile (SMEM_M, SMEM_N); one warpgroup is the producer (load tiles w/ TMA) warpgroup.
    //
    //
    // * if DEDICATED_PRODUCER_WG, one extra warpgroup is the producer
    // * if !DEDICATED_PRODUCER_WG, the 0th consumer is also the producer.
    DEVICE_INLINE bool is_producer_wg() const
    {
        return (threadIdx.x / 128u) == (DEDICATED_PRODUCER_WG ? consumer_wg_count() : 0u);
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

    __host__ DEVICE_INLINE static constexpr uint32_t cta_size()
    {
        // Optional 1 extra warpgroup for producer.
        return (DEDICATED_PRODUCER_WG + consumer_wg_count()) * 128;
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

    static DEVICE_INLINE void mbar_arrive(mbarrier_t& mbar)
    {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_ptr_u32(&mbar)));
    }

    static DEVICE_INLINE void mbar_wait(mbarrier_t& mbar, uint32_t parity)
    {
        asm volatile(
                "{.reg.pred P1; BEFORE_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni WAIT_DONE; bra.uni BEFORE_WAIT; WAIT_DONE: }"
        :
        : "r"(smem_ptr_u32(&mbar)), "r"(parity));
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
        mbarrier_t all_consumers_mbar;
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
        uint64_t enc = (val & 0x3FFFF) >> 4;
        DEVICE_ASSERT(val == enc << 4);
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

    DEVICE_INLINE void wg_accum_store_to_tile(float* tile, uint32_t row_stride, const WG_Accum& d) const
    {
        const uint32_t tid = threadIdx.x % 128u;

        #define X(REG_INDEX) { \
            const uint32_t r = (tid / 32u) * 16u + (tid % 32u) / 4u + ((REG_INDEX % 4u) / 2u) * 8u; \
            const uint32_t c = (tid % 4u) * 2u + (REG_INDEX / 4u) * 8 + (REG_INDEX % 2u); \
            tile[r * row_stride + c] = d.d##REG_INDEX; }
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
        wg_accum_store_to_tile(shared.aliased_per_wg_c_tile[consumer_wg_index()], WG_N, d);
    }

    // Fill shared memory A tile with SMEM_M×SMEM_K block starting at (cta_m_offset, cta_k_offset)
    // Fill shared memory B^T tile with SMEM_N×SMEM_K block starting at (cta_n_offset, cta_k_offset)
    // Due to the tiling of shared memory, this is done as SMEM_K / SMEM_K_INNER separate copies.
    DEVICE_INLINE void warp_async_load_block(Buffers& buffers, mbarrier_t& mbar,
                                             uint32_t cta_m_offset, uint32_t cta_n_offset, uint32_t cta_k_offset) const
    {
        if (elect_one_sync()) {
            constexpr uint32_t expect_count = (SMEM_M + SMEM_N) * SMEM_K * sizeof(float);
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(smem_ptr_u32(&mbar)), "n"(expect_count));

            for (uint32_t smem_k_offset = 0; smem_k_offset < SMEM_K; smem_k_offset += SMEM_K_INNER) {
                static_assert(tensorMap_a_box_m == SMEM_M_OUTER * SMEM_MN_INNER);
                static_assert(tensorMap_bT_box_n == SMEM_N_OUTER * SMEM_MN_INNER);
                static_assert(tensorMap_box_k == SMEM_K_INNER);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(smem_ptr_u32(buffers.a_tma_box(smem_k_offset))),
                      "l"(tensorMap_a),
                      "r"(smem_ptr_u32(&mbar)),
                      "r"(cta_k_offset + smem_k_offset), "r"(cta_m_offset)
                    : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%3, %4}], [%2];"
                    :
                    : "r"(smem_ptr_u32(buffers.bT_tma_box(smem_k_offset))),
                      "l"(tensorMap_bT),
                      "r"(smem_ptr_u32(&mbar)),
                      "r"(cta_k_offset + smem_k_offset), "r"(cta_n_offset)
                    : "memory");
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
        constexpr unsigned consumer_thread_count = consumer_wg_count() * 128u;

        cta_init_mbar_array<1>(shared.tile_fill_mbar);
        cta_init_mbar_array<consumer_thread_count>(shared.tile_read_mbar);
        cta_init_mbar_array<128>(shared.per_consumer_wg_mbar);
        if (threadIdx.x == 0) {
            init_mbar<consumer_thread_count>(shared.all_consumers_mbar);
        }
        asm volatile("fence.proxy.async;");

        if (canonical_warp_idx_sync() == 0 && elect_one_sync()) {
            prefetch_tensormap(tensorMap_a);
            prefetch_tensormap(tensorMap_bT);
            prefetch_tensormap(tensorMap_c);
        }
    }

    // CTA cooperates to fill or add to the output matrix block of size (SMEM_M, SMEM_N) starting at
    // (cta_m_offset, cta_n_offset); we process up to K_MAX_TILES input blocks on the K dimension starting
    // at cta_k_offset.
    // Requires smem-allocated ring buffer.
    template <bool ENABLE_PRODUCER_BRANCH, bool IS_CONSUMER>
    DEVICE_INLINE void cta_main_loop(uint32_t cta_m_offset, uint32_t cta_n_offset, const uint32_t cta_k_initial_offset,
                                     Shared& shared) const
    {
        DEVICE_ASSERT(cta_m_offset % SMEM_M == 0);
        DEVICE_ASSERT(cta_n_offset % SMEM_N == 0);
        DEVICE_ASSERT(cta_k_initial_offset % (SMEM_K * K_MAX_TILES) == 0);
        DEVICE_ASSERT(size_k % SMEM_K == 0);
        DEVICE_ASSERT(ENABLE_PRODUCER_BRANCH || !is_producer_wg());
        DEVICE_ASSERT(IS_CONSUMER || is_producer_wg());

        const uint32_t k_num_iters = min((size_k - cta_k_initial_offset) / SMEM_K, K_MAX_TILES);

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
                        const uint32_t ring_usage_parity = ((k_iter - RING_BUFFER_SIZE) / RING_BUFFER_SIZE) % 2u;
                        mbar_wait(shared.tile_read_mbar[ring_idx], ring_usage_parity);
                    }
                    if (threadIdx.x % 128u < 32u) {
                        warp_async_load_block(shared.aliased_input_ring_buffer[ring_idx],
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
            const uint32_t ring_usage_parity = (k_iter / RING_BUFFER_SIZE) % 2u;

            if constexpr (IS_CONSUMER) {
                mbar_wait(shared.tile_fill_mbar[ring_idx], ring_usage_parity);
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
                // the previous iteration may be overwritten.
                if (k_iter >= 1) {
                    asm volatile("wgmma.wait_group.sync.aligned 1;  // GMMA");
                    mbar_arrive(shared.tile_read_mbar[(k_iter - 1) % RING_BUFFER_SIZE]);
                    static_assert(RING_BUFFER_SIZE >= 2);
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
                // After wgmma finished, issue all-to-all consumer warpgroups sync,
                // indicating that SMEM is ready to be repurposed.
                mbar_arrive(shared.all_consumers_mbar);
                mbar_wait(shared.all_consumers_mbar, 0);

                // Now copy wgmma registers to shared memory C tile.
                wg_accum_to_shared(shared, accum);

                // 0th thread per consumer warpgroup waits for its own warpgroup and
                // issues TMA copy from shared tile to global.
                static_assert(WG_M == tensorMap_c_box_m);
                static_assert(WG_N == tensorMap_c_box_n);
                asm volatile("fence.proxy.async;");
                mbar_arrive(shared.per_consumer_wg_mbar[consumer_wg_index()]);
                if (threadIdx.x % 128u == 0) {
                    mbar_wait(shared.per_consumer_wg_mbar[consumer_wg_index()], 0);
                    asm volatile(
                    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"
                    " [%0, {%1, %2}], [%3];"
                    :
                    : "l"(tensorMap_c),
                      "r"(cta_n_offset + wg_n_offset), "r"(cta_m_offset + wg_m_offset),
                      "r"(smem_ptr_u32(&shared.aliased_per_wg_c_tile[consumer_wg_index()])));
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
            else {
                // Store tile directly to output memory, bypassing TMA and shared memory.
                wg_accum_store_to_tile(c + (cta_m_offset + wg_m_offset) * size_n + (cta_n_offset + wg_n_offset),
                                       size_n, accum);
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

        extern __shared__ char raw_smem[];
        Shared& smem = *reinterpret_cast<Shared*>(raw_smem);
        const auto cta_m_offset = cta_m_idx * SMEM_M, cta_n_offset = cta_n_idx * SMEM_N;

        cta_first_time_init(smem);
        __syncthreads();

        if constexpr (!DEDICATED_PRODUCER_WG) {
            cta_main_loop<true, true>(cta_m_offset, cta_n_offset, cta_k_offset, smem);
        }
        else if (is_producer_wg()) {
            cta_main_loop<true, false>(cta_m_offset, cta_n_offset, cta_k_offset, smem);
        }
        else {
            cta_main_loop<false, true>(cta_m_offset, cta_n_offset, cta_k_offset, smem);
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

        const uint32_t grid = m_cta(size_m) * n_cta(size_n) * k_cta(size_k);
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
        kernel<<<grid, block, smem, stream>>>(size_m, size_n, size_k, tensorMap_a, tensorMap_bT, tensorMap_c, c);
    }
};

}  // end namespace

void matmul_sm90(GPU_Tensors t, cudaStream_t stream)
{
    using namespace gemm_sm90;

    constexpr uint32_t smem_m = 192;
    constexpr uint32_t smem_n = 192;
    constexpr uint32_t smem_k = 32;
    constexpr uint32_t wg_m = 64;
    constexpr uint32_t wg_n = 96;
    constexpr uint32_t wg_k = 8;
    constexpr uint32_t cta_k_max_tiles = 8192u / smem_k;
    constexpr uint32_t cta_modulus = 4;
    constexpr uint32_t ring_buffer_size = 4;
    constexpr bool dedicated_producer = false;

    const uint32_t size_m = t.M;
    const uint32_t size_n = t.N;
    const uint32_t size_k = t.K;

    assert(!t.a_col_major);
    assert(t.b_col_major);
    assert(!t.c_col_major);

    if (size_m % smem_m == 0 && size_n % smem_n == 0 && size_k % smem_k == 0) {
        TiledMultiplier<smem_m, smem_n, smem_k, wg_m, wg_n, wg_k,
                        cta_k_max_tiles, cta_modulus, ring_buffer_size, dedicated_producer>::launch(
                stream, size_m, size_n, size_k, t.a, t.b, t.c);
    }
    else {
        assert(0);
    }
}
