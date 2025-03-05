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

#define DEVICE_INLINE __device__ __forceinline__

namespace gemm_sm80_impl {

using mbarrier_t = long long;

DEVICE_INLINE uint32_t smem_ptr_u32(const void* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

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

__device__ __forceinline__ void
load_a(WmmaA& rmem, const float* gmem, cuda::std::array<unsigned, 2> element_strides)
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

__device__ __forceinline__ void
load_b(WmmaB& rmem, const float* gmem, cuda::std::array<unsigned, 2> element_strides)
{
    const unsigned row_stride = element_strides[0];
    const unsigned col_stride = element_strides[1];
    const unsigned warp_lane = threadIdx.x % 32u;
    const float* gmem_thread_baseaddr = &gmem[warp_lane % 4u * row_stride + warp_lane / 4u * col_stride];
    rmem[0] = __float_as_uint(gmem_thread_baseaddr[0]);
    rmem[1] = __float_as_uint(gmem_thread_baseaddr[4 * row_stride]);
}

__device__ __forceinline__ void
store_d(float* gmem, WmmaD rmem, cuda::std::array<unsigned, 2> element_strides)
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

__device__ __forceinline__ void
wmma(WmmaD& d, WmmaA a, WmmaB b)
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
    uint32_t smem_i, smem_j, smem_k, ring_buffer_size;
    uint32_t warp_i, warp_j;
    uint32_t cta_modulus;
    uint32_t cta_per_sm;
};

template <TileConfig tile_config, bool b_col_major, bool c_col_major>
struct TiledMultiplier
{
    uint32_t size_i, size_j, size_k;
    float const* a;
    float const* b;
    float* c;

    static constexpr uint32_t SMEM_I = tile_config.smem_i;
    static constexpr uint32_t SMEM_J = tile_config.smem_j;
    static constexpr uint32_t SMEM_K = tile_config.smem_k;
    static constexpr uint32_t RING_BUFFER_SIZE = tile_config.ring_buffer_size;
    static constexpr uint32_t WARP_I = tile_config.warp_i;
    static constexpr uint32_t WARP_J = tile_config.warp_j;
    static constexpr uint32_t CTA_MODULUS = tile_config.cta_modulus;

    // One buffer of ring buffer.
    struct alignas(float4) Buffers
    {
        float a[SMEM_I * SMEM_K];  // Stores SMEM_I × SMEM_K of a [row major]
        float b[SMEM_K * SMEM_J];  // Stores SMEM_K × SMEM_J of b [as per b_col_major]
    };

    struct SmemLayout
    {
        Buffers buffers[RING_BUFFER_SIZE];
        mbarrier_t tile_fill_mbar[RING_BUFFER_SIZE];
        mbarrier_t tile_read_mbar[RING_BUFFER_SIZE];
    };

    static_assert(sizeof(SmemLayout) <= 100 << 10);

    // Accumulator tiles per warp
    struct WarpAccum
    {
        static_assert(WARP_I % MMA_I == 0);
        static_assert(WARP_J % MMA_J == 0);
        WmmaD ij[WARP_I / MMA_I][WARP_J / MMA_J];
    };

    // Helper for ring buffering and for waiting on an mbarrier while tracking the phase.
    struct RingState
    {
        // Phase bit tracking
        unsigned tile_fill_bits : RING_BUFFER_SIZE = 0;
        unsigned tile_read_bits : RING_BUFFER_SIZE = 0;

        unsigned consumer_ring_idx : 5 = 0;
        unsigned producer_ring_idx : 5 = 0;
        static_assert(RING_BUFFER_SIZE < 32);

        DEVICE_INLINE void tile_fill_wait_ring(SmemLayout& shared)
        {
            const uint32_t i = consumer_ring_idx;
            assert(i < RING_BUFFER_SIZE);
            mbar_wait(shared.tile_fill_mbar[i], (tile_fill_bits >> i) & 1u);
            tile_fill_bits ^= 1u << i;
        }

        DEVICE_INLINE void tile_read_wait_ring(SmemLayout& shared)
        {
            const uint32_t i = producer_ring_idx;
            assert(i < RING_BUFFER_SIZE);
            mbar_wait(shared.tile_read_mbar[i], (tile_read_bits >> i) & 1u);
            tile_read_bits ^= 1u << i;
        }

        static DEVICE_INLINE void mbar_wait(mbarrier_t& mbar, uint32_t parity)
        {
            asm volatile(
                    "{.reg.pred P1; BEFORE_WAIT: mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1; @P1 bra.uni WAIT_DONE; bra.uni BEFORE_WAIT; WAIT_DONE: }"
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
                producer_ring_idx = 0u;
            }
            else {
                producer_ring_idx = tmp + 1u;
            }
        }
    };

    template <uint32_t ARRIVE_THREADS>
    DEVICE_INLINE void init_mbar(mbarrier_t& mbar) const
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_ptr_u32(&mbar)), "n"(ARRIVE_THREADS));
    }

    template <uint32_t ARRIVE_THREADS, uint32_t COUNT>
    DEVICE_INLINE void cta_init_mbar_array(mbarrier_t (&mbar_array) [COUNT]) const
    {
        assert(COUNT < blockDim.x);
        if (threadIdx.x < COUNT) {
            init_mbar<ARRIVE_THREADS>(mbar_array[threadIdx.x]);
        }
    }

    static DEVICE_INLINE void mbar_arrive_classic(mbarrier_t& mbar)
    {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_ptr_u32(&mbar)));
    }

    static DEVICE_INLINE void mbar_arrive_cp_async(mbarrier_t& mbar)
    {
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(smem_ptr_u32(&mbar)));
    }

    DEVICE_INLINE void cta_first_time_init(SmemLayout& shared) const
    {
        cta_init_mbar_array<cta_size()>(shared.tile_fill_mbar);
        cta_init_mbar_array<cta_size()>(shared.tile_read_mbar);
    }

    __host__ __device__ __forceinline__ static constexpr uint32_t grid_size()
    {
        return 48 * tile_config.cta_per_sm;
    }

    __host__ __device__ __forceinline__ static constexpr uint32_t cta_per_sm()
    {
        return tile_config.cta_per_sm;
    }

    __host__ __device__ __forceinline__ static constexpr uint32_t cta_size()
    {
        // If output matrix is cut into (WARP_I, WARP_J) blocks, one warp handles one matrix block.
        static_assert(SMEM_I % WARP_I == 0);
        static_assert(SMEM_J % WARP_J == 0);
        return (SMEM_I / WARP_I) * (SMEM_J / WARP_J) * 32;
    }

    // If output matrix is cut into (SMEM_I, SMEM_J) blocks, one CTA handles one matrix block.

    __host__ __device__ __forceinline__ uint32_t i_cta() const
    {
        assert(size_i % SMEM_I == 0);
        return size_i / SMEM_I;
    }

    __host__ __device__ __forceinline__ uint32_t j_cta() const
    {
        assert(size_j % SMEM_J == 0);
        return size_j / SMEM_J;
    }

    // Fill shared memory A tile with SMEM_I×SMEM_K block starting at (cta_i_offset, cta_k_offset)
    // Fill shared memory B tile with SMEM_K×SMEM_J block starting at (cta_k_offset, cta_j_offset)
    __device__ __forceinline__ void cta_async_load_block(Buffers& buffers, uint32_t cta_i_offset,
                                                         uint32_t cta_j_offset, uint32_t cta_k_offset) const
    {
        for (unsigned work_idx = threadIdx.x; work_idx < SMEM_I * SMEM_K / 4u; work_idx += blockDim.x) {
            const unsigned thr_i_offset = work_idx / (SMEM_K / 4u);
            const unsigned thr_k_offset = 4 * (work_idx % (SMEM_K / 4u));
            cp_async4(&buffers.a[thr_i_offset*SMEM_K + thr_k_offset],
                      &a[(thr_i_offset + cta_i_offset) * size_k + cta_k_offset + thr_k_offset]);
        }
        for (unsigned work_idx = threadIdx.x; work_idx < SMEM_K * SMEM_J / 4u; work_idx += blockDim.x) {
            if constexpr (b_col_major) {
                const unsigned thr_j_offset = work_idx / (SMEM_K / 4u);
                const unsigned thr_k_offset = 4 * (work_idx % (SMEM_K / 4u));
                cp_async4(&buffers.b[thr_j_offset*SMEM_K + thr_k_offset],
                          &b[(thr_j_offset + cta_j_offset) * size_k + cta_k_offset + thr_k_offset]);
            }
            else {
                const unsigned thr_k_offset = work_idx / (SMEM_J / 4u);
                const unsigned thr_j_offset = 4 * (work_idx % (SMEM_J / 4u));
                cp_async4(&buffers.b[thr_k_offset*SMEM_J + thr_j_offset],
                          &b[(thr_k_offset + cta_k_offset) * size_j + cta_j_offset + thr_j_offset]);
            }
        }
    }

    // Static assignment of warps within CTA to per-warp output tiles (WARP_I, WARP_J) within
    // per-CTA output tile (SMEM_I, SMEM_J)
    __device__ __forceinline__ uint32_t get_warp_i_idx() const
    {
        const uint32_t warp_index = threadIdx.x / 32u;
        return warp_index / (SMEM_J / WARP_J);
    }

    __device__ __forceinline__ uint32_t get_warp_j_idx() const
    {
        const uint32_t warp_index = threadIdx.x / 32u;
        return warp_index % (SMEM_J / WARP_J);
    }

    // Accumulate warp-assigned matrix tile based on values in shared memory buffers.
    __device__ __forceinline__ void warp_accum_block(WarpAccum& accum, const Buffers& buffers) const
    {
        const uint32_t warp_i_idx = get_warp_i_idx();
        const uint32_t warp_j_idx = get_warp_j_idx();

        // B tiles pre-cached in registers for re-use.
        WmmaB mma_b_kj[SMEM_K / MMA_K][WARP_J / MMA_J];
        for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
            for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                const uint32_t j = warp_j_idx*WARP_J + mma_j_idx*MMA_J;
                const uint32_t k = mma_k_idx*MMA_K;
                if constexpr (b_col_major) {
                    load_b(mma_b_kj[mma_k_idx][mma_j_idx], &buffers.b[j*SMEM_K + k], {1, SMEM_K});
                }
                else {
                    load_b(mma_b_kj[mma_k_idx][mma_j_idx], &buffers.b[k*SMEM_J + j], {SMEM_J, 1});
                }
            }
        }

        for (uint32_t mma_i_idx = 0; mma_i_idx < WARP_I / MMA_I; ++mma_i_idx) {
            // Load A tiles needed.
            WmmaA mma_a_k[SMEM_K / MMA_K];
            const uint32_t i = warp_i_idx*WARP_I + mma_i_idx*MMA_I;
            for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                const uint32_t k = mma_k_idx*MMA_K;
                load_a(mma_a_k[mma_k_idx], &buffers.a[i*SMEM_K + k], {SMEM_K, 1});
            }
            for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
                for (uint32_t mma_k_idx = 0; mma_k_idx < SMEM_K / MMA_K; ++mma_k_idx) {
                    wmma(accum.ij[mma_i_idx][mma_j_idx], mma_a_k[mma_k_idx], mma_b_kj[mma_k_idx][mma_j_idx]);
                }
            }
        }
    }

    // CTA cooperates to fill the output matrix block of size (SMEM_I, SMEM_J) starting at (cta_i_offset, cta_j_offset).
    // Requires smem-allocated ring buffer.
    // Includes sync before operations (needed for ring buffer to be safe between usages ... this could be optimized to
    // pipeline across work items, but at least for my other sm_90a kernel, it wasn't worth it).
    __device__ __forceinline__ void cta_sync_compute_block(uint32_t cta_i_offset, uint32_t cta_j_offset,
                                                           RingState& ring, SmemLayout& smem)
    {
        __syncthreads();

        assert(cta_i_offset % SMEM_I == 0);
        assert(cta_j_offset % SMEM_J == 0);
        assert(size_k % SMEM_K == 0);

        WarpAccum accum{};
        const uint32_t K_LAG = RING_BUFFER_SIZE / 2u;
        const uint32_t k_blk_dim = size_k / SMEM_K;
        const uint32_t k_iter_count = k_blk_dim + K_LAG;

        for (uint32_t k_iter = 0; k_iter < k_iter_count; ++k_iter) {
            if (k_iter < k_blk_dim) {
                if (k_iter >= RING_BUFFER_SIZE) {
                    // Don't wait on tile_read_mbar for first RING_BUFFER_SIZE iterations.
                    ring.tile_read_wait_ring(smem);
                }
                cta_async_load_block(smem.buffers[ring.producer_ring_idx],
                                     cta_i_offset, cta_j_offset, k_iter * SMEM_K);
                mbar_arrive_cp_async(smem.tile_fill_mbar[ring.producer_ring_idx]);
                ring.advance_producer_ring_idx();
            }
            if (k_iter >= K_LAG) {
                // Accumulate smem buffer filled K_LAG iterations ago.
                ring.tile_fill_wait_ring(smem);
                warp_accum_block(accum, smem.buffers[ring.consumer_ring_idx]);
                if (k_iter + RING_BUFFER_SIZE < k_iter_count) {
                    // Don't signal tile_read_mbar on last RING_BUFFER_SIZE many iterations, to match producer skip.
                    mbar_arrive_classic(smem.tile_read_mbar[ring.consumer_ring_idx]);
                }
                ring.advance_consumer_ring_idx();
            }
        }

        const uint32_t warp_i_offset = get_warp_i_idx() * WARP_I + cta_i_offset;
        const uint32_t warp_j_offset = get_warp_j_idx() * WARP_J + cta_j_offset;
        if constexpr (c_col_major) {
            for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
                for (uint32_t mma_i_idx = 0; mma_i_idx < WARP_I / MMA_I; ++mma_i_idx) {
                    const auto i = warp_i_offset + MMA_I * mma_i_idx;
                    const auto j = warp_j_offset + MMA_J * mma_j_idx;
                    store_d(c + j * size_i + i, accum.ij[mma_i_idx][mma_j_idx], {1, size_i});
                }
            }
        }
        else {
            for (uint32_t mma_i_idx = 0; mma_i_idx < WARP_I / MMA_I; ++mma_i_idx) {
                for (uint32_t mma_j_idx = 0; mma_j_idx < WARP_J / MMA_J; ++mma_j_idx) {
                    const auto i = warp_i_offset + MMA_I * mma_i_idx;
                    const auto j = warp_j_offset + MMA_J * mma_j_idx;
                    store_d(c + i * size_j + j, accum.ij[mma_i_idx][mma_j_idx], {size_j, 1});
                }
            }
        }
    }

    __device__ __forceinline__ void kernel_main()
    {
        assert(gridDim.x == grid_size());
        assert(blockDim.x == cta_size());

        extern __shared__ char smem_bytes[];
        auto& smem = reinterpret_cast<SmemLayout&>(smem_bytes[0]);

        const uint32_t cta_rows = size_i / SMEM_I;
        const uint32_t cta_cols = size_j / SMEM_J;
        const uint32_t cta_col_remainder = cta_cols % CTA_MODULUS;
        const uint32_t superblock_count = cta_cols / CTA_MODULUS;
        const uint32_t superblock_cta_count = cta_rows * CTA_MODULUS;

        RingState ring{};
        cta_first_time_init(smem);

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

            cta_sync_compute_block(cta_i_idx * SMEM_I, cta_j_idx * SMEM_J, ring, smem);
        }
    }

    void launch(cudaStream_t stream);
};

template <typename Multiplier>
__global__ void
__launch_bounds__(Multiplier::cta_size() * Multiplier::cta_per_sm())
tiled_multiplier_kernel(Multiplier multiplier)
{
    multiplier.kernel_main();
}

template <TileConfig tile_config, bool b_col_major, bool c_col_major>
void TiledMultiplier<tile_config, b_col_major, c_col_major>::launch(cudaStream_t stream)
{
    using Multiplier = std::remove_reference_t<decltype(*this)>;
    const uint32_t grid = grid_size();
    const uint32_t block = cta_size();
    const uint32_t smem = sizeof(SmemLayout);
    cudaFuncSetAttribute(tiled_multiplier_kernel<Multiplier>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    tiled_multiplier_kernel<Multiplier> <<<grid, block, smem, stream>>>(*this);
}

constexpr TileConfig tile_config
{
    256, 192, 16, 3,  // SMEM I,J,K, ring buffer
    64, 96,           // WARP I,J
    128,  // cta modulus
    1,    // cta per sm
};

// A row major ("transposed" per BLAS notation), B column major.
void matmul_TN(uint32_t M, uint32_t N, uint32_t K, const float* a, const float* b, float* c, bool c_col_major,
               cudaStream_t stream)
{
    if (c_col_major) {
        TiledMultiplier<tile_config, true, true> multiplier{M, N, K, a, b, c};
        multiplier.launch(stream);
    }
    else {
        TiledMultiplier<tile_config, true, false> multiplier{M, N, K, a, b, c};
        multiplier.launch(stream);
    }
}

// A and B row major.
void matmul_TT(uint32_t M, uint32_t N, uint32_t K, const float* a, const float* b, float* c, bool c_col_major,
               cudaStream_t stream)
{
    if (c_col_major) {
        TiledMultiplier<tile_config, false, true> multiplier{M, N, K, a, b, c};
        multiplier.launch(stream);
    }
    else {
        TiledMultiplier<tile_config, false, false> multiplier{M, N, K, a, b, c};
        multiplier.launch(stream);
    }
}

}  // end namespace

void matmul_sm80(GPU_Tensors t, cudaStream_t stream) {
    using namespace gemm_sm80_impl;
    assert(matmul_sm80_supports(t));
    if (t.a_col_major != t.b_col_major) {
        // Ideally, we want A to be row major and B to be column major.
        // Swap and transpose matrices if not.
        if (t.a_col_major) {
            matmul_TN(t.N, t.M, t.K, t.b, t.a, t.c, !t.c_col_major, stream);
        }
        else {
            matmul_TN(t.M, t.N, t.K, t.a, t.b, t.c, t.c_col_major, stream);
        }
    }
    else {
        // Need all inputs row major.
        // Swap and transpose matrices if both column major.
        if (t.a_col_major) {
            matmul_TT(t.N, t.M, t.K, t.b, t.a, t.c, !t.c_col_major, stream);
        }
        else {
            matmul_TT(t.M, t.N, t.K, t.a, t.b, t.c, t.c_col_major, stream);
        }
    }
}

bool matmul_sm80_supports(GPU_Tensors t)
{
    using namespace gemm_sm80_impl;

    // Conservative
    static_assert(768 % tile_config.smem_i == 0);
    static_assert(768 % tile_config.smem_j == 0);
    static_assert(32 % tile_config.smem_k == 0);
    return t.M % 768 == 0 && t.N % 768 == 0 && t.K % 32 == 0;
};
