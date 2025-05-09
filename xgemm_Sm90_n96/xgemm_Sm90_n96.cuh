#pragma once
#include "xgemm_Sm90_n96.h"


/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=256, box1=32) */
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=96, box1=32) */
#include <cuda/std/array>


#ifdef __cplusplus
template <typename T, unsigned RegCount>
struct exo_Sm90_RmemMatrixA
{
    T a[RegCount];
    static constexpr unsigned reg_count = RegCount;
};
template <typename T, unsigned RegCount>
struct exo_Sm90_RmemMatrixD
{
    T d[RegCount];
    unsigned scale_d = 0;  // Set to 0 triggers zero-init on the next mma_async call.
    static constexpr unsigned reg_count = RegCount;
};
#endif

typedef struct Sm90_SmemMatrices_SW128 {
    char matrix_bytes[1024];
#ifdef __CUDACC__
    EXO_CUDA_INLINE Sm90_SmemMatrices_SW128& byte_offset(unsigned bytes)
    {
        return reinterpret_cast<Sm90_SmemMatrices_SW128&>(matrix_bytes[bytes]);
    }
#endif
} Sm90_SmemMatrices_SW128;
#include <assert.h>
#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#ifndef EXO_WIN_2F32_Sm90_RmemMatrix
#define EXO_WIN_2F32_Sm90_RmemMatrix
struct exo_win_2f32_Sm90_RmemMatrix{
    float * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2F32_Sm90_tensorMap_128_256_32
#define EXO_WIN_2F32_Sm90_tensorMap_128_256_32

struct exo_win_2f32_Sm90_tensorMap_128_256_32 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned exo_offsets[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_256_32_strides {
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned exo_strides[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_256_32_gmem_dim {
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned exo_dim[2];
};

static inline CUtensorMap exo_win_2f32_Sm90_tensorMap_128_256_32_encode(
        // Window dataptr, layout
        const void* globalAddress, struct exo_win_2f32_Sm90_tensorMap_128_256_32_strides gmem_stride,
        // Tensor size
        struct exo_win_2f32_Sm90_tensorMap_128_256_32_gmem_dim gmem_dim)
{
    assert(gmem_stride.exo_strides[2 - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    cuuint64_t globalDim[2];
    cuuint64_t allGlobalStrides[2];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[2];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_dim = 0; cu_dim < 2; ++cu_dim) {
        const uint32_t exo_dim = 2 - 1 - cu_dim;
        globalDim[cu_dim] = gmem_dim.exo_dim[exo_dim];
        allGlobalStrides[cu_dim] = ((cuuint64_t)gmem_stride.exo_strides[exo_dim]) * 4;
        elementStrides[cu_dim] = 1;
    }

    cuuint32_t boxDim[2] = { 32, 256 };
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {
        fprintf(stderr, "exo_win_2f32_Sm90_tensorMap_128_256_32_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

#endif
#ifndef EXO_WIN_2F32_Sm90_tensorMap_128_96_32
#define EXO_WIN_2F32_Sm90_tensorMap_128_96_32

struct exo_win_2f32_Sm90_tensorMap_128_96_32 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned exo_offsets[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_96_32_strides {
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned exo_strides[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_96_32_gmem_dim {
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned exo_dim[2];
};

static inline CUtensorMap exo_win_2f32_Sm90_tensorMap_128_96_32_encode(
        // Window dataptr, layout
        const void* globalAddress, struct exo_win_2f32_Sm90_tensorMap_128_96_32_strides gmem_stride,
        // Tensor size
        struct exo_win_2f32_Sm90_tensorMap_128_96_32_gmem_dim gmem_dim)
{
    assert(gmem_stride.exo_strides[2 - 1] == 1);

    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

    cuuint64_t globalDim[2];
    cuuint64_t allGlobalStrides[2];  // allGlobalStrides[0] unused by CUDA
    cuuint32_t elementStrides[2];

    // We translate from the Exo ordering (leftmost stride is most-significant)
    // to the CUDA ordering (leftmost stride is least-significant).
    for (uint32_t cu_dim = 0; cu_dim < 2; ++cu_dim) {
        const uint32_t exo_dim = 2 - 1 - cu_dim;
        globalDim[cu_dim] = gmem_dim.exo_dim[exo_dim];
        allGlobalStrides[cu_dim] = ((cuuint64_t)gmem_stride.exo_strides[exo_dim]) * 4;
        elementStrides[cu_dim] = 1;
    }

    cuuint32_t boxDim[2] = { 32, 96 };
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            (void*)globalAddress,
            globalDim,
            &allGlobalStrides[1],  // Cuda presumes least-significant dim is tightly-packed
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {
        fprintf(stderr, "exo_win_2f32_Sm90_tensorMap_128_96_32_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

#endif
#ifndef EXO_WIN_2F32C_Sm90_RmemMatrix
#define EXO_WIN_2F32C_Sm90_RmemMatrix
struct exo_win_2f32c_Sm90_RmemMatrix{
    const float * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_3F32_Sm90_SmemSwizzled_128
#define EXO_WIN_3F32_Sm90_SmemSwizzled_128
struct exo_win_3f32_Sm90_SmemSwizzled_128{
    Sm90_SmemMatrices_SW128 * const data;
    const int_fast32_t strides[3];
};
#endif
#ifndef EXO_WIN_3F32C_Sm90_SmemSwizzled_128
#define EXO_WIN_3F32C_Sm90_SmemSwizzled_128
struct exo_win_3f32c_Sm90_SmemSwizzled_128{
    const Sm90_SmemMatrices_SW128 * const data;
    const int_fast32_t strides[3];
};
#endif

namespace exo_CudaUtil_xgemm_Sm90_n96 {
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=256, box1=32) */
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=96, box1=32) */
EXO_CUDA_INLINE void
exo_Sm90_tma_to_smem_2d(void* dst, const CUtensorMap& tensorMap, cuda::std::array<unsigned, 2> exo_offsets,
                        uint32_t exo_tma_mbarrier, uint32_t n_bytes)

{
    // cute::elect_one_sync
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
    if (pred) {
        asm("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(exo_tma_mbarrier), "r"(n_bytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%2, %3}], [%4], %5;"
            :
            : "r"(exo_smemU32(dst)), "l"(&tensorMap),
              "r"(exo_offsets[1]), "r"(exo_offsets[0]),
              "r"(exo_tma_mbarrier), "n"(1152921504606846976)
            : "memory");
    }
}
/* Required by Sm90_mma_async_tf32(d,a,b,n=96) */
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    uint64_t enc = (val & 0x3FFFF) >> 4;
    return enc;
}

template <unsigned swizzle_bits>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(const void* smem_ptr, uint32_t mn_stride, uint32_t k_stride)
{
    return exo_matrix_descriptor_encode(exo_smemU32(smem_ptr))
           | exo_matrix_descriptor_encode(k_stride) << 16u
           | exo_matrix_descriptor_encode(mn_stride) << 32u
           | uint64_t(swizzle_bits) << 62;
}

/* Required by Sm90_mma_write_d_col_major_tf32(dst,src,n=96) */
template <bool ColumnMajor, typename Window, typename Reg>
EXO_CUDA_INLINE void exo_Sm90_store_d_reg(Window dst, Reg value, uint32_t reg_index)
{
    const uint32_t tid = threadIdx.x % 128u;
    const uint32_t r_base = (tid / 32u) * 16u + (tid % 32u) / 4u;
    const uint32_t c_base = (tid % 4u) * 2u;
    const uint32_t r = r_base + ((reg_index % 4u) / 2u) * 8u;
    const uint32_t c = c_base + (reg_index / 4u) * 8 + (reg_index % 2u);
    auto dst_ptr = reinterpret_cast<Reg*>(
            &dst.data[c * dst.strides[!ColumnMajor] + r * dst.strides[ColumnMajor]]);
    *dst_ptr = value;
}

}  // end namespace
// CUDA device function args -- duplicated in .c file
struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n96
{
    int_fast32_t M;  // M : size
    int_fast32_t N;  // N : size
    int_fast32_t K;  // K : size
    float* C;  // C : f32[N, M] @CudaGmemLinear
    CUtensorMap exo_data_A_tensorMap;  //     (Separate window data pointer)
    struct exo_win_2f32_Sm90_tensorMap_128_256_32 A_tensorMap;  // A_tensorMap : Window(src_type=f32[M, K], as_tensor=[f32][M, K], src_buf=A, idx='[0:M, 0:K]') @Sm90_tensorMap(128, 256, 32)
    CUtensorMap exo_data_B_tensorMap;  //     (Separate window data pointer)
    struct exo_win_2f32_Sm90_tensorMap_128_96_32 B_tensorMap;  // B_tensorMap : Window(src_type=f32[N, K], as_tensor=[f32][N, K], src_buf=B, idx='[0:N, 0:K]') @Sm90_tensorMap(128, 96, 32)
};

struct exo_Cuda0_xgemm_Sm90_wgmma_n96
{
  using exo_DeviceArgs = exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n96;

  static constexpr uint32_t exo_blockDim = 384;
  static constexpr uint32_t exo_clusterDim = 1;

  static constexpr unsigned exo_smemBytes = 180352;

  struct exo_Task
  {
    int_fast32_t m_task;
    int_fast32_t n_task;
  };

  struct exo_SyncState
  {
    // ringbar: barrier @ CudaMbarrier, ring=4, slice_count=1
    // (forward) mbarriers [0, 4]; arrive_count=32
    // (reverse) mbarriers [4, 8]; arrive_count=256
    unsigned ArriveIdx0_ringbar : 2 = 0;
    __device__ __forceinline__ uint32_t Arrive0_ringbar(char* exo_smem, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + 0 + 8*(slice * 4 + ArriveIdx0_ringbar));
      if (enable) {
        asm("// Arrive0_ringbar\n\tmbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbarrier_u32));
        // Advance ring buffer state
        ArriveIdx0_ringbar = ArriveIdx0_ringbar == 3 ? 0 : ArriveIdx0_ringbar + 1;
      }
      return mbarrier_u32;
    }
    unsigned AwaitIdx0_ringbar : 2 = 0;
    unsigned Parity0_ringbar : 4 = 0;
    __device__ __forceinline__ void Await0_ringbar(char* exo_smem, int slice) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + 0 + 8*(slice * 4 + AwaitIdx0_ringbar));
      const bool enable = true;
      if (enable) {
        // Wait for mbarrier ... PTX loop needed for this
        asm volatile("// Await0_ringbar\n\t{.reg.pred P1;\n\t"
                     "EXO_BEFORE_WAIT: mbarrier."
    #if __CUDA_ARCH__ >= 900
                     "try_wait"
    #else
                     "test_wait"
    #endif
                     ".parity.acquire.cta.shared::cta.b64\n\t"
                     "P1, [%0], %1;\n\t"
                     "@P1 bra.uni EXO_WAIT_DONE;\n\t"
                     "bra.uni EXO_BEFORE_WAIT; EXO_WAIT_DONE: }"::
            "r"(mbarrier_u32), "r"(1u & Parity0_ringbar >> AwaitIdx0_ringbar));
        // Flip parity
        Parity0_ringbar ^= 1u << AwaitIdx0_ringbar;
        // Advance ring buffer state
        AwaitIdx0_ringbar = AwaitIdx0_ringbar == 3 ? 0 : AwaitIdx0_ringbar + 1;
      }
    }
    unsigned ReverseArriveIdx0_ringbar : 2 = 0;
    __device__ __forceinline__ uint32_t ReverseArrive0_ringbar(char* exo_smem, int slice, bool enable) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + 32 + 8*(slice * 4 + ReverseArriveIdx0_ringbar));
      if (enable) {
        asm("// ReverseArrive0_ringbar\n\tmbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbarrier_u32));
        // Advance ring buffer state
        ReverseArriveIdx0_ringbar = ReverseArriveIdx0_ringbar == 3 ? 0 : ReverseArriveIdx0_ringbar + 1;
      }
      return mbarrier_u32;
    }
    unsigned ReverseAwaitIdx0_ringbar : 2 = 0;
    unsigned ReverseParity0_ringbar : 4 = 0;
    unsigned ReverseSkips0_ringbar : 3 = 0;
    __device__ __forceinline__ void ReverseAwait0_ringbar(char* exo_smem, int slice, int initial_skips = 0) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem + 32 + 8*(slice * 4 + ReverseAwaitIdx0_ringbar));
      const bool enable = ReverseSkips0_ringbar >= initial_skips;
      if (enable) {
        // Wait for mbarrier ... PTX loop needed for this
        asm volatile("// ReverseAwait0_ringbar\n\t{.reg.pred P1;\n\t"
                     "EXO_BEFORE_WAIT: mbarrier."
    #if __CUDA_ARCH__ >= 900
                     "try_wait"
    #else
                     "test_wait"
    #endif
                     ".parity.acquire.cta.shared::cta.b64\n\t"
                     "P1, [%0], %1;\n\t"
                     "@P1 bra.uni EXO_WAIT_DONE;\n\t"
                     "bra.uni EXO_BEFORE_WAIT; EXO_WAIT_DONE: }"::
            "r"(mbarrier_u32), "r"(1u & ReverseParity0_ringbar >> ReverseAwaitIdx0_ringbar));
        // Flip parity
        ReverseParity0_ringbar ^= 1u << ReverseAwaitIdx0_ringbar;
        // Advance ring buffer state
        ReverseAwaitIdx0_ringbar = ReverseAwaitIdx0_ringbar == 3 ? 0 : ReverseAwaitIdx0_ringbar + 1;
      }
      else {
        // ReverseAwait(ringbar) returns without waiting for mbarrier first <initial_skips> times
        ReverseSkips0_ringbar++;
      }
    }
  };

  static void
  exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);

  static __device__ __forceinline__ void
  exo_deviceTask_consumer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);

  static __device__ __forceinline__ void
  exo_deviceTask_producer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);

  static __device__ __forceinline__ void
  exo_deviceTask_unused(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);
};

inline void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs)
{
  cudaFuncSetAttribute(exo_deviceFunction0_xgemm_Sm90_wgmma_n96, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
  // TODO how expensive is it to query this every time?
  int exo_cudaDevice;
  cudaGetDevice(&exo_cudaDevice);
  int exo_SMs;
  cudaDeviceGetAttribute(&exo_SMs, cudaDevAttrMultiProcessorCount, exo_cudaDevice);
  const unsigned exo_gridDim = (unsigned(exo_SMs) / exo_clusterDim) * 1u;

  cudaLaunchConfig_t exo_launchConfig = {};
  exo_launchConfig.gridDim = dim3(exo_gridDim, 1, 1);
  exo_launchConfig.blockDim = dim3(exo_blockDim, 1, 1);
  exo_launchConfig.dynamicSmemBytes = exo_smemBytes;
  exo_launchConfig.stream = exo_cudaStream;

  cudaLaunchKernelEx(&exo_launchConfig, exo_deviceFunction0_xgemm_Sm90_wgmma_n96, exo_deviceArgs);
}

__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
{
    if (threadIdx.x == 0) {
      const auto mbarrier_u32 = exo_smemU32(exo_smem);
      for (int i = 0; i < 4; ++i) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;"::
          "r"(mbarrier_u32 + 0 + 8*i));
      }
      for (int i = 0; i < 4; ++i) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;"::
          "r"(mbarrier_u32 + 32 + 8*i));
      }
      asm("fence.proxy.async;");
    }
    __syncthreads();
}

__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_deviceTask_producer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n96;
  // Scope of named barrier ringbar
  ; // NO-OP
  ; // NO-OP
  auto& A_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[128]);
  auto& B_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[131200]);
  ; // NO-OP
  for (int k_iter = 0; k_iter < 1; k_iter++) {
    if (int tmp_1 = threadIdx.x; tmp_1 < 32) {
      {
        // CudaAsync(tma_to_smem_async)
        // Fence(empty_actor_kind, empty_actor_kind)
        const uint32_t exo_tma_mbarrier = exo_syncState.Arrive0_ringbar(exo_smem, 0, 0);
        // ReverseAwait(ringbar, cuda_temporal, ~4)
        exo_syncState.ReverseAwait0_ringbar(exo_smem, 0, 4);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &A_smem[((k_iter % 4) * (8192)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_A_tensorMap,
                    {{ exo_deviceArgs.A_tensorMap.exo_offsets[0] + (unsigned)(256 * exo_task.m_task), exo_deviceArgs.A_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    32768);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &B_smem[((k_iter % 4) * (3072)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_B_tensorMap,
                    {{ exo_deviceArgs.B_tensorMap.exo_offsets[0] + (unsigned)(96 * exo_task.n_task), exo_deviceArgs.B_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    12288);
        // Arrive(tma_to_smem_async, ringbar, 1)
        exo_syncState.Arrive0_ringbar(exo_smem, 0, 1);
      }
    }
    ; // NO-OP
  }
  for (int k_iter = 1; k_iter < ((exo_deviceArgs.K) / (32)); k_iter++) {
    if (int tmp_1 = threadIdx.x; tmp_1 < 32) {
      {
        // CudaAsync(tma_to_smem_async)
        // Fence(empty_actor_kind, empty_actor_kind)
        const uint32_t exo_tma_mbarrier = exo_syncState.Arrive0_ringbar(exo_smem, 0, 0);
        // ReverseAwait(ringbar, cuda_temporal, ~4)
        exo_syncState.ReverseAwait0_ringbar(exo_smem, 0, 4);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &A_smem[((k_iter % 4) * (8192)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_A_tensorMap,
                    {{ exo_deviceArgs.A_tensorMap.exo_offsets[0] + (unsigned)(256 * exo_task.m_task), exo_deviceArgs.A_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    32768);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &B_smem[((k_iter % 4) * (3072)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_B_tensorMap,
                    {{ exo_deviceArgs.B_tensorMap.exo_offsets[0] + (unsigned)(96 * exo_task.n_task), exo_deviceArgs.B_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    12288);
        // Arrive(tma_to_smem_async, ringbar, 1)
        exo_syncState.Arrive0_ringbar(exo_smem, 0, 1);
      }
    }
    ; // NO-OP
  }
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  // Fence(cuda_classic, cuda_classic)
  asm("barrier.cta.sync 0;");
}
__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_deviceTask_unused(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n96;
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  for (int k_iter = 0; k_iter < 1; k_iter++) {
    ; // NO-OP
    ; // NO-OP
  }
  for (int k_iter = 1; k_iter < ((exo_deviceArgs.K) / (32)); k_iter++) {
    ; // NO-OP
    ; // NO-OP
  }
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  ; // NO-OP
  // Fence(cuda_classic, cuda_classic)
  asm("barrier.cta.sync 0;");
}
__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_deviceTask_consumer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n96;
  // Scope of named barrier ringbar
  // Scope of named barrier cg
  exo_Sm90_RmemMatrixD<float, 48> D_rmem[2];
  auto& A_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[128]);
  auto& B_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[131200]);
  if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
    if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
      for (int m_mma = 0; m_mma < 2; m_mma++) {
        D_rmem[m_mma].scale_d = 0;
      }
    }
  }
  for (int k_iter = 0; k_iter < 1; k_iter++) {
    ; // NO-OP
    if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
      if (int tmp_2 = (threadIdx.x - 128); tmp_2 >= 32 && tmp_2 < 96) {
        ; // NO-OP
      }
      // Await(ringbar, wgmma_async, ~0)
      exo_syncState.Await0_ringbar(exo_smem, 0);
      if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
        {
          // CudaAsync(wgmma_async)
          // Fence(wgmma_fence_1, wgmma_fence_2)
          asm("wgmma.fence.sync.aligned;");
          for (int k_mma = 0; k_mma < 4; k_mma++) {
            for (int m_mma = 0; m_mma < 2; m_mma++) {
              asm volatile("{.reg .pred p;\n\tsetp.ne.b32 p, %50, 0;\n\twgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32\n\t{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47},\n\t%48, %49, p, 1, 1;\n}": "+f"(D_rmem[m_mma].d[0]), "+f"(D_rmem[m_mma].d[1]), "+f"(D_rmem[m_mma].d[2]), "+f"(D_rmem[m_mma].d[3]), "+f"(D_rmem[m_mma].d[4]), "+f"(D_rmem[m_mma].d[5]), "+f"(D_rmem[m_mma].d[6]), "+f"(D_rmem[m_mma].d[7]), "+f"(D_rmem[m_mma].d[8]), "+f"(D_rmem[m_mma].d[9]), "+f"(D_rmem[m_mma].d[10]), "+f"(D_rmem[m_mma].d[11]), "+f"(D_rmem[m_mma].d[12]), "+f"(D_rmem[m_mma].d[13]), "+f"(D_rmem[m_mma].d[14]), "+f"(D_rmem[m_mma].d[15]), "+f"(D_rmem[m_mma].d[16]), "+f"(D_rmem[m_mma].d[17]), "+f"(D_rmem[m_mma].d[18]), "+f"(D_rmem[m_mma].d[19]), "+f"(D_rmem[m_mma].d[20]), "+f"(D_rmem[m_mma].d[21]), "+f"(D_rmem[m_mma].d[22]), "+f"(D_rmem[m_mma].d[23]), "+f"(D_rmem[m_mma].d[24]), "+f"(D_rmem[m_mma].d[25]), "+f"(D_rmem[m_mma].d[26]), "+f"(D_rmem[m_mma].d[27]), "+f"(D_rmem[m_mma].d[28]), "+f"(D_rmem[m_mma].d[29]), "+f"(D_rmem[m_mma].d[30]), "+f"(D_rmem[m_mma].d[31]), "+f"(D_rmem[m_mma].d[32]), "+f"(D_rmem[m_mma].d[33]), "+f"(D_rmem[m_mma].d[34]), "+f"(D_rmem[m_mma].d[35]), "+f"(D_rmem[m_mma].d[36]), "+f"(D_rmem[m_mma].d[37]), "+f"(D_rmem[m_mma].d[38]), "+f"(D_rmem[m_mma].d[39]), "+f"(D_rmem[m_mma].d[40]), "+f"(D_rmem[m_mma].d[41]), "+f"(D_rmem[m_mma].d[42]), "+f"(D_rmem[m_mma].d[43]), "+f"(D_rmem[m_mma].d[44]), "+f"(D_rmem[m_mma].d[45]), "+f"(D_rmem[m_mma].d[46]), "+f"(D_rmem[m_mma].d[47]): "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&A_smem[((k_iter % 4) * (8192) + (8 * m_mma + 16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), 1024, 1024)), "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&B_smem[((k_iter % 4) * (3072) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), 1024, 1024)), "r"(D_rmem[m_mma].scale_d));
D_rmem[m_mma].scale_d = 1;
            }
          }
          // Arrive(wgmma_async, cg[wg], 1)
          asm("wgmma.commit_group.sync.aligned;");
        }
        if (k_iter >= 1) {
          // Await(cg[wg], cuda_classic, 1)
          asm("wgmma.wait_group.sync.aligned 1;");
        }
      }
      if (k_iter >= 1) {
        // ReverseArrive(cuda_classic, ringbar, 1)
        exo_syncState.ReverseArrive0_ringbar(exo_smem, 0, 1);
      }
    }
  }
  for (int k_iter = 1; k_iter < ((exo_deviceArgs.K) / (32)); k_iter++) {
    ; // NO-OP
    if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
      if (int tmp_2 = (threadIdx.x - 128); tmp_2 >= 32 && tmp_2 < 96) {
        ; // NO-OP
      }
      // Await(ringbar, wgmma_async, ~0)
      exo_syncState.Await0_ringbar(exo_smem, 0);
      if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
        {
          // CudaAsync(wgmma_async)
          // Fence(wgmma_fence_1, wgmma_fence_2)
          asm("wgmma.fence.sync.aligned;");
          for (int k_mma = 0; k_mma < 4; k_mma++) {
            for (int m_mma = 0; m_mma < 2; m_mma++) {
              asm volatile("{.reg .pred p;\n\tsetp.ne.b32 p, %50, 0;\n\twgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32\n\t{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47},\n\t%48, %49, p, 1, 1;\n}": "+f"(D_rmem[m_mma].d[0]), "+f"(D_rmem[m_mma].d[1]), "+f"(D_rmem[m_mma].d[2]), "+f"(D_rmem[m_mma].d[3]), "+f"(D_rmem[m_mma].d[4]), "+f"(D_rmem[m_mma].d[5]), "+f"(D_rmem[m_mma].d[6]), "+f"(D_rmem[m_mma].d[7]), "+f"(D_rmem[m_mma].d[8]), "+f"(D_rmem[m_mma].d[9]), "+f"(D_rmem[m_mma].d[10]), "+f"(D_rmem[m_mma].d[11]), "+f"(D_rmem[m_mma].d[12]), "+f"(D_rmem[m_mma].d[13]), "+f"(D_rmem[m_mma].d[14]), "+f"(D_rmem[m_mma].d[15]), "+f"(D_rmem[m_mma].d[16]), "+f"(D_rmem[m_mma].d[17]), "+f"(D_rmem[m_mma].d[18]), "+f"(D_rmem[m_mma].d[19]), "+f"(D_rmem[m_mma].d[20]), "+f"(D_rmem[m_mma].d[21]), "+f"(D_rmem[m_mma].d[22]), "+f"(D_rmem[m_mma].d[23]), "+f"(D_rmem[m_mma].d[24]), "+f"(D_rmem[m_mma].d[25]), "+f"(D_rmem[m_mma].d[26]), "+f"(D_rmem[m_mma].d[27]), "+f"(D_rmem[m_mma].d[28]), "+f"(D_rmem[m_mma].d[29]), "+f"(D_rmem[m_mma].d[30]), "+f"(D_rmem[m_mma].d[31]), "+f"(D_rmem[m_mma].d[32]), "+f"(D_rmem[m_mma].d[33]), "+f"(D_rmem[m_mma].d[34]), "+f"(D_rmem[m_mma].d[35]), "+f"(D_rmem[m_mma].d[36]), "+f"(D_rmem[m_mma].d[37]), "+f"(D_rmem[m_mma].d[38]), "+f"(D_rmem[m_mma].d[39]), "+f"(D_rmem[m_mma].d[40]), "+f"(D_rmem[m_mma].d[41]), "+f"(D_rmem[m_mma].d[42]), "+f"(D_rmem[m_mma].d[43]), "+f"(D_rmem[m_mma].d[44]), "+f"(D_rmem[m_mma].d[45]), "+f"(D_rmem[m_mma].d[46]), "+f"(D_rmem[m_mma].d[47]): "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&A_smem[((k_iter % 4) * (8192) + (8 * m_mma + 16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), 1024, 1024)), "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&B_smem[((k_iter % 4) * (3072) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), 1024, 1024)), "r"(D_rmem[m_mma].scale_d));
D_rmem[m_mma].scale_d = 1;
            }
          }
          // Arrive(wgmma_async, cg[wg], 1)
          asm("wgmma.commit_group.sync.aligned;");
        }
        if (k_iter >= 1) {
          // Await(cg[wg], cuda_classic, 1)
          asm("wgmma.wait_group.sync.aligned 1;");
        }
      }
      if (k_iter >= 1) {
        // ReverseArrive(cuda_classic, ringbar, 1)
        exo_syncState.ReverseArrive0_ringbar(exo_smem, 0, 1);
      }
    }
  }
  if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
    if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
      // Await(cg[wg], cuda_classic, 0)
      asm("wgmma.wait_group.sync.aligned 0;");
    }
    // ReverseArrive(cuda_classic, ringbar, ~0)
    exo_syncState.ReverseArrive0_ringbar(exo_smem, 0, -1);
  }
  if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
    if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
      for (int m_mma = 0; m_mma < 2; m_mma++) {
        exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[0], 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[1], 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[2], 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[3], 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[4], 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[5], 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[6], 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[7], 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[8], 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[9], 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[10], 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[11], 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[12], 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[13], 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[14], 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[15], 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[16], 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[17], 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[18], 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[19], 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[20], 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[21], 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[22], 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[23], 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[24], 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[25], 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[26], 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[27], 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[28], 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[29], 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[30], 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[31], 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[32], 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[33], 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[34], 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[35], 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[36], 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[37], 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[38], 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[39], 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[40], 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[41], 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[42], 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[43], 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[44], 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[45], 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[46], 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(96 * exo_task.n_task) * (exo_deviceArgs.M) + 64 * m_mma + 128 * exo_128thr_wg + 256 * exo_task.m_task], { exo_deviceArgs.M, 1 } }), D_rmem[m_mma].d[47], 47);
      }
    }
  }
  // Fence(cuda_classic, cuda_classic)
  asm("barrier.cta.sync 0;");
}
__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n96::exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
{
  exo_SyncState exo_syncState{};
  unsigned exo_taskIndex = 0;
  int32_t nreg;
  nreg = ((int32_t) 0);
  if (int tmp = threadIdx.x; tmp >= 128 && tmp < 384) {
    nreg = ((int32_t) 232);
  }
  if (int tmp = threadIdx.x; tmp >= 0 && tmp < 32) {
    nreg = ((int32_t) 40);
  }
  if (int tmp = threadIdx.x; tmp >= 32 && tmp < 128) {
    nreg = ((int32_t) 40);
  }
  if (nreg == ((int32_t) 40)) {
    asm("setmaxnreg.dec.sync.aligned.u32 40;");
    if (int tmp = threadIdx.x; tmp >= 0 && tmp < 32) {
      for (int exo_task_m_task = 0; exo_task_m_task < ((exo_deviceArgs.M) / (256)); exo_task_m_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (96)); exo_task_n_task++) {
          if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_producer(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m_task, exo_task_n_task });
        }
      }
    }
    if (int tmp = threadIdx.x; tmp >= 32 && tmp < 128) {
      for (int exo_task_m_task = 0; exo_task_m_task < ((exo_deviceArgs.M) / (256)); exo_task_m_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (96)); exo_task_n_task++) {
          if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_unused(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m_task, exo_task_n_task });
        }
      }
    }
  }
  if (nreg == ((int32_t) 232)) {
    asm("setmaxnreg.inc.sync.aligned.u32 232;");
    if (int tmp = threadIdx.x; tmp >= 128 && tmp < 384) {
      for (int exo_task_m_task = 0; exo_task_m_task < ((exo_deviceArgs.M) / (256)); exo_task_m_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (96)); exo_task_n_task++) {
          if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_consumer(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m_task, exo_task_n_task });
        }
      }
    }
  }
}

