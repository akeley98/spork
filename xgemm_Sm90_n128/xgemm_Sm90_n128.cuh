#pragma once
#include "xgemm_Sm90_n128.h"


/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=256, box1=32) */
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=128, box1=32) */
#include <cuda/std/array>



typedef struct Sm90_SmemMatrices_SW128 {
    char matrix_bytes[1024];
#ifdef __CUDACC__
    EXO_CUDA_INLINE Sm90_SmemMatrices_SW128& byte_offset(unsigned bytes)
    {
        return reinterpret_cast<Sm90_SmemMatrices_SW128&>(matrix_bytes[bytes]);
    }

    EXO_CUDA_INLINE uint64_t get_swizzle_bits() const
    {
        return 1;
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
#ifndef EXO_WIN_2F32_Sm90_tensorMap_128_128_32
#define EXO_WIN_2F32_Sm90_tensorMap_128_128_32

struct exo_win_2f32_Sm90_tensorMap_128_128_32 {
    // Stored in reverse-order as the raw CUtensorMap.
    // Leftmost offset is most-significant.
    unsigned exo_offsets[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_128_32_strides {
    // Stored in reverse-order as the raw CUtensorMap,
    // and in element count, not in bytes.
    // Leftmost stride is most-significant.
    unsigned exo_strides[2];
};

struct exo_win_2f32_Sm90_tensorMap_128_128_32_gmem_dim {
    // Stored in the reverse-order as the raw CUtensorMap.
    // Leftmost dimension is the most-significant.
    unsigned exo_dim[2];
};

static inline CUtensorMap exo_win_2f32_Sm90_tensorMap_128_128_32_encode(
        // Window dataptr, layout
        const void* globalAddress, struct exo_win_2f32_Sm90_tensorMap_128_128_32_strides gmem_stride,
        // Tensor size
        struct exo_win_2f32_Sm90_tensorMap_128_128_32_gmem_dim gmem_dim)
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

    cuuint32_t boxDim[2] = { 32, 128 };
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
        fprintf(stderr, "exo_win_2f32_Sm90_tensorMap_128_128_32_encode: error %i\n", (int)result);
        assert(0);
    }
    return tensorMap;
}

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

namespace exo_CudaUtil_xgemm_Sm90_n128 {
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=256, box1=32) */
/* Required by Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=128, box1=32) */
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
/* Required by Sm90_mma_async_tf32(d,a,b,M=128, N=128) */
/* Required by Sm90_mma_write_d_col_major_tf32(dst,src,M=128, N=128) */
template <bool ColumnMajor, typename Window, typename Reg>
EXO_CUDA_INLINE void exo_Sm90_store_d_reg(Window dst, Reg value, uint32_t m_offset, uint32_t reg_index)
{
    const uint32_t tid = threadIdx.x % 128u;
    const uint32_t r_base = (tid / 32u) * 16u + (tid % 32u) / 4u;
    const uint32_t c_base = (tid % 4u) * 2u;
    const uint32_t r = m_offset + r_base + ((reg_index % 4u) / 2u) * 8u;
    const uint32_t c = c_base + (reg_index / 4u) * 8 + (reg_index % 2u);
    auto dst_ptr = reinterpret_cast<Reg*>(
            &dst.data[c * dst.strides[!ColumnMajor] + r * dst.strides[ColumnMajor]]);
    *dst_ptr = value;
}

/* Required by Sm90_mma_async_tf32(d,a,b,M=128, N=128) */
/* Required by Sm90_mma_write_d_col_major_tf32(dst,src,M=128, N=128) */
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor_encode(uint32_t val)
{
    return (val & 0x3FFFF) >> 4;
}

template <typename Window>
EXO_CUDA_INLINE uint64_t exo_matrix_descriptor(Window window, uint32_t element_size, uint32_t mn_offset = 0)
{
    uint64_t mn_stride = window.strides[0] * element_size;
    return exo_matrix_descriptor_encode(exo_smemU32(window.data) + (mn_offset / 8u) * mn_stride)
           | exo_matrix_descriptor_encode(sizeof(*window.data)) << 16u
           | exo_matrix_descriptor_encode(mn_stride) << 32u
           | uint64_t(window.data->get_swizzle_bits()) << 62;
}

/* Required by Sm90_mma_async_tf32(d,a,b,M=128, N=128) */
/* Required by Sm90_mma_write_d_col_major_tf32(dst,src,M=128, N=128) */
struct exo_Sm90_RmemD_m128n128_f32 {
    float m0n0r0, m0n0r1, m0n0r2, m0n0r3, m0n8r0, m0n8r1, m0n8r2, m0n8r3, m0n16r0, m0n16r1, m0n16r2, m0n16r3, m0n24r0, m0n24r1, m0n24r2, m0n24r3, m0n32r0, m0n32r1, m0n32r2, m0n32r3, m0n40r0, m0n40r1, m0n40r2, m0n40r3, m0n48r0, m0n48r1, m0n48r2, m0n48r3, m0n56r0, m0n56r1, m0n56r2, m0n56r3, m0n64r0, m0n64r1, m0n64r2, m0n64r3, m0n72r0, m0n72r1, m0n72r2, m0n72r3, m0n80r0, m0n80r1, m0n80r2, m0n80r3, m0n88r0, m0n88r1, m0n88r2, m0n88r3, m0n96r0, m0n96r1, m0n96r2, m0n96r3, m0n104r0, m0n104r1, m0n104r2, m0n104r3, m0n112r0, m0n112r1, m0n112r2, m0n112r3, m0n120r0, m0n120r1, m0n120r2, m0n120r3, m64n0r0, m64n0r1, m64n0r2, m64n0r3, m64n8r0, m64n8r1, m64n8r2, m64n8r3, m64n16r0, m64n16r1, m64n16r2, m64n16r3, m64n24r0, m64n24r1, m64n24r2, m64n24r3, m64n32r0, m64n32r1, m64n32r2, m64n32r3, m64n40r0, m64n40r1, m64n40r2, m64n40r3, m64n48r0, m64n48r1, m64n48r2, m64n48r3, m64n56r0, m64n56r1, m64n56r2, m64n56r3, m64n64r0, m64n64r1, m64n64r2, m64n64r3, m64n72r0, m64n72r1, m64n72r2, m64n72r3, m64n80r0, m64n80r1, m64n80r2, m64n80r3, m64n88r0, m64n88r1, m64n88r2, m64n88r3, m64n96r0, m64n96r1, m64n96r2, m64n96r3, m64n104r0, m64n104r1, m64n104r2, m64n104r3, m64n112r0, m64n112r1, m64n112r2, m64n112r3, m64n120r0, m64n120r1, m64n120r2, m64n120r3;
    int scale_d;
};
/* Required by Sm90_mma_async_tf32(d,a,b,M=128, N=128) */
/* Required by Sm90_mma_write_d_col_major_tf32(dst,src,M=128, N=128) */
EXO_CUDA_INLINE void exo_Sm90_mma_async_ss_m128n128_f32_tf32_tf32(uint64_t a_descriptor_m0, uint64_t a_descriptor_m64, uint64_t b_descriptor, float& m0n0r0, float& m0n0r1, float& m0n0r2, float& m0n0r3, float& m0n8r0, float& m0n8r1, float& m0n8r2, float& m0n8r3, float& m0n16r0, float& m0n16r1, float& m0n16r2, float& m0n16r3, float& m0n24r0, float& m0n24r1, float& m0n24r2, float& m0n24r3, float& m0n32r0, float& m0n32r1, float& m0n32r2, float& m0n32r3, float& m0n40r0, float& m0n40r1, float& m0n40r2, float& m0n40r3, float& m0n48r0, float& m0n48r1, float& m0n48r2, float& m0n48r3, float& m0n56r0, float& m0n56r1, float& m0n56r2, float& m0n56r3, float& m0n64r0, float& m0n64r1, float& m0n64r2, float& m0n64r3, float& m0n72r0, float& m0n72r1, float& m0n72r2, float& m0n72r3, float& m0n80r0, float& m0n80r1, float& m0n80r2, float& m0n80r3, float& m0n88r0, float& m0n88r1, float& m0n88r2, float& m0n88r3, float& m0n96r0, float& m0n96r1, float& m0n96r2, float& m0n96r3, float& m0n104r0, float& m0n104r1, float& m0n104r2, float& m0n104r3, float& m0n112r0, float& m0n112r1, float& m0n112r2, float& m0n112r3, float& m0n120r0, float& m0n120r1, float& m0n120r2, float& m0n120r3, float& m64n0r0, float& m64n0r1, float& m64n0r2, float& m64n0r3, float& m64n8r0, float& m64n8r1, float& m64n8r2, float& m64n8r3, float& m64n16r0, float& m64n16r1, float& m64n16r2, float& m64n16r3, float& m64n24r0, float& m64n24r1, float& m64n24r2, float& m64n24r3, float& m64n32r0, float& m64n32r1, float& m64n32r2, float& m64n32r3, float& m64n40r0, float& m64n40r1, float& m64n40r2, float& m64n40r3, float& m64n48r0, float& m64n48r1, float& m64n48r2, float& m64n48r3, float& m64n56r0, float& m64n56r1, float& m64n56r2, float& m64n56r3, float& m64n64r0, float& m64n64r1, float& m64n64r2, float& m64n64r3, float& m64n72r0, float& m64n72r1, float& m64n72r2, float& m64n72r3, float& m64n80r0, float& m64n80r1, float& m64n80r2, float& m64n80r3, float& m64n88r0, float& m64n88r1, float& m64n88r2, float& m64n88r3, float& m64n96r0, float& m64n96r1, float& m64n96r2, float& m64n96r3, float& m64n104r0, float& m64n104r1, float& m64n104r2, float& m64n104r3, float& m64n112r0, float& m64n112r1, float& m64n112r2, float& m64n112r3, float& m64n120r0, float& m64n120r1, float& m64n120r2, float& m64n120r3, int scale_d)
{
  asm volatile("{\n"
  ".reg .pred p;\n"
  "setp.ne.b32 p, %66, 0;\n"
  "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
  "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
  "%64, %65, p, 1, 1;\n"
  "}"
  : "+f"(m0n0r0), "+f"(m0n0r1), "+f"(m0n0r2), "+f"(m0n0r3), "+f"(m0n8r0), "+f"(m0n8r1), "+f"(m0n8r2), "+f"(m0n8r3), "+f"(m0n16r0), "+f"(m0n16r1), "+f"(m0n16r2), "+f"(m0n16r3), "+f"(m0n24r0), "+f"(m0n24r1), "+f"(m0n24r2), "+f"(m0n24r3), "+f"(m0n32r0), "+f"(m0n32r1), "+f"(m0n32r2), "+f"(m0n32r3), "+f"(m0n40r0), "+f"(m0n40r1), "+f"(m0n40r2), "+f"(m0n40r3), "+f"(m0n48r0), "+f"(m0n48r1), "+f"(m0n48r2), "+f"(m0n48r3), "+f"(m0n56r0), "+f"(m0n56r1), "+f"(m0n56r2), "+f"(m0n56r3), "+f"(m0n64r0), "+f"(m0n64r1), "+f"(m0n64r2), "+f"(m0n64r3), "+f"(m0n72r0), "+f"(m0n72r1), "+f"(m0n72r2), "+f"(m0n72r3), "+f"(m0n80r0), "+f"(m0n80r1), "+f"(m0n80r2), "+f"(m0n80r3), "+f"(m0n88r0), "+f"(m0n88r1), "+f"(m0n88r2), "+f"(m0n88r3), "+f"(m0n96r0), "+f"(m0n96r1), "+f"(m0n96r2), "+f"(m0n96r3), "+f"(m0n104r0), "+f"(m0n104r1), "+f"(m0n104r2), "+f"(m0n104r3), "+f"(m0n112r0), "+f"(m0n112r1), "+f"(m0n112r2), "+f"(m0n112r3), "+f"(m0n120r0), "+f"(m0n120r1), "+f"(m0n120r2), "+f"(m0n120r3)
  : "l"(a_descriptor_m0), "l"(b_descriptor), "r"(scale_d)
  );
  asm volatile("{\n"
  ".reg .pred p;\n"
  "setp.ne.b32 p, %66, 0;\n"
  "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
  "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
  "%64, %65, p, 1, 1;\n"
  "}"
  : "+f"(m64n0r0), "+f"(m64n0r1), "+f"(m64n0r2), "+f"(m64n0r3), "+f"(m64n8r0), "+f"(m64n8r1), "+f"(m64n8r2), "+f"(m64n8r3), "+f"(m64n16r0), "+f"(m64n16r1), "+f"(m64n16r2), "+f"(m64n16r3), "+f"(m64n24r0), "+f"(m64n24r1), "+f"(m64n24r2), "+f"(m64n24r3), "+f"(m64n32r0), "+f"(m64n32r1), "+f"(m64n32r2), "+f"(m64n32r3), "+f"(m64n40r0), "+f"(m64n40r1), "+f"(m64n40r2), "+f"(m64n40r3), "+f"(m64n48r0), "+f"(m64n48r1), "+f"(m64n48r2), "+f"(m64n48r3), "+f"(m64n56r0), "+f"(m64n56r1), "+f"(m64n56r2), "+f"(m64n56r3), "+f"(m64n64r0), "+f"(m64n64r1), "+f"(m64n64r2), "+f"(m64n64r3), "+f"(m64n72r0), "+f"(m64n72r1), "+f"(m64n72r2), "+f"(m64n72r3), "+f"(m64n80r0), "+f"(m64n80r1), "+f"(m64n80r2), "+f"(m64n80r3), "+f"(m64n88r0), "+f"(m64n88r1), "+f"(m64n88r2), "+f"(m64n88r3), "+f"(m64n96r0), "+f"(m64n96r1), "+f"(m64n96r2), "+f"(m64n96r3), "+f"(m64n104r0), "+f"(m64n104r1), "+f"(m64n104r2), "+f"(m64n104r3), "+f"(m64n112r0), "+f"(m64n112r1), "+f"(m64n112r2), "+f"(m64n112r3), "+f"(m64n120r0), "+f"(m64n120r1), "+f"(m64n120r2), "+f"(m64n120r3)
  : "l"(a_descriptor_m64), "l"(b_descriptor), "r"(scale_d)
  );
}
}  // end namespace
// CUDA device function args -- duplicated in .c file
struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n128
{
    int_fast32_t M;  // M : size
    int_fast32_t N;  // N : size
    int_fast32_t K;  // K : size
    float* C;  // C : f32[N, M] @CudaGmemLinear
    CUtensorMap exo_data_A_tensorMap;  //     (Separate window data pointer)
    struct exo_win_2f32_Sm90_tensorMap_128_256_32 A_tensorMap;  // A_tensorMap : Window(src_type=f32[M, K], as_tensor=[f32][M, K], src_buf=A, idx='[0:M, 0:K]') @Sm90_tensorMap(128, 256, 32)
    CUtensorMap exo_data_B_tensorMap;  //     (Separate window data pointer)
    struct exo_win_2f32_Sm90_tensorMap_128_128_32 B_tensorMap;  // B_tensorMap : Window(src_type=f32[N, K], as_tensor=[f32][N, K], src_buf=B, idx='[0:N, 0:K]') @Sm90_tensorMap(128, 128, 32)
};

struct exo_Cuda0_xgemm_Sm90_wgmma_n128
{
  using exo_DeviceArgs = exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n128;

  static constexpr uint32_t exo_blockDim = 384;
  static constexpr uint32_t exo_clusterDim = 1;

  static constexpr unsigned exo_smemBytes = 196736;

  struct exo_Task
  {
    int_fast32_t m1_task;
    int_fast32_t n_task;
    int_fast32_t m0_task;
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
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs)
{
  cudaFuncSetAttribute(exo_deviceFunction0_xgemm_Sm90_wgmma_n128, cudaFuncAttributeMaxDynamicSharedMemorySize, exo_smemBytes);
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

  cudaLaunchKernelEx(&exo_launchConfig, exo_deviceFunction0_xgemm_Sm90_wgmma_n128, exo_deviceArgs);
}

__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
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
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceTask_producer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n128;
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
                    {{ exo_deviceArgs.A_tensorMap.exo_offsets[0] + (unsigned)(256 * (4 * exo_task.m1_task + exo_task.m0_task)), exo_deviceArgs.A_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    32768);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &B_smem[((k_iter % 4) * (4096)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_B_tensorMap,
                    {{ exo_deviceArgs.B_tensorMap.exo_offsets[0] + (unsigned)(128 * exo_task.n_task), exo_deviceArgs.B_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    16384);
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
                    {{ exo_deviceArgs.A_tensorMap.exo_offsets[0] + (unsigned)(256 * (4 * exo_task.m1_task + exo_task.m0_task)), exo_deviceArgs.A_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    32768);
        exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &B_smem[((k_iter % 4) * (4096)) / 256].byte_offset((0) * 4),
                    exo_deviceArgs.exo_data_B_tensorMap,
                    {{ exo_deviceArgs.B_tensorMap.exo_offsets[0] + (unsigned)(128 * exo_task.n_task), exo_deviceArgs.B_tensorMap.exo_offsets[1] + (unsigned)(32 * k_iter) }},
                    exo_tma_mbarrier,
                    16384);
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
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceTask_unused(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n128;
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
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceTask_consumer(char* exo_smem, exo_SyncState& exo_syncState, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task)
{
  namespace exo_CudaUtil = exo_CudaUtil_xgemm_Sm90_n128;
  // Scope of named barrier ringbar
  // Scope of named barrier cg
  exo_CudaUtil::exo_Sm90_RmemD_m128n128_f32 D_rmem;
  auto& A_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[128]);
  auto& B_smem = reinterpret_cast<Sm90_SmemMatrices_SW128 (&)[]>(exo_smem[131200]);
  if (int tmp_1 = threadIdx.x; tmp_1 >= 128) {
    if ([[maybe_unused]] int exo_128thr_wg = ((threadIdx.x - 128) / 128); 1) {
      D_rmem.scale_d = 0;
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
            exo_CudaUtil::exo_Sm90_mma_async_ss_m128n128_f32_tf32_tf32(exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &A_smem[((k_iter % 4) * (8192) + (16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4, 0), exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &A_smem[((k_iter % 4) * (8192) + (16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4, 64), exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &B_smem[((k_iter % 4) * (4096) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4), D_rmem.m0n0r0, D_rmem.m0n0r1, D_rmem.m0n0r2, D_rmem.m0n0r3, D_rmem.m0n8r0, D_rmem.m0n8r1, D_rmem.m0n8r2, D_rmem.m0n8r3, D_rmem.m0n16r0, D_rmem.m0n16r1, D_rmem.m0n16r2, D_rmem.m0n16r3, D_rmem.m0n24r0, D_rmem.m0n24r1, D_rmem.m0n24r2, D_rmem.m0n24r3, D_rmem.m0n32r0, D_rmem.m0n32r1, D_rmem.m0n32r2, D_rmem.m0n32r3, D_rmem.m0n40r0, D_rmem.m0n40r1, D_rmem.m0n40r2, D_rmem.m0n40r3, D_rmem.m0n48r0, D_rmem.m0n48r1, D_rmem.m0n48r2, D_rmem.m0n48r3, D_rmem.m0n56r0, D_rmem.m0n56r1, D_rmem.m0n56r2, D_rmem.m0n56r3, D_rmem.m0n64r0, D_rmem.m0n64r1, D_rmem.m0n64r2, D_rmem.m0n64r3, D_rmem.m0n72r0, D_rmem.m0n72r1, D_rmem.m0n72r2, D_rmem.m0n72r3, D_rmem.m0n80r0, D_rmem.m0n80r1, D_rmem.m0n80r2, D_rmem.m0n80r3, D_rmem.m0n88r0, D_rmem.m0n88r1, D_rmem.m0n88r2, D_rmem.m0n88r3, D_rmem.m0n96r0, D_rmem.m0n96r1, D_rmem.m0n96r2, D_rmem.m0n96r3, D_rmem.m0n104r0, D_rmem.m0n104r1, D_rmem.m0n104r2, D_rmem.m0n104r3, D_rmem.m0n112r0, D_rmem.m0n112r1, D_rmem.m0n112r2, D_rmem.m0n112r3, D_rmem.m0n120r0, D_rmem.m0n120r1, D_rmem.m0n120r2, D_rmem.m0n120r3, D_rmem.m64n0r0, D_rmem.m64n0r1, D_rmem.m64n0r2, D_rmem.m64n0r3, D_rmem.m64n8r0, D_rmem.m64n8r1, D_rmem.m64n8r2, D_rmem.m64n8r3, D_rmem.m64n16r0, D_rmem.m64n16r1, D_rmem.m64n16r2, D_rmem.m64n16r3, D_rmem.m64n24r0, D_rmem.m64n24r1, D_rmem.m64n24r2, D_rmem.m64n24r3, D_rmem.m64n32r0, D_rmem.m64n32r1, D_rmem.m64n32r2, D_rmem.m64n32r3, D_rmem.m64n40r0, D_rmem.m64n40r1, D_rmem.m64n40r2, D_rmem.m64n40r3, D_rmem.m64n48r0, D_rmem.m64n48r1, D_rmem.m64n48r2, D_rmem.m64n48r3, D_rmem.m64n56r0, D_rmem.m64n56r1, D_rmem.m64n56r2, D_rmem.m64n56r3, D_rmem.m64n64r0, D_rmem.m64n64r1, D_rmem.m64n64r2, D_rmem.m64n64r3, D_rmem.m64n72r0, D_rmem.m64n72r1, D_rmem.m64n72r2, D_rmem.m64n72r3, D_rmem.m64n80r0, D_rmem.m64n80r1, D_rmem.m64n80r2, D_rmem.m64n80r3, D_rmem.m64n88r0, D_rmem.m64n88r1, D_rmem.m64n88r2, D_rmem.m64n88r3, D_rmem.m64n96r0, D_rmem.m64n96r1, D_rmem.m64n96r2, D_rmem.m64n96r3, D_rmem.m64n104r0, D_rmem.m64n104r1, D_rmem.m64n104r2, D_rmem.m64n104r3, D_rmem.m64n112r0, D_rmem.m64n112r1, D_rmem.m64n112r2, D_rmem.m64n112r3, D_rmem.m64n120r0, D_rmem.m64n120r1, D_rmem.m64n120r2, D_rmem.m64n120r3, D_rmem.scale_d);
D_rmem.scale_d = 1;
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
            exo_CudaUtil::exo_Sm90_mma_async_ss_m128n128_f32_tf32_tf32(exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &A_smem[((k_iter % 4) * (8192) + (16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4, 0), exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &A_smem[((k_iter % 4) * (8192) + (16 * exo_128thr_wg) * (256) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4, 64), exo_CudaUtil::exo_matrix_descriptor(((struct exo_win_3f32c_Sm90_SmemSwizzled_128){ &B_smem[((k_iter % 4) * (4096) + 8 * k_mma) / 256].byte_offset((8 * k_mma) * 4), { 256, 32, 1 } }), 4), D_rmem.m0n0r0, D_rmem.m0n0r1, D_rmem.m0n0r2, D_rmem.m0n0r3, D_rmem.m0n8r0, D_rmem.m0n8r1, D_rmem.m0n8r2, D_rmem.m0n8r3, D_rmem.m0n16r0, D_rmem.m0n16r1, D_rmem.m0n16r2, D_rmem.m0n16r3, D_rmem.m0n24r0, D_rmem.m0n24r1, D_rmem.m0n24r2, D_rmem.m0n24r3, D_rmem.m0n32r0, D_rmem.m0n32r1, D_rmem.m0n32r2, D_rmem.m0n32r3, D_rmem.m0n40r0, D_rmem.m0n40r1, D_rmem.m0n40r2, D_rmem.m0n40r3, D_rmem.m0n48r0, D_rmem.m0n48r1, D_rmem.m0n48r2, D_rmem.m0n48r3, D_rmem.m0n56r0, D_rmem.m0n56r1, D_rmem.m0n56r2, D_rmem.m0n56r3, D_rmem.m0n64r0, D_rmem.m0n64r1, D_rmem.m0n64r2, D_rmem.m0n64r3, D_rmem.m0n72r0, D_rmem.m0n72r1, D_rmem.m0n72r2, D_rmem.m0n72r3, D_rmem.m0n80r0, D_rmem.m0n80r1, D_rmem.m0n80r2, D_rmem.m0n80r3, D_rmem.m0n88r0, D_rmem.m0n88r1, D_rmem.m0n88r2, D_rmem.m0n88r3, D_rmem.m0n96r0, D_rmem.m0n96r1, D_rmem.m0n96r2, D_rmem.m0n96r3, D_rmem.m0n104r0, D_rmem.m0n104r1, D_rmem.m0n104r2, D_rmem.m0n104r3, D_rmem.m0n112r0, D_rmem.m0n112r1, D_rmem.m0n112r2, D_rmem.m0n112r3, D_rmem.m0n120r0, D_rmem.m0n120r1, D_rmem.m0n120r2, D_rmem.m0n120r3, D_rmem.m64n0r0, D_rmem.m64n0r1, D_rmem.m64n0r2, D_rmem.m64n0r3, D_rmem.m64n8r0, D_rmem.m64n8r1, D_rmem.m64n8r2, D_rmem.m64n8r3, D_rmem.m64n16r0, D_rmem.m64n16r1, D_rmem.m64n16r2, D_rmem.m64n16r3, D_rmem.m64n24r0, D_rmem.m64n24r1, D_rmem.m64n24r2, D_rmem.m64n24r3, D_rmem.m64n32r0, D_rmem.m64n32r1, D_rmem.m64n32r2, D_rmem.m64n32r3, D_rmem.m64n40r0, D_rmem.m64n40r1, D_rmem.m64n40r2, D_rmem.m64n40r3, D_rmem.m64n48r0, D_rmem.m64n48r1, D_rmem.m64n48r2, D_rmem.m64n48r3, D_rmem.m64n56r0, D_rmem.m64n56r1, D_rmem.m64n56r2, D_rmem.m64n56r3, D_rmem.m64n64r0, D_rmem.m64n64r1, D_rmem.m64n64r2, D_rmem.m64n64r3, D_rmem.m64n72r0, D_rmem.m64n72r1, D_rmem.m64n72r2, D_rmem.m64n72r3, D_rmem.m64n80r0, D_rmem.m64n80r1, D_rmem.m64n80r2, D_rmem.m64n80r3, D_rmem.m64n88r0, D_rmem.m64n88r1, D_rmem.m64n88r2, D_rmem.m64n88r3, D_rmem.m64n96r0, D_rmem.m64n96r1, D_rmem.m64n96r2, D_rmem.m64n96r3, D_rmem.m64n104r0, D_rmem.m64n104r1, D_rmem.m64n104r2, D_rmem.m64n104r3, D_rmem.m64n112r0, D_rmem.m64n112r1, D_rmem.m64n112r2, D_rmem.m64n112r3, D_rmem.m64n120r0, D_rmem.m64n120r1, D_rmem.m64n120r2, D_rmem.m64n120r3, D_rmem.scale_d);
D_rmem.scale_d = 1;
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
      exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n0r0, 0, 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n0r1, 0, 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n0r2, 0, 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n0r3, 0, 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n8r0, 0, 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n8r1, 0, 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n8r2, 0, 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n8r3, 0, 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n16r0, 0, 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n16r1, 0, 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n16r2, 0, 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n16r3, 0, 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n24r0, 0, 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n24r1, 0, 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n24r2, 0, 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n24r3, 0, 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n32r0, 0, 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n32r1, 0, 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n32r2, 0, 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n32r3, 0, 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n40r0, 0, 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n40r1, 0, 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n40r2, 0, 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n40r3, 0, 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n48r0, 0, 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n48r1, 0, 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n48r2, 0, 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n48r3, 0, 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n56r0, 0, 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n56r1, 0, 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n56r2, 0, 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n56r3, 0, 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n64r0, 0, 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n64r1, 0, 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n64r2, 0, 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n64r3, 0, 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n72r0, 0, 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n72r1, 0, 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n72r2, 0, 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n72r3, 0, 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n80r0, 0, 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n80r1, 0, 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n80r2, 0, 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n80r3, 0, 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n88r0, 0, 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n88r1, 0, 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n88r2, 0, 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n88r3, 0, 47);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n96r0, 0, 48);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n96r1, 0, 49);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n96r2, 0, 50);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n96r3, 0, 51);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n104r0, 0, 52);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n104r1, 0, 53);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n104r2, 0, 54);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n104r3, 0, 55);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n112r0, 0, 56);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n112r1, 0, 57);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n112r2, 0, 58);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n112r3, 0, 59);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n120r0, 0, 60);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n120r1, 0, 61);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n120r2, 0, 62);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m0n120r3, 0, 63);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n0r0, 64, 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n0r1, 64, 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n0r2, 64, 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n0r3, 64, 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n8r0, 64, 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n8r1, 64, 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n8r2, 64, 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n8r3, 64, 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n16r0, 64, 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n16r1, 64, 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n16r2, 64, 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n16r3, 64, 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n24r0, 64, 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n24r1, 64, 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n24r2, 64, 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n24r3, 64, 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n32r0, 64, 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n32r1, 64, 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n32r2, 64, 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n32r3, 64, 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n40r0, 64, 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n40r1, 64, 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n40r2, 64, 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n40r3, 64, 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n48r0, 64, 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n48r1, 64, 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n48r2, 64, 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n48r3, 64, 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n56r0, 64, 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n56r1, 64, 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n56r2, 64, 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n56r3, 64, 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n64r0, 64, 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n64r1, 64, 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n64r2, 64, 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n64r3, 64, 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n72r0, 64, 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n72r1, 64, 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n72r2, 64, 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n72r3, 64, 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n80r0, 64, 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n80r1, 64, 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n80r2, 64, 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n80r3, 64, 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n88r0, 64, 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n88r1, 64, 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n88r2, 64, 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n88r3, 64, 47);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n96r0, 64, 48);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n96r1, 64, 49);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n96r2, 64, 50);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n96r3, 64, 51);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n104r0, 64, 52);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n104r1, 64, 53);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n104r2, 64, 54);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n104r3, 64, 55);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n112r0, 64, 56);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n112r1, 64, 57);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n112r2, 64, 58);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n112r3, 64, 59);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n120r0, 64, 60);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n120r1, 64, 61);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n120r2, 64, 62);
exo_CudaUtil::exo_Sm90_store_d_reg<true>(((struct exo_win_2f32){ &exo_deviceArgs.C[(128 * exo_task.n_task) * (exo_deviceArgs.M) + 128 * exo_128thr_wg + 256 * (4 * exo_task.m1_task + exo_task.m0_task)], { exo_deviceArgs.M, 1 } }), D_rmem.m64n120r3, 64, 63);
    }
  }
  // Fence(cuda_classic, cuda_classic)
  asm("barrier.cta.sync 0;");
}
__device__ __forceinline__ void
exo_Cuda0_xgemm_Sm90_wgmma_n128::exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
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
      for (int exo_task_m1_task = 0; exo_task_m1_task < ((((exo_deviceArgs.M) / (256)) + 3) / (4)); exo_task_m1_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (128)); exo_task_n_task++) {
          for (int exo_task_m0_task = 0; exo_task_m0_task < 4; exo_task_m0_task++) {
            if (4 * exo_task_m1_task + exo_task_m0_task < ((exo_deviceArgs.M) / (256))) {
              if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_producer(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m1_task, exo_task_n_task, exo_task_m0_task });
            }
          }
        }
      }
    }
    if (int tmp = threadIdx.x; tmp >= 32 && tmp < 128) {
      for (int exo_task_m1_task = 0; exo_task_m1_task < ((((exo_deviceArgs.M) / (256)) + 3) / (4)); exo_task_m1_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (128)); exo_task_n_task++) {
          for (int exo_task_m0_task = 0; exo_task_m0_task < 4; exo_task_m0_task++) {
            if (4 * exo_task_m1_task + exo_task_m0_task < ((exo_deviceArgs.M) / (256))) {
              if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_unused(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m1_task, exo_task_n_task, exo_task_m0_task });
            }
          }
        }
      }
    }
  }
  if (nreg == ((int32_t) 232)) {
    asm("setmaxnreg.inc.sync.aligned.u32 232;");
    if (int tmp = threadIdx.x; tmp >= 128 && tmp < 384) {
      for (int exo_task_m1_task = 0; exo_task_m1_task < ((((exo_deviceArgs.M) / (256)) + 3) / (4)); exo_task_m1_task++) {
        for (int exo_task_n_task = 0; exo_task_n_task < ((exo_deviceArgs.N) / (128)); exo_task_n_task++) {
          for (int exo_task_m0_task = 0; exo_task_m0_task < 4; exo_task_m0_task++) {
            if (4 * exo_task_m1_task + exo_task_m0_task < ((exo_deviceArgs.M) / (256))) {
              if (exo_taskIndex++ % (gridDim.x / exo_clusterDim) == blockIdx.x / exo_clusterDim) exo_deviceTask_consumer(exo_smem, exo_syncState, exo_deviceArgs, (struct exo_Task) { exo_task_m1_task, exo_task_n_task, exo_task_m0_task });
            }
          }
        }
      }
    }
  }
}

