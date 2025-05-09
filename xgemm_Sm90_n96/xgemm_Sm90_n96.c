#include "xgemm_Sm90_n96.h"




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
// CUDA device function args -- duplicated in .cuh file
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


/* relying on the following instruction..."
Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=256, box1=32)
exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &{dst_data},
                    {src_data},
                    {src_layout},
                    exo_tma_mbarrier,
                    32768);
*/

/* relying on the following instruction..."
Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=96, box1=32)
exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &{dst_data},
                    {src_data},
                    {src_layout},
                    exo_tma_mbarrier,
                    12288);
*/

/* relying on the following instruction..."
Sm90_mma_async_tf32(d,a,b,M=128, N=96)
exo_CudaUtil::exo_Sm90_mma_async_ss_m128n96_f32_tf32_tf32(exo_CudaUtil::exo_matrix_descriptor({a}, 4, 0), exo_CudaUtil::exo_matrix_descriptor({a}, 4, 64), exo_CudaUtil::exo_matrix_descriptor({b}, 4), {d_data}.m0n0r0, {d_data}.m0n0r1, {d_data}.m0n0r2, {d_data}.m0n0r3, {d_data}.m0n8r0, {d_data}.m0n8r1, {d_data}.m0n8r2, {d_data}.m0n8r3, {d_data}.m0n16r0, {d_data}.m0n16r1, {d_data}.m0n16r2, {d_data}.m0n16r3, {d_data}.m0n24r0, {d_data}.m0n24r1, {d_data}.m0n24r2, {d_data}.m0n24r3, {d_data}.m0n32r0, {d_data}.m0n32r1, {d_data}.m0n32r2, {d_data}.m0n32r3, {d_data}.m0n40r0, {d_data}.m0n40r1, {d_data}.m0n40r2, {d_data}.m0n40r3, {d_data}.m0n48r0, {d_data}.m0n48r1, {d_data}.m0n48r2, {d_data}.m0n48r3, {d_data}.m0n56r0, {d_data}.m0n56r1, {d_data}.m0n56r2, {d_data}.m0n56r3, {d_data}.m0n64r0, {d_data}.m0n64r1, {d_data}.m0n64r2, {d_data}.m0n64r3, {d_data}.m0n72r0, {d_data}.m0n72r1, {d_data}.m0n72r2, {d_data}.m0n72r3, {d_data}.m0n80r0, {d_data}.m0n80r1, {d_data}.m0n80r2, {d_data}.m0n80r3, {d_data}.m0n88r0, {d_data}.m0n88r1, {d_data}.m0n88r2, {d_data}.m0n88r3, {d_data}.m64n0r0, {d_data}.m64n0r1, {d_data}.m64n0r2, {d_data}.m64n0r3, {d_data}.m64n8r0, {d_data}.m64n8r1, {d_data}.m64n8r2, {d_data}.m64n8r3, {d_data}.m64n16r0, {d_data}.m64n16r1, {d_data}.m64n16r2, {d_data}.m64n16r3, {d_data}.m64n24r0, {d_data}.m64n24r1, {d_data}.m64n24r2, {d_data}.m64n24r3, {d_data}.m64n32r0, {d_data}.m64n32r1, {d_data}.m64n32r2, {d_data}.m64n32r3, {d_data}.m64n40r0, {d_data}.m64n40r1, {d_data}.m64n40r2, {d_data}.m64n40r3, {d_data}.m64n48r0, {d_data}.m64n48r1, {d_data}.m64n48r2, {d_data}.m64n48r3, {d_data}.m64n56r0, {d_data}.m64n56r1, {d_data}.m64n56r2, {d_data}.m64n56r3, {d_data}.m64n64r0, {d_data}.m64n64r1, {d_data}.m64n64r2, {d_data}.m64n64r3, {d_data}.m64n72r0, {d_data}.m64n72r1, {d_data}.m64n72r2, {d_data}.m64n72r3, {d_data}.m64n80r0, {d_data}.m64n80r1, {d_data}.m64n80r2, {d_data}.m64n80r3, {d_data}.m64n88r0, {d_data}.m64n88r1, {d_data}.m64n88r2, {d_data}.m64n88r3, {d_data}.scale_d);
{d_data}.scale_d = 1;
*/

/* relying on the following instruction..."
Sm90_mma_write_d_col_major_tf32(dst,src,M=128, N=96)
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n0r0, 0, 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n0r1, 0, 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n0r2, 0, 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n0r3, 0, 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n8r0, 0, 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n8r1, 0, 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n8r2, 0, 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n8r3, 0, 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n16r0, 0, 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n16r1, 0, 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n16r2, 0, 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n16r3, 0, 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n24r0, 0, 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n24r1, 0, 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n24r2, 0, 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n24r3, 0, 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n32r0, 0, 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n32r1, 0, 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n32r2, 0, 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n32r3, 0, 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n40r0, 0, 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n40r1, 0, 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n40r2, 0, 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n40r3, 0, 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n48r0, 0, 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n48r1, 0, 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n48r2, 0, 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n48r3, 0, 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n56r0, 0, 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n56r1, 0, 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n56r2, 0, 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n56r3, 0, 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n64r0, 0, 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n64r1, 0, 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n64r2, 0, 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n64r3, 0, 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n72r0, 0, 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n72r1, 0, 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n72r2, 0, 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n72r3, 0, 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n80r0, 0, 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n80r1, 0, 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n80r2, 0, 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n80r3, 0, 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n88r0, 0, 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n88r1, 0, 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n88r2, 0, 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m0n88r3, 0, 47);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n0r0, 64, 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n0r1, 64, 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n0r2, 64, 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n0r3, 64, 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n8r0, 64, 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n8r1, 64, 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n8r2, 64, 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n8r3, 64, 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n16r0, 64, 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n16r1, 64, 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n16r2, 64, 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n16r3, 64, 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n24r0, 64, 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n24r1, 64, 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n24r2, 64, 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n24r3, 64, 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n32r0, 64, 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n32r1, 64, 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n32r2, 64, 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n32r3, 64, 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n40r0, 64, 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n40r1, 64, 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n40r2, 64, 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n40r3, 64, 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n48r0, 64, 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n48r1, 64, 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n48r2, 64, 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n48r3, 64, 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n56r0, 64, 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n56r1, 64, 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n56r2, 64, 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n56r3, 64, 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n64r0, 64, 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n64r1, 64, 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n64r2, 64, 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n64r3, 64, 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n72r0, 64, 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n72r1, 64, 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n72r2, 64, 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n72r3, 64, 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n80r0, 64, 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n80r1, 64, 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n80r2, 64, 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n80r3, 64, 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n88r0, 64, 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n88r1, 64, 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n88r2, 64, 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.m64n88r3, 64, 47);
*/

/* relying on the following instruction..."
Sm90_zero_scale_d_f32(M,N,d)
{d_data}.scale_d = 0;
*/
// xgemm_Sm90_wgmma_n96(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @CudaGmemLinear,
//     B : f32[N, K] @CudaGmemLinear,
//     C : f32[N, M] @CudaGmemLinear
// )
void xgemm_Sm90_wgmma_n96( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
EXO_ASSUME(M % 256 == 0);
EXO_ASSUME(N % 96 == 0);
EXO_ASSUME(K % 32 == 0);
CUtensorMap exo_data_A_tensorMap = exo_win_2f32_Sm90_tensorMap_128_256_32_encode(&A[0], (struct exo_win_2f32_Sm90_tensorMap_128_256_32_strides){ { K, 1 } }, (struct exo_win_2f32_Sm90_tensorMap_128_256_32_gmem_dim){ { M, K } });
struct exo_win_2f32_Sm90_tensorMap_128_256_32 A_tensorMap = {};
CUtensorMap exo_data_B_tensorMap = exo_win_2f32_Sm90_tensorMap_128_96_32_encode(&B[0], (struct exo_win_2f32_Sm90_tensorMap_128_96_32_strides){ { K, 1 } }, (struct exo_win_2f32_Sm90_tensorMap_128_96_32_gmem_dim){ { N, K } });
struct exo_win_2f32_Sm90_tensorMap_128_96_32 B_tensorMap = {};
exo_cudaLaunch0_xgemm_Sm90_wgmma_n96(exo_cudaStream, (struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n96) { M, N, K, C, exo_data_A_tensorMap, A_tensorMap, exo_data_B_tensorMap, B_tensorMap });
}

