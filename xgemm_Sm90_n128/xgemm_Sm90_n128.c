#include "xgemm_Sm90_n128.h"



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
// CUDA device function args -- duplicated in .cuh file
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
Sm90_copy_tensor_to_smem_swizzled_2f32(dst,src,box0=128, box1=32)
exo_CudaUtil::exo_Sm90_tma_to_smem_2d(
                    &{dst_data},
                    {src_data},
                    {src_layout},
                    exo_tma_mbarrier,
                    16384);
*/

/* relying on the following instruction..."
Sm90_mma_async_tf32(d,a,b,n=128)
asm volatile("{{.reg .pred p;\n\tsetp.ne.b32 p, %66, 0;\n\twgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32\n\t{{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}},\n\t%64, %65, p, 1, 1;\n}}": "+f"({d_data}.d[0]), "+f"({d_data}.d[1]), "+f"({d_data}.d[2]), "+f"({d_data}.d[3]), "+f"({d_data}.d[4]), "+f"({d_data}.d[5]), "+f"({d_data}.d[6]), "+f"({d_data}.d[7]), "+f"({d_data}.d[8]), "+f"({d_data}.d[9]), "+f"({d_data}.d[10]), "+f"({d_data}.d[11]), "+f"({d_data}.d[12]), "+f"({d_data}.d[13]), "+f"({d_data}.d[14]), "+f"({d_data}.d[15]), "+f"({d_data}.d[16]), "+f"({d_data}.d[17]), "+f"({d_data}.d[18]), "+f"({d_data}.d[19]), "+f"({d_data}.d[20]), "+f"({d_data}.d[21]), "+f"({d_data}.d[22]), "+f"({d_data}.d[23]), "+f"({d_data}.d[24]), "+f"({d_data}.d[25]), "+f"({d_data}.d[26]), "+f"({d_data}.d[27]), "+f"({d_data}.d[28]), "+f"({d_data}.d[29]), "+f"({d_data}.d[30]), "+f"({d_data}.d[31]), "+f"({d_data}.d[32]), "+f"({d_data}.d[33]), "+f"({d_data}.d[34]), "+f"({d_data}.d[35]), "+f"({d_data}.d[36]), "+f"({d_data}.d[37]), "+f"({d_data}.d[38]), "+f"({d_data}.d[39]), "+f"({d_data}.d[40]), "+f"({d_data}.d[41]), "+f"({d_data}.d[42]), "+f"({d_data}.d[43]), "+f"({d_data}.d[44]), "+f"({d_data}.d[45]), "+f"({d_data}.d[46]), "+f"({d_data}.d[47]), "+f"({d_data}.d[48]), "+f"({d_data}.d[49]), "+f"({d_data}.d[50]), "+f"({d_data}.d[51]), "+f"({d_data}.d[52]), "+f"({d_data}.d[53]), "+f"({d_data}.d[54]), "+f"({d_data}.d[55]), "+f"({d_data}.d[56]), "+f"({d_data}.d[57]), "+f"({d_data}.d[58]), "+f"({d_data}.d[59]), "+f"({d_data}.d[60]), "+f"({d_data}.d[61]), "+f"({d_data}.d[62]), "+f"({d_data}.d[63]): "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&{a_data}, 1024, 1024)), "l"(exo_CudaUtil::exo_matrix_descriptor<1>(&{b_data}, 1024, 1024)), "r"({d_data}.scale_d));
{d_data}.scale_d = 1;
*/

/* relying on the following instruction..."
Sm90_mma_write_d_col_major_tf32(dst,src,n=128)
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[0], 0);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[1], 1);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[2], 2);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[3], 3);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[4], 4);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[5], 5);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[6], 6);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[7], 7);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[8], 8);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[9], 9);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[10], 10);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[11], 11);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[12], 12);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[13], 13);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[14], 14);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[15], 15);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[16], 16);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[17], 17);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[18], 18);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[19], 19);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[20], 20);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[21], 21);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[22], 22);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[23], 23);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[24], 24);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[25], 25);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[26], 26);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[27], 27);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[28], 28);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[29], 29);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[30], 30);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[31], 31);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[32], 32);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[33], 33);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[34], 34);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[35], 35);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[36], 36);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[37], 37);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[38], 38);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[39], 39);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[40], 40);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[41], 41);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[42], 42);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[43], 43);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[44], 44);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[45], 45);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[46], 46);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[47], 47);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[48], 48);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[49], 49);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[50], 50);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[51], 51);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[52], 52);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[53], 53);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[54], 54);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[55], 55);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[56], 56);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[57], 57);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[58], 58);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[59], 59);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[60], 60);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[61], 61);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[62], 62);
exo_CudaUtil::exo_Sm90_store_d_reg<true>({dst}, {src_data}.d[63], 63);
*/

/* relying on the following instruction..."
Sm90_zero_scale_d_tf32(d,n=128)
{d_data}.scale_d = 0;
*/
// xgemm_Sm90_wgmma_n128(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @CudaGmemLinear,
//     B : f32[N, K] @CudaGmemLinear,
//     C : f32[N, M] @CudaGmemLinear
// )
void xgemm_Sm90_wgmma_n128( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
EXO_ASSUME(M % 256 == 0);
EXO_ASSUME(N % 128 == 0);
EXO_ASSUME(K % 32 == 0);
CUtensorMap exo_data_A_tensorMap = exo_win_2f32_Sm90_tensorMap_128_256_32_encode(&A[0], (struct exo_win_2f32_Sm90_tensorMap_128_256_32_strides){ { K, 1 } }, (struct exo_win_2f32_Sm90_tensorMap_128_256_32_gmem_dim){ { M, K } });
struct exo_win_2f32_Sm90_tensorMap_128_256_32 A_tensorMap = {};
CUtensorMap exo_data_B_tensorMap = exo_win_2f32_Sm90_tensorMap_128_128_32_encode(&B[0], (struct exo_win_2f32_Sm90_tensorMap_128_128_32_strides){ { K, 1 } }, (struct exo_win_2f32_Sm90_tensorMap_128_128_32_gmem_dim){ { N, K } });
struct exo_win_2f32_Sm90_tensorMap_128_128_32 B_tensorMap = {};
exo_cudaLaunch0_xgemm_Sm90_wgmma_n128(exo_cudaStream, (struct exo_CudaDeviceArgs0_xgemm_Sm90_wgmma_n128) { M, N, K, C, exo_data_A_tensorMap, A_tensorMap, exo_data_B_tensorMap, B_tensorMap });
}

