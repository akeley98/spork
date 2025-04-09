
#pragma once
#ifndef EDITED_XGEMM_SM90_H
#define EDITED_XGEMM_SM90_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

#ifndef EXO_CUDA_HEADER_COMMON
#define EXO_CUDA_HEADER_COMMON
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#define EXO_CUDA_INLINE __device__ __forceinline__
EXO_CUDA_INLINE unsigned exo_smemU32(const void* smem_ptr)
{
    return (unsigned)__cvta_generic_to_shared(smem_ptr);
}
#endif
#endif

#ifndef EXO_CUDA_STREAM_GUARD
#define EXO_CUDA_STREAM_GUARD
static const cudaStream_t exo_cudaStream = 0;
#endif

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif



#include <stdio.h>
#include <stdlib.h>

#ifndef EXO_WIN_2F32
#define EXO_WIN_2F32
struct exo_win_2f32{
    float * const data;
    const int_fast32_t strides[2];
};
#endif
#ifndef EXO_WIN_2F32C
#define EXO_WIN_2F32C
struct exo_win_2f32c{
    const float * const data;
    const int_fast32_t strides[2];
};
#endif
// xgemm_Sm90_wgmma(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @CudaGmemLinear,
//     B : f32[N, K] @CudaGmemLinear,
//     C : f32[N, M] @CudaGmemLinear
// )
void edited_Sm90_wgmma( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C );



struct exo_CudaDeviceArgs0_edited_Sm90_wgmma;

#ifdef __CUDACC__
__global__ void exo_deviceFunction0_edited_Sm90_wgmma(__grid_constant__ const struct exo_CudaDeviceArgs0_edited_Sm90_wgmma exo_deviceArgs);
#endif
void exo_cudaLaunch0_edited_Sm90_wgmma(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs0_edited_Sm90_wgmma exo_deviceArgs);




#ifdef __cplusplus
}
#endif
#endif  // XGEMM_EDITED_SM90_H
