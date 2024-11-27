
#pragma once
#ifndef EXO_PAR_H
#define EXO_PAR_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

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


#ifndef EXO_WIN_1F32
#define EXO_WIN_1F32
struct exo_win_1f32{
    float * const data;
    const int_fast32_t strides[1];
};
#endif
#ifndef EXO_WIN_1F32C
#define EXO_WIN_1F32C
struct exo_win_1f32c{
    const float * const data;
    const int_fast32_t strides[1];
};
#endif
// bar(
//     x : size,
//     y : f32[x] @DRAM
// )
void bar( void *ctxt, int_fast32_t x, float* y );

// gemm_test(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void gemm_test( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C );

// gpu_gemm(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void gpu_gemm( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C );



#ifdef __cplusplus
}
#endif
#endif  // EXO_PAR_H