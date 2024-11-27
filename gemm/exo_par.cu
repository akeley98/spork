#include "exo_par.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

namespace{__global__ void bar_0026blk_EXOCU(int_fast32_t x, float* z);}
namespace{__global__ void bar_0034b_EXOCU();}
namespace{__global__ void gpu_gemm_0075cta_i_EXOCU(int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C);}

/* relying on the following instruction..."
avx2_reg_copy_ps(dst,src)
{dst_data} = {src_data};
*/
// bar(
//     x : size,
//     y : f32[x] @DRAM
// )
void bar( void *ctxt, int_fast32_t x, float* y ) {
EXO_ASSUME(x % 128 == 0);
__m256 avx_a[100];
__m256 avx_b[100];
avx_a[42] = avx_b[42];
float *z = (float*) malloc(10 * x * sizeof(*z));
// TODO implement barrier alloc
for (int_fast32_t foo = 0; foo < 1; foo++) {
  ; // NO-OP
}
#pragma omp parallel for
for (int_fast32_t i = 0; i < x; i++) {
  y[i] += 1.0f;
}
if (4 > 0) bar_0026blk_EXOCU<<<4, 1024>>>(x, z);
// TODO implement barrier free
free(z);
if (4 > 0) bar_0034b_EXOCU<<<4, 1024>>>();
// TODO LoopIR.SyncStmt Fence(cuda_all, cpu)
}

namespace {
__global__ void bar_0026blk_EXOCU(int_fast32_t x, float* z)
{
  if (auto blk = int((blockIdx.x) % unsigned(4)); 0 <= blk && blk < 4) {
    // TODO LoopIR.SyncStmt Arrive(cuda_sync, bar)
    // TODO LoopIR.SyncStmt Await(bar, cuda_sync)
    if (threadIdx.x < 8u) {
      // TODO LoopIR.SyncStmt Fence(wgmma_reg, wgmma_async_reg)
      if (auto i = int((threadIdx.x) % unsigned(8)); 0 <= i && i < 8) {
        z[2 * x + 3] += 1.0f;
      }
    }
    if (threadIdx.x < 8u) {
      if (auto i = int((threadIdx.x) % unsigned(8)); 0 <= i && i < 8) {
        z[2 * x + 2] += 2.0f;
      }
    }
  }
}
}
namespace {
__global__ void bar_0034b_EXOCU()
{
  if (auto b = int((blockIdx.x) % unsigned(4)); 0 <= b && b < 4) {
    if ((threadIdx.x / 32u) >= 7u && (threadIdx.x / 32u) < 31u) {
      if (auto wz = int(((threadIdx.x / 32 - 7u) / unsigned((3)*((5) - (1)))) % unsigned((12) - (10))) + 10; 10 <= wz && wz < 12) {
        if (auto wy = int(((threadIdx.x / 32 - 7u) / unsigned(((5) - (1)))) % unsigned(3)); 1) {
          if (auto wx = int(((threadIdx.x / 32 - 7u)) % unsigned((5) - (1))) + 1; 1) {
            if (auto t = int((threadIdx.x) % unsigned(32)); 0 <= t && t < 32) {
              for (int_fast32_t foof = 0; foof < 8; foof++) {
                for (int_fast32_t bogus = 0; bogus < 16; bogus++) {
                  ; // NO-OP
                }
              }
            }
          }
        }
      }
    }
  }
}
}
// gemm_test(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void gemm_test( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
EXO_ASSUME(M % 16 == 0);
EXO_ASSUME(N % 16 == 0);
EXO_ASSUME(K % 16 == 0);
for (int_fast32_t i = 0; i < M; i++) {
  for (int_fast32_t j = 0; j < N; j++) {
    float c;
    c = 0.0f;
    for (int_fast32_t k = 0; k < K; k++) {
      c += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = c;
  }
}
}

// gpu_gemm(
//     M : size,
//     N : size,
//     K : size,
//     A : f32[M, K] @DRAM,
//     B : f32[K, N] @DRAM,
//     C : f32[M, N] @DRAM
// )
void gpu_gemm( void *ctxt, int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C ) {
EXO_ASSUME(M % 16 == 0);
EXO_ASSUME(N % 16 == 0);
EXO_ASSUME(K % 16 == 0);
if ((((M) / (16))) * (((N) / (16))) > 0) gpu_gemm_0075cta_i_EXOCU<<<(((M) / (16))) * (((N) / (16))), 256>>>(M, N, K, A, B, C);
}

namespace {
__global__ void gpu_gemm_0075cta_i_EXOCU(int_fast32_t M, int_fast32_t N, int_fast32_t K, const float* A, const float* B, float* C)
{
  if (auto cta_i = int((blockIdx.x / unsigned((((N) / (16))))) % unsigned(((M) / (16)))); 0 <= cta_i && cta_i < ((M) / (16))) {
    if (auto cta_j = int((blockIdx.x) % unsigned(((N) / (16)))); 0 <= cta_j && cta_j < ((N) / (16))) {
      if (auto thr_i = int((threadIdx.x / unsigned((16))) % unsigned(16)); 0 <= thr_i && thr_i < 16) {
        if (auto thr_j = int((threadIdx.x) % unsigned(16)); 1) {
          float c;
          c = 0.0f;
          for (int_fast32_t k = 0; k < K; k++) {
            c += A[(thr_i + 16 * cta_i) * K + k] * B[k * N + thr_j + 16 * cta_j];
          }
          C[(thr_i + 16 * cta_i) * N + thr_j + 16 * cta_j] = c;
        }
      }
    }
  }
}
}
