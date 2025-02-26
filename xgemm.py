from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.spork.cuda_memory import *

m_tile = 16
n_tile = 16
k_tile = 8

launch = "exo_cudaLaunch0_xgemm_cuda(exo_cudaStream, (struct exo_CudaDeviceArgs0_xgemm_cuda) {M, N, K, A, B, C});"

h_snippet = """\
#include <cuda.h>
#include <cuda_runtime.h>
static const cudaStream_t exo_cudaStream = 0;  // TODO

struct exo_CudaDeviceArgs0_xgemm_cuda
{
    int M, N, K;
    const float* A;
    const float* B;
    float* C;
};

void exo_cudaLaunch0_xgemm_cuda(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs0_xgemm_cuda exo_deviceArgs);
#ifdef __CUDACC__
__global__ void exo_deviceFunction0_xgemm_cuda(__grid_constant__ const struct exo_CudaDeviceArgs0_xgemm_cuda exo_deviceArgs);
#endif
"""

cuh_snippet = """\
struct exo_Cuda0_xgemm_cuda
{
    using exo_DeviceArgs = exo_CudaDeviceArgs0_xgemm_cuda;

    static const uint32_t exo_blockDim = 256;

    struct exo_Task
    {
        int_fast32_t m2, n2;
    };

    static constexpr unsigned exo_smemBytes = 0;

    static __device__ void exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);
    static __device__ void exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs);
    static __device__ void exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task);
    static void exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs);
};

__device__ void exo_Cuda0_xgemm_cuda::exo_deviceSetup(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
{
}

__device__ void exo_Cuda0_xgemm_cuda::exo_deviceMainLoop(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs)
{
    unsigned exo_taskIndex = 0;
    for (int_fast32_t m2 = 0; m2 < exo_deviceArgs.M / 256; ++m2) {
        for (int_fast32_t n2 = 0; n2 < exo_deviceArgs.N / 128; ++n2) {
            if (exo_taskIndex++ % gridDim.x == blockIdx.x) {
                exo_deviceTask(exo_smem, exo_deviceArgs, {m2, n2});
            }
        }
    }
}

inline void exo_Cuda0_xgemm_cuda::exo_cudaLaunch(cudaStream_t exo_cudaStream, const exo_DeviceArgs& exo_deviceArgs)
{
    const uint32_t exo_gridDim = 48;
    exo_deviceFunction0_xgemm_cuda<<<exo_gridDim, exo_blockDim, exo_smemBytes, exo_cudaStream>>>(exo_deviceArgs);
}
"""

cu_snippet = """\
__global__ void exo_deviceFunction0_xgemm_cuda(__grid_constant__ const struct exo_CudaDeviceArgs0_xgemm_cuda exo_deviceArgs)
{
    extern __shared__ char exo_smem[];
    exo_Cuda0_xgemm_cuda::exo_deviceSetup(exo_smem, exo_deviceArgs);
    exo_Cuda0_xgemm_cuda::exo_deviceMainLoop(exo_smem, exo_deviceArgs);
}

void exo_cudaLaunch0_xgemm_cuda(cudaStream_t exo_cudaStream, struct exo_CudaDeviceArgs0_xgemm_cuda exo_deviceArgs)
{
    exo_Cuda0_xgemm_cuda::exo_cudaLaunch(exo_cudaStream, exo_deviceArgs);
}
"""

ext_snippets = {"h":h_snippet, "cu":cu_snippet, "cuh":cuh_snippet}

body_prefix = """__device__ void exo_Cuda0_xgemm_cuda::exo_deviceTask(char* exo_smem, const exo_DeviceArgs& exo_deviceArgs, exo_Task exo_task) {
  const float* A = exo_deviceArgs.A;
  const float* B = exo_deviceArgs.B;
  float* C = exo_deviceArgs.C;
  int M = exo_deviceArgs.M;
  int N = exo_deviceArgs.N;
  int K = exo_deviceArgs.K;
"""
body_suffix = "}"
body_ext = "cuh"

M0 = 16
N0 = 8

M1 = 256
N1 = 128

K0 = 16

@proc
def xgemm_cuda(M: size, N: size, K: size, A: f32[M,K] @ CudaGmemLinear, B: f32[K,N] @ CudaGmemLinear, C: f32[M,N] @ CudaGmemLinear):
    assert M % M1 == 0
    assert N % N1 == 0
    assert K % K0 == 0

    with ExtWithContext(launch, body_prefix, body_suffix, body_ext, ext_snippets):
        # These should be cuda_tasks loops
        for m2 in _codegen_par(0, M / M1, c_index="exo_task.m2", static_bounds=(None, None)):
            for n2 in _codegen_par(0, N / N1, c_index="exo_task.n2", static_bounds=(None, None)):
                # Per CTA code

                # Tiles
                A_tile : f32[M1, K0] @ CudaSmemLinear
                B_tile : f32[K0, N1] @ CudaSmemLinear

                # Zero-out accumulator
                accum : f32[M0, N0] @ CudaRmem
                for m0 in seq(0, M0, pragma_unroll=0):
                    for n0 in seq(0, N0, pragma_unroll=0):
                        accum[m0, n0] = 0

                for k1 in seq(0, K / K0):
                    Fence(cuda_sync, cuda_sync, codegen="__syncthreads();")

                    # Load A tile
                    for m1 in seq(0, M1 / 16):
                        for m0 in _codegen_par(0, 16, c_index="threadIdx.x / 16", static_bounds=(None, None)):
                            for k0 in _codegen_par(0, 16, c_index="threadIdx.x % 16", static_bounds=(None, None)):
                                A_tile[m1 * 16 + m0, k0] = A[m2 * M1 + m1 * 16 + m0, k1 * K0 + k0]

                    # Load B tile
                    for k0_seq in seq(0, 8):
                        for k0_par in _codegen_par(0, 2, c_index="threadIdx.x / 128", static_bounds=(None, None)):
                            for n0 in _codegen_par(0, 128, c_index="threadIdx.x % 128", static_bounds=(None, None)):
                                B_tile[k0_seq * 2 + k0_par, n0] = B[k1 * K0 + k0_seq * 2 + k0_par, n2 * N1 + n0]

                    Fence(cuda_sync, cuda_sync, codegen="__syncthreads();")

                    for m1 in _codegen_par(0, 16, c_index="threadIdx.x / 16", static_bounds=(None, None)):
                        for n1 in _codegen_par(0, 16, c_index="threadIdx.x % 16", static_bounds=(None, None)):
                            for m0 in seq(0, M0, pragma_unroll=0):
                                for n0 in seq(0, N0, pragma_unroll=0):
                                    for k0 in seq(0, K0):
                                        accum[m0,n0] += A_tile[m1 * M0 + m0, k0] * B_tile[k0, n1 * N0 + n0]

                # Write out accumulator
                for m1 in _codegen_par(0, 16, c_index="threadIdx.x / 16", static_bounds=(None, None)):
                    for n1 in _codegen_par(0, 16, c_index="threadIdx.x % 16", static_bounds=(None, None)):
                        for m0 in seq(0, M0, pragma_unroll=0):
                            for n0 in seq(0, N0, pragma_unroll=0):
                                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m0, n0]

