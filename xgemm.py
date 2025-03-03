from __future__ import annotations
import exo
from exo import *
from exo.stdlib.scheduling import *
from exo.spork.cuda_memory import *

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

    with CudaDeviceFunction(blockDim = 256):
        Fence(cpu_cuda_api, cuda_api)
        for m2 in cuda_tasks(0, M / M1):
            for n2 in cuda_tasks(0, N / N1):
                # Per CTA code

                # Tiles
                A_tile : f32[M1, K0] @ CudaSmemLinear
                B_tile : f32[K0, N1] @ CudaSmemLinear

                # Zero-out accumulator
                accum : f32[16, 16, M0, N0] @ CudaRmem
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, M0, pragma_unroll=0):
                            for n0 in seq(0, N0, pragma_unroll=0):
                                accum[m1, n1, m0, n0] = 0

                # K tiles loop
                for k1 in seq(0, K / K0):
                    Fence(cuda_classic, cuda_classic, codegen="__syncthreads();")

                    # Load A tile
                    for m1 in seq(0, M1 / 16):
                        for m0 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for k0 in cuda_threads(0, 16, unit=cuda_thread):
                                A_tile[m1 * 16 + m0, k0] = A[m2 * M1 + m1 * 16 + m0, k1 * K0 + k0]

                    # Load B tile
                    for k0_seq in seq(0, 8):
                        for k0_par in cuda_threads(0, 2, unit=128 * cuda_thread):
                            for n0 in cuda_threads(0, 128, unit=cuda_thread):
                                B_tile[k0_seq * 2 + k0_par, n0] = B[k1 * K0 + k0_seq * 2 + k0_par, n2 * N1 + n0]

                    Fence(cuda_classic, cuda_classic, codegen="__syncthreads();")

                    # Multiply and accumulate A and B tiles
                    for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                        for n1 in cuda_threads(0, 16, unit=cuda_thread):
                            for m0 in seq(0, M0, pragma_unroll=0):
                                for n0 in seq(0, N0, pragma_unroll=0):
                                    for k0 in seq(0, K0):
                                        accum[m1,n1,m0,n0] += A_tile[m1 * M0 + m0, k0] * B_tile[k0, n1 * N0 + n0]
                # End K tiles loop

                # Write out accumulator
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, M0, pragma_unroll=0):
                            for n0 in seq(0, N0, pragma_unroll=0):
                                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1,n1,m0, n0]


