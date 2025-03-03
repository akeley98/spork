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

Sm80_sync = 'asm volatile("cp.async.wait_all;" ::); __syncthreads();'

@instr("exo_Sm80_cpAsync16B(&{smem_data}, &{gmem_data});")
def tmp_cpAsync16B_f32(smem: [f32][4] @ CudaSmemLinear, gmem: [f32][4] @ CudaGmemLinear):
    for i in seq(0, 4):
        smem[i] = gmem[i]


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

                # Tiles (double buffered)
                A_tiles : f32[2, M1, K0] @ CudaSmemLinear
                B_tiles : f32[2, K0, N1] @ CudaSmemLinear

                # Zero-out accumulator
                accum : f32[16, 16, M0, N0] @ CudaRmem
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, M0, pragma_unroll=0):
                            for n0 in seq(0, N0, pragma_unroll=0):
                                accum[m1, n1, m0, n0] = 0

                # K tiles loop, double buffered
                # Don't accum tile in first iteration.
                # Don't load tile in last iteration.
                # 1 iteration delay between load and use.
                for k1 in seq(0, K / K0 + 1):
                    if k1 < K / K0:
                        # Load A tile
                        for m1 in seq(0, M1 / 64):
                            for m0 in cuda_threads(0, 64, unit=4 * cuda_thread):
                                for k0 in cuda_threads(0, 4, unit=cuda_thread):
                                    tmp_cpAsync16B_f32(A_tiles[k1 % 2, m1 * 64 + m0, 4 * k0 : 4 * k0 + 4],
                                                       A[m2 * M1 + m1 * 64 + m0, k1 * K0 + k0 * 4 : k1 * K0 + k0 * 4 + 4])

                        # Load B tile
                        for k0_seq in seq(0, 2):
                            for k0_par in cuda_threads(0, 8, unit=32 * cuda_thread):
                                for n0 in cuda_threads(0, 32, unit=cuda_thread):
                                    tmp_cpAsync16B_f32(B_tiles[k1 % 2, k0_seq * 8 + k0_par, 4 * n0 : 4 * n0 + 4],
                                                       B[k1 * K0 + k0_seq * 8 + k0_par, n2 * N1 + 4 * n0 : n2 * N1 + 4 * n0 + 4])

                    if k1 > 0:
                        # Multiply and accumulate A and B tiles from previous iteration.
                        for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                            for n1 in cuda_threads(0, 16, unit=cuda_thread):
                                for m0 in seq(0, M0, pragma_unroll=0):
                                    for n0 in seq(0, N0, pragma_unroll=0):
                                        for k0 in seq(0, K0):
                                            accum[m1,n1,m0,n0] += \
                                                A_tiles[1 - k1 % 2, m1 * M0 + m0, k0] \
                                                * B_tiles[1 - k1 % 2, k0, n1 * N0 + n0]

                    Fence(Sm80_generic, Sm80_generic, codegen=Sm80_sync)
                # End K tiles loop

                # Write out accumulator
                for m1 in cuda_threads(0, 16, unit=16 * cuda_thread):
                    for n1 in cuda_threads(0, 16, unit=cuda_thread):
                        for m0 in seq(0, M0, pragma_unroll=0):
                            for n0 in seq(0, N0, pragma_unroll=0):
                                C[m2 * M1 + m1 * M0 + m0, n2 * N1 + n1 * N0 + n0] = accum[m1,n1,m0, n0]
